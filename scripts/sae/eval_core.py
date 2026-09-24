"""SAEBench core eval of one of our SAEs at every target L0, next to SAEBench's own baselines.

Every run of train_sae.py -- hard or soft forward, fixed or random k -- is scored the same way:
as a plain hard TopK SAE (``sae_bench.custom_saes.topk_sae.TopKSAE``; ``ScoredTopKSAE`` below
when the top-k ranks a separate score matmul) built at each target k,
SAEBench's six L0s by default. A fixed-k run is scored off its training k too, which is the
point of comparison for the random-k runs.

Baselines are SAEBench's released SAEs at the same site and width, re-scored under the identical
core config: TopK (the recipe train_sae.py follows) and Matryoshka BatchTopK (the nested method
closest to "one SAE for many sparsities"), six trainers each. They are written once to
``<results>/core_baselines`` and shared by every run.

Core config = sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py's: 200
reconstruction batches, 2000 sparsity/variance batches, 16 prompts x 128 tokens of openwebtext,
special tokens excluded, featurewise statistics on, fp32. ``multiple_evals`` skips SAEs whose
result json already exists, and swallows per-SAE exceptions -- hence the missing-results check.

  UV_PROJECT_ENVIRONMENT=.venv-sae uv run --no-default-groups --group sae \
      python scripts/sae/eval_core.py --sae-dir results/sae/<tag>
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sae_bench.evals.core.main as core
from sae_bench.custom_saes.base_sae import BaseSAE
from sae_bench.custom_saes.batch_topk_sae import load_dictionary_learning_matryoshka_batch_topk_sae
from sae_bench.custom_saes.topk_sae import TopKSAE, load_dictionary_learning_topk_sae

from matryoshka_attribution import wandb_util

BASE_REPO = {("pythia-160m-deduped", 4096):
             "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108"}
BASELINES = {"TopK": ("TopK_pythia-160m-deduped__0108", load_dictionary_learning_topk_sae),
             "MatryoshkaBatchTopK": ("MatryoshkaBatchTopK_pythia-160m-deduped__0108",
                                     load_dictionary_learning_matryoshka_batch_topk_sae)}
METRICS = {"l0": ("sparsity", "l0"),
           "explained_variance": ("reconstruction_quality", "explained_variance"),
           "ce_loss_score": ("model_performance_preservation", "ce_loss_score")}


class ScoredTopKSAE(BaseSAE):
    """Inference form of a scores=sep SAE (matryoshka_attribution.sae): the k latents with the
    highest SEPARATE score s = (x - b_dec) @ W_score + b_score keep their relu activations."""

    def __init__(self, d_in, d_sae, k, model_name, hook_layer, device, dtype):
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype)
        self.W_score = nn.Parameter(torch.zeros(d_in, d_sae))
        self.b_score = nn.Parameter(torch.zeros(d_sae))
        self.k = k

    def encode(self, x):
        a = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        idx = ((x - self.b_dec) @ self.W_score + self.b_score).topk(self.k, dim=-1, sorted=False).indices
        return torch.zeros_like(a).scatter_(-1, idx, a.gather(-1, idx))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


def ours_at_k(sae_dir, k, device):
    tr = json.loads((sae_dir / "config.json").read_text())["trainer"]
    kw = dict(d_in=tr["activation_dim"], d_sae=tr["dict_size"], k=k, model_name=tr["lm_name"],
              hook_layer=tr["layer"], device=device, dtype=torch.float32)
    sd = torch.load(sae_dir / "ae.pt", map_location="cpu")
    # scores=acts and scores=pre select the same latents as SAEBench's TopKSAE (top-k of z and of
    # relu(z) agree on every positive latent; any extra picks have activation 0 either way).
    if tr.get("scores", "acts") == "sep":
        sae = ScoredTopKSAE(**kw)
        sae.load_state_dict(sd)
    else:
        sae = TopKSAE(**kw)
        sae.load_state_dict({**sd, "k": torch.tensor(k, dtype=torch.int)})
    sae.to(device=device, dtype=torch.float32)
    sae.cfg.architecture = "topk"
    return tr, sae


def run_core(selected, out, device):
    core.multiple_evals(
        selected_saes=[(rel, sae) for rel, (_, sae) in selected.items()],
        n_eval_reconstruction_batches=200, n_eval_sparsity_variance_batches=2000,
        eval_batch_size_prompts=16, exclude_special_tokens_from_reconstruction=True,
        # Both ON is not optional in sae-bench 0.6.0: the result json's MiscMetrics fields
        # (frac_alive, max encoder/decoder cosine sims, ...) are required and only computed with
        # these, so with them off every SAE fails pydantic validation at save time.
        compute_featurewise_density_statistics=True, compute_featurewise_weight_based_metrics=True,
        dataset="Skylion007/openwebtext", context_size=128, output_folder=str(out),
        dtype="float32", device=device)
    rows, missing = [], []
    for rel, (method, _) in selected.items():
        path = out / f"{rel}_custom_sae_eval_results.json"
        if not path.exists():
            missing.append(rel)
            continue
        m = json.loads(path.read_text())["eval_result_metrics"]
        rows.append({"sae": rel, "method": method,
                     **{name: m[a][b] for name, (a, b) in METRICS.items()}})
    return rows, missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[20, 40, 80, 160, 320, 640])
    p.add_argument("--baselines", nargs="*", default=list(BASELINES))
    p.add_argument("--n-trainers", type=int, default=6)
    wandb_util.add_args(p)
    p.add_argument("--wandb-name", default=None)
    args = p.parse_args()

    device = "cuda"
    sae_dir = Path(args.sae_dir)
    tag = sae_dir.name

    ours = {}
    for k in args.ks:
        tr, sae = ours_at_k(sae_dir, k, device)
        ours[f"{tag}_k{k}"] = (tag, sae)
    rows, missing = run_core(ours, sae_dir / "core", device)

    repo = BASE_REPO[(tr["lm_name"], tr["dict_size"])]
    base_out = sae_dir.parent / "core_baselines"
    base = {}
    for name in args.baselines:
        folder, loader = BASELINES[name]
        for i in range(args.n_trainers):
            base[f"saebench_{name}_trainer{i}"] = (name, loader(
                repo, f"{folder}/resid_post_layer_{tr['layer']}/trainer_{i}/ae.pt", tr["lm_name"],
                device, torch.float32, layer=tr["layer"], local_dir=str(sae_dir.parent / "downloaded_saes")))
    base_rows, base_missing = run_core(base, base_out, device)
    rows += base_rows
    missing += base_missing

    (sae_dir / "core" / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"{'sae':<60} {'L0':>7} {'FVE':>7} {'CE':>7}")
    for r in sorted(rows, key=lambda r: (r["method"], r["l0"])):
        print(f"{r['sae']:<60} {r['l0']:7.1f} {r['explained_variance']:7.4f} {r['ce_loss_score']:7.4f}")

    run = wandb_util.init("sae", args.wandb_name or f"{tag}_core", vars(args),
                          project=args.wandb_project, entity=args.wandb_entity, enabled=args.wandb,
                          group=f"{tr['lm_name']}/L{tr['layer']}/{tr['dict_size']}", job_type="core_eval")
    if run is not None:
        import wandb
        cols = ["sae", "method", *METRICS]
        run.log({"core": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
        run.summary.update({f"{r['sae']}/{m}": r[m] for r in rows for m in METRICS})
        run.finish()
    if missing:
        raise SystemExit(f"{len(missing)} SAEs have no core result (see log above): {missing}")


if __name__ == "__main__":
    main()
