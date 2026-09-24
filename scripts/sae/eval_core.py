"""SAEBench core eval of a sigmoid top-k SAE at every target L0, next to SAEBench's own baselines.

Our SAE is trained once over a k distribution, so it is scored as a plain hard TopK SAE
(``sae_bench.custom_saes.topk_sae.TopKSAE``) built at each target k -- SAEBench's six L0s by
default. The baselines are SAEBench's released SAEs at the same site and width, re-scored here
under the identical core config so every point shares one code version and one data draw:
TopK (the recipe we changed) and Matryoshka BatchTopK (the nested-dictionary method closest to
"one SAE for many sparsities"), six trainers each.

Core config = sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py's: 200
reconstruction batches, 2000 sparsity/variance batches, 16 prompts x 128 tokens of openwebtext,
special tokens excluded, fp32. ``multiple_evals`` skips SAEs whose result json already exists,
so a rerun only fills what is missing. It also swallows per-SAE exceptions, hence the explicit
missing-results check at the end.

  UV_PROJECT_ENVIRONMENT=.venv-sae uv run --no-default-groups --group sae \
      python scripts/sae/eval_core.py --sae-dir results/sae/pythia160m_l8_4k_sigtopk_uniform
"""

import argparse
import json
from pathlib import Path

import torch
import sae_bench.evals.core.main as core
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
           "ce_loss_score": ("model_performance_preservation", "ce_loss_score"),
           "kl_div_score": ("model_behavior_preservation", "kl_div_score")}


def ours_at_k(sae_dir, k, device):
    tr = json.loads((sae_dir / "config.json").read_text())["trainer"]
    sae = TopKSAE(d_in=tr["activation_dim"], d_sae=tr["dict_size"], k=k, model_name=tr["lm_name"],
                  hook_layer=tr["layer"], device=device, dtype=torch.float32)
    sd = torch.load(sae_dir / "ae.pt", map_location="cpu")
    sae.load_state_dict({**sd, "k": torch.tensor(k, dtype=torch.int)})
    sae.to(device=device, dtype=torch.float32)
    sae.cfg.architecture = "topk"
    return tr, sae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[20, 40, 80, 160, 320, 640])
    p.add_argument("--baselines", nargs="*", default=list(BASELINES))
    p.add_argument("--n-trainers", type=int, default=6)
    p.add_argument("--out", default=None, help="default: <sae-dir>/core")
    wandb_util.add_args(p)
    p.add_argument("--wandb-name", default=None)
    args = p.parse_args()

    device = "cuda"
    sae_dir = Path(args.sae_dir)
    out = Path(args.out or sae_dir / "core")
    tag = sae_dir.name

    selected = {}                                   # release name -> (method, SAE object)
    for k in args.ks:
        tr, sae = ours_at_k(sae_dir, k, device)
        selected[f"{tag}_k{k}"] = (tag, sae)
    repo = BASE_REPO[(tr["lm_name"], tr["dict_size"])]
    for name in args.baselines:
        folder, loader = BASELINES[name]
        for i in range(args.n_trainers):
            sae = loader(repo, f"{folder}/resid_post_layer_{tr['layer']}/trainer_{i}/ae.pt",
                         tr["lm_name"], device, torch.float32, layer=tr["layer"],
                         local_dir=str(out.parent / "downloaded_saes"))
            selected[f"saebench_{name}_trainer{i}"] = (name, sae)

    core.multiple_evals(
        selected_saes=[(rel, sae) for rel, (_, sae) in selected.items()],
        n_eval_reconstruction_batches=200, n_eval_sparsity_variance_batches=2000,
        eval_batch_size_prompts=16, exclude_special_tokens_from_reconstruction=True,
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
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"{'sae':<55} {'L0':>7} {'FVE':>7} {'CE':>7} {'KL':>7}")
    for r in sorted(rows, key=lambda r: (r["method"], r["l0"])):
        print(f"{r['sae']:<55} {r['l0']:7.1f} {r['explained_variance']:7.4f} "
              f"{r['ce_loss_score']:7.4f} {r['kl_div_score']:7.4f}")

    run = wandb_util.init("sae", args.wandb_name or f"{tag}_core", vars(args),
                          project=args.wandb_project, entity=args.wandb_entity,
                          enabled=args.wandb, group=f"{tr['lm_name']}/L{tr['layer']}/{tr['dict_size']}",
                          job_type="core_eval")
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
