"""MIB's node-level sweep with the intervention FLIPPED: patch the top-k, keep the rest clean.

    uv run python scripts/mib/eval_mib_noising.py --model gpt2 --task ioi \
        --dir mib_node_cause_topk_uniform_lr05 [--split validation] [--head 200]
    -> results/<dir>/<task>_<model>_<split>_noising.pkl

WHAT IT MEASURES. Every MIB number of ours is the sufficiency (denoising) sweep: the top-k stay
clean, the complement is corrupted. This runs the mirror, necessity (noising): at each of MIB's
ten proportions the top-k are corrupted and everything else is clean, through the fork's own
evaluate_area_under_curve(invert=True) -- same references, same grid, same trapezoids, same
per-example metric. For a ranking that puts the necessary components first, faithfulness
falls from ~1 fast; the scalar to read is `area_from_1` (area between 1 and the curve,
higher = the top-k break the behaviour sooner). Two accuracies come out: `accuracies` /
`acc_auc` (the clean answer still wins; falls) and `flip_accuracies` / `flip_acc_auc` (the
counterfactual answer wins -- the reverse-IIA reading; rises). `input` is never patched.

WHY. The objective ablation (scripts/mib/launch/submit_mib_node_mode_ablation_sc.sh: iso /
cause / joint) trains two of its three runs for necessity and MIB only ever scored them on
sufficiency. This is the direction they were trained for. VALIDATION ONLY (user decision
2026-09-16), llama3 capped at --head 200 like the validation tables.

The graph is built from the run's scores .pt exactly as eval_mib.py builds it for the
sufficiency eval (same index layout, same input handling), so the two pkls of a dir describe
one ranking. The reader is plots/plot_objective_ablation.py.
"""
import argparse
import os
import pickle
import sys
from functools import partial
from pathlib import Path

import torch
from matryoshka_attribution.deps import find_mib_path


def build_graph(Graph, model, scores, include_input):
    """Node scores onto an EAP-IG graph, mirroring eval_mib.py (input = max + 1 when unscored)."""
    graph = Graph.from_model(model, node_scores=True)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    off = 1 if include_input else 0
    input_score = scores[0].item() if include_input else None
    attn = scores[off:off + n_layers * n_heads].view(n_layers, n_heads).cpu()
    mlp = scores[off + n_layers * n_heads:].cpu()
    assert mlp.numel() == n_layers, f"scores has {scores.numel()} entries; expected {off}+{n_layers * n_heads}+{n_layers}"
    max_score = max(attn.abs().max().item(), mlp.abs().max().item()) + 1.0
    ns = torch.full((graph.n_forward,), float("nan"))
    for name, node in graph.nodes.items():
        if name == "logits":
            continue
        idx = graph.forward_index(node, attn_slice=False)
        if idx >= graph.n_forward:
            continue
        if name == "input":
            ns[idx] = input_score if input_score is not None else max_score
        elif name.startswith("a"):
            L, H = name.split("."); ns[idx] = attn[int(L[1:]), int(H[1:])].item()
        elif name.startswith("m"):
            ns[idx] = mlp[int(name[1:])].item()
    graph.nodes_scores = ns
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--dir", required=True, help="results/<dir> holding <task>_<model>_scores.pt")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--head", type=int, default=None, help="cap the eval set (llama3 validation: 200)")
    ap.add_argument("--mib-path", default=None)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    a.mib_path = str(find_mib_path(a.mib_path))
    sys.path.insert(0, a.mib_path)
    sys.path.insert(0, os.path.join(a.mib_path, "EAP-IG", "src"))
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME
    from MIB_circuit_track.dataset import HFEAPDataset

    d = Path("results") / a.dir
    sd = torch.load(d / f"{a.task}_{a.model}_scores.pt", map_location="cpu")
    scores = sd["scores"].float()
    include_input = bool(sd.get("args", {}).get("include_input", False))
    out = Path(a.output) if a.output else d / f"{a.task}_{a.model}_{a.split}_noising.pkl"

    if a.model in ("qwen2.5", "gemma2", "llama3"):
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[a.model],
                                                  attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[a.model])
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    graph = build_graph(Graph, model, scores, include_input)
    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[a.task]}", model.tokenizer,
                      split=a.split, task=a.task, model_name=a.model)
    if a.head is not None:
        ds.head(min(a.head, len(ds)))
    dl = ds.to_dataloader(batch_size=a.batch_size)
    metric = get_metric("logit_diff", a.task, model.tokenizer, model)
    metrics = partial(metric, mean=False, loss=False)

    extra = {}
    wec, area_under, area_from_1, average, faiths, accs, acc_auc = evaluate_area_under_curve(
        model, graph, dl, metrics, quiet=True, level="node", absolute=False,
        intervention="patching", invert=True, extra=extra)
    res = dict(direction="noising", source_dir=a.dir, split=a.split, head=a.head,
               weighted_edge_counts=wec, area_under=area_under, area_from_1=area_from_1,
               average=average, faithfulnesses=faiths, accuracies=accs, acc_auc=acc_auc,
               flip_accuracies=extra.get("flip_accuracies"), flip_acc_auc=extra.get("flip_acc_auc"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print(f"{a.task}/{a.model} [{a.split}, noising] {a.dir}: area_from_1={area_from_1:.4f} "
          f"area_under={area_under:.4f} acc_auc(base)={acc_auc:.4f} flip_acc_auc={res['flip_acc_auc']:.4f}")
    print("  faith", [round(x, 3) for x in faiths])
    print("  flip ", [round(x, 3) for x in res["flip_accuracies"]])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
