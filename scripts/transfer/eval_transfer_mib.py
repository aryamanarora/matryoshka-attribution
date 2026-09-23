"""Cross-task transfer eval, MIB harness: score task B's circuit eval with task A's ranking.

For one TARGET MIB task, loads llama3 once, then loops over a list of SOURCE score vectors
(LABEL:PATH specs) and runs the exact eval_mib Phase-2 eval (Graph.from_model + MIB's
evaluate_area_under_curve) for each. This is eval_mib.py's scoring phase with (a) many sources per
model load and (b) an incremental output json, so a preempted job resumes.

Sources may be either format that holds the full node-layout vector (index 0 = input, then
32x32 attention heads in (layer, head) order, then 32 MLPs -- LlamaAttributionHooks' layout,
the same one plots/plot_task_corr_heatmap.py::names() documents):
  - eval_mib.py's {task}_{model}_scores.pt dict (its "scores" entry is used; when the dict also
    carries attn_scores/mlp_scores they are asserted equal to the vector's slices, so a layout
    drift fails loudly instead of silently permuting the ranking);
  - eval_sva.py's raw [1057] .scores.pt tensor (--nodes node --include-input runs only).

Run (GPU):
  python scripts/transfer/eval_transfer_mib.py --model llama3 --task ioi \
      --sources ioi:results/softlog_sgd_lr_1.0/ioi_llama3_scores.pt \
                simple:results/sva_sweep_input/simple_llama3_node_sufficient_topk_sgd_bs1.scores.pt \
      --eval-examples 200 --batch-size 2 --output results/transfer_mib

Output: <output>/{task}_{model}_transfer.json = {label: {area_under, acc_auc, faithfulnesses,
accuracies, ...}}. The diagonal (task's own scores) should be in --sources too, so every cell
of the transfer matrix -- including the reference -- is computed by the same eval on the same
200 examples.
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mib"))  # eval_mib lives in the mib group
from eval_mib import MODEL_TL_NAMES, TASKS_TO_HF
from matryoshka_attribution.deps import find_mib_path


def load_source_vec(path):
    """Full node-layout score vector [n_nodes] from either on-disk format."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        vec = sd["scores"].float().flatten()
        # Layout tripwire: the dict's decomposed tensors must be slices of the vector.
        if "attn_scores" in sd and "mlp_scores" in sd:
            nl, nh = sd["attn_scores"].shape
            off = vec.numel() - (nl * nh + nl)   # 1 iff the input node is included
            assert torch.equal(vec[off:off + nl * nh].view(nl, nh).float(), sd["attn_scores"].float()), \
                f"{path}: scores vector does not match attn_scores -- layout drift"
            assert torch.equal(vec[off + nl * nh:].float(), sd["mlp_scores"].float()), \
                f"{path}: scores vector does not match mlp_scores -- layout drift"
        return vec
    return sd.float().flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, help="TARGET task (what gets evaluated)")
    ap.add_argument("--sources", nargs="+", required=True, metavar="LABEL:PATH")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-examples", type=int, default=200)
    ap.add_argument("--output", default="results/transfer_mib")
    ap.add_argument("--mib-path", default=None)
    args = ap.parse_args()

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    out_fn = outdir / f"{args.task}_{args.model}_transfer.json"
    results = json.load(open(out_fn)) if out_fn.exists() else {}

    specs = []
    for spec in args.sources:
        label, _, pth = spec.partition(":")
        if not pth:
            raise SystemExit(f"--sources wants LABEL:PATH, got {spec!r}")
        if label in results:
            print(f"skip {label} (already in {out_fn})")
            continue
        specs.append((label, pth, load_source_vec(pth)))   # load up front: fail before the GPU
    if not specs:
        print("nothing to do"); return

    mib_path = find_mib_path(args.mib_path)
    sys.path.insert(0, str(mib_path))
    sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.evaluation import evaluate_area_under_curve

    tl_name = MODEL_TL_NAMES[args.model]
    if args.model in ("gemma2", "llama3", "qwen2.5"):
        tl_model = HookedTransformer.from_pretrained(
            tl_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        tl_model = HookedTransformer.from_pretrained(tl_name)
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.cfg.ungroup_grouped_query_attention = True

    nl, nh = tl_model.cfg.n_layers, tl_model.cfg.n_heads
    graph = Graph.from_model(tl_model, node_scores=True)

    hf_task_name = f"mib-bench/{TASKS_TO_HF[args.task]}"
    ds = HFEAPDataset(hf_task_name, tl_model.tokenizer, split=args.split,
                      task=args.task, model_name=args.model)
    if args.eval_examples:
        ds.head(args.eval_examples)
    dataloader = ds.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, tl_model.tokenizer, tl_model)
    am = partial(metric, mean=False, loss=False)

    for label, pth, vec in specs:
        n_expected = 1 + nl * nh + nl
        if vec.numel() != n_expected:
            raise SystemExit(f"{pth}: {vec.numel()} scores, expected {n_expected} "
                             f"(input + {nl}x{nh} heads + {nl} MLPs)")
        # Same graph fill as eval_mib.py, from the unified vector.
        nst = torch.full((graph.n_forward,), float("nan"))
        for name, node in graph.nodes.items():
            if name == "logits":
                continue
            idx = graph.forward_index(node, attn_slice=False)
            if idx >= graph.n_forward:
                continue
            if name == "input":
                nst[idx] = vec[0].item()
            elif name.startswith("a"):
                p = name.split(".")
                nst[idx] = vec[1 + int(p[0][1:]) * nh + int(p[1][1:])].item()
            elif name.startswith("m"):
                nst[idx] = vec[1 + nl * nh + int(name[1:])].item()
        graph.nodes_scores = nst

        out = evaluate_area_under_curve(tl_model, graph, dataloader, am,
                                        level="node", absolute=False)
        wec, area_under, area_from_1, average, faith, acc, acc_auc = out
        results[label] = dict(source=pth, weighted_edge_counts=[float(w) for w in wec],
                              area_under=float(area_under), area_from_1=float(area_from_1),
                              average=float(average), faithfulnesses=[float(f) for f in faith],
                              accuracies=[float(a) for a in acc], acc_auc=float(acc_auc),
                              split=args.split, eval_examples=args.eval_examples)
        json.dump(results, open(out_fn, "w"), indent=2)   # incremental: survive preemption
        print(f"[{args.task} <- {label}] CPR={area_under:.4f} acc-AUC={acc_auc:.4f} -> {out_fn}")


if __name__ == "__main__":
    main()
