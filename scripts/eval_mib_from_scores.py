"""Replicate eval_mib.py Phase-2 eval EXACTLY, but from a saved *_scores.pt (no retraining).

Purpose: isolate whether eval_mib and MIB's run_evaluation.py diverge on the *same* circuit
because of a live code difference, or because the stored eval_mib pkls are stale (dataset/TL
drift since the LR sweep). Builds the graph identically to eval_mib (Graph.from_model +
node_scores_tensor from the saved attn/mlp scores) and calls evaluate_area_under_curve TODAY.

Run (GPU):
  python scripts/eval_mib_from_scores.py --model gemma2 --task arc_easy \
     --scores results/topklog_lr_0.05/arc_easy_gemma2_scores.pt --split validation --batch-size 4
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import torch

from eval_mib import MODEL_TL_NAMES, TASKS_TO_HF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-examples", type=int, default=None)
    ap.add_argument("--mib-path", default="./MIB-circuit-track")
    args = ap.parse_args()

    mib_path = Path(args.mib_path).resolve()
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

    sd = torch.load(args.scores, map_location="cpu")
    attn_scores, mlp_scores = sd["attn_scores"], sd["mlp_scores"]
    inc_input = sd["args"].get("include_input", True)
    input_score_val = sd["scores"][0].item() if inc_input else None

    graph = Graph.from_model(tl_model, node_scores=True)
    mx = max(attn_scores.abs().max().item(), mlp_scores.abs().max().item()) + 1.0
    nst = torch.full((graph.n_forward,), float("nan"))
    for name, node in graph.nodes.items():
        if name == "logits":
            continue
        idx = graph.forward_index(node, attn_slice=False)
        if idx >= graph.n_forward:
            continue
        if name == "input":
            nst[idx] = input_score_val if input_score_val is not None else mx
        elif name.startswith("a"):
            p = name.split(".")
            nst[idx] = attn_scores[int(p[0][1:]), int(p[1][1:])].item()
        elif name.startswith("m"):
            nst[idx] = mlp_scores[int(name[1:])].item()
    graph.nodes_scores = nst

    hf_task_name = f"mib-bench/{TASKS_TO_HF[args.task]}"
    ds = HFEAPDataset(hf_task_name, tl_model.tokenizer, split=args.split,
                      task=args.task, model_name=args.model)
    if args.eval_examples:
        ds.head(args.eval_examples)
    dataloader = ds.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, tl_model.tokenizer, tl_model)
    am = partial(metric, mean=False, loss=False)

    out = evaluate_area_under_curve(tl_model, graph, dataloader, am, level="node", absolute=False)
    wec, area_under, area_from_1, average, faith, acc, acc_auc = out
    print(f"FRESH eval_mib-style: acc-AUC={acc_auc:.4f} CPR={area_under:.4f}")
    print("acc curve:", [round(a, 3) for a in acc])


if __name__ == "__main__":
    main()
