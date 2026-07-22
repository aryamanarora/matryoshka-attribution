"""Compute the two method-independent CPR-normalisation anchors per (task, model):
  baseline_score  = clean model metric (mean logit diff)
  corrupted_score = fully-corrupted metric (all nodes ablated, apply_topn(0))

These are exactly the B, C in CPR's faithfulness = (ablated - C) / (B - C). They are NOT
saved by eval_mib, so raw logit-diff curves can't be recovered from the result pkls alone.
With B, C we invert the stored (normalised) faithfulness of EVERY method for free:
  raw_ablated(pct) = C + faithfulness(pct) * (B - C).

This reuses eval_mib's Phase-2 setup (TL model + graph + eval dataloader), but skips training
and the sparsity sweep — just 2 forward evals. Dumps results/anchors/{task}_{model}.json.

Run (per cell, GPU):
  python scripts/eval_mib_anchors.py --model gpt2 --task ioi --split validation --include-input \
      --output results/anchors
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch

from eval_mib import MODEL_TL_NAMES, TASKS_TO_HF  # reuse the exact name maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-examples", type=int, default=None,
                    help="Cap eval set (match a capped run's n so B,C align with its faithfulness)")
    ap.add_argument("--include-input", action="store_true")
    ap.add_argument("--mib-path", default="./MIB-circuit-track")
    ap.add_argument("--output", default="results/anchors")
    args = ap.parse_args()

    # put MIB + EAP-IG on the path (mirror eval_mib.py)
    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))
    sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))

    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from eap.evaluate import evaluate_baseline, evaluate_graph
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.dataset import HFEAPDataset

    # --- TL model (mirror eval_mib.py Phase 2) ---
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

    # --- graph: scores don't matter for anchors (baseline ignores the graph;
    #     apply_topn(0) corrupts everything), but nodes_scores must be non-None ---
    graph = Graph.from_model(tl_model, node_scores=True)
    graph.nodes_scores = torch.zeros(graph.n_forward)

    hf_task_name = f"mib-bench/{TASKS_TO_HF[args.task]}"
    eval_dataset = HFEAPDataset(hf_task_name, tl_model.tokenizer, split=args.split,
                                task=args.task, model_name=args.model)
    if args.eval_examples:
        eval_dataset.head(args.eval_examples)
    dataloader = eval_dataset.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, tl_model.tokenizer, tl_model)
    attribution_metric = partial(metric, mean=False, loss=False)

    # --- the two anchors (mirror evaluate_area_under_curve lines 17-20) ---
    baseline = evaluate_baseline(tl_model, dataloader, attribution_metric).mean().item()
    graph.apply_topn(0, True, level="node", prune=True)
    corrupted = evaluate_graph(tl_model, graph, dataloader, attribution_metric,
                               intervention="patching").mean().item()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"task": args.task, "model": args.model, "split": args.split,
           "n_eval": len(eval_dataset), "baseline": baseline, "corrupted": corrupted}
    (out / f"{args.task}_{args.model}.json").write_text(json.dumps(rec, indent=2))
    print(f"{args.task}/{args.model}: baseline={baseline:.4f} corrupted={corrupted:.4f} "
          f"(n={len(eval_dataset)}) -> {out}/{args.task}_{args.model}.json")


if __name__ == "__main__":
    main()
