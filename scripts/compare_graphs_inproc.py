"""In one process, same model + same dataloader + same metric, evaluate TWO graphs of the
identical MAttr circuit:
  A = eval_mib path:      Graph.from_model(tl, node_scores=True) + node_scores_tensor
  B = run_evaluation path: Graph.from_json(importances.json)
If A and B give different acc curves here, the divergence IS the graph object (isolated from
model/data/harness). If identical, the difference is elsewhere in the two scripts.
"""
import argparse, sys
from functools import partial
from pathlib import Path
import torch

from eval_mib import MODEL_TL_NAMES, TASKS_TO_HF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma2")
    ap.add_argument("--task", default="arc_easy")
    ap.add_argument("--scores", default="results/topklog_lr_0.05/arc_easy_gemma2_scores.pt")
    ap.add_argument("--importances", default="results/topklog_lr_0.05/arc_easy_gemma2_importances.json")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-examples", type=int, default=200)
    ap.add_argument("--mib-path", default="./MIB-circuit-track")
    args = ap.parse_args()

    mib = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib)); sys.path.insert(0, str(mib / "EAP-IG" / "src"))
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.evaluation import evaluate_area_under_curve

    tl_name = MODEL_TL_NAMES[args.model]
    tl = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
    tl.cfg.use_split_qkv_input = True; tl.cfg.use_attn_result = True
    tl.cfg.use_hook_mlp_in = True; tl.cfg.ungroup_grouped_query_attention = True

    # graph A (eval_mib)
    sd = torch.load(args.scores, map_location="cpu")
    a_s, m_s = sd["attn_scores"], sd["mlp_scores"]
    iv = sd["scores"][0].item() if sd["args"].get("include_input", True) else None
    gA = Graph.from_model(tl, node_scores=True)
    nst = torch.full((gA.n_forward,), float("nan"))
    for name, node in gA.nodes.items():
        if name == "logits": continue
        idx = gA.forward_index(node, attn_slice=False)
        if idx >= gA.n_forward: continue
        if name == "input": nst[idx] = iv if iv is not None else 0.0
        elif name.startswith("a"):
            p = name.split("."); nst[idx] = a_s[int(p[0][1:]), int(p[1][1:])].item()
        elif name.startswith("m"): nst[idx] = m_s[int(name[1:])].item()
    gA.nodes_scores = nst

    # graph B (from_json)
    gB = Graph.from_json(args.importances)

    # shared data + metric
    hf = f"mib-bench/{TASKS_TO_HF[args.task]}"
    ds = HFEAPDataset(hf, tl.tokenizer, split=args.split, task=args.task, model_name=args.model)
    if args.eval_examples: ds.head(args.eval_examples)
    dl = ds.to_dataloader(batch_size=args.batch_size)
    am = partial(get_metric("logit_diff", args.task, tl.tokenizer, tl), mean=False, loss=False)

    for tag, g in [("A eval_mib (from_model)", gA), ("B run_eval (from_json)", gB)]:
        out = evaluate_area_under_curve(tl, g, dl, am, level="node", absolute=False)
        acc, acc_auc = out[5], out[6]
        print(f"[{tag}] acc-AUC={acc_auc:.4f} floor={acc[0]:.3f} curve={[round(x,3) for x in acc]}")


if __name__ == "__main__":
    main()
