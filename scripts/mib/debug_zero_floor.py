"""Why does the EMPTY circuit under zero ablation read accuracy ~0.5 in eval_mib.py's pipeline
and exactly 0.0 in MIB's run_evaluation.py pipeline? (2026-09-20)

Both call the same evaluate_graph(intervention='zero') on the same model; the only thing that
differs is the Graph object. This script builds the graph both ways on one small cell and, at a
few top-n sizes, prints what is in the graph and the per-example logit-difference distribution
(how many are exactly 0, positive, negative). Read the n=0 rows: identical graphs must give
identical metrics.

    uv run python scripts/mib/debug_zero_floor.py [--task ioi --model gpt2 --head 100]
"""
import argparse
import sys
from functools import partial

import torch

from learning_to_attribute.deps import find_mib_path

mib_path = find_mib_path(None)
sys.path.insert(0, str(mib_path))
sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))

from eap.graph import Graph                                   # noqa: E402
from eap.evaluate import evaluate_graph, evaluate_baseline    # noqa: E402
from MIB_circuit_track.dataset import HFEAPDataset            # noqa: E402
from MIB_circuit_track.metrics import get_metric              # noqa: E402
from transformer_lens import HookedTransformer                # noqa: E402

MODEL_TL_NAMES = {"gpt2": "gpt2-small", "qwen2.5": "Qwen/Qwen2.5-0.5B",
                  "gemma2": "google/gemma-2-2b", "llama3": "meta-llama/Llama-3.1-8B"}
TASKS_TO_HF = {"ioi": "ioi", "mcqa": "copycolors_mcqa", "arithmetic_addition": "arithmetic_addition",
               "arithmetic_subtraction": "arithmetic_subtraction", "arc_easy": "arc_easy",
               "arc_challenge": "arc_challenge"}


def describe(tag, graph, model, dataloader, metric):
    ex = evaluate_graph(model, graph, dataloader, metric, quiet=True, intervention="zero").float()
    n0 = int((ex == 0).sum()); npos = int((ex > 0).sum()); nneg = int((ex < 0).sum())
    print(f"  {tag:<28s} nodes_in={int(graph.nodes_in_graph.sum()):4d} input_in={bool(graph.nodes_in_graph[0])!s:5s} "
          f"edges_in={int(graph.in_graph.sum()):6d} wec={graph.weighted_edge_count():8.1f} | "
          f"metric ==0:{n0:3d} >0:{npos:3d} <0:{nneg:3d}  mean={ex.mean():+.4f}  max|.|={ex.abs().max():.2e}  "
          f"first5={[round(float(v), 4) for v in ex[:5]]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--head", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--random-json", default=None,
                    help="a run_evaluation-style importances.json (e.g. the Random graph) to load "
                         "with Graph.from_json, as the MIB runner does")
    ap.add_argument("--ns", type=int, nargs="+", default=[0, 1, 2, 7])
    a = ap.parse_args()

    tl_name = MODEL_TL_NAMES[a.model]
    if a.model in ("gemma2", "llama3", "qwen2.5"):
        model = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(tl_name)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF[a.task]}", model.tokenizer, split="test",
                      task=a.task, model_name=a.model)
    ds.head(a.head)
    dl = ds.to_dataloader(batch_size=a.batch_size)
    metric = partial(get_metric("logit_diff", a.task, model.tokenizer, model), mean=False, loss=False)

    base = evaluate_baseline(model, dl, metric, quiet=True).float()
    print(f"baseline (no intervention): mean={base.mean():+.4f} >0:{int((base > 0).sum())}/{len(base)}")

    # --- graph B: eval_mib.py's construction: from_model + random scores, input = max+1 -------
    torch.manual_seed(0)
    gB = Graph.from_model(model, node_scores=True)
    scores = torch.full((gB.n_forward,), float("nan"))
    rnd = torch.rand(gB.n_forward)
    for name, node in gB.nodes.items():
        if name == "logits":
            continue
        idx = gB.forward_index(node, attn_slice=False)
        if idx >= gB.n_forward:
            continue
        scores[idx] = rnd[idx]
    scores[0] = float(rnd.max()) + 1.0          # eval_mib: input always kept
    gB.nodes_scores = scores.clone()
    # --- graph C: same, input unscored (NaN -> "unscored nodes stay in graph") ---------------
    gC = Graph.from_model(model, node_scores=True)
    sC = scores.clone(); sC[0] = float("nan"); gC.nodes_scores = sC
    # --- graph D: same, input scored LOW (so it is never selected at small n) ---------------
    gD = Graph.from_model(model, node_scores=True)
    sD = scores.clone(); sD[0] = -1.0; gD.nodes_scores = sD
    graphs = [("from_model, input=max+1", gB), ("from_model, input=NaN", gC), ("from_model, input=-1", gD)]
    # --- graph A: the MIB runner's construction: Graph.from_json(importances.json) -------------
    if a.random_json:
        gA = Graph.from_json(a.random_json)
        print(f"from_json: n_forward={gA.n_forward} scored={int((~torch.isnan(gA.nodes_scores)).sum())} "
              f"input score={float(gA.nodes_scores[0]):.4g} in_graph.sum()={int(gA.in_graph.sum())} "
              f"nodes_in_graph.sum()={int(gA.nodes_in_graph.sum())}")
        graphs.insert(0, ("from_json (runner)", gA))
    for tag, g in graphs:
        print(f"[{tag}] n_forward={g.n_forward} scored={int((~torch.isnan(g.nodes_scores)).sum())} "
              f"neurons_in_graph={'None' if g.neurons_in_graph is None else 'set'}")

    # corrupted reference exactly as evaluate_area_under_curve computes it
    for tag, g in graphs:
        g.apply_topn(0, True)
        describe(f"{tag} | ref apply_topn(0,True)", g, model, dl, metric)
    for n in a.ns:
        print(f"--- n = {n}")
        for tag, g in graphs:
            g.apply_topn(n, False, level="node", prune=True)
            describe(tag, g, model, dl, metric)


if __name__ == "__main__":
    main()
