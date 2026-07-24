"""Re-evaluate ALL MAttr Gemma-2 MIB cells in the MIB venv (transformer_lens 2.15.4, which
matches the HF reference; TL 3.x has a Gemma-2 forward bug). Overwrites the old pkls in place.

WHY: MAttr numbers were produced by scripts/eval_mib{,_edge}.py in the L2A venv (TL 3.2.1),
whose Gemma-2 forward disagrees with HuggingFace (see memory eval-mib-vs-run-evaluation-divergence).
Only the 3 Gemma cells (ioi/mcqa/arc_easy, gemma2) are affected; gpt2/qwen/llama are version-stable.

RUN IN THE MIB VENV, from the L2A repo root, one job per (level, split, task):
  PYTHONPATH=MIB-circuit-track:MIB-circuit-track/EAP-IG/src \
    MIB-circuit-track/.venv/bin/python scripts/reeval_gemma_mib.py --level node --split validation --task arc_easy

For each MAttr dir it rebuilds the circuit (node: Graph.from_json of importances.json; edge:
scores.pt -> graph.scores[real_edge_mask]), runs evaluate_area_under_curve, and OVERWRITES:
  - results/<dir>/<task>_gemma2_<split>.pkl                      (CPR tables + lr05 acc-AUC)
  - <mattr_accauc[_val]>/<dir>_patching_node/<t-dash>_gemma2_validation_abs-False.pkl (ablation acc-AUC)
"""
import argparse
import pickle
from pathlib import Path

import torch

L2A = Path("/home/guests/aryaman/learning-to-attribute")
R = L2A / "results"
MIBR = Path("/home/guests/aryaman/MIB-circuit-track/results")
HFTASK = {"ioi": "ioi", "mcqa": "copycolors_mcqa", "arc_easy": "arc_easy"}
HEAD = {"ioi": 200, "mcqa": 0, "arc_easy": 0}   # match run_accauc.sh baseline caps (ioi gemma2 -> 200)

DIRS = {
    ("node", "validation"): [
        "topklog_lr_0.05", "htklog_lr_0.05", "mib_node_detached_tau_log",
        "mib_node_bernoulli_reinforce_log", "mib_node_identity_sgd_log",
        "mib_node_identity_gumbel_sgd_log", "final_node", "htk_lr_0.05",
        "mib_node_hard_topk_gumbel", "mib_node_detached_tau", "mib_node_bernoulli_reinforce",
        "mib_node_identity_sgd", "mib_node_identity_gumbel_sgd_uniform",
    ],
    ("node", "test"): [
        "test_node_topk_log_lr05", "test_node_hard_topk_log_lr05", "test_node_hard_topk_uniform_lr05",
    ],
    ("edge", "validation"): [
        "mib_edge_topk_log_lr05", "mib_edge_hard_topk_log_lr05", "mib_edge_detached_tau",
        "mib_edge_bernoulli_reinforce", "mib_edge_identity_sgd_log",
        "mib_edge_hard_topk_uniform_lr05", "mib_edge_identity_sgd_uniform",
    ],
    ("edge", "test"): [
        "test_edge_topk_log_lr05", "test_edge_hard_topk_log_lr05", "test_edge_hard_topk_uniform_lr05",
    ],
    # extra dirs used ONLY by cpr_summary.tex + lr_sweep.tex (val only). All Gemma cells here
    # were also eval'd under the buggy L2A venv. Some (e.g. bern_lr_0.05) lack a circuit for
    # certain cells -> skipped per-cell below.
    ("node", "validation_extra"): [
        "mib_node_hard_topk", "mib_node_hard_topk_log", "mib_node_topk_log",       # lr=0.01 anchors
        "htklog_lr_0.005", "htklog_lr_0.1", "htklog_lr_0.3",                         # hard-log LR sweep
        "topklog_lr_0.005", "topklog_lr_0.1", "topklog_lr_0.3",                      # soft-log LR sweep
        "htk_lr_0.005", "htk_lr_0.1", "htk_lr_0.3",                                  # hard-unif LR sweep
        "bern_lr_0.01", "bern_lr_0.05", "bern_lr_0.3",                               # REINFORCE LR sweep
    ],
    ("edge", "validation_extra"): [
        "final_edge", "mib_edge_hard_topk", "mib_edge_hard_topk_uniform",           # cpr_summary edge
    ],
}
# acc-AUC ablation pkl locations (validation, node only); lr05 dirs read acc_auc from results/ pkl.
ACCAUC_VAL = {"final_node", "htk_lr_0.05"}   # -> mattr_accauc_val
ACCAUC_LR05 = {"topklog_lr_0.05", "htklog_lr_0.05"}  # no mattr_accauc write


def build_model(tl_name="google/gemma-2-2b"):
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
    m.cfg.use_split_qkv_input = True
    m.cfg.use_attn_result = True
    m.cfg.use_hook_mlp_in = True
    m.cfg.ungroup_grouped_query_attention = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["node", "edge"])
    ap.add_argument("--split", required=True, choices=["validation", "test", "validation_extra"])
    ap.add_argument("--task", required=True, choices=list(HFTASK))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dirs", default=None,
                    help="comma-separated subset of dirs to run (for parallel chunking); default = all for (level,split)")
    args = ap.parse_args()

    from eap.graph import Graph
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    import importlib.metadata as M
    assert M.version("transformer_lens").startswith("2."), \
        f"MUST run in MIB venv (TL 2.x); got {M.version('transformer_lens')}"

    task, model = args.task, "gemma2"
    tdash = task.replace("_", "-")
    split = "test" if args.split == "test" else "validation"   # actual eval split + pkl suffix
    model_tl = build_model()

    ds = HFEAPDataset(f"mib-bench/{HFTASK[task]}", model_tl.tokenizer, split=split,
                      task=task, model_name=model)
    if HEAD[task]:
        ds.head(HEAD[task])
    dl = ds.to_dataloader(batch_size=args.batch_size)
    from functools import partial
    am = partial(get_metric("logit_diff", task, model_tl.tokenizer, model_tl), mean=False, loss=False)
    print(f"== {args.level}/{args.split}/{task} gemma2  (n={len(ds)}, TL {M.version('transformer_lens')}) ==")

    dir_list = args.dirs.split(",") if args.dirs else DIRS[(args.level, args.split)]
    for d in dir_list:
        # skip cells with no saved circuit (partial coverage, e.g. bern_lr_0.05/ioi)
        circ = R / d / (f"{task}_{model}_importances.json" if args.level == "node"
                        else f"{task}_{model}_scores.pt")
        if not circ.exists():
            print(f"  {d:38s} SKIP (no circuit for {task}/{model})")
            continue
        # rebuild circuit
        if args.level == "node":
            g = Graph.from_json(str(circ))
        else:
            g = Graph.from_model(model_tl)
            sc = torch.load(circ, map_location="cpu")["scores"]
            real = g.real_edge_mask.bool()
            assert sc.numel() == int(real.sum()), f"{d}: score len {sc.numel()} != real edges {int(real.sum())}"
            g.scores[:] = float("-inf")
            g.scores[real] = sc.to(g.scores.dtype)

        out = evaluate_area_under_curve(model_tl, g, dl, am, level=args.level, absolute=False)
        wec, area_under, area_from_1, average, faith, acc, acc_auc = out
        rec = {"weighted_edge_counts": wec, "area_under": area_under, "area_from_1": area_from_1,
               "average": average, "faithfulnesses": faith, "accuracies": acc, "acc_auc": acc_auc}

        # overwrite the eval_mib-format pkl the CPR/lr05-acc tables read
        tgt = R / d / f"{task}_{model}_{split}.pkl"
        old = pickle.load(open(tgt, "rb")) if tgt.exists() else {}
        pickle.dump(rec, open(tgt, "wb"))
        print(f"  {d:38s} CPR {old.get('area_under', float('nan')):.3f}->{area_under:.3f}  "
              f"accAUC {old.get('acc_auc', float('nan')):.3f}->{acc_auc:.3f}  -> {tgt.name}")

        # overwrite the ablation acc-AUC pkl (main-table val, node, non-lr05 dirs only;
        # the *_extra dirs are not in the acc-AUC table)
        if args.split == "validation" and args.level == "node" and d not in ACCAUC_LR05:
            base = MIBR / ("mattr_accauc_val" if d in ACCAUC_VAL else "mattr_accauc")
            ap2 = base / f"{d}_patching_node" / f"{tdash}_{model}_validation_abs-False.pkl"
            ap2.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(rec, open(ap2, "wb"))
            print(f"      + acc-AUC pkl -> {base.name}/{ap2.parent.name}/{ap2.name}")


if __name__ == "__main__":
    main()
