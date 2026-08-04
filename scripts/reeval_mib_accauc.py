"""Backfill acc-AUC into MIB result pkls from ALREADY-SAVED circuits -- no retraining.

This is scripts/reeval_gemma_mib.py generalized off Gemma: same machinery (rebuild the circuit,
call evaluate_area_under_curve, write the record), but any model/task/level, and driven by an
explicit --dirs list instead of a hardcoded table. It exists because acc-AUC was added to the
pipeline after most runs had already been evaluated, so ~40 (dir, cell) pairs have a CPR number
and a trained circuit but no acc_auc.

Node dirs rebuild from <task>_<model>_importances.json (MIB graph format); edge dirs from
<task>_<model>_scores.pt, since eval_mib_edge.py never exported an importances.json and MIB's
run_evaluation.py therefore has nothing to consume -- which is why edge acc-AUC cannot go
through run_accauc_mattr.sh and needs this in-process path.

CPR IS NEVER OVERWRITTEN. If the target pkl already carries area_under, that value (and the
whole faithfulness curve behind it) is kept and only the acc fields are added. Those CPR numbers
are already printed in paper/tabs/mib_results.tex and were produced under a different
transformer_lens minor; silently re-deriving them here would move published cells as a side
effect of adding a column. Pass --clobber-cpr to take the fresh numbers instead.

Example count is REPRODUCED, not re-chosen: whatever the original run used (`eval_examples` in
the saved scores.pt args) is what the re-eval uses, so the new acc-AUC is measured on exactly
the cells the CPR number next to it was measured on. Node dirs have no scores.pt for some
methods, so they fall back to the run_accauc_mattr.sh cap rule (llama3 -> 200, gemma2/ioi ->
200, else full validation), which is what produced every acc-AUC already in the table.

RUN IN THE MIB VENV (TL 2.x -- the L2A venv's TL 3.2.1 has a Gemma-2 forward bug), from the L2A
repo root:
  PYTHONPATH=MIB-circuit-track:MIB-circuit-track/EAP-IG/src \
    MIB-circuit-track/.venv/bin/python scripts/reeval_mib_accauc.py \
      --level edge --split validation --model gpt2 --task ioi --dirs mib_edge_detached_tau
"""
import argparse
import pickle
from pathlib import Path

import torch

# Copied from eval_mib.py rather than imported: this runs in the MIB venv, which does not have
# the learning_to_attribute package installed, so `import eval_mib` dies at its own imports.
# reeval_gemma_mib.py inlines them for the same reason. Keep in sync with eval_mib.py.
MODEL_TL_NAMES = {"gpt2": "gpt2-small", "qwen2.5": "Qwen/Qwen2.5-0.5B",
                  "gemma2": "google/gemma-2-2b", "llama3": "meta-llama/Llama-3.1-8B"}
TASKS_TO_HF = {"ioi": "ioi", "mcqa": "copycolors_mcqa",
               "arithmetic_addition": "arithmetic_addition",
               "arithmetic_subtraction": "arithmetic_subtraction",
               "arc_easy": "arc_easy", "arc_challenge": "arc_challenge"}

L2A = Path("/home/guests/aryaman/learning-to-attribute")
R = L2A / "results"
MIBR = Path("/home/guests/aryaman/MIB-circuit-track/results")

# Where the node acc-AUC tables read from (scripts/make_mib_accauc_table.py). The lr05 pair keep
# acc_auc inside their own eval_mib pkl and are not mirrored; final_node/htk_lr_0.05 live in the
# _val dir. Everything else goes to mattr_accauc, alongside the 12 variants run_accauc_mattr.sh
# already produced -- a new location would just be a second thing every reader has to know about.
ACCAUC_VAL = {"final_node", "htk_lr_0.05"}
ACCAUC_LR05 = {"topklog_lr_0.05", "htklog_lr_0.05"}


def default_head(model, task, split):
    """run_accauc_mattr.sh's cap rule, used only when the dir has no saved args to copy."""
    if split == "test":
        return 0                      # test splits are <=1188 examples and scored in full
    if model == "llama3":
        return 200                    # 8B full validation OOMs / crawls; this is the dagger
    if model == "gemma2" and task == "ioi":
        return 200                    # ioi validation is 10k; the baseline sweep capped it
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["node", "edge"])
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--model", required=True, choices=list(MODEL_TL_NAMES))
    ap.add_argument("--task", required=True, choices=list(TASKS_TO_HF))
    ap.add_argument("--dirs", required=True, help="comma-separated results/ dirs")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--clobber-cpr", action="store_true",
                    help="also replace area_under/faithfulnesses with the fresh eval")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip cells whose pkl already has acc_auc (default: recompute)")
    args = ap.parse_args()

    from functools import partial
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    import importlib.metadata as IM
    assert IM.version("transformer_lens").startswith("2."), \
        f"MUST run in the MIB venv (TL 2.x); got {IM.version('transformer_lens')}"

    task, model, split = args.task, args.model, args.split
    tdash = task.replace("_", "-")
    dir_list = [d for d in args.dirs.split(",") if d]

    # The example cap has to be settled BEFORE the dataset is built, and it must be the same for
    # every dir in this job -- so take it from the first dir that saved its args, and only fall
    # back to the rule if none did. Mixing caps within one job would put two different n's in one
    # column of the table.
    head = None
    for d in dir_list:
        sp = R / d / f"{task}_{model}_scores.pt"
        if sp.exists():
            try:
                head = torch.load(sp, map_location="cpu", weights_only=False)["args"].get(
                    "eval_examples") or 0
                print(f"  cap {head or 'full'} copied from {d}/{sp.name}")
                break
            except Exception:
                pass
    if head is None:
        head = default_head(model, task, split)
        print(f"  cap {head or 'full'} from the default rule (no saved args)")

    tl_name = MODEL_TL_NAMES[model]
    if model in ("gemma2", "llama3", "qwen2.5"):
        tl_model = HookedTransformer.from_pretrained(
            tl_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        tl_model = HookedTransformer.from_pretrained(tl_name)
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.cfg.ungroup_grouped_query_attention = True

    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF[task]}", tl_model.tokenizer, split=split,
                      task=task, model_name=model)
    if head:
        ds.head(head)
    dl = ds.to_dataloader(batch_size=args.batch_size)
    am = partial(get_metric("logit_diff", task, tl_model.tokenizer, tl_model),
                 mean=False, loss=False)
    print(f"== {args.level}/{split}/{task}/{model}  n={len(ds)}  "
          f"TL {IM.version('transformer_lens')} ==")

    for d in dir_list:
        tgt = R / d / f"{task}_{model}_{split}.pkl"
        old = pickle.load(open(tgt, "rb")) if tgt.exists() else {}
        if args.skip_done and old.get("acc_auc") is not None:
            print(f"  {d:38s} SKIP (acc_auc already present)")
            continue
        circ = R / d / (f"{task}_{model}_importances.json" if args.level == "node"
                        else f"{task}_{model}_scores.pt")
        if not circ.exists():
            # a missing circuit is a TRAINING gap, not an eval gap -- say which, because the two
            # have wildly different costs and this script can only fix the second.
            print(f"  {d:38s} SKIP (no circuit -- needs training, not re-eval)")
            continue

        if args.level == "node":
            g = Graph.from_json(str(circ))
        else:
            g = Graph.from_model(tl_model)
            sc = torch.load(circ, map_location="cpu", weights_only=False)["scores"]
            real = g.real_edge_mask.bool()
            assert sc.numel() == int(real.sum()), \
                f"{d}: score len {sc.numel()} != real edges {int(real.sum())}"
            g.scores[:] = float("-inf")
            g.scores[real] = sc.to(g.scores.dtype)

        out = evaluate_area_under_curve(tl_model, g, dl, am, level=args.level, absolute=False)
        wec, area_under, area_from_1, average, faith, acc, acc_auc = out
        rec = {"weighted_edge_counts": wec, "area_under": area_under, "area_from_1": area_from_1,
               "average": average, "faithfulnesses": faith, "accuracies": acc, "acc_auc": acc_auc}
        if not args.clobber_cpr and old.get("area_under") is not None:
            for k in ("weighted_edge_counts", "area_under", "area_from_1", "average",
                      "faithfulnesses"):
                if k in old:
                    rec[k] = old[k]
        pickle.dump(rec, open(tgt, "wb"))
        drift = "" if old.get("area_under") is None else \
            f"  (CPR kept {old['area_under']:.3f}; fresh was {area_under:.3f})"
        print(f"  {d:38s} accAUC {acc_auc:.3f} -> {tgt.name}{drift}")

        # mirror into the acc-AUC table's location for node/validation, matching the layout
        # run_accauc_mattr.sh writes so make_mib_accauc_table.acc_mattr finds it unchanged.
        if args.level == "node" and split == "validation" and d not in ACCAUC_LR05:
            base = MIBR / ("mattr_accauc_val" if d in ACCAUC_VAL else "mattr_accauc")
            p2 = base / f"{d}_patching_node" / f"{tdash}_{model}_validation_abs-False.pkl"
            p2.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(rec, open(p2, "wb"))
            print(f"      + {base.name}/{p2.parent.name}/{p2.name}")


if __name__ == "__main__":
    main()
