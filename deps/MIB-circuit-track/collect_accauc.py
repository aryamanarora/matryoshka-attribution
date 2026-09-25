"""Collect acc-AUC (and CPR-AUC) for every node method across the 12 cells.

Reads each method's ORDINARY eval dir (run_evaluation.py has stored `acc_auc` alongside
`area_under` in every pkl since MIB_circuit_track/evaluation.py started returning it), falling
back to a legacy results/<dir>_accauc/ rerun only for methods scored before that change.
Prints one table per metric: rows = task/model cell, cols = method.

Adding a method: list its ordinary eval dir. No separate acc-AUC pass needs to be run.
Run:  .venv/bin/python collect_accauc.py
"""
import pickle, os, sys

CELLS = [
    ("gpt2", "ioi"), ("qwen2.5", "ioi"), ("qwen2.5", "mcqa"),
    ("gemma2", "ioi"), ("gemma2", "mcqa"), ("gemma2", "arc_easy"),
    ("llama3", "ioi"), ("llama3", "mcqa"), ("llama3", "arithmetic_addition"),
    ("llama3", "arithmetic_subtraction"), ("llama3", "arc_easy"), ("llama3", "arc_challenge"),
]
# tag -> ([output dirs, PRIMARY FIRST], method_name_saveable). GIM/AttnLRP postdate the
# acc_auc change so they have no legacy dir -- and gim_nomlp_accauc must NOT be listed as one,
# it belongs to the pre-scale_mlp_gate GIM.
METHODS = {
    "NAP-IG":    (["napig_repro_eval", "napig_ref_accauc"],   "EAP-IG-inputs_patching_node"),
    "NAP-local": (["napig_local_eval", "napig_local_accauc"], "EAP-IG-inputs-local_patching_node"),
    "IxG(1)":    (["ig1_eval", "ig1_accauc"],                 "EAP-IG-inputs_patching_node"),
    # Compute-matched to IxG(1) -- same one forward+backward per batch, alpha ~ U(0,1) per example
    # instead of the m=1 grid's degenerate alpha=1. Kept adjacent to it in the column order because
    # the ONLY reading of this arm that means anything is the difference between the two.
    "MC-IG(1)":  (["napig_mc_eval"],                          "EAP-IG-inputs-mc_patching_node"),
    "RelP":      (["relp_eval", "relp_accauc"],               "RelP_patching_node"),
    "RelP-qk":   (["relp_qkgrad_eval", "relp_qkgrad_accauc"], "RelP-qkgrad_patching_node"),
    "GIM":       (["gim_eval"],                               "GIM_patching_node"),
    "RelP+Shapley": (["relpshapley_eval", "relpshapley_accauc"], "RelPShapley_patching_node"),
    "AttnLRP":   (["attnlrp_eval"],                           "AttnLRP_patching_node"),
}
BASE = "results"


def load(odirs, mdir, task, model):
    """First listed dir that has this cell WITH an acc_auc; None if none does."""
    stask = task.replace("_", "-")
    for odir in odirs:
        p = f"{BASE}/{odir}/{mdir}/{stask}_{model}_validation_abs-False.pkl"
        if not os.path.exists(p):
            continue
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        # Pre-change pkls exist but carry acc_auc=None; skipping them is what makes the
        # legacy fallback reachable at all.
        if d.get("acc_auc") is not None:
            return d
    return None


def table(metric, label):
    hdr = f"{label + ' | cell':28}" + "".join(f"{m:>11}" for m in METHODS)
    print(hdr); print("-" * len(hdr))
    rows = []
    for model, task in CELLS:
        cell = f"{task}/{model}"
        vals = {}
        for m, (od, md) in METHODS.items():
            d = load(od, md, task, model)
            vals[m] = None if d is None else d.get(metric)
        rows.append(vals)
        s = f"{cell:28}" + "".join(
            (f"{vals[m]:>11.4f}" if vals[m] is not None else f"{'--':>11}") for m in METHODS)
        print(s)
    print("-" * len(hdr))
    for m in METHODS:
        done = [v[m] for v in rows if v[m] is not None]
        mean = sum(done) / len(done) if done else float("nan")
        print(f"  {m:12} mean over {len(done):2}/12 = {mean:.4f}")
    print()


if __name__ == "__main__":
    table("acc_auc", "acc-AUC")
    print("=" * 60)
    table("area_under", "CPR-AUC")
