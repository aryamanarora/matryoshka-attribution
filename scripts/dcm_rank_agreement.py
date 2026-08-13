"""Does DCM's circuit agree with the other node methods, or is it a coin flip?

The motivating question is "DCM is better than random, surely?" -- and rank correlation alone
does not answer it, because most of what DCM's rho measures is not DCM's circuit. Its raw
scores are ``clamp_(0,1)`` output, piled onto exactly 0.0 and 1.0; ``learn_scores_dcm``
(edge_pruning.py:348) then adds ``tie_eps * last_active/steps``, the step at which each unit
was last inside ``round(mask)``. That trajectory term breaks most of the ties, so rho is
dominated by PRUNING ORDER -- an ordering the optimizer generates as a side effect -- and not
by the 0/1 set the method actually returns.

So this reports three things per DCM run, and they have to be read together:

  ``ties``   fraction of nodes still sharing a score after the tie-break. Low here (0.1-0.5),
             which is the point: the rho next to it is a real ordering, just not the circuit's.
  ``rho``    Spearman vs each reference method's scores, same dirs as the heatmap figure.
  ``ovl``    of the units DCM KEPT (mask >= 0.5), the fraction the reference also puts in its
             top-|kept| by signed score -- MIB ranks with absolute=False, so signed, not |x|.
             This is the honest test: chance level is |kept|/N, in the ``kept/N`` column, and
             beating THAT is what "better than random" means.

Run: uv run python scripts/dcm_rank_agreement.py
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path("results")
R_MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")

# The 3 cells DCM was run on. Everything here is intersected down to these.
CELLS = [("ioi", "gpt2"), ("ioi", "qwen2.5"), ("mcqa", "qwen2.5")]

# Reference methods, same specs/dirs as plot_method_corr_heatmap.METHODS so the numbers are
# comparable to that figure cell-for-cell.
REFS = [
    ("MAttr",        "topklog_lr_0.05",                        "flat"),
    ("+hard",        "htklog_lr_0.05",                         "flat"),
    ("Node Pruning", "eprun_node_s0.5_ld",                     "graph"),
    ("DBM",          "eprun_node_ld_sig_lr0.3_l16.0",          "graph"),
    ("IG-5",         "napig_ref/EAP-IG-inputs_patching_node",  "nested"),
    ("IG-10",        "napig10/EAP-IG-inputs_patching_node",    "nested"),
    ("AttnLRP",      "attnlrp/AttnLRP_patching_node",          "nested"),
    ("GIM",          "gim/GIM_patching_node",                  "nested"),
    ("RelP+QK",      "relp_qkgrad/RelP-qkgrad_patching_node",  "nested"),
    ("I x G",        "ig1/EAP-IG-inputs_patching_node",        "nested"),
]

DCM_RUNS = [(d, lr) for d in ("0.01", "0.05", "0.2") for lr in ("0.01", "0.1", "1.0")]


def load(path):
    if not path.exists():
        return None
    nodes = json.load(open(path)).get("nodes", {})
    return {n: v["score"] for n, v in nodes.items()
            if n not in ("logits", "input") and isinstance(v, dict) and "score" in v}


def scores_for(loc, layout, task, model):
    if layout == "flat":
        return load(R / loc / f"{task}_{model}_importances.json")
    if layout == "graph":
        return load(R / loc / f"graph_{task}_{model}.json")
    return load(R_MIB / loc / f"{task.replace('_', '-')}_{model}" / "importances.json")


def tie_frac(s):
    """Fraction of nodes whose score is not unique. 1.0 = no ranking information at all."""
    v = np.asarray(list(s.values()))
    _, counts = np.unique(v, return_counts=True)
    return float((counts[counts > 1].sum()) / len(v))


def overlap(dcm, ref):
    """|kept set ∩ reference top-k| / k, with k = |kept|. Chance rate is k/N.

    Ranked by SIGNED score, not |score|: every pkl in this project is ``abs-False``, i.e.
    MIB's own top-k took the largest signed values. Using |x| here would rank a strongly
    excluded node (very negative mask logit for Node Pruning / DBM) as a top pick.
    """
    common = sorted(set(dcm) & set(ref))
    kept = [n for n in common if dcm[n] >= 0.5]
    k, N = len(kept), len(common)
    if k == 0 or k == N:
        return None, k, N            # empty or full: the statistic is undefined, not 0 or 1
    top = set(sorted(common, key=lambda n: -ref[n])[:k])
    return len(set(kept) & top) / k, k, N


def main():
    print(f"{'DCM run':>16}  {'cell':>16}  {'kept/N':>9}  {'ties':>5}   "
          + "  ".join(f"{r[0]:>12}" for r in REFS))
    for dens, lr in DCM_RUNS:
        loc = f"eprun_node_ld_dcm_d{dens}_lr{lr}"
        for task, model in CELLS:
            dcm = scores_for(loc, "graph", task, model)
            if not dcm:
                continue
            rhos, aps = [], []
            for _, rloc, rlayout in REFS:
                ref = scores_for(rloc, rlayout, task, model)
                if not ref:
                    rhos.append(np.nan); aps.append(None); continue
                common = sorted(set(dcm) & set(ref))
                rhos.append(spearmanr([dcm[n] for n in common], [ref[n] for n in common])[0]
                            if len(common) >= 4 else np.nan)
                aps.append(overlap(dcm, ref)[0])
            # kept/N is a property of the DCM run alone -- don't derive it from a reference.
            k = sum(1 for v in dcm.values() if v >= 0.5)
            N = len(dcm)
            cells = "  ".join(
                (f"{r:>5.2f}/{a:>5.2f}" if a is not None else f"{r:>5.2f}/{'--':>5}")
                if not np.isnan(r) else f"{'--':>11}"
                for r, a in zip(rhos, aps))
            print(f"{'d'+dens+' lr'+lr:>16}  {task+'/'+model:>16}  {k:>4}/{N:<4}  "
                  f"{tie_frac(dcm):>5.2f}   {cells}")
        print()
    print("cells are  rho / set-overlap.  chance overlap = kept/N (the 'kept/N' column).")


if __name__ == "__main__":
    main()
