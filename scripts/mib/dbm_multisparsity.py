"""CPR and IIA log-AUC for DBM's sparsity ladder, scored ON MIB's OWN GRID.

Reads what scripts/mib/eval_dbm_multisparsity.py measured -- each trained mask evaluated at the
L0 it converged to -- and turns the ladder into the two scalars the MIB tables report. ONE
implementation, imported by scripts/mib/make_mib_test_table.py and by the plot scripts, so the
table and the figures cannot disagree about what "DBM (multi-sparsity)" means.

THE PROTOCOL (2026-09-09). Scored on MIB's own ten proportions, so the row is comparable to
every existing MIB run. At each proportion the value is the one measured for the DENSEST TRAINED
MASK THAT FITS THE BUDGET -- the rung with the largest own-L0 <= int(p * n_scored) -- held until
a denser rung becomes affordable. Below the sparsest rung nothing fits, so the measured EMPTY
circuit is used. At p = 1 the FULL circuit is used. Then MIB's two trapezoid rules, unchanged.

Every value on the curve is therefore something that was actually measured on a trained mask.
Nothing is interpolated.

*** DO NOT SWITCH TO INTERPOLATION BETWEEN RUNGS. *** `curve(..., mode="interp")` exists and is
kept only so this note has something to point at. Resampling the ladder onto MIB's grid by
linear interpolation looks like the more natural way to put a sparsely-sampled curve on a common
axis, and it is wrong here for two reasons that were both measured:

  1. It changes nothing. A trapezoid over a piecewise-linear curve is grid-independent, so
     interpolating 7 knots onto 10 points and integrating gives the same number (to <= 0.011 on
     all four cells tested) as integrating over the 7 knots -- i.e. it keeps the coarse-sampling
     credit that scoring on MIB's grid was supposed to remove.
  2. It invents behaviour where there is none. Accuracy is close to a step, and interpolating in
     log p between the empty circuit at p = 0.001 and a sparsest rung at p ~ 0.13 puts a third of
     the log range below that rung: on mcqa/gemma2 and mcqa/qwen2.5 the interpolated curve claims
     ~0.33 and ~0.38 accuracy for a ONE-NODE circuit, where the measurements say 0.
     Worth +0.27 and +0.30 IIA over the honest value. The control: MAttr's own measured curve,
     subsampled to these 7 knots and interpolated back, gains +0.17 and +0.20 -- so the gain is
     a property of coarse sampling, not of DBM.

*** WHY MIB'S GRID AND NOT THE RUNGS' OWN. *** The first version of this file integrated over the
rungs' converged proportions, and that inflated IIA badly. Accuracy is close to a step function
-- near 0 until the circuit works, then 1 -- and the log-weighted trapezoid draws a straight line
between samples, so a grid with few points below the step collects area that a finer grid reads
as 0. Measured: MAttr's OWN accuracy curve, integrated over the rungs' 7-point grid instead of
MIB's 10, rose from 0.495 to 0.667 on mcqa/gemma2 and 0.493 to 0.697 on mcqa/qwen2.5 -- i.e. the
whole apparent DBM advantage was the grid, and it reversed once both were scored on the same one.
CPR was barely affected (<= 0.013) because its linear weight sits at the dense end. Never compare
AUCs computed over different x grids.

*** WHY p = 1 IS FORCED TO THE FULL CIRCUIT. *** Under the step rule alone, at p = 1 the row
would keep its densest rung (86 of 157 nodes on ioi/gpt2, faithfulness 2.59) rather than all of
them. The 0.5 -> 1 interval carries half of CPR's linear weight, so that one point decided the
comparison: with it, DBM beat every MAttr arm on CPR in all four landed cells (e.g. 1.999 vs
1.703 on ioi/gpt2); with the full circuit it loses in all four (1.659 vs 1.703). Every other row
in the table spends its whole budget at p = 1, where faithfulness is 1 by the definition of the
normalisation. Letting one row decline its budget is not a like-for-like comparison.

*** WHAT THE ROW ACTUALLY LOSES ON, measured 2026-09-09 on the 8-rung ladder. *** Decomposing
the IIA gap to MAttr grid point by grid point puts essentially all of it at p=0.05 and p=0.1 --
100% of mcqa/qwen2.5's 0.103 sits at p=0.05 alone. It is NOT that DBM's masks are worse at a
given size: at k=26 on that cell DBM scores accuracy 1.000 against MAttr's 0.920 at k=18. It is
that the ladder has no rung between k=8 and k=26, so the budget p=0.05 (k <= 18) falls back to
the k=8 mask and scores 0.040. The deficit is ladder RESOLUTION in the band k ~ [0.02N, 0.1N].
ioi/gpt2 is the control: its ladder happens to place rungs at k=1,5,15,27 through that band and
its deficit is 0.003. Extending the ladder to lambda=200 -- i.e. filling the SPARSE END -- moved
IIA by +0.000 to +0.005, because that was never where the gap was. See the L1S note in
scripts/mib/eval_dbm_multisparsity.py for why refining it further was declined.

*** STILL NOT THE SAME OBJECT AS THE ROWS AROUND IT, and the caption must say so. *** Eight
separately trained masks against every other row's single ranking, which the table's cost column
carries as 21k against 3k. A ranking yields every budget from one run; a per-lambda mask yields
one point per run -- and the paragraph above is the sharp form of that cost: buying the missing
budgets means buying more runs.
"""
import json
import math
from pathlib import Path

RESULTS = Path("results/dbm_multisparsity")
PCT = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.)


def load(task, model, split="test", base=RESULTS):
    p = Path(base) / f"{task}_{model}_{split}.json"
    return json.load(open(p)) if p.exists() else None


def curve(d, mode="step"):
    """(faithfulness, accuracy) at each of MIB's ten proportions.

    mode="step"    densest affordable mask, value held to the next rung -- THE REPORTED PROTOCOL
    mode="interp"  linear resampling of the ladder; kept only as the counter-example the module
                   docstring documents, never for a reported number
    """
    n = d["n_nodes"]
    rungs = sorted(d["points"], key=lambda r: r["k"])
    empty_f, empty_a = d["faith_curve"][0], d["acc_curve"][0]     # measured, not assumed
    full_f, full_a = d["faith_curve"][-1], d["acc_curve"][-1]
    if mode == "step":
        fs, accs = [], []
        for p in PCT:
            if p >= 1.0:
                fs.append(full_f); accs.append(full_a); continue
            fit = [r for r in rungs if r["k"] <= int(p * n)]
            fs.append(fit[-1]["faithfulness"] if fit else empty_f)
            accs.append(fit[-1]["accuracy"] if fit else empty_a)
        return fs, accs

    # Knots: the measured empty circuit at MIB's own lower limit, every rung at its converged
    # size, and the full circuit. p rather than k so the anchors need no special case.
    xs = [PCT[0]] + [r["k"] / n for r in rungs] + [1.0]
    ys_f = [empty_f] + [r["faithfulness"] for r in rungs] + [full_f]
    ys_a = [empty_a] + [r["accuracy"] for r in rungs] + [full_a]
    # A rung at or below the lower limit replaces the empty anchor rather than duplicating its x,
    # which would make the interpolation ill-defined.
    keep = [0] + [i for i in range(1, len(xs)) if xs[i] > xs[i - 1]]
    xs = [xs[i] for i in keep]; ys_f = [ys_f[i] for i in keep]; ys_a = [ys_a[i] for i in keep]
    lxs = [math.log(x) for x in xs]

    def interp(x, xg, yg):
        if x <= xg[0]:
            return yg[0]
        for i in range(1, len(xg)):
            if x <= xg[i]:
                t = (x - xg[i - 1]) / (xg[i] - xg[i - 1])
                return yg[i - 1] + t * (yg[i] - yg[i - 1])
        return yg[-1]

    fs = [full_f if p >= 1.0 else interp(p, xs, ys_f) for p in PCT]
    accs = [full_a if p >= 1.0 else interp(math.log(p), lxs, ys_a) for p in PCT]
    return fs, accs


def cpr_iia(d, mode="step"):
    """(CPR, IIA log-AUC, n_rungs) -- MIB's two trapezoid rules over MIB's own grid."""
    fs, accs = curve(d, mode)
    cpr = sum((PCT[i + 1] - PCT[i]) * (fs[i] + fs[i + 1]) / 2 for i in range(len(PCT) - 1))
    lx = [math.log(p) for p in PCT]
    iia = (sum((lx[i + 1] - lx[i]) * (accs[i] + accs[i + 1]) / 2
               for i in range(len(PCT) - 1)) / (lx[-1] - lx[0]))
    return cpr, iia, len(d["points"])


def rung_sets(split="test", base=RESULTS, attempted=False):
    """{cell: sorted lambdas} across every evaluated cell, for the consistency check below.

    attempted=False gives the rungs that LANDED (a point on the frontier); attempted=True gives
    the ladder the eval was ASKED for. The two differ exactly where a rung is degenerate.
    """
    out = {}
    for f in sorted(Path(base).glob(f"*_{split}.json")):
        d = json.load(open(f))
        v = (tuple(sorted(float(x) for x in d.get("l1s_requested", [])))
             if attempted else tuple(sorted(r["l1"] for r in d["points"])))
        out[f"{d['task']}/{d['model']}"] = v
    return out


def check_consistent(split="test", base=RESULTS):
    """Warn when cells were evaluated over different ATTEMPTED ladders.

    The Avg of a row whose cells rest on different ladders is not an average of one method. That
    is a live hazard whenever the ladder is extended, because the eval wave and the training wave
    land asynchronously: a cell evaluated before lambda=200 finished carries 7 rungs and its
    neighbour carries 8.

    *** THE CHECK IS ON THE ATTEMPTED LADDER, NOT ON WHAT LANDED, and that is deliberate. *** An
    L1 penalty does not pin L0, so a lambda that yields a usable mask on one cell collapses to
    zero gates on another -- at lambda=200, 3 of 11 cells train to no open gates at all and the
    eval skips them as degenerate. That is a PROPERTY OF THE METHOD at that sparsity, not a gap
    in the experiment, and under the hold protocol it costs nothing: a rung that fails to train
    simply supplies no point, and the budgets it would have covered stay with the next densest
    mask. Warning on it would fire on every future run and train the reader to ignore the one
    case that matters -- a cell evaluated against a stale ladder. Collapsed rungs are REPORTED
    below rather than warned about. Returns the distinct attempted ladders found.
    """
    import sys as _s
    tried = rung_sets(split, base, attempted=True)
    landed = rung_sets(split, base, attempted=False)
    distinct = sorted({v for v in tried.values()})
    if len(distinct) > 1:
        print(f"  WARNING DBM (multi-sparsity): {len(distinct)} different ATTEMPTED ladders "
              f"across {len(tried)} cells -- re-run the eval so every cell uses the same one:",
              file=_s.stderr)
        for v in distinct:
            who = [c for c, x in tried.items() if x == v]
            print(f"    {list(v)}: {who}", file=_s.stderr)
    # "No point" has THREE causes and the json does not separate them, so the note does not
    # guess. A rung contributes nothing when it (a) collapsed to zero open gates -- degenerate,
    # expected at high lambda; (b) had not finished training when the eval ran -- stale, and the
    # one case worth re-running; or (c) COLLIDED with a neighbour, converging to the same L0 and
    # hence the same proportion, where only the first is kept. (c) is real, not hypothetical:
    # on arc_easy/gemma2 lambda 0.2 and 0.6 both land at L0 ~ 138 of 234. It is harmless -- two
    # masks of equal size are two samples of one budget, and the frontier wants one -- but it
    # means a short ladder is not by itself evidence of a failed run. The eval log separates all
    # three ("degenerate run, skipped" / "SKIP l1=...: no graph/scores" / neither).
    for c in sorted(tried):
        missing = [x for x in tried[c] if x not in landed.get(c, ())]
        if missing:
            print(f"  note {c}: lambda {missing} contributed no point "
                  f"({len(landed.get(c, ()))} rungs on the frontier); see the eval log to tell "
                  f"a collapsed rung from an untrained one", file=_s.stderr)
    return distinct


def cell(task, model, split="test", base=RESULTS, mode="step"):
    """(CPR, IIA, n_rungs) or None when the cell has not been evaluated.

    Deliberately RECOMPUTES rather than reading the `cpr`/`iia` the eval script stored: those
    were integrated over the rungs' own grid, which the note above explains is not comparable.
    Keeping the protocol here means changing it needs no GPU time.
    """
    d = load(task, model, split, base)
    return cpr_iia(d, mode) if d else None
