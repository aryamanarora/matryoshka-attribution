"""plot_mib_test_avg with a second ROW facet: test acc AUC under the CPR AUC row.

Same two columns (node / edge), same bars in the same order, plus a second row of panels
showing each method's test acc-AUC average -- the log-weighted decision accuracy that
make_mib_accauc_table.py reports on validation, here read from the SAME test pkls the CPR
row is read from. The point of stacking them on one x axis: CPR rewards gap-padding
(see the budget-sweep note in make_mib_accauc_table.py), so a method can only be called
ahead if it leads in both rows, and a shared x is what lets the eye check that per bar.

EVERYTHING STRUCTURAL IS IMPORTED FROM plot_mib_test_avg, NOT COPIED: FAMILY, panel_rows()
(so DROP, the uniform-k recolour and the ordering guards all apply), tex_to_mpl, avg(), the
size constants. The CPR values here are therefore the parent figure's values by construction,
and a dir repointed in make_mib_test_table.py lands in both figures with no edit here.

WHERE THE ACC NUMBERS COME FROM. No new eval pass: run_evaluation.py and eval_mib.py have
both stored acc_auc in every pkl they write since MIB's evaluation.py started returning it,
and all 19 rows this figure draws have 11/11 test acc cells on disk (checked 2026-08-26).
The two loaders below mirror make_mib_test_table.load_cpr_auc / load_run_eval_cpr exactly,
reading "acc_auc" where those read "area_under" -- same files, different key, so the two rows
of a column describe the same run or the file itself is inconsistent.

THE ROWS THAT CANNOT HAVE AN ACC BAR are the literals transcribed from MIB's Table 1
(Random at node level, EAP-IG-inp (CF) at edge level -- the other literals are already in
the parent's DROP). MIB publishes no acc AUC, so their acc panels carry "n/a" text rather
than a bar. NO_ACC whitelists exactly those names: a real run that fell out of the dir maps
raises instead of quietly rendering as a literal, because "n/a" must mean "the source has no
number", never "the lookup missed".

BAR ORDER IS THE CPR ORDER IN BOTH ROWS -- the table's row order, via panel_rows(). The acc
row is NOT re-sorted by acc: the shared x axis is the figure's reason to exist, and where the
acc ordering disagrees with the CPR one (it does; that is the finding) the disagreement shows
as non-monotone bars, which is the honest rendering of it.

Raw matplotlib for the parent's reasons (free y across columns + width_ratios), doubled:
free y across ROWS too (CPR tops out near 6.5, acc near 1.0), which facet_grid also cannot
give per-cell. Theme via palette.RC / palette.furnish as everywhere else.

Run:  uv run python plots/plot_mib_test_avg_accauc.py
Out:  plots/mib_test_avg_accauc.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mib"))
import palette as P                                     # noqa: E402
import make_mib_test_table as T                         # noqa: E402
import plot_mib_test_avg as V                           # the parent figure -- shared structure  # noqa: E402

# Table names with no acc AUC anywhere: MIB Table 1 transcriptions, CPR-only by construction.
NO_ACC = {"Random", "EAP-IG-inp (CF)"}

# name -> how to load its acc_auc, per level. Built from the same constants collect() reads,
# so a repoint there moves the acc row too. Literals are absent on purpose (-> NO_ACC check).
ACC_DIRS = {
    "node": {**{name: ("mib", d) for name, d in T.OUR_NODE_METHODS},
             **{name: ("runeval", d, sub)
                for name, d, sub in list(T.GRAD_NODE_BASELINES)
                + list(T.MASK_NODE_BASELINES) + [T.NODE_PRUNING]},
             # The L1-ladder row: eval_dbm_multisparsity.py writes one JSON per cell with the
             # ladder's CPR (`cpr`, what the test table reads) and its IIA log-AUC (`iia`), the
             # same log-sparsity-weighted accuracy AUC the other rows store as acc_auc.
             T._M.DBM_MULTI_ROW: ("multi", "dbm_multisparsity")},
    "edge": {name: ("mib", d) for name, d in T.OUR_EDGE_METHODS},
}


def load_acc(src, task, model):
    """acc_auc for one test cell, from the same pkl the CPR row reads. None if absent."""
    if src[0] == "multi":                                # eval_dbm_multisparsity.py JSON
        p = T.RESULTS_BASE / src[1] / f"{task}_{model}_test.json"
        if not p.exists():
            return None
        try:
            with open(p) as f:
                v = json.load(f).get("iia")
            return round(v, 2) if v is not None else None
        except Exception:
            return None
    if src[0] == "mib":                                  # eval_mib layout (our runs)
        p = T.RESULTS_BASE / src[1] / f"{task}_{model}_test.pkl"
    else:                                                # run_evaluation.py layout (baselines)
        p = (T.RESULTS_BASE / src[1] / src[2]
             / f"{task.replace('_', '-')}_{model}_test_abs-False.pkl")
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            v = pickle.load(f).get("acc_auc")
        return round(v, 2) if v is not None else None
    except Exception:
        return None


def bars():
    """{level label: [(family, mpl label, cpr avg, acc avg | None)]}, in the parent's order.

    The CPR side repeats the parent's completeness gate (11/11 cells or raise); the acc side
    is held to the same standard for every row that HAS a source -- a partial acc average
    next to a complete CPR one would be the mixed-average defect the parent's gate exists to
    prevent, one metric over.
    """
    loaded, out = T.collect(), {}
    for level, level_lab, dagger in V.LEVELS:
        recs = []
        for fam, name, data in V.panel_rows(level, loaded):
            n = sum(1 for t, m, _ in T.COLUMNS if (t, m) in data)
            if n < len(T.COLUMNS):
                raise SystemExit(
                    f"{name} ({level}) averages {n}/{len(T.COLUMNS)} cells -- a bar cannot "
                    "show that. Add it to DROP, or wait for the missing cells.")
            src = ACC_DIRS[level].get(name)
            if src is None:
                if name not in NO_ACC:
                    raise SystemExit(
                        f"{name} ({level}) has no entry in ACC_DIRS and is not a known "
                        "CPR-only literal (NO_ACC) -- new row in make_mib_test_table.py? "
                        "add its dir here, or its name to NO_ACC if MIB never published acc")
                acc = None
            else:
                cells = [load_acc(src, t, m) for t, m, _ in T.COLUMNS]
                if any(c is None for c in cells):
                    missing = [f"{t}/{m}" for (t, m, _), c in zip(T.COLUMNS, cells) if c is None]
                    raise SystemExit(f"{name} ({level}): no acc_auc for {missing} in {src} -- "
                                     "old pkls predating acc_auc? see make_mib_accauc_table.py")
                acc = round(sum(cells) / len(cells), 2)
            lab = V.tex_to_mpl(name) + ("$^{\\dagger}$" if fam in dagger else "")
            recs.append((fam, lab, V.avg(data), acc))
        out[level_lab] = recs
    return out


def draw(ax, recs, which, ylabel, na_ok):
    """One panel: bars for metric index `which` (2=CPR, 3=acc), value printed over each."""
    xs = range(len(recs))
    vals = [r[which] for r in recs]
    ax.bar(xs, [0 if v is None else v for v in vals], width=V.BAR_W,
           color=[V.FAMILY[r[0]][1] for r in recs], lw=0, zorder=2)
    top = max(v for v in vals if v is not None) * 1.12   # headroom for the value labels
    for x, v in zip(xs, vals):
        if v is None:
            if not na_ok:
                raise SystemExit("missing value in a CPR panel -- collect() should prevent this")
            # Text where the bar would be, so the gap reads as "no published number", not as a
            # zero-height bar or a rendering accident.
            ax.annotate("n/a", (x, 0), textcoords="offset points", xytext=(0, 1.5),
                        ha="center", va="bottom", fontsize=V.FS_ANNOT, color="#888888", zorder=3)
        else:
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 1.5),
                        ha="center", va="bottom", fontsize=V.FS_ANNOT, zorder=3)
    ax.set_xlim(-0.5 - V.BAR_W / 4, len(recs) - 0.5 + V.BAR_W / 4)
    ax.set_ylim(0, top)
    ax.set_ylabel(ylabel, fontsize=V.FS_AXIS)
    P.furnish(ax)
    ax.grid(False, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=V.FS_TICK)
    ax.tick_params(axis="x", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/mib_test_avg_accauc.pdf")
    a = ap.parse_args()

    data = bars()
    plt.rcParams.update(P.RC)
    counts = [len(data[lab]) for _, lab, _ in V.LEVELS]
    fh = 2 * V.PANEL_H + V.HEAD + V.FOOT
    # sharex="col": the whole point is reading one method down both metrics, so the columns
    # must not be allowed to disagree about x -- and the tick labels then draw once, on the
    # bottom row, which is also where the parent's FOOT budget for them goes.
    fig, axes = plt.subplots(2, len(V.LEVELS), figsize=(V.FIG_W, fh), sharex="col",
                             gridspec_kw=dict(width_ratios=counts))
    for col, (_, level_lab, _) in enumerate(V.LEVELS):
        recs = axes[0][col], axes[1][col]
        # Two-line ylabels, unlike the parent's one-liners: at PANEL_H the single-line
        # "Node-level CPR AUC (↑)" is taller than its panel and collides with the row below.
        draw(recs[0], data[level_lab], 2, f"{level_lab}\nCPR AUC (↑)", na_ok=False)
        draw(recs[1], data[level_lab], 3, f"{level_lab}\nacc AUC (↑)", na_ok=True)
        recs[1].set_xticks(range(len(data[level_lab])))
        recs[1].set_xticklabels([lab for _, lab, _, _ in data[level_lab]],
                                fontsize=V.FS_TICK, rotation=45,
                                ha="right", rotation_mode="anchor")

    fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=0.6)
    top = 1.0 - V.HEAD / fh
    fig.subplots_adjust(top=top)
    # V.LEGEND, not V.FAMILY.values(): the uniform-k rows share the log-k hue, so the family
    # dict now holds two keys with one colour and only the de-duplicated LEGEND may be drawn.
    fig.legend(handles=[Patch(facecolor=c, label=V.tex_to_mpl(lab))
                        for lab, c in V.LEGEND],
               fontsize=V.FS_TICK, ncol=len(V.LEGEND), loc="lower center",
               bbox_to_anchor=(0.5, top), frameon=False, handlelength=1.2, handleheight=1.0,
               handletextpad=0.4, columnspacing=1.4)
    fig.savefig(a.out)
    print("wrote", a.out)

    for _, level_lab, _ in V.LEVELS:
        print(f"\n{level_lab}:  (CPR AUC / acc AUC)")
        for fam, lab, cpr, acc in data[level_lab]:
            acc_s = f"{acc:.2f}" if acc is not None else "n/a"
            print(f"  {V.FAMILY[fam][0]:<20} {lab:<26} {cpr:.2f} / {acc_s}")


if __name__ == "__main__":
    sys.exit(main())
