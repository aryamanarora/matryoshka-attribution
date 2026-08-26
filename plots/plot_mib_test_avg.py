"""tabs/mib_test_results.tex's Avg column as a bar chart, grouped by method family.

The test table is 13 columns wide and its point lives entirely in the last one: within each
level, ours > mask-learning > gradient > random. This is that column, nothing else, so the
ordering is readable without tracking eleven per-cell numbers across an adjustbox.

THE NUMBERS ARE THE TABLE'S OWN. make_mib_test_table.collect() is imported and called, so the
loaders, the results dirs and the skip guards are shared -- this figure cannot show a number the
table does not, and a dir repointed there lands here with no edit. The literal baselines
(NODE_BASELINES / EDGE_BASELINES, transcribed from MIB's Table 1) come from the same module's
constants through the same classify() call that assigns their families in the table.

BAR ORDER IS THE TABLE'S ROW ORDER, deliberately, including its worst-to-best-within-group
direction -- so bars grow to the RIGHT inside each family and the tallest bar of a family is its
last. A reader holding both should find the methods in the same sequence, and re-sorting
best-first here would silently produce a different ranking wherever the table's tie-breaking
differs. See make_mib_test_table.by_avg for the rule being mirrored.

FAMILY IS COLOUR, NOT A HEADING. The split is only four values and text headings would need room
the rotated tick labels have already spent. Hues are palette.py's, reused rather than invented,
so the families read the same here as in every other figure: cool blue = ours, warm orange =
gradient attribution, indigo = mask-learning baselines, grey = the random control.

THE TWO PANELS HAVE SEPARATE Y AXES AND MUST NOT BE COMPARED BY BAR HEIGHT. Node-level CPR AUC
tops out near 2.1 and edge-level near 6.5; drawn on one scale the whole node panel is a stub --
which is what a shared axis produced, and why each panel now carries its own ticks AND names the
level in its own axis title, so neither can be read without its scale. The levels are not
comparable anyway: different substrates with different unit counts, and MIB's area_under is not
normalised across them, so a shared axis would offer a comparison that does not exist. Panel
WIDTHS are proportional to their bar counts, for the matching reason -- equal halves would draw
the 5 edge bars three times as wide as the 16 node ones and read as emphasis.

WHY RAW MATPLOTLIB rather than the plotnine default. Exactly the combination above: free y
scales ACROSS COLUMNS plus width_ratios. ggplot2's facet_grid (which plotnine follows) frees y
only along ROWS and x only along columns, so `facet_grid(cols="level", scales="free")` silently
keeps one shared y -- verified, it draws the node panel as a stub. facet_wrap frees both but has
no `space="free_x"`, so the panels come out equal-width. The theme is matched by hand through
palette.RC / palette.furnish.

ONE CAVEAT THE Avg CANNOT SHOW, marked on the labels: at edge level our five llama3 cells are
scored on a 200-example subset (make_mib_test_table's EDGE_DAGGER), so those four Avgs mix capped
and full-split cells. Marked with the table's own dagger. It does not distort the comparison
WITHIN that family -- all four carry it equally -- but EAP-IG-inp beside them is full-split, so
the cross-family edge gap is the one number this affects.

DROPPED relative to the table (the table keeps all three; see DROP):
  NAP (CF), NAP-IG (CF)  MIB-published counterfactual gradient rows whose numbers do not
                         reproduce against our own runs of the same estimators -- NAP is
                         EAP-IG-inputs at m=1 (measured: Pearson = Spearman = 1.0 against our
                         implementation) yet the published row sits far below our I$\\times$G/IG
                         rows, and that gap is unexplained. An unexplained discrepancy is a
                         footnote a table can carry and a bar chart cannot.
  UGS                    averages 3 of the 11 cells, not 11. The table prints it because the
                         edge groups pass suppress_partial=False, and there the eight dashes
                         are right there in the row; a bar carries no such mark and would read
                         as a like-for-like average against 11-cell bars.
With those three gone every remaining bar is an 11/11-cell mean, which main() asserts.

Run:  uv run python plots/plot_mib_test_avg.py
Out:  plots/mib_test_avg.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import palette as P                                     # noqa: E402
import make_mib_test_table as T                         # collect() + the literal baselines  # noqa: E402

# family key -> (legend label, colour). Keys are internal; the labels are what the reader sees.
FAMILY = {
    "control":  ("Random control", P.METHOD["Random"]),
    "gradient": ("Gradient-based", P.METHOD["IG"]),
    "mask":     ("Mask-based", P.METHOD["Node Pruning"]),
    "ours":     ("\\ourmethod{} (ours)", P.METHOD["MAttr"]),
}
# Baseline rows the table keeps and this figure does not -- see the docstring for each. Matched
# against NODE_BASELINES / EDGE_BASELINES keys, and a name here that matches nothing raises
# rather than silently doing nothing, so a rename in the table cannot quietly un-drop a row.
DROP = ("NAP (CF)", "NAP-IG (CF)", "UGS")
LEVELS = [("node", "Node-level", ()), ("edge", "Edge-level", ("ours",))]

# 5.5in is iclr2026_conference.sty's \textwidth verbatim (line 49), so at width=\linewidth the
# figure is placed 1:1 and the sizes below are the sizes that reach the compiled PDF. Most other
# figures in paper/figs are drawn at 5.5 or 5.4 for the same reason; do not draw this one smaller
# and let LaTeX upscale it, which would push the tick labels past 7pt.
FIG_W = 5.5
# The plotting rect itself. Kept SHORT on purpose: this figure carries one number per method, so
# the y extent is doing no work beyond ordering the bars -- height here is spent on nothing and
# costs a column inch on a 9-page limit. FOOT is what the 45-degree tick labels need below the
# axis ("EAP-IG-inp (CF)" is the longest); tight_layout will steal it back from PANEL_H if it is
# short, shrinking the bars rather than clipping, so check the ink bbox after changing either.
PANEL_H = 1.05
HEAD, FOOT = 0.22, 0.62
FS_AXIS, FS_TICK, FS_ANNOT = 7, 6.5, 6
BAR_W = 0.72           # in category units, so bars are the same thickness in both panels


def tex_to_mpl(s):
    """A table row label as mathtext. Only the forms this table actually uses are handled.

    Deliberately NOT a general LaTeX translator: a silent partial translation would put a stray
    backslash in the paper, so anything unrecognised RAISES. \\ourmethod{} -> MAttr follows
    plot_lr_sweep_summary.main()'s replace, for the same reason -- the macro is undefined here.
    """
    s = s.replace("\\ourmethod{}", "MAttr")
    s = re.sub(r"\{=\}", "$=$", s.replace("$", "\x00")).replace("\x00", "$")
    s = s.replace("$$", "")
    if "\\" in s.replace("\\times", ""):
        raise SystemExit(f"unhandled LaTeX in row label {s!r} -- extend tex_to_mpl()")
    return s


def avg(data):
    """Mean over the cells present, matching make_mib_test_table.main()'s row_avg."""
    vs = [v for v in (data.get((t, m)) for t, m, _ in T.COLUMNS) if v is not None]
    return round(sum(vs) / len(vs), 2) if vs else None


def ordered(rows, suppress_partial):
    """Family members in the table's row order -- worst-to-best by Avg, incomplete rows last.

    Mirrors make_mib_test_table.main()'s nested by_avg(); kept as a copy rather than imported
    because it closes over that function's locals. If the table's ordering rule changes, change
    it here too -- the docstring's promise that the two agree is the thing being maintained.
    """
    def key(item):
        _, data = item
        a = None if (suppress_partial and len(data) < len(T.COLUMNS)) else avg(data)
        return (a is None, a if a is not None else 0.0)
    return sorted(rows, key=key)


def panel_rows(level, loaded):
    """[(family, name, data)] for one level, in table order, with DROP applied."""
    ours_nodes, mask_nodes, grad_nodes, ours_edges = loaded
    if level == "node":
        groups = [
            ("control", [(n, T.NODE_BASELINES[n])
                         for n in T.CONTROL_NAMES if n in T.NODE_BASELINES], True),
            ("gradient", T.classify(T.NODE_BASELINES, "gradient") + list(grad_nodes.items()), True),
            ("mask", list(mask_nodes.items()), True),
            # complete_or_skip already held ours to 11/11, so suppress_partial has nothing to act
            # on -- passed explicitly here for the same reason main() passes it explicitly.
            ("ours", list(ours_nodes.items()), False),
        ]
    else:
        groups = [
            ("gradient", T.classify(T.EDGE_BASELINES, "gradient"), False),
            ("mask", T.classify(T.EDGE_BASELINES, "mask"), False),
            ("ours", list(ours_edges.items()), False),
        ]
    out = []
    for fam, rows, sup in groups:
        for name, data in ordered([r for r in rows if r[0] not in DROP], sup):
            out.append((fam, name, data))
    return out


def bars():
    """{level label: [(family, mpl label, value)]}, in table order, DROP applied.

    Also the completeness gate: with DROP applied every bar must be an 11/11-cell mean, and a
    partial one raises rather than being drawn. That is stricter than the table, which prints
    partial baseline rows with their dashes visible -- a bar has nowhere to put the dashes.
    """
    loaded, out = T.collect(), {}
    for level, level_lab, dagger in LEVELS:
        recs = []
        for fam, name, data in panel_rows(level, loaded):
            n = sum(1 for t, m, _ in T.COLUMNS if (t, m) in data)
            if n < len(T.COLUMNS):
                raise SystemExit(
                    f"{name} ({level}) averages {n}/{len(T.COLUMNS)} cells -- a bar cannot show "
                    "that. Add it to DROP, or wait for the missing cells.")
            lab = tex_to_mpl(name) + ("$^{\\dagger}$" if fam in dagger else "")
            recs.append((fam, lab, avg(data)))
        out[level_lab] = recs
    missing = [n for n in DROP
               if n not in T.NODE_BASELINES and n not in T.EDGE_BASELINES]
    if missing:
        raise SystemExit(f"DROP names not present in the baseline dicts: {missing} -- "
                         "renamed in make_mib_test_table.py? update DROP or drop the name")
    return out


def draw(ax, recs, ylabel):
    """One level's panel: bars left to right in table order, value printed over each."""
    xs = range(len(recs))
    ax.bar(xs, [v for _, _, v in recs], width=BAR_W,
           color=[FAMILY[f][1] for f, _, _ in recs], lw=0, zorder=2)
    for x, (_, _, v) in zip(xs, recs):
        # offset points, not data units: the two panels have different y ranges, so a constant
        # in data units sits flush on one panel's bars and floats above the other's.
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 1.5),
                    ha="center", va="bottom", fontsize=FS_ANNOT, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab, _ in recs], fontsize=FS_TICK, rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5 - BAR_W / 4, len(recs) - 0.5 + BAR_W / 4)
    ax.set_ylim(0, max(v for _, _, v in recs) * 1.12)   # headroom for the value labels
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    P.furnish(ax)
    ax.grid(False, axis="x")            # vertical rules behind bars are pure noise
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/mib_test_avg.pdf")
    a = ap.parse_args()

    data = bars()
    plt.rcParams.update(P.RC)
    counts = [len(data[lab]) for _, lab, _ in LEVELS]
    fh = PANEL_H + HEAD + FOOT
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(FIG_W, fh),
                             gridspec_kw=dict(width_ratios=counts))
    for ax, (_, level_lab, _) in zip(axes, LEVELS):
        # The level goes in the AXIS TITLE, not a panel title: the two panels carry different y
        # scales, so the level and the units belong to the same piece of text, and a reader
        # cannot pick up a number here without also picking up which axis it is on.
        draw(ax, data[level_lab], f"{level_lab} CPR AUC (↑)")

    fig.tight_layout(pad=0.3, w_pad=1.0)
    top = 1.0 - HEAD / fh
    fig.subplots_adjust(top=top)
    fig.legend(handles=[Patch(facecolor=c, label=tex_to_mpl(lab)) for lab, c in FAMILY.values()],
               fontsize=FS_TICK, ncol=len(FAMILY), loc="lower center", bbox_to_anchor=(0.5, top),
               frameon=False, handlelength=1.2, handleheight=1.0, handletextpad=0.4,
               columnspacing=1.4)
    fig.savefig(a.out)
    print("wrote", a.out)

    for _, level_lab, _ in LEVELS:
        print(f"\n{level_lab}:")
        for fam, lab, v in data[level_lab]:
            print(f"  {FAMILY[fam][0]:<20} {lab:<26} {v:.2f}")


if __name__ == "__main__":
    sys.exit(main())
