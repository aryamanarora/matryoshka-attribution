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
gradient attribution, indigo = mask-learning baselines, grey = the random control. Both
k-schedules of ours share the blue -- see FAMILY.

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

COST IS A SECOND ROW OF BARS UNDER THE CPR PANELS (2026-09-11, requested), on a LINEAR axis in
backward passes, aligned bar-for-bar with the CPR row so "how well" and "what it cost" are read
straight down. It replaced grey cost strings printed over each bar: fifteen strings like
"0.1--1k" had to be parsed to discover that the best trained method is also the cheapest, and
on the tallest bar the string collided with the panel title. Linear rather than log (tried,
rejected): log made the 6x gap between 0.5k and 3k legible but flattened the 24k and 30k rows
that are the point; linear makes those tower and prints the number over any bar too short to
read (see TINY_FRAC), which is where the fine distinctions now live. A RANGE is drawn as a solid
bar to its minimum and a lighter bar of the same hue on to its maximum, so "0.1--1k" is a sliver
plus a pale sliver rather than a string; the caption has to say so, since an unexplained pale
bar reads as a second series.

ONE CAVEAT THE Avg CANNOT SHOW, and this figure no longer marks: at edge level our five llama3
cells are scored on a 200-example subset (make_mib_test_table's EDGE_DAGGER), so those Avgs mix
capped and full-split cells. The table carries the dagger; the figure dropped it (2026-09-11,
requested; see LEVELS). It does not distort the comparison WITHIN that family -- all our edge
rows carry it equally -- but EAP-IG-inp beside them is full-split, so the cross-family edge gap
is the one number this affects.

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
Also dropped, from OUR rows rather than the baselines: both SGD arms at both levels, by
optimiser rather than by name -- see DROP_OURS_OPT for why the survivors keep the table's
"$+$ Adam" labels.

With those three gone every remaining bar is an 11/11-cell mean, which main() asserts.

Run:  uv run python plots/plot_mib_test_avg.py
Out:  plots/mib_test_avg.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import argparse
import os
import re
import sys

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mib"))
import palette as P                                     # noqa: E402
import make_mib_test_table as T                         # collect() + the literal baselines  # noqa: E402
import make_mib_table as _M   # cost constants only -- the table's own, so a re-costing reaches this figure  # noqa: E402

# family key -> (legend label, colour). Keys are internal; the labels are what the reader sees.
#
# "ours_uni" IS THE SAME COLOUR AS "ours" (2026-09-08, requested): both k-schedules draw in
# MAttr blue and the legend carries ONE "ours" key -- see LEGEND. The uniform-k rows kept a
# separate FAMILY key only so panel_rows() can still count them (UNI_PER_LEVEL) and so the
# split can be restored by changing one hue here; nothing downstream may assume the two
# differ. THIS FIGURE NOW DISAGREES WITH plots/plot_mib_accauc_cpr_scatter.py, which still
# gives uniform-k its own hue (G_MUNI = P.METHOD["+hard"], Wong bluish green) because there
# it is a distinct point on the frontier and the two schedules are the axis being compared.
# The old comment here argued a method cannot be blue in one figure and green in its
# neighbour; that is still the risk, so if the scatter stays on the same page as this figure,
# recolour it too rather than reverting this.
FAMILY = {
    "control":  ("Random control", P.METHOD["Random"]),
    "gradient": ("Gradient-based", P.METHOD["IG"]),
    "mask":     ("Mask-based", P.METHOD["Node Pruning"]),
    "ours":     ("\\ourmethod{} (ours)", P.METHOD["MAttr"]),
    "ours_uni": ("\\ourmethod{}, unif. $k$ (ours)", P.METHOD["MAttr"]),
}
# Legend keys, in order. NOT FAMILY.values(): "ours_uni" now shares "ours"'s hue, so drawing
# both would print two identical swatches under two different names -- which reads as a colour
# distinction the figure does not make. Derived by de-duplicating on colour rather than by
# hardcoding the survivors, so restoring a separate uniform-k hue brings its key back on its
# own. Read by plot_mib_test_avg_accauc.py too, which draws the same legend.
LEGEND = list({c: lab for lab, c in reversed(list(FAMILY.values()))}.items())
LEGEND = [(lab, c) for c, lab in reversed(LEGEND)]
# Substring that marks a uniform-k row in make_mib_test_table's OUR_* name strings ("$+$ unif
# $k$", "$+$ Adam, unif $k$"). Matched rather than hardcoded per name so an Adam/SGD relabel
# does not silently drop the recolour; UNI_PER_LEVEL then asserts the match still finds them,
# so a rename to something without "unif" raises instead of quietly turning the bars blue.
# 0 since 2026-09-15: the one surviving MAttr row per level IS the uniform-k run and is drawn
# as the plain method, so no drawn row carries the "unif" mark any more (the log-k arms are the
# marked ones now, and both are dropped). The hook stays so a split can be restored.
UNI_MARK, UNI_PER_LEVEL = "unif", 0
# Baseline rows the table keeps and this figure does not -- see the docstring for each. Matched
# against NODE_BASELINES / EDGE_BASELINES keys, and a name here that matches nothing raises
# rather than silently doing nothing, so a rename in the table cannot quietly un-drop a row.
# "EAP-IG-inp (CF)" dropped 2026-09-17: it is MIB's published number for the 5-step grid, and the
# edge panel now draws our own re-run of that setting ("IG ($m{=}5$)", GRAD_EDGE_BASELINES),
# named as the node panel names it. The table keeps both; the figure has one bar per method.
DROP = ("NAP (CF)", "NAP-IG (CF)", "UGS", "EAP-IG-inp (CF)")
# OUR rows to drop, by OPTIMISER (2026-09-08, requested): the figure shows Adam only, so the
# SGD log-k and SGD uniform-k arms are dropped at BOTH levels -- 4 rows, 2 per level.
#
# KEYED ON THE RESULTS DIR, NOT THE ROW NAME, and that is the whole point. Under the
# 2026-08-24 convention the bare "\ourmethod{}" row IS the SGD one and "$+$ Adam" is the
# ablation, so a name-keyed drop would read as dropping the headline method and would invert
# silently if that convention is ever flipped back. make_mib_table.opt_of() is the same
# function the TABLE files these rows by, so this figure drops exactly the rows the table
# prints under its SGD block.
#
# Rows whose eval wave is STILL LANDING. They are skipped with a printed line instead of
# aborting the figure. Everything else that comes up short of 11/11 still raises -- that guard
# catches a dir silently losing cells, and a bare "skip anything incomplete" rule would let
# that failure vanish into stdout. A name here is a promise to remove it once the cells land;
# an entry that is already complete raises too, so the list cannot rot unnoticed.
# 2026-09-17: the four dir-backed edge baselines (make_mib_test_table.GRAD_EDGE_BASELINES /
# MASK_EDGE_BASELINES) while their eval waves land. "IG ($m{=}5$)", "IG ($m{=}10$)" and
# "Expected Gradients" also name COMPLETE node rows; the guard is per row, so those still draw.
# Keyed by (level, name): the first three names also label COMPLETE node rows, and a name-only
# key read those as "the pending row has landed" the moment the edge rows were skipped entirely.
# 2026-09-17 (later): the three EAP-IG-family edge test evals landed 12/12; only Edge Pruning's
# validation-then-test wave is still mid-flight.
PENDING = (("edge", T._M.EPRUN_NAME["edge"]),)
DROP_OURS_OPT = "sgd"
# Also dropped BY NAME (2026-09-11, requested): the log-k Adam arm, so the only MAttr bar left
# is the uniform-k Adam one, drawn as the plain method (see RENAME_OURS). Keyed on the table's
# row string like RENAME_OURS, and guarded the same way: a name here that matches no row raises.
# 2026-09-15: the tables now follow scripts/mib/mattr_variants.py (headline = uniform k, Adam),
# so the log-k Adam arm is labelled "$+$ log $k$" and the surviving uniform-k Adam row is
# ALREADY the bare \ourmethod{} -- the RENAME_OURS patch below is therefore empty, and the
# "MAttr means a different run here than in the table" defect described above is closed.
# "$+$ $10\\times$ steps" (node only, 2026-09-17): the figure keeps one MAttr bar per level, the
# 500-step headline; the 10x row is the table's. Node therefore drops 4 of its 5 rows, edge 3 of 4.
DROP_OURS_NAMES = (T.MV.label(k="log"), "$+$ $10\\times$ steps")
OURS_DROPPED_PER_LEVEL = {"node": 4, "edge": 3}
# Display names for the two survivors (2026-09-08, requested). With no non-Adam MAttr left in
# the figure, "$+$ Adam" is an ablation marker pointing at nothing, so the rows are drawn as
# the plain method and its one k-schedule ablation.
#
# *** THIS MAKES "MAttr" MEAN A DIFFERENT RUN HERE THAN IN tabs/mib_test_results.tex. *** The
# table still follows the 2026-08-24 convention where the unmarked \ourmethod{} row is SGD
# (test_node_softlog_sgd_lr_1.0) and Adam is the "$+$ Adam" ablation; this figure's unmarked
# "MAttr" bar is the ADAM dir. A reader holding both will find the same name on two different
# runs -- worth 0.00 CPR at node level (1.89 vs 1.89) but 0.31 at edge (6.23 vs 5.92). Fix it
# properly by flipping the table to Adam-default, not by patching more names in here; this map
# is keyed on the table's current strings and will raise the moment they change.
# 2026-09-11 (requested): uniform-k IS the figure's MAttr. The log-k arm is gone (see
# DROP_OURS_NAMES), so the surviving uniform-k Adam row is drawn as the plain method name.
RENAME_OURS = {}   # the headline row is bare \ourmethod{} in the table itself since 2026-09-15
# Baseline rows relabelled for the figure only (2026-09-11, requested). The table keeps MIB's
# own name for its published edge row; here it is drawn under the node panel's naming, since
# it IS the 5-step IG grid (see its COST entry) and "EAP-IG-inp (CF)" beside "IG ($m{=}5$)"
# reads as two methods. Same guard as RENAME_OURS: a key that matches no drawn row raises.
RENAME_BASELINES = {
    # The L1 ladder of 8 DBM runs (make_mib_table.DBM_MULTI_COST); "sweep" says what it is
    # in one word, and the cost row beneath shows what the sweep costs.
    "DBM (multi-sparsity)": "DBM (sweep)",
    # "NP" (2026-09-16, requested): the full name is the widest tick of the node panel.
    "Node Pruning": "NP",
    "Node Pruning (multi-sparsity)": "NP (sweep)",
    # Rotated tick; the full name is twice the width of any other and pushes the axis down.
    # "EG" is the abbreviation plot_mib_accauc_cpr_scatter.py already uses for the same row.
    "Expected Gradients": "EG",
}
RENAME = {**RENAME_OURS, **RENAME_BASELINES}
# Third element = families whose rows carry the llama3 dagger at that level. EMPTY at both levels
# since 2026-09-11 (requested): the edge rows used to carry it ("ours", "ours_uni") for the
# 200-example llama3 cells, but the mark was dropped from the figure -- the caveat is the
# table's (make_mib_test_table.EDGE_DAGGER) and stays there. The hook is kept so it can be
# restored per level without re-deriving which families it applies to; if it is, list
# "ours_uni" alongside "ours" -- it is a colour split, not a scoring one.
LEVELS = [("node", "Node-level", ()), ("edge", "Edge-level", ())]

# === Training cost, drawn as the lower row of bars (see draw_cost) ===
#
# WHY THE FIGURE CARRIES IT AT ALL: this table has no cost column (method + 11 cells + Avg), so
# without this the bar chart says "DBM (multi-sparsity) is second-best at node level" and gives
# the reader no way to learn that it spends EIGHT training runs to every other mask row's one.
# It also makes the cheap-and-best point visible in one glance: \ourmethod{} is 0.5k, the least
# of any TRAINED method here and 6x under the mask baselines it beats.
#
# UNIT is make_mib_table's: backward passes through the model, in sequences, to fit ONE cell.
# The numbers are that module's constants rather than literals, so a re-costing reaches this
# figure too -- only the NAME -> constant mapping lives here, because the test table's row names
# are not the validation table's (whose nested grad_cost/mask_cost do this same job).
#
# RANGES ARE KEPT AS RANGES. The gradient methods cost (steps x examples) and the example count
# varies by cell, so "0.1--1k" is the honest statement and a single number would not be. It is
# tempting to draw the upper end only to simplify the bars; that would say \ourmethod{} (0.5k)
# is cheaper than I x G, when on the smallest cells I x G is 0.1k -- five times cheaper. The
# strings stay the display form (they are also the tiny-bar labels, see draw_cost); parse_cost
# turns them into the (min, max) the bars are drawn from. The en-dash is spelled out here
# rather than reusing the LaTeX "--" because these strings go to matplotlib.
_R = lambda s: s.replace("--", "\u2013")
COST = {
    "node": {
        # Not a training cost and not a gradient cost: a random ranking is drawn, not fitted.
        "Random": "0",
        "I$\\times$G": _R(_M.COST_GRAD_IG1), "RelP": _R(_M.COST_GRAD_IG1),
        "RelP$+$QK": _R(_M.COST_GRAD_IG1), "GIM": _R(_M.COST_GRAD_IG1),
        "AttnLRP": _R(_M.COST_GRAD_IG1),
        # Expected Gradients is m=1 -- alpha ~ U(0,1) drawn per example instead of a fixed grid -- so
        # it costs exactly what I x G costs. That it lands with IG-10 at a 10th of IG-10's cost
        # is the whole reason the row is interesting, and it is invisible without this column.
        "Expected Gradients": _R(_M.COST_GRAD_IG1),
        "IG ($m{=}5$)": _R(_M.COST_GRAD_IG5),
        # The 10- and 30-step rows declare their own cost with the rows themselves.
        "IG ($m{=}10$)": _R(dict((d, c) for d, _, c in _M.NAPIG_STEP_ROWS)["$+$ 10 IG steps"]),
        "IG ($m{=}30$)": _R(dict((d, c) for d, _, c in _M.NAPIG_STEP_ROWS)["$+$ 30 IG steps"]),
        "DBM": _M.COST_EPRUN, "Node Pruning": _M.COST_EPRUN,
        "DBM (multi-sparsity)": _M.DBM_MULTI_COST,
        "Node Pruning (multi-sparsity)": _M.NP_MULTI_COST,
        # Every MAttr row costs the same 500 steps; keyed on the table's own row list so a relabel
        # (2026-09-15: headline is now the bare \ourmethod{}) cannot strand a row without a cost.
        **{name: _M.COST_OURS["node"] for name, _ in T.OUR_NODE_METHODS},
        "$+$ $10\\times$ steps": "5k",   # never drawn (DROP_OURS_NAMES); kept correct anyway
    },
    "edge": {
        # MIB's published EAP-IG-inputs is the 5-step grid, the same setting our repro row
        # carries in the validation table.
        "EAP-IG-inp (CF)": _R(_M.COST_GRAD_IG5),
        # Our own edge runs (make_mib_test_table.GRAD_EDGE_BASELINES / MASK_EDGE_BASELINES).
        "IG ($m{=}5$)": _R(_M.COST_GRAD_IG5),
        "IG ($m{=}10$)": _R(dict((d, c) for d, _, c in _M.EAPIG_EDGE_STEP_ROWS)["$+$ 10 IG steps"]),
        "Expected Gradients": _R(_M.COST_GRAD_IG1),
        _M.EPRUN_NAME["edge"]: _M.COST_EPRUN_EDGE,
        **{name: _M.COST_OURS["edge"] for name, _ in T.OUR_EDGE_METHODS},
    },
}


# 5.5in is iclr2027_conference.sty's \textwidth verbatim (line 49), so at width=\linewidth the
# figure is placed 1:1 and the sizes below are the sizes that reach the compiled PDF. Most other
# figures in paper/figs are drawn at 5.5 or 5.4 for the same reason; do not draw this one smaller
# and let LaTeX upscale it, which would push the tick labels past 7pt.
FIG_W = 5.5
# The plotting rect itself. Kept SHORT on purpose: this figure carries one number per method, so
# the y extent is doing no work beyond ordering the bars -- height here is spent on nothing and
# costs a column inch on a 9-page limit. FOOT is what the 45-degree tick labels need below the
# axis ("EAP-IG-inp (CF)" is the longest); tight_layout will steal it back from PANEL_H if it is
# short, shrinking the bars rather than clipping, so check the ink bbox after changing either.
# PANEL_H gained 0.12 when the level moved from the y-axis label into a panel TITLE: a title
# lives inside the axes bbox, so without this the bars would have paid for it and the two
# figures' bar heights would no longer match across revisions.
PANEL_H = 0.85   # 1.17 -> 0.85 on 2026-09-16 (requested): the bar row was taller than its content needs
# HEAD grew 0.22 -> 0.30 with the panel titles: the legend is anchored AT the axes top, so a
# title (which lives inside the axes bbox) butts straight up against it. The extra head plus
# the +0.03 offset on bbox_to_anchor below is what puts ~4pt of air between "Node-level" and
# the legend row; without it the two read as one line.
HEAD, FOOT = 0.30, 0.50   # FOOT 0.70 -> 0.50 (2026-09-16, requested): the 45-degree labels left ~0.2in of air under them
# The cost row. Short: it carries one bar per method and its only job is the ratio between
# them, which survives at this height; the 45-degree method names hang off ITS x axis, so FOOT
# is measured from here, not from the CPR panel. Set beside PANEL_H so the two rows' shares
# are visible in one place.
STRIP_H = 0.40
# Headroom above the tallest cost bar. 1.30 rather than the CPR panels' 1.14 because the
# tallest bar here has no label to clear -- only the TINY ones are labelled -- but the edge
# panel's 5k bars otherwise ran into the top spine.
STRIP_HEADROOM = 1.30
# A cost bar shorter than this fraction of the strip's tallest bar gets its string printed
# over it. On a linear axis with a 30k ceiling, 0.5k is one pixel; the number is what makes
# "MAttr costs 0.5k, Node Pruning 3k" readable at all. Bars above the threshold are legible
# as bars and stay unlabelled so the row is not fifteen strings again.
TINY_FRAC = 0.12
# How baseline bars are drawn relative to OURS, in both rows and in the legend. Family is
# always the hue. Three options were tried on 2026-09-11 and "solid" was kept (requested):
#   "solid"   -- every bar at full strength; nothing singles ours out but its colour
#   "muted"   -- baselines filled with their family colour blended MUTE of the way to white
#   "outline" -- baselines hollow, outlined in their family colour
FILLED = ("ours", "ours_uni")
BASELINE_STYLE = "solid"
MUTE = 0.55           # fraction of the way from the family colour to white
BAR_LW = 0.8
# A ranged cost on a filled bar: solid to the minimum, RANGE_ALPHA of the same hue on to the
# maximum. On an outlined bar: solid outline to the minimum, DASHED outline on to the maximum.
RANGE_ALPHA = 0.35
RANGE_DASH = (0, (2.0, 1.2))
# Left margin, figure fraction: clears "Backward passes" plus the widest tick label of either
# row ("30k"). Measured against the render, like the old supylabel x was.
LEFT = 0.095
FS_AXIS, FS_TICK, FS_ANNOT = 7, 6.5, 6
BAR_W = 0.72           # in category units, so bars are the same thickness in both panels
COST_COLOR = "#8a8a8a"   # the tiny-bar cost label: secondary to the black CPR values above


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


def parse_cost(s):
    """A COST display string -> (min, max) backward passes. "0.1\u20131k" -> (100, 1000); "3k" -> (3000, 3000).

    Only the forms COST actually holds are handled; anything else raises rather than drawing a
    bar of the wrong height. "0" (the random control) is (0, 0): a bar of no height, which on a
    linear axis is the honest picture of a ranking that is drawn, not fitted.
    """
    body = s.replace("\u2013", "-")
    if not re.fullmatch(r"\d+(\.\d+)?(-\d+(\.\d+)?)?k?", body):
        raise SystemExit(f"cannot parse cost string {s!r} -- extend parse_cost()")
    unit = 1000 if body.endswith("k") else 1
    parts = [float(x) * unit for x in body.rstrip("k").split("-")]
    return parts[0], parts[-1]


def tint(colour, frac):
    """`colour` blended `frac` of the way to white."""
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def bar_style(fam):
    """Fill/edge kwargs for a bar of family `fam`: full strength for FILLED, demoted otherwise."""
    colour = FAMILY[fam][1]
    if fam in FILLED:
        # Our bars carry a white diagonal hatch (2026-09-16, requested) so they stand out from
        # the solid baselines; the legend swatch goes through this same function.
        return dict(facecolor=colour, edgecolor="white", hatch="///", lw=0)
    if BASELINE_STYLE == "solid":
        return dict(color=colour, lw=0)
    if BASELINE_STYLE == "muted":
        return dict(color=tint(colour, MUTE), lw=0)
    if BASELINE_STYLE == "outline":
        return dict(facecolor="none", edgecolor=colour, lw=BAR_LW)
    raise SystemExit(f"unknown BASELINE_STYLE {BASELINE_STYLE!r}")


def avg(data):
    """Mean over the cells present, matching make_mib_test_table.main()'s row_avg."""
    vs = [v for v in (data.get((t, m)) for t, m, _ in T.COLUMNS) if v is not None]
    return round(sum(vs) / len(vs), 2) if vs else None


def sem(data):
    """Standard error of the row mean ACROSS THE 11 TEST CELLS.

    *** THIS IS CELL HETEROGENEITY, NOT MEASUREMENT NOISE, AND IT IS NOT A COMPARISON BAR. ***
    Each run is deterministic given its seed; the spread here is that ioi/gpt2 and mcqa/llama3
    are different problems, so the bar says "how much does this method's score vary across the
    benchmark", not "how well is this number determined".
    *** IT ALSO OVERSTATES THE UNCERTAINTY OF ANY COMPARISON BETWEEN TWO BARS. *** Every method
    is evaluated on the SAME eleven cells, so differences are paired and the paired s.e. is much
    smaller: MAttr minus Node Pruning is 0.23 +- 0.11 paired against +- 0.16 read off two
    independent bars, and MAttr minus its Adam twin is -0.00 +- 0.02 paired against +- 0.13.
    Overlapping error bars here therefore do NOT mean two methods are indistinguishable. Say so
    in the caption, or use a paired test.
    """
    vs = [v for v in (data.get((t, m)) for t, m, _ in T.COLUMNS) if v is not None]
    if len(vs) < 2:
        return None
    return float(np.std(vs, ddof=1) / math.sqrt(len(vs)))


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


def ours_kept(level, rows):
    """`rows` minus the arms whose results dir is DROP_OURS_OPT, with a count guard.

    Raises rather than silently keeping an SGD bar if the dir lists are renamed or repointed:
    a figure that quietly regrows a dropped method is the failure this guard exists for.
    """
    dirs = dict(T.OUR_NODE_METHODS if level == "node" else T.OUR_EDGE_METHODS)
    kept, dropped = [], []
    for name, data in rows:
        # A name collect() returned but the dir map does not have cannot be classified, so it
        # is louder to fail than to guess the optimiser and keep it.
        if name not in dirs:
            raise SystemExit(f"{level}: our row {name!r} is not in make_mib_test_table.OUR_"
                             f"{level.upper()}_METHODS -- cannot resolve its optimiser")
        drop = T._M.opt_of(dirs[name]) == DROP_OURS_OPT or name in DROP_OURS_NAMES
        (dropped if drop else kept).append(name)
    if len(dropped) != OURS_DROPPED_PER_LEVEL[level]:
        raise SystemExit(f"{level}: dropped {len(dropped)} rows ({dropped}), expected "
                         f"{OURS_DROPPED_PER_LEVEL[level]} -- repointed or renamed in "
                         "make_mib_test_table.OUR_*_METHODS? update DROP_OURS_OPT / "
                         "DROP_OURS_NAMES")
    unknown = [n for n in DROP_OURS_NAMES if n not in dirs and level != "edge"]   # the 10x row is node-only
    if unknown:
        raise SystemExit(f"{level}: DROP_OURS_NAMES {unknown} match no row in "
                         "make_mib_test_table.OUR_*_METHODS -- renamed? update DROP_OURS_NAMES")
    return [(n, d) for n, d in rows if n in set(kept)]


_EXTRA = None


def extra():
    """make_mib_test_table.collect_extra(), loaded once: (causal_nodes, grad_edges, mask_edges).
    The activation-patching rows are 3-cell by design and are never drawn (a bar cannot show 3/12),
    so only the two edge groups are used here."""
    global _EXTRA
    if _EXTRA is None:
        _EXTRA = T.collect_extra()
    return _EXTRA


def panel_rows(level, loaded):
    """[(family, name, data)] for one level, in table order, with DROP applied."""
    ours_nodes, mask_nodes, grad_nodes, ours_edges = loaded
    _, grad_edges, mask_edges = extra()
    if level == "node":
        groups = [
            ("control", [(n, T.NODE_BASELINES[n])
                         for n in T.CONTROL_NAMES if n in T.NODE_BASELINES], True),
            ("gradient", T.classify(T.NODE_BASELINES, "gradient") + list(grad_nodes.items()), True),
            ("mask", list(mask_nodes.items()), True),
            # complete_or_skip already held ours to 11/11, so suppress_partial has nothing to act
            # on -- passed explicitly here for the same reason main() passes it explicitly.
            ("ours", ours_kept("node", list(ours_nodes.items())), False),
        ]
    else:
        groups = [
            ("gradient", T.classify(T.EDGE_BASELINES, "gradient") + list(grad_edges.items()), False),
            ("mask", T.classify(T.EDGE_BASELINES, "mask") + list(mask_edges.items()), False),
            ("ours", ours_kept("edge", list(ours_edges.items())), False),
        ]
    out = []
    for fam, rows, sup in groups:
        for name, data in ordered([r for r in rows if r[0] not in DROP], sup):
            # Recoloured AFTER ordering, not sorted into their own group: the bars stay in the
            # table's worst-to-best order within "ours", so the green ones interleave with the
            # blue exactly where their Avg puts them. Splitting the group would reorder the
            # panel to serve the colour, which is backwards.
            out.append((f"{fam}_uni" if fam == "ours" and UNI_MARK in name else fam, name, data))
    n_uni = sum(1 for f, _, _ in out if f == "ours_uni")
    if n_uni != UNI_PER_LEVEL:
        raise SystemExit(f"{level}: {n_uni} rows matched {UNI_MARK!r}, expected {UNI_PER_LEVEL} "
                         "-- renamed in make_mib_test_table.OUR_*_METHODS? update UNI_MARK")
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
                if (level, name) in PENDING:
                    print(f"PENDING {name} ({level}): {n}/{len(T.COLUMNS)} cells -- row skipped "
                          "until its eval wave finishes")
                    continue
                raise SystemExit(
                    f"{name} ({level}) averages {n}/{len(T.COLUMNS)} cells -- a bar cannot show "
                    "that. Add it to DROP or PENDING, or wait for the missing cells.")
            # Cost is drawn by draw_cost as the lower row of bars (2026-09-11). Three earlier
            # placements were tried against the rendering and rejected: as a second tick-label
            # LINE it lands down-and-right of the name at 45 degrees, i.e. visually between its
            # own method and the next one; INLINE in the tick label it pushes the longest label
            # to "DBM (multi-sparsity) . 24k" and costs 0.24in of FOOT; as grey text ABOVE the
            # value it is unreadable as a comparison and collides with the panel title on the
            # tallest bar. A row with no COST entry RAISES -- a bar silently drawn without its cost is
            # exactly the omission this annotation exists to prevent, and new rows arrive by
            # editing lists, not this map.
            if name not in COST[level]:
                raise SystemExit(
                    f"{name!r} ({level}) has no COST entry -- add one (the validation table's "
                    "rendered cost column in paper/tabs/mib_results.tex is the ground truth) "
                    "or the bar would claim a free method.")
            lab = tex_to_mpl(RENAME.get(name, name)) \
                + ("$^{\\dagger}$" if fam in dagger else "")
            recs.append((fam, lab, avg(data), sem(data), COST[level][name]))
        out[level_lab] = recs
    seen = {name for _, lab, _ in LEVELS for _, name, _ in ()} | {
        name for level, _, _ in LEVELS for _, name, _ in panel_rows(level, loaded)}
    unused = [k for k in RENAME if k not in seen]
    if unused:
        raise SystemExit(f"RENAME keys match no drawn row: {unused} -- renamed in "
                         "make_mib_test_table.OUR_*_METHODS / EDGE_BASELINES? update RENAME_OURS "
                         "or RENAME_BASELINES")
    # A pending (level, name) is stale once THAT level's row is drawn complete. A row that is not
    # drawn at all (its dir has no cells yet, so the table's guard skipped it) is still pending.
    stale = [(l, n) for l, n in PENDING
             if any(nm == n and sum(1 for t, m, _ in T.COLUMNS if (t, m) in d) == len(T.COLUMNS)
                    for f, nm, d in panel_rows(l, loaded))]
    if stale:
        raise SystemExit(f"PENDING rows are now complete: {stale} -- remove them from PENDING "
                         "so the completeness guard covers them again")
    missing = [n for n in DROP
               if n not in T.NODE_BASELINES and n not in T.EDGE_BASELINES]
    if missing:
        raise SystemExit(f"DROP names not present in the baseline dicts: {missing} -- "
                         "renamed in make_mib_test_table.py? update DROP or drop the name")
    return out


def draw(ax, recs, title):
    """One level's panel: bars left to right in table order, value printed over each.

    `title` names the LEVEL. It used to be the y-axis label ("Node-level CPR AUC"), because the
    two panels carry different y scales and a number here cannot be read without knowing which
    axis it is on. The label is now a single shared "CPR" for the figure (2026-09-08,
    requested), so the level moved to a panel title rather than being dropped -- the reason it
    has to appear somewhere is unchanged.
    """
    xs = range(len(recs))
    for x, r in zip(xs, recs):
        ax.bar(x, r[2], width=BAR_W, zorder=2, **bar_style(r[0]))
    # Caps drawn in black on top of every bar colour, thin enough not to read as part of the bar.
    ax.errorbar(list(xs), [r[2] for r in recs],
                yerr=[r[3] or 0.0 for r in recs], fmt="none", ecolor="#000000",
                elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=4)
    for x, (_, _, v, e, cost) in zip(xs, recs):
        # offset points, not data units: the two panels have different y ranges, so a constant
        # in data units sits flush on one panel's bars and floats above the other's.
        # Anchored above the WHISKER, not the bar: at these spreads the cap sits well inside
        # the old label position and the two collided on every row.
        ax.annotate(f"{v:.2f}", (x, v + (e or 0.0)), textcoords="offset points", xytext=(0, 1.5),
                    ha="center", va="bottom", fontsize=FS_ANNOT, zorder=5)
    # The method names hang off the COST row below (draw_cost), which shares this x axis
    # bar-for-bar; ticks are kept here so the two rows' bars are pinned to the same positions.
    ax.set_xticks(list(xs))
    ax.set_xticklabels([])
    ax.set_xlim(-0.5 - BAR_W / 4, len(recs) - 0.5 + BAR_W / 4)
    # One line of annotation (the value) above each whisker; 1.14 clears it under the title.
    ax.set_ylim(0, max(r[2] + (r[3] or 0.0) for r in recs) * 1.28)   # 1.14 -> 1.28 (2026-09-16): the tallest bar's label was tight against the frame
    ax.set_title(title, fontsize=FS_AXIS, pad=2)
    P.furnish(ax)
    ax.grid(False, axis="x")            # vertical rules behind bars are pure noise
    # All four spines, black -- P.furnish sets them. The top and right used to be hidden; the
    # paper's panels now carry a full frame (figs/baseline_strongreject) and a half-open box
    # here would be the only one of its kind.
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", length=0)


def draw_cost(sx, recs):
    """The cost row for one level: a bar per method, LINEAR axis, aligned under draw()'s bars.

    A range is a solid bar to its minimum plus a RANGE_ALPHA bar of the same hue on to its
    maximum. Bars shorter than TINY_FRAC of the row's tallest get their COST string printed over
    them (grey, a shade smaller than the CPR values), because on this axis they are otherwise a
    line at the floor; the tall ones are legible as bars and stay unlabelled. Ticks are whole
    thousands ("10k"), no minor ticks: this row is read as ratios, not values, except where the
    label supplies the value.
    """
    xs = list(range(len(recs)))
    ranges = [parse_cost(r[4]) for r in recs]
    top = max(hi for _, hi in ranges)
    for x, (fam, _, _, _, cost), (lo, hi) in zip(xs, recs, ranges):
        colour = FAMILY[fam][1]
        sx.bar(x, lo, width=BAR_W, zorder=2, **bar_style(fam))
        if hi > lo:
            if fam in FILLED or BASELINE_STYLE == "solid":
                sx.bar(x, hi - lo, bottom=lo, width=BAR_W, color=colour, alpha=RANGE_ALPHA,
                       lw=0, zorder=2)
            elif BASELINE_STYLE == "muted":
                # the extension is RANGE_ALPHA of the already-muted fill, same as for ours
                sx.bar(x, hi - lo, bottom=lo, width=BAR_W, color=tint(colour, MUTE),
                       alpha=RANGE_ALPHA, lw=0, zorder=2)
            else:
                sx.bar(x, hi - lo, bottom=lo, width=BAR_W, facecolor="none", edgecolor=colour,
                       lw=BAR_LW, ls=RANGE_DASH, zorder=2)
        if hi < TINY_FRAC * top:
            sx.annotate(cost, (x, hi), textcoords="offset points", xytext=(0, 1.2),
                        ha="center", va="bottom", fontsize=FS_ANNOT - 0.7, color=COST_COLOR,
                        zorder=5)
    sx.set_ylim(0, top * STRIP_HEADROOM)
    # ~3 ticks per strip whatever its range: the edge panel tops out at 10k (IG m=10) and the
    # 2.5k step gave it six labels (0 .. 12.5k) on a 0.4in strip.
    step = 10_000 if top >= 20_000 else 5_000 if top >= 10_000 else 2_500
    ticks = np.arange(0, top * STRIP_HEADROOM, step)
    sx.set_yticks(ticks)
    sx.set_yticklabels(["0" if v == 0 else f"{v / 1000:g}k" for v in ticks], fontsize=FS_TICK)
    sx.set_xticks(xs)
    sx.set_xticklabels([r[1] for r in recs], fontsize=FS_TICK, rotation=45,
                       ha="right", rotation_mode="anchor")
    sx.set_xlim(-0.5 - BAR_W / 4, len(recs) - 0.5 + BAR_W / 4)
    P.furnish(sx)
    sx.grid(False, axis="x")
    sx.tick_params(axis="x", length=0)
    sx.tick_params(axis="y", labelsize=FS_TICK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/mib_test_avg.pdf")
    a = ap.parse_args()

    data = bars()
    plt.rcParams.update(P.RC)
    counts = [len(data[lab]) for _, lab, _ in LEVELS]
    fh = PANEL_H + STRIP_H + HEAD + FOOT
    # Two rows: CPR over cost, column widths by bar count as before, row heights by the two
    # constants so neither row's share depends on what tight_layout would have negotiated --
    # it cannot lay out a gridspec with height_ratios reliably, so margins are set explicitly.
    fig, axes = plt.subplots(2, len(LEVELS), figsize=(FIG_W, fh),
                             gridspec_kw=dict(width_ratios=counts,
                                              height_ratios=[PANEL_H, STRIP_H]))
    for col, (_, level_lab, _) in enumerate(LEVELS):
        draw(axes[0, col], data[level_lab], level_lab)
        draw_cost(axes[1, col], data[level_lab])

    # One y label per ROW, on the left column, aligned so the two read as one axis edge.
    axes[0, 0].set_ylabel("CPR (↑)", fontsize=FS_AXIS)
    axes[1, 0].set_ylabel("Cost (↓)", fontsize=FS_AXIS)
    fig.align_ylabels(axes[:, 0])
    top = 1.0 - HEAD / fh
    fig.subplots_adjust(top=top, bottom=FOOT / fh, left=LEFT, right=0.995,
                        wspace=0.22, hspace=0.10)
    # Swatches follow the bars: filled only for the families in FILLED. LEGEND is de-duplicated
    # on colour, so look the family key up by colour to decide.
    fam_of = {c: k for k, (_, c) in FAMILY.items()}
    fig.legend(handles=[Patch(label=tex_to_mpl(lab), **bar_style(fam_of[c])) for lab, c in LEGEND],
               fontsize=FS_TICK, ncol=len(LEGEND), loc="lower center",
               bbox_to_anchor=(0.5, top + 0.03),
               frameon=False, handlelength=1.2, handleheight=1.0, handletextpad=0.4,
               columnspacing=1.4)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)   # eyeballing sibling
    print("wrote", a.out)

    for _, level_lab, _ in LEVELS:
        print(f"\n{level_lab}:")
        for fam, lab, v, e, cost in data[level_lab]:
            print(f"  {FAMILY[fam][0]:<20} {lab:<26} {v:.2f} "
                  f"+- {e if e else float('nan'):.2f}   cost {cost}")


if __name__ == "__main__":
    sys.exit(main())
