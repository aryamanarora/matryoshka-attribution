"""Every \\ourmethod{} ablation's average MIB validation score, as bars: CPR and IIA log-AUC.

THE CUT. All node-level rows of make_mib_table.OUR_METHODS whose dir is complete on BOTH metrics
over the 11 validation cells -- 15 of 15 as of 2026-09-05. Most were only ever run at ONE
learning rate, which is what makes bars the right form: figs/optimizer_lr.pdf spends its whole x
axis on a coordinate where 11 of these 15 arms have a single value, so they read as scattered
dots against four curves. Here the LR is a label, not a geometry.

*** THE LR IS IN THE TICK LABEL AND IT IS LOAD-BEARING. *** These arms sit at 0.01, 0.05, 0.1, 1
and 3, because make_mib_table pins each row to its OWN swept optimum rather than to a shared
value -- the whole point of that decision being that the optimizer does not matter once tuned
but the LR it is tuned at does. Two bars at different LRs are therefore not a controlled
comparison of the ablation alone, and a reader who cannot see the LR would assume they are.
See make_mib_table.OUR_METHODS for the per-row reasoning.

TWO PANELS, SEPARATE Y AXES, AND THEY MUST NOT BE COMPARED BY BAR HEIGHT. CPR AUC runs to ~2.1
and IIA log-AUC to ~0.5; they are different metrics on different scales, not two views of one
number. Each panel carries its own ticks. Bars are SORTED BY CPR, in both panels, so a row can be
tracked across -- and the places where the two disagree are the figure's content: the IIA panel
is visibly non-monotone, with `+ id-STE, Gum.` and its uniform-k twin far higher there than
their CPR rank, and `+ unif k, + id-STE` far lower.

COLOUR IS THE OPTIMIZER, per palette.py's paper-wide rule (SGD black, Adam blue), because it is
the one property of each row that its label does not already carry: the k-schedule is spelled by
the "$+$ unif $k$" prefix and the LR by the parenthetical.

NUMBERS COME FROM plot_mib_accauc_cpr_scatter.build_lr_rows, the same loader the LR figures use,
so these bars cannot disagree with them -- and it applies the same "11/11 cells on BOTH metrics
or it is not plotted" bar, printing every exclusion.

*** THE TWO "hard bwd" ROWS HAVE NO SINGLE LR *** -- results/mib_node_bernoulli_reinforce{,_log}
ran at 0.1 on 10 cells and 0.01 on ioi/llama3, which is why mib_results.tex prints "0.01/0.1".
Their label says "0.01/0.1" rather than picking one.

Run:  uv run python plots/plot_ablation_bars.py
Out:  plots/ablation_bars.pdf  (+ .png)
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mib"))
import palette as P                                    # noqa: E402
import plot_mib_accauc_cpr_scatter as S                # build_lr_rows, RC, delatex  # noqa: E402
import make_mib_table as M                             # OUR_METHODS, ours_lr, opt_of  # noqa: E402

# The paper's headline \ourmethod{} row at node level, pinned by DIR rather than by display name
# -- "\ourmethod{}" legitimately labels one row per optimizer, so a name match would take
# whichever came last. See make_mib_table.OUR_METHODS.
HEADLINE = "topklog_lr_0.05"

# The single MIB cell drawn as a rule across each bar, against the 11-cell mean the bar height
# is. (task, model) -- COLUMNS' own spelling.
#
# *** THE RULE CAN SIT ABOVE OR BELOW ITS BAR AND BOTH HAPPEN. *** It is one cell of the eleven
# being averaged, not an error bar or a bound, so a bar whose rule is far from its top is telling
# you this ablation's score is carried unevenly across cells -- which for ioi/qwen2.5 is the
# common case, since it is the cell where several ablations collapse to the 0.25 CPR floor while
# the other ten hold up.
REF_CELL = ("ioi", "qwen2.5")
REF_LW, REF_HALO = 0.9, 1.9

# 5.5in is iclr2026_conference.sty's \textwidth, so at width=\linewidth this is placed 1:1.
FIG_W = 5.5
# The bars are the data, so PANEL_H is the last thing to cut -- FOOT is where the height
# actually goes. A 45-degree label's height is length * sin(45), so the tick font is the lever:
# 5.0 -> 4.5pt takes ~10% off a 24-character worst case. Steeper rotation is not an option (at
# 30 degrees the height would drop but each label would be 0.87 * its length WIDE, against a
# ~0.17in bar pitch).
PANEL_H = 1.02
HEAD, FOOT = 0.42, 1.00     # legend band + the LR header row above the panels; rotated tick
#                             labels below. Longest label is "+ unif k, + id-STE, Gum." at 24
#                             chars, down from 31 once the LR moved out of it.
FS_AXIS, FS_TICK, FS_ANNOT, FS_LEG, FS_LR = 6.5, 4.5, 4.2, 5.8, 4.0
BAR_W = 0.74
METRICS = [("cpr", "CPR AUC (↑)"), ("acc", "Compactness (↑)")]


def rows():
    """[(label, optimizer, cpr, acc)] in the figure's bar order.

    ORDER: the headline \\ourmethod{}+Adam row first, then the remaining Adam ablations, then the
    SGD ones, each block sorted by CPR descending. The two dashed rules main() draws sit at those
    two boundaries, so the reference is separated from the ablations and the optimizers do not
    interleave.

    SORTED BY CPR IN BOTH PANELS, not by each panel's own metric. Sorting each panel
    independently would put a different row at position 1 in each and destroy the one comparison
    this figure is for -- the places where the two metrics DISAGREE. With a shared order those
    disagreements are visible as a non-monotone IIA panel, which is exactly what the reader
    should see. It also means the IIA panel is deliberately NOT in descending order.
    """
    spec = []
    for name, dirn, level, grp in M.OUR_METHODS:
        if level != "node":
            continue
        lr = M.ours_lr(dirn) or ""
        if not lr:
            continue
        # unifk() is the table's own rule: log k is the unmarked default, uniform k the marked
        # ablation. Reused rather than restated so the bars carry the table's labels exactly.
        disp = "$+$ unif $k$" if grp == "uniform" and name.startswith("\\ourmethod") else (
            f"$+$ unif $k$, {name}" if grp == "uniform" else name)
        spec.append((disp, dirn, lr, M.opt_of(dirn)))
    # build_lr_rows keys on the path label; the DIR is the only unique key here, because the same
    # display name legitimately appears under both optimizers.
    got = {r["grp"]: r for r in
           S.build_lr_rows([(d, d, [(lr.split("/")[-1], d)]) for _, d, lr, _ in spec])}
    out = []
    for disp, dirn, lr, opt in spec:
        if dirn not in got:
            print(f"  dropped {disp} ({dirn}): incomplete on one of the two metrics")
            continue
        # LR is kept OUT of the tick label and drawn as its own row under the axis (see main).
        # Appending " (0.05)" put 7 characters onto a 22-character worst case -- ~24% of the
        # figure's bottom margin spent on four digits, since a 45-degree label's height scales
        # with its length. A two-line tick label was tried first and is worse: at 45 degrees the
        # lines are offset ALONG the rotation, so line 2 of one label lands on line 1 of the next.
        # Single-cell reference: ioi / qwen2.5. Read from the same loader the mean comes from,
        # so the tick and the bar it sits on cannot come from different pkls.
        cell = S._pair(dirn, *REF_CELL)
        out.append((S.delatex(disp), opt, got[dirn]["cpr"], got[dirn]["acc"], dirn, lr,
                    cell[1], cell[0]))
    head = [r for r in out if r[4] == HEADLINE]
    rest = [r for r in out if r[4] != HEADLINE]
    adam = sorted((r for r in rest if r[1] == "adam"), key=lambda r: -r[2])
    sgd = sorted((r for r in rest if r[1] == "sgd"), key=lambda r: -r[2])
    # Boundaries are returned with the rows so main() cannot draw a rule in the wrong place if
    # the headline dir is ever repointed or an arm changes optimizer.
    return head + adam + sgd, [len(head) - 0.5, len(head) + len(adam) - 0.5]


def main():
    data, cuts = rows()
    if not data:
        raise SystemExit("no complete node-level ablations")
    plt.rcParams.update(S.RC)
    fh = PANEL_H + HEAD + FOOT
    fig, axes = plt.subplots(1, len(METRICS), figsize=(FIG_W, fh))
    colour = {"adam": P.METHOD["MAttr"], "sgd": P.METHOD["MAttr (SGD)"]}

    for ax, (key, ylab) in zip(axes, METRICS):
        vals = [r[2] if key == "cpr" else r[3] for r in data]
        refs = [r[6] if key == "cpr" else r[7] for r in data]
        ax.bar(range(len(data)), vals, width=BAR_W,
               color=[colour[r[1]] for r in data], lw=0, zorder=2)
        for x, v in enumerate(vals):
            # ANCHORED ABOVE WHICHEVER IS HIGHER, the bar or its reference rule. The value label
            # sits just over the bar top, and where the ioi/qwen rule exceeds the mean it lands
            # in exactly that space -- `+unif k, -c_k` on IIA rendered as "0|30" with the rule
            # struck through it. The printed number is still the BAR's value; only its position
            # moves.
            rv = refs[x]
            top = v if rv is None else max(v, rv)
            ax.annotate(f"{v:.2f}", (x, top), textcoords="offset points", xytext=(0, 1.8),
                        ha="center", va="bottom", fontsize=FS_ANNOT, rotation=90, zorder=6)
        for x, rv in enumerate(refs):
            if rv is None:
                continue
            # White with a dark halo, because the rule has to read on a blue bar, on a black bar
            # AND on the white background above a bar it exceeds. No flat colour does all three.
            ax.plot([x - BAR_W / 2, x + BAR_W / 2], [rv, rv], color="#ffffff", lw=REF_LW,
                    solid_capstyle="butt", zorder=5,
                    path_effects=[pe.withStroke(linewidth=REF_HALO, foreground="#000000")])
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([r[0] for r in data], fontsize=FS_TICK, rotation=45,
                           ha="right", rotation_mode="anchor")
        # LR row ABOVE the panel, reading as a header over the bars rather than as a second
        # rank of tick labels under them. Rotated 90 so "0.01/0.1" fits inside the ~0.17in bar
        # pitch, which it cannot do horizontally. get_xaxis_transform() is (data x, axes y), so
        # the row sits at the frame whatever the y limits do -- and because it is OUTSIDE the
        # axes it cannot collide with the tallest bar's value label, which is what putting it
        # inside at y~0.98 would have risked.
        for x, r in enumerate(data):
            ax.text(x, 1.012, r[5], transform=ax.get_xaxis_transform(), rotation=90,
                    ha="center", va="bottom", fontsize=FS_LR, color="#444444")
        ax.set_xlim(-0.5 - BAR_W / 4, len(data) - 0.5 + BAR_W / 4)
        # Dashed rules: after the headline row, and between the Adam and SGD blocks. Drawn
        # BEHIND the bars (zorder 1) so a rule never cuts across a bar it does not separate.
        for c in cuts:
            ax.axvline(c, color="#999999", lw=0.5, ls=(0, (2, 2)), zorder=1)
        # ylim spans the reference rules too: on several ablations ioi/qwen2.5 sits ABOVE the
        # 11-cell mean, and a rule clipped at the frame would read as a bar that has none.
        ax.set_ylim(0, max(vals + [r for r in refs if r is not None]) * 1.22)
        ax.set_ylabel(ylab, fontsize=FS_AXIS)
        ax.grid(True, lw=0.25, color="#dddddd")
        ax.set_axisbelow(True)
        ax.grid(False, axis="x")                # vertical rules behind bars are pure noise
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.tick_params(axis="x", length=0)

    fig.tight_layout(pad=0.3, w_pad=1.0)
    # HEAD is shared by two things stacked above the panels: the LR header row sits just over
    # the axes frame, and the legend goes above THAT, anchored to the FIGURE top. Anchoring the
    # legend to the axes top (which is what `bbox_to_anchor=(0.5, top)` did) put it straight
    # through the LR row -- the row is drawn outside the axes, so "above the axes" is where it
    # already is.
    fig.subplots_adjust(top=1.0 - HEAD / fh)
    fig.legend(handles=[Patch(facecolor=colour["adam"], label="Adam"),
                        Patch(facecolor=colour["sgd"], label="SGD"),
                        Line2D([0], [0], color="#ffffff", lw=REF_LW, label="IOI / Qwen cell",
                               path_effects=[pe.withStroke(linewidth=REF_HALO,
                                                           foreground="#000000")])],
               fontsize=FS_LEG, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.998),
               frameon=False, handlelength=1.2, handleheight=1.0, handletextpad=0.4,
               columnspacing=1.4)
    out = "plots/ablation_bars.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {out} ({len(data)} ablations)")

    print(f"\n{'ablation':<34}{'opt':<6}{'CPR':>7}{'IIA':>8}")
    print(f"  {'':<32}{'':<6}{'mean':>7}{'mean':>8}{'ioi/qwen':>10}{'ioi/qwen':>10}")
    for lab, opt, c, a, _, lr, rc, ra in data:
        f = lambda v: f"{v:.3f}" if v is not None else "--"
        print(f"  {lab + ' (' + lr + ')':<32}{opt:<6}{c:>7.3f}{a:>8.3f}{f(rc):>10}{f(ra):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
