"""tabs/mib_results.tex as a bar chart: every method, every cell, both levels.

THE TABLE IS 15 COLUMNS x 41 ROWS AND NOBODY READS IT CELL BY CELL. plots/plot_mib_test_avg.py
solves that by collapsing to the Avg column, which shows the family ordering and throws away every
per-cell fact -- which method wins on which model, where the 0.25 CPR floor is being hit, whether a
family's margin is uniform or carried by two cells. This is the other projection: one facet per
(task, model) cell, all rows kept.

*** THE NUMBERS ARE PARSED OUT OF paper/tabs/mib_results.tex, NOT RECOMPUTED. *** Its sibling
imports make_mib_test_table.collect(); the validation generator has no equivalent -- make_mib_table
builds NODE_BASELINES / EDGE_BASELINES inside main(), so there is nothing importable that yields
the rows. Rather than refactor a 1,000-line table generator (or, worse, reimplement its loaders and
let the two drift), this reads the rendered table. The invariant is then literal rather than
argued: this figure cannot show a number the table does not, and `make_mib_table.py` followed by
this script is the whole update path. The cost is that the parser is coupled to the table's
formatting, so it FAILS LOUDLY -- an unparseable cell, an unknown family header or a row count
that disagrees with the column header raises instead of silently dropping bars.

ROW ORDER, FAMILY GROUPING AND LABELS ARE THE TABLE'S, in the table's own order, so a reader
holding both finds the methods in the same sequence. Bold/underline (best/second per cell) is
dropped: at 41 bars per row it would be invisible, and the same information is the bar height.

COLOUR IS THE TABLE'S SECTION HEADER, with the optimizer split kept. That split is why "ours" is
two hues and not one: palette.py's paper-wide rule is SGD -> BLACK, Adam -> BLUE, unconditionally,
and the table files our rows under exactly those two headers. It is load-bearing here in a way it
is not in the Avg figure -- both blocks contain rows labelled "MAttr" and "$+$ hard", so within a
panel the hue is the only thing telling them apart.

*** Y IS SHARED DOWN EACH COLUMN AND THE TWO COLUMNS ARE NOT COMPARABLE. *** Node CPR AUC tops out
at 2.96 and edge at 10.59; MIB's area_under is not normalised across levels (different substrates,
different unit counts), so a shared axis would offer a comparison that does not exist -- the same
reason plot_mib_test_avg gives each panel its own ticks. Sharing DOWN a column is deliberate: it
keeps the cells comparable to each other, which is what makes a flat panel (arc_easy/gemma2 at
edge level, max 4.61 against ioi/gpt2's 10.59) legible as "this cell is harder" rather than being
rescaled away.

PARTIAL ROWS ARE DRAWN WHERE THEY HAVE DATA. UGS covers 3 of 11 cells and is dropped from the Avg
figure because a 3-cell mean cannot sit beside an 11-cell one; per cell there is no such problem,
and its absence from the other eight facets is the honest rendering of the same fact.

THE 0.25 FLOOR is CPR's value for a circuit that has collapsed, not a small number -- a bar at 0.25
means the method failed on that cell, and several do. Marked with a dotted rule so a row of short
bars is not read as a smooth gradient down to zero.

Run:  uv run python plots/plot_mib_cells.py
Out:  plots/mib_cells.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mib"))
import palette as P                                     # noqa: E402

TABLE = "paper/tabs/mib_results.tex"
# Column order and headers are read off the table's own header row rather than imported from
# make_mib_table.COLUMNS, so the figure stays correct if the table is regenerated with a
# different column set. SHORT names because they are drawn as facet strips inside a ~0.6in
# panel: "Arithmetic" and "arc_challenge" do not fit rotated.
TASK_SHORT = {"IOI": "IOI", "Arithmetic": "Arith", "MCQA": "MCQA",
              "ARC (E)": "ARC-e", "ARC (C)": "ARC-c"}
MODEL_SHORT = {"GPT": "GPT-2", "Qwen": "Qwen", "Gemma": "Gemma", "Llama": "Llama"}
# The table's \textbf{...} section headers -> (family key, legend label, colour).
#
# SGD black / Adam blue is palette.py's rule, not a choice made here; see the module docstring
# for why it does real work in this figure. Gradient orange and mask indigo match
# plot_mib_test_avg's families so the two figures read the same.
FAMILY = {
    "Gradient attribution": ("gradient", "Gradient attribution", P.METHOD["IG"]),
    "Mask learning":        ("mask", "Mask learning", P.METHOD["Node Pruning"]),
    "\\ourmethod{}":        ("ours_sgd", "\\ourmethod{}$+$SGD (ours)", P.METHOD["MAttr (SGD)"]),
    "\\ourmethod{}$+$Adam": ("ours_adam", "\\ourmethod{}$+$Adam (ours)", P.METHOD["MAttr"]),
}
LEVELS = [("Node-level", "Node-level CPR AUC (↑)"), ("Edge-level", "Edge-level CPR AUC (↑)")]
CPR_FLOOR = 0.25        # CPR of a collapsed circuit -- a floor, not a small value

# 5.5in is iclr2027_conference.sty's \textwidth verbatim, so at width=\linewidth this is placed
# 1:1 and the sizes below are the sizes that reach the compiled PDF.
FIG_W = 5.5
PANEL_H = 0.60          # per facet row; 11 rows, so this is most of the height
HEAD, FOOT = 0.45, 1.35  # legend + column titles; rotated tick labels ("$+$ unif $k$, $+$ hard,
#                          $+$ Gum." is the longest at 24 chars)
FS_AXIS, FS_TICK, FS_STRIP, FS_LEG = 7, 5.2, 5.5, 6
# VALUE LABELS ARE ROTATED AND SMALL, and both are forced by the bar pitch. The node panel is
# ~3.5in wide for 28 bars, i.e. 9pt per bar; horizontal "1.85" needs ~10pt at 4pt type, so
# horizontal labels overlap at ANY size that is still legible. Rotated they need only their cap
# height across the bar and spend their length on the y axis instead, which is what YHEAD buys
# back. 4.3pt is below this repo's 6-7pt house minimum -- unavoidable at 41 labels per facet
# row, and the reason the bar heights remain the primary encoding.
FS_VAL = 4.3
YHEAD = 1.32            # ylim headroom multiplier: exactly the rotated label's length
BAR_W = 0.78


def tex_to_mpl(s):
    """A table label as mathtext. Deliberately NOT a general translator -- anything unrecognised
    RAISES, because a silent partial translation puts a stray backslash in the paper."""
    s = s.replace("\\ourmethod{}", "MAttr").replace("\\quad", "").strip()
    s = re.sub(r"\\underline\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\textbf\{(.*?)\}", r"\1", s)
    s = re.sub(r"\{=\}", "$=$", s.replace("$", "\x00")).replace("\x00", "$")
    s = s.replace("$$", "")
    if "\\" in s.replace("\\times", ""):
        raise SystemExit(f"unhandled LaTeX in label {s!r} -- extend tex_to_mpl()")
    return s


def parse_cell(c):
    """One table cell -> float or None. Strips the heat colour, the dagger and the best/second
    markup. The table went to math mode + \\cellcolor on 2026-09-02; both spellings of the
    dagger and both of the bold are accepted so this keeps working against an older table."""
    c = re.sub(r"\\cellcolor\[HTML\]\{[0-9A-Fa-f]{6}\}", "", c.strip())
    c = c.replace("$^{\\dagger}$", "").replace("^{\\dagger}", "").replace("$", "")
    c = re.sub(r"\\(?:textbf|mathbf|underline)\{(.*?)\}", r"\1", c).strip()
    if c in ("---", "--", ""):
        return None
    try:
        return float(c)
    except ValueError:
        raise SystemExit(f"unparseable cell {c!r} in {TABLE} -- extend parse_cell()")


def parse_table(path):
    """(columns, rows) from the rendered table.

    columns: [(task, model)] short display names, in table order.
    rows:    [(level, family_key, mpl_label, {(task, model): value})]
    """
    lines = [ln.rstrip() for ln in open(path)]
    cols, rows, level, fam = None, [], None, None
    for ln in lines:
        s = ln.strip()
        if s.startswith("\\textbf{Method}"):
            # Header row: Method & LR & Bwd. & <11 model names> & Avg. The TASK grouping lives in
            # the \multicolumn line above it, which is parsed separately below.
            cols = [c.strip() for c in s.split("&")[3:-1]]
            continue
        if "\\multicolumn" in s and "\\textit{" in s:
            m = re.search(r"\\textit\{(.*?)\}", s)
            level = m.group(1)
            continue
        if s.startswith("\\textbf{") and s.endswith("\\\\") and "&" not in s:
            name = re.match(r"\\textbf\{(.*)\}\s*\\\\", s).group(1)
            if name not in FAMILY:
                raise SystemExit(f"unknown section header {name!r} in {TABLE} -- add to FAMILY")
            fam = FAMILY[name][0]
            continue
        if not s.startswith("\\quad "):
            continue
        parts = [p.strip() for p in s.rstrip("\\").split("&")]
        label = tex_to_mpl(parts[0])
        vals = [parse_cell(c) for c in parts[3:3 + len(cols)]]
        rows.append((level, fam, label, vals))
    if cols is None or not rows:
        raise SystemExit(f"{TABLE}: parsed {len(rows)} rows, header={'yes' if cols else 'no'}")

    # Task grouping comes from the \multicolumn header two lines above \textbf{Method}. Rebuilt
    # by expanding each \multicolumn{n} to n copies and each bare entry to one, which is the only
    # way to line the task names up with the model names without hardcoding the layout.
    hdr = next(ln for ln in lines if "\\multicolumn{4}{c}{IOI}" in ln)
    tasks = []
    for f in [f.strip() for f in hdr.strip().rstrip("\\").split("&")][3:]:
        m = re.match(r"\\multicolumn\{(\d+)\}\{c\}\{(.*)\}", f)
        if m:
            tasks += [m.group(2)] * int(m.group(1))
        elif f:
            tasks.append(f)
    tasks = tasks[:len(cols)]
    if len(tasks) != len(cols):
        raise SystemExit(f"{TABLE}: {len(tasks)} task headers for {len(cols)} model columns")
    columns = [(TASK_SHORT.get(t, t), MODEL_SHORT.get(m, m)) for t, m in zip(tasks, cols)]
    return columns, [(lv, fm, lab, dict(zip(columns, vs))) for lv, fm, lab, vs in rows]


def nice_ticks(top, nmax=3):
    """At most `nmax` round ticks covering [0, top]."""
    for step in (0.5, 1, 2, 5, 10, 20, 50):
        if top / step <= nmax:
            return [step * i for i in range(int(top / step) + 1)]
    return [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/mib_cells.pdf")
    a = ap.parse_args()

    columns, rows = parse_table(TABLE)
    per_level = {lv: [r for r in rows if r[0] == lv] for lv, _ in LEVELS}
    for lv, _ in LEVELS:
        if not per_level[lv]:
            raise SystemExit(f"no {lv} rows parsed from {TABLE}")
    # Shared per COLUMN, not globally -- see the docstring.
    ytop = {lv: max(v for _, _, _, d in per_level[lv] for v in d.values() if v is not None)
            for lv, _ in LEVELS}
    # Ticks from the DATA max, not from ylim -- ylim carries YHEAD's label headroom, so letting a
    # locator work on it puts a tick in the empty band above the tallest bar (node read 0/2 with
    # a stray 4 clipped off; edge read 0/5/10 by luck).
    yticks = {lv: nice_ticks(t) for lv, t in ytop.items()}

    plt.rcParams.update(P.RC)
    nrow = len(columns)
    fh = nrow * PANEL_H + HEAD + FOOT
    fig, axes = plt.subplots(nrow, len(LEVELS), figsize=(FIG_W, fh),
                             gridspec_kw=dict(
                                 width_ratios=[len(per_level[lv]) for lv, _ in LEVELS]))
    for ri, cell in enumerate(columns):
        for ci, (lv, ylab) in enumerate(LEVELS):
            ax = axes[ri][ci]
            recs = per_level[lv]
            vals = [d.get(cell) for _, _, _, d in recs]
            ax.bar([i for i, v in enumerate(vals) if v is not None],
                   [v for v in vals if v is not None], width=BAR_W,
                   color=[FAMILY_BY_KEY[f][2] for (_, f, _, _), v in zip(recs, vals)
                          if v is not None], lw=0, zorder=2)
            ax.axhline(CPR_FLOOR, color="#999999", lw=0.4, ls=(0, (1, 2)), zorder=1)
            ax.set_xlim(-0.5 - BAR_W / 4, len(recs) - 0.5 + BAR_W / 4)
            ax.set_ylim(0, ytop[lv] * YHEAD)
            for i, v in enumerate(vals):
                if v is None:
                    continue
                # offset points, not data units: the two columns have different y ranges, so a
                # constant in data units sits flush on one and floats above the other.
                ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 1.2),
                            ha="center", va="bottom", rotation=90, fontsize=FS_VAL, zorder=3)
            P.furnish(ax)
            ax.grid(False, axis="x")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", labelsize=FS_TICK, length=1.5, pad=1.2)
            ax.set_yticks(yticks[lv])
            ax.tick_params(axis="x", length=0)
            ax.set_xticks(range(len(recs)))
            if ri == nrow - 1:
                ax.set_xticklabels([lab for _, _, lab, _ in recs], fontsize=FS_TICK,
                                   rotation=45, ha="right", rotation_mode="anchor")
            else:
                ax.set_xticklabels([])
            if ri == 0:
                ax.set_title(ylab, fontsize=FS_AXIS, pad=3)
        # Facet strip on the right edge of the row, ggplot-style: the cell name is ~0.5in at
        # 5.5pt rotated, which fits a PANEL_H-tall panel where a left-hand ylabel would not.
        axes[ri][-1].text(1.015, 0.5, "/".join(cell), transform=axes[ri][-1].transAxes,
                          rotation=270, va="center", ha="left", fontsize=FS_STRIP)

    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.4)
    # HEAD is reserved ABOVE the axes for the column titles AND the legend, in that vertical
    # order. tight_layout does not know about the figure-level legend, so without this the two
    # land on the same line -- which is exactly what happened.
    fig.subplots_adjust(top=1.0 - HEAD / fh)
    seen, handles = set(), []
    for _, fam, _, _ in rows:
        if fam not in seen:
            seen.add(fam)
            _, lab, col = FAMILY_BY_KEY[fam]
            handles.append(Patch(facecolor=col, label=tex_to_mpl(lab)))
    fig.legend(handles=handles, fontsize=FS_LEG, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 0.999), frameon=False, handlelength=1.2, handleheight=1.0,
               handletextpad=0.4, columnspacing=1.6)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {a.out} (+ .png)   {len(columns)} cells x "
          f"{'+'.join(str(len(per_level[lv])) for lv, _ in LEVELS)} methods")
    for lv, _ in LEVELS:
        n_part = sum(1 for _, _, _, d in per_level[lv]
                     if sum(v is not None for v in d.values()) < len(columns))
        print(f"  {lv:<11} {len(per_level[lv])} rows, y to {ytop[lv]:.2f}, "
              f"{n_part} partial")
    return 0


FAMILY_BY_KEY = {v[0]: v for v in FAMILY.values()}

if __name__ == "__main__":
    sys.exit(main())
