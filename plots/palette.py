"""Single source of truth for method colours across every figure.

Import this instead of writing hex codes into a figure script. Before this module the same
4-method palette was hand-copied into five files (plot_accauc_vs_faithauc, the cause and k1
figures via its METHODS dict, plot_mib_accauc_cpr_scatter, plot_mib_curves,
plot_mib_logitdiff_curves), so a recolour either touched all five or left the paper showing two
different colours for the same method.

Colours are Wong / Tol colourblind-safe stock, chosen so the two gradient baselines separate by
LIGHTNESS as well as hue: IG is a light orange (L*~72), I×G a dark wine (L*~30). An earlier pass
used vermillion + reddish purple, which is well separated numerically (dE 74 in normal vision)
but reads as one warm family at 2.6pt marker size -- lightness is the cue that survives both
small markers and every CVD type.

  ours      = cool  (MAttr blue, +hard bluish green)
  baselines = warm  (IG orange, I×G wine)

Run `python plots/palette.py` to re-verify: it simulates deuteranopia / protanopia / tritanopia
(Machado et al. 2009, severity 1.0) and prints the worst pairwise CIE76 dE. Rule of thumb: dE>20
is comfortably distinguishable. Keep the worst-case above that if you change anything.
"""

OTHER = "#cccccc"   # un-highlighted baselines in the MIB scatter

# canonical name -> hex
METHOD = {
    "MAttr": "#0072b2",   # Wong blue
    "+hard": "#009e73",   # Wong bluish green
    "IG":    "#e69f00",   # Wong orange   (light)
    "I×G":   "#882255",   # Tol wine      (dark)
    "GIM":   "#56b4e9",   # Wong sky blue; 5th series, MIB curve figures only
    # Wong vermillion: the LRP-family gradient baseline on SVA+. Warm like IG (they are both
    # gradient methods) but separated by lightness, L* 54 vs 72 -- the same cue the IG / I×G
    # pair relies on. Its worst CVD distance is dE 18.3 (deuteranopia, vs IG), which is under
    # the dE>20 rule of thumb but above this palette's pre-existing worst pair (16.1), so it
    # does not become the binding constraint. Checked with `python plots/palette.py`.
    "AttnLRP": "#d55e00",
    # Tol sand: "stepless" IG (alpha ~ U(0,1) per example at m=1) on SVA+. It is an IG ESTIMATOR
    # variant, not a new method, so it belongs in the warm gradient family with IG and AttnLRP --
    # but the family already holds three hues, and this is the fourth. Sand is the ONLY candidate
    # tried that does not become the new binding constraint: its worst CVD distance is 17.0 dE
    # (tritanopia, vs Random), just above this palette's pre-existing worst pair of 16.1 (+hard
    # vs GIM, tritanopia). Every other warm collapses against the family it has to join --
    # goldenrod #b8860b 3.2 dE and olive #999933 13.5 (both vs AttnLRP), sienna #a65628 15.2 and
    # #7f3b08 10.7 (vs I×G), salmon #ee8866 10.0 (vs +hard). The comment on "MAttr (SGD)" below
    # already identified sand and black as the only two additions that leave 16.1 intact; black
    # went there, sand comes here. The 17.0 is against Random, which in the scatter also carries
    # its own marker SHAPE (a star, LOSS_SHAPE[NO_LOSS]), so hue is not the only cue there.
    # Re-verify with `python plots/palette.py`.
    "Stepless IG": "#ddcc77",
    "Node Pruning": "#332288",   # Tol indigo; mask-learning baseline, MIB scatter + curves
    # Wong reddish purple: the other mask-learning baseline, so it stays in Node Pruning's
    # cool-purple family while separating from it by lightness (L* ~60 vs ~24). Deliberately
    # NOT the warm #d98d3a it started as -- that is a near-twin of IG's Wong orange, and in
    # the curve figures the two would cross each other in every panel.
    "DBM": "#cc79a7",
    # MAttr with Adam -> SGD. Previously aliased to MAttr's blue and separated by linetype, on
    # the argument that an optimizer change is not a different method; that made it undrawable
    # in any figure where linetype is already spoken for or absent (the acc/faith scatter),
    # so it now has a hex.
    #
    # BLACK, and that is a measured choice rather than a stylistic one. Every cool hue this
    # could plausibly have taken collapses the palette: navy 10.6 dE (vs Node Pruning,
    # tritanopia), Tol light blue 9.6 (vs GIM), steel 8.7 (vs GIM), Tol teal 4.3 (vs DBM) --
    # all BELOW the palette's pre-existing worst pair of 16.1 (+hard vs GIM, tritanopia), i.e.
    # each would have become the new binding constraint. Of everything tried only black and Tol
    # sand (#ddcc77) leave that 16.1 intact; black wins on separation (30.9 dE worst case, vs
    # I×G under protanopia) and stays out of the warm family, which this palette reserves for
    # the gradient BASELINES. It does break the "ours = cool" rule, so read it as a neutral
    # rather than a cool -- the alternative was making some other pair harder to tell apart.
    # Re-verify with `python plots/palette.py`.
    "MAttr (SGD)": "#000000",
    # The random-ranking floor. Achromatic on purpose: it is a REFERENCE, not a competitor, so
    # it should not read as another method fighting for attention. Same value as OTHER, which is
    # already this palette's "un-highlighted" grey -- and under a black marker edge it reads as
    # a hollow marker, which is the right visual weight for a floor.
    #
    # It has to be a LIGHT grey. A neutral is separated from the hues only by lightness, and
    # mid-greys land on top of the desaturated CVD renderings of the chromatic entries: checked
    # against _check(), #999999 gives 7.1 dE (deuteranopia, vs DBM), #aaaaaa 9.6, #888888 9.3
    # (vs +hard), #666666 8.4 (vs I×G) -- every one of them BELOW this palette's pre-existing
    # worst pair of 16.1 (+hard vs GIM, tritanopia), i.e. each would have become the new binding
    # constraint. #cccccc and lighter leave that 16.1 untouched.
    "Random": OTHER,
}
OTHER = "#cccccc"   # un-highlighted baselines in the MIB scatter

# Font setup for the raw-matplotlib figures. Lives here for the same reason the hexes do: it was
# hand-copied into plot_mib_accauc_cpr_scatter and then again into every figure that wanted to
# match it, which is the duplication this module exists to end. plot_mib_accauc_cpr_scatter.RC
# is now an alias for this dict, so the two cannot drift.
#
# Inter matches the plotnine theme the rest of the paper's figures use; these figures are raw
# matplotlib (label placement needs per-annotation control), so the font has to be set on
# rcParams by hand -- plotnine's theme does not reach it. mathtext gets Inter too: on the DejaVu
# default, "$k^\\star$" would render in a visibly different face from the text beside it. cal/sf/tt
# are unused, but a "custom" fontset resolves all of them at import time and the cal default
# ("cursive") is not installed, so leaving them emits a findfont warning on every run. fonttype
# 42 embeds real TrueType outlines instead of Type-3, which is what arXiv wants.
RC = {
    "font.family": "Inter", "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42, "text.color": "#000000",
    "axes.labelcolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000",
}

# Axis furniture every raw-matplotlib figure here shares: hairline grid behind the data,
# half-weight spines. A function rather than more rcParams because `set_axisbelow` and the spine
# widths are per-Axes, and figures in this repo mix gridded panels with ungridded ones.
GRID_LW, GRID_COLOR, SPINE_LW = 0.25, "#dddddd", 0.5


def furnish(ax):
    """Apply the shared grid/spine treatment to one Axes."""
    ax.grid(True, lw=GRID_LW, color=GRID_COLOR)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)

# Qualitative suitability marks (+ / o / -) in the teaser figure's method-property table.
# Not method colours -- a separate three-level ordinal scale -- but kept here so the whole
# paper still has exactly one file with hex codes in it. Wong colourblind-safe stock:
# bluish green / orange / vermillion, which also separate by lightness (L* 60 / 72 / 54).
RATING = {"+": "#009e73", "o": "#e69f00", "-": "#d55e00"}

# The two poles of the contrast traced in the ViT sparsity-ladder figure: `pos` is the class
# being explained, `neg` the class it is explained against. Keyed by ROLE, not by animal, so
# the explained class is always the blue line whichever way round the run was set up. A
# different semantic space from METHOD (classes, not attribution methods), so the hexes are
# deliberately reused rather than new ones invented -- Wong blue / orange, L* 47 vs 72.
CLASS = {"pos": "#0072b2", "neg": "#e69f00"}

# spellings that appear as series labels in individual figures
ALIASES = {
    "MAttr (log)": "MAttr", "stopk-log": "MAttr", "MAttr-cause": "MAttr",
    "+hard (log)": "+hard", "soft-log": "+hard", "+hard-cause": "+hard",
    "IxG": "I×G", "I$\\times$G": "I×G", "NAP-IG": "IG",
    # The IG step-count ladder is ONE method at three integration budgets, so all three share
    # IG's orange and separate by LINETYPE in the figure, not by hue. Giving them their own
    # hexes was tried and rejected: the warm-dark region this palette leaves free is already
    # boxed in by I×G (wine, L*30) and AttnLRP (vermillion, L*54). A 3-step OrRd ramp
    # (#e69f00 / #cc4c02 / #8c2d04) lands its dark end 13.1 dE from I×G under tritanopia --
    # below this palette's pre-existing worst pair of 16.1, so it would have become the new
    # binding constraint -- and its mid tone is 7.9 dE from AttnLRP in NORMAL vision, i.e. a
    # near-twin of an existing method. Ordered hyperparameter -> ordered linetype is also the
    # more honest encoding: colour is reserved for "different method" everywhere else here.
    "IG (5 steps)": "IG", "IG (10 steps)": "IG", "IG (30 steps)": "IG",
    # MAttr-SGD now has its own hex (see METHOD) -- these are just its other spellings. NOTE
    # this is the one place the IG-ladder reasoning above does NOT apply: that ladder stays
    # linetype-encoded because all three rungs appear in figures that HAVE a spare linetype,
    # whereas the SGD arm had to be droppable from the acc/faith scatter for want of one.
    "MAttr-SGD": "MAttr (SGD)", "softsgd-log": "MAttr (SGD)",
}


def color(name):
    """Colour for a method under any of its label spellings; OTHER for unknown names."""
    return METHOD.get(ALIASES.get(name, name), OTHER)


def _check():   # python plots/palette.py
    import itertools
    import numpy as np

    def s2lin(c):
        c = c / 255.0
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    WP = np.array([0.95047, 1.0, 1.08883])

    def lab(rgb):
        x = s2lin(np.asarray(rgb, float)) @ M.T / WP
        f = np.where(x > 0.008856, np.cbrt(x), 7.787 * x + 16 / 116)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

    CVD = {
        "normal": np.eye(3),
        "deuter": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                            [-0.011820, 0.042940, 0.968881]]),
        "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
        "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
    }

    def sim(hexs, m):
        rgb = np.array([int(hexs[i:i + 2], 16) for i in (1, 3, 5)], float)
        o = np.clip(s2lin(rgb) @ m.T, 0, 1)
        srgb = np.where(o <= 0.0031308, o * 12.92, 1.055 * o ** (1 / 2.4) - 0.055)
        return np.clip(srgb * 255, 0, 255)

    names = list(METHOD)
    print("worst pairwise CIE76 dE per vision type (>20 = comfortably distinguishable)")
    overall = (1e9, None)
    for vis, m in CVD.items():
        labs = {n: lab(sim(METHOD[n], m)) for n in names}
        pairs = [(np.linalg.norm(labs[a] - labs[b]), a, b)
                 for a, b in itertools.combinations(names, 2)]
        d, a, b = min(pairs)
        print(f"  {vis:7s} {d:6.1f}  ({a} vs {b})")
        overall = min(overall, (d, f"{vis}: {a} vs {b}"))
        if vis == "normal":
            print(f"          IG vs I×G {np.linalg.norm(labs['IG'] - labs['I×G']):.1f}, "
                  f"L* {labs['IG'][0]:.0f} vs {labs['I×G'][0]:.0f}")
    print(f"\noverall worst: {overall[0]:.1f} ({overall[1]})")


if __name__ == "__main__":
    _check()
