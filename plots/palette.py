"""Single source of truth for method colours across every figure.

Import this instead of writing hex codes into a figure script. Before this module the same
4-method palette was hand-copied into five files (plot_accauc_vs_faithauc, the cause and k1
figures via its METHODS dict, plot_mib_accauc_cpr_scatter, plot_mib_curves,
the old plot_mib_logitdiff_curves), so a recolour either touched all five or left the paper showing two
different colours for the same method.

Colours are Wong / Tol colourblind-safe stock, chosen so the two gradient baselines separate by
LIGHTNESS as well as hue: IG is a light orange (L*~72), I×G a dark wine (L*~30). An earlier pass
used vermillion + reddish purple, which is well separated numerically (dE 74 in normal vision)
but reads as one warm family at 2.6pt marker size -- lightness is the cue that survives both
small markers and every CVD type.

  ours      = cool  (MAttr blue, +hard bluish green)
  baselines = warm  (IG orange, I×G wine)

THE OPTIMIZER RULE, which every figure follows without exception:

  SGD  -> BLACK   ("MAttr (SGD)")
  Adam -> BLUE    ("MAttr")

Unconditionally -- whether or not both optimizers appear on the same panel. Two earlier
attempts at something cleverer both failed the same way. Sharing MAttr's blue across
optimizers and separating by linetype left MAttr+SGD blue in plot_mib_accauc_cpr_scatter and
black in plot_optimizer_lr / plot_adamsgd_mlp_diag. Making the hue CONDITIONAL on whether a
panel showed one optimizer or two was worse still: the arm then changed colour between two cuts
of the SAME figure. If a figure needs a third MAttr arm, give it a hex here; do not re-purpose
blue or black.

"Blue = Adam" means the Adam configuration we RECOMMEND, i.e. eps=1e-2 at neuron scale. The
library-default eps=1e-8 arm is the broken one and carries the violet as a marked variant --
see "MAttr (Adam, default eps)" below.

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
    # Tol sand: Expected Gradients (alpha ~ U(0,1) per example at m=1) on SVA+. It is an IG ESTIMATOR
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
    "Expected Gradients": "#ddcc77",
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
    # MAttr with Adam at the LIBRARY-DEFAULT eps=1e-8, as distinct from the eps=1e-2 arm we
    # actually ship (scripts/sva/launch/submit_sva_eps.sh), which takes the canonical MAttr blue.
    #
    # WHICH WAY ROUND THIS GOES IS THE POINT. The paper-wide rule is SGD black / Adam blue,
    # unconditionally, so "blue" has to be the Adam configuration we recommend. Default-eps Adam
    # is the BROKEN one at neuron scale -- its update degenerates to ~sign(g) and costs ~0.10 IIA
    # AUC on the MLP substrates -- so it is the variant worth marking, and it gets the violet.
    # A third optimizer setting of the same method needs a hex rather than a linetype: the
    # acc/faith scatter already spends shape on the training loss and fill on the method.
    #
    # BLUEVIOLET, and it is the first addition here that gets to stay in the "ours = cool"
    # family rather than breaking the rule the way black had to. Measured over a 4096-point RGB
    # grid against the whole palette under all three CVD simulations, the violet region is by a
    # wide margin the emptiest part of the space this palette leaves free -- its worst-case
    # distance is 28.1 dE (deuteranopia, vs Node Pruning), against 24.0 for the next candidate
    # tried (orchid #b366ff), 22.6 (deep violet #6a00cc) and 21.1 (#b07aff). Everything in the
    # muted-purple region that looked more natural next to the Wong/Tol stock fails outright:
    # Tol purple #AA4499 8.0 dE (vs MAttr), plum #6a3d9a 8.2, amethyst #9966dd 12.8 -- each
    # BELOW this palette's pre-existing worst pair of 16.05 (+hard vs GIM, tritanopia), i.e.
    # each would have become the new binding constraint. Blueviolet leaves that 16.05 intact.
    #
    # ONE CAVEAT, stated because this palette's stated preference is lightness separation: at
    # L* 42 this sits close to MAttr's blue (L* 47), so the two are told apart by HUE, not by
    # lightness -- the one pair here that leans on the cue this module normally treats as the
    # backup. That is deliberate and is why a violet was required to clear 20 dE by a margin
    # rather than merely clear it: the hue distance has to do all the work. Semantically it is
    # also the right place for it, since ours are cool and the warm hues are reserved for the
    # gradient baselines. Re-verify with `python plots/palette.py`.
    "MAttr (Adam, default eps)": "#8a2be2",
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
    # Mid-dark grey, not OTHER's #cccccc (2026-09-11, requested): as an outlined marker or a
    # bar beside saturated hues the light grey vanished at print size. Only plot_mib_test_avg
    # and plot_accauc_vs_faithauc read this entry; OTHER is unchanged for its other users.
    # #6e6e6e is the best of seven greys tried against the check below (worst 11.6 dE, I×G vs
    # Random, deuteranopia): #8c8c8c / #999999 / #a0a0a0 collide with DBM's reddish purple at
    # 7-8.5, #666666 and below with I×G at 8.4 and 4.3. It is under the palette's previous
    # 16.1 floor; accepted because Random also sits at the origin and is labelled in both
    # figures, so hue is not its only cue.
    "Random": "#6e6e6e",
}

# WITHIN-FAMILY TINTS, kept OUT of METHOD on purpose. These are lightness variants of a hue
# already in METHOD, used where one method appears at two settings and the pair is also
# distinguished by label or linetype -- they are not competing hues and must not be scanned by
# _check() as if they were. Adding "Node Pruning (KL)" to METHOD directly dropped the palette's
# reported worst case from 16.1 to 8.1 dE (GIM vs the tint, protanopia), which is a meaningless
# number: GIM and the two Node Pruning arms never appear in the same panel, and the tint's real
# job is to separate from Node Pruning's own indigo, which it does by L* 62 vs 24.
TINT = {
    # Node Pruning's KL-objective arm against its logit-diff one, in the MIB scatter. Lived as
    # a local hex in plot_mib_accauc_cpr_scatter.py until 2026-08-29.
    "Node Pruning (KL)": "#9d95d1",
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
# Panel frames are BLACK across the paper (2026-09-08), matching figs/baseline_strongreject.
# Set explicitly rather than left to matplotlib's axes.edgecolor default so a future rcParams
# change cannot quietly grey them, and so the plotnine figures have one value to copy: their
# theme_bw default panel_border is grey20, which read visibly lighter next to these.
SPINE_COLOR = "#000000"


def furnish(ax):
    """Apply the shared grid/spine treatment to one Axes: full black frame + faint grid."""
    ax.grid(True, lw=GRID_LW, color=GRID_COLOR)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_LW)
        sp.set_color(SPINE_COLOR)
        sp.set_visible(True)

# Outline colours for the "beats baseline" marks on plots/plot_epsgrid_facets.py. THREE MUTUALLY
# EXCLUSIVE categories, one outline per cell, so a cell is marked once and the colour says which
# baseline(s) it cleared.
#
# Two of the three REUSE THE BEATEN BASELINE'S OWN METHOD COLOUR -- an orange outline means it
# cleared IG (orange everywhere in this paper) and a black one means it cleared MAttr+SGD (black,
# per the SGD rule above). That is the whole reason no new hue was invented for them: the reader
# already knows those two colours, and inventing a second orange for "the thing IG is" would be
# the exact duplication this module exists to prevent.
#
# "both" is the one new hex, and it is deliberately NOT a method colour. It has to separate from
# orange AND black AND from the two backgrounds these outlines sit on (viridis: dark purple ->
# yellow; RdBu_r: blue -> white -> red). Purple clears all four; a red or magenta would collide
# with RdBu_r's warm arm, and a green with viridis's middle. Every outline is drawn over a white
# halo (path_effects.withStroke), so contrast against the cell is already handled -- what these
# three must do is separate from EACH OTHER.
OUTLINE = {"ig": METHOD["IG"], "sgd": METHOD["MAttr (SGD)"], "both": "#7b3294"}

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
    # The big-eps Adam arm, under the key plot_accauc_vs_faithauc.parse_method emits and under
    # the two spellings the figures label it with. The eps VALUE is part of every spelling on
    # purpose: a future eps=1e-1 arm is a different series, and an alias that matched on "eps"
    # alone would silently colour it the same.
    # SGD is BLACK and Adam is BLUE, unconditionally, across every figure. So the Adam arm we
    # actually ship (eps=1e-2) takes the canonical MAttr blue, and the violet marks the
    # LIBRARY-DEFAULT eps arm instead -- the broken one, which is the variant worth flagging.
    "stopk-log-eps1e-2": "MAttr",
    "MAttr (Adam, ε=10⁻²)": "MAttr", "MAttr+Adam": "MAttr", "Adam": "MAttr",
    # Back-compat: this was the entry's name until 2026-08-29, when blue and violet swapped
    # roles. Kept so an un-updated caller gets the right colour rather than OTHER's grey --
    # which is exactly what it silently got during the swap.
    "MAttr (Adam eps=1e-2)": "MAttr",
    "stopk-log": "MAttr (Adam, default eps)",
    "MAttr (Adam, def. eps)": "MAttr (Adam, default eps)",
    # SGD spellings, so a figure can pass parse_method's key or a display label either way.
    "MAttr+SGD": "MAttr (SGD)", "SGD": "MAttr (SGD)", "softsgd-unif": "MAttr (SGD)",
}


def color(name):
    """Colour for a method under any of its label spellings; OTHER for unknown names.

    Searches METHOD then TINT, so a caller does not need to know which of the two a series
    lives in -- that split exists for _check()'s benefit, not the caller's.
    """
    key = ALIASES.get(name, name)
    return METHOD.get(key, TINT.get(key, OTHER))


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
