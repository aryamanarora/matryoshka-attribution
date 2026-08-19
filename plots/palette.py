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
    "Node Pruning": "#332288",   # Tol indigo; mask-learning baseline, MIB scatter + curves
    # Wong reddish purple: the other mask-learning baseline, so it stays in Node Pruning's
    # cool-purple family while separating from it by lightness (L* ~60 vs ~24). Deliberately
    # NOT the warm #d98d3a it started as -- that is a near-twin of IG's Wong orange, and in
    # the curve figures the two would cross each other in every panel.
    "DBM": "#cc79a7",
}
OTHER = "#cccccc"   # un-highlighted baselines in the MIB scatter

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
    # Same reasoning as the IG ladder: MAttr-SGD is the SAME forward and the same backward as
    # MAttr, differing only in the optimizer, so it shares MAttr's blue and separates by
    # linetype. It is deliberately not given a hex of its own -- the cool region is already
    # occupied by +hard (bluish green), GIM (sky blue) and Node Pruning (indigo), and a fifth
    # cool hue would become this palette's binding CVD constraint for what is a hyperparameter
    # change, not a different method.
    "MAttr (SGD)": "MAttr", "MAttr-SGD": "MAttr", "softsgd-log": "MAttr",
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
