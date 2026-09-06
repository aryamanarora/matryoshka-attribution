"""Did MAttr converge on the ViT teaser? -- the sufficiency-AUC probe traced during training.

`scripts/vit/vit_teaser_attr.py` evaluates, every `--probe-every` steps, the hard top-k
sufficiency curve of the *current* score ranking over MIB's sparsity grid and records its
log-spaced area (`mattr/auc_trace` in the npz). That is the quantity being optimised; the
raw training loss is not a convergence signal, because k is resampled every step and the
loss trace mostly tracks which k was drawn.

Plots four 2000-step seeds against one 20000-step run, with the AUC that the two strongest
baselines reach (from `faithfulness.json`) as reference lines.

    uv run python plots/plot_vit_teaser_convergence.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (aes, element_blank, element_line, element_text, geom_hline, geom_line,
                      geom_text, ggplot, labs, scale_alpha_manual, scale_size_manual,
                      scale_x_log10, theme, theme_bw, theme_set)

from palette import METHOD

# plotnine renders through matplotlib but does not set this, and the default (Type 3) is
# not a real embedded outline font
plt.rcParams["pdf.fonttype"] = 42

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(3.3, 1.9),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

RUNS = [("seed 42 (2k steps)", "results/vit_teaser_pixelate"),
        ("seed 43 (2k steps)", "results/vit_teaser_pix_seed43"),
        ("seed 44 (2k steps)", "results/vit_teaser_pix_seed44"),
        ("seed 45 (2k steps)", "results/vit_teaser_pix_seed45"),
        ("seed 42 (20k steps)", "results/vit_teaser_pix_long")]
REFS = ["kernelshap", "attnlrp"]
LABEL = {"kernelshap": "KernelSHAP", "attnlrp": "AttnLRP"}


SMOOTH_STEPS = 200   # width of the smoothing window, in training steps


def rolling(y, spacing):
    """Centred rolling median over a fixed number of *training steps*.

    The per-probe values are noisy (each probe is a handful of corruption draws), so the
    trend is what carries the convergence claim, not any single point. The window is set in
    steps rather than probes because the runs are probed at different intervals -- a fixed
    probe-count window would smooth the 20k run over 4x more training than the 2k ones and
    drag its early points up towards its plateau.
    """
    w = max(3, round(SMOOTH_STEPS / max(spacing, 1)))
    return pd.Series(y).rolling(w, center=True, min_periods=1).median().to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faithfulness", default="results/vit_teaser_pixelate/faithfulness.json")
    ap.add_argument("--out", default="paper/figs/vit_teaser_convergence.pdf")
    args = ap.parse_args()

    rows = []
    for name, path in RUNS:
        t = np.load(Path(path) / "attributions.npz")["mattr/auc_trace"]
        spacing = float(np.median(np.diff(t[:, 0]))) if len(t) > 1 else 1.0
        rows.append(pd.DataFrame({"step": np.maximum(t[:, 0], 1),
                                  "auc": rolling(t[:, 1], spacing), "run": name,
                                  "length": "20k steps" if "20k" in name else "2k steps"}))
    df = pd.concat(rows)

    logf = np.log(json.loads(Path(args.faithfulness).read_text())["fracs"])
    curves = json.loads(Path(args.faithfulness).read_text())["curves"]
    ref = {m: float(np.trapezoid(curves[m], logf) / (logf[-1] - logf[0])) for m in REFS}

    p = (ggplot(df, aes("step", "auc", group="run"))
         + geom_hline(yintercept=list(ref.values()), linetype="dashed", size=0.3, color="#000000")
         + geom_text(pd.DataFrame({"step": df["step"].max(),
                                   "auc": [v + 0.07 for v in ref.values()],
                                   "label": [LABEL[m] for m in ref]}),
                     aes("step", "auc", label="label"), size=5.5, ha="right", va="bottom",
                     inherit_aes=False)
         + geom_line(aes(alpha="length", size="length"), color=METHOD["MAttr"])
         + scale_alpha_manual(values={"2k steps": 0.45, "20k steps": 1.0})
         + scale_size_manual(values={"2k steps": 0.4, "20k steps": 0.7})
         + scale_x_log10(breaks=[1, 10, 100, 1000, 10000],
                         labels=["1", "10", "10²", "10³", "10⁴"])
         + labs(x="Training step", y="Sufficiency AUC"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, verbose=False)
    p.save(out.with_suffix(".png"), dpi=300, verbose=False)
    print(f"wrote {out}")
    for m, v in ref.items():
        print(f"  reference {LABEL[m]:<12} AUC {v:.3f}")
    for name, path in RUNS:
        t = np.load(Path(path) / "attributions.npz")["mattr/auc_trace"]
        tail = t[t[:, 0] >= 0.5 * t[-1, 0], 1]
        print(f"  {name:<20} final-half AUC {tail.mean():.3f} +- {tail.std():.3f}")


if __name__ == "__main__":
    main()
