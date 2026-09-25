"""Path-weight densities of MAttr's first update, one panel per optimiser regime, one line per
budget schedule (paper/sections/gradient.tex, "Path weights under general schedules").

    uv run python plots/plot_path_weights.py   ->  paper/figs/path_weights.pdf

The first update is a path integral of the centred IxG effect along t (the intervention
coefficient of the uniform mask: t = 0 is the base run, t = 1 fully source-patched). Which
weight rho(t) multiplies the integrand depends on the optimiser and on the density p(t) that
the budget schedule induces on t = 1 - k/N:

    SGD, and the leading term of Adam at large eps :  rho   ~ p(t) t(1-t)
    first correction of Adam at large eps          :  rho_2 ~ p(t) t^2 (1-t)^2
    Adam at tiny eps (sign regime)                 :  rho   = p(t)

    uniform k       p(t) = 1                 on [1/N, 1-1/N]
    log-uniform k   p(t) = 1/((1-t) ln N)    on [0,   1-1/N]
    logit k         p(t) ~ 1/(t(1-t))        on [1/N, 1-1/N]

Every curve is normalised to integrate to 1 on its support (numerically, so the truncated
log-uniform / logit weights are exact rather than the O(1/N) closed forms). N only sets the
truncation; the shapes are N-free apart from the edge mass of the two singular schedules. The
y-axis is linear and clipped at 3: the log-uniform weight reaches N / ln N at t = 1 - 1/N and
the logit weight diverges at both ends, so those run off the panel by design.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (aes, coord_cartesian, element_blank, element_line, element_text,
                      facet_wrap, geom_line, ggplot, labs, scale_color_brewer,
                      scale_x_continuous, scale_y_continuous, theme, theme_bw, theme_set)

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
N = 1024

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.9),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

# grid on [1/N, 1-1/N], dense at both ends so the singular schedules are resolved
edge = np.geomspace(1 / N, 0.5, 600)
t = np.unique(np.concatenate([edge, 1 - edge]))

SCHEDULES = {                      # induced density p(t), up to normalisation
    "uniform $k$":     lambda t: np.ones_like(t),
    "log-uniform $k$": lambda t: 1 / (1 - t),
    "logit $k$":       lambda t: 1 / (t * (1 - t)),
}
REGIMES = {                        # weight = p(t) * gate(t), up to normalisation
    "SGD / Adam, large ε (leading)":  lambda t: t * (1 - t),
    "Adam, large ε (correction)":     lambda t: t ** 2 * (1 - t) ** 2,
    "Adam, tiny ε":                   lambda t: np.ones_like(t),
}
rows = []
for sched, p in SCHEDULES.items():
    for regime, gate in REGIMES.items():
        w = p(t) * gate(t)
        w = w / np.trapezoid(w, t)
        rows.append(pd.DataFrame({"t": t, "rho": w, "schedule": sched, "regime": regime}))
df = pd.concat(rows)
df["schedule"] = pd.Categorical(df["schedule"], list(SCHEDULES))
df["regime"] = pd.Categorical(df["regime"], list(REGIMES))

fig = (
    ggplot(df, aes("t", "rho", color="schedule"))
    + geom_line(size=0.6)
    + facet_wrap("~regime", nrow=1)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_continuous(breaks=[0, 0.25, 0.5, 0.75, 1], labels=["0", ".25", ".5", ".75", "1"])
    + scale_y_continuous(breaks=[0, 1, 2, 3])
    + coord_cartesian(ylim=(0, 3))          # log-uniform / logit weights diverge at the ends
    + labs(x="Path position $t$ (0 = base, 1 = source)", y="Path weight $\\rho(t)$", color="")
)
fig.save(OUT / "path_weights.pdf")
print(f"-> {OUT / 'path_weights.pdf'}")
