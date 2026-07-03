"""CE vs % dims kept (sparsity sweep), averaged over the 29 tasks: learned vs random
ordering, full vs lean. Reads results/pythia1b_multitask_das{,_lean}.pkl. Run on sc."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_ribbon, labs, scale_x_log10,
    scale_color_manual, theme_bw, theme_set, theme, element_text,
)

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(theme_bw(base_size=8) + theme(text=element_text(family="Inter")))

full = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
lean = pickle.load(open(R / "pythia1b_multitask_das_lean.pkl", "rb"))
sp = np.array(full["sparsities"]) * 100  # percent


def mean_curve(d, key):
    vals = [ev[key] for ev in d["per_task_eval"].values() if ev.get(key)]
    a = np.array(vals)
    return a.mean(0), a.std(0) / np.sqrt(len(a))


rows = []
for d, tag in [(full, "full (3000)"), (lean, "lean (1500)")]:
    for key, lab in [("eval_learned_ce", "learned"), ("eval_random_ce", "random")]:
        m, se = mean_curve(d, key)
        for i, pct in enumerate(sp):
            rows.append({"pct": pct, "ce": m[i], "lo": m[i]-se[i], "hi": m[i]+se[i],
                         "series": f"{lab} — {tag}"})
df = pd.DataFrame(rows)
order = ["learned — full (3000)", "learned — lean (1500)",
         "random — full (3000)", "random — lean (1500)"]
df["series"] = pd.Categorical(df["series"], categories=order, ordered=True)
pal = {"learned — full (3000)": "#e41a1c", "learned — lean (1500)": "#fb9a99",
       "random — full (3000)": "#377eb8", "random — lean (1500)": "#a6cee3"}

p = (ggplot(df, aes("pct", "ce", color="series", fill="series"))
     + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.15, color="none")
     + geom_line(size=0.7) + geom_point(size=1.0)
     + scale_x_log10()
     + scale_color_manual(values=pal)
     + labs(x="% of subspace dims kept (log)", y="CE (mean over 29 tasks)",
            color="", fill="", title="Shared-basis DAS: CE vs sparsity")
     + theme(figure_size=(4.3, 2.7), legend_position="right",
             legend_text=element_text(size=6)))
# drop fill from legend
from plotnine import guides
p = p + guides(fill=None)
p.save(OUT / "multitask_ce_curve.png", dpi=150)
print("Saved", OUT / "multitask_ce_curve.png")

# also a learned-only faint per-task spaghetti for the full run
rows2 = []
for t, ev in full["per_task_eval"].items():
    if ev.get("eval_learned_ce"):
        for i, pct in enumerate(sp):
            rows2.append({"pct": pct, "ce": ev["eval_learned_ce"][i], "task": t})
d2 = pd.DataFrame(rows2)
mc, _ = mean_curve(full, "eval_learned_ce")
dm = pd.DataFrame({"pct": sp, "ce": mc})
p2 = (ggplot(d2, aes("pct", "ce", group="task"))
      + geom_line(alpha=0.18, size=0.3, color="#888888")
      + geom_line(dm, aes("pct", "ce"), color="#e41a1c", size=1.0, inherit_aes=False)
      + scale_x_log10()
      + labs(x="% of subspace dims kept (log)", y="CE",
             title="Per-task CE curves (full run; red = mean)")
      + theme(figure_size=(4.3, 2.7)))
p2.save(OUT / "multitask_ce_curve_pertask.png", dpi=150)
print("Saved", OUT / "multitask_ce_curve_pertask.png")
