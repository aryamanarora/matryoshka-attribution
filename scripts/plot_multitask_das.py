"""Plots for multitask shared-rotation DAS: cross-task overlap matrix, loss curves,
per-task CE@5%. Reads results/pythia1b_multitask_das{,_lean}.pkl. Run on sc."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, geom_text, geom_line, geom_point, geom_col, labs,
    theme_bw, theme_set, theme, element_text, element_blank, coord_flip,
    scale_fill_gradient, scale_color_manual, scale_fill_manual, position_dodge,
)

R = Path("results")
OUT = Path("paper/figs")
OUT.mkdir(parents=True, exist_ok=True)
theme_set(theme_bw(base_size=8) + theme(text=element_text(family="Inter")))


def short(t):
    return t.replace("syntaxgym/", "")


full = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
lean = pickle.load(open(R / "pythia1b_multitask_das_lean.pkl", "rb"))

# 1) Cross-task overlap matrix heatmap
tasks = [short(t) for t in full["overlap_tasks"]]
M = np.array(full["overlap_matrix"])
rows = [{"a": tasks[i], "b": tasks[j], "j": M[i, j]}
        for i in range(len(tasks)) for j in range(len(tasks))]
df = pd.DataFrame(rows)
df["a"] = pd.Categorical(df["a"], categories=tasks, ordered=True)
df["b"] = pd.Categorical(df["b"], categories=tasks[::-1], ordered=True)
p = (ggplot(df, aes("a", "b", fill="j")) + geom_tile()
     + scale_fill_gradient(low="#ffffff", high="#08306b", limits=[0, 1])
     + labs(x="", y="", fill="Jaccard\n(top-5% dims)",
            title="Cross-task feature overlap (shared DAS basis, 29 tasks)")
     + theme(figure_size=(8, 7), axis_text_x=element_text(rotation=90, size=5, ha="center"),
             axis_text_y=element_text(size=5), panel_grid=element_blank()))
p.save(OUT / "multitask_overlap.png", dpi=150)
print("Saved", OUT / "multitask_overlap.png")

# 2) Loss curves full vs lean
def loss_df(d, tag):
    return pd.DataFrame([{"step": s, "loss": l, "run": tag} for s, l in d["loss_log"]])
ld = pd.concat([loss_df(full, "full (3000)"), loss_df(lean, "lean (1500)")])
p = (ggplot(ld, aes("step", "loss", color="run")) + geom_line(size=0.6)
     + scale_color_manual(values={"full (3000)": "#e41a1c", "lean (1500)": "#377eb8"})
     + labs(x="mixed step", y="mean train loss (CE)", color="",
            title="Multitask DAS training loss")
     + theme(figure_size=(4, 2.4), legend_position="top"))
p.save(OUT / "multitask_loss.png", dpi=150)
print("Saved", OUT / "multitask_loss.png")

# 3) Per-task CE@5% (full vs lean), sorted by full
def ce5(d):
    return {short(t): ev["eval_learned_ce"][4] for t, ev in d["per_task_eval"].items()
            if ev.get("eval_learned_ce")}
cf, cl = ce5(full), ce5(lean)
order = sorted(cf, key=lambda t: cf[t])
rows = []
for t in order:
    rows.append({"task": t, "ce": cf[t], "run": "full (3000)"})
    if t in cl:
        rows.append({"task": t, "ce": cl[t], "run": "lean (1500)"})
cd = pd.DataFrame(rows)
cd["task"] = pd.Categorical(cd["task"], categories=order, ordered=True)
p = (ggplot(cd, aes("task", "ce", fill="run"))
     + geom_col(position=position_dodge(width=0.7), width=0.65)
     + scale_fill_manual(values={"full (3000)": "#e41a1c", "lean (1500)": "#377eb8"})
     + coord_flip()
     + labs(x="", y="CE @ 5% sparsity (lower = shared basis serves task)", fill="",
            title="Per-task circuit quality under shared basis")
     + theme(figure_size=(5, 6), axis_text_y=element_text(size=5), legend_position="top"))
p.save(OUT / "multitask_ce.png", dpi=150)
print("Saved", OUT / "multitask_ce.png")
