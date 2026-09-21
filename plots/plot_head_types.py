"""Where each attribution method ranks the published circuit heads of a MIB task, one heatmap
per task with a named head taxonomy (test-split rankings):

  --task ioi    IOI / GPT-2 small, the Wang et al. (2023) circuit (26 heads in 7 classes; the
                negative name movers are the copy-suppression heads of McDougall et al. 2023).
                -> paper/figs/ioi_head_types.pdf
  --task arith  Arithmetic (+ and -) / Llama-3.1-8B, the Nikankin et al. (2025) circuit heads
                (Table 2: the operator head a2.h2, the operand heads a15.h13 / a16.h21, the heads
                specific to each operator) and the MLPs the paper puts in the circuit (MLP 0 at the
                operand/operator positions; the middle/late MLPs, 16-31, at the final position,
                where its heuristic neurons live). Two row blocks, one per operator, so the
                operator-specific heads can be read against BOTH cells.
                -> paper/figs/arith_head_types.pdf

Rows are the key MIB node-level methods in test-table order; columns the circuit nodes grouped by
class. Cell = the node's rank among all nodes of that cell (157 for GPT-2, 1057 for Llama).
Ranks in the top half print as positive numbers in blue (1 = darkest); ranks in the bottom half
print as NEGATIVE numbers counted from the bottom in red (-1 = last, darkest), so "this method
actively ranks the node as harmful" reads as red at a glance rather than as a large number. White
is the median rank. A method that "finds the circuit" reads as a blue row.

Node Pruning and DBM are mask learners, not rankers: their rows show membership of the learned
circuit (check = gate open, i.e. mask logit > 0: sigmoid(logit) > 0.5 for DBM's gate, and for
Node Pruning's hard-concrete gate the sign of log_alpha) rather than a rank.

Rank = 1 + #nodes with a strictly larger score, i.e. exactly the order MIB's
evaluate_area_under_curve consumes (absolute=False). Gradient baselines are attributed on the train
split and are split-independent (the fork's importances.json); MAttr / IntInv read their test-run
importances; Node Pruning and DBM read the learned mask logits in graph_<task>_<model>.json.
Random and the no-learning control were in the first cut and dropped (requested 2026-09-20).

    uv run python plots/plot_head_types.py --task ioi
    uv run python plots/plot_head_types.py --task arith
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (aes, element_blank, element_rect, element_text, facet_grid, geom_text,
                      geom_tile, ggplot, labs, scale_color_identity, scale_fill_cmap,
                      scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning_to_attribute.deps import mib_results_dir  # noqa: E402

R = Path("results")
R_MIB = mib_results_dir()
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=5.5, rotation=90, hjust=0.5, vjust=0.5),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.015,
        panel_spacing_y=0.03,
        panel_border=element_rect(color="#000", size=0.4),
        strip_background=element_blank(),
        strip_text=element_text(size=6.2),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="right",
        legend_direction="vertical",
        legend_box_margin=0,
    )
)

# --- the circuits ----------------------------------------------------------------------------
# IOI / GPT-2 small (Wang et al. 2023). Column order follows the flow of the circuit: early heads
# that detect the duplicated name, the induction/S-inhibition path, then the heads that write the
# answer. Class labels are the strip text; "\n" keeps each strip to two short lines.
CLASSES_IOI = [
    ("Duplicate\ntoken",          ["a0.h1", "a0.h10", "a3.h0"]),
    ("Previous\ntoken",           ["a2.h2", "a4.h11"]),
    ("Induction",                 ["a5.h5", "a5.h8", "a5.h9", "a6.h9"]),
    ("S-inhibition",              ["a7.h3", "a7.h9", "a8.h6", "a8.h10"]),
    ("Name\nmover",               ["a9.h6", "a9.h9", "a10.h0"]),
    ("Backup\nname mover",        ["a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10",
                                   "a11.h2", "a11.h9"]),
    ("Negative NM /\ncopy suppr.", ["a10.h7", "a11.h10"]),
]
# Arithmetic / Llama-3.1-8B (Nikankin et al. 2025, Table 2, LxHy -> ax.hy). The shared heads are
# in both operators' circuits: a2.h2 attends to the operator, a15.h13 to the second operand,
# a16.h21 to the first, a14.h12 is unlabelled; a5.h3 / a5.h31 are +-only, a13.h21 / a13.h22 are
# --only. The MLP classes are the paper's layer ranges, not a per-layer list. (One facet per
# single head made the strip labels collide; the shared heads share a strip.)
CLASSES_ARITH = [
    ("Shared\nheads",      ["a2.h2", "a15.h13", "a16.h21", "a14.h12"]),
    ("+\nonly",            ["a5.h3", "a5.h31"]),
    ("−\nonly",       ["a13.h21", "a13.h22"]),
    ("MLP\n0",             ["m0"]),
    ("Late MLPs (16–31):\nheuristic neurons", [f"m{l}" for l in range(16, 32)]),
]
TASKS = {
    # Widths: the IOI cut fits \textwidth; the arithmetic one has 38 columns and 3-4 digit ranks,
    # so it is drawn wider (an appendix figure, scaled by \includegraphics[width=\textwidth]).
    "ioi":   dict(cells=[("ioi", "gpt2", "IOI / GPT-2")], classes=CLASSES_IOI,
                  out="ioi_head_types", width=5.5, height=2.1),
    "arith": dict(cells=[("arithmetic_addition", "llama3", "Arith. (+)\nLlama"),
                         ("arithmetic_subtraction", "llama3", "Arith. (−)\nLlama")],
                  classes=CLASSES_ARITH, out="arith_head_types", width=7.2, height=3.9),
}

# --- methods: (row label, path, layout), top-to-bottom by test-table CPR Avg -----------------
# flat   = results/<dir>/<task>_<model>_importances.json   (eval_mib.py layout; MAttr, IntInv)
# graph  = results/<dir>/graph_<task>_<model>.json          (Node Pruning / DBM mask logits; drawn
#                                                            as circuit MEMBERSHIP, logit > 0)
# nested = <fork results>/<dir>/<task-dash>_<model>/importances.json (run_attribution.py baselines)
# The 4th field locates the method's TEST-split CPR pkl (the test table's own dirs), drawn in a
# last grey column so the ranking can be read against the score it earns:
#   ("ours", dir)       results/<dir>/<task>_<model>_test.pkl                    (eval_mib.py)
#   ("mib", dir, sub)   results/<dir>/<sub>/<task-dash>_<model>_test_abs-False.pkl (run_evaluation)
METHODS = [
    ("MAttr",              "test_node_topk_uniform_lr05",                    "flat",
     ("ours", "test_node_topk_uniform_lr05")),
    ("IntInv",             "test_node_actpatch_noise",                       "flat",
     ("ours", "test_node_actpatch_noise")),
    ("Node Pruning",       "eprun_node_s0.5_ld",                             "graph",
     ("mib", "eprun_eval_s0.5_ld", "EdgePruning_patching_node")),
    ("DBM",                "eprun_node_ld_sig_lr0.3_l16.0",                  "graph",
     ("mib", "eprun_eval_ld_sig_lr0.3_l16.0", "EdgePruning_patching_node")),
    ("AttnLRP",            "attnlrp/AttnLRP_patching_node",                  "nested",
     ("mib", "attnlrp_eval", "AttnLRP_patching_node")),
    ("Expected Gradients", "napig_mc/EAP-IG-inputs-mc_patching_node",        "nested",
     ("mib", "napig_mc_test", "EAP-IG-inputs-mc_patching_node")),
    ("IG (m=10)",          "napig10/EAP-IG-inputs_patching_node",            "nested",
     ("mib", "napig10_test", "EAP-IG-inputs_patching_node")),
    ("I×G",                "ig1/EAP-IG-inputs_patching_node",                "nested",
     ("mib", "ig1_test", "EAP-IG-inputs_patching_node")),
]
MEMBER_U = 0.62      # fill for the membership rows, as a fraction of the scale's half-range
CPR_CLASS = "Test\nCPR"


def cpr(spec, task, model):
    """Test-split CPR AUC (area_under) of the method on this cell, or None if not on disk."""
    loc = spec[3]
    p = (R / loc[1] / f"{task}_{model}_test.pkl" if loc[0] == "ours"
         else R / loc[1] / loc[2] / f"{task.replace('_', '-')}_{model}_test_abs-False.pkl")
    if not p.exists():
        print(f"no test pkl for {spec[0]} on {task}/{model}: {p}")
        return None
    with open(p, "rb") as f:
        return float(pickle.load(f)["area_under"])


def signed(rank, n):
    """Signed log distance from the median rank: +log10(half/rank) in the top half (rank 1 ->
    +log10(half)), -log10(half/(n+1-rank)) in the bottom half (rank n -> -log10(half)).
    Symmetric, so one diverging scale serves both halves and the two extremes are equally dark;
    a log ratio, so a 157-node and a 1057-node cell read alike."""
    half = (n + 1) / 2
    return np.log10(half / rank) if rank <= half else -np.log10(half / (n + 1 - rank))


def label(rank, n):
    """Top half: the rank. Bottom half: minus the rank from the bottom (-1 = last)."""
    return str(rank) if rank <= (n + 1) / 2 else f"−{n + 1 - rank}"


def load(path):
    d = json.load(open(path)); nodes = d.get("nodes", d)
    return {n: v["score"] for n, v in nodes.items() if n != "logits" and "score" in v}


def scores(spec, task, model):
    _, loc, layout = spec[:3]
    if layout == "flat":
        return load(R / loc / f"{task}_{model}_importances.json")
    if layout == "graph":
        return load(R / loc / f"graph_{task}_{model}.json")
    return load(R_MIB / loc / f"{task.replace('_', '-')}_{model}" / "importances.json")


def ranks(s):
    """1 + number of nodes with a strictly larger score (ties share the better rank)."""
    v = np.array(list(s.values()))
    return {n: int((v > s[n]).sum()) + 1 for n in s}


ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=list(TASKS), default="ioi")
a = ap.parse_args()
T = TASKS[a.task]
CLASSES = list(T["classes"])
classed = {h for _, hs in CLASSES for h in hs}
# Last facet (requested 2026-09-21): the ten highest-ranked nodes OUTSIDE the taxonomy under MAttr,
# so the figure also shows what MAttr promotes that the paper did not name. With several cells the
# ten are picked by mean MAttr rank across the cells (the columns must be shared by the row
# blocks); the printed line says what they are.
mattr = METHODS[0]; assert mattr[0] == "MAttr"
mean_rank = pd.DataFrame({cell: ranks(scores(mattr, task, model))
                          for task, model, cell in T["cells"]}).mean(axis=1)
extra = [n for n in mean_rank.sort_values().index if n not in classed][:10]
CLASSES.append(("Other:\nMAttr top-10", extra))
print("MAttr top-10 outside the taxonomy (mean rank over cells):",
      ", ".join(f"{n} ({mean_rank[n]:g})" for n in extra))
CLASS_OF = {h: c for c, hs in CLASSES for h in hs}
HEADS = [h for _, hs in CLASSES for h in hs]

rows = []; L = None
for task, model, cell in T["cells"]:
    for spec in METHODS:
        sc = scores(spec, task, model); n = len(sc); rk = ranks(sc); member = spec[2] == "graph"
        Lc = np.log10((n + 1) / 2)
        assert L is None or abs(L - Lc) < 1e-9, "all cells of a task must share a node count"
        L = Lc
        for h in HEADS:
            r = rk[h]
            if member:
                inc = sc[h] > 0
                rows.append(dict(cell=cell, method=spec[0], head=h, cls=CLASS_OF[h], rank=r,
                                 u=(MEMBER_U if inc else -MEMBER_U) * L,
                                 txt="✓" if inc else "✗", txt_col="#000000"))
            else:
                u = signed(r, n)
                rows.append(dict(cell=cell, method=spec[0], head=h, cls=CLASS_OF[h], rank=r, u=u,
                                 txt=label(r, n),
                                 txt_col="#ffffff" if abs(u) > L - np.log10(12) else "#000000"))
        if member:
            print(f"{cell.replace(chr(10), ' ')} {spec[0]}: {sum(v > 0 for v in sc.values())}/{n} "
                  f"nodes in the learned circuit, {sum(sc[h] > 0 for h in HEADS)}/{len(HEADS)} "
                  f"of the published circuit nodes")
        v = cpr(spec, task, model)     # grey cell, no fill on the rank scale
        rows.append(dict(cell=cell, method=spec[0], head="CPR", cls=CPR_CLASS, rank=np.nan,
                         u=np.nan, txt="—" if v is None else f"{v:.2f}", txt_col="#000000"))
df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], [m[0] for m in METHODS][::-1])   # first method on top
df["head"] = pd.Categorical(df["head"], HEADS + ["CPR"])
df["cls"] = pd.Categorical(df["cls"], [c for c, _ in CLASSES] + [CPR_CLASS])
df["cell"] = pd.Categorical(df["cell"], [c for _, _, c in T["cells"]])

# Per-class median rank of the ranking methods, printed so the figure's reading can be quoted.
rankers = [m[0] for m in METHODS if m[2] != "graph"]
for cell in df["cell"].cat.categories:
    sub = df[(df["cell"] == cell) & df["method"].isin(rankers)]
    print("==", cell.replace("\n", " "))
    print(sub.groupby(["method", "cls"], observed=True)["rank"].median().unstack()
             .reindex(rankers).to_string())

# Legend: the decades of the top half, the median, the decades of the bottom half.
dec = int(np.ceil(L))          # 1.9 -> 1, 10; 2.72 -> 1, 10, 100
breaks = [L - i for i in range(dec)] + [0] + [-(L - i) for i in range(dec)][::-1]
labels = ([f"{10 ** i:d}" for i in range(dec)] + [f"{round(10 ** L)}"]
          + [f"−{10 ** i:d}" for i in range(dec)][::-1])
facets = (facet_grid(cols="cls", scales="free_x", space="free_x") if len(T["cells"]) == 1
          else facet_grid(rows="cell", cols="cls", scales="free_x", space="free_x"))
p = (
    ggplot(df, aes("head", "method", fill="u"))
    + geom_tile(color="#ffffff", size=0.3)
    + geom_text(aes(label="txt", color="txt_col"), size=4.4)
    + facets
    + scale_fill_cmap("RdBu", name="Rank", limits=(-L, L), breaks=breaks, labels=labels,
                      na_value="#e8e8e8")
    + scale_x_discrete(expand=(0, 0))
    + scale_y_discrete(expand=(0, 0))
    + labs(x="", y="")
    + theme(legend_key_height=44, legend_key_width=5, figure_size=(T["width"], T["height"]),
            strip_text_y=element_text(size=6.2, angle=-90))
)
# geom_text's colour is data-driven (white on dark cells); take it verbatim, no legend.
p = p + scale_color_identity()

out = OUT / f"{T['out']}.pdf"
p.save(out, verbose=False)
p.save(out.with_suffix(".png"), dpi=200, verbose=False)
print("wrote", out)
