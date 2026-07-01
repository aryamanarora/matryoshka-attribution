"""Faceted sufficiency plot: x = k-budget (log), y = accuracy. SAE vs DAS-64 per task (L12)."""
import json, glob, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAE_DIR = "results/sae_node_sweep"
DAS_DIR = "results/das64_sweep"
tasks = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{SAE_DIR}/*/results.json"))

def load(d, task):
    p = f"{d}/{task}/results.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("curve")

n = len(tasks)
ncol = 5
nrow = math.ceil(n / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
for i, task in enumerate(tasks):
    ax = axes[i // ncol][i % ncol]
    sae = load(SAE_DIR, task)
    das = load(DAS_DIR, task)
    if sae:
        ax.plot(sae["k"], sae["learned_acc"], "-o", ms=3, color="C0", label="SAE (node)")
        if "random_acc" in sae:
            ax.plot(sae["k"], sae["random_acc"], ":", color="C0", alpha=0.5, label="SAE random")
    if das:
        ax.plot(das["k"], das["learned_acc"], "-s", ms=3, color="C1", label="DAS-64")
        if "random_acc" in das:
            ax.plot(das["k"], das["random_acc"], ":", color="C1", alpha=0.5, label="DAS random")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.9, color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.set_title(task, fontsize=8)
    ax.tick_params(labelsize=7)
    if i % ncol == 0:
        ax.set_ylabel("accuracy", fontsize=8)
    if i // ncol == nrow - 1:
        ax.set_xlabel("k-budget (log)", fontsize=8)
for j in range(n, nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
# one shared legend
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower right", fontsize=9, ncol=4, bbox_to_anchor=(0.98, 0.0))
fig.suptitle("Sufficiency: accuracy vs k-budget, layer 12 (uniform-k training) — SAE features vs DAS-64", fontsize=12)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
out = "results/sufficiency_plot.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out, f"({n} tasks)")
