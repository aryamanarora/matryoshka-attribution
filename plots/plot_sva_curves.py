"""Full-page appendix: the IIA (or faithfulness) curve vs sparsity, per granularity x task.

WHAT THE HEADLINE FIGURES COMPRESS. plots/plot_accauc_vs_faithauc.py reduces each run to one
number -- the log-weighted AUC of the curve drawn here -- and then averages over task-groups. Two
methods with the same AUC can get there completely differently: one recovering the behaviour at
k=10 and plateauing, another climbing slowly and only catching up at k=10^5. That difference is
the whole practical content of "how big is the circuit", and it is invisible in the scatter. This
figure is the uncompressed version, and it is the reason it has to be full page.

LAYOUT is task (rows) x granularity (columns), which is "granularity x task" read the other way
up. Deliberately not the transpose: at 5.5in a 4-column grid gives ~1.3in-wide panels, where a
24-point curve is legible, while 8 columns gives 0.69in and the curves turn into scribble. The
comparison the columns carry -- the SAME task attributed over four different variable sets -- is
also the one worth putting side by side.

X IS THE FRACTION OF UNITS KEPT, not raw k. The granularities differ by four orders of magnitude
in size (node ~1k, MLP 2.75M, SAE 6.29M), so raw k would give every column its own axis and make
the columns incomparable -- which is precisely the comparison the layout is for. Fraction keeps
one axis for all of them. Note this is the x the AUC is NOT computed on: eval_sva integrates over
log k, so a curve's area here is not its reported AUC. Read shape, not area.

Y. IIA is a bounded accuracy, so it shares one axis everywhere and the panels are directly
comparable. Faithfulness is unbounded and its scale is set by the (task, granularity) pair --
the SAE column reaches ~1 where the MLP column on the same task reaches ~10 -- so --metric faith
gives every panel its OWN y. Anything coarser (shared, or per-row) flattens the SAE column to a
line at zero. Faith panels therefore cannot be compared by height at all; read shape only. Neither metric is comparable across ABLATION settings,
which is why this figure draws the patched sweep only.

THE RANDOM FLOOR IS DRAWN IN EVERY PANEL and is not decoration: at node/MLP it sits near 0.02,
but the SAE columns and the arithmetic tasks are where methods get close to it, and a curve that
never separates from grey is the single most useful thing a reader can see here.

MISSING SERIES ARE EXPECTED, not a bug. The SAE columns carry only the methods that have been
run there (IG, I x G, both MAttr arms, Random); AttnLRP, Node Pruning, DBM and Stepless IG exist
at node/MLP only. main() prints a per-column inventory so a gap is always attributable.

Run:  uv run python plots/plot_sva_curves.py                 -> plots/sva_curves_iia.pdf
      uv run python plots/plot_sva_curves.py --metric faith  -> plots/sva_curves_faith.pdf
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                   # noqa: E402
import plot_accauc_vs_faithauc as V                   # parse_method, on_model, METHODS  # noqa: E402

RES = "results/sva_sweep"          # patched, -input: one setting, see the docstring
LOSS = "logit_diff"
TASKS = V.SVA + V.ARITH            # the 8 every granularity covers; ARC-E/IOI are node-only
# SAE (resid) DROPPED 2026-09-03 (requested), matching plot_accauc_vs_faithauc's default cut.
#
# (substrate, results tree, label). A COLUMN IS A (BASIS, TREE) PAIR, NOT A BASIS -- `node`
# appears twice, once per tree, so keying columns on the substrate alone would collapse them.
#   sva_sweep        the -input runs, i.e. the substrate proper
#   sva_sweep_input  the same substrate PLUS the input-embedding node as an extra scored unit,
#                    which competes for the top-k budget and so moves every learned score. For
#                    the GRADIENT methods the non-input scores are bit-identical to the -input
#                    run (verified on node / resid_sae_span / mlp_sae_span), so those two columns
#                    differ only by the one extra unit; for MAttr they differ throughout.
#   sva_sweep_ferr   --sae-error frozen. results/sva_sweep's SAE runs use the legacy `absorb`
#                    intervention, under which keeping ONE reconstruction-error node restores its
#                    entire (layer, position) site regardless of the latents. See
#                    plot_accauc_vs_faithauc.SUBSTRATE_RES for the full note; the tree is
#                    imported from there so the two figures cannot draw different runs under the
#                    same column name.
INPUT_RES = "results/sva_sweep_input"
GRAN = [("node", RES, "Node"),
        ("node", INPUT_RES, "Node, $+$input"),
        ("mlp", RES, "MLP neuron"),
        ("mlp+attn_head", RES, "MLP+Attn"),
        ("mlp_sae_span", V.SUBSTRATE_RES["mlp_sae_span"], "SAE (MLP out)")]
# THE SAE COLUMN READS A DIFFERENT TREE. Imported from plot_accauc_vs_faithauc rather than
# restated, so the two figures cannot end up drawing different runs under the same column name.
# results/sva_sweep's SAE runs use the legacy `absorb` error intervention, under which keeping ONE
# reconstruction-error node restores its entire (layer, position) site regardless of the latents;
# results/sva_sweep_ferr is the same tasks and methods under --sae-error frozen, where the error
# node is an ordinary scored unit. See that module's SUBSTRATE_RES for the full note.
#
# A substrate with an override reads ONLY from it, never falling back to RES, so a column can
# never mix two interventions; a file for that substrate sitting in the other tree is skipped.
# Draw order = legend order; ours last so they sit on top where curves overlap.
# AttnLRP and MAttr (log) -- the default-eps Adam arm -- dropped 2026-08-29 (requested). Both
# still exist on disk and in the registry; they are out of THIS figure only. Dropping the
# default-eps arm also removes the one series whose faith curve spikes to ~10 mid-sparsity
# (the gap-padding signature), so the remaining panels no longer need a y range set by it.
ORDER = ["Random", "IxG", "IG", "mc_ig", "eprun-s090", "sig_lr0.3_l16.0",
         "stopk-log-eps1e-2", "softsgd-log"]
METRIC = {"iia": ("acc_base", "IIA (base preferred over source)", True),
          "faith": ("faithfulness", "Faithfulness", False)}

FIG_W = 5.5
FS_TITLE, FS_TICK, FS_LAB, FS_LEG = 6.5, 5, 6.5, 5.5


def curves(metric_key):
    """{(gran, task, method key): (fraction-kept, y)} for the patched logit-diff runs."""
    out = {}
    for res in sorted({g[1] for g in GRAN}):
        for f in glob.glob(f"{res}/*.json"):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if d.get("loss") != LOSS or not V.on_model(d):
                continue
            if (d["nodes"], res) not in {(g[0], g[1]) for g in GRAN}:
                continue        # this tree holds no column for that substrate
            m = V.parse_method(os.path.basename(f), d)
            if m is None or m not in V.METHODS:
                continue
            y = (d.get("iso_metrics", {}) or {}).get(metric_key) if metric_key == "acc_base" \
                else d.get(metric_key)
            if not y or "n_nodes" not in d or not d.get("total"):
                continue
            x = np.asarray(d["n_nodes"], float) / d["total"]
            # Random is 3 seeds under one key; average them so the floor is one line, not three.
            k = ((d["nodes"], res), d["task"], m)
            prev = out.get(k)
            out[k] = (x, np.asarray(y, float)) if prev is None else \
                (x, (prev[1] + np.asarray(y, float)) / 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="iia", choices=list(METRIC))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    key, ylab, shared_y = METRIC[a.metric]
    out = a.out or f"plots/sva_curves_{a.metric}.pdf"

    C = curves(key)
    # A granularity needs curves on at least half the tasks to earn a column. Without this the
    # MLP-output SAE (2 runs on nounpp, the smoke) adds a column that is empty in 7 of 8 rows,
    # which reads as "that basis failed everywhere" rather than "it has barely been run".
    gran = [((g, res), lab) for g, res, lab in GRAN
            if len({k[1] for k in C if k[0] == (g, res)}) >= len(TASKS) / 2]
    if not gran:
        raise SystemExit(f"no curves found in {RES}")

    plt.rcParams.update(P.RC)
    fig_h = 0.78 * len(TASKS) + 1.0
    fig, axes = plt.subplots(len(TASKS), len(gran), figsize=(FIG_W, fig_h),
                             sharex=True, sharey=shared_y, squeeze=False)

    drawn = set()
    for r, task in enumerate(TASKS):
        for c, (g, glab) in enumerate(gran):
            ax = axes[r][c]
            for m in ORDER:
                v = C.get((g, task, m))
                if v is None:
                    continue
                x, y = v
                drawn.add((g, m))
                is_rand = m == "Random"
                ax.plot(x, y, lw=1.1 if is_rand else 0.8, color=V.METHODS[m][1],
                        ls=(0, (2, 1.6)) if is_rand else "solid",
                        zorder=1 if is_rand else 2)
            ax.set_xscale("log")
            ax.set_xlim(1e-6, 1.4)
            P.furnish(ax)
            ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1.5)
            if r == 0:
                ax.set_title(glab, fontsize=FS_TITLE, pad=3)
            if c == 0:
                ax.set_ylabel(task.replace("_", "\n"), fontsize=FS_TITLE, rotation=0,
                              ha="right", va="center", labelpad=10)
        if not shared_y:
            # Faithfulness: y is INDEPENDENT PER PANEL, not per row. Its scale is set by the
            # (task, granularity) pair together, not by the task alone -- the SAE column reaches
            # ~1 where the MLP column on the same task reaches ~10, so a per-row limit squashes
            # every SAE panel to a flat line near zero and hides its shape entirely. The cost is
            # that panels cannot be compared by height, which is already true across rows and is
            # what the docstring says to do anyway: read shape, not level.
            for c, (g, _) in enumerate(gran):
                ys = [C[(g, task, m)][1] for m in ORDER if (g, task, m) in C]
                if not ys:
                    continue
                lo, hi = min(v.min() for v in ys), max(v.max() for v in ys)
                pad = 0.08 * (hi - lo or 1)
                axes[r][c].set_ylim(lo - pad, hi + pad)
    if shared_y:
        for row in axes:
            row[0].set_ylim(0, 1)

    fig.supxlabel("fraction of units kept", fontsize=FS_LAB, y=0.012)
    fig.supylabel(ylab, fontsize=FS_LAB, x=0.005)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.45, rect=(0.012, 0.022, 1, 0.955))
    handles = [Line2D([], [], color=V.METHODS[m][1], lw=1.1 if m == "Random" else 0.9,
                      ls=(0, (2, 1.6)) if m == "Random" else "solid",
                      label=V.METHODS[m][0])
               for m in ORDER if any(mm == m for _, mm in drawn)]
    fig.legend(handles=handles, fontsize=FS_LEG, ncol=min(5, len(handles)),
               loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               handlelength=2.0, handletextpad=0.5, columnspacing=1.4)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)

    print("\nseries per granularity (a gap is coverage, not a bug):")
    for g, glab in gran:
        have = [V.METHODS[m][0] for m in ORDER if (g, m) in drawn]
        print(f"  {glab:<14} {len(have)}: {', '.join(have)}")


if __name__ == "__main__":
    sys.exit(main())
