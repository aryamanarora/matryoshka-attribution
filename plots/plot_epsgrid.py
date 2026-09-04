"""fig:optimiser-eps's four-panel row, redrawn on other substrates.

*** THIS IS NOT A SECOND IMPLEMENTATION. *** It calls plot_adamsgd_mlp_diag.fig_eps_grid, the
same function that draws figs/adamsgd_epsgrid*.pdf, with a different results tree, references and
panel list. Two hand-maintained copies of a figure this fiddly (quarter-width canvases tuned per
panel so four subfigures land at equal rendered width, per-colormap text-contrast rules,
horizontal colourbars) drift on the first edit, and the point of putting these rows on facing
pages is that they are drawn identically. Need a knob this does not expose? Add a parameter
there, do not fork the loop.

SUBSTRATES (--substrate):
  mlp_sae   MLP-output SAE latents, --sae-error frozen (results/saefrozen_epslr). 5.24M units.
  node      MIB node level: 32L x (32 heads + 1 MLP) = 1,056 units (results/epslr_node).

*** THE REFERENCES MUST MATCH THE GRID'S INTERVENTION. *** The SAE rows correlate against
frozen-mode IG and MAttr+SGD run on the same cell, NOT the results/sva_sweep runs the MLP-neuron
row uses -- those are `absorb`, and correlating a frozen grid against an absorb reference folds
an intervention change into a hyperparameter figure. `node` has no SAE reconstruction error and
so no intervention to match, which is why it reads its references straight out of
results/sva_sweep.

*** rho vmax IS PER SUBSTRATE AND THAT IS THE POINT, NOT AN INCONSISTENCY. *** Observed ranges:
node rho-vs-IG 0.33-0.74 and rho-vs-SGD 0.41-0.87, against the MLP-output SAE's 0.01-0.08 and
0.01-0.35. A scale shared across substrates would render every SAE cell as the same dark square
and throw away the only variation those panels have. Compare panels WITHIN a row; across rows,
read the colourbar.

WHY node IS THE CONTROL ROW. The eps story is that at ~2.29M mask logits Adam's update
degenerates to sign(g)*lr; at 1,056 units -- 2,170x smaller -- it should not bite. It does not:
the accuracy grid spans 0.477-0.526 (18 of 24 cells within 0.011 of the best) where the SAE and
MLP-neuron grids span ~0.3, and no cell exceeds faith-AUC 1.0, so the gap-padding corner those
rows have does not exist here either. The untuned reference (0.526) is already the grid maximum:
there is no tuning headroom at node level at all.

ONE TASK, ONE SEED (addition / llama3 / logit_diff), like the MLP-neuron row.

Run:  uv run python plots/plot_epsgrid.py --substrate node
      uv run python plots/plot_epsgrid.py --substrate mlp_sae
Out:  plots/{node,sae}_epsgrid{,_faith,_rho,_rho_sgd}.pdf  (+ .png)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_adamsgd_mlp_diag as D                      # noqa: E402

FERR = "results/sae_ferr/smoke"
SUBSTRATES = {
    "mlp_sae": dict(
        res="results/saefrozen_epslr", prefix="sae",
        refs={"rho_ig": f"{FERR}/addition_llama3_mlp_sae_span_ig_ferr.scores.pt",
              "rho_sgd": f"{FERR}/addition_llama3_mlp_sae_span_"
                         "sufficient_topk_sgd_ferr_bs1.scores.pt"},
        rho_vmax=(0.30, 0.60)),
    "node": dict(
        res="results/epslr_node", prefix="node",
        refs={"rho_ig": "results/sva_sweep/addition_llama3_node_ig.scores.pt",
              "rho_sgd": "results/sva_sweep/"
                         "addition_llama3_node_sufficient_topk_sgd_bs1.scores.pt"},
        rho_vmax=(0.80, 0.90)),
}


def specs(prefix, rho_vmax):
    """fig_eps_grid's panel tuples: (key, out name, colourbar label, cmap, vmin, vmax, center,
    canvas w, canvas h, (label, tick, cell) pt).

    Geometry copied verbatim from the MLP-neuron row's first four entries. The 1.10 / 1.055
    width split is what makes four 0.24\\textwidth subfigures print at equal size after
    bbox_inches="tight" trims each to its own content -- do not "tidy" it into one number.
    """
    return [
        ("acc_auc", f"{prefix}_epsgrid", "Accuracy AUC", "viridis", 0.0, 0.55, None,
         1.10, 1.20, (5.5, 5.0, 5.0)),
        ("faith_auc", f"{prefix}_epsgrid_faith", "Faithfulness AUC", "RdBu_r", 0.0, 2.0, 1.0,
         1.055, 1.20, (5.5, 5.0, 5.0)),
        ("rho_ig", f"{prefix}_epsgrid_rho", "Spearman $\\rho$ vs. IG", "viridis",
         0.0, rho_vmax[0], None, 1.055, 1.20, (5.5, 5.0, 5.0)),
        ("rho_sgd", f"{prefix}_epsgrid_rho_sgd", "Spearman $\\rho$ vs. SGD", "viridis",
         0.0, rho_vmax[1], None, 1.055, 1.20, (5.5, 5.0, 5.0)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="node", choices=list(SUBSTRATES))
    a = ap.parse_args()
    c = SUBSTRATES[a.substrate]
    D.fig_eps_grid(res=c["res"], refs=c["refs"],
                   specs=specs(c["prefix"], c["rho_vmax"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
