"""Compare DAS (learned rotation) vs DAS-identity (no rotation) on CE vs sparsity.

Shows whether learned rotation achieves lower CE at sparser masks (= more compact subspace).

Usage:
    uv run python plots/plot_das_comparison.py results/pythia1b_npi_any_das_scores.pt results/pythia1b_npi_any_das_identity_scores.pt
"""

import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_ribbon, geom_vline,
    labs, facet_wrap, scale_x_log10, scale_color_manual, scale_fill_manual,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
)

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(8, 5),
        axis_title=element_text(size=10),
        axis_text=element_text(size=8),
        legend_text=element_text(size=9),
        legend_title=element_blank(),
        panel_grid_major=element_line(size=0.5, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=9, face="plain"),
    )
)

COLORS = {"DAS (learned R)": "#e94560", "DAS (identity R)": "#1a1a2e",
           "Random (DAS)": "#e94560", "Random (identity)": "#1a1a2e"}


def load_eval(path):
    d = torch.load(path, weights_only=False, map_location="cpu")
    ev = d.get("sparsity_eval") or d.get("eval_results")
    return ev, d


def main():
    das_path = Path(sys.argv[1])
    id_path = Path(sys.argv[2])

    ev_das, d_das = load_eval(das_path)
    ev_id, d_id = load_eval(id_path)

    rows = []
    for label, ev in [("DAS (learned R)", ev_das), ("DAS (identity R)", ev_id)]:
        sp = ev["sparsities"]
        ce_l = ev["eval_learned_ce"]
        ce_r = ev["eval_random_ce"]
        ce_l_std = ev.get("eval_learned_ce_std")
        n = len(ev.get("eval_learned_ce_all", [[]])) or 1
        for i, frac in enumerate(sp):
            se = ce_l_std[i] / n**0.5 if ce_l_std else 0
            rows.append({"sparsity": frac, "CE": ce_l[i], "method": label, "type": "learned",
                         "CE_lo": ce_l[i] - 1.96*se, "CE_hi": ce_l[i] + 1.96*se})
            rows.append({"sparsity": frac, "CE": ce_r[i], "method": label, "type": "random",
                         "CE_lo": ce_r[i], "CE_hi": ce_r[i]})

    df = pd.DataFrame(rows)

    df_learned = df[df["type"] == "learned"].copy()
    df_random = df[df["type"] == "random"].copy()

    has_ci = df_learned["CE_lo"].ne(df_learned["CE"]).any()

    p = (
        ggplot(df_learned, aes(x="sparsity", y="CE", color="method"))
        + geom_line(size=1)
        + geom_point(size=2)
        + (geom_ribbon(aes(ymin="CE_lo", ymax="CE_hi", fill="method"), alpha=0.15, color="none") if has_ci else geom_line(size=0))
        + geom_line(data=df_random, linetype="dashed", size=0.5, alpha=0.5)
        + scale_color_manual(values=["#e94560", "#1a1a2e"])
        + scale_fill_manual(values=["#e94560", "#1a1a2e"])
        + scale_x_log10()
        + labs(x="Fraction of dims kept", y="Cross-entropy loss",
               title="DAS: Learned rotation vs identity — CE at varying sparsity")
    )

    out = das_path.parent / "das_comparison_ce.png"
    p.save(out, dpi=150)
    print(f"Saved {out}")
    p.save(out.with_suffix(".pdf"), dpi=300)

    # Print summary: area under CE-sparsity curve
    for label, ev in [("DAS (learned R)", ev_das), ("DAS (identity R)", ev_id)]:
        sp = ev["sparsities"]
        ce = ev["eval_learned_ce"]
        ce_r = ev["eval_random_ce"]
        auc = np.trapezoid(ce, sp)
        auc_r = np.trapezoid(ce_r, sp)
        print(f"{label}: AUC(CE)={auc:.4f}  AUC(random)={auc_r:.4f}  ratio={auc/auc_r:.4f}")

    # CE at specific sparsity levels
    print("\nCE at key sparsities:")
    print(f"{'Sparsity':<12} {'DAS (R)':<12} {'Identity':<12} {'Delta':<12} {'R random':<12} {'Id random':<12}")
    for i, frac in enumerate(ev_das["sparsities"]):
        ce_r = ev_das["eval_learned_ce"][i]
        ce_id = ev_id["eval_learned_ce"][i]
        rr = ev_das["eval_random_ce"][i]
        ir = ev_id["eval_random_ce"][i]
        delta = ce_id - ce_r
        print(f"{frac:<12.3f} {ce_r:<12.4f} {ce_id:<12.4f} {delta:<+12.4f} {rr:<12.4f} {ir:<12.4f}")

    # Per-example significance test if available
    das_all = ev_das.get("eval_learned_ce_all")
    id_all = ev_id.get("eval_learned_ce_all")
    if das_all and id_all:
        from scipy.stats import wilcoxon
        das_arr = np.array(das_all)
        id_arr = np.array(id_all)
        print("\nWilcoxon signed-rank test (per-example CE, DAS < identity):")
        for i, frac in enumerate(ev_das["sparsities"]):
            d = das_arr[:, i]
            ident = id_arr[:, i]
            if np.allclose(d, ident):
                print(f"  keep={frac:.1%}: identical")
                continue
            stat, p = wilcoxon(d, ident, alternative="less")
            print(f"  keep={frac:.1%}: DAS={d.mean():.4f} Id={ident.mean():.4f} "
                  f"delta={d.mean()-ident.mean():+.4f} p={p:.4g} {'*' if p < 0.05 else ''}")


if __name__ == "__main__":
    main()
