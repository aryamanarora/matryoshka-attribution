"""Assemble every result of the replication into the markdown tables we actually quote.

    uv run python scripts/vw/vw_summary.py --run results/vw/base >> results/vw/interference_vw.md

Reads whatever the pipeline has written (`log.json`, `report.json`, `prune.json`,
`whatkept.json`) and emits five tables, each with the note's own published figure alongside
ours where the note has one. Nothing is hardcoded from the note except its published
numbers, which are collected in `NOTE` here and in `vw_report.NOTE` so they sit in one
place and can be checked against the source.
"""

import argparse
import json
from pathlib import Path

# Published numbers from Turner, Wu & Batson (2026). The pruning figures are over ALL SIX
# weight families; ours are the Tokens->Logits family alone, so they are a reference point
# rather than a matched comparison, and every table below says so.
NOTE = {"train_loss": 3.33, "test_loss": 3.38, "params": 2_883_584,
        "prune": {0.55: 0.000146, 0.30: 0.0107, 0.15: 0.0702},
        "frac_helpful": 0.476, "frac_dead": 0.127,
        "median_absW_over_max": 1 / 3, "median_fisher_over_max": 1e-4,
        "density_90pct_mass": 0.0243, "density_99pct_mass": 0.136}

PRETTY = {
    "weight_abs": "Virtual weight magnitude", "weight": "Virtual weight (signed)",
    "era": "Expected attribution", "fisher": "Fisher effectiveness",
    "helpfulness": "**Helpfulness (oracle)**", "random_s0": "Random",
}


def name(k):
    if k in PRETTY:
        return PRETTY[k]
    if k.startswith("mattr_adam"):
        eps = k.split("eps")[-1].replace("_row", "").replace("_col", "").replace("_long", "")
        lr = k.split("_lr")[-1].split("_eps")[0]
        g = " row" if "_row" in k else (" col" if "_col" in k else "")
        return f"**MAttr (Adam, lr {lr}, eps {eps}{g}{', long' if '_long' in k else ''})**"
    if k.startswith("mattr_sgd"):
        return f"MAttr (SGD, lr {k.split('_lr')[-1].split('_eps')[0]})"
    if k.startswith("ig"):
        return "Expected Gradients" + (" (long)" if "long" in k else "")
    if k.startswith("ixg"):
        return "I x G"
    return k


def load(run, f):
    p = run / f
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--prune", default="prune.json")
    args = ap.parse_args()
    log, rep = load(args.run, "log.json"), load(args.run, "report.json")
    pr, wk = load(args.run, args.prune), load(args.run, "whatkept.json")
    out = []

    if log:
        out += ["### The model", "",
                "| | note | ours |", "|---|---|---|",
                f"| parameters | ≈2.9M | {log['params'][0]:,} |",
                f"| excl. embed/unembed | ≈0.79M | {log['params'][1]:,} |",
                f"| final train loss | {NOTE['train_loss']} | {log['train_loss_tail20']:.3f} |",
                f"| held-out loss | {NOTE['test_loss']} | {log['val_loss_full']:.3f} |",
                f"| learning rate | *not stated* | {log['args']['lr']} "
                f"(decay {log['args']['decay_frac']}, warmup {log['args']['warmup']}) |", ""]

    if rep:
        d = rep["dist"]
        out += ["### The weight population (note: all six families; ours: Tokens→Logits only)",
                "", "| | note | ours |", "|---|---|---|",
                f"| weights | 331,350,016 | {d['n_weights']:,} |",
                f"| median \\|W\\| / max \\|W\\| | ~1/3 | {d['median_absW_over_max']:.3f} |",
                f"| median Fisher / max Fisher | ~1e-4 | {d['median_fisher_over_max']:.2e} |",
                f"| Fisher spread (orders of magnitude) | ~10 | {d['fisher_log10_range']:.1f} |",
                f"| fraction with positive helpfulness | 47.6% | {100*d['frac_helpful']:.1f}% |",
                f"| dead (source never active) | 12.7% | {100*d['frac_dead']:.1f}% |",
                f"| significantly helpful / harmful | — | {100*d['frac_sig_helpful']:.1f}% / "
                f"{100*d['frac_sig_harmful']:.1f}% |",
                f"| density for 90% / 99% of helpfulness mass | 2.43% / 13.6% | "
                f"{100*d['density_for_90pct_mass']:.2f}% / {100*d['density_for_99pct_mass']:.2f}% |",
                f"| most-effective helpful ÷ most-effective harmful | \"an order of magnitude "
                f"or more\" | {d.get('helpful_over_harmful_fisher', float('nan')):.1f}x |", ""]

    if pr:
        ds = [0.55, 0.30, 0.15, 0.05, 0.01, 0.001]
        cols = [d for d in ds if d in pr["densities"]]
        out += [f"### Δ loss vs density, held-out ({pr['seqs']} sequences of shard 10)", "",
                f"Full model {pr['full_loss']:.4f}; with the whole Tokens→Logits family removed "
                f"{pr['empty_loss']:.4f} (Δ {pr['empty_loss']-pr['full_loss']:+.4f}).", "",
                "| ranking | " + " | ".join(f"d={d}" for d in cols) + " |",
                "|---|" + "---|" * len(cols)]
        rows = sorted(pr["curves"], key=lambda k: pr["curves"][k][pr["densities"].index(0.05)]
                      if 0.05 in pr["densities"] else 0)
        for k in rows:
            v = [pr["curves"][k][pr["densities"].index(d)] - pr["full_loss"] for d in cols]
            out.append(f"| {name(k)} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
        out += ["| *note, all six families* | " + " | ".join(
            f"{NOTE['prune'].get(d, ''):.4f}" if d in NOTE["prune"] else "—" for d in cols) + " |", ""]

    if rep and rep.get("agreement"):
        out += ["### Agreement with the helpfulness oracle (all 21M weights)", "",
                f"Base rate of significantly-helpful weights: {rep['base_rate_sig_helpful']:.4f}.",
                "", "| ranking | ρ vs helpfulness | ρ vs Fisher | precision@n | harmful rate |",
                "|---|---|---|---|---|"]
        for k, a in sorted(rep["agreement"].items(),
                           key=lambda kv: -kv[1]["spearman_vs_helpfulness"]):
            pk = next(x for x in a if x.startswith("precision@"))
            hk = next(x for x in a if x.startswith("harmful_rate@"))
            out.append(f"| {name(k)} | {a['spearman_vs_helpfulness']:+.3f} | "
                       f"{a['spearman_vs_fisher']:+.3f} | {a[pk]:.3f} | {a[hk]:.4f} |")
        out.append("")

    if wk and rep:
        d, tot = rep["dist"], wk["none"] - wk["full"]
        out += ["### How badly marginal helpfulness overcounts", "",
                f"Summed over all {d['n_weights']:,} weights, POSITIVE mean helpfulness comes "
                f"to **{d['sum_pos_helpfulness']:.1f} nats/token** (negative: "
                f"{d['sum_neg_helpfulness']:.3f}). Removing the entire Tokens→Logits family at "
                f"once costs **{tot:.3f} nats/token**. Marginal ablation therefore overstates "
                f"the family's worth by **{d['sum_pos_helpfulness']/tot:.0f}x**.", "",
                "That ratio is the note's footnoted caveat — \"up to nonlinear effects in "
                "removing multiple weights at once\" — made quantitative. Helpfulness is a "
                "faithful measure of what one weight does *given all the others*; it is not a "
                "budget that can be summed, and any argument that reasons from per-weight "
                "helpfulness to how many weights a sparse model needs is crossing that gap.", ""]

    if wk:
        out += ["### What each kept set IS", "",
                f"Full {wk['full']:.4f} · no direct path {wk['none']:.4f} · no direct path plus "
                f"its MEAN effect {wk['mean_all']:.4f} — i.e. the family's average logit "
                f"contribution alone, a fixed unigram-like vector, recovers "
                f"{100*(wk['none']-wk['mean_all'])/(wk['none']-wk['full']):.1f}% of what the "
                f"whole family is worth.", ""]
        if "none_refit" in wk:
            out += [f"A free per-target bias is worth {wk['none']-wk['none_refit']:+.4f} nats to "
                    f"the no-direct-path model and {wk['full']-wk['full_refit']:+.4f} to the "
                    f"full model.", ""]
        dens = sorted({r["density"] for r in wk["rows"].values()}, reverse=True)
        out += ["| ranking | density | loss | mean-effect only | median \\|W\\| ÷ pop | "
                "% positive | % position rows | ρ(unigram) | bias worth |",
                "|---|---|---|---|---|---|---|---|"]
        for key, r in wk["rows"].items():
            k = key.rsplit("@", 1)[0]
            out.append(f"| {name(k)} | {r['density']} | {r['loss']:.4f} | "
                       f"{r['loss_meanonly']:.4f} | {r['median_absW_kept_over_pop']:.2f} | "
                       f"{100*r['frac_positive']:.0f}% | {100*r['frac_position_rows']:.1f}% "
                       f"(base {100*r['frac_position_rows_base']:.0f}%) | "
                       f"{r['rho_unigram']:+.2f} | "
                       f"{r['bias_worth']:+.4f} |" if "bias_worth" in r else
                       f"| {name(k)} | {r['density']} | {r['loss']:.4f} | "
                       f"{r['loss_meanonly']:.4f} | {r['median_absW_kept_over_pop']:.2f} | "
                       f"{100*r['frac_positive']:.0f}% | {100*r['frac_position_rows']:.1f}% "
                       f"(base {100*r['frac_position_rows_base']:.0f}%) | "
                       f"{r['rho_unigram']:+.2f} | — |")
        out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
