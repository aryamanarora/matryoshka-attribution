"""LaTeX table of the top-10 Llama Scope SAE features per arithmetic-wild task, from the
multitask MAttr+SAE run (results/arith_sae.pkl). Scores are per (layer, span, feature);
we rank (layer, feature) at the last-token span. Feature description + top/bottom logits
+ Neuronpedia link (llama3.1-8b, llamascope-res-32k) pulled from Neuronpedia (cached).

Usage: python scripts/gen_arith_sae_feature_table.py [--out paper/tabs/arith_sae_top_features.tex]
"""
import argparse, json, os, pickle, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from gen_sae_feature_table import fetch_feature, toktags, tex_escape, load_cache

MODEL = "llama3.1-8b"
SAE_SET = "{layer}-llamascope-res-32k"          # 32k width == 8x expansion (d_sae=32768)
NP_URL = "https://www.neuronpedia.org/{model}/{sae}/{idx}"
TOPK = 10
SPAN_LAST = 2                                   # span order: input, offset, last_token
TASK_ORDER = ["addition", "months", "weekdays", "hours"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results/arith_sae.pkl")
    ap.add_argument("--out", default="results/arith_sae_top_features.tex")
    ap.add_argument("--cache", default="results/neuronpedia_cache.json")
    args = ap.parse_args()

    r = pickle.load(open(args.pkl, "rb"))
    L, I = r["n_layers"], r["scores"][r["tasks"][0]].numel() // (r["n_layers"] * 3)
    cache = load_cache(args.cache)
    tasks = [t for t in TASK_ORDER if t in r["tasks"]] + [t for t in r["tasks"] if t not in TASK_ORDER]

    groups = []
    for t in tasks:
        S = r["num_spans"][t]
        v = r["scores"][t].view(L, S, I)[:, SPAN_LAST, :]      # [L, d_sae]
        flat = torch.argsort(v.flatten(), descending=True)[:TOPK]
        rows = []
        for idx in flat.tolist():
            layer, feat = idx // I, idx % I
            sae = SAE_SET.format(layer=layer)
            info = fetch_feature(MODEL, sae, feat, cache)
            url = NP_URL.format(model=MODEL, sae=sae, idx=feat)
            rows.append((layer, feat, info["desc"], info["pos"], info["neg"], url))
            print(f"{t:9s} L{layer:2d} feat {feat:6d}  {info['desc']}")
        groups.append((t, rows))

    json.dump(cache, open(args.cache, "w"), indent=0)

    lines = [
        "% Requires \\usepackage{booktabs}, \\usepackage{hyperref}, and \\toktag{} (token chip).",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{@{}r l p{0.30\\linewidth} p{0.21\\linewidth} p{0.21\\linewidth}@{}}",
        "\\toprule",
        "Layer & Feat. & Description & Top logits & Bottom logits \\\\",
    ]
    for t, rows in groups:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{tex_escape(t)}}}}} \\\\")
        for layer, feat, desc, pos, neg, url in rows:
            link = f"\\href{{{url}}}{{{feat}}}"
            lines.append(f"{layer} & {link} & {tex_escape(desc)} "
                         f"& {toktags(pos)} & {toktags(neg)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        ("\\caption{Top-10 Llama Scope SAE features per arithmetic-wild task "
         "(Llama-3.1-8B, \\texttt{llamascope-res-32k}) by sufficient-MAttr importance at "
         "the last token, ranked over all layers. Feature IDs link to Neuronpedia; "
         "descriptions are auto-interp labels; top/bottom logits are the most "
         "promoted/suppressed output tokens.}"),
        "\\label{tab:arith-sae-top-features}",
        "\\end{table}",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
