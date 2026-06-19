"""LaTeX table of the top-10 MLP neurons per arithmetic-wild task (results/arith_mlp.pkl,
sufficient MAttr), with Transluce positive- and negative-activation descriptions.

Scores are per (layer, span, neuron); we rank (layer, neuron) at the last-token span.
Descriptions come from the Transluce neuron-description service (Llama-3.1-8B-Instruct;
note we attribute on the *base* model, so labels are approximate). Cached locally.

Usage: python scripts/gen_arith_mlp_neuron_table.py [--out paper/tabs/arith_mlp_neuron_table.tex]
"""
import argparse, json, os, pickle, sys, time, urllib.request, urllib.parse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from gen_sae_feature_table import tex_escape, load_cache

API = "https://transluce--neuron-data-server-fastapi-app.modal.run/read_specific_file"
NEURON_URL = "https://neurons.transluce.org/{layer}/{neuron}/+"
TOPK = 10
SPAN_LAST = 2
TASK_ORDER = ["addition", "months", "weekdays", "hours"]


def ascii_clean(s):
    return "".join(c for c in s if ord(c) < 128)


def fetch_transluce(layer, neuron, sign, cache):
    """Top explanation for a (layer, neuron, sign); sign='+' = positive activation,
    '-' = negative activation. (The API keys the description file on '+'/'-'; any other
    value returns the negative/default file.)"""
    key = f"{layer}/{neuron}/{sign}"
    if key in cache:
        return cache[key]
    # URL-encode params: a literal '+' in a query string means space, so sign='+'
    # must be sent as %2B to select the positive-activation description file.
    qs = urllib.parse.urlencode({"layer": layer, "neuron": neuron, "sign": sign})
    url = f"{API}?{qs}"
    desc = "(none)"
    try:
        with urllib.request.urlopen(url, timeout=40) as f:
            d = json.load(f)
        exps = d.get("explanation_summary") or []
        if exps:
            desc = exps[0][0].strip()
    except Exception as e:
        desc = f"(fetch error: {e})"
    cache[key] = desc
    time.sleep(0.25)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results/arith_mlp.pkl")
    ap.add_argument("--out", default="results/arith_mlp_neuron_table.tex")
    ap.add_argument("--cache", default="results/transluce_cache.json")
    args = ap.parse_args()

    r = pickle.load(open(args.pkl, "rb"))
    L, I = r["n_layers"], r["intermediate_size"]
    cache = load_cache(args.cache)
    tasks = [t for t in TASK_ORDER if t in r["tasks"]] + [t for t in r["tasks"] if t not in TASK_ORDER]

    groups = []
    for t in tasks:
        S = r["num_spans"][t]
        v = r["scores"][t].view(L, S, I)[:, SPAN_LAST, :]      # [L, intermediate]
        flat = torch.argsort(v.flatten(), descending=True)[:TOPK]
        rows = []
        for idx in flat.tolist():
            layer, neuron = idx // I, idx % I
            pos = fetch_transluce(layer, neuron, "+", cache)
            neg = fetch_transluce(layer, neuron, "-", cache)
            url = NEURON_URL.format(layer=layer, neuron=neuron)
            rows.append((layer, neuron, pos, neg, url))
            print(f"{t:9s} L{layer:2d} n{neuron:6d}  +{pos[:60]}")
        groups.append((t, rows))

    json.dump(cache, open(args.cache, "w"), indent=0)

    header = "Layer & Neuron & Positive-activation description & Negative-activation description \\\\"
    caption = ("Top-10 MLP neurons per arithmetic-wild task (Llama-3.1-8B) by "
               "sufficient-MAttr importance at the last token, ranked over all layers. "
               "Neuron IDs link to Transluce; positive/negative descriptions are Transluce "
               "auto-generated labels for the neuron's positive/negative activation "
               "(Llama-3.1-8B-Instruct).")
    lines = [
        "% Requires \\usepackage{booktabs}, \\usepackage{hyperref}, \\usepackage{longtable}.",
        "\\begingroup\\scriptsize",
        "\\begin{longtable}{@{}r l p{0.36\\linewidth} p{0.36\\linewidth}@{}}",
        f"\\caption{{{caption}}}\\label{{tab:arith-mlp-neurons}}\\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{4}{c}{{\\tablename\\ \\thetable{} -- continued}} \\\\",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        "\\midrule \\multicolumn{4}{r}{\\textit{continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for t, rows in groups:
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{{tex_escape(t)}}}}} \\\\")
        for layer, neuron, pos, neg, url in rows:
            link = f"\\href{{{url}}}{{{neuron}}}"
            lines.append(f"{layer} & {link} & {tex_escape(ascii_clean(pos))} "
                         f"& {tex_escape(ascii_clean(neg))} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
