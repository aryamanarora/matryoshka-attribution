"""Generate a LaTeX table of the top Gemma Scope SAE feature per CausalGym subtask,
from the SAE-feature MAttr sweep (results/sae_node_sweep/<task>/scores.pt).

For each subtask: top feature = argmax of the learned MAttr score over the d_sae SAE
features (the trailing error-node score, if present, is excluded). Feature description
+ Neuronpedia link are pulled from the Neuronpedia API (gemma-2-2b, layer-12 res-16k
Gemma Scope), cached locally so reruns are offline-fast.

Usage: python scripts/gen_sae_feature_table.py [--sweep results/sae_node_sweep]
       [--out results/sae_top_features.tex]
"""
import argparse, glob, json, os, time, urllib.request
import torch

MODEL = "gemma-2-2b"
NP_API = "https://www.neuronpedia.org/api/feature/{model}/{sae}/{idx}"
NP_URL = "https://www.neuronpedia.org/{model}/{sae}/{idx}"


def sae_set_from_id(sae_id):
    # "layer_12/width_16k/average_l0_82" -> "12-gemmascope-res-16k"
    layer = sae_id.split("layer_")[1].split("/")[0]
    width = sae_id.split("width_")[1].split("/")[0]
    return f"{layer}-gemmascope-res-{width}"


def tex_escape(s):
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(c, c) for c in s)


def load_cache(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def fetch_description(model, sae, idx, cache):
    key = f"{model}/{sae}/{idx}"
    if key in cache:
        return cache[key]
    url = NP_API.format(model=model, sae=sae, idx=idx)
    desc = "(no description)"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(req, timeout=40) as f:
            d = json.load(f)
        exps = d.get("explanations") or []
        if exps:
            desc = exps[0].get("description", desc).strip()
    except Exception as e:
        desc = f"(fetch error: {e})"
    cache[key] = desc
    time.sleep(0.3)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sae_node_sweep")
    ap.add_argument("--out", default="results/sae_top_features.tex")
    ap.add_argument("--cache", default="results/neuronpedia_cache.json")
    args = ap.parse_args()

    cache = load_cache(args.cache)
    rows = []
    for d in sorted(glob.glob(os.path.join(args.sweep, "*/"))):
        task = os.path.basename(d.rstrip("/"))
        meta = json.load(open(os.path.join(d, "results.json")))["args"]
        sae_set = sae_set_from_id(meta["sae_id"])
        sc = torch.load(os.path.join(d, "scores.pt"))
        d_sae = int(meta.get("d_sae") or sc.numel())
        # exclude trailing error-node score (error_mode == "node" adds one extra entry)
        n_feat = sc.numel() - (1 if meta.get("error_mode") == "node" else 0)
        feat = int(torch.argmax(sc[:n_feat]))
        desc = fetch_description(MODEL, sae_set, feat, cache)
        url = NP_URL.format(model=MODEL, sae=sae_set, idx=feat)
        rows.append((task, sae_set, feat, desc, url))
        print(f"{task:28s} feat {feat:6d}  {desc}")

    json.dump(cache, open(args.cache, "w"), indent=0)

    # emit LaTeX (booktabs + hyperref only; feature id hyperlinks to Neuronpedia)
    sae_set = rows[0][1] if rows else "12-gemmascope-res-16k"
    lines = [
        "% Requires \\usepackage{booktabs} and \\usepackage{hyperref} in the preamble.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}l l p{0.55\\linewidth}@{}}",
        "\\toprule",
        "Subtask & Feature & Description \\\\",
        "\\midrule",
    ]
    for task, sset, feat, desc, url in rows:
        feat_link = f"\\href{{{url}}}{{{feat}}}"
        lines.append(f"\\texttt{{{tex_escape(task)}}} & {feat_link} & {tex_escape(desc)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        ("\\caption{Top Gemma Scope SAE feature (gemma-2-2b, "
         f"{tex_escape(sae_set)}) per CausalGym subtask by sufficient-MAttr importance. "
         "Feature IDs link to Neuronpedia; descriptions are Neuronpedia auto-interp labels.}"),
        "\\label{tab:sae-top-features}",
        "\\end{table}",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}  ({len(rows)} subtasks)")


if __name__ == "__main__":
    main()
