"""Aggregate the per-model outputs of run_model.py into the paper's analyses.

  Fig 2   accuracy per format per model
  Sec 3.2 top-1% Jaccard overlap numeric-vs-verbal circuits; across-model Pearson r with accuracy
  Fig 3   item-level point-biserial r between numeric-circuit loading and correctness
  App C   patch-restored fraction (numeric circuit vs random matched units) vs correctness
  App E   circuits from correct-only / incorrect-only numeric items
  App F   reciprocity: any format's circuit predicting any other format's accuracy
  Fig 4   (models with probes_<fmt>.json) linear probability model + LMG decomposition

Every analysis is reported for the paper's method (attribution patching, ``ap``) and for MAttr
(``mattr``) side by side.  Writes results/arith_formats/summary.json and summary.md.
"""

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FORMATS  # noqa: E402

VERBAL = ("english", "spanish", "italian")
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/arith_formats")
METHODS = ("mattr", "ap")


def pbr(x, y):
    """Point-biserial r (== Pearson with a binary y) and p; nan if degenerate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or y.std() == 0 or x.std() == 0:
        return float("nan"), float("nan")
    r, p = stats.pointbiserialr(y, x)
    return float(r), float(p)


def fmt_r(r, p):
    if np.isnan(r):
        return "  n/a  "
    star = "*" if p < 0.05 else " "
    return f"{r:+.2f}{star}"


def load_model(d):
    if not (d / "circuits.json").exists():
        return None
    M = {"name": d.name, "circuits": json.load(open(d / "circuits.json")), "behav": {}, "items": {},
         "itemmattr": {}, "probes": {}}
    for f in FORMATS:
        if (d / f"behav_{f}.json").exists():
            M["behav"][f] = json.load(open(d / f"behav_{f}.json"))
        if (d / f"items_{f}.json").exists():
            M["items"][f] = json.load(open(d / f"items_{f}.json"))
        if (d / f"itemmattr_{f}.json").exists():
            M["itemmattr"][f] = json.load(open(d / f"itemmattr_{f}.json"))
        if (d / f"probes_{f}.json").exists():
            M["probes"][f] = json.load(open(d / f"probes_{f}.json"))
    return M


def correct_map(M, f):
    return {r["id"]: float(r["correct"]) for r in M["behav"][f]["items"]}


def lmg(X, y, names):
    """Lindeman-Merenda-Gold relative importance: average over all orderings of the R^2
    increment of each predictor (all-subsets OLS)."""
    n, p = X.shape

    def r2(cols):
        if not cols:
            return 0.0
        A = np.column_stack([np.ones(n), X[:, cols]])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        return 1 - res.var() / y.var()

    cache = {}
    def R(cols):
        key = tuple(sorted(cols))
        if key not in cache:
            cache[key] = r2(list(key))
        return cache[key]

    out = {}
    for j in range(p):
        others = [k for k in range(p) if k != j]
        tot, cnt = 0.0, 0
        for size in range(len(others) + 1):
            for sub in itertools.combinations(others, size):
                tot += R(list(sub) + [j]) - R(list(sub))
                cnt += 1
        out[names[j]] = tot / cnt
    return out, R(list(range(p)))


def ols(X, y, names):
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    s2 = (res @ res) / (n - p - 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), n - p - 1)
    return {nm: {"beta": float(b), "p": float(pp)} for nm, b, pp in zip(["const"] + names, beta, pv)}


def main():
    models = [m for m in (load_model(d) for d in sorted(ROOT.iterdir()) if d.is_dir()) if m]
    if not models:
        print("no finished models under", ROOT)
        return
    S = {"models": [m["name"] for m in models]}
    lines = []
    P = lines.append

    # ------------------------------------------------------------------ Fig 2: accuracy
    P("## Accuracy per format (Fig. 2)\n")
    P("| model | " + " | ".join(FORMATS) + " | valid pairs (num/en/es/it) |")
    P("|---|" + "---|" * (len(FORMATS) + 1))
    S["acc"] = {}
    for m in models:
        acc = {f: m["behav"][f]["acc"] for f in FORMATS if f in m["behav"]}
        nv = [str(m["behav"][f]["n_valid"]) for f in FORMATS if f in m["behav"]]
        S["acc"][m["name"]] = acc
        P(f"| {m['name']} | " + " | ".join(f"{100 * acc.get(f, float('nan')):.1f}" for f in FORMATS)
          + " | " + "/".join(nv) + " |")
    med = {f: np.median([S["acc"][n][f] for n in S["acc"] if f in S["acc"][n]]) for f in FORMATS}
    P("| **median** | " + " | ".join(f"{100 * med[f]:.1f}" for f in FORMATS) + " | |")
    P("\nPaper medians: numeric 87.4, English 36.4, Spanish 17.9, Italian 6.8.\n")

    # ------------------------------------------------------------------ Sec 3.2: overlap
    P("## Top-1% circuit overlap with the numeric circuit (Sec. 3.2)\n")
    S["jaccard"] = {}
    P("| model | " + " | ".join(f"{meth} {f[:2]}" for meth in METHODS for f in VERBAL) + " | mattr-vs-ap (num) |")
    P("|---|" + "---|" * (2 * len(VERBAL) + 1))
    for m in models:
        J = m["circuits"]["jaccard"]
        row = {}
        for meth in METHODS:
            for f in VERBAL:
                row[f"{meth}:{f}"] = J.get(f"{meth}:numeric|{meth}:{f}", float("nan"))
        row["mattr_vs_ap"] = J.get("mattr:numeric|ap:numeric", float("nan"))
        S["jaccard"][m["name"]] = row
        P(f"| {m['name']} | " + " | ".join(f"{row[f'{meth}:{f}']:.3f}" for meth in METHODS for f in VERBAL)
          + f" | {row['mattr_vs_ap']:.3f} |")
    P("| **mean** | " + " | ".join(
        f"{np.nanmean([S['jaccard'][n][f'{meth}:{f}'] for n in S['jaccard']]):.3f}" for meth in METHODS for f in VERBAL)
      + f" | {np.nanmean([S['jaccard'][n]['mattr_vs_ap'] for n in S['jaccard']]):.3f} |")
    P("\nPaper (AP): English 0.145, Spanish 0.078, Italian 0.069.\n")

    P("### Across-model correlation between overlap and accuracy\n")
    P("| method | format | Pearson r | p | Spearman rho | N |")
    P("|---|---|---|---|---|---|")
    S["overlap_acc_corr"] = {}
    for meth in METHODS:
        for f in VERBAL:
            xs = [S["jaccard"][n][f"{meth}:{f}"] for n in S["jaccard"]]
            ys = [S["acc"][n][f] for n in S["jaccard"]]
            ok = [i for i in range(len(xs)) if not np.isnan(xs[i])]
            xs, ys = [xs[i] for i in ok], [ys[i] for i in ok]
            if len(xs) >= 3:
                r, p = stats.pearsonr(xs, ys); rho, _ = stats.spearmanr(xs, ys)
            else:
                r = p = rho = float("nan")
            S["overlap_acc_corr"][f"{meth}:{f}"] = {"r": float(r), "p": float(p), "rho": float(rho), "n": len(xs)}
            P(f"| {meth} | {f} | {r:+.2f} | {p:.3f} | {rho:+.2f} | {len(xs)} |")
    P("\nPaper (AP): English r=0.75 (p=0.003), Spanish r=0.58 (p=0.04), Italian r=0.47 (p=0.10), N=13.\n")

    # ------------------------------------------------------------------ sufficiency curves
    P("## Sufficiency of the top-k units (restored preference fraction, patched clean into x')\n")
    S["curves"] = {m["name"]: m["circuits"]["curves"] for m in models}
    fracs = None
    for m in models:
        for f in FORMATS:
            if f in m["circuits"]["curves"]:
                fracs = list(next(iter(m["circuits"]["curves"][f].values())).keys())
                break
        if fracs:
            break
    if fracs:
        P("Mean over models of the restored fraction at each top-k fraction (own-format circuit).\n")
        P("| format | ranking | " + " | ".join(fracs) + " |")
        P("|---|---|" + "---|" * len(fracs))
        for f in FORMATS:
            for rk in ("mattr", "ap", "random", "mattr_numeric", "ap_numeric"):
                vals = []
                for fr in fracs:
                    v = [m["circuits"]["curves"][f][rk][fr] for m in models
                         if f in m["circuits"]["curves"] and rk in m["circuits"]["curves"][f]]
                    vals.append(np.mean(v) if v else float("nan"))
                if not all(np.isnan(vals)):
                    P(f"| {f} | {rk} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        P("")

    # ------------------------------------------------------------------ Fig 3: loading vs correctness
    P("## Item-level: numeric-circuit loading predicts correctness (Fig. 3, point-biserial r)\n")
    P("Loading = sum of the item's AP attributions over the circuit's units (paper's definition), "
      "for the circuit found by each method.  `mattr-item` = per-item MAttr scores summed over the "
      "MAttr numeric circuit (subset of items).\n")
    S["pbr"] = defaultdict(dict)
    cols = [f"{meth}:{f}" for f in VERBAL for meth in METHODS] + [f"item:{f}" for f in VERBAL]
    P("| model | " + " | ".join(cols) + " |")
    P("|---|" + "---|" * len(cols))
    for m in models:
        cells = []
        for f in VERBAL:
            for meth in METHODS:
                key = f"{meth}:numeric"
                if f in m["items"] and key in m["items"][f]["loading"]:
                    L = m["items"][f]["loading"][key]; C = correct_map(m, f)
                    ids = m["items"][f]["ids"]
                    r, p = pbr([L[str(i)] for i in ids], [C[i] for i in ids])
                else:
                    r = p = float("nan")
                S["pbr"][m["name"]][f"{meth}:{f}"] = {"r": r, "p": p}
                cells.append(fmt_r(r, p))
        for f in VERBAL:
            if f in m["itemmattr"] and "mattr:numeric" in m["itemmattr"][f]["loading"]:
                L = m["itemmattr"][f]["loading"]["mattr:numeric"]; C = correct_map(m, f)
                ids = m["itemmattr"][f]["ids"]
                r, p = pbr([L[str(i)] for i in ids], [C[i] for i in ids])
            else:
                r = p = float("nan")
            S["pbr"][m["name"]][f"item:{f}"] = {"r": r, "p": p}
            cells.append(fmt_r(r, p))
        P(f"| {m['name']} | " + " | ".join(cells) + " |")
    P("| **median** | " + " | ".join(
        f"{np.nanmedian([S['pbr'][n][c]['r'] for n in S['pbr']]):+.2f}" for c in cols) + " |")
    P("| **# sig (p<.05)** | " + " | ".join(
        str(sum(1 for n in S['pbr'] if S['pbr'][n][c]['p'] < 0.05)) + f"/{len(models)}" for c in cols) + " |")
    P("\nPaper (AP): English 12/13 significant, median 0.38 (0.11-0.56); Spanish 11/13, median 0.32; "
      "Italian 9/13, median 0.29.\n")

    # binned accuracy vs loading (Fig 3B), pooled per model with per-model decile bins
    S["loading_bins"] = {}
    for meth in METHODS:
        for f in VERBAL:
            rows = []
            for m in models:
                key = f"{meth}:numeric"
                if f not in m["items"] or key not in m["items"][f]["loading"]:
                    continue
                L = m["items"][f]["loading"][key]; C = correct_map(m, f)
                ids = m["items"][f]["ids"]
                x = np.array([L[str(i)] for i in ids]); y = np.array([C[i] for i in ids])
                ent = {r["id"]: r["entropy"] for r in m["behav"][f]["items"]}
                e = np.array([ent[i] for i in ids])
                q = np.quantile(x, np.linspace(0, 1, 11))
                b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, 9)
                for k in range(10):
                    if (b == k).any():
                        rows.append({"model": m["name"], "bin": k, "loading": float(x[b == k].mean()),
                                     "acc": float(y[b == k].mean()), "entropy": float(e[b == k].mean()),
                                     "n": int((b == k).sum())})
            S["loading_bins"][f"{meth}:{f}"] = rows

    # ------------------------------------------------------------------ App C: causal validation
    P("## Causal validation: patch-restored fraction vs correctness (App. C)\n")
    P("Point-biserial r between the restored fraction of the preference when the numeric circuit's "
      "units are patched clean into x' and item correctness; `rand` = random units matched per "
      "layer. Wilcoxon across models: circuit vs random.\n")
    S["patch"] = defaultdict(dict)
    cols = [f"{meth}:{f}:{k}" for f in VERBAL for meth in METHODS for k in ("circ", "rand")]
    P("| model | " + " | ".join(c.replace(":", " ") for c in cols) + " |")
    P("|---|" + "---|" * len(cols))
    for m in models:
        cells = []
        for f in VERBAL:
            for meth in METHODS:
                for k in ("circ", "rand"):
                    key = f"{meth}:numeric" + ("" if k == "circ" else ":random0")
                    if f in m["items"] and key in m["items"][f]["patch"]:
                        Pt = m["items"][f]["patch"][key]; C = correct_map(m, f)
                        ids = m["items"][f]["ids"]
                        r, p = pbr([Pt[str(i)] for i in ids], [C[i] for i in ids])
                        mean_rf = float(np.mean([Pt[str(i)] for i in ids]))
                    else:
                        r = p = mean_rf = float("nan")
                    S["patch"][m["name"]][f"{meth}:{f}:{k}"] = {"r": r, "p": p, "mean_restored": mean_rf}
                    cells.append(fmt_r(r, p))
        P(f"| {m['name']} | " + " | ".join(cells) + " |")
    P("| **median** | " + " | ".join(
        f"{np.nanmedian([S['patch'][n][c]['r'] for n in S['patch']]):+.2f}" for c in cols) + " |")
    P("| **mean restored** | " + " | ".join(
        f"{np.nanmean([S['patch'][n][c]['mean_restored'] for n in S['patch']]):.3f}" for c in cols) + " |")
    S["patch_wilcoxon"] = {}
    for meth in METHODS:
        for f in VERBAL:
            a = [S["patch"][n][f"{meth}:{f}:circ"]["r"] for n in S["patch"]]
            b = [S["patch"][n][f"{meth}:{f}:rand"]["r"] for n in S["patch"]]
            ok = [i for i in range(len(a)) if not (np.isnan(a[i]) or np.isnan(b[i]))]
            if len(ok) >= 5:
                w = stats.wilcoxon([a[i] for i in ok], [b[i] for i in ok], alternative="greater")
                S["patch_wilcoxon"][f"{meth}:{f}"] = {"p": float(w.pvalue), "n": len(ok)}
                P(f"- {meth} / {f}: circuit > random, Wilcoxon p = {w.pvalue:.4f} (N={len(ok)})")
    P("\nPaper (AP): circuit > random, Wilcoxon p<0.001 in English/Spanish, p<0.01 in Italian.\n")

    # ------------------------------------------------------------------ App E: correct/incorrect circuits
    P("## Circuits from correct-only vs incorrect-only numeric items (App. E)\n")
    P("Point-biserial r of loading (verbal items) on the numeric circuit derived from all / "
      "correct / incorrect numeric items.\n")
    S["appE"] = defaultdict(dict)
    subs = ("numeric", "numeric_correct", "numeric_incorrect")
    cols = [f"{meth}:{f}:{s.replace('numeric', 'num')}" for f in VERBAL for meth in METHODS for s in subs]
    P("| model | " + " | ".join(c.replace(":", " ") for c in cols) + " |")
    P("|---|" + "---|" * len(cols))
    for m in models:
        cells = []
        for f in VERBAL:
            for meth in METHODS:
                for s in subs:
                    key = f"{meth}:{s}"
                    if f in m["items"] and key in m["items"][f]["loading"]:
                        L = m["items"][f]["loading"][key]; C = correct_map(m, f)
                        ids = m["items"][f]["ids"]
                        r, p = pbr([L[str(i)] for i in ids], [C[i] for i in ids])
                    else:
                        r = p = float("nan")
                    S["appE"][m["name"]][f"{meth}:{f}:{s}"] = {"r": r, "p": p}
                    cells.append(fmt_r(r, p))
        P(f"| {m['name']} | " + " | ".join(cells) + " |")
    P("| **median** | " + " | ".join(
        f"{np.nanmedian([S['appE'][n][c.replace(':num', ':numeric')]['r'] for n in S['appE']]):+.2f}"
        for c in cols) + " |")
    P("")

    # ------------------------------------------------------------------ App F: reciprocity
    P("## Reciprocity: circuit of format A predicting accuracy in format B (App. F)\n")
    P("Median across models of the point-biserial r between loading on A's circuit and correctness "
      "on B's items (rows: circuit source A; columns: evaluated format B).\n")
    S["recip"] = {}
    for meth in METHODS:
        P(f"**{meth}**\n")
        P("| circuit \\ items | " + " | ".join(FORMATS) + " |")
        P("|---|" + "---|" * len(FORMATS))
        for a in FORMATS:
            cells = []
            for b in FORMATS:
                rs = []
                for m in models:
                    key = f"{meth}:{a}"
                    if b in m["items"] and key in m["items"][b]["loading"]:
                        L = m["items"][b]["loading"][key]; C = correct_map(m, b)
                        ids = m["items"][b]["ids"]
                        r, p = pbr([L[str(i)] for i in ids], [C[i] for i in ids])
                        if not np.isnan(r):
                            rs.append(r)
                S["recip"][f"{meth}:{a}->{b}"] = rs
                cells.append(f"{np.median(rs):+.2f} (n={len(rs)})" if rs else "n/a")
            P(f"| {a} | " + " | ".join(cells) + " |")
        P("")
    P("Paper (AP): any format's circuit predicts any other's accuracy (r 0.26-0.53); numeric->verbal "
      "0.30-0.38 > verbal->numeric 0.26-0.28.\n")

    # ------------------------------------------------------------------ Fig 4: probes + confidence
    S["fig4"] = {}
    for m in models:
        if not m["probes"]:
            continue
        P(f"## Loading vs probes and confidence: {m['name']} (Fig. 4 / Sec. 3.4)\n")
        P("Linear probability model, standardised predictors: correct ~ loading + probe_resid + "
          "probe_mlp + mean_logprob + entropy.  beta / p, then LMG share of R^2.\n")
        for meth in METHODS + ("mattr-patch",):
            for f in VERBAL:
                if f not in m["items"] or f not in m["probes"]:
                    continue
                if meth == "mattr-patch":       # causal loading: restored fraction, MAttr circuit
                    if "mattr:numeric" not in m["items"][f]["patch"]:
                        continue
                    L = m["items"][f]["patch"]["mattr:numeric"]
                else:
                    key = f"{meth}:numeric"
                    if key not in m["items"][f]["loading"]:
                        continue
                    L = m["items"][f]["loading"][key]
                C = correct_map(m, f)
                pr = m["probes"][f]
                beh = {r["id"]: r for r in m["behav"][f]["items"]}
                ids = [i for i in m["items"][f]["ids"] if str(i) in pr["resid"]]
                names = ["loading", "probe_resid", "probe_mlp", "mean_logprob", "entropy"]
                X = np.array([[L[str(i)], pr["resid"][str(i)], pr["mlp"][str(i)],
                               beh[i]["mean_logprob"], beh[i]["entropy"]] for i in ids])
                y = np.array([C[i] for i in ids])
                Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
                coef = ols(Xs, y, names)
                shares, r2 = lmg(Xs, y, names)
                S["fig4"][f"{m['name']}:{meth}:{f}"] = {"ols": coef, "lmg": shares, "r2": r2, "n": len(ids),
                                                        "probe_cv_auc": pr.get("cv_auc"), "probe_layer": pr.get("layer")}
                P(f"**{meth} / {f}** (N={len(ids)}, R^2={r2:.3f}; probe layers {pr.get('layer')}, "
                  f"CV AUC {pr.get('cv_auc')})\n")
                P("| predictor | beta | p | LMG share of R^2 (% of variance) |")
                P("|---|---|---|---|")
                for nm in names:
                    P(f"| {nm} | {coef[nm]['beta']:+.3f} | {coef[nm]['p']:.2g} | {100 * shares[nm]:.1f} |")
                P("")
        P("Paper (Llama-3.1-8B, AP): loading beta=0.127 p<1e-4 (11.0% of variance) in English; "
          "0.030 p=0.002 (5.3%) Spanish; 0.031 p=0.006 (6.0%) Italian.\n")

    text = "\n".join(lines)
    (ROOT / "summary.md").write_text(text)
    json.dump(S, open(ROOT / "summary.json", "w"), indent=1, default=float)
    print(text)


if __name__ == "__main__":
    main()
