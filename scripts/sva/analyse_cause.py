"""Rank attribution methods on the CAUSE direction: patch the top-k units to the source
activations (complement clean) and ask whether the output flips to the SOURCE label.

The metric the paper calls Cause (causal-abstraction.tex) is, per budget k,
    acc_source(k) = mean_i 1[logit_source > logit_base]   under top-k <- source
and every eval_sva.py json already carries it as cause_metrics["acc_source"] (curve) and
cause_accsrc_auc (its log-k AUC). Nothing here is re-evaluated; this only reads jsons.

Per (substrate, task, method, loss) we report
  * cause_accsrc_auc         (higher = flips sooner)
  * k*_50 / k*_90            smallest k on the grid with acc_source >= 0.5 / 0.9
  * acc_auc                  the iso metric, for contrast
and then rank methods per substrate averaged over tasks (task-group averaged like
plot_accauc_vs_faithauc.group_avg so ARC-E/IOI don't get outvoted by the 4 SVA tasks).

Also matches every cause-TRAINED MAttr run (results/sva_sweep_cause, tag necessary_*) to its
iso-trained twin (results/sva_sweep, tag sufficient_*) and reports the paired delta.

Run:  uv run python scripts/sva/analyse_cause.py [--sub node|mlp|mlp+attn_head|all]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots"))
import plot_accauc_vs_faithauc as R   # noqa: E402  (parse_method, on_model, task groups)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


def auc_of(ys, ks):
    lx = np.log10(np.asarray(ks, float)); ya = np.asarray(ys, float)
    return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))


def kstar(curve, ks, thr):
    for x, a in zip(ks, curve):
        if a >= thr:
            return float(x)
    return float("nan")


def parse_any(fname, d):
    """parse_method plus the cause-trained MAttr tags it deliberately drops."""
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    # mode prefix FIRST: parse_method's hard_topk branch does not look at the prefix, so a
    # `necessary_hard_topk_*` tag would otherwise come back as an iso "soft-log".
    if tag.startswith("necessary_") or tag.startswith("joint_"):
        mode = tag.split("_")[0]
        t2 = tag.replace(mode + "_", "sufficient_", 1)
        m = R.parse_method(fname.replace(tag, t2), d)
        return (m, "cause" if mode == "necessary" else "joint") if m else (None, None)
    m = R.parse_method(fname, d)
    return (m, "iso") if m is not None else (None, None)


def load(dirs):
    rows = []
    for res in dirs:
        for f in sorted(glob.glob(res + "/*.json")):
            d = json.load(open(f))
            if not R.on_model(d):
                continue
            m, trained = parse_any(os.path.basename(f), d)
            if m is None:
                continue
            cm, ks = d["cause_metrics"], d["n_nodes"]
            rows.append(dict(
                dir=res, task=d["task"], model=d["model"], sub=d["nodes"], method=m,
                trained=trained, loss=d["loss"] if m != "Random" else "n/a",
                cause_accsrc_auc=d["cause_accsrc_auc"], cause_auc=d["cause_auc"],
                cause_psrc_auc=d["cause_psrc_auc"],
                k50=kstar(cm["acc_source"], ks, 0.5), k90=kstar(cm["acc_source"], ks, 0.9),
                acc_src_k1=cm["acc_source"][0],
                acc_auc=d["acc_auc"], faith_auc=d["faith_auc"], total=d["total"],
                file=os.path.basename(f)))
    return pd.DataFrame(rows)


def group_avg(df, col):
    """Task-group average (SVA, Arith, ARC-E, IOI) of `col`, requiring the substrate's full
    group set (R.REQUIRED) so partial sweeps don't compete with complete ones."""
    out = []
    for (sub, method, trained, loss), g in df.groupby(["sub", "method", "trained", "loss"]):
        gm, n = [], 0
        for gname, tasks in R.GROUPS:
            v = g[g.task.isin(tasks)][col]
            if len(v):
                gm.append(v.mean()); n += len(v)
        need = R.REQUIRED.get(sub, [gn for gn, _ in R.GROUPS])
        have = [gn for gn, tasks in R.GROUPS if g.task.isin(tasks).any()]
        if not all(gn in have for gn in need):
            continue
        out.append(dict(sub=sub, method=method, trained=trained, loss=loss, n=n,
                        **{col: float(np.mean(gm))}))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="all")
    ap.add_argument("--dirs", nargs="+",
                    default=["results/sva_sweep", "results/sva_sweep_cause"])
    a = ap.parse_args()
    df = load(a.dirs)
    if a.sub != "all":
        df = df[df["sub"] == a.sub]
    print(f"{len(df)} runs; substrates {sorted(df['sub'].unique())}; "
          f"trained modes {df.trained.value_counts().to_dict()}")

    # 1. method x loss ranking on cause_accsrc_auc, per substrate (task-group averaged)
    for col in ("cause_accsrc_auc", "acc_auc"):
        ga = group_avg(df, col)
        for sub, g in ga.groupby("sub"):
            piv = g.pivot_table(index=["method", "trained"], columns="loss", values=col)
            piv["best"] = piv.max(axis=1)
            print(f"\n=== {col} | substrate={sub} | task-group avg (higher=better) ===")
            print(piv.sort_values("best", ascending=False).round(3).to_string())

    # 2. per-task table for the headline methods at ce loss (the paper's default)
    print("\n=== per-task cause_accsrc_auc (loss=ce, plus Random) ===")
    sel = df[(df.loss.isin(["ce", "n/a"]))]
    for sub, g in sel.groupby("sub"):
        piv = g.assign(m=g.method + np.where(g.trained == "iso", "", "[" + g.trained + "]")) \
               .pivot_table(index="m", columns="task", values="cause_accsrc_auc")
        print(f"\n--- {sub} ---"); print(piv.round(3).to_string())
        piv = g.assign(m=g.method + np.where(g.trained == "iso", "", "[" + g.trained + "]")) \
               .pivot_table(index="m", columns="task", values="k50")
        print(f"--- {sub}: k*(acc_source>=0.5) ---"); print(piv.round(0).to_string())

    # 3. paired: cause-trained vs iso-trained MAttr (same task/sub/method/loss)
    c = df[df.trained == "cause"]; i = df[df.trained == "iso"]
    key = ["task", "sub", "method", "loss"]
    pr = c.merge(i, on=key, suffixes=("_cause", "_iso"))
    if len(pr):
        print("\n=== cause-trained vs iso-trained MAttr (paired) ===")
        for col in ("cause_accsrc_auc", "k50", "k90", "acc_auc", "faith_auc"):
            pr[f"d_{col}"] = pr[f"{col}_cause"] - pr[f"{col}_iso"]
        show = pr[key + ["cause_accsrc_auc_iso", "cause_accsrc_auc_cause", "d_cause_accsrc_auc",
                         "k50_iso", "k50_cause", "acc_auc_iso", "acc_auc_cause"]]
        print(show.sort_values(key).round(3).to_string(index=False))
        print("\nmean delta by method x loss:")
        print(pr.groupby(["method", "loss"])[["d_cause_accsrc_auc", "d_acc_auc", "d_faith_auc"]]
                .mean().round(3).to_string())
        print("win-rate (cause-trained beats iso-trained on cause_accsrc_auc):",
              f"{(pr.d_cause_accsrc_auc > 0).mean():.2f}  n={len(pr)}")
    # 4. does cause-trained MAttr beat IG (the best iso-trained cause ranking) cell-for-cell?
    ig = df[(df.method == "IG")][["task", "sub", "loss", "cause_accsrc_auc", "k50", "k90"]]
    for trained in ("cause", "joint", "iso"):
        m = df[(df.trained == trained) & df.method.isin(["stopk-log-eps1e-2", "softsgd-log", "stopk-log"])]
        if not len(m):
            continue
        pr = m.merge(ig, on=["task", "sub", "loss"], suffixes=("", "_IG"))
        pr["d_vs_IG"] = pr.cause_accsrc_auc - pr.cause_accsrc_auc_IG
        print(f"\n=== {trained}-trained MAttr vs IG (same task/sub/loss), cause_accsrc_auc ===")
        print(pr.groupby(["sub", "method", "loss"]).agg(
            n=("d_vs_IG", "size"), mean_d=("d_vs_IG", "mean"),
            win=("d_vs_IG", lambda x: (x > 0).mean()),
            mattr=("cause_accsrc_auc", "mean"), IG=("cause_accsrc_auc_IG", "mean"),
            k90_mattr=("k90", "median"), k90_IG=("k90_IG", "median")).round(3).to_string())
    df.to_csv("results/cause_analysis.csv", index=False)
    print("\nwrote results/cause_analysis.csv")


if __name__ == "__main__":
    main()
