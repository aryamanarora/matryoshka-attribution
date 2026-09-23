"""Generate the two hyperparameter tables of paper/sections/hparams.tex:

    uv run python scripts/mib/make_hparams_tables.py
        -> paper/tabs/hparams_mib.tex        (tab:hparams-mib, node-level MIB methods of fig:cpr-mib-test)
        -> paper/tabs/hparams_mib_edge.tex   (tab:hparams-mib-edge, edge-level MIB methods of the same figure)
        -> paper/tabs/hparams_sva.tex   (tab:hparams-sva, the SVA+ methods of fig:acc-faith)

Both tables carry the SAME MAttr rows as the results tables they sit next to, in the same order
and with the same labels, because the rows come from the same lists:
  MIB  -> make_mib_test_table.OUR_NODE_METHODS / OUR_EDGE_METHODS (label, results dir), and the
          optimizer / LR / steps / k-schedule / Adam-eps of each row are READ FROM THE RUN
          (the `args` dict saved in results/<dir>/<cell>_scores.pt), not typed here.
  SVA+ -> make_sva_table.BLOCKS' \\ourmethod{} block (V.METHODS key, label); optimizer and
          k-schedule are read from a representative run's JSON, eps and steps from its filename
          tag (`_eps1e-2`, `_s5000`), and the LR from SVA_LR below, which mirrors the sweep
          launchers (scripts/sva/launch/*.sh: 0.05 on the neuron/node substrates, 0.5 on SAE
          latents, SGD 1.0) -- the one value the runs do not record.
The headline (bare \\ourmethod{}) and every relative label come from scripts/mib/mattr_variants.py,
so flipping the headline there flips these tables with the results tables.

The non-MAttr rows (Node Pruning, DBM, the gradient methods, the leaderboard rows) are fixed
specs kept in this file, next to the run they describe; they change when a recipe changes, not
when the headline does.
"""

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mib"))
sys.path.insert(0, str(ROOT / "scripts" / "sva"))
import mattr_variants as MV            # noqa: E402
import make_mib_test_table as T        # noqa: E402  (OUR_*_METHODS: label -> results dir)
import make_sva_table as S             # noqa: E402  (BLOCKS: V.METHODS key -> label)
V = S.V                                # plots/plot_accauc_vs_faithauc: parse_method, SUBSTRATE_RES

TABS = ROOT / "paper" / "tabs"
RESULTS = ROOT / "results"

# SVA+ learning rates by (optimizer, substrate family). Not recorded in the run JSONs; these are
# the values the launchers pass (scripts/sva/launch/submit_adam_eps_followup.sh etc.).
SVA_LR = {("adam", "neuron"): "0.05", ("adam", "sae"): "0.5", ("sgd", "neuron"): "1", ("sgd", "sae"): "1"}

HEADER = [
    "\\begin{adjustbox}{max width=\\textwidth}",
    "\\begin{tabular}{lllllll}",
    "\\toprule",
    "\\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{3}{c}{\\textbf{Scores}} & "
    "\\multirow{2}{*}{\\textbf{HParams}} & \\multirow{2}{*}{\\textbf{Data}} & "
    "\\multirow{2}{*}{\\textbf{Bwd./Data}} \\\\",
    "\\cmidrule(lr){2-4}",
    "& \\textbf{Optimiser} & \\textbf{LR} & $\\boldsymbol{\\epsilon}$ & & & \\\\",
    "\\midrule",
]
FOOTER = ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]


def num(x):
    """0.05 -> $0.05$, 3.0 -> $3$, 1.0 -> $1$."""
    s = f"{float(x):g}"
    return f"${s}$"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def eps_tex(eps):
    mant, exp = f"{float(eps):.0e}".split("e")
    return f"$10^{{{int(exp)}}}$" if float(mant) == 1 else f"${mant}\\times 10^{{{int(exp)}}}$"


def optimiser_cell(opt):
    return "Adam" if opt == "adam" else "SGD"


def eps_cell(opt, eps):
    """Adam's epsilon; SGD has none."""
    return eps_tex(eps) if opt == "adam" else "---"


def schedule_cell(k):
    return {"log": "$k$ log-uniform", "uniform": "$k$ uniform"}[k]


def row(label, cells, indent):
    return ("\\quad " if indent else "") + label + " & " + " & ".join(cells) + " \\\\"


def mattr_rows(pairs, describe):
    """(label, key) pairs -> table rows; the headline (bare \\ourmethod{}) is unindented."""
    out = []
    for label, key in pairs:
        d = describe(key)
        if d is None:          # variant not run on the substrates this table covers
            continue
        opt, lr, steps, k, eps, n_ex = d
        out.append(row(label, [optimiser_cell(opt), lr, eps_cell(opt, eps), schedule_cell(k), n_ex, "1"],
                       indent=label != MV.OURMETHOD))
    return out


# ---------------------------------------------------------------- MIB (tab:hparams-mib)
def mib_args(results_dir):
    """The saved `args` of one cell of a MIB results dir (all cells share the recipe)."""
    d = RESULTS / results_dir
    cands = sorted(d.glob("*_scores.pt"))
    if not cands:
        raise SystemExit(f"no *_scores.pt in results/{results_dir} -- cannot read its hparams")
    pref = [p for p in cands if p.name.startswith("ioi_gpt2")] or cands
    a = torch.load(pref[0], map_location="cpu", weights_only=False)["args"]
    return a, pref[0].name


def describe_mib(results_dir):
    a, src = mib_args(results_dir)
    opt = a.get("optimizer") or "adam"        # pre-2026-08 dirs saved no key: Adam was the default
    eps = a.get("adam_eps", 1e-8)
    steps = a["steps"]
    return opt, num(a["lr"]), steps, a["k_schedule"], eps, thousands(steps)


# cells: optimiser, LR, eps, path, # examples, steps/example. Node Pruning and DBM use torch's
# default Adam eps (src/matryoshka_attribution/edge_pruning.py constructs the optimiser without one).
NONE3 = ["---", "---", "---"]
MIB_FIXED_NODE = [
    ("Node Pruning", ["Adam", "$0.8$", "$10^{-8}$", "hard-concrete, $s = 0.5$", "3{,}000", "1"]),
    ("DBM", ["Adam", "$0.3$", "$10^{-8}$", "sigmoid, $\\tau\\!:\\,50\\!\\to\\!0.1$, $\\lambda_{L_1} = 6$",
             "3{,}000", "1"]),
    None,
    ("Expected Gradients", NONE3 + ["$\\alpha \\sim U(0,1)$, seed 0", "100--1{,}000", "1"]),
    ("IG ($m{=}30$)", NONE3 + ["$\\alpha = j/30$, $j = 1 \\dots 30$", "100--1{,}000", "30"]),
    ("IG ($m{=}10$)", NONE3 + ["$\\alpha = j/10$, $j = 1 \\dots 10$", "100--1{,}000", "10"]),
    ("IG ($m{=}5$)", NONE3 + ["$\\alpha = j/5$, $j = 1 \\dots 5$", "100--1{,}000", "5"]),
    ("I$\\times$G", NONE3 + ["$\\alpha = 0$", "100--1{,}000", "1"]),
    ("RelP, RelP$+$QK", NONE3 + ["--- (LRP rule)", "100--1{,}000", "1"]),
    ("AttnLRP, GIM", NONE3 + ["--- (LRP rule)", "100--1{,}000", "1"]),
    ("NAP (CF), NAP-IG (CF) & \\multicolumn{6}{l}{as published (MIB leaderboard, counterfactual)}", None),
    ("Random (control) & \\multicolumn{6}{l}{as published (MIB leaderboard)}", None),
]
MIB_FIXED_EDGE = [
    ("EAP-IG-inp (CF) & \\multicolumn{6}{l}{as published (MIB leaderboard, counterfactual)}", None),
    ("UGS & \\multicolumn{4}{l}{as published} & 7.2k--114k seq. & ---", None),
]


def fixed_rows(spec):
    out = []
    for item in spec:
        if item is None:
            out.append("\\midrule")
        elif item[1] is None:
            out.append(item[0] + " \\\\")
        else:
            out.append(row(item[0], item[1], indent=False))
    return out


def build_mib():
    """Node-level table."""
    L = list(HEADER)
    L += mattr_rows(T.OUR_NODE_METHODS, describe_mib)
    L.append("\\midrule")
    L += fixed_rows(MIB_FIXED_NODE)
    L += FOOTER
    return "\n".join(L) + "\n"


def build_mib_edge():
    """Edge-level table."""
    L = list(HEADER)
    L += mattr_rows(T.OUR_EDGE_METHODS, describe_mib)
    L.append("\\midrule")
    L += fixed_rows(MIB_FIXED_EDGE)
    L += FOOTER
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- SVA+ (tab:hparams-sva)
# Neuron and SAE substrates only: the table does not describe the node-level runs, and
# node-only variants are dropped from it.
SVA_TREES = [("mlp", V.SUBSTRATE_RES["mlp"], "neuron"),
             ("mlp_sae_span", V.SUBSTRATE_RES["mlp_sae_span"], "sae")]


def sva_run(key, sub, res):
    """One representative logit-diff run of V.METHODS key `key` on substrate `sub`: its JSON
    dict and filename (any task; the recipe is per (method, substrate))."""
    import json
    for p in sorted((ROOT / res).glob(f"*_{sub.replace('+', '-')}_*.json")):
        if p.name.endswith(("_ce.json", "_acc.json")) or ".scores" in p.name:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("nodes") != sub or d.get("loss", "logit_diff") != "logit_diff":
            continue
        if V.parse_method(p.name, d) == key:
            return d, p.name
    return None, None


def describe_sva(key):
    per = {}
    for sub, res, fam in SVA_TREES:
        d, fname = sva_run(key, sub, res)
        if d is None:
            continue
        opt = d.get("optimizer", "adam")
        m = re.search(r"_eps([0-9.e-]+?)_", fname)
        eps = float(m.group(1)) if m else 1e-8
        m = re.search(r"_s(\d{3,})", fname)
        steps = int(m.group(1)) if m else 2000
        # optimizer "none" is the no-learning control (frozen scores): it has no LR at all,
        # so SVA_LR has no entry for it and the LR cell prints a dash.
        per[sub] = (opt, SVA_LR.get((opt, fam)), steps, d.get("k_schedule", "log"), eps)
    if not per:
        return None
    opt, _, _, k, eps = per.get("mlp") or per["mlp_sae_span"]
    lrs = sorted({v[1] for v in per.values() if v[1] is not None}, key=float)
    if not lrs:
        lr = "---"
    else:
        lr = num(lrs[0]) if len(lrs) == 1 else " / ".join(num(x) for x in lrs) + " (SAE)" * 0
    if len(lrs) > 1:
        mlp_lr = per["mlp"][1] if per.get("mlp") and per["mlp"][1] is not None else lrs[0]
        sae_lr = per["mlp_sae_span"][1] if per.get("mlp_sae_span") and per["mlp_sae_span"][1] is not None else lrs[-1]
        lr = f"{num(mlp_lr)} ({num(sae_lr)} SAE)"
    steps = sorted({v[2] for v in per.values()})
    n_ex = " / ".join(thousands(x) for x in steps)
    return opt, lr, steps, k, eps, n_ex


SVA_FIXED = [
    None,
    ("Node Pruning", ["Adam", "$0.8$", "$10^{-8}$", "hard-concrete, $s = 0.9$", "5{,}000", "1"]),
    ("DBM", ["Adam", "$0.3$", "$10^{-8}$", "sigmoid, $\\tau\\!:\\,50\\!\\to\\!0.1$, $\\lambda_{L_1} = 6$",
             "5{,}000", "1"]),
    None,
    ("Expected Gradients", NONE3 + ["$\\alpha \\sim U(0,1)$, seed 42", "5{,}000$^{*}$", "1"]),
    ("IG ($m{=}10$)", NONE3 + ["$\\alpha = j/10$, $j = 1 \\dots 10$", "500$^{*}$", "10"]),
    ("I$\\times$G", NONE3 + ["$\\alpha = 0$", "5{,}000$^{*}$", "1"]),
    ("AttnLRP", NONE3 + ["--- (LRP rule)", "5{,}000$^{*}$", "1"]),
    ("Random (control)", NONE3 + ["i.i.d.\\ uniform scores, seeds 42--44", "---", "---"]),
]
SVA_FOOTNOTE = ("\\multicolumn{7}{l}{\\footnotesize $^{*}$compute-matched "
                "to \\ourmethod{}'s pass budget, capped at the full train pool (no repetition).} \\\\")


def build_sva():
    ours = [members for title, members in S.BLOCKS if title == MV.OURMETHOD][0]
    L = list(HEADER)
    L += mattr_rows([(disp, key) for key, disp in ours], describe_sva)
    L += fixed_rows(SVA_FIXED)
    L.append("\\bottomrule")
    L.append(SVA_FOOTNOTE)
    L += ["\\end{tabular}", "\\end{adjustbox}"]
    return "\n".join(L) + "\n"


def main():
    for fname, build in (("hparams_mib.tex", build_mib), ("hparams_mib_edge.tex", build_mib_edge),
                         ("hparams_sva.tex", build_sva)):
        tex = build()
        (TABS / fname).write_text(tex)
        print(f"-> {TABS / fname}  ({tex.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
