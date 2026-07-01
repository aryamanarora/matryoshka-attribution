"""Paper-ready: all SVA eval metrics vs circuit size, faceted by metric, coloured by method,
one figure per eval direction (iso = keep top-k clean; cause = corrupt top-k).
Reproduce: uv run python plots/sva_eval_metrics.py
"""
import json
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_grid, facet_wrap, labs,
    scale_x_log10, scale_color_brewer, scale_color_manual, scale_linetype_manual, theme_set, theme_bw, theme,
    element_text, element_line, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

RES = "results/sva"
TASK, MODEL = "nounpp", "llama3"
METHODS = [  # (label, file tag) — order = legend/colour order
    ("MAttr suff", "sufficient_hard_topk_adam"),
    ("MAttr suff (idSGD)", "sufficient_hard_topk_identity_sgd"),
    ("MAttr suff (CE)", "sufficient_hard_topk_adam_ce"),
    ("MAttr nec", "necessary_hard_topk_adam"),
    ("MAttr nec (idSGD)", "necessary_hard_topk_identity_sgd"),
    ("MAttr nec (CE)", "necessary_hard_topk_adam_ce"),
    ("RelP", "relp"),
    ("IxG", "ixg"),
    ("IG", "ig"),
]
NODESETS = [("mlp", "MLP"), ("mlp-attn_dim", "MLP + attn")]  # (file key, facet label)
# metric key -> facet title (proper capitalisation)
METRICS = {
    "faithfulness": "Faithfulness (norm. logit diff)",
    "logit_diff": "Logit difference",
    "logit_base": "Logit (base)",
    "logit_source": "Logit (source)",
    "p_base": "P(base)",
    "p_source": "P(source)",
    "ce_base": "Cross-entropy (base)",
    "ce_source": "Cross-entropy (source)",
    "acc_base": "Acc: P(base) > P(source)",
    "acc_source": "Acc: P(source) > P(base)",
}

import math as _math
def _metric_curve(cur, mkey):
    """Return the per-sparsity curve for mkey, deriving CE from P(.) when absent in old JSONs.
    Returns None if unavailable (e.g. logit_base in a pre-metric run)."""
    if mkey in cur:
        return cur[mkey]
    if mkey == "ce_base" and "p_base" in cur:
        return [-_math.log(max(p, 1e-9)) for p in cur["p_base"]]
    if mkey == "ce_source" and "p_source" in cur:
        return [-_math.log(max(p, 1e-9)) for p in cur["p_source"]]
    return None

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
def log_labels(breaks):
    return [f"10{str(int(round(np.log10(b)))).translate(_SUP)}" if b > 0 else "0" for b in breaks]


def make(direction, fname, task, methods, nodesets):
    rows = []
    for label, tag in methods:
        for nkey, nlabel in nodesets:
            fp = f"{RES}/{task}_{MODEL}_{nkey}_{tag}.json"
            try:
                d = json.load(open(fp))
            except FileNotFoundError:
                print("skip (missing):", fp); continue
            cur = d[f"{direction}_metrics"]
            for mkey, mtitle in METRICS.items():
                curve = _metric_curve(cur, mkey)
                if curve is None:
                    continue
                for n, v in zip(d["n_nodes"], curve):
                    rows.append({"method": label, "nodes": nlabel, "n_nodes": n,
                                 "metric": mtitle, "value": v})
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], [m[0] for m in methods])
    df["metric"] = pd.Categorical(df["metric"], list(METRICS.values()))
    p = (ggplot(df, aes("n_nodes", "value", color="method"))
         + geom_hline(yintercept=[0, 1], linetype="dashed", color="#cccccc", size=0.25)
         + geom_line(size=0.5) + geom_point(size=0.4)
         + scale_x_log10(labels=log_labels)
         + scale_color_brewer(type="qual", palette="Set1")
         + labs(x="Circuit size (nodes)", y="Value", color=""))
    n_metrics = df["metric"].nunique()
    if len(nodesets) > 1:  # facet metric x node-set
        df["nodes"] = pd.Categorical(df["nodes"], [n[1] for n in nodesets])
        p = p + facet_grid("metric ~ nodes", scales="free_y") + theme(figure_size=(5.5, 1.2 * n_metrics))
    else:                   # single node set -> wrap metrics (ncol=3, ~1.5in per row)
        nrow = -(-n_metrics // 3)
        p = p + facet_wrap("metric", ncol=3, scales="free_y") + theme(figure_size=(5.5, 1.5 * nrow))
    p.save(fname, verbose=False)
    print("wrote", fname)


# --- SVA (nounpp): 9 methods x {MLP, MLP+attn} ---
make("iso", "plots/sva_eval_metrics_iso.pdf", "nounpp", METHODS, NODESETS)
make("cause", "plots/sva_eval_metrics_cause.pdf", "nounpp", METHODS, NODESETS)

# --- NPI subj-relc: MAttr iso x {logit-diff, logit, CE} + RelP/IxG/IG, MLP only ---
NPI = [
    ("MAttr iso logit-diff", "sufficient_hard_topk_adam"),
    ("MAttr iso logit", "sufficient_hard_topk_adam_logit"),
    ("MAttr iso CE", "sufficient_hard_topk_adam_ce"),
    ("RelP", "relp"), ("IxG", "ixg"), ("IG", "ig"),
]
# same MAttr losses but uniform-k schedule (vs log-uniform default above)
NPI_UNIF = [
    ("MAttr logit-diff (log k)", "sufficient_hard_topk_adam"),
    ("MAttr logit-diff (unif k)", "sufficient_hard_topk_adam_uniformk"),
    ("MAttr logit (log k)", "sufficient_hard_topk_adam_logit"),
    ("MAttr logit (unif k)", "sufficient_hard_topk_adam_logit_uniformk"),
    ("MAttr CE (log k)", "sufficient_hard_topk_adam_ce"),
    ("MAttr CE (unif k)", "sufficient_hard_topk_adam_ce_uniformk"),
]
make("iso", "plots/npi_eval_metrics_iso.pdf", "npi_any_subj-relc", NPI, [("mlp", "MLP")])
make("cause", "plots/npi_eval_metrics_cause.pdf", "npi_any_subj-relc", NPI, [("mlp", "MLP")])

# --- NPI: log-k vs uniform-k for each loss ---
make("iso", "plots/npi_eval_metrics_iso_kshed.pdf", "npi_any_subj-relc", NPI_UNIF, [("mlp", "MLP")])
make("cause", "plots/npi_eval_metrics_cause_kshed.pdf", "npi_any_subj-relc", NPI_UNIF, [("mlp", "MLP")])

# same log-k vs uniform-k comparison but at bs=1
NPI_UNIF_BS1 = [
    ("logit-diff log-k", "sufficient_hard_topk_adam_bs1"),
    ("logit-diff unif-k", "sufficient_hard_topk_adam_uniformk_bs1"),
    ("logit-diff adapt-k", "sufficient_hard_topk_adam_adaptivek_bs1"),
    ("logit log-k", "sufficient_hard_topk_adam_logit_bs1"),
    ("logit unif-k", "sufficient_hard_topk_adam_logit_uniformk_bs1"),
    ("CE log-k", "sufficient_hard_topk_adam_ce_bs1"),
    ("CE unif-k", "sufficient_hard_topk_adam_ce_uniformk_bs1"),
    ("hinge log-k", "sufficient_hard_topk_adam_hinge_bs1"),
    ("hinge adapt-k", "sufficient_hard_topk_adam_hinge_adaptivek_bs1"),
]
make("iso", "plots/npi_eval_metrics_iso_kshed_bs1.pdf", "npi_any_subj-relc", NPI_UNIF_BS1, [("mlp", "MLP")])
make("cause", "plots/npi_eval_metrics_cause_kshed_bs1.pdf", "npi_any_subj-relc", NPI_UNIF_BS1, [("mlp", "MLP")])

# --- NPI: MAttr (log-k) bs=1 vs bs=8 per loss, vs gradient methods (9 lines) ---
NPI_BS = [
    ("MAttr logit-diff bs8", "sufficient_hard_topk_adam"),
    ("MAttr logit-diff bs1", "sufficient_hard_topk_adam_bs1"),
    ("MAttr logit bs8", "sufficient_hard_topk_adam_logit"),
    ("MAttr logit bs1", "sufficient_hard_topk_adam_logit_bs1"),
    ("MAttr CE bs8", "sufficient_hard_topk_adam_ce"),
    ("MAttr CE bs1", "sufficient_hard_topk_adam_ce_bs1"),
    ("RelP", "relp"), ("IxG", "ixg"), ("IG", "ig"),
]
# decision-focused: hinge loss (saturating margin) vs the gap-maximisers + gradients
NPI_HINGE = [  # all bs=1
    ("MAttr hinge", "sufficient_hard_topk_adam_hinge_bs1"),
    ("MAttr prob", "sufficient_hard_topk_adam_prob_bs1"),
    ("MAttr logit-diff", "sufficient_hard_topk_adam_bs1"),
    ("MAttr CE", "sufficient_hard_topk_adam_ce_bs1"),
    ("RelP", "relp"), ("IxG", "ixg"), ("IG", "ig"),
]
make("iso", "plots/npi_eval_metrics_iso_bs.pdf", "npi_any_subj-relc", NPI_BS, [("mlp", "MLP")])
make("cause", "plots/npi_eval_metrics_cause_bs.pdf", "npi_any_subj-relc", NPI_BS, [("mlp", "MLP")])

# --- NPI: hinge (decision-focused) vs gap-maximisers vs gradients ---
make("iso", "plots/npi_eval_metrics_iso_hinge.pdf", "npi_any_subj-relc", NPI_HINGE, [("mlp", "MLP")])
make("cause", "plots/npi_eval_metrics_cause_hinge.pdf", "npi_any_subj-relc", NPI_HINGE, [("mlp", "MLP")])

# --- NPI: 2k vs 8k steps, bs=1, log-k, for logit-diff / CE / prob ---
NPI_8K = [
    ("logit-diff 2k", "sufficient_hard_topk_adam_bs1"),
    ("logit-diff 8k", "sufficient_hard_topk_adam_bs1_s8000"),
    ("CE 2k", "sufficient_hard_topk_adam_ce_bs1"),
    ("CE 8k", "sufficient_hard_topk_adam_ce_bs1_s8000"),
    ("prob 2k", "sufficient_hard_topk_adam_prob_bs1"),
    ("prob 8k", "sufficient_hard_topk_adam_prob_bs1_s8000"),
]
make("iso", "plots/npi_eval_metrics_iso_8k.pdf", "npi_any_subj-relc", NPI_8K, [("mlp", "MLP")])
make("cause", "plots/npi_eval_metrics_cause_8k.pdf", "npi_any_subj-relc", NPI_8K, [("mlp", "MLP")])


# --- NPI: color=method, linetype=2k/8k, with gradient methods (1-shot, dotted) ---
# (base, steps, tag)
NPI_8K_GRAD = [
    ("logit-diff", "2k", "sufficient_hard_topk_adam_bs1"),
    ("logit-diff", "8k", "sufficient_hard_topk_adam_bs1_s8000"),
    ("CE", "2k", "sufficient_hard_topk_adam_ce_bs1"),
    ("CE", "8k", "sufficient_hard_topk_adam_ce_bs1_s8000"),
    ("prob", "2k", "sufficient_hard_topk_adam_prob_bs1"),
    ("prob", "8k", "sufficient_hard_topk_adam_prob_bs1_s8000"),
    ("IG", "1-shot", "ig"),
    ("RelP", "1-shot", "relp"),
    ("IxG", "1-shot", "ixg"),
]


def make_lt(direction, fname, task="npi_any_subj-relc"):
    rows = []
    for base, steps, tag in NPI_8K_GRAD:
        fp = f"{RES}/{task}_{MODEL}_mlp_{tag}.json"
        try:
            d = json.load(open(fp))
        except FileNotFoundError:
            print("skip (missing):", fp); continue
        cur = d[f"{direction}_metrics"]
        for mkey, mtitle in METRICS.items():
            curve = _metric_curve(cur, mkey)
            if curve is None:
                continue
            for n, v in zip(d["n_nodes"], curve):
                rows.append({"method": base, "steps": steps, "n_nodes": n,
                             "metric": mtitle, "value": v})
    df = pd.DataFrame(rows)
    bases = ["logit-diff", "CE", "prob", "IG", "RelP", "IxG"]
    df["method"] = pd.Categorical(df["method"], bases)
    df["steps"] = pd.Categorical(df["steps"], ["2k", "8k", "1-shot"])
    df["metric"] = pd.Categorical(df["metric"], list(METRICS.values()))
    p = (ggplot(df, aes("n_nodes", "value", color="method", linetype="steps"))
         + geom_hline(yintercept=[0, 1], linetype="dashed", color="#cccccc", size=0.25)
         + geom_line(size=0.5)
         + scale_x_log10(labels=log_labels)
         + scale_color_brewer(type="qual", palette="Set1")
         + scale_linetype_manual(values={"2k": "solid", "8k": "dashed", "1-shot": "dotted"})
         + facet_wrap("metric", ncol=3, scales="free_y")
         + labs(x="Circuit size (nodes)", y="Value", color="", linetype="Steps")
         + theme(figure_size=(5.5, 1.5 * -(-df["metric"].nunique() // 3))))
    p.save(fname, verbose=False)
    print("wrote", fname)


make_lt("iso", "plots/npi_eval_metrics_iso_8k_grad.pdf")
make_lt("cause", "plots/npi_eval_metrics_cause_8k_grad.pdf")

# --- NPI: adaptive-k vs log-k (A/B per loss/bs), with IG/RelP reference ---
NPI_ADK = [
    ("hinge bs1 log-k", "sufficient_hard_topk_adam_hinge_bs1"),
    ("hinge bs1 adapt-k", "sufficient_hard_topk_adam_hinge_adaptivek_bs1"),
    ("logit-diff bs1 log-k", "sufficient_hard_topk_adam_bs1"),
    ("logit-diff bs1 adapt-k", "sufficient_hard_topk_adam_adaptivek_bs1"),
    ("logit-diff bs8 log-k", "sufficient_hard_topk_adam"),
    ("logit-diff bs8 adapt-k", "sufficient_hard_topk_adam_adaptivek"),
    ("IG", "ig"), ("RelP", "relp"),
]
make("iso", "plots/npi_eval_metrics_iso_adaptivek.pdf", "npi_any_subj-relc", NPI_ADK, [("mlp", "MLP")])
make("cause", "plots/npi_eval_metrics_cause_adaptivek.pdf", "npi_any_subj-relc", NPI_ADK, [("mlp", "MLP")])

# --- NPI: per-span (mlp_span, variable length) -- CE MAttr + IG/RelP/IxG ---
NPI_SPAN = [
    ("MAttr CE (iso-trained)", "sufficient_hard_topk_adam_ce_bs1"),
    ("MAttr CE (cause-trained)", "necessary_hard_topk_adam_ce_bs1"),
    ("IG", "ig"), ("RelP", "relp"), ("IxG", "ixg"),
]
# all MAttr loss targets on span (iso, bs=1) + gradients
NPI_SPAN_LOSS = [
    ("MAttr acc (T=1)", "sufficient_hard_topk_adam_acc_bs1"),
    ("MAttr acc (T=.5)", "sufficient_hard_topk_adam_acc_bs1_t05"),
    ("MAttr hinge", "sufficient_hard_topk_adam_hinge_bs1"),
    ("MAttr logit-diff", "sufficient_hard_topk_adam_bs1"),
    ("MAttr CE", "sufficient_hard_topk_adam_ce_bs1"),
    ("MAttr prob", "sufficient_hard_topk_adam_prob_bs1"),
    ("IG", "ig"), ("RelP", "relp"), ("IxG", "ixg"),
]
make("iso", "plots/npi_eval_metrics_iso_span.pdf", "npi_any_subj-relc", NPI_SPAN,
     [("mlp_span", "MLP per-span")])
make("cause", "plots/npi_eval_metrics_cause_span.pdf", "npi_any_subj-relc", NPI_SPAN,
     [("mlp_span", "MLP per-span")])
make("iso", "plots/npi_eval_metrics_iso_span_loss.pdf", "npi_any_subj-relc", NPI_SPAN_LOSS,
     [("mlp_span", "MLP per-span")])
make("cause", "plots/npi_eval_metrics_cause_span_loss.pdf", "npi_any_subj-relc", NPI_SPAN_LOSS,
     [("mlp_span", "MLP per-span")])

# --- NPI span: CE iso/cause x log / log_both k-schedule (4 methods) ---
NPI_SPAN_LB = [
    ("CE iso (log)", "sufficient_hard_topk_adam_ce_bs1"),
    ("CE iso (log_both)", "sufficient_hard_topk_adam_ce_logboth_bs1"),
    ("CE cause (log)", "necessary_hard_topk_adam_ce_bs1"),
    ("CE cause (log_both)", "necessary_hard_topk_adam_ce_logboth_bs1"),
]
make("iso", "plots/npi_eval_metrics_iso_span_logboth.pdf", "npi_any_subj-relc", NPI_SPAN_LB,
     [("mlp_span", "MLP per-span")])
make("cause", "plots/npi_eval_metrics_cause_span_logboth.pdf", "npi_any_subj-relc", NPI_SPAN_LB,
     [("mlp_span", "MLP per-span")])

# --- NPI span: CE iso/cause x log/log_both, with OLD (suppress-base) vs FIXED (force-source) cause ---
NPI_SPAN_FIX = [
    ("iso (log)", "sufficient_hard_topk_adam_ce_bs1"),
    ("iso (log_both)", "sufficient_hard_topk_adam_ce_logboth_bs1"),
    ("cause log [suppress]", "necessary_hard_topk_adam_ce_oldsupp_bs1"),
    ("cause log [force-src]", "necessary_hard_topk_adam_ce_bs1"),
    ("cause log_both [suppress]", "necessary_hard_topk_adam_ce_logboth_oldsupp_bs1"),
    ("cause log_both [force-src]", "necessary_hard_topk_adam_ce_logboth_bs1"),
]
make("iso", "plots/npi_eval_metrics_iso_span_cefix.pdf", "npi_any_subj-relc", NPI_SPAN_FIX,
     [("mlp_span", "MLP per-span")])
make("cause", "plots/npi_eval_metrics_cause_span_cefix.pdf", "npi_any_subj-relc", NPI_SPAN_FIX,
     [("mlp_span", "MLP per-span")])

# --- NPI span: 12 variants = loss(CE/LD/acc) x direction(iso/cause) x schedule(log/log_both) ---
# col=loss, color=train direction, linetype=schedule. (loss, direction, schedule, tag)
SPAN12 = [
    ("CE", "iso", "log", "sufficient_hard_topk_adam_ce_bs1"),
    ("CE", "iso", "log_both", "sufficient_hard_topk_adam_ce_logboth_bs1"),
    ("CE", "cause", "log", "necessary_hard_topk_adam_ce_bs1"),
    ("CE", "cause", "log_both", "necessary_hard_topk_adam_ce_logboth_bs1"),
    ("logit-diff", "iso", "log", "sufficient_hard_topk_adam_bs1"),
    ("logit-diff", "iso", "log_both", "sufficient_hard_topk_adam_logboth_bs1"),
    ("logit-diff", "cause", "log", "necessary_hard_topk_adam_bs1"),
    ("logit-diff", "cause", "log_both", "necessary_hard_topk_adam_logboth_bs1"),
    ("acc T=.5", "iso", "log", "sufficient_hard_topk_adam_acc_bs1_t05"),
    ("acc T=.5", "iso", "log_both", "sufficient_hard_topk_adam_acc_logboth_bs1_t05"),
    ("acc T=.5", "cause", "log", "necessary_hard_topk_adam_acc_bs1_t05"),
    ("acc T=.5", "cause", "log_both", "necessary_hard_topk_adam_acc_logboth_bs1_t05"),
]


# gradient reference column: no direction/schedule; colour distinguishes the 3 methods
SPAN12_GRAD = [("Gradient", "IG", "log", "ig"), ("Gradient", "RelP", "log", "relp"),
               ("Gradient", "IxG", "log", "ixg")]
_GRID_COL = {"iso": "#e41a1c", "cause": "#377eb8",         # MAttr train direction
             "IG": "#4daf4a", "RelP": "#984ea3", "IxG": "#ff7f00"}  # gradient methods


def make_grid(direction, fname, task="npi_any_subj-relc"):
    rows = []
    for loss, tdir, sched, tag in SPAN12 + SPAN12_GRAD:
        fp = f"{RES}/{task}_{MODEL}_mlp_span_{tag}.json"
        try:
            d = json.load(open(fp))
        except FileNotFoundError:
            print("skip (missing):", fp); continue
        cur = d[f"{direction}_metrics"]
        for mkey, mtitle in METRICS.items():
            curve = _metric_curve(cur, mkey)
            if curve is None:
                continue
            for n, v in zip(d["n_nodes"], curve):
                rows.append({"loss": loss, "dir": tdir, "sched": sched, "n_nodes": n,
                             "metric": mtitle, "value": v})
    df = pd.DataFrame(rows)
    df["loss"] = pd.Categorical(df["loss"], ["CE", "logit-diff", "acc T=.5", "Gradient"])
    df["dir"] = pd.Categorical(df["dir"], ["iso", "cause", "IG", "RelP", "IxG"])
    df["sched"] = pd.Categorical(df["sched"], ["log", "log_both"])
    df["metric"] = pd.Categorical(df["metric"], list(METRICS.values()))
    p = (ggplot(df, aes("n_nodes", "value", color="dir", linetype="sched"))
         + geom_hline(yintercept=[0, 1], linetype="dashed", color="#cccccc", size=0.25)
         + geom_line(size=0.5)
         + scale_x_log10(labels=log_labels)
         + scale_color_manual(values=_GRID_COL, name="Train / method")
         + scale_linetype_manual(values={"log": "solid", "log_both": "dashed"})
         + facet_grid("metric ~ loss", scales="free_y")
         + labs(x="Circuit size (nodes)", y="Value", linetype="k-sched")
         + theme(figure_size=(6.8, 1.15 * df["metric"].nunique())))
    p.save(fname, verbose=False)
    print("wrote", fname)


make_grid("iso", "plots/npi_eval_metrics_iso_span12.pdf")
make_grid("cause", "plots/npi_eval_metrics_cause_span12.pdf")


# --- NPI mlp+attn_head_span: acc loss x {iso,cause,joint} x {Adam, SGD+id-STE} + gradients ---
# color = mode/method, linetype = optimizer.
ACC_OPT = [
    ("iso", "Adam", "sufficient_hard_topk_adam_acc_bs1_t05"),
    ("iso", "SGD+id-STE", "sufficient_hard_topk_identity_sgd_acc_bs1_t05"),
    ("cause", "Adam", "necessary_hard_topk_adam_acc_bs1_t05"),
    ("cause", "SGD+id-STE", "necessary_hard_topk_identity_sgd_acc_bs1_t05"),
    ("joint", "Adam", "joint_hard_topk_adam_acc_bs1_t05"),
    ("joint", "SGD+id-STE", "joint_hard_topk_identity_sgd_acc_bs1_t05"),
    ("IG", "gradient", "ig"), ("RelP", "gradient", "relp"), ("IxG", "gradient", "ixg"),
]
_ACC_COL = {"iso": "#e41a1c", "cause": "#377eb8", "joint": "#4daf4a",
            "IG": "#984ea3", "RelP": "#ff7f00", "IxG": "#a65628"}


def make_acc_opt(direction, fname, task="npi_any_subj-relc"):
    rows = []
    for grp, opt, tag in ACC_OPT:
        fp = f"{RES}/{task}_{MODEL}_mlp-attn_head_span_{tag}.json"
        try:
            d = json.load(open(fp))
        except FileNotFoundError:
            print("skip (missing):", fp); continue
        cur = d[f"{direction}_metrics"]
        for mkey, mtitle in METRICS.items():
            curve = _metric_curve(cur, mkey)
            if curve is None:
                continue
            for n, v in zip(d["n_nodes"], curve):
                rows.append({"grp": grp, "opt": opt, "n_nodes": n, "metric": mtitle, "value": v})
    df = pd.DataFrame(rows)
    df["grp"] = pd.Categorical(df["grp"], ["iso", "cause", "joint", "IG", "RelP", "IxG"])
    df["opt"] = pd.Categorical(df["opt"], ["Adam", "SGD+id-STE", "gradient"])
    df["metric"] = pd.Categorical(df["metric"], list(METRICS.values()))
    nrow = -(-df["metric"].nunique() // 3)
    p = (ggplot(df, aes("n_nodes", "value", color="grp", linetype="opt"))
         + geom_hline(yintercept=[0, 1], linetype="dashed", color="#cccccc", size=0.25)
         + geom_line(size=0.5)
         + scale_x_log10(labels=log_labels)
         + scale_color_manual(values=_ACC_COL, name="mode / method")
         + scale_linetype_manual(values={"Adam": "solid", "SGD+id-STE": "dashed", "gradient": "dotted"})
         + facet_wrap("metric", ncol=3, scales="free_y")
         + labs(x="Circuit size (nodes)", y="Value", linetype="optimizer")
         + theme(figure_size=(6.5, 1.5 * nrow)))
    p.save(fname, verbose=False)
    print("wrote", fname)


make_acc_opt("iso", "plots/npi_eval_metrics_iso_accopt.pdf")
make_acc_opt("cause", "plots/npi_eval_metrics_cause_accopt.pdf")

# --- NPI SAE: acc-iso, MLP-out vs resid SAE x {Adam, SGD+id-STE} ---
SAE_METHODS = [
    ("Adam", "sufficient_hard_topk_adam_acc_bs1_t05"),
    ("SGD+id-STE", "sufficient_hard_topk_identity_sgd_acc_bs1_t05"),
]
SAE_NODESETS = [("mlp_sae_span", "MLP-out SAE"), ("resid_sae_span", "resid SAE")]
make("iso", "plots/npi_eval_metrics_iso_sae.pdf", "npi_any_subj-relc", SAE_METHODS, SAE_NODESETS)
make("cause", "plots/npi_eval_metrics_cause_sae.pdf", "npi_any_subj-relc", SAE_METHODS, SAE_NODESETS)

# --- NPI acc-iso: neuron vs DAS-64 vs SAE, MLP-out & resid (5 node families, overlaid) ---
NODE5 = [  # (label, nodeset key)
    ("neuron (MLP)", "mlp_span"),
    ("DAS-64 MLP-out", "das_mlp_span"),
    ("DAS-64 resid", "das_resid_span"),
    ("SAE MLP-out", "mlp_sae_span"),
    ("SAE resid", "resid_sae_span"),
]
def make_node5(direction, fname, task="npi_any_subj-relc",
               tag="sufficient_hard_topk_adam_acc_bs1_t05"):
    rows = []
    for label, nkey in NODE5:
        try:
            d = json.load(open(f"{RES}/{task}_{MODEL}_{nkey}_{tag}.json"))
        except FileNotFoundError:
            print("skip:", nkey); continue
        cur = d[f"{direction}_metrics"]
        for mkey, mtitle in METRICS.items():
            curve = _metric_curve(cur, mkey)
            if curve is None:
                continue
            for n, v in zip(d["n_nodes"], curve):
                rows.append({"method": label, "n_nodes": n, "metric": mtitle, "value": v})
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], [m[0] for m in NODE5])
    df["metric"] = pd.Categorical(df["metric"], list(METRICS.values()))
    nrow = -(-df["metric"].nunique() // 3)
    p = (ggplot(df, aes("n_nodes", "value", color="method"))
         + geom_hline(yintercept=[0, 1], linetype="dashed", color="#cccccc", size=0.25)
         + geom_line(size=0.5) + geom_point(size=0.35)
         + scale_x_log10(labels=log_labels)
         + scale_color_brewer(type="qual", palette="Set1")
         + facet_wrap("metric", ncol=3, scales="free_y")
         + labs(x="Circuit size (nodes)", y="Value", color="")
         + theme(figure_size=(6.5, 1.5 * nrow)))
    p.save(fname, verbose=False); print("wrote", fname)

make_node5("iso", "plots/npi_eval_metrics_iso_node5.pdf")
make_node5("cause", "plots/npi_eval_metrics_cause_node5.pdf")

# --- NPI DAS-64: MLP-out vs resid x {acc, CE} loss ---
DAS_METHODS = [("acc", "sufficient_hard_topk_adam_acc_bs1_t05"),
               ("CE", "sufficient_hard_topk_adam_ce_bs1")]
DAS_NODESETS = [("das_mlp_span", "DAS MLP-out"), ("das_resid_span", "DAS resid")]
make("iso", "plots/npi_eval_metrics_iso_das.pdf", "npi_any_subj-relc", DAS_METHODS, DAS_NODESETS)
make("cause", "plots/npi_eval_metrics_cause_das.pdf", "npi_any_subj-relc", DAS_METHODS, DAS_NODESETS)
