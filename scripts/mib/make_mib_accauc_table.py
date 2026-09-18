"""MIB validation table with acc-AUC (log-weighted decision accuracy in [0,1]) instead of CPR.
Mirrors mib_results.tex (node level): gradient baselines + all MAttr variants (Adam/SGD, log-k
default + '+unif k' ablations). Swept methods (hard/soft log, uniform) at lr=0.05.

acc_auc sources: baselines -> the method's ORDINARY eval dir, falling back to a legacy *_accauc
re-eval only for methods scored before acc_auc existed; MAttr -> results/mattr_accauc_val/
(eval-only re-eval of each variant's circuit). 2 decimals; llama cells daggered (n=200 cap).

ADDING A NEW BASELINE: list its ordinary eval dir in BASELINES and stop. run_evaluation.py has
stored acc_auc in every pkl it writes since MIB_circuit_track/evaluation.py started returning
it, so acc-AUC comes free with the CPR run and NO separate acc-AUC pass needs to be launched.

Run from repo root:  uv run python scripts/mib/make_mib_accauc_table.py  ->  paper/tabs/mib_accauc_results.tex
"""
import pickle
from pathlib import Path
import make_mib_table as M   # reuse COLUMNS + node OUR_METHODS + unifk/opt_of conventions
from learning_to_attribute.deps import mib_results_dir

# The MIB fork's results tree (deps/MIB-circuit-track/results on the juice3 checkout, the
# legacy Tilde path elsewhere). Was hardcoded to Tilde until 2026-09-16, which on sc made every
# row read from here (all gradient baselines, the 10 legacy MAttr ablations) silently vanish.
MIB = mib_results_dir()
L2A = Path("results")
MATTR_ACC = MIB / "mattr_accauc"       # lr01 ablations (acc_auc already computed)
MATTR_REEVAL = MIB / "mattr_accauc_val"  # htk_lr_0.05, final_node (re-eval)
OUTPUT = Path("paper/tabs/mib_accauc_results.tex")
COLUMNS = M.COLUMNS
# M's, not a second copy -- same reasoning as opt_of below. This set decides which rows get
# the n=200 dagger, so a copy that misses a newly swept dir does not just look different from
# the CPR table, it silently drops a caveat that table carries.
IOI_LLAMA_CAPPED = M.IOI_LLAMA_CAPPED
# Dirs whose acc_auc lives in the eval_mib validation pkl (results/<dir>/<task>_<model>_
# validation.pkl) rather than in a run_evaluation.py re-eval folder.
#
# mib_node_topk_uniform_lr05 was MISSING here, and that is why the "+ unif k" row showed only
# its three gemma2 cells: unrouted dirs fall through to MATTR_ACC, and mattr_accauc holds
# nothing for this variant except the three pkls the TL 2.15.4 gemma re-eval left behind.
# Nothing needed evaluating -- all 11 acc_auc values were already on disk, just not looked at.
#
# Root cause was a half-finished repoint, not a missing run: make_mib_table.py:73 moved the
# "+ unif k" row from final_node to the lr=0.05 dir when submit_softuni_lr05.sh produced it,
# and this module's routing was left describing the old dir (final_node is still named in
# REEVAL_DIRS below, where it is now dead -- it is no longer in OUR_METHODS at all).
#
# Safe to reroute rather than a change of measurement: on the three cells present in BOTH
# sources the values agree to 2dp (ioi 0.403/0.40, mcqa 0.485/0.48, arc_easy 0.478/0.48).
#
# The two SGD optimizer ablations (submit_softlog_sgd_lr.sh, submit_softuni_sgd_lr.sh). Both
# are new enough that their eval_mib pkls carry acc_auc directly, so they belong here and NOT
# in the legacy fallback -- an unrouted dir falls through to MATTR_ACC, which holds nothing for
# it, and the row would render all-dashes forever while the numbers sat in results/. That is
# the failure the "+ unif k" note above describes, so listing the dir when the sweep is
# submitted (not when someone notices the blank row) is the habit that avoids repeating it.
#
# THIS SET IS KEYED BY DIR, SO IT MOVES WHEN A ROW IS REPOINTED. When the SGD rows went from
# the matched lr=0.05 to their own optima (make_mib_table.py OUR_METHODS), the old
# softlog_sgd_lr_0.05 entry stopped matching and BOTH SGD rows silently dropped out of this
# table while staying in the CPR one. The `SKIP row ... no acc_auc yet` warning in main() is
# what surfaced it; keep that warning, it is the only thing standing between a repoint and two
# tables that disagree about which rows exist.
#
# Renamed off "LR05_" because it no longer holds only lr=0.05 dirs -- see the same rename of
# make_mib_table.IOI_LLAMA_CAPPED.
EVALMIB_ACC = {"htklog_lr_0.05", "topklog_lr_0.05", "mib_node_topk_uniform_lr05",
               "softlog_sgd_lr_1.0", "softuni_sgd_lr_3.0",
               "mib_node_topk_uniform_lr05_eps1e-2", "mib_node_topk_uniform_lr05_5k",
               "mib_node_topk_uniform_frozen", "mib_node_topk_log_frozen",
               "mib_node_topkid_uniform_frozen", "mib_node_topkid_log_frozen"}
# htk_lr_0.05 predates evaluation.py returning acc_auc, so its eval_mib pkl has acc_auc=None
# and it genuinely needs the re-eval folder. final_node is vestigial (see above); it is kept
# only so the entry does not have to be re-derived if that row is ever restored.
REEVAL_DIRS = {"htk_lr_0.05", "final_node"}

# (display, [results dirs, PRIMARY FIRST], method_saveable). The primary is the same ordinary
# eval dir make_mib_table.EXTRA_NODE_BASELINES reads for CPR, so the two tables describe the
# same run; the trailing `*_accauc` entries are legacy re-evaluations kept only because those
# methods were scored before evaluation.py returned acc_auc and their original pkls have
# acc_auc=None. Per cell, the first dir that yields a value wins.
#
# Mixing the two is safe and was checked rather than assumed: over every cell present in both,
# the max |CPR AUC difference| between a method's _eval and its _accauc rerun is 0.02
# (RelP 0.009, IxG 0.002, RelP+QK 0.020, RelP+Shapley 0.000, Conductance 0.008), i.e.
# run-to-run nondeterminism and not a different evaluation setting.
#
# AttnLRP and GIM have NO legacy dir on purpose -- they postdate the change, so their ordinary
# eval dir already carries acc_auc and adding a `*_accauc` rerun for them would be wasted GPU.
# (There IS a gim_nomlp_accauc on disk; it belongs to the pre-scale_mlp_gate GIM and must not
# be listed here, or a corrected-GIM row would silently fill from the buggy run.)
BASELINES = [
    ("NAP-IG", ["napig_ref_eval", "napig_ref_accauc"], "EAP-IG-inputs_patching_node"),
    # ig-steps 10 / 30, mirroring make_mib_table.NAPIG_STEP_ROWS -- see the long note there for
    # why the shipped 5-step default is not a converged integral. Both tables must carry the
    # rows or they contradict each other: acc-AUC is where the 5-step run looks WORST (four
    # cells pinned at the ~0.05 floor, vs 0.43--0.49 at 10 steps), so a CPR table that shows a
    # competitive NAP-IG next to an acc-AUC table that still shows it collapsed would read as
    # the metric disagreeing when it is only the step count.
    #
    # Single dir each, no legacy `*_accauc` fallback: these runs postdate evaluation.py
    # returning acc_auc, so their ordinary eval pkls already carry it -- no extra GPU pass.
    # (Confirmed, not assumed: the 10-step pkls have non-null acc_auc for every finished cell.)
    ("$+$ 10 IG steps", ["napig10_eval"], "EAP-IG-inputs_patching_node"),
    ("$+$ 30 IG steps", ["napig30_eval"], "EAP-IG-inputs_patching_node"),
    # Expected Gradients: alpha ~ U(0,1) drawn per example at m=1 instead of a fixed grid. Same
    # integral, unbiased at every m, and at m=1 it costs exactly what I x G costs -- so it is a
    # rung of the ladder above in QUALITY while sitting at the bottom of it in COST, which is
    # the whole reason it is worth a row. Dir is MIB-side only (napig_mc_eval, 12/12 validation
    # cells); the subfolder is EAP-IG-inputs-mc_patching_node, NOT the grid arms' folder --
    # run_napig_mc.sh passes --method EAP-IG-inputs-mc precisely so it cannot overwrite them.
    ("Expected Gradients", ["napig_mc_eval"], "EAP-IG-inputs-mc_patching_node"),
    ("Conductance", ["napig_local_eval", "napig_local_accauc"], "EAP-IG-inputs-local_patching_node"),
    ("I$\\times$G", ["ig1_eval", "ig1_accauc"], "EAP-IG-inputs_patching_node"),
    ("RelP", ["relp_eval", "relp_accauc"], "RelP_patching_node"),
    ("RelP+QK", ["relp_qkgrad_eval", "relp_qkgrad_accauc"], "RelP-qkgrad_patching_node"),
    ("RelP+Shapley", ["relpshapley_eval", "relpshapley_accauc"], "RelPShapley_patching_node"),
    ("AttnLRP", ["attnlrp_eval"], "AttnLRP_patching_node"),
    ("GIM", ["gim_eval"], "GIM_patching_node"),
]
NODE_METHODS = [(n, r, g) for n, r, l, g in M.OUR_METHODS if l == "node"]   # (name, dir, group)
EDGE_METHODS = [(n, r, g) for n, r, l, g in M.OUR_METHODS if l == "edge"]

# EDGE-LEVEL BASELINES, added 2026-09-02. There was no edge section here until then and the
# reason was never that the numbers did not exist -- run_evaluation.py has stored acc_auc in
# every pkl it writes for years, so all ten of our edge dirs and all three gradient baselines
# were sitting at 11/11 coverage while this table showed node only. The CPR table has always
# had both levels; this one silently did not, which reads as "acc-AUC has nothing to say about
# edges" rather than "nobody wrote the section".
#
# ROW SET follows make_mib_table's EDGE section, plus Expected Gradients. The extra row is this
# table's own precedent, not an invention: the node section here already carries a Expected Gradients
# row that the node CPR table does not, on the argument that at m=1 it costs what I x G costs
# while scoring like the grid arms -- and eapig_mc_eval is 11/11 at edge too. Edge Pruning has
# no edge-level results at all (eprun_eval_*/EdgePruning_patching_edge is empty), so UGS is the
# only mask learner here, at the 3 cells it can run (docs/ugs_baseline.md).
EDGE_BASELINES_ACC = [
    ("EAP-IG-inp (CF, repro)", ["eapig_clean_eval"], "EAP-IG-inputs_patching_edge"),
    ("$+$ 10 IG steps", ["eapig_clean10_eval"], "EAP-IG-inputs_patching_edge"),
    ("Expected Gradients", ["eapig_mc_eval"], "EAP-IG-inputs-mc_patching_edge"),
]
UGS_ACC = (M.UGS_DIR, "UGS_patching_edge")
# EVERY EDGE DIR READS acc_auc STRAIGHT OUT OF ITS eval_mib PKL, like the EVALMIB_ACC node dirs
# and unlike the older node dirs that need a MATTR_ACC re-eval folder. Membership rather than a
# hand-listed set: the edge sweep postdates evaluation.py returning acc_auc, so every dir in
# OUR_METHODS at edge level qualifies by construction and a new one cannot be forgotten -- which
# is exactly how the SGD rows fell out of this table once before (see EVALMIB_ACC's note).
EDGE_DIRS = {r for _, r, _ in EDGE_METHODS}

# Mask-learning baselines (own header). UGS is edge-only so it cannot appear in this
# node-level table at all; Node Pruning runs at node level on every model. The budget, not the
# ranking, is what a mask learner actually optimizes, so it is a reported setting rather than a
# hidden default -- but the full M.EPRUN_SPARSITIES sweep is twelve near-identical rows, so as
# in the CPR table (M.EPRUN_SHOW) we show the best budget per objective and no more.
#
# *** THE PICK IS M.EPRUN_SHOW's, READ FROM THERE RATHER THAN RESTATED. *** This module used to
# keep its own ({s=0.99, s=0.99_ld}, the acc-AUC argmax) on the argument that each table should
# name the budget best under the metric it reports. Each half of that was defensible; together
# they put two identically-named "Node Pruning" rows one page apart that were different runs,
# and once make_mib_table dropped s= from the label (it now prints "(KL)" / "(LD)") the reader
# had no way to tell. Both tables are s=0.95 as of 2026-09-02 -- see M.EPRUN_SHOW for why that
# budget and not either metric's argmax. Importing the dict means they cannot drift again.
#
# WHAT THIS COSTS HERE, since the number moved in the baseline's favour and should not be
# quoted as a method result: Node Pruning's acc-AUC row goes 0.46 -> 0.46 (KL, a 2dp tie with
# s=0.99) and 0.38 -> 0.36 (LD). CPR and acc-AUC rank the budgets in near-opposite orders --
# CPR rewards a bigger circuit, the same gap-padding sensitivity that motivated reporting
# acc-AUC in the first place -- so a shared budget is necessarily off-argmax under one of them.
MASK_BASELINES = [(M.eprun_label("node", M.EPRUN_SHOW["node"][d]), L2A / d, "EdgePruning_patching_node")
                  for _, d in M.EPRUN_SPARSITIES if d in M.EPRUN_SHOW["node"]]
# Edge-level Edge Pruning (submit_eprun_edge_sc.sh), the level's own pick from the same dict.
EDGE_MASK_BASELINES = [(M.eprun_label("edge", M.EPRUN_SHOW["edge"][d]), L2A / d, "EdgePruning_patching_edge")
                       for _, d in M.EPRUN_SPARSITIES if d in M.EPRUN_SHOW["edge"]]
# Single-node interchange intervention, IntInv (M.ACTPATCH_NODE_ROWS): eval_mib-format pkls.
CAUSAL_BASELINES = M.ACTPATCH_NODE_ROWS
# DBM (pyvene sigmoid mask) is in the CPR table's mask block via M.SIGMOID_MASK_ROWS but was
# missing here, even though its eval pkls carry acc_auc like every other run -- so the acc-AUC
# table was silently comparing MAttr against a smaller set of mask learners than the CPR table.
# Its dirs are also written by run_evaluation.py under EdgePruning_patching_node (same code
# path, only the mask parameterization differs), hence the same sub-folder name.
MASK_BASELINES += [(disp, L2A / d, "EdgePruning_patching_node")
                   for disp, d in M.SIGMOID_MASK_ROWS]


# opt_of is M's, not a local copy. There WAS a local one here that only matched "identity",
# and it went stale the moment make_mib_table's grew a second SGD arm (softlog_sgd_*): this
# table filed that run under the "\ourmethod{}-Adam" header while the CPR table had it under
# SGD -- two tables making contradictory claims about which optimizer a run used. The module
# docstring already says this file mirrors make_mib_table's conventions; importing the
# function is what makes that true instead of aspirational.
opt_of = M.opt_of


def _acc(p):
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb")).get("acc_auc")
        return round(v, 2) if v is not None else None
    except Exception:
        return None


# L2A/results and MIB-circuit-track/results are mirrors of the same run_evaluation.py outputs,
# but neither is complete: some dirs (e.g. the quarantined _stale_tl321 wave) exist only in L2A,
# and a freshly finished job lands MIB-side until it is copied over -- napig_ref_eval and
# eapig_clean_eval both originate there. Searching both is what makes "just add the eval dir" true
# regardless of which side a run happens to be on.
ROOTS = (L2A, MIB)


def acc_base(dirs, sub, t, m):
    """acc_auc for one cell, from the first (root, dir) that has a value.

    Returns None only if NO listed dir has this cell, so a half-copied mirror or a legacy dir
    that never covered a cell degrades to a dash rather than to a wrong number.
    """
    fn = f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl"
    for d in dirs:
        for root in ROOTS:
            v = _acc(root / d / sub / fn)
            if v is not None:
                return v
    return None


def acc_mattr(dir_, t, m):
    if dir_ in EVALMIB_ACC or dir_ in EDGE_DIRS:       # acc_auc straight from the eval_mib pkl
        return _acc(L2A / dir_ / f"{t}_{m}_validation.pkl")
    base = MATTR_REEVAL if dir_ in REEVAL_DIRS else MATTR_ACC
    v = _acc(base / f"{dir_}_patching_node" / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")
    if v is None:
        # Cells evaluated after evaluation.py started returning acc_auc carry it in their own
        # eval_mib pkl and were never mirrored into the re-eval folders (the arithmetic_addition
        # cells of 2026-09-15, the gemma2 cells of the TL 2.15.4 re-eval). Same value, same
        # protocol; the mirror is just where the OLDER cells of these dirs had to be re-scored.
        v = _acc(L2A / dir_ / f"{t}_{m}_validation.pkl")
    return v


def fmt(v, bold=False, underline=False, dagger=False, color=None):
    """One table cell, matching make_mib_table.fmt -- MATH mode, dagger inside the same group,
    \\cellcolor leading. The heat ramp itself is M.cell_color, imported rather than restated so
    the two tables cannot end up on different scales."""
    if v is None:
        return "---"
    s = f"{v:.2f}"
    if bold:
        s = f"\\mathbf{{{s}}}"
    elif underline:
        s = f"\\underline{{{s}}}"
    if dagger:
        s = "^{\\dagger}" + s
    s = f"${s}$"
    return f"\\cellcolor[HTML]{{{color}}}{s}" if color else s


def row_avg(data):
    vs = [v for v in (data.get((t, m)) for t, m, _ in COLUMNS) if v is not None]
    return round(sum(vs) / len(vs), 2) if vs else None


def mark_k(name, group):
    """k-schedule mark for a row of OUR_METHODS group `group`, from the shared headline
    definition (scripts/mib/mattr_variants.py) -- the headline schedule's rows stay unmarked."""
    return M.MV.mark_k(name, M.MV.GROUP_K[group])


def collect(level):
    """[(header, [(display, data, dagger_cells)])] for one level, in render order."""
    llama = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
    if level == "node":
        grad = []
        for disp, d, sub in BASELINES:
            data = {(t, m): acc_base(d, sub, t, m) for t, m, _ in COLUMNS}
            # Same warning make_mib_table prints. A partial row shows no Avg (see `full`), so it
            # cannot masquerade as a finished one -- but silence at the terminal is how a row
            # stays half-empty for a week without anyone noticing.
            n = sum(v is not None for v in data.values())
            if 0 < n < len(COLUMNS):
                print(f"  NOTE {disp} ({level}): {n}/{len(COLUMNS)} cells ({'/'.join(d)})"
                      " -- still running")
            grad.append((disp, data, llama))
        mask = []
        for disp, base, sub in MASK_BASELINES:
            data = {(t, m): _acc(base / sub / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")
                    for t, m, _ in COLUMNS}
            if any(v is not None for v in data.values()):
                mask.append((disp, data, llama))
        causal = []
        for disp, d in CAUSAL_BASELINES:
            data = {(t, m): _acc(L2A / d / f"{t}_{m}_validation.pkl") for t, m, _ in COLUMNS}
            if any(v is not None for v in data.values()):
                causal.append((disp, data, set()))
    else:
        grad = []
        for disp, dirs, sub in EDGE_BASELINES_ACC:
            data = {(t, m): acc_base(dirs, sub, t, m) for t, m, _ in COLUMNS}
            n = sum(v is not None for v in data.values())
            if 0 < n < len(COLUMNS):
                print(f"  NOTE {disp} ({level}): {n}/{len(COLUMNS)} cells -- still running")
            grad.append((disp, data, llama))
        d, sub = UGS_ACC
        ugs = {(t, m): _acc(L2A / d / sub / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")
               for t, m, _ in COLUMNS}
        mask = [("UGS", ugs, llama)] if any(v is not None for v in ugs.values()) else []
        for disp, base, sub in EDGE_MASK_BASELINES:
            data = {(t, m): _acc(base / sub / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")
                    for t, m, _ in COLUMNS}
            if any(v is not None for v in data.values()):
                mask.append((disp, data, llama))
        causal = []
    return grad, mask, causal


def main():
    ncols = len(COLUMNS)
    L = ["\\begin{adjustbox}{max width=\\textwidth}",
         # CENTRED, not right-aligned: \cellcolor paints the whole cell including \tabcolsep,
         # so a right-aligned number sits hard against the right edge of its own swatch. Same
         # change and same reason as make_mib_table's column spec.
         "\\begin{tabular}{l" + "c" * ncols + "@{\\quad}c}", "\\toprule",
         "& \\multicolumn{4}{c}{IOI} & \\multicolumn{2}{c}{Arithmetic} & \\multicolumn{3}{c}{MCQA} & "
         "\\multicolumn{2}{c}{ARC (E)} & ARC (C) & \\\\",
         "\\cmidrule(lr){2-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-12} \\cmidrule(lr){13-13}",
         "\\textbf{Method} & " + " & ".join(h for _, _, h in COLUMNS) + " & \\textbf{Avg} \\\\"]

    for li, (level, title, methods) in enumerate(
            [("node", "Node-level, acc-AUC", NODE_METHODS),
             ("edge", "Edge-level, acc-AUC", EDGE_METHODS)]):
        grad, mask, causal = collect(level)
        mattr = {d: {(t, m): acc_mattr(d, t, m) for t, m, _ in COLUMNS} for _, d, _ in methods}
        all_data = [dd for _, dd, _ in grad + mask + causal] + list(mattr.values())
        if not any(any(v is not None for v in dd.values()) for dd in all_data):
            print(f"SKIP {level} section: no acc_auc anywhere")
            continue

        # *** BEST / SECOND / HEAT ARE ALL PER LEVEL. *** Node acc-AUC tops out near 0.60 and
        # edge near 0.96 on the same column, so a shared scale would bold an edge cell in every
        # column and shade the whole node section white. Same reason make_mib_table computes
        # best_in_col per level rather than once.
        best, second, crange = {}, {}, {}
        for t, m, _ in COLUMNS:
            vals = sorted({dd[(t, m)] for dd in all_data if dd.get((t, m)) is not None},
                          reverse=True)
            best[(t, m)] = vals[0] if vals else None
            second[(t, m)] = vals[1] if len(vals) > 1 else None
            crange[(t, m)] = (min(vals), max(vals)) if vals else None

        # An average over a subset of columns is not comparable to one over all 11, so rows with
        # missing cells neither print an average, compete for the bolded best, nor stretch the
        # heat scale.
        def full(dd):
            return all(dd.get((t, m)) is not None for t, m, _ in COLUMNS)

        avs = sorted({a for a in (row_avg(dd) for dd in all_data if full(dd)) if a is not None},
                     reverse=True)
        abest, asec = (avs[0] if avs else None), (avs[1] if len(avs) > 1 else None)
        crange["avg"] = (min(avs), max(avs)) if avs else None

        def emit(disp, data, dcells, _b=best, _s=second, _c=crange, _f=full,
                 _ab=abest, _as=asec):
            cells = []
            for t, m, _ in COLUMNS:
                v = data.get((t, m))
                is_best = v is not None and v == _b[(t, m)]
                cells.append(fmt(v, bold=is_best,
                                 underline=(v is not None and not is_best
                                            and v == _s[(t, m)]),
                                 dagger=((t, m) in dcells and v is not None),
                                 color=M.cell_color(v, _c[(t, m)])))
            a = row_avg(data) if _f(data) else None
            cells.append(fmt(a, bold=(a is not None and a == _ab),
                             underline=(a is not None and a != _ab and a == _as),
                             color=M.cell_color(a, _c["avg"])))
            return f"\\quad {disp} & " + " & ".join(cells) + " \\\\"

        L.append("\\midrule")
        L.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        if grad:
            L.append("\\textbf{Gradient attribution} \\\\")
            L += [emit(d, dd, dc) for d, dd, dc in grad]
        if mask:
            L.append("\\textbf{Mask learning} \\\\")
            L += [emit(d, dd, dc) for d, dd, dc in mask]
        if causal:
            L.append("\\textbf{Interchange intervention} \\\\")
            L += [emit(d, dd, dc) for d, dd, dc in causal]

        llama_ioi = {("ioi", "llama3")}
        edge_llama = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
        # Headline optimizer block first and unmarked, headline k-schedule rows first and
        # unmarked within it -- the same order make_mib_table.emit_ours takes from
        # scripts/mib/mattr_variants.py, so the two tables cannot disagree about the headline.
        for opt, label in M.MV.OPT_ORDER:
            by_group = {grp: [(n, d) for n, d, g in methods if g == grp and opt_of(d) == opt]
                        for grp in M.MV.GROUP_K}
            if not any(by_group.values()):
                continue
            L.append(f"\\textbf{{{label}}} \\\\")
            for grp in M.MV.GROUP_ORDER:
                for n, d in by_group[grp]:
                    disp = mark_k(n, grp)
                    if not any(v is not None for v in mattr[d].values()):
                        # Same rule as make_mib_table's emit_ours: a row of 12 "---" claims a run
                        # that was scored and produced nothing, which is a wrong statement rather
                        # than a blank. Drop it until its first cell lands; it reappears on the
                        # next regeneration with no edit here. Announced, never silent.
                        print(f"SKIP row {disp!r} ({d}): no acc_auc yet")
                        continue
                    # Node ours are capped on ioi/llama3 only (IOI_LLAMA_CAPPED); every edge
                    # llama3 cell is --head 200, as in make_mib_table's EDGE_LLAMA_DAGGER.
                    dc = edge_llama if level == "edge" else (
                        llama_ioi if d in IOI_LLAMA_CAPPED else set())
                    L.append(emit(disp, mattr[d], dc))

    L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUTPUT} ({len(NODE_METHODS)} node + {len(EDGE_METHODS)} edge MAttr rows)")


if __name__ == "__main__":
    main()
