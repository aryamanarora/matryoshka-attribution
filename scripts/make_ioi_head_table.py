"""Top-10 and bottom-10 nodes by attribution for EVERY MIB cell, with the heads of the
published IOI circuit colour-coded by their role in the one cell that has such a circuit.

The MIB analogue of make_sva_neuron_table.py: same question ("which units does each method
reach for FIRST, and which does it rule out?"), same ranking convention, same chip-colouring
idea -- but the highlight means something stronger here. In the SVA table a colour marks a
neuron that HAPPENS to recur across cells, which is an internal consistency check with no
ground truth behind it. Here a colour marks a head that Wang et al. identified as part of the
IOI circuit by path patching, so the chips are an EXTERNAL reference.

Run:  uv run python scripts/make_ioi_head_table.py  ->  paper/tabs/ioi_top_nodes.tex

*** COLOURS APPEAR IN THE ioi/gpt2 BLOCK ONLY, AND THAT IS NOT AN OMISSION. ***
ioi/gpt2 is the only MIB cell with a published, head-level, role-annotated circuit to check
against. The other ten have no role taxonomy: MIB ships a reference graph only for InterpBench
(run_evaluation.py:42), not for the real task/model cells, so there is nothing to colour with.
The head indices themselves are gpt2-specific, so they are not carried over to ioi/qwen2.5,
ioi/gemma2 or ioi/llama3 either -- `a9.h9` in Llama-3 is a different head in a different model
and chipping it would fabricate the reference this table exists to check against.

The uncoloured blocks are still worth printing: they show WHERE in the model each method looks
(early MLPs vs late heads, `input` vs computation) and how much the methods agree with each
other, both of which are readable without a ground-truth circuit.

RANKING. Raw score, sorted DESCENDING -- not |score|. Same reason as the SVA table, and here
it is pinned to MIB's own code: Graph.apply_topn (EAP-IG/src/eap/graph.py:491) does
`torch.argsort(node_score_copy.view(-1), descending=True)` and every pkl the paper reads is
`abs-False`, so this really is the order nodes enter each circuit and drives the left end of
every CPR curve. Ranking by |score| would show a different set than our own numbers came from.

Note what descending-raw does at the BOTTOM: a head that pushes the metric the wrong way lands
at rank n, not at rank 1. Every gradient method puts the two negative name movers (10.7, 11.10)
in the last two slots of the ioi/gpt2 block for exactly this reason. They are published circuit
members ranked dead last, which is a property of the ranking convention and not a method
failure -- do not read those two chips as misses.

WHAT COUNTS AS A NODE. Every entry of the circuit's `nodes` dict that carries a score --
attention heads, MLPs, and `input`. `logits` is excluded because it has no score (it is
`in_graph` only). MLPs and `input` are real nodes and are shown, just never chipped.

WHY THE BOTTOM 10 TOO. The top block asks "what does the method reach for first?"; the bottom
block asks the sharper question, "what does it rule out?" A coloured chip down there is a
method placing a path-patched IOI head among the least important nodes in the model, which is
a substantive disagreement with the reference rather than a missed hit -- and unlike a top-10
miss it cannot be explained away by `input`/`m0` occupying slots.

NO SUMMARY ROWS. Counting hits per column invited exactly the reading the counts cannot
support: methods differ in whether they rank `input` and the early MLPs highly (MAttr puts m0
and input 2nd and 3rd; the mask learners put `input` first), so an all-nodes hit count scores
that as worse head recovery when it is not, and the head-only correction needed its own row to
say so. The ranks themselves are the evidence; the reader can see the chips.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mib_table as M   # noqa: E402  single source of truth for the 11-cell column set

RESULTS_MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")
RESULTS_L2A = Path("results")
TABDIR = Path("paper/tabs")
TOPN = 10

# One column per method, left to right. Three shapes of source file, all with the same
# {'nodes': {name: {'score': float, ...}}} payload, so one loader reads them; they differ only
# in how the cell is spelled into the path, hence a formatter per method rather than a dir.
#   - MIB run_attribution.py output:  <dir>/<task-with-hyphens>_<model>/importances.json
#   - L2A mask output:                <dir>/graph_<task>_<model>.json
#   - L2A MAttr output:               <dir>/<task>_<model>_importances.json
# The gradient set is the one the MIB node table compares (minus RelP+QK and RelP+Shapley,
# which would take the table to ten columns without adding a distinct story -- both are RelP
# variants and both are in tabs/mib_results.tex already).
def _mib(dirn, sub):
    return lambda t, m: RESULTS_MIB / dirn / sub / f"{t.replace('_', '-')}_{m}" / "importances.json"


def _graph(dirn):
    return lambda t, m: RESULTS_L2A / dirn / f"graph_{t}_{m}.json"


def _mattr(dirn):
    return lambda t, m: RESULTS_L2A / dirn / f"{t}_{m}_importances.json"


METHODS = [
    ("NAP-IG",      _mib("napig_ref", "EAP-IG-inputs_patching_node")),
    (r"I$\times$G", _mib("ig1", "EAP-IG-inputs_patching_node")),
    ("RelP",        _mib("relp", "RelP_patching_node")),
    ("AttnLRP",     _mib("attnlrp", "AttnLRP_patching_node")),
    ("GIM",         _mib("gim", "GIM_patching_node")),
    # Node Pruning at the single config the test table and figures show (logit-diff, s=0.5),
    # per the comment on HEADLINE_EPRUN in make_mib_table.py -- not the KL s=0.9 run, which
    # optimises something other than what CPR measures. (The acc-AUC table names a different
    # budget; this table shows circuits, not either metric, so it follows the headline config.)
    ("Node Pruning", _graph("eprun_node_s0.5_ld")),
    # pyvene sigmoid mask, the same lr/L1 config make_mib_table.py's SIGMOID_MASK_ROWS names.
    ("DBM",         _graph("eprun_node_ld_sig_lr0.3_l16.0")),
    # MAttr headline = soft top-k forward, log-k, lr=0.05 (CLAUDE.md's results-dir table).
    (r"\ourmethod{}", _mattr("topklog_lr_0.05")),
]

# Cells in the same order as every other MIB table, so a reader can move between them without
# re-sorting. Imported rather than retyped: a cell added to make_mib_table appears here too.
CELLS = M.COLUMNS
TASK_NAME = {"ioi": "IOI", "arithmetic_subtraction": "Arithmetic (subtraction)",
             "mcqa": "MCQA", "arc_easy": "ARC (easy)", "arc_challenge": "ARC (challenge)"}
MODEL_NAME = {"gpt2": "GPT-2", "qwen2.5": "Qwen-2.5", "gemma2": "Gemma-2", "llama3": "Llama-3"}
# The one cell the IOI reference applies to. Everything about the chips keys off this.
CHIPPED_CELL = ("ioi", "gpt2")

# The IOI circuit of Wang et al., Figure 2 (wang2022interpretability), transcribed head-for-head
# including the heads that figure draws in parentheses as minor members (0.10, 5.8, 5.9) -- they
# are part of the published circuit and dropping them would quietly inflate every denominator.
# Names are MIB's node naming, a<layer>.h<head>.
#
# Colours are lifted from that figure's own boxes so the table and the figure can be read
# together without a translation step: previous-token cream, duplicate-token pink, induction
# yellow, S-inhibition periwinkle, name-mover mint, negative-name-mover peach, backup green.
# They are pastels for the same reason the SVA palette is: hyperref renders the link text on
# top of them in darkblue (colorlinks=true), and a saturated chip buries it.
CLASSES = [
    ("prev", "Previous token",      "F5E3C3", ["a2.h2", "a4.h11"]),
    ("dup",  "Duplicate token",     "F7CDE4", ["a0.h1", "a3.h0", "a0.h10"]),
    ("ind",  "Induction",           "FBF3C4", ["a5.h5", "a6.h9", "a5.h8", "a5.h9"]),
    ("sinh", "S-inhibition",        "D8DDF0", ["a7.h3", "a7.h9", "a8.h6", "a8.h10"]),
    ("nm",   "Name mover",          "CDEBDC", ["a9.h9", "a9.h6", "a10.h0"]),
    ("neg",  "Negative name mover", "FBDCC4", ["a10.h7", "a11.h10"]),
    ("bnm",  "Backup name mover",   "DFEDC8", ["a9.h0", "a9.h7", "a10.h1", "a10.h2",
                                               "a10.h6", "a10.h10", "a11.h2", "a11.h9"]),
]
ROLE = {h: key for key, _, _, heads in CLASSES for h in heads}
N_IOI = len(ROLE)

HEAD_RE = re.compile(r"a(\d+)\.h(\d+)$")


def load_ranking(path):
    """[node name] sorted by descending raw score, for one method's circuit.

    Filters entries without a 'score' key rather than assuming a fixed node set: `logits`
    carries only `in_graph`, and a KeyError there would be a crash rather than a wrong table,
    but the filter also survives any future node type that is structural-only.
    """
    nodes = json.load(open(path))["nodes"]
    scored = [(k, v["score"]) for k, v in nodes.items()
              if isinstance(v, dict) and "score" in v]
    scored.sort(key=lambda kv: -kv[1])
    return [k for k, _ in scored]


def fmt_node(name):
    """MIB's `a9.h9` / `m0` / `input` as it should be typeset."""
    m = HEAD_RE.match(name)
    if m:
        return r"%s.%s" % (m.group(1), m.group(2))     # 9.9 -- the IOI paper's own notation
    if name == "input":
        return r"\textit{input}"
    return name                                        # m0, m1, ...


def main():
    # rankings[(task, model)][label]; a cell is emitted only if every method has a circuit for
    # it, since a block with a blank column invites reading a missing file as "found nothing".
    rankings, missing = {}, []
    for task, model, _ in CELLS:
        per_cell = {}
        for label, pathf in METHODS:
            p = pathf(task, model)
            if not p.exists():
                missing.append(f"{task}/{model} {label}: {p}")
                continue
            per_cell[label] = load_ranking(p)
        if per_cell:
            rankings[(task, model)] = per_cell
    if missing:
        print("MISSING circuits:\n  " + "\n  ".join(missing), file=sys.stderr)
    if not rankings:
        sys.exit("no circuits found")

    # Sanity, per cell: every method must rank the same node set, or the "top 10 of the same n"
    # framing is false and the columns are not comparable. Checked WITHIN a cell only -- node
    # counts legitimately differ ACROSS cells (157 for gpt2, 235 gemma2, 361 qwen2.5, 1057
    # llama3), which is exactly why the rank labels are computed per block.
    n_nodes = {}
    for cell, per_cell in rankings.items():
        sets = {lab: frozenset(r) for lab, r in per_cell.items()}
        if len(set(sets.values())) != 1:
            ref = next(iter(sets.values()))
            for lab, s in sets.items():
                if s != ref:
                    print(f"  ! {cell} {lab} node set differs: +{sorted(s - ref)[:5]} "
                          f"-{sorted(ref - s)[:5]}", file=sys.stderr)
            sys.exit("node sets differ across methods -- refusing to write a table that implies "
                     "they are the same ranking problem")
        n_nodes[cell] = len(next(iter(sets.values())))

    cols = [(lab, f) for lab, f in METHODS
            if any(lab in pc for pc in rankings.values())]
    ncol = len(cols)
    # Same width arithmetic as make_sva_neuron_table: tabcolsep (3pt) lands on both sides of
    # 2*ncol interior gaps, the label column takes a fixed 0.14, and the method columns share
    # the rest. Derived rather than hardcoded so adding a column cannot silently overflow.
    LABW = 0.14
    W = (1.0 - 2 * ncol * 3.0 / 397.0 - LABW) / ncol
    # Node ids are short here (`10.10` at gpt2, `31.31` at llama3 -- 5 chars either way) where
    # SVA's were 12, so the same >=65pt threshold that forced \scriptsize there is not the
    # binding constraint -- but keep the rule rather than a new constant, so the two tables stay
    # typographically matched.
    font = r"\small" if W * 397 >= 65 else r"\scriptsize"
    col = r">{\raggedright\arraybackslash}p{%.3f\textwidth}" % W
    hdr = ["& " + " & ".join(r"\textbf{%s}" % lab for lab, _ in cols) + r" \\", r"\midrule"]

    def chip(cell, name):
        """Role colour, but only in the cell the IOI reference actually describes."""
        body = fmt_node(name)
        if cell == CHIPPED_CELL and name in ROLE:
            return r"\colorbox{ioi%s}{%s}" % (ROLE[name], body)
        return body

    # longtable, not tabular: eleven blocks of 21 rows is several pages, and a tabular that does
    # not fit overflows off the bottom of the page silently rather than breaking. That also
    # means sections/task_analysis.tex must keep \input-ing this at top level -- a longtable
    # inside a table float is an error. longtable centres itself by default (\LTleft/\LTright
    # are \fill), which is where the horizontal centring comes from; do not wrap it in center.
    # The method header repeats on every continuation page (\endhead) since a reader landing
    # mid-table otherwise cannot tell which column is which.
    L = [r"% Requires \usepackage{booktabs,longtable,colortbl,array}; \input at top level "
         r"(NOT inside a table float).",
         r"% Generated by scripts/make_ioi_head_table.py -- do not edit by hand.",
         *[r"\definecolor{ioi%s}{HTML}{%s}" % (key, hexv) for key, _, hexv, _ in CLASSES],
         "{" + font,
         r"\setlength{\tabcolsep}{3pt}",
         # \colorbox pads 3pt by default, which would shove the chips into the neighbouring
         # column and open the line spacing inside every cell.
         r"\setlength{\fboxsep}{1pt}",
         r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{longtable}{@{}p{%.2f\textwidth} *{%d}{%s}@{}}" % (LABW, ncol, col),
         r"\toprule", *hdr, r"\endfirsthead",
         r"\toprule", *hdr, r"\endhead",
         r"\bottomrule"]

    # Legend in \endlastfoot, not \endfoot: repeating it under every page would cost a chunk of
    # the vertical space an eleven-block table has none of. The notation gloss is not optional:
    # heads print in the IOI paper's own `layer.head` form so the table can be read against its
    # Figure 2, but the same columns also hold `m5` and `input`, and next to those an entry like
    # `0.4` or `6.0` reads as a decimal number rather than as head 4 of layer 0.
    legend = r"\quad ".join(
        r"\colorbox{ioi%s}{\strut~}~%s" % (key, name) for key, name, _, _ in CLASSES)
    legend = (r"Attention heads are \emph{layer}.\emph{head}; $\mathrm{m}\ell$ is the layer-$\ell$ "
              r"MLP. Colours mark the IOI-circuit role assigned by \citet{wang2022interpretability} "
              r"and so appear in the IOI\,/\,GPT-2 block only, the one cell with a published "
              r"role-annotated circuit:~" + legend)
    L += [r"\multicolumn{%d}{@{}p{0.97\textwidth}@{}}{\scriptsize %s} \\" % (ncol + 1, legend),
          r"\endlastfoot"]

    row = 0   # global, so the alternating shade does not reset at a block boundary and read
              # as the start of a new table
    for b_i, (task, model, _) in enumerate(CELLS):
        cell = (task, model)
        if cell not in rankings:
            continue
        n = n_nodes[cell]
        if b_i:
            L.append(r"\addlinespace[3pt]")
        # \\* forbids a page break directly after the block header, so a cell name can never be
        # orphaned at the foot of a page from the rows it labels.
        L.append(r"\multicolumn{%d}{@{}l}{\textbf{%s\,/\,%s} \textnormal{\scriptsize (%d nodes)}} \\*[2pt]"
                 % (ncol + 1, TASK_NAME[task], MODEL_NAME[model], n))
        # Two blocks of TOPN rows over the SAME ranking, labelled by absolute rank so the
        # elision between them is unambiguous: 1..10 and (n-9)..n out of n.
        for i in list(range(TOPN)) + [None] + list(range(n - TOPN, n)):
            shade = r"\rowcolor[HTML]{F7F7F7}" if row % 2 else ""
            row += 1
            if i is None:
                # Elided middle. \vdots goes in EVERY column, not just the label: a rule alone
                # would read as a block separator, whereas dots across the full width say "each
                # of these rankings continues".
                L.append(f"{shade}$\\vdots$ & " + " & ".join([r"$\vdots$"] * ncol) + r" \\")
                continue
            cells = [chip(cell, rankings[cell][lab][i]) if lab in rankings[cell] else ""
                     for lab, _ in cols]
            L.append(f"{shade}{i + 1}. & " + " & ".join(cells) + r" \\")

    L += [r"\end{longtable}", "}"]

    TABDIR.mkdir(parents=True, exist_ok=True)
    out = TABDIR / "ioi_top_nodes.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}  ({ncol} methods x {len(rankings)} cells, top/bottom {TOPN}, "
          f"{N_IOI} IOI heads in the reference)")
    # Stdout only -- deliberately NOT rows in the table (see the docstring). Printed anyway
    # because it is the cheapest way to notice a column that has gone wrong or stale.
    if CHIPPED_CELL in rankings:
        for lab, _ in cols:
            r = rankings[CHIPPED_CELL].get(lab)
            if not r:
                continue
            top = sum(1 for x in r[:TOPN] if x in ROLE)
            bot = sum(1 for x in r[-TOPN:] if x in ROLE)
            print(f"  ioi/gpt2 {lab:22s} IOI heads in top{TOPN}={top:2d}  in bottom{TOPN}={bot:2d}")


if __name__ == "__main__":
    main()
