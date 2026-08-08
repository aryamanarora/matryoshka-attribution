"""Top-10 and bottom-10 nodes by attribution for the IOI / GPT-2 MIB cell, with the heads of
the published IOI circuit colour-coded by their role.

The MIB analogue of make_sva_neuron_table.py: same question ("which units does each method
reach for FIRST?"), same ranking convention, same chip-colouring idea -- but the highlight
means something stronger here. In the SVA table a colour marks a neuron that HAPPENS to
recur across cells, which is an internal consistency check with no ground truth behind it.
Here a colour marks a head that Wang et al. identified as part of the IOI circuit by
path patching, so the chips are an EXTERNAL reference: a method whose top-10 is all coloured
has rediscovered a hand-derived circuit, and one whose top-10 is bare has not.

Run:  uv run python scripts/make_ioi_head_table.py  ->  paper/tabs/ioi_top_nodes.tex

WHY THIS CELL AND NO OTHER. ioi/gpt2 is the only MIB cell with a published, head-level,
role-annotated circuit to check against. The other ten cells have MIB's own curated circuits
but no role taxonomy, so there would be nothing to colour with. This is a one-cell table by
necessity, which is why it is a tabular (floatable) rather than the SVA table's longtable.

RANKING. Raw score, sorted DESCENDING -- not |score|. Same reason as the SVA table, and here
it is pinned to MIB's own code: Graph.apply_topn (EAP-IG/src/eap/graph.py:491) does
`torch.argsort(node_score_copy.view(-1), descending=True)` and every pkl the paper reads is
`abs-False`, so this really is the order nodes enter each circuit and drives the left end of
every CPR curve. Ranking by |score| would show a different set than our own numbers came from.

WHAT COUNTS AS A NODE. Every entry of the circuit's `nodes` dict that carries a score --
attention heads, MLPs, and `input`. `logits` is excluded because it has no score (it is
`in_graph` only). MLPs and `input` are real nodes and are shown, just never chipped: the IOI
taxonomy is head-level and inventing an MLP role would be fabricating the reference we are
checking against.

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

RESULTS_MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")
RESULTS_L2A = Path("results")
TABDIR = Path("paper/tabs")
TOPN = 10

# One column per method, left to right. Two shapes of source file, both with the same
# {'nodes': {name: {'score': float, ...}}} payload, so they are read by one loader:
#   - MIB run_attribution.py output:  <dir>/<Method>_patching_node/ioi_gpt2/importances.json
#   - L2A mask/MAttr output:          <dir>/ioi_gpt2_importances.json or graph_ioi_gpt2.json
# The gradient set is the one the MIB node table compares (minus RelP+QK and RelP+Shapley,
# which would take the table to ten columns without adding a distinct story -- both are RelP
# variants and both are in tabs/mib_results.tex already).
METHODS = [
    ("NAP-IG",  RESULTS_MIB / "napig_ref/EAP-IG-inputs_patching_node/ioi_gpt2/importances.json"),
    (r"I$\times$G", RESULTS_MIB / "ig1/EAP-IG-inputs_patching_node/ioi_gpt2/importances.json"),
    ("RelP",    RESULTS_MIB / "relp/RelP_patching_node/ioi_gpt2/importances.json"),
    ("AttnLRP", RESULTS_MIB / "attnlrp/AttnLRP_patching_node/ioi_gpt2/importances.json"),
    ("GIM",     RESULTS_MIB / "gim/GIM_patching_node/ioi_gpt2/importances.json"),
    # Node Pruning at the single config the test table and figures show (logit-diff, s=0.5),
    # per the comment on HEADLINE_EPRUN in make_mib_table.py -- not the KL s=0.9 run, which
    # optimises something other than what CPR measures.
    ("Node Pruning", RESULTS_L2A / "eprun_node_s0.5_ld/graph_ioi_gpt2.json"),
    # pyvene sigmoid mask, the same lr/L1 config make_mib_table.py's SIGMOID_MASK_ROWS names.
    ("DBM",     RESULTS_L2A / "eprun_node_ld_sig_lr0.3_l16.0/graph_ioi_gpt2.json"),
    # MAttr headline = soft top-k forward, log-k, lr=0.05 (CLAUDE.md's results-dir table).
    (r"\ourmethod{}", RESULTS_L2A / "topklog_lr_0.05/ioi_gpt2_importances.json"),
]

# The IOI circuit of Wang et al., Figure 2 (wang2022interpretability), transcribed head-for-head including the
# heads that figure draws in parentheses as minor members (0.10, 5.8, 5.9) -- they are part of
# the published circuit and dropping them would quietly inflate every "of 26" denominator.
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
    rankings, missing = {}, []
    for label, path in METHODS:
        if not path.exists():
            missing.append(f"{label}: {path}")
            continue
        rankings[label] = load_ranking(path)
    if missing:
        print("MISSING circuits:\n  " + "\n  ".join(missing), file=sys.stderr)
    cols = [(lab, p) for lab, p in METHODS if lab in rankings]
    if not cols:
        sys.exit("no circuits found")

    # Sanity: every method must rank the same node set, or the "top 10 of the same 157" framing
    # is false and the counts below are not comparable across columns. gpt2-small is
    # 1 input + 12*12 heads + 12 MLPs = 157.
    sets = {lab: frozenset(r) for lab, r in rankings.items()}
    if len(set(sets.values())) != 1:
        ref = sets[cols[0][0]]
        for lab, s in sets.items():
            if s != ref:
                print(f"  ! {lab} node set differs: +{sorted(s - ref)[:5]} "
                      f"-{sorted(ref - s)[:5]}", file=sys.stderr)
        sys.exit("node sets differ across methods -- refusing to write a table that implies "
                 "they are the same ranking problem")
    n_nodes = len(next(iter(sets.values())))

    ncol = len(cols)
    # Same width arithmetic as make_sva_neuron_table: tabcolsep (3pt) lands on both sides of
    # 2*ncol interior gaps, the label column takes a fixed 0.14, and the method columns share
    # the rest. Derived rather than hardcoded so adding a column cannot silently overflow.
    LABW = 0.14
    W = (1.0 - 2 * ncol * 3.0 / 397.0 - LABW) / ncol
    # Node ids are short here (`10.10` is the widest, 5 chars) where SVA's were 12, so the
    # same >=65pt threshold that forced \scriptsize there is not the binding constraint -- but
    # keep the rule rather than a new constant, so the two tables stay typographically matched.
    font = r"\small" if W * 397 >= 65 else r"\scriptsize"
    col = r">{\raggedright\arraybackslash}p{%.3f\textwidth}" % W
    hdr = ("& " + " & ".join(r"\textbf{%s}" % lab for lab, _ in cols) + r" \\")

    def chip(name):
        body = fmt_node(name)
        return r"\colorbox{ioi%s}{%s}" % (ROLE[name], body) if name in ROLE else body

    L = [r"% Requires \usepackage{booktabs,colortbl,array}. Floatable (tabular, not longtable).",
         r"% Generated by scripts/make_ioi_head_table.py -- do not edit by hand.",
         *[r"\definecolor{ioi%s}{HTML}{%s}" % (key, hexv) for key, _, hexv, _ in CLASSES],
         "{" + font,
         r"\setlength{\tabcolsep}{3pt}",
         # \colorbox pads 3pt by default, which would shove the chips into the neighbouring
         # column and open the line spacing inside every cell.
         r"\setlength{\fboxsep}{1pt}",
         r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{tabular}{@{}p{%.2f\textwidth} *{%d}{%s}@{}}" % (LABW, ncol, col),
         r"\toprule", hdr, r"\midrule"]

    # Two blocks of TOPN rows over the SAME ranking, labelled by absolute rank so the elision
    # between them is unambiguous: 1..10 and (n-9)..n out of n_nodes. Row parity is carried by
    # the absolute row index rather than the within-block one, so the alternating shade does not
    # reset and read as a new table at the join.
    ranks = list(range(TOPN)) + [None] + list(range(n_nodes - TOPN, n_nodes))
    for row, i in enumerate(ranks):
        shade = r"\rowcolor[HTML]{F7F7F7}" if row % 2 else ""
        if i is None:
            # Elided middle. \vdots goes in EVERY column, not just the label: a rule alone would
            # read as the section break the table already uses before the legend, whereas dots
            # across the full width say "each of these rankings continues".
            L.append(f"{shade}$\\vdots$ & " + " & ".join([r"$\vdots$"] * ncol) + r" \\")
            continue
        cells = [chip(rankings[lab][i]) if i < len(rankings[lab]) else "" for lab, _ in cols]
        # Faint alternating shade: ncol near-identical short ids per row are easy to slip a
        # row on, exactly as in the SVA table.
        L.append(f"{shade}{i + 1}. & " + " & ".join(cells) + r" \\")

    # Legend. It sits inside the tabular as a full-width row so it travels with the table
    # wherever the float lands, rather than as free text that can drift away from it.
    L.append(r"\midrule")
    # The notation gloss is not optional: heads print in the IOI paper's own `layer.head` form
    # so the table can be read against its Figure 2, but the same columns also hold `m5` and
    # `input`, and next to those an entry like `0.4` or `6.0` reads as a decimal number rather
    # than as head 4 of layer 0. One clause removes the ambiguity.
    legend = r"\quad ".join(
        r"\colorbox{ioi%s}{\strut~}~%s" % (key, name) for key, name, _, _ in CLASSES)
    legend = (r"Attention heads are \emph{layer}.\emph{head}; $\mathrm{m}\ell$ is the layer-$\ell$ "
              r"MLP. Colours mark the IOI-circuit role assigned by \citet{wang2022interpretability}:~"
              + legend)
    L.append(r"\multicolumn{%d}{@{}p{0.97\textwidth}@{}}{\scriptsize %s} \\" % (ncol + 1, legend))
    L += [r"\bottomrule", r"\end{tabular}", "}"]

    TABDIR.mkdir(parents=True, exist_ok=True)
    out = TABDIR / "ioi_top_nodes.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}  ({ncol} methods, top/bottom {TOPN} of {n_nodes} nodes, "
          f"{N_IOI} IOI heads in the reference)")
    # Stdout only -- deliberately NOT rows in the table (see the docstring). Printed anyway
    # because it is the cheapest way to notice a column that has gone wrong or stale.
    for lab, _ in cols:
        r = rankings[lab]
        top = sum(1 for x in r[:TOPN] if x in ROLE)
        bot = sum(1 for x in r[-TOPN:] if x in ROLE)
        print(f"  {lab:22s} IOI heads in top{TOPN}={top:2d}  in bottom{TOPN}={bot:2d}")


if __name__ == "__main__":
    main()
