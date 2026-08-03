"""Top-5 MLP neurons by attribution, per SVA subtask x method.

Reads the per-unit score tensors the SVA sweep already writes
(results/sva_sweep/<task>_llama3_mlp_<tag>.scores.pt) and reports, for each subtask and
method, the five neurons each method ranks FIRST -- i.e. the first units it puts into the
circuit.

DEFAULT LAYOUT IS COMPACT (one page): identity + position/score only. `--descriptions` adds
each neuron's top positive and negative description from Transluce, which is a much richer
table but runs to four pages -- one per subtask -- because a described cell is 6-8 typeset
lines instead of 2. Every neuron id is a hyperlink to Transluce either way, so the compact
table does not lose access to the descriptions, only their inlining.

Run:  uv run python scripts/make_sva_neuron_table.py   ->  paper/tabs/sva_top_neurons.tex
      (--descriptions for the 4-page version; --no-fetch renders from cache only, e.g. offline)

WHY llama3-only: the `mlp` substrate was only ever swept on llama3 (the other MIB models are
node-level), so there is exactly one model here and no cross-model column to add.

RANKING. Raw score, sorted DESCENDING -- not |score|. That is not a stylistic choice: it is
what evaluate.sparsity_sweep does (`flat.argsort(descending=True)`), so these really are the
units that enter each method's circuit at the smallest budget and drive the left end of every
faithfulness curve in the paper. Ranking by |score| here would show a DIFFERENT set of neurons
than the ones our own numbers were computed from, for IG/IxG especially, whose scores are
signed effects rather than importances.

NEURON vs UNIT. The substrate is per-(layer, position, neuron) -- 32 x 6 x 14336 = 2752512
units -- but the question is about neurons, and one neuron can occupy several of the top
slots at different positions. So we deduplicate to distinct (layer, neuron) keeping each
neuron's best-scoring position, and report that position. "Top 5" therefore means 5 distinct
neurons, which is usually deeper into the raw ranking than slot 5.

*** Transluce descriptions are for Llama-3.1-8B-INSTRUCT; our runs use the BASE model
    (meta-llama/Llama-3.1-8B, see eval_sva.MODEL_FULLNAMES). ***
Identical architecture and identical neuron indexing, so index i means the same slot in both,
but instruction tuning updates the MLP weights -- a described neuron is evidence about what
that slot computes, not proof about what it computes in our model. Read the descriptions as
indicative. This caveat belongs in the caption; do not drop it.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fingerprint_tables import parse_method  # noqa: E402  (one naming source of truth)

RES = Path("results/sva_sweep")
TABDIR = Path("paper/tabs")
CACHE = Path("results/.transluce_cache.json")
MODEL, SUBSTRATE, TOPN = "llama3", "mlp", 5

# Transluce's neuron-data server. sign=+/- selects the positive/negative activation direction;
# the response's explanation_summary is a [description, score] list already sorted best-first.
API = "https://transluce--neuron-data-server-fastapi-app.modal.run/read_specific_file"

TASKS = [("simple", "Simple"), ("nounpp", "Noun PP"),
         ("rc", "RC"), ("within_rc", "Within RC")]
# Methods and the order they appear. Keys are parse_method()'s outputs; the tag suffix picks
# the loss, and no suffix = logit-diff, the loss every headline SVA number in the paper uses.
# Same four series as plot_accauc_vs_faithauc's FIGURE_METHODS, so this table and that figure
# describe the same runs.
METHODS = [("IG", "ig"), ("IxG", "ixg"), ("eprun-s090", "eprun_s090"),
           ("stopk-log", "sufficient_topk_adam_bs1"),
           ("stopk-unif", "sufficient_topk_adam_uniformk_bs1")]
LABELS = {"IG": "IG", "IxG": r"I$\times$G", "eprun-s090": "Node Pruning",
          "stopk-log": r"\ourmethod{}", "stopk-unif": r"\ourmethod{} $+$ unif $k$"}
# Truncation budget per description. The layout is one column per METHOD, so this shrinks with
# the number of methods: five columns across \textwidth leave ~0.176\textwidth each, i.e. ~68pt,
# i.e. ~15 characters per typeset line at \small -- so 45 wraps to about three lines.
DESC_CHARS = 45
# Transluce's neuron browser, the same URL scheme tabs/arith_mlp_neuron_table.tex links to.
NEURON_URL = "https://neurons.transluce.org/%d/%d/+"

# Neurons that surface in many (subtask, method) cells get a categorical highlight, so that
# "these two columns keep picking the SAME unit" is visible at a glance instead of something
# you verify by reading 100 six-digit ids. Colour = identity, nothing else: it does not encode
# rank, score or count.
#
# The threshold is 4 of the 20 cells because the recurrence distribution has a clean gap there
# -- 8 neurons appear 4-7 times, then it drops straight to 2 with nothing at 3 -- so this is
# reading a break in the data, not imposing a cutoff. It also happens to land exactly on the
# palette size. If a rerun changes that, main() says so rather than silently recolouring.
RECUR_MIN = 4
# ColorBrewer Pastel1, with two substitutions made after looking at a rendered page. Pastels
# because the link text sits ON these and hyperref renders it darkblue (colorlinks=true in the
# preamble), so saturated chips would bury it -- but pastel has a floor: Pastel1's FFFFCC and
# F2F2F2 are so close to white that a chip in them reads as no chip at all, and on an F7F7F7
# shaded row it disappears outright. FFFFCC -> a yellow with enough body to survive the
# shading, F2F2F2 was already unused, and FDDAEC (pink) -> teal, since against FBB4AE (salmon)
# it was the one pair that needed a second look. Hues are spread so that the three most
# frequent neurons -- which take slots 0-2 -- land on red/blue/green.
PALETTE = ["FBB4AE", "B3CDE3", "CCEBC5", "DECBE4",
           "FED9A6", "8DD3C7", "E8DE6B", "DCC49A"]


def load_run(task, tag):
    """(scores tensor, meta dict) for one cell, or (None, None) if that run is missing."""
    stem = RES / f"{task}_{MODEL}_{SUBSTRATE}_{tag}"
    pt, js = Path(str(stem) + ".scores.pt"), Path(str(stem) + ".json")
    if not pt.exists() or not js.exists():
        return None, None
    meta = json.load(open(js))
    # Guard the identity of the run rather than trusting the filename: parse_method is the
    # repo's naming authority and a tag that no longer maps to the expected key means the
    # sweep was renamed under us, which would silently mislabel every row below.
    got = parse_method(pt.name.replace(".scores.pt", ".json"), meta)
    return torch.load(pt, map_location="cpu", weights_only=False), (meta | {"_method": got})


def decode(idx, meta):
    """Flat unit index -> (layer, position, neuron).

    Byte-for-byte the same arithmetic as LlamaAttributionHooks.decode_index (models/llama.py,
    the `mask_type == "mlp"` branch), which is the authority for this layout; it is duplicated
    rather than imported because decode_index is an instance method and instantiating the
    hooker would mean loading 8B of weights just to divide two integers. It also matches the
    writer side, eval_sva.gradient_scores: `off = li * P * N` then a [P, N] block reshaped
    row-major, i.e. index = layer*(P*N) + pos*N + neuron.

    Getting this wrong is the failure mode with no symptom -- every description would attach
    to the wrong neuron and the table would still look entirely plausible -- so main() asserts
    the shape identity L*P*N == total == numel before trusting it.

    That assert only pins the shape, so the ORDERING was checked separately against a
    signature a transposed decode could not reproduce: under this decode, positions 0 and 1
    hold *exactly* 0.0 IG score in all four subtasks (BOS and the first token are identical in
    the clean and patch prompts, so g.(clean - patch) is identically zero there), while the
    top-200 units concentrate at pos 2 and pos P-1 -- the subject and the verb, which is where
    subject-verb agreement lives. Note P is per-subtask (simple 3, nounpp 6, rc 7,
    within_rc 6), read from each run's json rather than assumed.
    """
    N, P = meta["intermediate_size"], meta["seq_len"]
    layer, rem = divmod(int(idx), P * N)
    pos, neuron = divmod(rem, N)
    return layer, pos, neuron


def top_neurons(scores, meta, n=TOPN):
    """Top-n DISTINCT (layer, neuron) by descending raw score, best position kept."""
    order = torch.argsort(scores, descending=True)
    out, seen = [], set()
    for idx in order.tolist():
        layer, pos, neuron = decode(idx, meta)
        if (layer, neuron) in seen:
            continue
        seen.add((layer, neuron))
        out.append(dict(layer=layer, neuron=neuron, pos=pos, score=float(scores[idx])))
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------- Transluce descriptions
# Cache format version. v1 stored only the single best description per (layer, neuron, sign);
# v2 stores the full ranked candidate list, which is what makes the renderability fallback in
# describe() possible. A v1 file on disk is discarded rather than misread.
CACHE_V = 2
_raw = json.load(open(CACHE)) if CACHE.exists() else {}
_cache = _raw.get("data", {}) if _raw.get("v") == CACHE_V else {}

# pdflatex-renderability. The paper builds with pdfTeX and loads neither inputenc nor fontenc
# (checked in iclr2026_conference.log), so a Cyrillic or CJK codepoint in a description is not
# a cosmetic issue -- it is "Unicode character ... not set up for use with LaTeX" and a failed
# Overleaf build. Many descriptions quote non-Latin activating tokens, so this is not rare.
_PUNCT = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "--",
          "—": "---", "…": "...", " ": " ", "→": "->", "·": "."}


def normalize(s):
    for a, b in _PUNCT.items():
        s = s.replace(a, b)
    return s


def renderable(s):
    """True if pdflatex can typeset s with this preamble (ASCII + Latin-1/Extended-A)."""
    return all(ord(c) < 0x180 for c in normalize(s))


def describe(layer, neuron, sign, fetch=True):
    """Best-scoring RENDERABLE description for one (layer, neuron, sign), or None.

    The API returns five candidate descriptions ranked by score, so when the top one quotes a
    non-Latin token we fall back to the next renderable candidate instead of mangling the text
    with elisions. Only if all five are unrenderable do we give up and return None -- that
    loses one cell rather than corrupting every cell that mentions a Russian token.

    Cached failures are stored and NOT retried -- a neuron with no description is a fact about
    the database, not a transient error, and re-requesting the whole missing set on every
    render would hammer a service we do not own.
    """
    key = f"{layer}/{neuron}/{sign}"
    if key not in _cache:
        if not fetch:
            return None
        url = f"{API}?" + urllib.parse.urlencode(
            {"layer": layer, "neuron": neuron, "sign": sign})
        cands = []
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    cands = json.load(r).get("explanation_summary") or []
                break
            except Exception as e:                   # noqa: BLE001 - network, any failure retries
                if attempt == 2:
                    print(f"  ! {key}: {type(e).__name__} {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        _cache[key] = cands
    return next((normalize(c[0]) for c in _cache[key] if c and c[0] and renderable(c[0])), None)


def save_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"v": CACHE_V, "data": _cache}, open(CACHE, "w"))


# ---------------------------------------------------------------- LaTeX
def tex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def fmt_desc(s, sign):
    """Escape, bold the {{...}} activating-token markers, truncate on a word boundary.

    `sign` prefixes the line with + / - so the two descriptions in a stacked cell stay
    distinguishable without a header to point at.
    """
    # The negative line is grayed WHOLE (mark included) here rather than by the caller -- the
    # caller used to wrap this return value in a second \textcolor{gray}{...}, which nested and
    # left the marker double-wrapped.
    grey = sign == "-"
    mark = r"$%s$~" % ("+" if sign == "+" else "-")
    wrap = (lambda x: r"\textcolor{gray}{%s}" % x) if grey else (lambda x: x)
    if not s:
        return wrap(mark + "---")
    s = " ".join(s.split())
    if len(s) > DESC_CHARS:
        cut = s[:DESC_CHARS]
        # Prefer a word boundary, but only if one is reasonably near the end -- otherwise a
        # long unbroken token would collapse the cell to a couple of characters.
        sp = cut.rfind(" ")
        s = (cut[:sp] if sp > DESC_CHARS * 0.6 else cut) + "..."
    s = tex_escape(s)
    # Transluce wraps the activating token as {{tok}}; escaping turned those into \{\{tok\}\}.
    s = re.sub(r"\\\{\\\{(.*?)\\\}\\\}", r"\\textbf{\1}", s)
    # Some descriptions also carry raw markdown bold from the explainer model (e.g. **"creepy"**),
    # which would otherwise print as literal asterisks.
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return wrap(mark + s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="render from cache only")
    ap.add_argument("--descriptions", action="store_true",
                    help="inline the Transluce descriptions (4 pages instead of 1)")
    args = ap.parse_args()
    desc_mode = args.descriptions

    blocks, missing = [], []
    for task, tlabel in TASKS:
        rows = []
        for mkey, tag in METHODS:
            scores, meta = load_run(task, tag)
            if scores is None:
                missing.append(f"{task}/{tag}")
                continue
            assert meta["_method"] == mkey, f"{task}/{tag} parses as {meta['_method']}, not {mkey}"
            assert meta["intermediate_size"] * meta["seq_len"] * meta["num_layers"] == \
                meta["total"] == scores.numel(), f"layout mismatch in {task}/{tag}"
            rows.append((mkey, top_neurons(scores, meta)))
        if rows:
            blocks.append((task, tlabel, rows))

    # Recurrence over every (subtask, method) cell, and the colour each recurring neuron keeps
    # everywhere it appears. Ordered by (count desc, layer, neuron) so a rerun on unchanged
    # scores reproduces the same assignment byte for byte -- a table whose colours shuffle
    # between renders is worse than no colours, because the reader's memory of "the pink one"
    # silently goes stale.
    counts = Counter((n["layer"], n["neuron"])
                     for _, _, rs in blocks for _, ns in rs for n in ns)
    recur = sorted((k for k, v in counts.items() if v >= RECUR_MIN),
                   key=lambda k: (-counts[k], k))
    if len(recur) > len(PALETTE):
        print(f"NOTE: {len(recur)} neurons recur >={RECUR_MIN}x but the palette holds "
              f"{len(PALETTE)}; colouring the {len(PALETTE)} most frequent, rest left plain.",
              file=sys.stderr)
        recur = recur[:len(PALETTE)]
    color = {k: f"recur{i}" for i, k in enumerate(recur)}

    todo = {(n["layer"], n["neuron"]) for _, _, rs in blocks for _, ns in rs for n in ns}
    todo = sorted(k for k in todo if any(f"{k[0]}/{k[1]}/{s}" not in _cache for s in "+-"))
    if todo and desc_mode and not args.no_fetch:
        print(f"fetching {len(todo)} neurons from Transluce ({2 * len(todo)} requests)...")
        for i, (layer, neuron) in enumerate(todo, 1):
            for sign in "+-":
                describe(layer, neuron, sign)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
                save_cache()
        save_cache()

    # One column per method, read left to right; rank 1-5 down the rows, task as a row group.
    # Each cell stacks its lines with \newline (legal in a p-column, unlike \\ which would end
    # the table row): identity over position/score in compact mode, plus the two descriptions
    # under them in --descriptions mode. Rank rows alternate a faint shade -- in description
    # mode because a cell is 6-8 lines tall and rows that deep are hard to track across five
    # columns, in compact mode because five columns of near-identical "l30.n11158" strings are
    # easy to slip a row on.
    #
    # longtable, not tabular, in BOTH modes: description mode is four pages, and while compact
    # mode fits on one, a tabular that later stops fitting overflows off the bottom of the page
    # silently rather than breaking. That also means iclr2026_conference.tex must keep \input-ing
    # this file at top level -- a longtable inside a table float is an error.
    # The method header is repeated on every continuation page (\endhead) since a reader landing
    # mid-table otherwise has no way to tell which column is which.
    ncol = len(METHODS)
    # \textwidth is ~397pt here. tabcolsep is added on both sides of all 6 columns except at the
    # two @{} edges (2*6-2 = 10 gaps = 30pt = 0.076), and the rank column takes ~0.02, so the
    # method columns share what is left. Overshooting shows up as an Overfull \hbox, not as a
    # visibly broken table, so this is computed rather than eyeballed.
    W = (1.0 - 0.076 - 0.02) / ncol
    # Columns are ragged-right, not justified. At ~68pt a justified column cannot stretch its
    # interword glue enough to absorb a long word and overflows into its neighbour instead --
    # that alone accounted for most of the Overfull \hbox warnings this table used to emit.
    # >{...} needs the array package, which colortbl already \RequirePackage's (verified in the
    # build log), so this adds no preamble requirement beyond what \rowcolor already forces.
    col = r">{\raggedright\arraybackslash}p{%.3f\textwidth}" % W
    hdr = ["& " + " & ".join(r"\textbf{%s}" % LABELS[k] for k, _ in METHODS) + r" \\", r"\midrule"]
    def chip(key, body):
        """Wrap a neuron id in its recurrence colour, or leave it plain if it does not recur."""
        return r"\colorbox{%s}{%s}" % (color[key], body) if key in color else body

    L = [r"% Requires \usepackage{booktabs,longtable,colortbl,hyperref}; \input at top level "
         r"(NOT inside a table float).",
         *[r"\definecolor{recur%d}{HTML}{%s}" % (i, PALETTE[i]) for i in range(len(recur))],
         r"{\small",
         r"\setlength{\tabcolsep}{3pt}",
         # \colorbox's default 3pt padding would push the chips into the neighbouring column and
         # open up the line spacing inside every cell; 1pt keeps the highlight tight to the id.
         r"\setlength{\fboxsep}{1pt}",
         r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{longtable}{@{}l *{%d}{%s}@{}}" % (ncol, col),
         r"\toprule", *hdr, r"\endfirsthead",
         r"\toprule", *hdr, r"\endhead",
         r"\bottomrule",
         # Legend, so a colour is decodable without hunting for its other occurrences. It goes in
         # \endlastfoot rather than \endfoot: repeating it under all four pages would cost a
         # quarter of the vertical space the \newpage-per-subtask layout just bought.
         r"\multicolumn{%d}{@{}p{0.97\textwidth}@{}}{\scriptsize Recurring in $\geq$%d of the "
         r"%d cells:~%s} \\" % (
             ncol + 1, RECUR_MIN, len(blocks) * ncol,
             r"\quad ".join(
                 r"\colorbox{%s}{$\ell$%d.n%d}~$\times$%d" % (color[k], k[0], k[1], counts[k])
                 for k in recur)),
         r"\endlastfoot"]
    for task, tlabel, rows in blocks:
        by = {mk: ns for mk, ns in rows}
        # One subtask per page, in description mode only. A described block is ~45 typeset lines
        # and a page body holds ~53, so left to itself longtable breaks a block roughly in half
        # and the continuation page opens on "2." with nothing saying which subtask it belongs
        # to (the \endhead repeats the method names, not the row-group label). Forcing the break
        # makes every page self-labelling. Compact mode is ~50 lines TOTAL, so the same \newpage
        # would turn a one-page table into four near-empty ones.
        if desc_mode and task != blocks[0][0]:
            L.append(r"\newpage")
        elif task != blocks[0][0]:
            L.append(r"\addlinespace[3pt]")
        # \\* forbids a page break directly after the task header, so a subtask name can never
        # be orphaned at the foot of a page from the rows it labels.
        L.append(r"\multicolumn{%d}{@{}l}{\textbf{%s}} \\*[2pt]" % (ncol + 1, tlabel))
        for r_i in range(TOPN):
            cells = []
            for mkey, _ in METHODS:
                ns = by.get(mkey) or []
                if r_i >= len(ns):
                    cells.append("")
                    continue
                n = ns[r_i]
                url = NEURON_URL % (n["layer"], n["neuron"])
                # A plain space, not ~, between the id and the pos/score: "l30.n11158 p5 / 0.523"
                # is ~95pt of text in a 68pt column, so tying it together guarantees an overfull
                # box. A breakable space lets it sit on one line when it fits and wrap when not.
                ident = chip((n["layer"], n["neuron"]),
                             r"\href{%s}{\textbf{$\ell$%d.n%d}}" % (url, n["layer"], n["neuron"]))
                # Compact mode forces the break before the position/score instead of letting it
                # wrap. The two together are ~80pt of text in a 68pt column, so SOME cells
                # wrapped and some (short ids) did not, and a row whose five cells break
                # differently reads as noise. It costs no vertical space: every row already
                # contained a wrapped cell, so the row was two lines tall regardless. Description
                # mode keeps the breakable space -- there the id line is followed by two more
                # lines anyway, so a forced break buys nothing and would add 20 lines to a
                # layout that is already at the page limit.
                sep = " " if desc_mode else r"\newline "
                cell = (r"%s%s\textcolor{gray}{\scriptsize p%d\,/\,%.3g}"
                        % (ident, sep, n["pos"], n["score"]))
                if desc_mode:
                    cell += r"\newline %s\newline %s" % (
                        fmt_desc(describe(n["layer"], n["neuron"], "+", fetch=False), "+"),
                        fmt_desc(describe(n["layer"], n["neuron"], "-", fetch=False), "-"))
                cells.append(cell)
            shade = r"\rowcolor[HTML]{F7F7F7}" if r_i % 2 else ""
            L.append(f"{shade}{r_i + 1}. & " + " & ".join(cells) + r" \\")
    L += [r"\end{longtable}", r"}"]

    TABDIR.mkdir(parents=True, exist_ok=True)
    out = TABDIR / "sva_top_neurons.tex"
    out.write_text("\n".join(L) + "\n")
    n_rows = sum(len(ns) for _, _, rs in blocks for _, ns in rs)
    extra = ""
    if desc_mode:
        n_desc = sum(1 for _, _, rs in blocks for _, ns in rs for n in ns
                     if describe(n["layer"], n["neuron"], "+", fetch=False))
        extra = f", {n_desc}/{n_rows} with a + description"
    print(f"wrote {out}  ({len(blocks)} subtasks, {n_rows} neuron rows, "
          f"{'descriptions' if desc_mode else 'compact'}{extra})")
    if missing:
        print("MISSING runs:", ", ".join(missing))


if __name__ == "__main__":
    main()
