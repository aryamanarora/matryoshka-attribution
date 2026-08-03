"""Top-5 MLP neurons by attribution, per SVA subtask x method, with Transluce descriptions.

Reads the per-unit score tensors the SVA sweep already writes
(results/sva_sweep/<task>_llama3_mlp_<tag>.scores.pt) and reports, for each subtask and
method, the five neurons each method ranks FIRST -- i.e. the first units it puts into the
circuit. For llama3 each neuron is annotated with its top positive and top negative
description from Transluce's neuron-description database.

Run:  uv run python scripts/make_sva_neuron_table.py   ->  paper/tabs/sva_top_neurons.tex
      (add --no-fetch to render from the cache only, e.g. offline)

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
METHODS = [("IG", "ig"), ("IxG", "ixg"),
           ("eprun-s090", "eprun_s090"), ("stopk-log", "sufficient_topk_adam_bs1")]
LABELS = {"IG": "IG", "IxG": r"I$\times$G", "eprun-s090": "Node Pruning",
          "stopk-log": r"\ourmethod{}"}
# Truncation budget per description. The layout is one column per METHOD, so this shrinks with
# the number of methods -- four columns across \textwidth leaves ~0.21\textwidth each, which is
# roughly 55 characters over two typeset lines at \footnotesize.
DESC_CHARS = 55


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
    args = ap.parse_args()

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

    todo = {(n["layer"], n["neuron"]) for _, _, rs in blocks for _, ns in rs for n in ns}
    todo = sorted(k for k in todo if any(f"{k[0]}/{k[1]}/{s}" not in _cache for s in "+-"))
    if todo and not args.no_fetch:
        print(f"fetching {len(todo)} neurons from Transluce ({2 * len(todo)} requests)...")
        for i, (layer, neuron) in enumerate(todo, 1):
            for sign in "+-":
                describe(layer, neuron, sign)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
                save_cache()
        save_cache()

    # One column per method, read left to right; rank 1-5 down the rows, task as a row group.
    # Each cell stacks the neuron's identity over its two descriptions via \newline (legal in a
    # p-column, unlike \\ which would end the table row). Rank rows alternate a faint shade
    # because a cell is 5-6 typeset lines tall and unshaded rows of that height are hard to
    # track across four columns.
    ncol = len(METHODS)
    W = 0.21                       # p-column width as a fraction of \textwidth
    L = [r"{\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{tabular}{@{}l *{%d}{p{%.2f\textwidth}}@{}}" % (ncol, W), r"\toprule",
         "& " + " & ".join(r"\textbf{%s}" % LABELS[k] for k, _ in METHODS) + r" \\"]
    for task, tlabel, rows in blocks:
        by = {mk: ns for mk, ns in rows}
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{@{}l}{\textbf{%s}} \\[2pt]" % (ncol + 1, tlabel))
        for r_i in range(TOPN):
            cells = []
            for mkey, _ in METHODS:
                ns = by.get(mkey) or []
                if r_i >= len(ns):
                    cells.append("")
                    continue
                n = ns[r_i]
                pos_desc = describe(n["layer"], n["neuron"], "+", fetch=False)
                neg_desc = describe(n["layer"], n["neuron"], "-", fetch=False)
                cells.append(
                    r"\textbf{$\ell$%d.n%d}~\textcolor{gray}{\scriptsize p%d\;/\;%.3g}\newline "
                    r"%s\newline %s"
                    % (n["layer"], n["neuron"], n["pos"], n["score"],
                       fmt_desc(pos_desc, "+"), fmt_desc(neg_desc, "-")))
            shade = r"\rowcolor[HTML]{F7F7F7}" if r_i % 2 else ""
            L.append(f"{shade}{r_i + 1}. & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"}"]

    TABDIR.mkdir(parents=True, exist_ok=True)
    out = TABDIR / "sva_top_neurons.tex"
    out.write_text("\n".join(L) + "\n")
    n_rows = sum(len(ns) for _, _, rs in blocks for _, ns in rs)
    n_desc = sum(1 for _, _, rs in blocks for _, ns in rs for n in ns
                 if describe(n["layer"], n["neuron"], "+", fetch=False))
    print(f"wrote {out}  ({len(blocks)} subtasks, {n_rows} neuron rows, "
          f"{n_desc}/{n_rows} with a + description)")
    if missing:
        print("MISSING runs:", ", ".join(missing))


if __name__ == "__main__":
    main()
