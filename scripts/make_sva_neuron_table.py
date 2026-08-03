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
DESC_CHARS = 88          # truncation budget per description cell


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
_cache = json.load(open(CACHE)) if CACHE.exists() else {}


def describe(layer, neuron, sign, fetch=True):
    """Best-scoring description for one (layer, neuron, sign), or None. Disk-cached.

    Cached failures are stored as None and NOT retried on later runs -- a neuron with no
    description is a fact about the database, not a transient error, and re-requesting the
    whole missing set on every render would hammer a service we do not own.
    """
    key = f"{layer}/{neuron}/{sign}"
    if key in _cache:
        return _cache[key]
    if not fetch:
        return None
    url = f"{API}?" + urllib.parse.urlencode({"layer": layer, "neuron": neuron, "sign": sign})
    val = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                summary = json.load(r).get("explanation_summary") or []
            val = summary[0][0] if summary else None
            break
        except Exception as e:                       # noqa: BLE001 - network, any failure retries
            if attempt == 2:
                print(f"  ! {key}: {type(e).__name__} {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    _cache[key] = val
    return val


def save_cache():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(_cache, open(CACHE, "w"))


# ---------------------------------------------------------------- LaTeX
def tex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def fmt_desc(s):
    """Escape, bold the {{...}} activating-token markers, truncate on a word boundary."""
    if not s:
        return r"\textcolor{gray}{---}"
    s = " ".join(s.split())
    if len(s) > DESC_CHARS:
        cut = s[:DESC_CHARS]
        # Prefer a word boundary, but only if one is reasonably near the end -- otherwise a
        # long unbroken token would collapse the cell to a couple of characters.
        sp = cut.rfind(" ")
        s = (cut[:sp] if sp > DESC_CHARS * 0.6 else cut) + "..."
    s = tex_escape(s)
    # Transluce wraps the activating token as {{tok}}; escaping turned those into \{\{tok\}\}.
    return re.sub(r"\\\{\\\{(.*?)\\\}\\\}", r"\\textbf{\1}", s)


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

    L = [r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{llrr p{0.29\textwidth} p{0.29\textwidth}}", r"\toprule",
         r"\textbf{Method} & \textbf{Neuron} & \textbf{Pos} & \textbf{Score} & "
         r"\textbf{Top $+$ description} & \textbf{Top $-$ description} \\"]
    for bi, (task, tlabel, rows) in enumerate(blocks):
        L.append(r"\midrule")
        L.append(r"\multicolumn{6}{l}{\textit{%s}} \\" % tlabel)
        for mkey, ns in rows:
            for i, n in enumerate(ns):
                L.append(" & ".join([
                    LABELS[mkey] if i == 0 else "",
                    r"$\ell$%d.n%d" % (n["layer"], n["neuron"]),
                    str(n["pos"]),
                    "%.3g" % n["score"],
                    fmt_desc(describe(n["layer"], n["neuron"], "+", fetch=False)),
                    fmt_desc(describe(n["layer"], n["neuron"], "-", fetch=False)),
                ]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]

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
