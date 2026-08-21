"""Top-5 units by attribution, per training loss x SVA subtask x method.

Reads the per-unit score tensors the SVA sweep already writes
(results/sva_sweep/<task>_llama3_<substrate>_<tag>.scores.pt) and reports, for each subtask
and method, the five units each method ranks FIRST -- i.e. the first units it puts into the
circuit.

FOUR TABLES, one --substrate/--include-input combination each (see SUBSTRATES):

  --substrate mlp                  -> sva_top_neurons.tex     per-(layer, pos, neuron)
  --substrate mlp+attn_head        -> sva_top_mlp_attn.tex    the above PLUS per-(layer, pos, head)
  --substrate node                 -> sva_top_nodes.tex       MIB granularity: MLP block, attn head
  --substrate node --include-input -> sva_top_nodes_input.tex the above PLUS the embedding node

The last two are the same 1056 attn-head/MLP-block units with and without one extra unit at
flat index 0, and they are SEPARATE SWEEPS, not one sweep re-read: --include-input changes
what the mask can hold, so every method re-ranks under it. They live in different results
dirs (results/sva_sweep vs results/sva_sweep_input) for that reason, and --include-input
selects the dir; see submit_sva_node_pruning.sh, which is where the split originates.

Reading the two node tables against each other is the point of having both: if the input node
enters at rank 1 and the rest of the column is otherwise unchanged, the method is spending its
first pick on "the prompt matters" and the with-input faithfulness curve gains nothing that
tells you where in the network SVA is computed.

SECTIONED BY TRAINING LOSS (logit-diff, CE, accuracy), task within loss. All three losses
were swept for all six methods, and which units a method reaches for first is exactly the
thing the loss is expected to move -- showing only the logit-diff third made that
unfalsifiable. 3 losses x 8 tasks x 6 methods = 144 cells, 720 unit rows.

The 8 tasks are the 4 SVA subtasks and the 4 goodfire-ai/arithmetic-wild tasks (see TASKS).
The arithmetic half carries PRIOR-WORK MARKS -- \\star for a neuron \\citet{feucht2026arithmetic}
publish, \\dagger for the layer-18 MLP block they localise to -- the SVA+ counterpart of the
IOI table's role colours, and the only unit-level published circuit available for these tasks.
See the GOODFIRE block for what each mark does and does not claim. Every run prints its mark
counts, and a count of zero is a result, not a failure: it says no method's top-5 on that task
contains a unit the prior work names.

DEFAULT LAYOUT IS COMPACT (~3 pages): the identifier alone. `--descriptions` adds
each neuron's top positive and negative description from Transluce, which is a much richer
table but runs to twelve pages -- one per loss x subtask -- because a described cell is 6-8
typeset lines instead of 2. Every neuron id is a hyperlink to Transluce either way, so the
compact table does not lose access to the descriptions, only their inlining.

Run:  uv run python scripts/make_sva_neuron_table.py   ->  paper/tabs/sva_top_neurons.tex
      (--descriptions for the long version; --no-fetch renders from cache only, e.g. offline)

WHY llama3-only: the SVA subtasks were only ever swept on llama3, at every substrate, so there
is exactly one model here and no cross-model column to add.

RANKING. Raw score, sorted DESCENDING -- not |score|. That is not a stylistic choice: it is
what evaluate.sparsity_sweep does (`flat.argsort(descending=True)`), so these really are the
units that enter each method's circuit at the smallest budget and drive the left end of every
faithfulness curve in the paper. Ranking by |score| here would show a DIFFERENT set of neurons
than the ones our own numbers were computed from, for IG/IxG especially, whose scores are
signed effects rather than importances.

NEURON vs UNIT. At the positional substrates (`mlp`, `mlp+attn_head`) the score index is
per-(layer, position, neuron) -- 32 x 6 x 14336 = 2752512 units -- but the question is about
neurons, and one neuron can occupy several of the top slots at different positions. So we
deduplicate to distinct (component, layer, neuron-or-head) keeping each unit's best-scoring
position, and report that position. "Top 5" therefore means 5 distinct units, which is usually
deeper into the raw ranking than slot 5. At `node` there is no position axis and every index is
already a distinct unit, so the dedupe is a no-op there and the printed rank is the raw rank.

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

TABDIR = Path("paper/tabs")
CACHE = Path("results/.transluce_cache.json")
MODEL, TOPN = "llama3", 5
# Query heads per layer, asserted rather than assumed: the head count is NOT in the runs'
# metadata, so layout() derives it from `total` and cross-checks it here. That check is what
# distinguishes a with-input node run from a without-input one (1057 vs 1056 = 32*32 + 32 + i),
# and reading one as the other would shift every attn head by one layer with no visible symptom.
NUM_HEADS = {"llama3": 32}

# --substrate / --include-input -> (results dir, on-disk substrate tag, output stem, recurrence
# cut). The tag is the run's own meta["nodes"] with `+` -> `-`, which is the substitution
# eval_sva.py makes when it builds a filename. `recur` is set per substrate because the
# recurrence distribution is a property of the substrate SIZE, not a global constant: 2.7M
# positional units almost never collide across cells, 1056 node units collide constantly, and a
# single threshold would either colour nothing in one table or everything in the other.
#
# Where each value comes from (the descending count list main() prints is the evidence):
#   mlp            14  the original gap: 24 22 18 16 14 14 14 | 9 7 7 ...
#   mlp+attn_head  14  MATCHED to mlp, not gap-read. This distribution is a smooth staircase --
#                      20 17 15 14 12 11 10 9 9 9 | 6 -- whose only gap in the palette range is
#                      at >=9, and that selects 10 units for 8 colours AND cuts a three-way tie
#                      at 9. With no gap to read, comparability wins: a chip here means the same
#                      frequency as a chip in the mlp table, which is the table it is read against.
#   node           29  gap 30 | 24 (nothing sits at 29, so >=29 and >=30 select the same 3)
#   node+input     29  gap 29 29 | 22
# The two node values are deliberately EQUAL rather than separately gap-read, for the same
# comparability reason: these two tables exist to be read against each other.
SUBSTRATES = {
    ("mlp", False): dict(res="results/sva_sweep", tag="mlp",
                         out="sva_top_neurons", recur=14),
    ("mlp+attn_head", False): dict(res="results/sva_sweep", tag="mlp-attn_head",
                                   out="sva_top_mlp_attn", recur=14),
    ("node", False): dict(res="results/sva_sweep", tag="node",
                          out="sva_top_nodes", recur=29),
    ("node", True): dict(res="results/sva_sweep_input", tag="node",
                         out="sva_top_nodes_input", recur=29),
}

# Transluce's neuron-data server. sign=+/- selects the positive/negative activation direction;
# the response's explanation_summary is a [description, score] list already sorted best-first.
API = "https://transluce--neuron-data-server-fastapi-app.modal.run/read_specific_file"

TASKS = [("simple", "Simple"), ("nounpp", "Noun PP"),
         ("rc", "RC"), ("within_rc", "Within RC"),
         # goodfire-ai/arithmetic-wild, same model and substrates. These are NOT agreement
         # tasks, so a unit recurring across an SVA cell and an arith cell is a much stronger
         # claim than one recurring across two SVA subtasks -- which is the reason to have them
         # in the same table rather than a parallel one, and the reason the recurrence cuts in
         # SUBSTRATES had to be re-read once these landed (12 cells per loss became 24).
         ("addition", "Addition"), ("months", "Months"),
         ("weekdays", "Weekdays"), ("hours", "Hours")]

# ---------------------------------------------------------------- prior-work annotation
# The arithmetic half of this table is the one place in the SVA+ appendix where a PUBLISHED
# unit-level circuit exists to check against, so it gets the same treatment the IOI table gives
# \citet{wang2022interpretability}'s head roles: units the prior work names are marked in place.
#
# `arith_wild_l18_neurons.json` is goodfire-ai/arithmetic-wild's own `src/neurons_per_task.json`,
# vendored (results/ and paper/ are gitignored, and a table generator that reads a sibling home
# directory silently stops annotating on any other machine). Its contents are LAYER-18
# down_proj-INPUT neuron indices -- the same 14336-wide index space this table's `mlp` substrate
# uses -- selected in that repo's src/neuron_selection.ipynb by write score
# omega = ||n^T S|| / ||n|| > 0.4 onto the task's layer-18 DAS *output* subspace.
#
# TWO marks, because the two substrate families can support two different claims and conflating
# them would overclaim at `node`:
#   \star    (neuron substrates) this EXACT neuron is in that task's published set.
#   \dagger  (node substrates)   this is the layer-18 MLP BLOCK, the one the published neurons
#            live in. A far weaker statement -- the block holds 14336 neurons and the published
#            set is 15-28 of them -- so it is worded as "the block they localise to", not as a
#            hit, and it is never drawn at the neuron substrates where the exact test is available.
# Both are drawn ONLY in the four arithmetic blocks, and only against that task's OWN list: the
# four lists overlap heavily (weekdays' 15 are a subset of addition's 28), so marking a
# cross-task member would turn "the method found the weekdays neurons" into a near-tautology.
GOODFIRE = json.load(open(Path(__file__).resolve().parent.parent
                          / "src/learning_to_attribute/data/arith_wild_l18_neurons.json"))
GF_LAYER, GF_NEURONS = GOODFIRE["layer"], GOODFIRE["neurons"]
GF_CITE = r"\citet{feucht2026arithmetic}"
# Training losses, as a top-level section each. Keys are the runs' own meta["loss"]; the middle
# field is the tag fragment that selects them on disk (empty = logit-diff, the loss every
# headline SVA number in the paper uses). Order matches plot_sva_sweep.LOSS_ORDER so the two
# artifacts read top-to-bottom the same way.
LOSSES = [("logit_diff", "", "Logit difference"),
          ("ce", "_ce", "Cross-entropy"),
          ("acc", "_acc", "Accuracy")]
# Methods and the order they appear. Keys are parse_method()'s outputs; %s takes the loss
# fragment above. It is a template rather than a suffix because the fragment does not land at
# the end for the two MAttr rows -- `sufficient_topk_adam_ce_bs1`, not `..._bs1_ce`.
# Same series as plot_accauc_vs_faithauc's FIGURE_METHODS, so this table and that figure
# describe the same runs. DBM's fragment DOES land at the end (eval_sva.py appends _ce/_acc to
# the whole tag), unlike the MAttr rows -- which is why the template is a %s slot, not a suffix.
METHODS = [("IG", "ig%s"), ("IxG", "ixg%s"), ("eprun-s090", "eprun_s090%s"),
           ("sig_lr0.3_l16.0", "sig_lr0.3_l16.0%s"),
           ("stopk-log", "sufficient_topk_adam%s_bs1"),
           ("stopk-unif", "sufficient_topk_adam%s_uniformk_bs1"),
           # Same gate, same backward, Adam -> SGD (lr=1.0). Only the log-k arm is a column: the
           # unif-k one would be an eighth, and unif-k is where the two OPTIMIZERS differ most
           # (Adam drops ~0.19 acc-AUC log -> unif at the neuron substrates, SGD 0.006), so a
           # unif-k SGD column would be read as a k-schedule statement about \ourmethod{} when it
           # is really about Adam. Adding a column also re-opens the 14-of-72 recurrence cut
           # below -- main() prints the counts either side of the gap; check it after a rerun.
           ("softsgd-log", "sufficient_topk_sgd%s_bs1")]
LABELS = {"IG": "IG", "IxG": r"I$\times$G", "eprun-s090": "Node Pruning",
          "sig_lr0.3_l16.0": "DBM",
          "stopk-log": r"\ourmethod{}", "stopk-unif": r"\ourmethod{} $+$ unif $k$",
          "softsgd-log": r"\ourmethod{} $+$ SGD"}
# Truncation budget per description. The layout is one column per METHOD, so the space per
# description shrinks with the number of methods -- but the font drops a step at six columns
# (see `font` in make()), and the two roughly cancel: ~68pt at \small and ~59pt at \scriptsize
# are both ~15 characters per typeset line, so 45 still wraps to about three lines.
DESC_CHARS = 45
# Transluce's neuron browser, the same URL scheme tabs/arith_mlp_neuron_table.tex links to.
NEURON_URL = "https://neurons.transluce.org/%d/%d/+"

# Neurons that surface in many (subtask, method) cells get a categorical highlight, so that
# "these two columns keep picking the SAME unit" is visible at a glance instead of something
# you verify by reading 100 six-digit ids. Colour = identity, nothing else: it does not encode
# rank, score or count.
#
# The threshold is 14 of the 72 cells because the recurrence distribution has a clean gap
# there -- 7 neurons appear 14-24 times, then nothing at all between 13 and 10, then the tail
# resumes at 9 -- so this is reading a break in the data, not imposing a cutoff. It also lands
# inside the palette size. (It was 4-of-20 when the table covered one loss; the same gap-reading
# rule gave 14 once all three losses were in, and 4-of-72 would select 20 neurons for 8
# colours.) Adding the DBM column moved the cell count 60 -> 72 without moving the threshold or
# the selected set at all: DBM's top-5s never land on any of the 7, so the counts either side of
# the gap are unchanged. main() prints the counts either side of the cut so a rerun that moves
# the gap is visible rather than silently recoloured.
#
# THE RULE, not the number, is what carries over to the other substrates: read the gap in the
# descending count list main() prints on every run, don't reuse 14. Each substrate's value lives
# in SUBSTRATES[...]["recur"] above; `--recur-min` overrides it for one render.
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


def load_run(res, task, sub_tag, tag):
    """(scores tensor, meta dict) for one cell, or (None, None) if that run is missing."""
    stem = res / f"{task}_{MODEL}_{sub_tag}_{tag}"
    pt, js = Path(str(stem) + ".scores.pt"), Path(str(stem) + ".json")
    if not pt.exists() or not js.exists():
        return None, None
    meta = json.load(open(js))
    # Guard the identity of the run rather than trusting the filename: parse_method is the
    # repo's naming authority and a tag that no longer maps to the expected key means the
    # sweep was renamed under us, which would silently mislabel every row below.
    got = parse_method(pt.name.replace(".scores.pt", ".json"), meta)
    return torch.load(pt, map_location="cpu", weights_only=False), (meta | {"_method": got})


def layout(meta, sub, inp):
    """Heads per layer for this run, asserting the flat score layout is the one decode() assumes.

    The head count is not stored in the runs' metadata, so it is DERIVED from `total` by
    subtracting the parts whose size is known, and the derivation is only well-posed if the
    layout is as expected -- which is what makes it a check and not an assumption:

      mlp            L*P*N                       (no heads to derive; returns None)
      mlp+attn_head  L*P*N + L*P*H
      node           [1 if include_input] + L*H + L

    Both node variants are pinned by exact divisibility alone: without input 1056 - 32 = 1024 =
    32*32, with input 1057 - 1 - 32 = 1024, and swapping the two leaves 1025 or 1023, neither
    divisible by 32. So pointing --include-input at the wrong results dir is an AssertionError,
    not a table in which every attention head silently sits one layer off.
    """
    L, P, N = meta["num_layers"], meta["seq_len"], meta["intermediate_size"]
    total = meta["total"]
    if sub == "mlp":
        assert L * P * N == total, f"mlp layout: {L}*{P}*{N} != {total}"
        return None
    if sub == "mlp+attn_head":
        rest, per = total - L * P * N, L * P
    else:
        rest, per = total - (1 if inp else 0) - L, L
    assert rest > 0 and rest % per == 0, f"{sub} layout: {rest} not divisible by {per}"
    H = rest // per
    assert H == NUM_HEADS[MODEL], f"derived {H} heads, expected {NUM_HEADS[MODEL]}"
    return H


def decode(idx, meta, sub, H, inp):
    """Flat unit index -> unit dict: kind (mlp/attn/input), layer, and head/neuron/pos if any.

    Byte-for-byte the same arithmetic as LlamaAttributionHooks.decode_index (models/llama.py),
    which is the authority for these layouts; it is duplicated rather than imported because
    decode_index is an instance method and instantiating the hooker would mean loading 8B of
    weights just to divide two integers. The positional branch also matches the writer side,
    eval_sva.gradient_scores: `off = li * P * N` then a [P, N] block reshaped row-major, i.e.
    index = layer*(P*N) + pos*N + neuron.

    Getting this wrong is the failure mode with no symptom -- every description would attach
    to the wrong neuron and the table would still look entirely plausible -- so layout() pins
    the shape before this is trusted.

    That assert only pins the shape, so the ORDERING was checked separately against a
    signature a transposed decode could not reproduce: under this decode, positions 0 and 1
    hold *exactly* 0.0 IG score in all four subtasks (BOS and the first token are identical in
    the clean and patch prompts, so g.(clean - patch) is identically zero there), while the
    top-200 units concentrate at pos 2 and pos P-1 -- the subject and the verb, which is where
    subject-verb agreement lives. Note P is per-subtask (simple 3, nounpp 6, rc 7,
    within_rc 6), read from each run's json rather than assumed.

    NOTE the two substrates order their blocks differently, and it is not a typo: `mlp+attn_head`
    is [all MLP | all attn] with MLP first, while `node` is [input | all attn | all MLP] with
    attn first. That is how the hooker lays them out; decode_index has the same asymmetry.
    """
    idx = int(idx)
    L, P, N = meta["num_layers"], meta["seq_len"], meta["intermediate_size"]
    if sub == "node":
        if inp and idx == 0:
            return dict(kind="input", layer=-1)
        i = idx - (1 if inp else 0)
        if i < L * H:
            return dict(kind="attn", layer=i // H, head=i % H)
        return dict(kind="mlp", layer=i - L * H)
    if sub == "mlp" or idx < L * P * N:
        layer, rem = divmod(idx, P * N)
        pos, neuron = divmod(rem, N)
        return dict(kind="mlp", layer=layer, pos=pos, neuron=neuron)
    layer, rem = divmod(idx - L * P * N, P * H)
    pos, head = divmod(rem, H)
    return dict(kind="attn", layer=layer, pos=pos, head=head)


def unit_key(u):
    """Identity for dedupe / recurrence colouring: everything about the unit EXCEPT position.

    Position is excluded because the chip means "this unit recurs across cells" and the same
    neuron shows up at different positions; folding position in would report a recurrence that
    was never measured. At `node` there is no position and the key is the whole unit.
    """
    if u["kind"] == "input":
        return ("input",)
    return (u["kind"], u["layer"], u.get("head", u.get("neuron")))


def unit_label(u):
    r"""Chip text for a unit: `$\ell$30.n11158`, `$\ell$12.h7`, `$\ell$12.mlp`, or `input`."""
    if u["kind"] == "input":
        return r"\textsc{input}"
    if u["kind"] == "attn":
        return r"$\ell$%d.h%d" % (u["layer"], u["head"])
    if "neuron" in u:
        return r"$\ell$%d.n%d" % (u["layer"], u["neuron"])
    return r"$\ell$%d.mlp" % u["layer"]


def gf_mark(u, task, sub):
    r"""`$^{\star}$` / `$^{\dagger}$` / "" -- see the GOODFIRE block for what each claims."""
    if task not in GF_NEURONS or u["kind"] != "mlp" or u["layer"] != GF_LAYER:
        return ""
    if "neuron" in u:
        return r"$^{\star}$" if u["neuron"] in GF_NEURONS[task] else ""
    return r"$^{\dagger}$" if sub == "node" else ""


def gf_legend(gf, sub):
    """Legend row(s) for whichever prior-work marks this render actually produced.

    Returns a list so it can be spliced into \\endlastfoot, and is EMPTY when no mark occurred --
    the four arithmetic blocks are always rendered, so an absent legend means the marks found
    nothing, which main() reports on stderr rather than by printing a promise the table does not
    keep.
    """
    n_arith = sum(1 for t, _ in TASKS if t in GF_NEURONS)
    parts = []
    if gf[r"$^{\star}$"]:
        parts.append(r"$^{\star}$~one of the %d--%d layer-%d MLP neurons %s localise that task's "
                     r"cyclic arithmetic to (%d occurrences here)"
                     % (min(map(len, GF_NEURONS.values())), max(map(len, GF_NEURONS.values())),
                        GF_LAYER, GF_CITE, gf[r"$^{\star}$"]))
    if gf[r"$^{\dagger}$"]:
        parts.append(r"$^{\dagger}$~the layer-%d MLP block, which is where %s localise those "
                     r"neurons -- at this granularity the block is one unit, so this marks the "
                     r"right \emph{site}, not the right neurons (%d occurrences)"
                     % (GF_LAYER, GF_CITE, gf[r"$^{\dagger}$"]))
    if not parts:
        return []
    return [r"\multicolumn{%d}{@{}p{0.97\textwidth}@{}}{\scriptsize Marks apply to the %d "
            r"arithmetic blocks only, against each task's own published set:~%s} \\"
            % (len(METHODS) + 1, n_arith, r"\quad ".join(parts))]


def top_units(scores, meta, sub, H, inp, n=TOPN):
    """Top-n DISTINCT units by descending raw score, each unit's best position kept."""
    order = torch.argsort(scores, descending=True)
    out, seen = [], set()
    for idx in order.tolist():
        u = decode(idx, meta, sub, H, inp)
        k = unit_key(u)
        if k in seen:
            continue
        seen.add(k)
        out.append(u | {"score": float(scores[idx])})
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
                    help="inline the Transluce descriptions (12 pages instead of ~3)")
    ap.add_argument("--substrate", default="mlp",
                    choices=sorted({s for s, _ in SUBSTRATES}),
                    help="score granularity; picks the results dir and the output file")
    ap.add_argument("--include-input", action="store_true",
                    help="node only: read results/sva_sweep_input, whose mask carries the "
                         "embedding node at flat index 0")
    ap.add_argument("--recur-min", type=int, default=None,
                    help="override this substrate's recurrence colouring cut")
    args = ap.parse_args()
    desc_mode = args.descriptions
    sub, inp = args.substrate, args.include_input
    if (sub, inp) not in SUBSTRATES:
        ap.error(f"--include-input was only ever swept at --substrate node, not {sub} "
                 f"(there is no results/sva_sweep_input run for it)")
    cfg = SUBSTRATES[(sub, inp)]
    res, recur_min = Path(cfg["res"]), args.recur_min or cfg["recur"]
    # Descriptions are per-NEURON facts from Transluce's database; at `node` a unit is a whole
    # MLP block or attention head and there is nothing to look up, so every cell would render
    # two "---" lines and quadruple the page count to say nothing.
    if desc_mode and sub == "node":
        ap.error("--descriptions is neuron-level; --substrate node has no neuron indices")

    blocks, missing = [], []
    for lkey, lsuf, llabel in LOSSES:
        for task, tlabel in TASKS:
            rows = []
            for mkey, tmpl in METHODS:
                tag = tmpl % lsuf
                scores, meta = load_run(res, task, cfg["tag"], tag)
                if scores is None:
                    missing.append(f"{task}/{tag}")
                    continue
                assert meta["_method"] == mkey, \
                    f"{task}/{tag} parses as {meta['_method']}, not {mkey}"
                # The loss is asserted from the run's own metadata, not inferred from the tag
                # fragment. The fragment is a filename convention; meta["loss"] is what the
                # trainer actually optimised, and a section headed "Cross-entropy" that
                # silently held logit-diff runs is exactly the error this table cannot show.
                assert meta.get("loss") == lkey, \
                    f"{task}/{tag} was trained with loss={meta.get('loss')}, not {lkey}"
                # Same idea for the substrate: the filename tag says `node`, meta["nodes"] says
                # what the hooker actually masked, and layout() then pins the flat layout that
                # decode() is about to assume.
                assert meta.get("nodes") == sub, \
                    f"{task}/{tag} was swept at nodes={meta.get('nodes')}, not {sub}"
                assert meta["total"] == scores.numel(), f"layout mismatch in {task}/{tag}"
                H = layout(meta, sub, inp)
                rows.append((mkey, top_units(scores, meta, sub, H, inp)))
            if rows:
                blocks.append((lkey, llabel, task, tlabel, rows))

    # Recurrence over every (subtask, method) cell, and the colour each recurring neuron keeps
    # everywhere it appears. Ordered by (count desc, layer, neuron) so a rerun on unchanged
    # scores reproduces the same assignment byte for byte -- a table whose colours shuffle
    # between renders is worse than no colours, because the reader's memory of "the pink one"
    # silently goes stale.
    units = {}                              # key -> a representative unit, for the legend label
    for *_, rs in blocks:
        for _, ns in rs:
            for n in ns:
                units.setdefault(unit_key(n), n)
    counts = Counter(unit_key(n) for *_, rs in blocks for _, ns in rs for n in ns)
    # Prior-work marks, counted BEFORE rendering so the legend can name only the marks that
    # actually occur. A legend that promises a $\star$ no cell carries reads as "the methods
    # missed them" only if you scan all 720 ids to confirm; the honest version is to drop the
    # entry and say so in the run log instead, which is what main()'s stderr line does.
    gf = Counter(m for _, _, task, _, rs in blocks for _, ns in rs for n in ns
                 if (m := gf_mark(n, task, sub)))
    recur = sorted((k for k, v in counts.items() if v >= recur_min),
                   key=lambda k: (-counts[k], k))
    # Truncation is not just cosmetic: the legend below says "recurring in >=recur_min cells", and
    # if the palette cut off some of the units that clear recur_min, that sentence is false. So
    # the legend switches to describing what is actually coloured -- and the tie warning matters
    # because a cut through a plateau of equal counts colours some members and not others with no
    # rule a reader could infer.
    truncated = len(recur) > len(PALETTE)
    if truncated:
        print(f"NOTE: {len(recur)} units recur >={recur_min}x but the palette holds "
              f"{len(PALETTE)}; colouring the {len(PALETTE)} most frequent, rest left plain.",
              file=sys.stderr)
        recur = recur[:len(PALETTE)]
        if counts[recur[-1]] == max(counts[k] for k in counts if k not in set(recur)):
            print(f"  ! the cut splits a tie at {counts[recur[-1]]}x -- units with identical "
                  f"counts are coloured differently. Raise --recur-min above it.", file=sys.stderr)
    color = {k: f"recur{i}" for i, k in enumerate(recur)}
    # Show the cut: the smallest count kept vs the largest dropped. The threshold is justified by
    # a gap in this distribution, and a gap is the one property a constant cannot assert about
    # itself -- if a rerun closes it these two numbers land next to each other and say so. The
    # descending count list next to them is what a NEW substrate's threshold is read off.
    kept = min((counts[k] for k in recur), default=None)
    drop = max((v for k, v in counts.items() if k not in color), default=None)
    print(f"[{cfg['out']}] recurrence cut at >={recur_min}: {len(recur)} coloured "
          f"(lowest kept {kept}x, highest dropped {drop}x)", file=sys.stderr)
    print("  counts, descending: "
          + " ".join(str(v) for v in sorted(counts.values(), reverse=True)[:24]) + " ...",
          file=sys.stderr)

    # Only MLP units with a neuron index are describable; at mlp+attn_head the attn rows have no
    # Transluce entry and are simply skipped rather than fetched and cached as failures.
    todo = {(n["layer"], n["neuron"]) for *_, rs in blocks for _, ns in rs for n in ns
            if n["kind"] == "mlp" and "neuron" in n}
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
    # A compact cell is now a single identifier; --descriptions mode stacks the two Transluce
    # lines under it with \newline (legal in a p-column, unlike \\ which would end the table
    # row). Rank rows alternate a faint shade -- in description mode because a cell is 5-7 lines
    # tall and rows that deep are hard to track across five columns, in compact mode because
    # five columns of near-identical "l30.n11158.p5" strings are easy to slip a row on.
    #
    # longtable, not tabular, in BOTH modes: description mode is four pages, and while compact
    # mode fits on one, a tabular that later stops fitting overflows off the bottom of the page
    # silently rather than breaking. That also means iclr2026_conference.tex must keep \input-ing
    # this file at top level -- a longtable inside a table float is an error.
    # The method header is repeated on every continuation page (\endhead) since a reader landing
    # mid-table otherwise has no way to tell which column is which.
    ncol = len(METHODS)
    # \textwidth is ~397pt here. tabcolsep (3pt, set below) is added on both sides of all
    # ncol+1 columns except at the two @{} edges, i.e. 2*ncol gaps, and the rank column takes
    # ~0.02, so the method columns share what is left. Derived from ncol rather than written
    # out as a constant: adding a method used to leave the old five-column number in place, and
    # overshooting shows up as an Overfull \hbox, not as a visibly broken table.
    W = (1.0 - 2 * ncol * 3.0 / 397.0 - 0.02) / ncol
    # Columns are ragged-right, not justified. At ~68pt a justified column cannot stretch its
    # interword glue enough to absorb a long word and overflows into its neighbour instead --
    # that alone accounted for most of the Overfull \hbox warnings this table used to emit.
    # >{...} needs the array package, which colortbl already \RequirePackage's (verified in the
    # build log), so this adds no preamble requirement beyond what \rowcolor already forces.
    col = r">{\raggedright\arraybackslash}p{%.3f\textwidth}" % W
    # A chip is a \colorbox, i.e. an UNBREAKABLE box: ragged-right cannot rescue an id wider than
    # its column, it just hangs over the neighbour. The widest id on this data is 12 characters
    # ($\ell$31.n13964.p2); digits and lowercase run ~0.5em, so ~59pt at \small (9pt) against a
    # 58.8pt column at ncol=6 -- over the edge, where ncol=5 had 71.8pt and 13pt of slack. One
    # step down the size ladder buys back ~11%% of the width, which restores the slack.
    font = r"\small" if W * 397 >= 65 else r"\scriptsize"
    hdr = ["& " + " & ".join(r"\textbf{%s}" % LABELS[k] for k, _ in METHODS) + r" \\", r"\midrule"]
    def chip(key, body):
        """Wrap a unit id in its recurrence colour, or leave it plain if it does not recur."""
        return r"\colorbox{%s}{%s}" % (color[key], body) if key in color else body

    L = [r"% Requires \usepackage{booktabs,longtable,colortbl,hyperref}; \input at top level "
         r"(NOT inside a table float).",
         *[r"\definecolor{recur%d}{HTML}{%s}" % (i, PALETTE[i]) for i in range(len(recur))],
         "{" + font,
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
         r"\multicolumn{%d}{@{}p{0.97\textwidth}@{}}{\scriptsize %s in $\geq$%d of the "
         r"%d cells:~%s} \\" % (
             ncol + 1,
             f"The {len(PALETTE)} most frequent, each recurring" if truncated else "Recurring",
             kept if truncated else recur_min, len(blocks) * ncol,
             r"\quad ".join(
                 r"\colorbox{%s}{%s}~$\times$%d" % (color[k], unit_label(units[k]), counts[k])
                 for k in recur)),
         *gf_legend(gf, sub),
         r"\endlastfoot"]
    for b_i, (lkey, llabel, task, tlabel, rows) in enumerate(blocks):
        by = {mk: ns for mk, ns in rows}
        # One subtask per page, in description mode only. A described block is ~45 typeset lines
        # and a page body holds ~53, so left to itself longtable breaks a block roughly in half
        # and the continuation page opens on "2." with nothing saying which subtask it belongs
        # to (the \endhead repeats the method names, not the row-group label). Forcing the break
        # makes every page self-labelling. Compact mode is ~12 lines per block, so the same
        # \newpage would turn a three-page table into twelve near-empty ones.
        if desc_mode and b_i:
            L.append(r"\newpage")
        elif b_i:
            L.append(r"\addlinespace[3pt]")
        # Two levels of row group: training loss over subtask. They have to be told apart at a
        # glance or a reader scanning for "RC" cannot tell which of the three RC blocks they
        # landed in, so the loss header gets a rule above it and small caps, the subtask header
        # stays flush-left bold. The rule is what actually does the work -- \textsc alone is too
        # quiet a difference at \small.
        if not b_i or lkey != blocks[b_i - 1][0]:
            # No rule for the first section (the column header's own \midrule is right above
            # it), and none in description mode (every block opens a page there, so \endhead
            # has just drawn one) -- either would render as a double rule.
            if b_i and not desc_mode:
                L.append(r"\midrule")   # the 3pt block spacer above already opens the gap
            L.append(r"\multicolumn{%d}{@{}l}{\textsc{\textbf{%s} loss}} \\*[3pt]"
                     % (ncol + 1, llabel))
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
                describable = n["kind"] == "mlp" and "neuron" in n
                # One identifier, "l30.n11158.p5" -- layer, unit, position (positional substrates
                # only; `node` units have no position and stop at "l30.mlp"). The raw score used
                # to sit next to it and is gone: it is not comparable across columns (IG's
                # signed effects and a mask's logits are different quantities in different
                # units), so a reader could only ever compare it DOWN a column, which is the one
                # thing the rank number already says.
                #
                # \colorbox covers only the l.n part and the position is appended outside it,
                # because the chip means "this UNIT recurs across cells" and position is not part
                # of that identity -- the same neuron shows up at different positions, and
                # highlighting the position with it would claim a recurrence that was not
                # measured.
                cell = chip(unit_key(n), unit_label(n))
                if "pos" in n:
                    cell += ".p%d" % n["pos"]
                # Prior-work mark, OUTSIDE the chip and after the position, for the same reason
                # the position sits outside it: the chip means "recurs across cells", and a unit
                # can be both recurring and published, so the two annotations have to compose
                # rather than one overwriting the other.
                cell += gf_mark(n, task, sub)
                # Only MLP neurons get the Transluce link; there is no browser page for an
                # attention head or a whole MLP block, so those ids stay plain text. Nesting the
                # link OUTSIDE the box rather than the reverse keeps the whole id one uniform
                # hyperref colour instead of a highlighted darkblue stem and a black tail.
                if describable:
                    cell = r"\href{%s}{%s}" % (NEURON_URL % (n["layer"], n["neuron"]), cell)
                if desc_mode and describable:
                    cell += r"\newline %s\newline %s" % (
                        fmt_desc(describe(n["layer"], n["neuron"], "+", fetch=False), "+"),
                        fmt_desc(describe(n["layer"], n["neuron"], "-", fetch=False), "-"))
                cells.append(cell)
            shade = r"\rowcolor[HTML]{F7F7F7}" if r_i % 2 else ""
            L.append(f"{shade}{r_i + 1}. & " + " & ".join(cells) + r" \\")
    L += [r"\end{longtable}", r"}"]

    TABDIR.mkdir(parents=True, exist_ok=True)
    out = TABDIR / f"{cfg['out']}.tex"
    out.write_text("\n".join(L) + "\n")
    n_rows = sum(len(ns) for *_, rs in blocks for _, ns in rs)
    extra = ""
    if desc_mode:
        n_desc = sum(1 for *_, rs in blocks for _, ns in rs for n in ns
                     if n["kind"] == "mlp" and "neuron" in n
                     and describe(n["layer"], n["neuron"], "+", fetch=False))
        extra = f", {n_desc}/{n_rows} with a + description"
    # The component mix is the headline fact for the two mixed substrates -- "IG's top-5 is all
    # attention and MAttr's is all MLP" is not visible from a row count -- so it is printed
    # rather than left to be eyeballed off the rendered page.
    mix = Counter(n["kind"] for *_, rs in blocks for _, ns in rs for n in ns)
    print(f"wrote {out}  ({len(blocks)} loss x subtask blocks, {n_rows} unit rows, "
          f"{'descriptions' if desc_mode else 'compact'}{extra})")
    print("  component mix: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items())))
    # A zero here is a RESULT, not a bug: it says no method's top-5 on that task contains a unit
    # the prior work names. Print it either way so the two cases are distinguishable without
    # grepping the .tex, and print the per-task split because the four lists differ in size and
    # `hours` is the one our own multitask run already failed to recover (ARITHMETIC_WILD_REPORT).
    n_arith_cells = sum(1 for _, _, t, _, _ in blocks if t in GF_NEURONS) * len(METHODS)
    per_task = Counter(t for _, _, t, _, rs in blocks for _, ns in rs for n in ns
                       if gf_mark(n, t, sub))
    star, dag = gf[r"$^{\star}$"], gf[r"$^{\dagger}$"]
    print(f"  prior work ({GOODFIRE['_paper'].split(':')[0]}): {star} exact-neuron marks, "
          f"{dag} layer-{GF_LAYER}-block marks over {n_arith_cells} arithmetic cells"
          + (f"  [{', '.join(f'{k} {v}' for k, v in sorted(per_task.items()))}]" if per_task else ""))
    if missing:
        print("MISSING runs:", ", ".join(missing))


if __name__ == "__main__":
    main()
