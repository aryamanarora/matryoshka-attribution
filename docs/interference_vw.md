# Replicating "Characterizing interference weights in a tiny language model"

Source: Turner, Wu & Batson, *Characterizing interference weights in a tiny language
model*, Transformer Circuits Thread, 2026-08-21
(`transformer-circuits.pub/2026/interference_effectiveness_helpfulness/`). **No code was
released.** Companion to the earlier toy-model replication in
`../mask-learning-finetuning/docs/interference_toy.md`, which covers the note this one
descends from (*A Toy Model of Interference Weights*, 2025-07-29).

Code here: `scripts/vw/vw_data.py`, `vw_model.py`, `vw_train.py`, `vw_scores.py`,
`vw_mattr.py`, `vw_prune.py`, `vw_whatkept.py`, `vw_report.py`, `vw_examples.py`,
`vw_check.py`; figures `plots/plot_vw_{prune,effhelp,source}.py`; drivers
`scripts/vw_stage{1a,1b,2,3}.sh` through `scripts/vw/launch/vw.sbatch`.

## Summary

> **Figure-by-figure audit.** `interference_vw_audit.md` goes through the note's figures from
> *Effectiveness and helpfulness* onward, redraws each for the best MAttr arm
> (`scripts/vw/vw_audit.py`), and records whether the claim survives. Short version: the pruning
> and tail claims hold; "effectiveness does not isolate helpful weights" and "the model contains
> a middle regime of mixed weights" are properties of the sign-blind Fisher score and largely
> vanish under a signed learned or gradient ranking; the "half of weights are helpful" density
> floor is not a floor for any joint selection.

**This document is a chronological record, and several results correct earlier ones.** Three
of the corrections came from hypotheses raised mid-run and two of them changed the verdict,
so the superseded claims are marked in place rather than deleted. If you read one section,
read this one.

### The replication succeeds

Everything checkable against the note reproduces: the transformer (held-out loss **3.3796**
against its 3.38), the weight population (**49.0%** of weights have positive helpfulness
against its 47.6%), its motivating interference weight (three of the six largest virtual
weights out of ' IN ' point at tokens that never once follow it in 97.7M tokens), its Fisher
pruning numbers (**+0.001 / +0.052** at density 0.55 / 0.30 against +0.000146 / +0.0107), its
per-node bifurcation figure, and even a footnote (' T ' is more likely to follow ' IN ' than
' E ': 3,817 vs 3,395). A transcoder at full published spec (FVU 0.042, L0 68) adds a second
weight family. The one unstated knob, the learning-rate schedule, is recovered from the
note's own transcoder paragraph.

### The method comparison: it depends on the family, and the split is mechanistic

**Both verdicts are measured under the bias-preserving `--ablation mean` control**, which
matters because three of the four MAttr advantages found here dissolved under it.

| family | winner, controlled | margin |
|---|---|---|
| **Tokens→Logits** (one-hot source) | **Fisher effectiveness** above d=0.05 (3–90x); MAttr from d=0.05 to 0.01 | see Result 19 |
| **Features→Logits** (transcoder features, continuous source) | **MAttr (Adam)** | beats Fisher at **every** density by **1.6–4.6x**, and the exact oracle by more |

On the feature family — the basis the note trains a transcoder to obtain — a learned mask
beats both the note's metric and its ground truth, and **prunes 85% of the family while
improving held-out loss by 0.017 nats**. The MAttr-minus-Fisher gap there is 0.098 under both
ablations, i.e. none of it is bias restoration.

The two families differ in one respect that explains the split: the token family is dominated
by large **suppressive** weights whose ablation effect is exponential in `w` (helpfulness mass
2,610 : 1 positive-to-negative, against 6.5 : 1 for features). Every method that estimates an
*effect size* is mis-calibrated there, and the ranking that wins is the one that never files a
big weight low — Fisher's non-negative `w²`. Where that regime is absent, the joint objective
wins, because what is left to get right is which *set* to keep.

One caveat survives: **rank agreement with the exact oracle does not predict pruning
performance** (Expected Gradients: ρ 0.62 vs Fisher's 0.33, and it loses badly).

> **A CLAIM THIS DOCUMENT MADE AND THEN RETRACTED.** Earlier revisions argued that "no single
> MAttr configuration is good across the whole density curve, and Fisher is". That was read
> off tables and figures whose series list predated stage E, and it is **false**. The
> batch-32 arm (`E_eps1e-8_lr0.01_s12000_b32`) has no hole: on the full family, zero
> ablation, it is −0.005 / −0.002 / +0.888 / +0.982 / +0.831 / +0.752 across densities
> 0.55 → 0.002 against Fisher's +0.001 / +0.052 / +1.027 / +2.252 / +2.146 / +1.279 — better
> at every density. The "needs density-specific tuning" story was an artifact of which arms
> were in the table. Tuning still moves MAttr by **260x**, and the winning cell is
> ε=1e-8 / low lr / batch 32, which is *not* the configuration this project recommends.
> **Answered by stage J**: the full-family win does NOT survive mean ablation above density
> 0.05 (best arm +0.029 / +0.028 / +0.903 / +0.987 at d = 0.55 / 0.30 / 0.15 / 0.05 against
> Fisher's +0.001 / +0.015 / +0.280 / +0.744; MAttr wins only at d = 0.01, +0.407 vs +0.750).
> The Features→Logits win does survive it (Result 18).

### The three things worth taking away, none of which are leaderboard results

1. **A pruning curve on this family is 37–42% a measurement of bias restoration.** The direct
   path's average contribution is a fixed vector over the vocabulary — a unigram prior — that
   zero-ablation destroys and an optimised mask rebuilds. It is large enough to reverse
   method orderings, and neither the note nor our own prior work controls for it.
   `vw_prune.py --ablation mean` is a one-line control that removes it.
2. **Summed marginal helpfulness overstates a family's worth by ~78x** (91.4 nats summed
   against 1.17 measured), so density estimates obtained by summing it are not safe.
3. **Path-linear attribution under-reads suppressive weights by 40–250x above |W| ~ 3**, and
   is numerically exact below it. The ablation effect is exponential in `w`; any linear
   functional of `w` cannot represent it. This is why Expected Gradients has the best within-row
   agreement with the oracle and still loses the pruning curve — it files ~20% of the model's
   entire value in the bottom of its ranking — and why every winning MAttr arm is the
   **magnitude-blind ε = 1e-8** one this project's guidance calls broken. Counting how *often*
   a weight helps beats estimating how *much*, when how much is exponential.

### Reading order for the corrections

Results 6 and 11 state the method comparison on **untuned** MAttr arms; Result 14 corrects
them. Result 13 first claimed the k-schedule was irrelevant and is corrected in place to
"matters only in combination with ε". Result 10 contains one coincidence explicitly flagged
as **not** a finding. Result 12 rejects a plausible mechanism ("saturated gradients") with the
measurement that refutes it.

## What is replicated, and what is not

The note trains a 1L transformer, fits a transcoder to its MLP, and expands the pair into
**six families** of virtual weights (331M of them) between tokens, positions, transcoder
features and logits. It then scores every weight by two quantities:

- **effectiveness** — the magnitude of the weight's effect on the output distribution,
  measured as a second-order (Fisher) estimate of the KL between the model with and
  without it;
- **helpfulness** — the average change in loss when the weight is ablated. Positive means
  the weight was helping.

**We replicate the transformer exactly and one weight family: Tokens→Logits.** That family
is `W_TL = [W_E ; P] @ W_U`, 5120 × 4096 = **20,971,520** virtual weights, and it is the
right one to take first for three reasons the note itself gives:

1. It needs **no transcoder**, so nothing here depends on the quality of a dictionary. The
   note makes the same point: it hopes the Tokens→Logits evidence is convincing "regardless
   of [the reader's] views on the 'right' decomposition of model components".
2. Its **helpfulness has an exact closed form** (note, Appendix › *Helpfulness computations
   for weight families targeting the logits*). The note can only afford helpfulness for a
   7,765-weight sample over the whole model; on this family we compute it for **all 21M
   weights** in one streaming pass. That turns the note's expensive oracle into a complete
   ground truth.
3. It carries the note's own motivating example (' IN ' → ' utions ').

Not replicated: the transcoder and the five families that need it (Features→Logits,
Tokens→Features, the two OV families, QK), and therefore the note's feature-level
interpretability vignettes. Their qualitative content is specific to their trained model
and would not transfer token-for-token in any case.

## The model

Every architectural and optimisation detail is stated in the note's Appendix › *Training
details* and is followed verbatim in `scripts/vw/vw_model.py` / `vw_train.py`. The two
published parameter counts are asserted, not just aimed at:

| | note | ours |
|---|---|---|
| parameters | ≈2.9M | 2,883,584 (asserted) |
| excluding embed/unembed | ≈0.79M | 786,432 (asserted) |
| train tokens | ≈9.8×10⁷ (95,464 sequences) | 97,755,136 (95,464 × 1024) |
| steps × batch | 11,933 × ? | 11,933 × 8 (95,464/11,933 = 8.0 exactly) |

**The learning rate is the one knob the note does not state.** It gives β, weight decay,
autocast dtype, the gradient-clipping rule, batch, steps and token count — and, for the
*transcoder* in the very next paragraph, a full LR and schedule ("2×10⁻⁵ … linearly decayed
to zero over the final 20%"). The asymmetry looks like an omission rather than a choice, so
`vw_train.py` sweeps LR and also tests whether that same house-style decay applies here.
The acceptance test is the note's own published losses: train ≈3.33, test 3.38.

## The tokenizer reconstruction validates itself

"a 4,096-token BPE vocabulary obtained by truncating the tokenizer of the publicly released
Pleias-1.2B model to its first 4,096 token ids, keeping only the BPE merges whose inputs
and output all survive the truncation" — 3,849 of 65,289 merges survive. The check that
this is the right tokenizer is the note's own worked example: under it, ACETYLCHOLINE
tokenizes as `AC ET Y L CH OL IN E`, so the ' IN ' → ' E ' pair the note analyses falls out
of the reconstruction rather than being assumed.

(13 of the 256 byte-level base tokens fall outside the first 4,096 ids and are lost. That
is a property of the stated truncation, not a choice made here.)

## The corpus

Common Corpus (PleIAs), filtered to `language == "English" or language_type == "Code"` — the
note's "disjunction over the corpus's language/language_type metadata columns". The note
says "the corpus copy is split into ten file shards; nine are used for training and the
tenth is held out"; the HF repo is *literally* laid out as `common_corpus_1..10`, so shards
1–9 are training and shard 10 is the held-out split. Files are taken round-robin across the
nine so the training mix is not one shard's sources.

## Both closed forms are verified, not trusted

`scripts/vw/vw_check.py` (CPU, ~20 s) recomputes Fisher effectiveness and helpfulness for
sampled weights of a tiny model by **brute force** — literally zeroing the weight in `W_TL`,
re-running the forward, and taking the mean loss difference — and requires agreement.

One thing that test taught us, recorded because it would otherwise look like a bug in the
formula: a single Tokens→Logits weight moves the mean loss by ~1e-9 nats, which a
**float32** mean of ~4 nats cannot resolve. The first version of the test failed at 72%
relative error with the closed form exact to 5 digits *per position*. The brute-force
reference must be float64.

## What we expect to find, written down before the runs

The comparison this replication exists to make is between the note's **marginal** scores and
a **joint** one. Fisher effectiveness and helpfulness both ask "what does removing this one
weight do to the full model". The pruning curve asks "which *set* of weights can be
removed". The note flags the gap once, in a footnote — "Up to nonlinear effects in removing
multiple weights at once" — and then reasons across it, concluding

> We were unable to reach a highly sparse and interpretable model using Fisher effectiveness
> as the filtering criterion, and we suspect, based on our helpfulness computations, that no
> saliency scheme will perform much better.

MAttr is not a saliency scheme; it optimises the selection directly. So three things are
predictions rather than assumptions, and are recorded here before the numbers exist:

1. **MAttr should beat Fisher on the loss-vs-density curve**, by more at low density.
2. **MAttr may beat the helpfulness ORACLE at low density.** That is the sharp one: it
   would mean the marginal/joint gap is not a technicality but the dominant effect, in a
   trained transformer rather than a toy MLP.
3. **If (2) happens, the mechanism should be the one the toy-model replication found** — the
   kept set standing in for a constant the mask cannot write down. Here that constant has a
   name: the direct path's average contribution to logit `j` is a fixed vector over the
   vocabulary, i.e. a **unigram prior**. `scripts/vw/vw_whatkept.py` tests this directly by
   replacing each kept set with its mean effect and re-measuring.

If (1) fails, MAttr has nothing to add here and that is the finding. If (2) and (3) both
hold, the honest reading is the toy note's: a top-k mask over a weight family is a *model
class*, not a *lens*, and a sparsity curve that looks good for the bias-restoration reason
is measuring capacity, not attribution.

## One structural fact about ERA on this family

`era = |W_ij| · P(source i active)` is constant in `i` within a source row, so **ERA and raw
weight magnitude induce exactly the same ordering within any one source's weights.** ERA can
only re-rank *across* sources. Anywhere the note shows a per-node ranking for this family,
an ERA column would be a copy of the magnitude column. (The note's ERA/TWERA discussion is
about families with a gated target, where the coactivation term does vary per pair.)
## Result 1: the transformer replicates, and the missing schedule is the whole gap

The learning rate and schedule are the only unstated knobs. Sweeping them (8 cells, ~40 s
each on one H100), held-out loss on shard 10:

| run | lr | decay | warmup | train (tail-20) | **held-out** |
|---|---|---|---|---|---|
| **lrwd_3e-3** | 3e-3 | 0.2 | 500 | 3.721 | **3.3796** |
| lrd_2e-3 | 2e-3 | 0.2 | 0 | 3.716 | 3.4107 |
| lrd_3e-3 | 3e-3 | 0.2 | 0 | 3.758 | 3.4312 |
| lrd_1e-3 | 1e-3 | 0.2 | 0 | 3.828 | 3.5612 |
| lrwd_6e-3 | 6e-3 | 0.2 | 500 | 3.984 | 3.5967 |
| lr_2e-3 | 2e-3 | — | 0 | 3.785 | 3.6027 |
| lr_1e-3 | 1e-3 | — | 0 | 3.868 | 3.6770 |
| lr_3e-3 | 3e-3 | — | 0 | 3.850 | 3.6808 |
| *note* | *not stated* | | | *3.33* | ***3.38*** |

**Held-out loss lands at 3.3796 against the note's 3.38.** The decisive knob is the
schedule, not the learning rate: at every LR, adding the linear decay-to-zero over the final
20% — *the schedule the note states for its transcoder and not for its transformer* — buys
0.07–0.25 nats, and no constant-LR cell gets below 3.60. Warmup adds a little on top. This
is the "read the source before sweeping" lesson paying off in the other direction: the
answer-shaped thing was in the adjacent paragraph.

**One honest caveat.** Our train loss (3.72) sits *above* our held-out loss (3.38), where
the note's is below it (3.33 vs 3.38). Two things contribute: our train figure is a
20-minibatch average at batch 8 (a noisy estimator that the note's single-step number also
is), and our shard-10 held-out slice is evidently easier relative to our training mix than
theirs. So we match their *test* number and do not match their *train* number, and the
corpus slices are not equivalent. The selection was made on held-out loss.

Two further ambiguities we resolved by choice, neither stated in the note: "interleaved
sin/cos table with maximum wavelength 2^16" is implemented as the Vaswani table with base
2^16 (an alternative reading puts the *longest* wavelength at 2^16, i.e. base 2^16/2π); and
data order is one shuffled permutation consumed once ("single pass with no data repetition").

## Result 2: the weight population replicates on the one family we can check

Scoring is over **all 97,659,672 training positions** — the complete corpus, so these are
exact expectations over the empirical training distribution, not Monte-Carlo estimates of
them. (The note estimates over ~537M tokens sampled from its corpus.)

| | note (six families) | ours (Tokens→Logits) |
|---|---|---|
| weights | 331,350,016 | 20,971,520 |
| **fraction with positive mean helpfulness** | **47.6%** | **49.0%** |
| dead | 12.7% | 0.21% |
| median Fisher ÷ max Fisher | ~1e-4 | 4.3e-8 |

The headline distributional claim — that helpful and harmful weights are split almost evenly
across the population — **replicates to within 1.4 points**, which is more than this
comparison had any right to give, since ours is one family of six and a different model.

"Dead" does not transfer and should not: the note's dead weights are mostly pairs whose
*feature* never coactivates with its target, and this family has no features in it. A
Tokens→Logits weight is dead only if its source token or position never appears at all,
which for a 4,096-token vocabulary over 97.7M tokens is almost nothing.
## Result 3: marginal helpfulness overstates the family by ~100x

Summed over all 20,971,520 weights, positive mean helpfulness comes to **91.35 nats/token**
(negative: 0.035). Removing the entire Tokens→Logits family at once costs of order **1
nat/token**. So marginal ablation overstates the family's worth by roughly **two orders of
magnitude**.

This is the note's own footnote — "up to nonlinear effects in removing multiple weights at
once" — made quantitative, and it matters for the note's conclusion. Helpfulness is a
faithful measure of what one weight does *given all the others are present*. It is not a
budget that can be summed, and the note's density estimates ("we'd only need 2.43% density
[for 90% of positive helpfulness mass] … 13.6% [for 99%]") are computed by summing exactly
this quantity. Our own version of those numbers, on this family, is 9.47% and 16.1%.

The mechanism is visible in the closed form. For a target `j` that is not the true next
token, ablating weight `w` changes the loss by `log(1 - p_j(1 - e^{-w}))`. That is
**asymmetric in the sign of `w`**: at `w = +2` it is about `-0.87 p_j`, but at `w = -2` it
is `+6.4 p_j`, because `e^{-w}` blows up. Removing a *suppressive* weight lets a suppressed
token back into the softmax, and that costs far more than removing an equally large
excitatory one gains. Helpfulness in this family is therefore dominated by suppression.

### The consequence: signed virtual weight is worse than random

| ranking | ρ vs helpfulness | precision@n | (base rate 0.439) |
|---|---|---|---|
| Expected attribution (\|W\|·P(source)) | **+0.403** | 0.573 | |
| Fisher effectiveness | +0.331 | 0.548 | |
| Virtual weight magnitude | +0.309 | 0.514 | |
| Random | +0.000 | 0.438 | |
| **Virtual weight, signed** | **−0.728** | 0.036 | |

Ranking by signed virtual weight descending — the ranking the *toy-model* note uses, where
it is correct because that toy's real weights are drawn from U[0,1] and are positive by
construction — is strongly **anti**-correlated with helpfulness in a real transformer, and
picks significantly-helpful weights at 3.6% against a 43.9% base rate. Sign conventions
carried over from the toy model do not survive the move to a trained model.

Note also that the three analytic rankings sit only 7–13 points above the base rate. Fisher
effectiveness is the note's own metric and it is doing real work, but on this family it is
not close to the oracle.

## Result 4: the family is dominated by the fixed position embedding

| source rows | weights | median \|W\| | median Fisher | significantly helpful | positive helpfulness mass |
|---|---|---|---|---|---|
| token (4,096) | 16,777,216 | 1.43 | 2.7e-10 | 30.1% | 1.82 |
| **position (1,024)** | 4,194,304 | **5.84** | **8.1e-7** | **98.9%** | **89.54** |

Position rows are 20% of the family and carry **98% of its positive helpfulness mass**.
This is structural, not behavioural: position enters through a *fixed*, non-learned
sinusoidal table whose rows have norm √(d_m/2) ≈ 11.3, while token embeddings are
initialised at 0.0375 and trained from there. A position row's virtual weights are
systematically ~4× larger in magnitude and ~3,000× larger in Fisher effectiveness for a
reason that has nothing to do with what they do.

**The note's model is built the same way and shows the same thing** — of its newline
feature it writes "The largest positive Tokens→Features virtual weights to this feature
originate from positions", and then spends a paragraph showing those position weights are
*misleading* (the Tokens→Features and OV position inputs cancel to a near-constant). So
this is replicated rather than an artifact introduced here; but it does mean that any
aggregate over this family is mostly a statement about the position embedding, which is why
every table above is also reported split by source type.
## How to reproduce

```bash
# corpus + tokenizer (~5 min, downloads ~2 GB of Common Corpus parquet and deletes as it goes)
uv run python scripts/vw/vw_data.py

# both closed forms against brute-force ablation (~20 s, CPU, no data needed)
uv run python scripts/vw/vw_check.py

# A: LR sweep, pick the cell matching the note's held-out loss, score all 21M weights
sbatch -J vw_A --time=01:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageA.sh        # ~8 min on one H100

# B: MAttr + gradient baselines, pruning sweeps, composition analysis, figures
sbatch -J vw_B --time=01:30:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageB.sh

# C: the token-rows-only cut
sbatch -J vw_C --time=00:40:00 --dependency=afterany:<B> scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageC.sh

# D: is MAttr's flat tail the k-schedule?  (log_both / uniform) + the lr x eps corner
#    + the granularity block
sbatch -J vw_D --time=01:20:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageD.sh

# E: is it undertraining or gradient noise?  48,000 steps; batch 32 and 64
sbatch -J vw_E --time=02:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageE.sh

# F: an independent k per BATCH ITEM (a new variant; see vw_mattr.py --k-per-item)
sbatch -J vw_F --time=02:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageF.sh

# the transcoder (full published spec, ~17 min) and the SECOND weight family
sbatch -J vw_tc --time=01:15:00 scripts/vw/launch/vw.sbatch scripts/vw/vw_transcoder.py --run results/vw/base
sbatch -J vw_G  --time=02:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageG.sh
```

The transcoder reaches **FVU 0.042 / L0 68** at the note's stated 100,000 steps x 16,384
activations, in 17 minutes on one H100.

Costs, measured: the transformer is **40 s** per training run; the full scoring pass over all
97.7M training positions is **under a minute** (~1.9M positions/s); one MAttr run over
20,971,520 mask units at 3,000 steps is **57 s**. The whole replication is well under an
hour of one H100. Nothing here needs the cluster except patience for a slot.

### Two bugs worth not rediscovering

- `torch.quantile` **refuses tensors above ~16M elements**, and this family has 21M. It
  raises `RuntimeError: quantile() input tensor is too large` rather than falling back, and
  it took out a whole stage the first time. Score diagnostics read a fixed random subsample.
- `W = A @ B` where `A`, `B` are `nn.Parameter`s is a **non-leaf** tensor, so
  `W.clone().requires_grad_(True)` stays non-leaf and its `.grad` is silently `None` after
  `backward()` -- the IG path in `vw_feat.py` died on `acc += Wv.grad`. Detach at the source.
  (Caught by a CPU smoke test before it reached the cluster, as was a 17 GB cache in the same
  file. Both smoke tests cost about two minutes and each saved a queued GPU job.)
- A `str.replace` used to patch a script **silently no-ops when the pattern does not match**.
  A `np.clip(y, 1e-5, None)` left in the plotting call therefore survived an edit that was
  meant to remove it, and drew every NEGATIVE dL — the result that a pruned model beats the
  full one, i.e. the most interesting thing in the study — flat on the zero line. It was
  caught only by checking a figure against the JSON it was drawn from. **Assert on every
  programmatic patch**, and read figures against their source numbers rather than trusting
  that an edit landed.
- Editing a **shell script while bash is executing it** shifts the interpreter's byte offset
  and can make it run garbage. This bit us mid-run on `scripts/vw/launch/vw_cpu.sh`; the fix was to
  restore the file's exact original bytes (bash was still early in it) and apply the change
  elsewhere. Python entry points are safe to edit — they are fully loaded at process start.
## Result 5: Adam's eps is a real knob at 21M mask units, and its signature is textbook

Mask learning here is over **weights** — the mask multiplies entries of `W_TL` — so the
lineage is `MaskedDelta`-style weight masking (the "base" is the fully ablated path, the
"delta" is the virtual weight matrix), not MIB's node/edge activation masking. Two axes come
with that framing: Adam's eps, and unit granularity (`weight` / `row` / `col`, per
`masks/layout.py`).

At 20,971,520 mask logits nearly every per-step gradient sits far below torch's default
`eps=1e-8`, so Adam's update degenerates to `sign(g)` and the learned score becomes a signed
*count of steps* with all effect magnitude divided out. Measured, same lr, same 3,000 steps:

| | p99/p50 of \|score\| | max \|score\| |
|---|---|---|
| Adam, ε = 1e-2 | **40.2** | 8.77 |
| Adam, ε = 1e-8 (torch default) | **4.65** | 31.83 |

Both halves of the signature are present: the dynamic range **collapses** (40 → 4.6, because
a step count has far less spread than an effect size) while the absolute magnitude
**inflates** (8.8 → 31.8, because scores now grow like lr × steps rather than tracking
anything about the model). This is the same failure recorded at 2.29M mask logits in the
parent project's neuron-scale runs, reproduced here at 9× the width. `ε = 1e-2` is the arm
to use; the default is drawn as a marked variant in every figure.
## Result 6: the headline — two regimes, and each metric owns one

Δ loss vs density on the held-out shard. Removing the whole family costs **+1.172**, and
that is the number every row should be read against: a ranking whose Δ exceeds it is worse
than keeping nothing at all.

| ranking | d=0.70 | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.02 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|---|---|
| Helpfulness (oracle) | **−0.019** | **−0.019** | **−0.018** | +0.416 | +1.708 | +1.593 | +1.450 | +1.278 |
| Fisher effectiveness | +0.000 | +0.001 | +0.051 | +1.041 | +2.250 | +2.144 | +1.893 | +1.279 |
| Expected attribution | +0.004 | +0.016 | +0.144 | +1.350 | +1.280 | +1.301 | +1.315 | +1.292 |
| Virtual weight magnitude | +0.033 | +0.127 | +0.755 | +1.243 | +1.197 | +1.189 | +1.187 | +1.182 |
| MAttr (Adam, ε=1e-8) | +0.089 | +0.090 | +0.095 | +0.954 | +1.265 | +1.142 | +1.121 | +1.113 |
| **MAttr (Adam, ε=1e-2)** | +1.352 | +1.350 | +1.351 | +1.364 | +1.229 | **+1.075** | **+0.995** | **+0.946** |
| MAttr (SGD, lr 1.0) | +1.361 | +1.359 | +1.359 | +1.372 | +1.236 | +1.081 | +0.999 | +0.949 |
| Expected Gradients | +1.432 | +1.428 | +1.429 | +1.450 | +1.349 | +1.183 | +1.084 | +0.976 |
| Random | +2.607 | +2.686 | +2.458 | +2.080 | +1.622 | +1.386 | +1.280 | +1.195 |
| *note, all six families* | | *+0.000146* | *+0.0107* | *+0.0702* | | | | |

**Fisher effectiveness replicates the note's own pruning numbers.** +0.001 at density 0.55
and +0.051 at 0.30, against the note's +0.000146 and +0.0107 — same order of magnitude and
same shape, on a different model and one family instead of six. The note's qualitative claim
holds here too: Fisher beats raw virtual weight magnitude at every density (0.051 vs 0.755 at
d=0.30), which is its central pruning result.

Beyond that the table says three things the note does not.

**(a) The oracle prunes 70% of the family for free, and *negatively*.** Ranking by
helpfulness and keeping the top 30% *lowers* held-out loss by 0.018 nats. Removing the 70%
of virtual weights whose individual ablation helps is a real improvement to the model, which
is as direct a demonstration of interference weights as this setup can produce.

**(b) Below density ≈0.02, every marginal score is worse than keeping nothing, and only the
optimised ones are not.** Fisher at d=0.02 costs +2.144 and the oracle +1.593, against
+1.172 for deleting the entire family; MAttr+Adam reaches **+0.946** at d=0.002. So the
2% most-*effective* weights are a worse model than no direct path at all, while 0.2% of
weights chosen by optimising the selection are meaningfully better than it.

This is the marginal/joint gap, and it is not a technicality. The note writes "we suspect,
based on our helpfulness computations, that no saliency scheme will perform much better".
On this family that is *true of saliency schemes* — every one of them, including the exact
oracle, is beaten at the sparse end by a method that is not one.

The mechanism is the one the sibling toy replication identified, in its transformer form:
the direct path's contribution to the logits must be **balanced**. Its average over the
corpus is a fixed vector over the vocabulary — effectively a unigram prior — and an
unbalanced subset of individually-important weights is worse than no subset at all.
`vw_whatkept.py` and the `--ablation mean` control test that directly.

**(c) MAttr fails at the dense end, badly, and it is not ties.** From d=0.7 to d=0.15
MAttr and Expected Gradients sit flat at ≈+1.35 to +1.45 — worse than keeping nothing, where Fisher
is at +0.001. Every score is non-zero (`frac_nonzero = 1.0`), so this is a real ordering,
not an artifact of untouched units ranked by index. This reproduces the parent project's
`global-kl-lrs-and-sgd-dense-end` observation on a new substrate.

**The ε trade-off is crisp and points at the mechanism.** ε=1e-8 — the arm that is *broken*
in the sense of Result 5 — is far **better** at the dense end (+0.089 vs +1.352 at d=0.7)
and worse at the sparse end. That is what the `sign(g)` degeneracy predicts: a score with
its magnitude divided out orders weights by how *often* one helps rather than by how much,
and frequency is the better criterion when you keep most of them and need the kept set
balanced. So "ε=1e-2 is the right setting" is regime-dependent here, not universal.
## Result 7: the note's worked example replicates, down to its footnote

`scripts/vw/vw_examples.py` reproduces both halves of the note's *A motivating example*.

### The interference weight

The note: "the largest [Tokens→Logits] virtual weight [from ' IN '] votes to complete the
word with ' utions '. This token **never** follows ' IN ' in the training set, so every time
this virtual weight affects the output, it only makes the model's loss worse."

Ours, source `'IN'` (row 903, active at 43,165 positions), top-6 by raw virtual weight, with
the corpus co-occurrence count measured rather than asserted:

| target | W | Fisher | helpfulness | times it followed ' IN ' |
|---|---|---|---|---|
| `'TER'` | +5.384 | 1.39e-4 | +4.31e-5 | 977 |
| `'SA'` | +4.412 | 2.53e-6 | **−1.24e-7** | 3 |
| `'DE'` | +4.293 | 2.74e-5 | +1.01e-5 | 299 |
| `'vir'` | +4.251 | 1.07e-7 | **−1.16e-8** | **0** |
| `'tig'` | +4.147 | 7.10e-8 | **−8.14e-9** | **0** |
| `'ST'` | +4.124 | 4.09e-5 | +2.56e-5 | 724 |

**Three of the six largest virtual weights out of ' IN ' point at tokens that never once
follow it in 97.7M tokens of training text, and all three have negative helpfulness.**
`'vir'` and `'tig'` are our `' utions '`. Their Fisher effectiveness is 3 orders of magnitude
below the effective weights from the same source — the note reports "around three orders of
magnitude below that of the most effective weights from the same token".

### The Fisher re-sort, and the note's footnote

The note: "Re-sorting ' IN ''s weights by Fisher effectiveness not only drops ' utions ', it
also lifts the ' E ' from ACETYLCHOLINE to second place." Ours, top-6 by Fisher:

| target | W | helpfulness | followed |
|---|---|---|---|
| `'TER'` | +5.384 | +4.31e-5 | 977 |
| `'T'` | +2.825 | +7.54e-5 | 3,817 |
| **`'E'`** | +2.948 | +6.88e-5 | 3,395 |
| `'ES'` | +3.369 | +1.93e-5 | 1,522 |
| `'D'` | +2.437 | +2.20e-5 | 6,097 |
| `'IT'` | +3.795 | +1.90e-5 | 652 |

The re-sort drops every zero-count target and lifts `'E'` — the ACETYLCHOLINE continuation —
to **third**. Every one of the six is a real continuation of ' IN ' with hundreds to
thousands of occurrences.

And the note's footnote on this figure reads: *"' T ', ranked at the top, is indeed more
likely to follow ' IN ' than ' E '."* In our corpus `'T'` follows ' IN ' **3,817** times
against `'E'`'s **3,395** — so the footnote's ordering holds here too, on a different corpus
sample and a separately trained model. That is a coincidence we did not go looking for.

### Where it does not replicate

The note's model gets the answer: "' E ' is the one continuation every path scores somewhat
highly". Ours ranks `'E'` **6th** in the summed logits (direct 8th, attention 19th, MLP
487th), so it does not actually complete ACETYLCHOLINE. Our model is weaker than theirs on
train loss (3.72 vs 3.33) and this is what that difference looks like at the token level.
The *structure* of the observation replicates — no path individually ranks `'E'` top, each
path's own maximum is a different wrong token (`'TER'` / `'ET'` / `'}}'`) — but the consensus
is not strong enough to win.

### ERA cannot re-rank within a source, as predicted

The `by era` and `by weight_abs` orderings above are **identical**, exactly as the structural
argument in the setup section says they must be: `era = |W_ij| · P(source i)` and `P(source
i)` is constant within a source row. Any per-node ERA column for this family is a copy of the
magnitude column.
## The figures, and what each one shows

`paper/figs/vw_source_IN.pdf` (`plots/plot_vw_source.py`) — the note's per-node panel for
` IN `→logits, with one extra panel per attribution method. Colour is the note's significance
convention (grey where the 95% CI on helpfulness includes zero).

- **Virtual weight, signed and magnitude**: no relationship to helpfulness. Helpful and
  harmful weights are spread across the whole range, which is the note's "Virtual weight
  shows a weak relationship to helpfulness (panels 1 and 2)".
- **Fisher effectiveness**: the note's central claim, replicated cleanly — the points
  **bifurcate into a rising helpful arm and a falling harmful arm, both widening with
  effectiveness**, with the most helpful extending furthest. The note: "Weights with
  increasing effectiveness bifurcate into helpful and harmful weights, with the most helpful
  extending to the largest effectiveness values."
- **Expected Gradients**: a tight diagonal on *both* arms. Within a single source row it is the best
  predictor of helpfulness of anything here — unsurprising once stated, since `-w ∂L/∂w` is
  literally the first-order expansion of the ablation effect that helpfulness measures
  exactly. Its weakness is across rows, not within them.
- **MAttr (both optimizers)**: a hockey stick. The two arms separate by sign as they should,
  but there is a **large grey cloud at high MAttr score** — weights the mask wants to keep
  whose individual helpfulness is not significantly different from zero. That is the
  marginal/joint gap visible in a single node: MAttr is keeping weights for what they do
  *as a set*, and the oracle, which ablates them one at a time, cannot see it.

`paper/figs/vw_effhelp.pdf` (`plots/plot_vw_effhelp.py`) — the aggregate version: helpfulness
against rank under each score, over all 21M weights, plus the cumulative
positive-helpfulness mass each ranking captures. Fisher's harmful weights appear only beyond
rank ~1e6; virtual weight magnitude has them at every rank. MAttr's mass curve rises faster
than Fisher's at the sparse end and then **plateaus around 80%**, only completing at density
≈1 — the dense-end failure of Result 6, visible as a shape.

`paper/figs/vw_prune.pdf` — Result 6's table as a figure, with the note's own three annotated
points drawn as reference stars. `vw_prune_mean.pdf` is the mean-ablation control and
`vw_prune_token.pdf` the token-rows-only cut.
## Result 8: unit granularity is the biggest lever, larger than optimizer or ε

This is mask learning over **weights**, so it inherits `masks/layout.py`'s granularity axis:
one score per virtual weight (`weight`, 20,971,520 units — the note's own unit), one per
**source** row (`row`, 5,120 — "does this token's or position's direct path to the logits
survive at all"), or one per **target** logit (`col`, 4,096).

Changing only that, at fixed optimizer, learning rate and ε:

| MAttr (Adam, lr 0.05, ε=1e-2) | d=0.70 | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.002 |
|---|---|---|---|---|---|---|
| per **weight** | +1.352 | +1.350 | +1.350 | +1.362 | +1.228 | **+0.947** |
| per **row** (source) | **+0.018** | **+0.057** | **+0.253** | **+0.641** | **+0.971** | +1.166 |
| per **col** (target logit) | +0.638 | +0.824 | +1.002 | +1.083 | +1.109 | +1.164 |

A **24× improvement at the dense end** from changing nothing but what one score governs. And
the row arm is not merely rescued — at densities 0.15 and 0.05 it is the **best ranking of
any kind measured here**, beating Fisher effectiveness (+1.027, +2.252), the helpfulness
oracle (+0.414, +1.713) and every per-weight MAttr arm. That covers the middle of the curve,
which is where the note's own operating point sits (it quotes density 0.15–0.55).

The mechanism is the balance story of Result 6 again. A subset of individually-important
*weights* need not be balanced, and an unbalanced direct path is worse than none. A subset of
*rows* cannot have that problem: each kept row contributes its whole logit vector, so the
kept set stays internally consistent by construction. `col` is the same idea applied to the
wrong axis — choosing which logits may receive a direct path is neither balanced nor
interpretable, and it performs accordingly.

**Practical reading.** For a weight family with this structure, the granularity of the mask
is worth more than any hyperparameter of the optimiser. That is a claim about where to spend
effort, and it is the opposite of where this project's tuning effort has historically gone.

## Result 9: the mean-ablation control shows how much of this is a constant

Under **zero** ablation, removing the whole Tokens→Logits family costs **+1.176** nats. Under
**mean** ablation — where each removed weight's contribution is replaced by its expectation
`W_ij · P(source i)` rather than by zero, so the family's average logit contribution survives
— it costs **+0.736**.

So **37% of what this family is worth is a fixed vector over the vocabulary**: a unigram-like
prior that the direct path installs on every token, and which zero-ablation destroys as a
side effect of pruning. Any sparsity curve measured with zero ablation is partly measuring
whether a ranking happens to rebuild that constant, which is not attribution.

This is the transformer form of the mechanism the sibling toy replication found, where
MAttr+Adam's kept off-circuit weights turned out to be standing in for a per-row bias the
mask could not write down. `scripts/vw/vw_whatkept.py` measures the same thing three further
ways for each kept set: its loss when replaced by its mean effect alone, its correlation with
log unigram frequency, and what a freely re-fitted per-target bias is still worth to it.
### And how much of MAttr's win it explains: most of it, but not all

Same rankings, same held-out split, mean ablation instead of zero (removing the whole family
now costs **+0.736** rather than +1.176):

| ranking | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|
| Fisher effectiveness | **+0.001** | **+0.015** | **+0.280** | +0.744 | +0.750 | +0.576 |
| Expected attribution | +0.016 | +0.145 | +1.303 | +0.985 | +0.729 | +0.737 |
| Helpfulness (oracle) | +0.018 | +0.015 | +0.390 | +1.211 | +1.028 | +0.844 |
| MAttr (Adam, row) | +0.057 | +0.249 | +0.632 | +0.773 | +0.731 | +0.733 |
| MAttr (Adam, ε=1e-2, weight) | +1.677 | +1.676 | +1.669 | +1.173 | +0.664 | **+0.544** |
| Expected Gradients (long) | +1.459 | +1.461 | +1.451 | +1.106 | +0.661 | +0.543 |
| I×G | +2.278 | +2.286 | +2.272 | +1.773 | +1.180 | +0.927 |

Read against the zero-ablation table, three things change and two do not.

**Most of MAttr's sparse-end advantage was bias restoration.** At d=0.002 the MAttr-minus-
Fisher gap collapses from **0.334 nats** (0.947 vs 1.281) under zero ablation to **0.032**
(0.544 vs 0.576) under mean ablation. About 90% of what looked like "MAttr finds a better
sparse circuit" was "MAttr rebuilds the constant that zero-ablation deleted". The same is
true of the row arm's win in the middle of the curve: at d=0.15 it beat Fisher 0.641 vs 1.027
with the constant destroyed, and loses to it 0.632 vs 0.280 with the constant preserved.

**A residual advantage survives, and it is real but small.** MAttr and Expected Gradients still hold
the sparse end under the control (0.544 / 0.543 against Fisher's 0.576 and the oracle's
0.844), so optimising the selection does buy something beyond the constant — roughly a tenth
of what the uncontrolled comparison suggested.

**The oracle is the biggest loser.** Under mean ablation helpfulness costs **+0.844** at
d=0.002, *worse than deleting the whole family* (+0.736), and it is beaten by raw signed
virtual weight (+0.721). Marginal helpfulness is close to useless as a sparse-selection
criterion once you stop crediting it for the constant.

**What does not change: MAttr's dense-end failure is not a bias artifact.** Per-weight MAttr
is at +1.677 at d=0.55 under the control, worse than it was without it. And Fisher's
dominance at d ≥ 0.15 is if anything cleaner here.

**The honest summary of Results 6–9.** Under the note's own ablation, MAttr wins the sparse
end and row-granularity MAttr wins the middle. Under a control that removes the confound,
Fisher effectiveness owns density ≥ 0.15 and MAttr/IG own density ≤ 0.01 by a small margin.
The note's sentence — "no saliency scheme will perform much better" — survives the control at
the densities the note actually operates at, and fails at the sparse extreme it does not
reach. The sibling toy replication's warning transfers intact: a top-k mask over a weight
family is a **model class**, not a **lens**, and any sparsity curve where the mask beats the
thing it was fitted to should be re-measured with the constant held fixed before it is
believed.
## Result 10: the free-bias diagnostic, transplanted from the toy model

The sibling replication settled its mechanism with one table: what a freely re-fitted bias is
worth to each model. A kept set that has *already* built the constant into its weights gains
nothing from a free bias; one that is missing it gains a lot. Here (4,096 free parameters —
one per target logit — fitted on train by Adam, everything else frozen, reported held-out):

| model | held-out loss | with a free per-target bias | the bias is worth |
|---|---|---|---|
| full model | 3.3721 | 3.3957 | **−0.024** |
| no direct path at all | 4.5437 | 4.0461 | **+0.498** |

The full model gains **nothing** — its direct path already supplies whatever a per-target
bias could (the small negative is the refit being fitted on train and generalising slightly
worse). Deleting the direct path costs 1.176 nats, and **0.498 of that, 42%, is recoverable
by 4,096 free numbers.** The corpus-mean effect alone, with no fitting at all, recovers 37.9%.

Now the same diagnostic applied to each ranking's kept set at density 0.02:

| kept set (d=0.02) | held-out loss | a free bias is still worth | % of kept weights in position rows | ρ(mean effect, log unigram freq) |
|---|---|---|---|---|
| Fisher effectiveness | 5.517 | **+1.513** | 84.6% | −0.22 |
| I×G | 5.451 | +0.801 | 76.9% | −0.78 |
| Virtual weight (signed) | 4.689 | +0.745 | 0.0% | −0.68 |
| Expected attribution | 4.673 | +0.615 | 76.5% | +0.37 |
| Random | 4.758 | +0.569 | 19.9% (base) | +0.47 |
| Helpfulness (oracle) | 4.965 | +0.555 | 98.5% | −0.24 |
| Expected Gradients | 4.556 | +0.428 | 83.5% | −0.51 |
| MAttr (Adam, ε=1e-2) | 4.447 | +0.370 | 84.9% | −0.32 |
| MAttr (Adam, ε=1e-2, long) | **4.343** | +0.363 | 84.4% | −0.19 |
| MAttr (Adam, lr 0.01, ε=1e-8) | 4.446 | **+0.272** | 98.3% | −0.20 |

**The ordering is the finding.** The rankings that do best on loss are exactly the ones a free
bias is worth least to — MAttr's kept sets need +0.27 to +0.37 more, Fisher's needs **+1.51**.
Fisher selects a sparse set that is badly mis-biased, and most of its 5.517 is that. This is
the toy replication's mechanism reproduced in a transformer, with the same direction and the
same interpretation: *what the optimised kept sets are for, in part, is supplying a constant
the marginal scores never think to preserve.*

### One coincidence in `whatkept.json` that is NOT a finding

For `mattr_adam_lr0.05_eps0.01_row` at d=0.02 the `loss` and `loss_meanonly` columns agree to
six decimals (4.4576282 vs 4.4576224). That looks like "this kept set acts purely as a
constant" and it is **not** true — checked directly: the set is 103 rows (89 position, 14
token), its per-position contribution has rms deviation **2.38** from its corpus mean, and
84% of positions receive nothing from it at all. Two quite different modifications happen to
cost the same mean loss on this eval set. Do not re-derive the stronger claim from the JSON.
## Result 11: on the token rows alone — the linguistically meaningful part — the note's own metric wins outright

Holding all 1,024 fixed-sinusoid position rows permanently on and pruning only the
16,777,216 token→logit weights. Removing all of them costs **+0.630**.

| ranking | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.02 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|---|
| **Helpfulness (oracle)** | **−0.020** | **−0.020** | **−0.019** | **−0.012** | **+0.005** | **+0.035** | **+0.216** |
| **Fisher effectiveness** | +0.000 | +0.002 | +0.006 | +0.019 | +0.039 | +0.068 | +0.230 |
| MAttr (Adam, lr 0.01, ε=1e-8) | +0.034 | +0.034 | +0.037 | +0.057 | +0.116 | +0.208 | +0.475 |
| MAttr (Adam, ε=1e-2, long) | +0.155 | +0.155 | +0.155 | +0.160 | +0.169 | +0.183 | +0.304 |
| Expected attribution | +0.008 | +0.039 | +0.116 | +0.305 | +0.448 | +0.518 | +0.604 |
| MAttr (Adam, ε=1e-2) | +0.217 | +0.217 | +0.217 | +0.220 | +0.227 | +0.241 | +0.345 |
| Virtual weight magnitude | +0.040 | +0.163 | +0.298 | +0.436 | +0.503 | +0.541 | +0.592 |
| I×G | +0.577 | +0.577 | +0.578 | +0.582 | +0.592 | +0.606 | +0.681 |
| Random | +0.695 | +0.912 | +0.927 | +0.806 | +0.709 | +0.668 | +0.639 |

> **SUPERSEDED IN PART BY RESULT 14.** The MAttr arms in the table above are the untuned
> ones. With the (lr x eps x schedule) grid filled in, the best MAttr arm on this same cut is
> `eps=1e-8, uniform` at **+0.008 / +0.009 / +0.014 / +0.031 / +0.062 / +0.487** across the
> six densities, i.e. Fisher's lead shrinks from 3-10x to **1.6-2.3x** in the middle of the
> curve (27x at d=0.55, 2.1x at d=0.002). The ORDERING is unchanged and stable --
> oracle > Fisher > tuned MAttr > ERA / weight / IG -- but "beats every MAttr variant by
> 3-10x" is a statement about untuned cells and should be read as "beats the best tuned MAttr
> arm by 1.6-2.3x".

**Fisher effectiveness beats every MAttr variant at every density, and the oracle beats
Fisher.** This is the cleanest cut of the problem — no fixed position embedding dominating
the aggregate, and the weights being ranked are exactly the interpretable ones (which token
votes for which continuation) — and on it the note's own cheap metric is simply the better
ranking.

It also strengthens the note's central pruning claim rather than weakening it. The note
reports 0.0107 nats at 70% sparsity over all six families; here, **keeping only the top 2% of
token→logit weights by helpfulness costs 0.005 nats of the 0.630 the family is worth, and by
Fisher 0.039.** 98% of this family can go for under 1% of its value.

**MAttr's curves are flat**, which is the diagnosis of why it loses: +0.217 at d=0.55 and
+0.227 at d=0.02 for the headline arm. Its top 0.2% already captures everything its top 55%
captures, so the ordering below the head carries almost no information. That is what a
log-uniform `k` schedule optimises for — the head — and it is exactly the failure Result 6
saw at the dense end, now visible across the whole curve once the position rows stop masking
it.

## Corrected summary of the method comparison

Results 6, 9 and 11 have to be read together, and the honest version is less flattering to
our method than Result 6 alone suggested:

| setting | who wins |
|---|---|
| full family, zero ablation (the note's own object) | MAttr at density ≤ 0.02; row-granularity MAttr in the middle; Fisher at the dense end |
| full family, **mean ablation** (constant preserved) | Fisher at density ≥ 0.15; MAttr/IG at ≤ 0.01 by 0.03 nats |
| **token rows only** (position embedding held fixed) | **Fisher at every density; the oracle better still.** Tuned MAttr is third, behind by 1.6-2.3x in the middle of the curve (Result 14) |

So: **MAttr does not beat Fisher effectiveness at attributing this model**, though once
tuned it comes within a factor of two on the clean cut rather than an order of magnitude. Its
apparent *wins* on the full family come from two confounds that the controls isolate — the fixed sinusoidal
position rows (20% of weights, 98% of the positive helpfulness mass) and the unigram-like
constant that zero-ablation destroys and an optimised mask rebuilds.

What the comparison *did* produce is a methodological result rather than a leaderboard one,
and it is the more useful of the two:

1. **A pruning curve on this family is 37–42% a measurement of bias restoration.** That is
   large enough to reverse method orderings, and neither the note nor our own prior work
   controls for it. `--ablation mean` is a one-line control that removes it.
2. **The marginal/joint gap is real but small once controlled** — 0.03 nats, not 0.33.
3. **Summed marginal helpfulness overstates a family's worth by ~78×**, so density estimates
   derived by summing it (the note's 2.43% / 13.6%) are not safe.
4. **Unit granularity moves results more than optimizer or ε** — but on the clean sub-problem
   that lever is not enough to catch a good saliency score either.

The note's own sentence stands, and our attempt to break it is the evidence: *"we suspect,
based on our helpfulness computations, that no saliency scheme will perform much better."*
On the sub-problem where attribution means something, nothing we ran performs better, and
the thing that looked like it did was measuring something else.
### The fully controlled cut: token rows AND mean ablation

Both confounds removed at once — position rows held fixed, the constant preserved. Removing
all token→logit weights costs **+0.569**.

| ranking | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.02 | d=0.002 |
|---|---|---|---|---|---|---|
| **Fisher effectiveness** | **+0.000** | **+0.002** | **+0.009** | **+0.036** | **+0.069** | +0.250 |
| Helpfulness (oracle) | +0.016 | +0.018 | +0.015 | +0.024 | +0.045 | **+0.235** |
| MAttr (Adam, lr 0.01, ε=1e-8) | +0.062 | +0.065 | +0.069 | +0.087 | +0.143 | +0.472 |
| MAttr (Adam, ε=1e-2, long) | +0.154 | +0.156 | +0.155 | +0.158 | +0.168 | +0.290 |
| MAttr (Adam, ε=1e-2) | +0.209 | +0.210 | +0.210 | +0.212 | +0.220 | +0.327 |
| Expected attribution | +0.008 | +0.039 | +0.116 | +0.305 | +0.445 | +0.572 |
| Virtual weight magnitude | +0.039 | +0.161 | +0.310 | +0.457 | +0.510 | +0.560 |
| Random | +0.697 | +0.915 | +0.922 | +0.781 | +0.668 | +0.581 |

Same verdict, cleaner: **Fisher effectiveness wins at every density from 0.55 to 0.02, by
3–10× over the best MAttr arm**, and the oracle takes the sparsest point. One thing does
change from the uncontrolled token-rows table — the oracle stops going *negative* at the
dense end (+0.016 rather than −0.019), which confirms that the "pruning 70% improves the
model" result of Result 6 was itself partly the zero-ablation constant rather than pure
interference removal. The improvement is real but smaller than the uncontrolled number said.

This is the table to quote for the method comparison. Everything else in this document is
either a replication of the note (Results 1, 2, 7) or a measurement of the confounds
(Results 3, 4, 9, 10).
## Result 12: why the good attribution methods lose the pruning curve — the inhibition blind spot

Results 6/11 say Fisher wins the loss-vs-density curve. That is true and it is also
misleading, because **rank agreement with the exact oracle says the opposite**:

| ranking | global ρ vs helpfulness | mean **within-row** ρ | across-row ρ |
|---|---|---|---|
| Fisher effectiveness | +0.332 | **+0.054** | +0.981 |
| Expected attribution | +0.404 | −0.002 | +0.944 |
| Virtual weight magnitude | +0.310 | −0.002 | +0.666 |
| Expected Gradients (long) | +0.623 | **+0.567** | +0.980 |
| I×G | +0.491 | **+0.578** | — |
| MAttr (Adam, lr 0.01, ε=1e-8) | **+0.737** | +0.402 | +0.818 |
| MAttr (Adam, ε=1e-2) | +0.583 | +0.368 | +0.986 |

**Fisher is essentially uninformative within a source row (ρ = 0.054)** — it knows which
tokens matter, not which continuation a token votes for — while Expected Gradients and MAttr are at
0.37–0.58. On the attribution task the note's metric is the *worst* of these, not the best.

### The reconciliation: a heavy tail, invisible to Spearman

What a density-0.55 prune actually reads is how much value sits in the ranking's **bottom**:

| ranking | % of total positive helpfulness in its bottom 45% |
|---|---|
| Fisher effectiveness | **0.0%** |
| MAttr (Adam, lr 0.01, ε=1e-8) | 0.1% |
| MAttr (Adam, row) | 0.2% |
| Expected Gradients (long) | **19.4%** |
| MAttr (Adam, ε=1e-2) | **21.5%** |
| Expected Gradients (3k) | **24.0%** |

Expected Gradients throws away a fifth of the model's value at d=0.55 while being the better ranking
almost everywhere else. Spearman cannot see this: rank correlation is blind to a small number
of extremely valuable weights being placed low, and the mass here is very concentrated.

### What gets misplaced, and why it is derivable

| | IG's misplaced weights (top 20k by lost helpfulness) | all weights |
|---|---|---|
| in **position** rows | **98.6%** | 20.0% |
| **W < 0** (suppressive) | **98.6%** | 44.8% |
| median \|W\| | **5.444** | 1.909 |

Globally, **suppressive weights hold 98.0% of the positive helpfulness mass but only 58.6% of
IG's score mass.** The cause is the closed form of Result 3: ablating `w` on a non-target
changes the loss by `log(1 - p_j(1 - e^{-w}))`, which is **exponential in −w**. Expected Gradients
integrates `-w ∂L/∂w` along a straight path and is linear in `w` by construction, so at
`w = -5.4` it under-reads the true effect by orders of magnitude. Fisher's `w²` is not
exponential either, but it is superlinear and non-negative, so it never files a large
suppressive weight in the tail — which is the only property the pruning curve rewards.

### The failure is exponential in |W|, and sharply localised

Under-read factor = true helpfulness ÷ Expected Gradients score, by weight magnitude:

| \|W\| bucket (W<0) | n | median helpfulness | median IG score | under-read |
|---|---|---|---|---|
| [0,1) | 2,557,104 | 1.78e-10 | 2.73e-10 | **0.7x** |
| [1,2) | 1,486,330 | 7.02e-10 | 8.31e-10 | 0.8x |
| [2,3) | 668,829 | 1.27e-09 | 1.25e-09 | 1.0x |
| [3,4) | 416,799 | 5.24e-07 | 1.29e-08 | **40x** |
| [4,5) | 438,984 | 4.44e-06 | 6.35e-08 | **70x** |
| [5,6) | 549,641 | 1.20e-05 | 9.57e-08 | **126x** |
| [6,20) | 1,622,034 | 2.31e-05 | 9.31e-08 | **248x** |
| all W>0 | 640,909 | 4.24e-08 | 1.32e-08 | 3.2x |

**Expected Gradients is numerically near-exact below |W| ~ 3 and degrades exponentially above it.**
That is the closed form: relative to a linear estimate the true effect grows as
`(e^{|w|} - 1)/|w|`, which predicts 20x at |w| = 4.5 and 44x at |w| = 5.5 against 70x and
126x measured -- the same growth law, right order, the residual factor coming from IG's
alpha-averaging visiting less-suppressed states.

**NOT saturated gradients, and it is worth being precise about which failure this is.** The
obvious story -- "a suppressive weight has driven p_j to zero, so its gradient vanishes" --
is FALSE here and was checked: the misplaced weights have a median |mean dL/dw| of 5.2e-8
against the population's 1.1e-9, i.e. gradients **46x LARGER** than typical. Nothing is
vanishing. A linear functional of `w` simply cannot represent an effect exponential in `w`,
however large the gradient is. (The note separately cites "saturated gradients" as a reason
to prefer a second-order estimate over expected attribution; that is a different problem of
theirs, and conflating the two would misattribute this one.)

**The note anticipated the right failure.** Its Fisher appendix: *"A linearized estimate,
freezing the activation gate and taking Δf = sw when the feature is active... gives 0
attribution for inhibition effects that deactivate a feature... We instead compute Δf
**counterfactually**, passing the ablated pre-activation exactly through the nonlinearity"*;
and in the main text, *"we also use a counterfactual variant for some weights that helps
handle inhibition."* They hit inhibition and left the linear regime to fix it. What is new
here is the **size**, measured against an exact oracle: ~20% of a weight family's entire
value misplaced into the bottom of the ranking, by every path-linear method we ran.

**The actionable version.** Any gradient-path attribution over logit-targeting weights in a
softmax model has this bias, and it is worst exactly where interference analysis is most
interesting — suppression. The fix is not a better optimiser or budget; it is to score the
true ablation rather than its first-order expansion, which for this family is a closed form
that costs one streaming pass (`scripts/vw/vw_scores.py`).
## Result 13: the k-schedule matters, but only in combination with ε

Result 6(c) proposed that MAttr's flat, catastrophic dense end comes from `k_schedule="log"`
drawing k log-uniformly over [1, 20971520], so that essentially every step supervises the
sparse end of the ordering. `schedules.py` ships the two-sided fix (`log_both`, half its draws
on `total - log-uniform`, i.e. on which weights to exclude LAST) and the opposite control
(`uniform`, most of its mass at large k). Run as a discriminator, full family, zero ablation:

| MAttr (Adam, lr 0.05, ε=1e-2) | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.01 |
|---|---|---|---|---|---|
| `log` (default) | +1.350 | +1.350 | +1.363 | +1.228 | +0.994 |
| `log_both` | **+1.434** | +1.435 | +1.453 | +1.338 | +1.070 |
| `uniform` | **+1.236** | +1.236 | +1.252 | +1.166 | +0.961 |
| *Fisher effectiveness* | *+0.001* | *+0.052* | *+1.027* | *+2.252* | *+1.893* |

At ε=1e-2 the hypothesis fails: `uniform` improves the dense end by 9% (1.350 → 1.236)
against a gap to Fisher of three orders of magnitude, and `log_both` — designed for precisely
this failure — makes it *worse*.

**But the schedule interacts strongly with ε, and at ε=1e-8 it is a 10x effect:**

| schedule | ε=1e-2 (lr 0.05) | ε=1e-8 (lr 0.05) |
|---|---|---|
| `log` (default) | +1.350 | +0.088 |
| `log_both` | +1.434 | +0.097 |
| **`uniform`** | +1.236 | **+0.0085** |

`uniform` + ε=1e-8 reaches **+0.0085 at d=0.55**, the best MAttr arm measured and within one
order of magnitude of Fisher's +0.0008, where the default configuration (`log`, ε=1e-2) is
1,500x away. So the honest statement is not "the k-schedule is irrelevant" but **"the
k-schedule matters only once the score update is magnitude-blind"** — which is itself
evidence for the mechanism below, since a `sign(g)` update is the one whose accumulated score
is a count of *how often* a weight was on the right side, and counts are exactly what a
schedule that samples densities uniformly makes comparable across weights.

**What does move it, by 40x, is ε and the learning rate.** `lr 0.01 / ε = 1e-8` reaches
**+0.034** at d=0.55 on the *default* `log` schedule. Same schedule, same optimizer, same
budget, same granularity — only the step-size rule differs. Taken with Result 5 (ε = 1e-8
makes Adam's update `sign(g)`, discarding gradient magnitude) and Result 12 (the misplaced
weights are large suppressive ones whose true effect is exponential while any magnitude-based
estimate is not), the surviving explanation is that **magnitude-sensitive score updates
inherit the same exponential blind spot the path-linear methods have, and a sign-only update
sidesteps it** — because "how often does including this weight help" is not fooled by an
effect size the estimator cannot represent.

That is a hypothesis, not a result: it predicts that reducing gradient *noise* (bigger
batches, more steps, a k per batch item) should help the ε=1e-2 arm only marginally, while
never closing the gap to the ε=1e-8 arm. Stages E and F test it.
## Result 14: MAttr's dense-end collapse was an untuned configuration, not a property

With the (lr x eps x schedule x granularity) block filled in — 37 rankings on the full
family, zero ablation, ranked at density 0.55:

| rank | ranking | d=0.55 | d=0.30 | d=0.15 |
|---|---|---|---|---|
| 1 | Helpfulness (oracle) | −0.0197 | −0.0186 | +0.414 |
| 2 | Fisher effectiveness | +0.0008 | +0.0516 | +1.027 |
| **3** | **MAttr (Adam, lr 0.002, ε=1e-8)** | **+0.0052** | **+0.0127** | +1.432 |
| 4 | MAttr (Adam, lr 0.05, ε=1e-8, `uniform`) | +0.0085 | +0.0157 | +0.736 |
| 5 | MAttr (Adam, lr 0.005, ε=1e-8) | +0.0158 | +0.0222 | +1.326 |
| 6 | Expected attribution | +0.0164 | +0.1455 | +1.356 |
| 7 | MAttr (Adam, lr 0.01, ε=1e-8) | +0.0338 | +0.0389 | +1.368 |
| … | MAttr (Adam, lr 0.05, ε=1e-2) — *the arm Result 6 reported* | **+1.350** | +1.350 | +1.363 |

**Tuned MAttr is third of thirty-seven at the dense end, beats expected attribution, and is
6.5x from Fisher — where the configuration Result 6 quoted was 1,500x away.** The swing from
hyperparameters alone is **260x** (lr 0.002 / ε=1e-8 at +0.0052 against lr 0.05 / ε=1e-2 at
+1.350). Result 6(c)'s "MAttr fails at the dense end" was a statement about one untuned cell,
and should be read as such.

**Every arm in the top group is ε = 1e-8 at a small learning rate.** That is the
magnitude-blind regime of Result 5 — the one this project's standing guidance calls broken,
because it discards effect size. Here discarding effect size is an *advantage*, and Result 12
says why: the effect size these estimators would need to represent is exponential in `w`, so
a score that counts how *often* a weight helps is better calibrated than one that tries to
measure how *much*.

**But no single MAttr arm is good across the whole curve**, and Fisher is. lr 0.002 / ε=1e-8
is 3rd at d=0.55 and +1.43 at d=0.15; the row-granularity arms invert that (+0.055 dense,
+0.64 at d=0.15). Choosing the MAttr configuration requires knowing the density you care
about, which a saliency score does not. That is a real practical disadvantage and it survives
the tuning.
## Result 15: batch size is the lever, not steps — and it survives the control

Result 6 and Result 11 measured MAttr at batch 8 (8,192 tokens per step) against Fisher and
helpfulness integrating analytically over all 97.7M positions. That is the axis this study
had never moved. Matched at **384M tokens seen**, on the fully controlled cut (token rows,
mean ablation):

| arm (Adam, ε=1e-8, lr 0.01) | d=0.55 | d=0.15 | d=0.05 | d=0.02 |
|---|---|---|---|---|
| 3,000 steps x batch 8 (the original) | +0.062 | +0.069 | +0.087 | +0.143 |
| 48,000 steps x batch 8 | +0.077 | +0.084 | +0.106 | +0.136 |
| **12,000 steps x batch 32** | **+0.030** | **+0.033** | **+0.041** | **+0.059** |
| *Fisher effectiveness* | *+0.000* | *+0.009* | *+0.036* | *+0.069* |

**16x more steps at batch 8 makes it slightly WORSE; 4x the batch at the same token count
makes it 2.3-2.6x BETTER.** So the earlier arms were not undertrained in the "needs more
optimisation" sense — they were noise-limited per step. The improvement survives mean
ablation, so unlike every previous MAttr gain in this study it is not bias restoration.

At this setting MAttr finally **beats Fisher at one density** under full control (+0.059 vs
+0.069 at d=0.02) and stays within 1.1-3.7x elsewhere, against 3-10x for the untuned arms.
Under ZERO ablation the same arm beats Fisher from d=0.55 to d=0.02 and goes negative
(−0.005), but ~70% of that margin is the constant, which is the fourth time in this document
that a MAttr advantage has mostly dissolved under `--ablation mean`.

## Result 16: an independent k per batch item is a null

`--k-per-item` (a new variant; see `vw_mattr.py`) draws a separate k for every example, so a
batch of 32 supervises 32 densities per step instead of one, at no extra compute. Against its
exact non-`kitem` twins, controlled cut:

| arm | d=0.55 | d=0.05 |
|---|---|---|
| ε=1e-2, lr 0.05, b32 — plain | +0.175 | +0.178 |
| ε=1e-2, lr 0.05, b32 — **per item** | +0.173 | +0.177 |
| ε=1e-8, lr 0.01, b8 — plain | +0.062 | +0.087 |
| ε=1e-8, lr 0.01, b8 — **per item** | +0.066 | +0.094 |

**About 1% in either direction, inside noise, and slightly worse on the better arm.**

That is worth stating alongside Result 15 rather than on its own, because together they
localise the problem. Batch size and per-item k both reduce variance, but only one of them
helps — so the binding noise is **example noise in the gradient estimate, not k-sampling
noise**. Three independent k-side interventions have now come back flat (`uniform` and
`log_both` in Result 13, per-item k here), and the two axes that move results are ε/lr
(260x, Result 14) and batch (2.5x, Result 15). Anyone tuning MAttr on a problem of this shape
should spend their compute there and not on the k schedule.
## Result 17: the second weight family — Features→Logits confirms the mechanism

`scripts/vw/vw_transcoder.py` trains the note's transcoder at its full published spec (100,000
steps x 16,384 activations, **FVU 0.042, L0 68**, 17 minutes on one H100), and
`vw_scores_feat.py` scores the resulting family, `W_FL = W_dec^T @ W_U`, 4,096 x 4,096 =
**16,777,216** virtual weights, over the full 97,659,672 training positions. Same two exact
closed forms; the difference is that the source activation is a **continuous** feature
activation rather than a one-hot indicator, so `|s w|` rarely reaches the regime where
`e^{-sw}` dominates.

### The exponential asymmetry, and everything that followed from it, is regime-specific

| | Tokens→Logits | Features→Logits |
|---|---|---|
| Σ positive helpfulness | 91.35 | **0.616** |
| Σ \|negative helpfulness\| | 0.035 | **0.094** |
| positive : negative | **2,610 : 1** | **6.5 : 1** |
| family's actual worth (ablate all) | 1.176 | 0.797 |
| **summed marginal ÷ actual** | **78x (overstates)** | **0.77x (UNDERSTATES)** |
| fraction helpful | 49.0% | 57.8% |
| dead | 0.21% | 0.0% |

**Result 3's 78x overcounting does not merely shrink on this family, it inverts.** That is
the strongest available evidence that it is caused by the exponential term and not by
marginal-vs-joint effects in general — and it sharpens the practical warning: summing
marginal helpfulness is unsafe *on families with large suppressive weights*, not everywhere.

### The rank-agreement / tail-safety dissociation replicates

| ranking | global ρ (tok → feat) | within-row ρ | % of value in bottom 45% (tok → feat) |
|---|---|---|---|
| Fisher effectiveness | +0.33 → **+0.14** | +0.05 → +0.15 | 0.0% → **0.4%** |
| Expected Gradients | +0.62 → +0.61 | +0.57 → +0.59 | 19.4% → **13.9%** |
| I×G | +0.49 → +0.70 | +0.58 → **+0.80** | — → **27.8%** |
| MAttr (Adam, ε=1e-8) | +0.74 → +0.39 | +0.40 → +0.36 | 0.1% → **1.1%** |

Fisher has the *worst* rank agreement with the exact oracle on this family (0.14) and the
*best* tail (0.4%); I×G has the best within-row agreement of anything measured anywhere here
(**0.80**) and the worst tail (27.8%). **The dissociation is a cross-family property, not a
quirk of Tokens→Logits.**

One prediction of Result 12 is only **partly** borne out and is recorded as such: the
path-linear blind spot should have largely vanished on a family without the large-|sw|
regime, and it shrank (19.4% → 13.9%) rather than disappearing, while I×G's got worse. So the
exponential asymmetry explains part of the tail failure, not all of it.

### And the oracle is beaten by the cheap metric

| ranking | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.01 |
|---|---|---|---|---|---|
| **Fisher effectiveness** | **+0.0007** | **+0.007** | **+0.033** | **+0.107** | **+0.289** |
| Helpfulness (oracle) | +0.094 | +0.097 | +0.111 | +0.174 | +0.330 |
| Expected attribution | +0.022 | +0.136 | +0.336 | +0.527 | +0.694 |
| Virtual weight magnitude | +0.055 | +0.170 | +0.291 | +0.473 | +0.673 |

**Fisher beats exact marginal helpfulness at every density on this family** — the ground
truth for "what does removing this one weight do" is a *worse* criterion for choosing a set
than a cheap second-order proxy. That is the marginal/joint gap of Result 3 in its starkest
form, and it is an argument for the note's own metric that the note does not make.
## Result 18: on the transcoder-feature family, MAttr beats Fisher — and it survives the control

This is the one place in this study where a MAttr advantage is real. The bias-preserving
control was run because three previous MAttr advantages dissolved under it; this one does not.

**Features→Logits, mean ablation (the control), 14 rankings, held-out.** Removing the whole
family costs **+0.806**:

| ranking | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.02 | d=0.002 |
|---|---|---|---|---|---|---|
| **MAttr (Adam, ε=1e-8, lr 0.01, batch 32, 12k)** | **−0.011** | **−0.021** | **−0.017** | **+0.027** | **+0.087** | **+0.297** |
| MAttr (Adam, ε=1e-8, lr 0.01, batch 8, 3k) | +0.008 | +0.006 | +0.030 | +0.102 | +0.170 | +0.373 |
| Fisher effectiveness | +0.003 | +0.018 | +0.057 | +0.125 | +0.198 | +0.491 |
| Helpfulness (oracle) | +0.098 | +0.089 | +0.094 | +0.161 | +0.237 | +0.504 |
| MAttr (Adam, ε=1e-2, batch 32, 12k) | +0.143 | +0.137 | +0.143 | +0.197 | +0.261 | +0.497 |

**MAttr beats Fisher at every density by 1.6–4.6x, and beats the exact helpfulness oracle by
a wide margin.** At d=0.05 the MAttr-minus-Fisher gap is **0.098 under zero ablation and
0.098 under mean ablation** — identical, i.e. *none* of this advantage is the constant. And
from d=0.55 to d=0.15 the curve is **negative**: keeping the top 15% of the family by MAttr
score *improves* held-out loss by 0.017 nats over the full model.

### Why this family and not the other one

| | Tokens→Logits | Features→Logits |
|---|---|---|
| source activation | one-hot indicator | continuous |
| helpfulness asymmetry (pos : neg) | 2,610 : 1 | 6.5 : 1 |
| summed marginal ÷ actual worth | 78x | 0.77x |
| winner, controlled | **Fisher** (MAttr only at d=0.02) | **MAttr, every density** |
| was MAttr's edge bias restoration? | yes, ~70–90% | **no, 0%** |

The token family is dominated by large suppressive virtual weights whose ablation effect is
exponential (Result 12). Every method that has to *estimate an effect size* is mis-calibrated
there, and the ranking that wins is the one that never files a big weight low — Fisher's
non-negative `w^2`. The feature family has no such regime, and there the joint objective wins,
because what is left to get right is which *set* to keep, which is what MAttr optimises and
what a marginal score cannot see. The oracle's poor showing on this family is the same point
from the other side: exact marginal helpfulness is a *worse* set-selection criterion than
either method here.

**This is the family the note's programme is about.** The transcoder exists precisely so that
the model can be read in a feature basis rather than a token basis, and this is the result on
that basis: a learned mask beats both the note's metric and its ground truth.

### The configuration matters and is not the default

The winning arm is **ε = 1e-8, lr 0.01, batch 32, 12,000 steps**. Batch 32 buys 3.8x over
batch 8 at the same setting (+0.027 vs +0.102 at d=0.05), consistent with Result 15, and
ε = 1e-2 — the setting this project recommends — is **7x worse** (+0.197). Anyone repeating
this should tune ε and batch first and ignore the k-schedule (Results 13, 16).

## Result 19: stage J — more steps do not help, a bigger batch does a little, and the dense-end gap is not a budget artefact

Stage J pushed the token-family budget past stage E's ceiling (which was stuck at ~393M tokens
for every "high budget" arm). Controlled cut (token rows, mean ablation), 1,024 held-out
sequences:

| arm (Adam, ε=1e-8) | tokens seen | d=0.55 | d=0.30 | d=0.15 | d=0.05 | d=0.02 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|---|---|
| lr 0.01, 12,000 × b32 (stage E) | 393M | +0.030 | +0.034 | +0.033 | +0.041 | +0.059 | +0.088 | +0.323 |
| lr 0.01, 48,000 × b32 | 1.57B | +0.040 | +0.044 | +0.045 | +0.055 | +0.066 | +0.086 | +0.331 |
| lr 0.005, 48,000 × b32 | 1.57B | +0.034 | +0.038 | +0.038 | +0.047 | +0.060 | +0.084 | +0.325 |
| **lr 0.01, 24,000 × b64** | 1.57B | **+0.026** | **+0.029** | **+0.028** | **+0.035** | **+0.050** | **+0.080** | +0.337 |
| *Fisher effectiveness* | *exact* | *+0.0003* | *+0.002* | *+0.009* | *+0.036* | *+0.069* | *+0.101* | *+0.250* |

**The "still descending" extrapolation of stage E was wrong.** Four times the steps at batch 32
made the arm *worse* at every density above 0.01 (+0.040 vs +0.030 at d=0.55), and halving the
learning rate only partly recovered it (+0.034). Doubling the batch to 64 at twice the steps
gave the best arm, 15% better than 12k × b32 — diminishing returns on the one lever that works
(2.4x from batch 8 to 32, 1.15x from 32 to 64). The best arm now beats Fisher from d=0.05
(+0.035 vs +0.036, marginal) to d=0.01 (+0.080 vs +0.101) and loses at d=0.002 and at every
density above 0.05, by 3x (d=0.15) to 90x (d=0.55). Fisher's dense-end lead on the token rows
is a property of the methods, not of the compute.

Under zero ablation on the full family the b64 arm is better than Fisher at every density
(−0.010 / −0.007 / +0.859 / +0.914 / +0.739 at d=0.55 / 0.30 / 0.15 / 0.05 / 0.01 vs +0.001 /
+0.052 / +1.027 / +2.252 / +1.893), and under mean ablation it loses above d=0.05 — the same
verdict as the controlled token-rows cut. Stage J also re-ran the audit (`interference_vw_audit.md`)
with this arm as the token-family best; the qualitative verdicts are unchanged and the token
numbers moved in MAttr's favour by small amounts (harmful share of a node's top-10 0.7% → 0.4%,
harmful share of the top 10% 1.5% → 0.7%).
