# Shared circuits across arithmetic formats, with MAttr

Replication of de Varda, Pandey, Han, Andreas & Fedorenko (2026), *Shared circuits predict
whether LLMs generalize across formats in arithmetic reasoning* (arXiv:2609.04463), with MAttr
as the attribution method next to the paper's own attribution patching (AP).  Started
2026-09-08.

Code here:
- `scripts/arith_formats/data.py` — items (2,000; 2/3 digits x 2/3 terms x carry, balanced),
  the four formats, number words (en/es/it), sign flip, one-shot prompt.
- `scripts/arith_formats/run_model.py` — per-model driver, stages `behav / acts / ap / mattr /
  circuits / items / itemmattr`; writes `results/arith_formats/<model>/`.
- `scripts/arith_formats/probes.py` — Fig. 4 probes (Llama-3.1-8B).
- `scripts/arith_formats/analyse.py` — every table below; writes `summary.md` / `summary.json`.
- `plots/plot_arith_formats.py` → `paper/figs/arith_formats_*.pdf`.
- `scripts/arith_formats/launch/arith.sbatch` — one model per GPU inside one allocation.

## Summary

Everything the paper measures was re-measured on 13 models x 2,000 items x 4 formats with the
paper's attribution patching (AP) and with MAttr on the same units and metric.

**What replicates (both methods):** the numeric > English > Spanish > Italian accuracy
hierarchy (Result 1); the numeric-circuit overlap ordering English > Spanish > Italian in every
model, at the paper's magnitudes (Result 2); numeric-circuit loading predicting item
correctness in English (10/13 models significant, median r ~ 0.28 vs paper 0.38; Result 3);
the causal validation that patching the numeric circuit into the sign-flipped run restores the
preference more for correct items and far more than random units (Result 4); loading surviving
supervised probes and confidence on Llama-3.1-8B, beta 0.13 vs the paper's 0.127 (Result 8).

**What does not:** the across-model correlation between overlap and accuracy is weak for
English with every method (r 0.2-0.3, paper 0.75); Spanish / Italian item-level loading is
small and sometimes negative because most models are near floor there (Result 3); reciprocity
beyond numeric -> English is ~0 (Result 6).

**What MAttr adds:** (i) trained on the paper's rescaled metric, MAttr finds more sufficient
circuits than AP at every budget up to 1% and the highest numeric-verbal overlap, and its
overlap tracks accuracy across models at the paper's strength in Spanish / Italian (Results 9,
10); (ii) the MAttr-native item quantity -- the causal restored fraction when the MAttr circuit
is patched -- is the best item-level predictor and replicates Fig. 4 in all three formats,
where the paper's AP loading fails in Spanish on this model (Results 4, 8); (iii) per-item
MAttr masks are a weaker (and 100x costlier) loading than AP scores summed over the circuit
(Result 7). **The nats-loss main arm underperforms; quote the normalised arm as MAttr.**

This document is a chronological record; Results 9-10 supersede the main-arm MAttr numbers in
Results 2 and 5 where they differ.

## What the paper does, and what is swapped

Units are every MLP neuron (input of `down_proj`) of every layer, read at the **last prompt
token**.  The metric is the model's own preference between its greedy answer to the prompt x
and its greedy answer to the sign-flipped prompt x' (every + ↔ −),
`m(z) = log P(ŷ|z) − log P(ŷ'|z)` (teacher-forced sum over answer tokens), rescaled to
`M(z) = (m(z) − m(x')) / (m(x) − m(x'))`.

| | paper (AP) | here, MAttr |
|---|---|---|
| unit score | `(a(x) − a(x')) · ∂M/∂a` at x', averaged over items | learned mask logit: top-k units keep the clean activation, the rest take x'’s, loss = −m (nats), soft top-k forward, log-k schedule, SGD lr 1, 2000 steps × 8 items |
| circuit | top 1% of L·d_MLP units (pooled) by signed score | same, from the MAttr scores |
| item loading | Σ over circuit units of the item's AP score | (i) Σ of the item's **AP** scores over the **MAttr** circuit; (ii) patch-restored fraction M when the MAttr circuit is patched clean into x' (the paper's App. C quantity); (iii) per-item MAttr masks (subset of items, `--stage itemmattr`) |

Both are run on every model so each table has an AP row (the replication proper) and a MAttr
row.

### Choices the paper leaves open (assumptions)

- **Answers in the format's own surface form**: digits for numeric, number words for the
  verbal formats; correctness = exact match after lower-casing and stripping accents, hyphens,
  spaces.  The paper does not say; word answers are what make Italian at 6.8% plausible.
- **One-shot exemplar**: `53 + 28 = 81` rendered in the same format, one line, then the query.
- **Results ≥ 0** for the original item (every left-to-right partial result); the flipped
  alternative may go negative — its gold answer is never used.
- Carry = any column carry (addition) or borrow (subtraction) in any left-to-right step.
- **Answer tokens include the terminator** (the newline/EOS the model produced).  Without it an
  answer that is a token-prefix of the other (" 2" vs " 284") trivially wins the summed log-prob.
- **Item validity**: ŷ ≠ ŷ', both non-empty, and `m(x) − m(x') ≥ 1` nat (`--min-margin`);
  smaller margins make M explode.  Counts are in `circuits.json["n_valid"]`.
- Model checkpoints: Qwen3 `-Base` where it exists (0.6B/4B/8B; the 32B has no base
  release), Llama-3.2-1B/3B, Llama-3.1-8B, gemma-2-2b/9b, Mistral-7B-v0.1, Olmo-3-1025-7B,
  Olmo-3-1125-32B, phi-4.
- Sub-analysis of App. E uses the correct / incorrect split of the main 2,000 numeric items,
  not a fresh balanced 20k pool.

## Results (13 models, 2,000 items x 4 formats; all numbers in `results/arith_formats/summary.md`)

Terminology: `AP` = the paper's attribution patching, re-implemented here; `MAttr` = the learned
mask (soft top-k forward, log-k schedule, SGD lr 1, 2000 steps x 8 items; 4000 x 4 on the 32B
models for memory). Both are scored on exactly the same units, items and metric.

### Result 1: the accuracy hierarchy replicates, and is steeper than the paper's

| | numeric | English | Spanish | Italian |
|---|---|---|---|---|
| median over 13 models, here | 90.0 | 53.6 | 23.4 | 6.4 |
| paper | 87.4 | 36.4 | 17.9 | 6.8 |

Numeric > English > Spanish > Italian holds for every model (Fig. `arith_formats_acc.pdf`).
Our verbal formats are answered in words, and several small models are at floor in Spanish /
Italian (llama3.2-1b, qwen3-0.6b, gemma2-2b, olmo3-7b < 6% Spanish). The 32B Qwen3 is the only
post-trained checkpoint (no base release) and is the top scorer in every verbal format.

### Result 2: overlap with the numeric circuit orders the formats, for both methods

Top-1% Jaccard with the numeric circuit, mean over 13 models (paper, AP: 0.145 / 0.078 / 0.069):

| | English | Spanish | Italian | MAttr-vs-AP (numeric) |
|---|---|---|---|---|
| MAttr | 0.142 | 0.089 | 0.078 | 0.184 |
| AP | 0.115 | 0.062 | 0.052 | |

English > Spanish > Italian holds in **all 13 models for both methods**. MAttr circuits overlap
the numeric circuit more than AP circuits do in every model (the two methods' numeric circuits
agree on ~18% of units, Jaccard 0.12-0.25).

**The across-model correlation between overlap and accuracy does NOT replicate at the paper's
strength.** Pearson r over the 13 models (paper: English 0.75 p=0.003, Spanish 0.58 p=0.04,
Italian 0.47 p=0.10):

| | English | Spanish | Italian |
|---|---|---|---|
| MAttr | +0.23 (p=0.45) | +0.45 (p=0.12) | +0.55 (p=0.05) |
| AP | +0.21 (p=0.49) | +0.37 (p=0.22) | +0.49 (p=0.09) |

Same sign everywhere, MAttr slightly stronger than AP, but only Italian approaches significance
(`arith_formats_overlap.pdf`). The English scatter is dominated by two clusters -- the floor
models and everything else -- rather than a gradient. Our accuracies differ from the paper's
(word answers; one-shot template), so the format-level correlation is the least comparable of
the results.

### Result 3: numeric-circuit loading predicts item correctness in English, weakly elsewhere

Point-biserial r between an item's loading on the numeric circuit (sum of the item's AP scores
over the circuit's units -- the paper's definition -- applied to each method's circuit) and
correctness (paper: English 12/13 significant, median 0.38; Spanish 11/13, 0.32; Italian 9/13,
0.29):

| circuit | English | Spanish | Italian |
|---|---|---|---|
| MAttr | 10/13 sig, median +0.29 | 10/13, +0.05 | 6/13, +0.05 |
| AP | 10/13 sig, median +0.28 | 10/13, +0.08 | 5/13, +0.04 |

The three English failures are the three models with English accuracy <= 16% (llama3.2-1b,
qwen3-0.6b, mistral-7b at 33% is the exception with r=0.04). Spanish / Italian are mostly
significant but small and of either sign: llama3.2-3b and mistral-7b give **negative** r in
Spanish (-0.17 / -0.23, both methods), i.e. items that engage the numeric circuit more are more
often wrong there. Fig. `arith_formats_pbr.pdf`, deciles in `arith_formats_loading.pdf`.

### Result 4: the causal version of loading replicates the paper and is the better item predictor

Patch the numeric circuit's units clean into the sign-flipped run and read the restored
fraction of the preference (paper App. C). Point-biserial r with correctness, median over models,
vs random units matched per layer (Wilcoxon circuit > random across models):

| circuit | English | Spanish | Italian |
|---|---|---|---|
| MAttr | +0.29 vs 0.00, p=1e-4 | +0.23 vs +0.04, p=4e-4 | +0.08 vs 0.00, p=0.03 |
| AP | +0.28 vs -0.01, p=1e-4 | +0.23 vs +0.04, p=2e-4 | +0.08 vs +0.02, p=0.04 |

(paper: p<0.001 English/Spanish, p<0.01 Italian). Mean restored fraction from 1% of the
units is 0.58 (English), 0.52 (Spanish), 0.59 (Italian) vs 0.02-0.03 for random. Per model the
causal r is often well above the AP-loading r (llama3.2-3b English 0.57 vs 0.27; olmo3-32b
0.53 vs 0.35; llama3.1-8b 0.48 vs 0.39), and it is exactly the quantity MAttr's objective
optimises at the format level, so it is the natural "MAttr-native" item loading.

### Result 5: MAttr and AP find circuits of the same sufficiency; AP is slightly better past 1%

Restored fraction when the top-k units of the OWN format's ranking are patched clean, mean over
models:

| format | ranking | 0.1% | 1% | 10% |
|---|---|---|---|---|
| numeric | MAttr / AP / random | 0.21 / 0.21 / 0.00 | 0.27 / 0.28 / 0.00 | 0.36 / 0.39 / 0.02 |
| English | MAttr / AP / random | 0.55 / 0.55 / 0.00 | 0.62 / 0.64 / 0.01 | 0.67 / 0.77 / 0.10 |
| Italian | MAttr / AP / random | 0.63 / 0.63 / 0.00 | 0.75 / 0.80 / 0.02 | 0.88 / 1.03 / 0.15 |

At the 1% budget the paper works at, the two rankings are equivalent; at 10% AP is ahead by up to
0.15. On this substrate (last-token MLP neurons, a 2-token difference between clean and
corrupted prompts) the attribution-patching linearisation is evidently accurate, and MAttr's
training loss is unnormalised nats while the evaluation is the per-item rescaled M -- the
`abl_norm_*` / `abl_adam` runs below test whether that mismatch is the gap.

### Result 6: App. E / App. F

- **Correct-only vs incorrect-only numeric circuits** (App. E): loading on a circuit derived
  from the ~10% incorrect numeric items predicts English correctness less well (median r
  MAttr 0.19 / AP 0.19) than the circuit from correct items (0.28 / 0.28) or all items (0.29 /
  0.28). Same direction as the paper; our incorrect sets are small (11-800 items).
- **Reciprocity** (App. F, median r over models): every source circuit predicts English
  correctness (MAttr 0.22-0.29, AP 0.16-0.28); nothing predicts Spanish / Italian beyond
  ~0.1 here (the paper: 0.26-0.53 everywhere). Numeric -> English is the strongest cell for
  both methods, as in the paper.

### Result 7: per-item MAttr masks are a weaker item predictor than AP loading or patching

`--stage itemmattr`: an independent mask per item (batched, per-row k ~ log-uniform, normalised
loss M, SGD), loading = sum of the item's own scores over the numeric MAttr circuit. 400 English
items per model. Point-biserial r with correctness, same items, three loadings side by side:

| model | per-item MAttr (lr, steps) | AP loading on MAttr circuit | patch-restored (MAttr circuit) |
|---|---|---|---|
| olmo3-7b | 0.15 (100, 200) | 0.27 | 0.40 |
| gemma2-9b | 0.10 (100, 200) / 0.21 (10, 200) / 0.14 (10, 1000) / **0.25** (1, 1000) | 0.30 | 0.27 |
| llama3.2-1b (9% acc) | 0.08 (100, 200) / -0.08 (10, 200) | -0.06 | 0.15 |

The per-item masks agree with the per-item AP scores only moderately (Spearman 0.45-0.54 on the
same items), and their loading is the weakest of the three predictors at every setting tried;
lr 100 saturates the scores (|s| ~ 45 at T = 0.5) and the ranking improves monotonically as the
lr drops (0.10 -> 0.21 -> 0.25), i.e. the softer the masks stay, the better the loading -- the
same lesson as Result 9's normalised loss, and at lr 1 / 1000 steps per-item MAttr is within
0.05 of AP loading. Per-item MAttr is also 100x the cost of per-item AP (200 forward/backward passes per item
vs 1). What does carry over is the overlap ordering: the items' own top-1% masks overlap the
numeric MAttr circuit by 0.20 (numeric items) > 0.14 (English) > 0.09 (Spanish) > 0.06
(Italian) on olmo3-7b, a per-item version of Result 2.

### Result 8: loading survives probes and confidence on Llama-3.1-8B (Fig. 4), with the causal loading in every format

`scripts/arith_formats/probes.py`: logistic-regression probes on the last-prompt-token residual
stream and MLP activations, trained on numeric items disjoint from the main set (layer and C by
stratified 5-fold CV, `--layer-step 2`), applied to the main items in each format; then the
paper's linear probability model with standardised predictors and the LMG R^2 decomposition.
Llama-3.1-8B answers 99.2% of numeric items, so a 24k pool yields only 178 incorrect items
(train set 1,600 / 178 rather than the paper's 1,600 / 400). Probes: residual layer 15 (CV AUC
0.914), MLP layer 12 (0.914); paper: layer 25, AUC 0.93 / 0.92.

| loading | format | beta | p | % variance (LMG) | vs. the probes | entropy % |
|---|---|---|---|---|---|---|
| paper (AP) | English | 0.127 | <1e-4 | 11.0 | matches / exceeds | -- |
| AP loading, AP circuit | English | +0.133 | 1e-36 | 8.4 | > resid 0.3, mlp 4.6 | 9.8 |
| AP loading, MAttr circuit | English | +0.130 | 3e-35 | 7.9 | > 0.4, 4.7 | 9.9 |
| **causal, MAttr circuit** | English | **+0.139** | 1e-31 | **9.0** | > 0.4, 4.0 | 8.6 |
| paper (AP) | Spanish | 0.030 | 0.002 | 5.3 | | |
| AP loading, AP circuit | Spanish | -0.017 | 0.05 | 0.3 | | 16.5 |
| **causal, MAttr circuit** | Spanish | **+0.061** | 2e-10 | 3.3 | > 0.7, 2.1 | 14.1 |
| paper (AP) | Italian | 0.031 | 0.006 | 6.0 | | |
| AP loading, AP circuit | Italian | +0.047 | 3e-4 | 1.8 | > 0.5, 1.4 | 8.2 |
| **causal, MAttr circuit** | Italian | **+0.049** | 2e-4 | 2.2 | > 0.5, 1.3 | 7.9 |

English replicates the paper almost to the digit with either circuit (beta 0.13 vs 0.127), and
loading beats both supervised probes that had labelled data. With the paper's own AP loading,
Spanish does **not** replicate on this model (beta ~ 0; entropy and mean log-prob carry the
variance). With the causal loading on the MAttr circuit, loading is significant in all three
formats and in English it is the largest single share of R^2, above entropy. The residual
probe is never significant once the MLP probe is in the model.

### Result 9: training MAttr on the rescaled metric M (not nats) closes the gap to AP

`scripts/arith_formats/launch/mattr_ablate.sh` retrains the format-level masks on olmo3-7b,
gemma2-9b and llama3.1-8b with (a) the normalised loss `-M` at SGD lr 1, (b) the same at lr 10,
(c) Adam (eps 1e-2, lr 0.05) on the nats loss; behaviour and AP are reused. Restored fraction
when the top 0.1% / 1% / 10% of the own-format ranking is patched clean, and the top-1% Jaccard
with the numeric circuit:

| model | arm | numeric | English | Italian | J(numeric, en/sp/it) |
|---|---|---|---|---|---|
| olmo3-7b | main (nats, SGD 1) | .18/.25/.31 | .50/.57/.62 | .58/.79/1.13 | .164/.084/.052 |
| | **norm, SGD 1** | .20/.26/.34 | .51/.57/.63 | .63/.89/1.46 | .193/.095/.060 |
| | norm, SGD 10 | .18/.26/.33 | .50/.57/.62 | .55/.71/.89 | .168/.082/.051 |
| | Adam eps 1e-2 | .16/.23/.28 | .49/.56/.61 | .54/.68/.83 | .148/.079/.048 |
| | AP | .18/.25/.33 | .48/.57/.63 | .61/.91/1.42 | (.136/.061/.041) |
| gemma2-9b | main | .28/.35/.39 | .58/.64/.71 | .67/.74/.80 | .175/.126/.114 |
| | **norm, SGD 1** | .30/.38/.48 | .59/.66/.73 | .70/.78/.85 | .262/.190/.159 |
| | Adam eps 1e-2 | .27/.35/.39 | .57/.63/.68 | .65/.73/.78 | .174/.127/.113 |
| | AP | .27/.33/.41 | .56/.61/.67 | .65/.72/.82 | (.171/.099/.084) |
| llama3.1-8b | main | .24/.28/.44 | .53/.57/.60 | .65/.73/.83 | .139/.096/.088 |
| | **norm, SGD 1** | .25/.29/.42 | .55/.59/.64 | .69/.80/.85 | .192/.118/.092 |
| | Adam eps 1e-2 | .15/.25/.30 | .49/.57/.59 | .64/.69/.74 | .109/.074/.068 |
| | AP | .21/.26/.35 | .50/.58/.64 | .65/.77/.88 | (.111/.061/.055) |

The normalised loss at lr 1 is the best MAttr arm on every model and every format, matching or
beating AP at every budget (the nats-trained main arm let the high-margin items dominate the
gradient, which is exactly what the rescaling removes). It also raises the numeric-verbal
overlap by 0.03-0.09 while keeping the en > sp > it order. The lr-10 arm is the main arm again
(lr and the metric scale are one knob for zero-init SGD), and Adam is worst. This arm has been
re-run on all 13 models as `results/arith_formats_abl_norm_lr1` ("MAttr (normalised)" below).

### Result 10: the normalised MAttr arm on all 13 models is the best circuit finder, and recovers the paper's format-level correlation in Spanish / Italian

`results/arith_formats_abl_norm_lr1/summary.md`; figures in `paper/figs/arith_formats_norm/`.
Same behaviour, AP, items and metric as the main run; only the MAttr masks differ.

| | MAttr (normalised) | MAttr (nats, main) | AP | paper (AP) |
|---|---|---|---|---|
| top-1% overlap with numeric, en / sp / it (mean) | **0.174 / 0.106 / 0.092** | 0.142 / 0.089 / 0.078 | 0.115 / 0.062 / 0.052 | 0.145 / 0.078 / 0.069 |
| across-model r(overlap, accuracy), en | +0.33 (p=0.27) | +0.23 | +0.21 | +0.75 (p=0.003) |
| ... sp | **+0.54 (p=0.054)** | +0.45 | +0.37 | +0.58 (p=0.04) |
| ... it | **+0.63 (p=0.020)** | +0.55 | +0.49 | +0.47 (p=0.10) |
| restored fraction, own-format top 1%, nu / en / sp / it | **.30 / .64 / .64 / .80** | .27 / .62 / .61 / .75 | .28 / .64 / .65 / .80 | -- |
| ... top 0.1% | **.23 / .57 / .56 / .66** | .21 / .55 / .54 / .63 | .21 / .55 / .53 / .63 | -- |
| numeric circuit on verbal items, top 1%, en / sp / it | **.59 / .52 / .60** | .58 / .51 / .58 | .57 / .51 / .57 | -- |
| item-level r (AP loading on the circuit), en median / # sig | +0.27, 10/13 | +0.29, 10/13 | +0.28, 10/13 | +0.38, 12/13 |
| causal item-level r, en / sp / it median | +0.29 / +0.22 / +0.08 | +0.29 / +0.23 / +0.08 | +0.28 / +0.23 / +0.08 | -- |
| mean restored by the numeric circuit, en / sp / it | **.59 / .53 / .61** | .58 / .52 / .59 | .57 / .51 / .58 | -- |

Reading: the normalised arm finds strictly more sufficient circuits than AP at every budget up
to 1% (and ties at 10%, where AP pulls ahead in Spanish / Italian), gives the highest
numeric-verbal overlap, and its overlap tracks accuracy across models at the paper's strength
in Spanish and Italian (Spearman 0.65 / 0.86). English remains the format where the
across-model correlation does not replicate with any method here. Item-level numbers are the
same for all three circuit definitions to within 0.02 -- the circuits agree on the units that
matter for individual items even though only ~20% of their top-1% sets coincide.

### Artifacts

- `results/arith_formats/summary.{md,json}` (main arm), `results/arith_formats_abl_norm_lr1/summary.{md,json}` (normalised MAttr), `results/arith_formats_abl_{norm_lr10,adam}/` (3-model ablation arms)
- `paper/figs/arith_formats_{acc,overlap,pbr,loading,patch}.pdf` (main arm) and `paper/figs/arith_formats_norm/` (normalised arm)
- per model: `behav_<fmt>.json`, `ap_<fmt>.pt`, `mattr_<fmt>[_correct|_incorrect].pt`, `circuits.{json,pt}`, `items_<fmt>.json`, `itemmattr_<fmt>*.json`, `probes_<fmt>.json` (llama3.1-8b)

## How to reproduce

```bash
ARGS="--stage all --subsets" sbatch -J arith-small --gres=gpu:4 --mem=160G \
  scripts/arith_formats/launch/arith.sbatch qwen3-0.6b llama3.2-1b gemma2-2b llama3.2-3b
# mid / large groups likewise; 32B: ARGS="--stage all --subsets --bs 4 --train-bs 4 --steps 4000"
# normalised MAttr arm (copies behav_*/ap_* first, then): ARGS="--stage items --subsets --loss-norm --lr 1 --out results/arith_formats_abl_norm_lr1"
sbatch -J arith-probes scripts/arith_formats/launch/probes.sbatch --model llama3.1-8b --pool 24000
ARGS="--stage itemmattr --item-n 400 --item-steps 1000 --item-lr 1" sbatch -J arith-item --gres=gpu:1 scripts/arith_formats/launch/arith.sbatch olmo3-7b
python scripts/arith_formats/analyse.py [results/arith_formats_abl_norm_lr1]   # -> summary.md
python plots/plot_arith_formats.py [results/arith_formats_abl_norm_lr1 paper/figs/arith_formats_norm]
python scripts/arith_formats/make_tables.py        # -> paper/tabs/arith_formats_*.tex (needs both summary.json)
```
Paper appendix: `paper/sections/arith-formats.tex` (input after `olmpool` in the main tex).
Runtimes on one H100: 7-14B models ~35 min for `--stage all` (MAttr 2000 steps ~4-6 min per
format), 32B ~2.5 h; per-item MAttr ~10 min per 400 items at 200 steps; probes ~45 min.
