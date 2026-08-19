# SAE attribution experiments

SAE-latent attribution with MAttr on CausalGym, and how it compares with a gradient
baseline. Sections are independent experiments; **numbers are only comparable within a
section** (they differ in `n_eval` and in which runs they aggregate).

## 1. k-schedule and soft-vs-hard top-k pilot

### Question

Compare three matched MAttr SAE attribution configurations:

A. hard top-k + uniform k
B. hard top-k + log-k
C. soft top-k + log-k

A→B isolates the k-schedule.
B→C isolates the soft differentiable top-k relaxation.

### Main result

Values are **IIA log-AUC** (log-weighted AUC of interchange intervention accuracy over k).

| Task | Hard + uniform | Hard + log | Soft + log | log-k effect (B−A) | soft effect (C−B) |
|---|---|---|---|---|---|
| npi_ever_subj-relc | 0.5227 | 0.7684 | 0.7669 | **+0.2457** | −0.0016 |
| filler_gap_embed_3 | 0.5396 | 0.5623 | 0.5618 | **+0.0227** | −0.0005 |

- Log-k improves compact SAE attribution on both tasks, dramatically on NPI and weakly on filler-gap.
- Soft top-k gives no measurable improvement over hard/STE at fixed log-k in this two-task pilot.
- The strong/weak task pattern closely matches the historical tied-score sweep, although absolute
  values are NOT comparable because this experiment uses per-span scores.

![SAE attribution pilot](pilot_abc_two_task.png)

## 2. Multi-task objective robustness: MAttr vs I×G

A gradient-attribution baseline in the same variable set. Because the intervention
`new = a_cf + (m ⊙ (f_b − f_cf)) W_dec + m_err(ε_b − ε_cf)` is **exactly affine in the mask**,
`∂ℓ/∂m` *is* the paper's I×G score (eq. 21) with the `(h(b) − h(s))` factor already inside the
derivative; the gradient is taken at `m = 1` (the base point). The reconstruction-error node gets
eq. 21 with `H` = the error term and is deliberately not special-cased. This makes I×G an
unusually strong baseline in this parameterization — the base–source displacement is represented
exactly by the affine mask parameterization; the remaining first-order approximation comes from
the model downstream of the intervention layer.

**Setup.** Six CausalGym tasks × two seeds (0, 1) × {MAttr soft top-k + log-k, I×G} ×
{CE, logit-difference} = 48 runs. `google/gemma-2-2b`, layer 12, Gemma Scope width-16k
(`average_l0_82`), per-span SAE latent scores plus a per-span reconstruction-error node, Iso
(denoising) intervention, 4000 MAttr steps / 4000 I×G examples, `n_eval = 200`, identical
hard-top-k evaluator and k-grid throughout. Two tasks (`npi_ever_subj-relc`,
`garden_npz_v-trans`) were run first; the other four were **preselected from quantiles of the
historical 29-task log-k-effect distribution before any MAttr-vs-I×G result on them was
observed**, to avoid choosing tasks favourable to either method.

Values are IIA log-AUC, mean over the two seeds. Robustness contrast =
`[I×G(logit-diff) − I×G(CE)] − [MAttr(logit-diff) − MAttr(CE)]`.

| task | family | robustness contrast | MAttr − I×G (logit-diff) | MAttr − I×G (CE) |
|---|---|---|---|---|
| `npi_ever_subj-relc` | NPI | **+0.3582** | +0.0110 | +0.3693 |
| `npi_any_obj-relc` | NPI | **+0.2860** | +0.0405 | +0.3265 |
| `garden_npz_obj_mod` | garden-path | +0.0733 | +0.0124 | +0.0856 |
| `garden_npz_v-trans` | garden-path | +0.0621 | +0.0295 | +0.0916 |
| `agr_sv_num_subj-relc` | agreement | +0.0577 | +0.0092 | +0.0668 |
| `filler_gap_pp` | filler-gap | +0.0246 | +0.0094 | +0.0340 |

**Aggregate.**
- The robustness contrast is **positive in all 6 tasks and in both seeds** (median **+0.068**,
  range +0.025 to +0.358).
- The matched-logit-diff MAttr edge is **positive in all 6 tasks and in both seeds**
  (median +0.012, range +0.009 to +0.041) — a consistent *small* advantage, not dominance.
- Effect size is **heterogeneous across task families**: the smaller of the two NPI robustness
  contrasts (+0.286) is ~3.9× the largest non-NPI contrast (+0.073), so we report the per-task
  distribution rather than a single pooled headline number.
- Under CE, **I×G reaches IIA ≥ 0.9 in only 2 of 12 (task, seed) cells**, while MAttr reaches it
  in 9 of 12. Under logit-difference both methods reach it in 11 of 12.

![Objective robustness](sae_objective_robustness_final.png)

**Interpretation.** Across six CausalGym tasks, MAttr is consistently less sensitive than I×G to
the attribution objective. The objective-sensitivity contrast is positive in every task and both
seeds (median +0.068 IIA log-AUC). With logit-difference, I×G nearly matches MAttr, although
MAttr retains a small consistent edge on all six tasks; under cross-entropy the gap is
substantially larger. The effect size is heterogeneous across task families, with substantially
larger robustness gaps on the two NPI tasks, so we report the per-task distribution rather than a
single pooled headline number. In this overcomplete basis, what MAttr mainly buys over first-order
attribution is insensitivity to the choice of objective rather than a uniformly better ranking.

**Caveats.**
- Two seeds per task; `filler_gap_pp` is the weakest cell (+0.025 mean, seed values +0.033/+0.016)
  and its effect is only ~1.4× its seed spread, so it is positive but not individually resolved.
- One SAE, one layer, one model; only CE and logit-difference were tested (not soft accuracy).
- **Do not pool these `n_eval = 200` values with the `n_eval = 80` k-schedule numbers above** —
  different held-out sets *and* different training streams.

## 3. Ranking stability across objectives

Section 2 measures whether the *causal curve* moves when the objective changes. It does not say
whether the methods keep selecting the **same SAE variables**. Every completed run stores its full
attribution-score vector (`scores.pt`), so that question is answerable with **zero new model
compute**, using the evaluator's own ranking convention: `scores.topk(k)` on the raw signed score,
descending — no `abs()`, no normalisation.

Overlap of the top-`k` selected variables, mean over 6 tasks × 2 seeds:

| comparison | top-8 | top-16 | top-32 | top-64 | top-128 | RBO(.9) | RBO(.98) |
|---|---|---|---|---|---|---|---|
| **MAttr CE↔logit-diff** | 0.885 | 0.870 | **0.844** | 0.729 | 0.649 | 0.885 | 0.795 |
| **I×G CE↔logit-diff** | 0.625 | 0.615 | **0.534** | 0.512 | 0.511 | 0.573 | 0.549 |
| MAttr↔I×G @ logit-diff | 0.708 | 0.745 | 0.773 | 0.750 | 0.760 | 0.768 | 0.756 |
| MAttr↔I×G @ CE | 0.510 | 0.583 | 0.539 | 0.533 | 0.537 | 0.483 | 0.529 |

RBO is rank-biased overlap, `(1−p)·Σ_d p^(d−1)·|A_d∩B_d|/d`, truncated at depth 256.

- **MAttr keeps 0.844 of its top-32 variables when the objective changes; I×G keeps 0.534.**
- The ordering holds task by task and seed by seed: the MAttr−I×G stability gap is positive in
  23 of 24 (task, seed, k∈{8,64}) cells; the single exception is an exact tie, not a reversal.
- The effect is **concentrated near the top of the ranking**. MAttr's overlap decays with depth
  (0.885 → 0.649 from k=8 to k=128) while I×G's is flat near 0.51, so at large `k` the two are
  much closer. This is a top-of-ranking phenomenon, not a whole-vector one.
- The two methods also agree with each other far more under logit-difference (0.773 at k=32) than
  under cross-entropy (0.539) — the same objective that separates them causally.
- Causal robustness and ranking stability are **associated**, not shown to be causally linked: we
  have no intervention establishing that ranking change produces IIA loss, and a common cause
  (a weaker CE gradient signal) is equally consistent with the data.

**Reconstruction-error node.** It is genuinely high-ranked — in the top-8 in 12/12 MAttr cells
under both objectives and in the top-64 in all 48 — so it is retained, matching the evaluator.
Excluding it shifts the overlaps by at most 0.052 and changes no conclusion
(`--exclude-error-node` runs this check).

**Why not full-vector rank correlation.** I×G leaves ~116k of ~131k coordinates at exactly zero
(JumpReLU sparsity), so a whole-vector Spearman is dominated by an arbitrary tie ordering over
variables no method would ever select: it returns ≈0.76–0.81 for *all four* comparisons above and
even ranks I×G as the more stable method. Top-heavy metrics are required here, not merely
preferred.

![Causal and ranking robustness](sae_causal_and_ranking_robustness.png)

*Causal and ranking robustness across attribution objectives. Left: change in IIA log-AUC between
the cross-entropy and logit-difference objectives (smaller = more stable). Right: overlap of the
top-32 SAE variables selected under the two objectives (larger = more stable). Bars are means over
two seeds; error bars span the seeds. MAttr is more stable than I×G on both measures across six
CausalGym tasks.*

## 4. Matching the paper's soft-accuracy objective

The paper's SVA+ grid evaluates three objectives: logit-difference, cross-entropy, and soft
accuracy. For compatibility we ran the third, reusing the repository's canonical
`attribution_loss(name="acc")` (`--loss acc` / `--grad-loss acc`). **Seed 0 only**, same protocol
as section 2; IIA log-AUC:

| task | MAttr CE | MAttr soft-acc | MAttr LD | I×G CE | I×G soft-acc | I×G LD |
|---|---|---|---|---|---|---|
| npi_ever_subj-relc | 0.7237 | 0.7602 | 0.7548 | 0.3397 | 0.7361 | 0.7421 |
| npi_any_obj-relc | 0.6926 | 0.6975 | 0.6947 | 0.3623 | 0.6653 | 0.6573 |
| agr_sv_num_subj-relc | 0.8253 | 0.8324 | 0.8298 | 0.7532 | 0.8162 | 0.8192 |
| garden_npz_v-trans | 0.6151 | 0.6276 | 0.6219 | 0.5221 | 0.5937 | 0.5946 |
| garden_npz_obj_mod | 0.6775 | 0.6992 | 0.6941 | 0.5940 | 0.6875 | 0.6791 |
| filler_gap_pp | 0.6548 | 0.6677 | 0.6648 | 0.6131 | 0.6574 | 0.6564 |

- **Soft accuracy tracks logit-difference for both methods on all six tasks** (|acc − LD| is
  0.001–0.008 throughout); cross-entropy remains the outlier, and far more so for I×G.
- Objective range across the three (max − min): MAttr median **0.013**, I×G median **0.083**;
  I×G's range is larger on **6/6** tasks.
- MAttr keeps a small positive matched-objective advantage under soft accuracy
  (median **+0.020**, positive 6/6), consistent with CE (+0.088) and logit-difference (+0.014).
- Ranking agrees: `acc↔LD` is the most stable objective pair for both methods
  (top-32 overlap 0.932 MAttr, 0.875 I×G), while both CE-involving pairs separate the methods.

**Structural caveat — these are not three independent objective directions.** With
`d = logit[y_base] − logit[y_source]`, logit-difference is `−d` and soft accuracy is `1 − σ(d)`
(up to the repository's additive-constant convention). Both are monotone functions of the *same*
scalar, so their per-example gradients are collinear, differing only by the positive weight
`σ'(d)`; we verified this numerically (gradient cosine similarity 1.000000 over random logits,
versus ≈0.63 between either and cross-entropy). We therefore treat soft accuracy as a
compatibility check with the paper's experimental grid rather than an independent third objective
direction — and for the same reason we did **not** run a second seed for it: a replication would
confirm the behaviour of a near-duplicate of an objective already measured at two seeds.

## 5. Why is the objective choice so consequential for I×G?

Sections 2 and 3 show *that* I×G's attributions move when the objective changes and MAttr's do
not; they do not show *why*. The natural explanation is **saturation**. I×G differentiates at the
unmasked base point `m=1`, where the two objectives have exact output gradients

```
grad_z L_CE = softmax(z) - e_base        grad_z L_LD = -e_base + e_source
```

so CE's gradient vanishes as `p_base -> 1` while the logit-difference gradient has constant norm
`sqrt(2)`. That predicts CE should be least reliable exactly where the model is most confident.

This is answerable with forward passes alone, but no stored run contains it: I×G is closed-form,
so those runs never wrote per-example losses or logits (`losses` is an empty list in all 24 I×G
cells). `collect_base_diagnostics.py` therefore **reconstructs the exact gradient stream** each
I×G run consumed. The dataset holds an isolated `random.Random(seed)` and nothing between the
eval-set draw and the gradient loop touches it, so `(task, seed)` fully determines the stream; the
script re-derives all 4000 examples and checks them against every invariant the original runs
recorded (`eval_collection_draws`, `n_eval_distinct_keys`, `n_rejected_eval_draws`,
`n_train_distinct`, `train_eval_overlap`). **All 12 streams match exactly.** It then runs only the
base prompt forward — no SAE, no hook, no backward — and stores per-example scalars. At `m=1` the
intervention is algebraically the identity,
`a_cf + (f_b - f_cf) W_dec + [(a_b - dec f_b) - (a_cf - dec f_cf)] = a_b`, so the plain base
forward *is* the I×G base point.

### The saturation hypothesis is refuted

Over all 48,000 attribution examples (6 tasks × 2 seeds × 4000):

| statistic | value |
|---|---|
| median `p_base` | 0.019 |
| p99 / max `p_base` | 0.399 / 0.695 |
| examples with `p_base > 0.9` | **0** |
| min `‖grad_z L_CE‖ / sqrt(2)` | 0.221 |
| examples with `‖grad_z L_CE‖ / sqrt(2) < 0.1` | **0** |

The saturated regime is not rare on these tasks, it is **empty** — CE's gradient norm never falls
below ~22% of the logit-difference norm. Saturation cannot be why CE behaves differently here.

### What the geometry actually is

The probability mass is somewhere else entirely: median `p_source` = 0.0008 and median
`p_other` = 0.975. Since `grad_z L_CE = p - e_base` places weight `p_j` on *every* vocabulary
token while `grad_z L_LD` is supported on `{base, source}` alone, CE's gradient is only

```
cosine(grad_z L_CE, grad_z L_LD) = 0.677     (p10 0.626, p90 0.699)
```

aligned with the causal base-vs-source direction — and this is **uniform across tasks** (0.654 to
0.686). The two objectives genuinely ask different questions of the model, but for a reason
unrelated to confidence: roughly **97% of CE's gradient mass is spent on tokens that the
interchange intervention never involves**. I×G takes one gradient at this point, so it inherits
that geometry directly and completely.

![CE gradient geometry](sae_ce_gradient_geometry.png)

This analysis measures the base point only. It does **not** explain MAttr's robustness: MAttr's
scores are the endpoint of an optimisation through thousands of *masked* points, whose operating
point these diagnostics deliberately do not touch.

### Which tasks swing most remains unexplained

The two NPI tasks are large outliers in objective sensitivity (I×G log-AUC swing +0.399 and +0.288
vs +0.033 to +0.085 elsewhere). No base-point statistic accounts for that split. Two-seed means,
with the NPI pair listed first:

| candidate | ρ | separates NPI? | gap/range | seeds |
|---|---|---|---|---|
| median `p_base` | +0.143 | no | −0.06 | 0/2 |
| median `‖grad_z L_CE‖` | −0.429 | no | −0.06 | 0/2 |
| median `p_other` | −0.257 | no | −0.02 | 0/2 |
| median CE↔LD cosine | +0.200 | no | −0.01 | 0/2 |
| ranking churn@32 | +0.551 | no | −0.00 | — |
| median `\|margin\|` | +0.486 | YES | +0.39 | 2/2 |
| *(the outcome itself)* | — | YES | +0.56 | — |

`ρ` is Spearman over n=6 and is **descriptive only**. Separation is weak evidence by itself: with
2 NPI vs 4 non-NPI a random ordering separates with probability `2/C(6,2)` = 13.3%, so across the
seven candidates there is a ~63% chance that at least one separates by luck. The normalised gap is
what discriminates, and on shuffled data a chance separation is razor-thin (~0.09) against the
outcome's own 0.56.

Only median `|margin|` separates, and it is fragile: it holds at mean/median/p25 but **fails at
p75 and p90**, and it mis-orders within groups (`filler_gap_pp` has the third-largest `|margin|`
and the *smallest* swing). We record it as a correlate, not a mechanism.

**The earlier ranking-churn explanation is also negative and is retained as such**: churn@32
correlates at ρ = +0.551 but does not separate the NPI pair at all (NPI [0.625, 0.406] vs non-NPI
[0.547, 0.406, 0.500, 0.312] — fully interleaved), so the within-group ordering carries that ρ.

Held-out (200 eval) and attribution-stream (4000) distributions agree closely — median cosine
differs by less than 0.004 on every task — so none of this is specific to the training stream.

## 6. Interpretation

In SAE space, MAttr is substantially less sensitive than I×G to switching between cross-entropy
and margin-based attribution objectives. This robustness appears both in causal evaluation
(section 2) and in the identity of the highest-ranked SAE variables (section 3). Under
margin-based objectives I×G approaches MAttr, suggesting that MAttr's advantage in this
parameterisation is primarily robustness to objective choice rather than uniform dominance. This
is plausible mechanistically: the intervention is affine in the mask, so the base–source
displacement is represented exactly by the mask parameterisation and first-order attribution is
unusually well suited to this basis.

Section 5 measures where that objective sensitivity comes from. It is **not** gradient saturation
— the saturated regime is empty across all 48,000 attribution examples. It is that cross-entropy
spends ~97% of its output-gradient mass on tokens outside the `{base, source}` pair the
intervention actually manipulates, leaving it ~0.68-cosine aligned with the causal direction on
every task. I×G takes a single gradient at that point and inherits this directly.

We do **not** claim that MAttr dominates I×G, that I×G fails in SAE space, that three independent
objectives were tested, that ranking change causes IIA degradation, that MAttr is
objective-invariant, or that the NPI-sized effect is representative of the task distribution. We
also do **not** claim to explain MAttr's robustness (section 5 measures only the base point, not
MAttr's masked operating point), nor to explain *which* tasks are most objective-sensitive — no
base-point statistic separates the NPI outliers, and the one that does (`|margin|`) is fragile.

## 7. Reproduction

All three objectives, both methods, six tasks (`--loss` drives MAttr, `--grad-loss` drives I×G;
`acc` is the paper's soft accuracy). Seeds 0 and 1 for `ce`/`logit_diff`, seed 0 only for `acc`:

```bash
TASKS="npi_ever_subj-relc npi_any_obj-relc agr_sv_num_subj-relc \
       garden_npz_v-trans garden_npz_obj_mod filler_gap_pp"
for T in $TASKS; do
  D=results/sae_variant_pilot/$T/n200
  for S in 0 1; do for L in ce logit_diff; do
    python scripts/attribute_sae.py --task syntaxgym/$T --n-eval 200 --seed $S \
        --method mattr --variant topk --k-schedule log --loss $L --steps 4000 \
        --output $D/MAttr_${L}_s$S
    python scripts/attribute_sae.py --task syntaxgym/$T --n-eval 200 --seed $S \
        --method ixg --grad-loss $L --grad-examples 4000 \
        --output $D/IxG_${L}_s$S
  done; done
  python scripts/attribute_sae.py --task syntaxgym/$T --n-eval 200 --seed 0 \
      --method mattr --variant topk --k-schedule log --loss acc --steps 4000 \
      --output $D/MAttr_acc_s0
  python scripts/attribute_sae.py --task syntaxgym/$T --n-eval 200 --seed 0 \
      --method ixg --grad-loss acc --grad-examples 4000 \
      --output $D/IxG_acc_s0
done
```

Ranking stability (section 3) reads the `scores.pt` those runs already wrote — no model compute:

```bash
python sae_pilot/analyze_ranking_stability.py --root results/sae_variant_pilot
python sae_pilot/analyze_ranking_stability.py --root results/sae_variant_pilot \
    --objectives ce acc ld --seeds 0            # soft-accuracy contrasts (section 4)
python sae_pilot/analyze_ranking_stability.py --root results/sae_variant_pilot \
    --exclude-error-node                        # reconstruction-error-node sensitivity
```

Base-point diagnostics (section 5). The collector is **forward-only** — it loads no SAE, takes no
backward pass, and reruns no attribution; it reconstructs each I×G gradient stream and asserts it
against the invariants stored by the original runs before doing any GPU work:

```bash
python sae_pilot/collect_base_diagnostics.py --device cuda \
    --grad-examples 4000 --n-eval 200 \
    --results-root results/sae_variant_pilot --out results/sae_base_diag

python sae_pilot/analyze_base_diagnostics.py \
    --diag results/sae_base_diag --root results/sae_variant_pilot \
    --figure sae_pilot/sae_ce_gradient_geometry.png
```

Collection is ~21 min on one A100 for 6 tasks × 2 seeds × (4000 attribution + 200 held-out)
examples; the analysis is pure CPU.

## Shared experimental setup

- `google/gemma-2-2b`
- layer 12 (residual stream, output of the decoder block)
- Gemma Scope width-16k SAE, `average_l0_82`
- per-span scores (one score per (span, latent), plus a per-span reconstruction-error node)
- Iso / denoising objective (top-k held clean, complement patched to the source; CE to the base label)
- 4000 steps
- seed 0
- 80 held-out evaluation examples (the k-schedule pilot only; the objective-robustness
  section above uses 200 and is not numerically comparable)
- identical hard-top-k evaluation across A/B/C — the arms differ only in how the ranking is trained

Reproduce (one arm; vary `--variant` / `--k-schedule` for the others):

```bash
python scripts/attribute_sae.py --task syntaxgym/npi_ever_subj-relc \
    --variant topk --k-schedule log --seed 0 --steps 4000 \
    --output results/sae_variant_pilot/npi_ever_subj-relc/C_soft_log_s0
```

## Implementation notes

- deterministic seeding added (`--seed`, covering Python / NumPy / torch / CUDA and the dataset);
- genuine held-out evaluation added (a fixed set of distinct examples, fixed before training);
- train/eval key overlap asserted to zero against the examples training actually consumed;
- canonical log-k and soft-top-k variants supported (`--k-schedule`, `--variant`, routed through
  `learning_to_attribute.masks.build_mask` rather than a local mask implementation);
- training distribution preserved via rejection of held-out examples — training still draws from
  the original generator, rejecting only keys in the held-out set, rather than sampling uniformly
  from a materialised pool;
- gradient baseline added over the same SAE variable set (`--method ixg`, `--grad-loss`,
  `--grad-examples`), reusing `run_intervened` and the evaluator verbatim so only the ranking
  source differs;
- MAttr training objective made configurable (`--loss {ce,logit_diff}`) via the canonical
  `learning_to_attribute.losses.attribution_loss`; `ce` is bit-identical to the previous
  hardcoded `F.cross_entropy(logits, base_label)`.

## Caveats for section 1 (k-schedule pilot)

- Only two tasks and one seed.
- `n_eval = 80`, so the evaluation resolution is 0.0125 IIA; every C−B difference observed is at or
  below one example.
- The historical sweep used **tied** scores (one score per latent, shared across positions) whereas
  these runs use **per-span** scores, so a given k is a different fraction of the unit set. Compare
  qualitative effects, not absolute AUC or k values.
- The soft-vs-hard result should be described as **"no measurable difference in this pilot"**, not as
  general equivalence — a true effect smaller than the evaluation resolution would be invisible here.
- `filler_gap_embed_3` has a high random-feature floor (IIA 0.463) and its curve has not saturated at
  the largest k on the evaluation grid, so it has little dynamic range for any method to exploit.
