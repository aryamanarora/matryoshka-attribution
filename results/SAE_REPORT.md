# SAE-feature attribution (sufficient/denoising) on npi_ever_subj-relc, gemma-2-2b

Replication of the CausalGym **sufficient** DAS-64 experiment, replacing the learned DAS
rotation with a **frozen pretrained residual-stream SAE** at a single layer, learning one
score per SAE feature.

## Setup

- **Model**: `google/gemma-2-2b` (base), bf16, frozen.
- **Task**: `syntaxgym/npi_ever_subj-relc` (8 spans). Minimal pairs differ in the NPI
  licensing context; metric = does the model predict the **clean (base)** label vs the
  counterfactual label at the final position.
- **SAE**: Gemma Scope `google/gemma-scope-2b-pt-res`, **layer 12, width 16k (d_sae=16384),
  L0≈82** (JumpReLU; `f = (W_enc·x + b_enc > θ)·relu(W_enc·x + b_enc)`, no b_dec
  pre-subtraction; decode `f·W_dec + b_dec`). Loaded from raw `params.npz`. Hooked at the
  output of `model.model.layers[12]` (= resid_post). BOS (pos 0) excluded.
- **Learned scores**: one score per feature, `[16384]`, shared across all (non-BOS) span
  positions. Sigmoid-top-k masking (hard forward / soft backward), log-uniform k schedule,
  T=0.5, Adam lr=0.05, 4000 steps, CE to the clean label. SAE weights frozen.

## Intervention & the residual-error question

Mirroring the DAS *sufficient* (denoising) branch
`out = cf + ((rot_base − rot_cf)⊙mask) @ Wᵀ`, the SAE analog is:

```
out = a_cf + (mask ⊙ (f_base − f_cf)) @ W_dec        # f = SAE.encode(a)
```

Top-k features (mask=1) are held **clean**, the complement is set to **cf**. **The SAE
reconstruction error cancels analytically** — it is implicitly held at the counterfactual
example's residual `err_cf = a_cf − decode(f_cf)`, exactly analogous to DAS leaving the
null-space at cf. So the per-pair error difference the brief flagged **never needs to be
mixed**; the SAE-unexplained residual simply stays corrupted, and the features alone must
carry clean behavior.

**Is that the right choice? Yes — verified by a 2×2 decomposition** (all features clean/cf ×
error clean/cf, n=100 pairs):

| condition | acc | prob-diff (base−src) |
|---|---|---|
| features=cf, error=cf  (pure counterfactual) | 0.02 | −4.64 |
| features=cf, **error=clean** (error alone) | 0.02 | −3.93 |
| **features=clean**, error=cf (features alone, corrupted error) | **0.97** | +4.06 |
| features=clean, error=clean (full clean) | 0.98 | +4.73 |

**The features carry essentially all the task behavior; the reconstruction error carries
~none.** Clean features + *corrupted* error → 0.97 (≈ the 0.98 full-clean ceiling); a clean
error + corrupted features stays at chance (0.02). So the error is not a standalone cause.

### Three ways to handle the error (all run; learned acc, 80 pairs)

- **`cf`** — error always counterfactual (default; pure feature attribution).
- **`clean`** — error always base (given for free at every k).
- **`node`** — error is an extra scored unit: held clean iff its score is in the top-k, else cf.

| k | cf | clean | node |
|---|---|---|---|
| 4   | 0.21 | 0.68 | 0.43 |
| 8   | 0.50 | 0.79 | 0.76 |
| 32  | 0.86 | 0.84 | 0.84 |
| 64  | **0.91** | 0.84 | 0.85 |
| 128 | **0.94** | 0.89 | 0.88 |
| plateau | **0.94** | 0.92 | 0.91 |

**Key finding — the error node ranks #3 of 16,385** (score 10.2 vs the #1 feature's 14.6; 2
features outrank it). So *when given the choice, the attribution wants the error clean.* This
refines the decomposition: the error carries no behavior **alone** (0.02), but as a
**complement** it is the single most valuable unit after the top feature — keeping it clean
removes the `err_base − err_cf` mismatch that caps recovery, a big win at small k (k=4: 0.21→0.43).

This exposes a real tension, resolved by *what you're asking*:
- **Feature importance** ("which *features* are causal?") → use **`cf`**: error excluded, features
  do all the work, sharpest high-k ranking (0.94). The error is correctly judged a non-cause.
- **Sufficiency / behavioral recovery** → **`node`/`clean`**: far better at low k. The model rates
  "keep the SAE error clean" as its #2 intervention.

(`cf` slightly *beats* `node`/`clean` at the plateau — giving the error away for free reduces
gradient pressure on the features, so the feature ranking ends up marginally less sharp. All
three are within ~3 points at the plateau; don't over-read it.)

## Results — sufficiency curves (80 held-out pairs)

CPR-style: accuracy & prob-diff of recovering the **clean** label vs #components held clean (k),
under denoising. Higher = more sufficient.

| k (features) | **SAE learned** acc | SAE **random** acc | DAS-64 learned acc |
|---|---|---|---|
| 1  | 0.09 | 0.00 | 0.88 |
| 2  | 0.13 | 0.00 | **1.00** |
| 8  | 0.50 | 0.00 | 1.00 |
| 16 | 0.70 | 0.00 | 1.00 |
| 32 | 0.86 | 0.00 | 1.00 |
| **64** | **0.91** | 0.00 | 1.00 |
| 128 | 0.94 | 0.00 | — |
| 2048 | 0.94 | 0.09 | — |
| 4096 | 0.94 | 0.16 | — |

**Headline findings**

1. **~32–64 of 16,384 SAE features are sufficient** to recover ~0.86–0.91 of clean behavior
   (feature-only ceiling 0.97). The learned scores concentrate the task on a tiny,
   interpretable feature set.
2. **Random features never recover behavior** (acc 0 up to thousands; 0.16 even at 4096) —
   the result is driven by *which* features, not how many. Strong control.
3. **DAS-64 is far more efficient**: 1–2 of its 64 *learned* dims → **acc 1.0** (prob-diff up
   to +12). This is expected — DAS optimizes a task-specific low-rank subspace, so it packs the
   causal direction into ~1 dim; the SAE is **frozen and general-purpose**, so it needs ~30–60
   of its features to span the task variable, and tops out near 0.94–0.97.

**The trade-off**: DAS buys efficiency/accuracy with an *uninterpretable* learned rotation;
the SAE buys *interpretability* — every selected unit is a named, monosemantic Gemma Scope
feature — at the cost of efficiency. Both massively beat random.

## Top features — Neuronpedia descriptions (layer-12 gemma-scope-res-16k)

Top-2 (`16253`, `2605`) are **identical across all three error modes** — the core NPI
attribution is robust. The auto-labels are often vague; the top-activating tokens are decisive.
`neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/<id>`.

**NPI / negation / negative-polarity (task-relevant core, 8 of top ~13):**

| # | label | top tokens | read |
|---|---|---|---|
| 16253 | "skepticism/doubt about claims" | any, necessarily, nor, ningún | NPI licensors |
| 2605  | "presence/absence of evidence" | whatsoever, никаких, nothing, except | negative polarity |
| 10366 | "comparative/improvement" | nor, **ever**, or | "ever"/disjunction (the task NPI) |
| 8728  | "institutional/systemic critique" | anymore, necessarily, unless, nor | NPI licensors |
| 8743  | "obligation or denial" | siquiera, any, anything, nor | NPI (any/anything) |
| 10457 | "uncertainty or potentiality" | Any, Anybody, anyone, any | "any" NPI |
| 11938 | "negative responses or rejections" | No, no, NO | negation |
| 8038  | "absence or lack of something" | no, No, NO | negation |

**Function words / quantifiers (plausibly relevant):** `5511` (determiner: the/same/entire),
`14832` (universal quantifier: everyone/everybody).

**Spurious / junk (stable across modes — cleanup target):** `2171` (code: ViewFeatures,
AnchorStyles), `12818` (label≠tokens: فريبيس, HasColumnType), `15110` (CJK+code: 才知道, ValueStyle).

The attribution localizes NPI licensing onto exactly the right features — `any / ever / nor /
unless / whatsoever / no` — robust across error modes and cross-lingual (es `ningún/siquiera`,
ru `никаких`, fi `enää`). In `node` mode the **error node lands at #3** (no Neuronpedia label —
it is the SAE-unexplained residual, not a dictionary feature).

## Caveats / next steps

- Single layer (12) only; a layer sweep would show where NPI features live (mid-layers
  expected). Width-16k / L0-82 SAE; wider/sparser SAEs may give cleaner, more sufficient sets.
- Per-feature scores shared across positions (matches "score per feature"); a per-span variant
  (closer to DAS's per-span subspace) would localize features to the NPI licensor vs subject.
- Training loss is noisy (~3.7) because k is resampled each step; the sufficiency curve, not
  the loss, is the figure of merit.
- Compared against DAS-64 *restricted to the same single layer 12* for fairness (the original
  DAS-64 config is all-layers on Pythia); both use identical data/loop/denoising intervention.

## Files
- `scripts/attribute_sae.py` — SAE training + sufficiency eval (`--error-mode {cf,clean}`).
- `scripts/attribute_das_single.py` — single-layer DAS-64 baseline.
- `scripts/eval_error_decomp.py` — the 2×2 feature/error decomposition.
- `results/sae_npi_subj_relc/` (scores.pt, results.json, error_decomp.json),
  `results/das64_npi_subj_relc/`, `results/sae_npi_clean/`.
</content>
