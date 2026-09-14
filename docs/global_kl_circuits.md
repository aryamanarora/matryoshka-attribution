# Global (task-free) circuits: zero ablation + KL to the unintervened model

Every other circuit experiment in this repo is *task* circuit discovery: a minimal pair, a
counterfactual, and a logit diff on one answer token. This one asks the task-free question —
**which MLP neurons and attention heads does Llama-3.1-8B need to reproduce its own next-token
distribution on ordinary text?** — and it turns out the answer is shaped very differently.

    scripts/global_kl/eval_global_kl.py     the experiment (train + sweep)
    scripts/global_kl/collect_global_kl.py  cross-arm table + rank agreement
    scripts/global_kl/fetch_fineweb_edu.py  data                -> data/fineweb_edu_200.jsonl
    scripts/global_kl/launch/submit_global_kl.sh   the four arms       -> results/global_kl
    scripts/global_kl/launch/submit_global_kl_lr.sh  the LR bracket    -> results/global_kl_lr
    scripts/global_kl/launch/global_kl.sbatch              thin runner (.venv; no TransformerLens, so no gemma2 caveat)

## What is different from eval_sva / eval_mib

**No counterfactual.** The ablation is to zero (or, with `--ablation mean`, to the corpus mean),
so the only reference is the model's own unintervened forward on the same tokens.

**The objective is KL(p_clean || p_masked) averaged over all 128 positions,** not a logit diff
on one token. It is minimized directly — lower is better, no sign flip.

**Nodes are tied over token position.** `--nodes mlp_tied+attn_head_tied` gives one score per
(layer, neuron) and per (layer, head): 32L x 14336 + 32L x 32 = **459,776** nodes on 8B. A
position-indexed node cannot be "globally important" — it is not even defined on a document of
a different length. The layout was added to `LlamaAttributionHooks` for this; `mlp_tied`
(neurons only) and `node` (MIB granularity) also work.

**The sparsity grid is two-sided** (`--sparsity-grid log_both`, the default). See below for why
a one-sided log grid cannot see this experiment at all.

## Result 1: under zero ablation there is no sparse global circuit

Damage is roughly **additive at ~0.005 nats of KL per zeroed neuron**. Zeroing the 241
lowest-ranked nodes of 459,776 already costs KL = 1.0. The best arm has to KEEP ~50% of the
network to stay under KL 1.0, and at 42% kept the model is *worse* (KL 12.5) than with
everything ablated (KL 11.0) — the curve is non-monotonic.

This is a property of zero ablation, not of any attribution method: it holds for the random
ordering too. Zeroing deletes a neuron's constant contribution along with its input-dependent
one, and every neuron has a non-zero mean activation feeding the residual stream.

Two consequences:

* **The informative regime is at the top of the grid, not the bottom.** The first round used a
  one-sided log grid whose second-highest point was 42% kept; every arm looked like noise. Hence
  `log_both`, which resolves `1 - 1e-3` as finely as `1e-3`. Do not go back to the one-sided grid.
* **`--ablation mean`** holds the constant part and removes only the variation, which is what
  "is this node doing work here?" is supposed to mean. Implemented and smoke-tested (it drops
  the all-ablated KL from 85.4 to 9.6 on a Llama-3.2-1B smoke), **not yet run at 8B**.

## Result 2: the metric is degenerate for gradient-at-clean methods

KL to the clean model is exactly 0 with exactly 0 gradient at the clean point. So **IxG is
identically zero** here — `eval_global_kl.py` refuses the arm rather than emitting a bf16 noise
ranking. Expected Gradients survives only because α~U(0,1) puts its gradient at perturbed points, and
it does not survive well: KL-AUC 11.71 against a random ordering's 11.91, flat at KL ≈ 12 at
every budget, top-1 agreement ≈ 0.

Do not read this as a verdict on Expected Gradients. It is a property of the metric — the SVA/MIB
harnesses use a logit diff, which has a perfectly good gradient at the clean point.
`--metric ce` (next-token cross-entropy on the real text) is the non-degenerate alternative if
a gradient-at-clean baseline is wanted.

## Result 3: MAttr arms, tuned

`results/global_kl` ran SGD at lr=1.0 and Adam at lr=0.05, both imported from the MIB
node-level sweeps. **Neither transfers**: the 12-point bracket in `results/global_kl_lr` puts
SGD's optimum at **lr=100** (100x off) and Adam's at **lr=0.2** (4x off). Read the round-1
optimizer gap as an artifact of that, not as a finding.

Tuned against tuned (KL-AUC, lower better / top-1-agreement-AUC, higher better):

| arm                  | KL-AUC | agr-AUC | KL @ 0.05% kept | KL @ 87% kept |
|----------------------|--------|---------|-----------------|---------------|
| MAttr + Adam lr 0.2  | **5.39** | **0.164** | **6.56**    | **0.236**     |
| MAttr + SGD lr 100   | 6.86   | 0.056   | 6.65            | 3.598         |
| Expected Gradients m=1      | 11.71  | 0.002   | 10.55           | 11.934        |
| magnitude (null)     | 11.99  | 0.016   | -               | -             |
| random ordering      | 11.91  | 0.022   | 14.93           | 0.649         |

(`m` in "Expected Gradients m=1" is the number of MC draws, `--ig-steps`. alpha is drawn per EXAMPLE,
so m=1 over 100 documents is 100 independent draws and the estimator error falls like
1/sqrt(n_examples), not 1/sqrt(m). m=1 is compute-matched to IxG at one backward per document.)

**Adam wins tuned-vs-tuned**, and the shape of the win is the interesting part: at the sparse
end the two are a tie (6.56 vs 6.65), and the entire gap is at the dense end. **SGD's ranking is
worse than a random one at the dense end at every LR tested** (KL @ 87% kept: 1.97-5.92 across
lr 0.1-10000, against random's 0.649), while every Adam run beats random there. That is a
structural difference, not a tuning one, and it has no verified mechanism yet — it is NOT a dead
tail in the score vector (SGD's scores have no exact zeros; median |s| = 0.085 at lr=100).

`--k-schedule log_both` is the obvious thing to try: plain log-k spends nearly every draw at
small k, so it only ever supervises which nodes to KEEP FIRST, never which to DROP LAST. But
Adam runs the same schedule and does not fail there, so the schedule cannot be the whole story.
Implemented and smoke-tested, not yet run at 8B.

## Result 4: ranking anatomy — the arms do not agree with each other

`scripts/global_kl/analyse_global_ranks.py` reads the saved `*_scores.pt` (CPU only, no model).

**Heads are hugely over-represented** in every real arm. They are 0.22% of nodes; in the
top-100 they are 90% (SGD), 37% (Expected Gradients), 18% (magnitude), 5% (Adam). Random sits at the
0.22% null. Unsurprising — a head writes 128 dimensions, a neuron writes one scaled column —
but it means a "top-k node circuit" is a head circuit at small k and a neuron circuit at large k.

**The arms have opposite depth biases** (% of top-1000, null = 6.2% per two-layer band):

| arm | L0-1 | L30-31 |
|---|---|---|
| Expected Gradients | **43.8%** | 7.2% |
| MAttr + Adam | 11.9% | 26.5% |
| MAttr + SGD | 14.4% | 35.1% |
| magnitude (null) | 0.3% | **65.5%** |

Expected Gradients is 7x enriched at the input end, which is the expected artifact of integrating along
the *input-embedding* path: nodes nearest the embedding see the largest activation change along
it. Both MAttr arms lean toward the output instead.

**On heads the arms are mutually uncorrelated.** Two random 256-subsets of 1024 heads overlap 64
by chance, so read these against 64:

| pair | rho (1024 heads) | top-64 | top-256 |
|---|---|---|---|
| Adam vs SGD | +0.024 | 2/64 | 59 |
| Adam vs random | -0.055 | 4/64 | 57 |
| SGD vs magnitude | +0.068 | 7/64 | 64 |
| Adam vs Expected Gradients | **+0.153** | **9/64** | **92** |
| Expected Gradients vs magnitude | +0.115 | 14/64 | 92 |

Adam and SGD differ only in optimizer — same method, objective, data and k-schedule — and share
**2 of their top 64 heads**, below what either shares with a random ranking. **Do not interpret
an individual head out of these runs.** The only above-chance pairs are Adam/Expected Gradients and
Expected Gradients/magnitude.

## Result 5: the magnitude null is REFUTED

`--method magnitude` ranks nodes by the RMS L2 norm each writes into the residual stream, with
no objective and no gradients (see `magnitude_scores`). It is the null for the additive-damage
picture in Result 1: if damage is proportional to deleted activation mass, "important" might
mean nothing more than "large".

It is not. Magnitude scores **KL-AUC 11.99 against a random ordering's 11.91** — no better than
random, and slightly worse on agreement-AUC (0.016 vs 0.022). Tuned MAttr+Adam is 5.39. So the
learned ranking is finding something real rather than re-deriving node size, and the "damage is
additive in deleted mass" observation constrains *how much* you can ablate, not *which* nodes
are worth keeping.

Its own anatomy is a clean sanity check on the method: magnitude is 65.5% concentrated in the
last two layers and 0.3% in the first two, i.e. the late layers write the biggest vectors — and
that concentration buys it nothing on the sweep.

## Result 6: what IS consistent across methods — neurons yes, heads no

Low pairwise Jaccard does not rule out a small strongly-agreed core, so test the core directly
against chance (`analyse_global_ranks.py --consensus ...`). **Compute it WITHIN component type**:
over all nodes the intersection is inflated by the fact that every arm puts heads near the top,
so an all-node top-k consensus partly measures "they all like heads", not "they all like this
head".

Across MAttr+Adam, MAttr+SGD and Expected Gradients:

| population | in top-k of all 3 | chance | enrichment |
|---|---|---|---|
| neurons, k=1,000  | 14 | 0.005 | **2,946x** |
| neurons, k=10,000 | 471 | 4.75 | **99x** |
| heads, k=64  | 0 | 0.25 | — |
| heads, k=256 | 18 | 16.0 | 1.1x |

**There is a robust consensus neuron core and no consensus head circuit.** The most consistent
head is L21.h19 (ranks #29/#66/#29 among all nodes), but heads as a class agree at chance, which
matches the pairwise result in Result 4. Do not read a head list off these runs.

The 471-neuron core is **U-shaped in depth**: layer 0 holds 19% of it (null 3.1%) and layers
29-31 hold 32% (null 9.4%). And it contains two families that a single explanation does not
cover — their magnitude (write-norm) ranks, of 458,752:

    L0.n13080   worst-of-3 #173    magnitude #400,128
    L0.n3273    worst-of-3 #237    magnitude #381,139
    L0.n8237    worst-of-3 #500    magnitude #398,584
    L0.n9308    worst-of-3 #656    magnitude #417,901
    L31.n8118   worst-of-3 #624    magnitude #18
    L29.n12010  worst-of-3 #450    magnitude #55
    L23.n13591  worst-of-3 #493    magnitude #724
    L31.n2437   worst-of-3 #578    magnitude #1,169

The layer-0 members are among the **smallest** writers in the model — bottom decile by norm —
and all three methods still rank them near the top. The late-layer members are the very largest.
So the core is not a norm ranking (39% of it is inside magnitude's own top-10k against a 2.2%
chance rate, but its most consistent members are anti-correlated with norm).

**Nothing is consistently UNIMPORTANT.** Zero nodes sit in the bottom-100 or bottom-1000 of all
three arms; bottom-10,000 is 2.7x chance and bottom-100,000 is 1.1x. The methods agree far more
about what matters than about what does not — consistent with Result 3's finding that the tail
of the SGD ranking is anti-informative.

### Correction to Result 2: Expected Gradients is not noise

Its sweep score is at random, but its ranking is not uninformative. **49.5% of the 951
adam+sgd consensus neurons are also in Expected Gradients's top-10k, against 2.2% for a noise arm**
(a 22x enrichment), and adding it as a third arm above sharpens the core rather than thinning it
at random. Read Result 2 as "this ORDERING does not reproduce the model at any budget" — which
is what a sweep measures — not as "this ranking identifies nothing". The two are compatible
because IG's top is 44% layer-0 nodes: a set can be genuinely important and still be far too
small to carry the sweep.

### Reading the tables

`KL-AUC` is a trapezoid mean against log10(sparsity) over the whole two-sided grid; it is the
reliable summary. The "smallest circuit reaching a tolerance" thresholds in
`collect_global_kl.py` are brittle at the grid resolution — Adam lr=0.05 and lr=0.2 differ by a
whole grid step at KL <= 1.0 purely because their KL at 50% kept straddles the threshold
(0.967 vs 1.048). Compare AUCs and per-budget columns, not thresholds.

## Cost and mechanics

An 8B arm is cheap: ~13 training steps/s, so 1000 steps is ~80 s, and the full sweep (42 masks
x 13 eval batches) is ~15 s on an H100. A whole arm is ~4 min including model load, and the
LR bracket's 12 cells ran concurrently in about that.

`--reuse-scores` (passed by both submit scripts) makes a resubmit reuse the learned scores on
disk and re-run only the sweep, so changing `--sparsity-grid` costs a sweep rather than a
retrain. It is verified bit-identical (KL-AUC 8.0878 on both passes of a smoke run). Delete
`results/<dir>/*_scores.pt` to force real retraining.

The KL reference is recomputed each step by a `hooker.mask = None` forward rather than cached:
one extra forward per step, but it keeps memory flat and scales to longer `--seq-len` / larger
`--n-train` than a cached `[n, L, 128256]` logprob table would.
