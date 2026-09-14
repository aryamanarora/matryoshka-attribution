# Figure-by-figure audit of *Characterizing interference weights in a tiny language model*

Source: Turner, Wu & Batson (2026), the whole note: main text and appendix
(`transformer-circuits.pub/2026/interference_effectiveness_helpfulness/`). Companion to
`interference_vw.md`, which is the chronological record of the replication; this document is
organised by the note's figures instead.

**Method.** Every figure in the note that ranks or filters weights is drawn for one ranking, Fisher
effectiveness, and its caption is a claim about that ranking. On the two families whose
helpfulness we have exactly for every weight (Tokens→Logits, 16.8M token-row weights;
Features→Logits, 16.8M weights), the same figure can be drawn for any ranking. `scripts/vw/vw_audit.py`
redraws each one for the best MAttr arm in that setting and the baselines, and writes the
statistic each caption rests on to `results/vw/base/audit.json`. Figures:
`paper/figs/vw_audit_regimes.pdf` (the note's helpfulness-vs-rank panel per ranking, both
families), `paper/figs/vw_audit_curves.pdf` (harmful share of the kept set and positive
helpfulness mass captured, against density), `paper/figs/vw_source_IN.pdf` (the per-node panel),
`paper/figs/vw_prune*.pdf` (the pruning figure), `paper/figs/vw_audit_hist.pdf` (the appendix's
mean-helpfulness histogram). LaTeX: `paper/tabs/vw_audit.tex`.

"Best MAttr" is the best arm measured on that family's controlled pruning curve: Adam, lr 0.01,
ε = 1e-8; for Tokens→Logits **24,000 steps × batch 64** (`J_eps1e-8_lr0.01_s24000_b64`, stage J),
for Features→Logits 12,000 × batch 32 (`H_…`).

> **Stage J update.** The detail tables below were written against the 12k × b32 token arm
> (`E_…`) and are left as they were; `audit.json`, `paper/tabs/vw_audit.tex` and the figures
> were regenerated with the b64 arm. What changed on the token family (E → J): harmful share of
> a node's top-10 0.7% → **0.4%**, nodes with a harmful weight in their top-10 3.5% → **2.8%**,
> harmful share of the top 1% / 10% 0.3% / 1.5% → **0.2% / 0.7%**, first harmful at rank 670 →
> 575, helpful weights kept at d=0.15 36.1% → 37.5%, positive mass at d=0.15 0.986 → 0.989,
> 99%-mass density 0.847 → 0.180, rank agreement 0.930 → 0.936, practically-zero share of the
> kept set at d=0.01 17% → **8.7%** (now below Fisher's 10.8%), controlled ΔL at d=0.15 +0.033
> → **+0.028** (Fisher +0.009). The `IN` memberships are unchanged (E and T at rank 2 and 1,
> the zero-count targets held out to density > 0.78, ES/DE kept at 0.006–0.007). No verdict
> flips. Note also that 4x more steps at batch 32 made the arm *worse*; see Result 19 of
> `interference_vw.md`.

"Significantly helpful / harmful" is the note's convention: the 95% Gaussian CI on mean
helpfulness excludes zero. Everything is exact over all 97,659,672 training positions.

## The verdicts, one line each

| # | Note figure / claim | Replicates under Fisher? | Under the best MAttr | Verdict |
|---|---|---|---|---|
| 0a | *Which of these signals implement functional circuits?* (ACETYLCHOLINE path decomposition): no path ranks ' E ' top; each path's largest vote is a different wrong token | yes in structure (direct / attention / MLP maxima are ' TER ' / ' ET ' / ' }} '); our weaker model ranks ' E ' 6th overall rather than 1st | as a membership question: the best MAttr set keeps IN→E and IN→T at density < 0.001 and drops the three zero-count weights until density 0.79–0.95 | **upheld** (structure), model-specific detail differs |
| 0b | *Rewriting paths as virtual weights*: six families, 331M weights, exact forward pass | two families built (21.0M + 16.8M), both closed forms verified against brute-force ablation | n/a | **partial replication**, by design |
| 1 | *Instantiating the IN→Logit weights without interference*: sorting by effectiveness drops ' utions ' and lifts ' E ' to 2nd | yes (drops all zero-count targets; ' E ' 3rd) | yes, and cleaner: ' E ' 2nd, all top-8 are real continuations | **upheld**; MAttr matches or betters |
| 2 | *Effectiveness concentrates helpful and harmful weights*: Fisher bifurcates into a helpful and a harmful arm; \|W\| shows a weak relationship | yes | MAttr shows a hockey stick: arms separate by sign, plus a grey cloud of non-significant weights at high score | **upheld** for Fisher; MAttr's high-score cloud is the marginal/joint gap in one node |
| 3 | "Virtual weights appear approximately normally distributed, Fisher roughly lognormally, varying by 10 orders" | qualitatively; ours spans 24 orders on one family | not an attribution claim | **upheld** (n/a to MAttr) |
| 4 | Chinese feature: sometimes the largest \|W\| are helpful; "not inherently misleading, but not guaranteed" | population version: on features, a node's top-10 by \|W\| is 68% helpful / 12% harmful; on tokens 63% / 33% | MAttr: 85% / 6.5% (features), 91% / 0.7% (tokens) | **upheld**, and quantified: \|W\| is a coin flip on tokens, decent on features |
| 5 | *Harmful weights overlap with helpful ones* (Feature 3013): effectiveness "surfaces but does not isolate" — harmful weights remain intermingled at high effectiveness | yes: 12% of a feature's top-10 by Fisher is harmful, 56% of features have one; on tokens 8.6% / 31% | features: 6.5% / 43% of nodes; tokens: **0.7% / 3.5%** (12× fewer harmful, 9× fewer nodes) | **upheld for Fisher; overturned as a general limit on tokens**, halved on features |
| 6 | Newline feature: position inputs cancel; Tokens→Features weights are misleading | not built (no Tokens→Features family) | — | **not audited** |
| 7 | *The most effective weights are helpful*: three regimes (ineffective ≈ 0, a mixed middle, a helpful-only tail); most effective helpful weight out-measures any harmful one by ≥10× | yes: 514× (tokens), 43× (features); harmful share of the top 0.01% is 0 on both families (top 0.1%: 0 tokens, 0.8% features) | the mixed middle is a Fisher artefact: MAttr's harmful share of the top **10%** is 1.5% (tokens) against Fisher's 46%; harmful weights sit at the bottom of MAttr's ranking, not in a middle band | **upheld for Fisher; the "middle regime" is a property of a sign-blind score, not of the model** |
| 8 | *Fisher effectiveness can cheaply filter tens of percent*: ≈0.01 nats at density 0.30, <0.1 at 0.15; far better than \|W\| at every density | yes (+0.002 / +0.009 controlled; +0.052 / +1.03 full family zero) | full family, zero ablation: MAttr beats Fisher at every density; controlled token rows: Fisher wins above 0.05; features: MAttr wins everywhere, negative to 0.15 | **upheld** for the note's operating point on tokens under control; **overturned** on features and on the note's own uncontrolled object |
| 9 | "At 1% [density] the pruned model's performance is significantly compromised" | Fisher at 0.01: +0.10 of 0.57 (tokens, controlled), +0.27 of 0.81 (features) | MAttr at 0.01: +0.09 (tokens), +0.14 (features) — 15–17% of the family's value | **qualified**: at 1% density MAttr keeps 83–85% of what the family is worth |
| 10 | *Weight filters in this basis won't yield much sparser models*: 47.6% of weights have positive helpfulness, so keeping every helpful weight is a floor of tens of percent | 49.0% positive (full family); 30.1% / 42.2% significantly helpful (tokens / features) | irrelevant as a floor: MAttr at density 0.15 discards **64%** (tokens) / **77%** (features) of significantly-helpful weights and the loss is −0.003 / −0.024 (zero) or +0.033 / −0.017 (mean) | **premise refuted as a floor** — counted helpful weights do not bound density; mass does |
| 11 | Helpfulness-mass guide: 90% of positive mass at 2.43% density, 99% at 13.6% | oracle: 0.44% / 4.2% (tokens), 3.4% / 18% (features); Fisher 0.54% / 13% and 7.1% / 42% | MAttr's break-even kept sets hold 95–99% of positive mass; sets holding 90% cost +0.03 to +0.09 nats | **qualified**: 99% mass is roughly the break-even guide, 90% is too aggressive; summing mass overstates the family's worth 78× on tokens (Result 3) |
| 12 | "We don't encourage readers to read much into our change from ERA/TWERA to Fisher effectiveness … these metrics are quite similar" | ρ(Fisher, ERA) = 0.76 over all weights, but ERA cannot re-rank within a source row on this family (it is \|W\| · P(source)); pruning: ERA +0.116 vs Fisher +0.009 at density 0.15 | — | **fails on Tokens→Logits** with the plain ERA definition; caveat: the note's appendix ERA uses gradient attributions for logit targets, which we did not implement |
| 13 | Discussion: "we suspect … that no saliency scheme will perform much better" than Fisher for filtering | — | on tokens, controlled: nothing beats Fisher above density 0.05 (MAttr within 1.1–3.7×, still improving with budget); on features: MAttr beats Fisher 1.6–4.6× and beats the exact oracle at every density | **upheld on the token family at the densities the note operates at; overturned on the feature family** |
| A1 | *ACETYLCHOLINE pairplot* (path effects against each other over the vocabulary) | qualitative only (Result 7 of `interference_vw.md`) | n/a | **not audited** (display of 0a) |
| A2 | *Single-weight helpfulness histograms*: per-batch ΔL is broad and two-sided even when the mean is confidently positive; "recovering the sign takes a lot of data" | replicated as a token budget: the median token-row weight needs 6.7×10⁵ tokens for its CI to exclude zero, the median feature weight 1.7×10⁷; with 10⁶ tokens only 57% / 29% of weights are sign-recoverable, with 10⁹ 97% / 88% | the learned ranking recovers keep/drop for 98% of well-determined weights (z ≥ 5) on tokens from 393M *sampled* tokens, IG 94–99%; Fisher, sign-blind, is at chance (51%) on this classification | **upheld** for per-weight ablation statistics; a signed score reads every weight from every token and does not pay that cost |
| A3 | *Mean helpfulness histograms*: "centered near zero, roughly symmetric, helpful side slightly heavier" | features: 58% / 42% by count, helpful mass 6.5× the harmful; token rows: **63% harmful by count**, helpful mass **52×** the harmful (median helpful weight 4× the median harmful one) | n/a | **upheld on features; on tokens the symmetry is in count only**, the mass is one-sided |
| A4 | *Sampled mean-helpfulness CDF* | same data as A3 | n/a | **upheld** |
| A5 | *Helpfulness mass distribution*: sample sums 1.44×10⁻⁴ positive vs 1.12×10⁻⁵ negative (13 : 1) | 52 : 1 (token rows), 2,610 : 1 (full family incl. positions), 6.5 : 1 (features) | the ratio predicts which family MAttr wins (Result 18) | **upheld in direction**; the ratio is family-specific by two orders of magnitude |
| A6 | *Fisher effectiveness CDF*: continuum, no bimodality | yes: 99.8% / 100% of weights have non-zero Fisher, spread over 24 orders | n/a | **upheld** |
| A7 | *Fisher mass distribution*: 315M of 331M non-zero; effectiveness mass 3.62 on positive weights vs 1.59 on negative (30% negative) | **does not replicate on tokens**: negative token-row weights carry 0.35% of Fisher mass (3.965 vs 0.014) while holding 98% of positive helpfulness mass; features 9% (0.523 vs 0.052) | this is the mechanism of Result 12: `p_j(1−p_j)` vanishes for a weight that fully suppresses its target, so Fisher cannot see inhibition on this family | **discrepancy**, explained; the note's own appendix flags the same blind spot and patches it with a counterfactual variant for feature-targeting weights |
| A8 | *Fisher vs expected attribution*: quantitatively similar | ρ = 0.76 over all weights; identical within a source row on this family by construction (see #12) | — | **similar in rank, not in pruning** (#12) |
| A9 | *Feature 1254 position input*: direct and OV position inputs cancel | not built | — | **not audited** |
| A10 | *Virtual weight magnitude vs helpfulness*: the helpful/harmful tail gap present for some paths, not consistently | ratio of the largest helpful to the largest harmful \|W\| is 1.3 (tokens) and 1.2 (features), against Fisher's 514 and 43 | — | **upheld** |
| A11 | *Threshold eval per family and sign*: Fisher beats \|W\| for every family and sign except negative Features→Logits | replicates, including the exception: Fisher loses to \|W\| on negative feature weights below density 0.02 | MAttr is best on every sign of both families at every density (negative to d=0.15 on positive weights of both, to 0.30 on negative feature weights); it does not share Fisher's negative-feature failure | **upheld, exception included; MAttr removes the exception**. Also: 90% of negative token weights are individually helpful and jointly worth 0.011 nats; 87% of positive ones are individually harmful and jointly worth 0.58 |
| A12 | *ROPE plots*: with a region of practical equivalence at ε = budget/N_family, most significant weights are practically zero | at ε = budget/N (2.9×10⁻⁷): 98.1% / 97.8% practically zero, **1.0% / 1.1% practically positive**, ≤0.1% practically negative; at ε = 10⁻⁶ × budget, 0.2% / 0.1% positive | at density 0.01 the kept sets are 11% (Fisher), 12% (IG), 17% (MAttr best), 95% (\|W\|) practically zero on tokens; 6% / 14% / 23% / 93% on features; at density 0.15 every ranking is ≥ 85% practically zero because only 1.9% / 2.2% of weights are not | **upheld**, and it sharpens #10: by the note's own yardstick ~1% of each family is practically non-zero, yet the loss-neutral density is ~15%, so the loss is carried jointly by weights that are individually practically zero |
| A13 | *Full helpfulness table*: proportion helpful / harmful by significance, per family | 30.1% / 60.0% (token rows), 43.9% / 48.0% (full family), 42.2% / 25.8% (features) | n/a | **upheld** (tens of percent each way) |

## Details by figure

### 1. The IN → logits example

Top-8 targets out of `IN` under each score, with the training-corpus count of how often the
target actually follows `IN` and the significance of its helpfulness (`he` helpful, `ha`
harmful, `ns` not significant):

| ranking | top-8 (target[count, sig]) |
|---|---|
| Virtual weight magnitude | TER[977,he] SA[3,ns] DE[299,he] **vir[0,ha] tig[0,ha]** ST[724,he] **ú[0,ha]** NA[7,he] |
| Fisher effectiveness | TER[977,he] T[3817,he] E[3395,he] ES[1522,he] D[6097,he] IT[652,he] ST[724,he] AL[583,he] |
| Expected Gradients | T[3817] E[3395] TER[977] S[1956] ST[724] D[6097] C[2342] UT[148], all helpful |
| MAttr (default) | T[3817] E[3395] TER[977] S[1956] D[6097] N[599] IT[652] P[402], all helpful |
| **MAttr (best)** | **T[3817] E[3395]** S[1956] D[6097] C[2342] ,[764] N[599] av[72], all helpful |
| Helpfulness (oracle) | T E TER ST S D ES IT, all helpful |

Three of the eight largest virtual weights out of `IN` point at tokens that never follow it;
every other ranking removes all three. The note's footnote (' T ' more likely than ' E ') holds
for every score that gets the order right: T (3,817) above E (3,395). MAttr puts ' E ' second,
exactly where the note's Fisher sort put it.

### 2, 4, 5. Per-node panels: does a score isolate helpful weights within a node?

For every live source node, the top-n weights by each score (`audit.json → per_node`):

| Tokens→Logits, top-10 per token | helpful | harmful | not sig. | nodes with ≥1 harmful |
|---|---|---|---|---|
| Virtual weight magnitude / ERA | 0.631 | 0.329 | 0.040 | 0.663 |
| Fisher effectiveness | 0.817 | 0.086 | 0.097 | 0.315 |
| MAttr (default) | 0.855 | 0.025 | 0.121 | 0.078 |
| **MAttr (best)** | 0.914 | **0.007** | 0.080 | **0.035** |
| Expected Gradients | 0.926 | 0.003 | 0.071 | 0.018 |
| Helpfulness (oracle) | 0.944 | 0.002 | 0.054 | 0.005 |

| Features→Logits, top-10 per feature | helpful | harmful | not sig. | nodes with ≥1 harmful |
|---|---|---|---|---|
| Virtual weight magnitude / ERA | 0.684 | 0.117 | 0.199 | 0.481 |
| Fisher effectiveness | 0.770 | 0.122 | 0.109 | 0.563 |
| MAttr (default) | 0.768 | 0.113 | 0.119 | 0.586 |
| **MAttr (best)** | 0.846 | **0.065** | 0.089 | **0.427** |
| Expected Gradients | 0.846 | 0.070 | 0.084 | 0.424 |
| Helpfulness (oracle) | 0.998 | 0.000 | 0.002 | 0.000 |

The note's Feature-3013 observation — harmful weights intermingled with helpful ones at high
effectiveness — is the typical case for Fisher, not an unlucky feature: **more than half of
all features have a significantly harmful weight among their ten most effective**. On the token
family a learned or gradient score removes that problem almost entirely (0.7% harmful in the
top-10, 3.5% of nodes affected). On the feature family it is halved, not removed: MAttr, IG and
Fisher all leave 6–12% harmful weights in a feature's top-10, and only the oracle is clean.
Note the untuned MAttr default is no better than Fisher on features; the whole gain is the
ε = 1e-8 / batch-32 arm.

Also visible in the tables: on features, `|W|` puts *fewer* harmful weights in a node's top-10
than Fisher (11.7% vs 12.2%) but more non-significant ones (19.9% vs 10.9%). That is the
note's "virtual weights are not inherently misleading" in numbers: on the feature basis raw
magnitude is a mediocre guide, not a random one; on the token basis it is worse than a coin
flip (33% harmful against a 30% base rate of helpful weights).

### 7. The three-regime figure

Along each ranking, over the whole family (`audit.json → along`):

| Tokens→Logits | first harmful at rank | harmful share, top 0.01% | top 0.1% | top 1% | top 10% |
|---|---|---|---|---|---|
| Virtual weight magnitude | 27 | 0.828 | 0.881 | 0.930 | 0.906 |
| Fisher effectiveness | 18,688 | 0.000 | 0.000 | 0.091 | **0.459** |
| MAttr (default) | 4,556 | 0.000 | 0.000 | 0.007 | 0.011 |
| **MAttr (best)** | 670 | 0.001 | 0.001 | 0.003 | **0.015** |
| Expected Gradients | 7,092 | 0.000 | 0.000 | 0.001 | 0.001 |
| Helpfulness (oracle) | 6,138,728 | 0 | 0 | 0 | 0 |

| Features→Logits | first harmful at rank | top 0.01% | top 0.1% | top 1% | top 10% |
|---|---|---|---|---|---|
| Virtual weight magnitude | 344 | 0.007 | 0.033 | 0.077 | 0.143 |
| Fisher effectiveness | 3,679 | 0.000 | 0.008 | 0.088 | **0.241** |
| MAttr (default) | 144 | 0.015 | 0.034 | 0.093 | 0.109 |
| **MAttr (best)** | 178 | 0.014 | 0.034 | 0.064 | **0.063** |
| Expected Gradients | 155 | 0.005 | 0.018 | 0.048 | 0.059 |
| Helpfulness (oracle) | 9,699,154 | 0 | 0 | 0 | 0 |

The note reads its figure as three regimes: ineffective weights near zero helpfulness, a
"middle regime of helpful and harmful weights with marginal effectiveness", and a helpful-only
tail at the top. Its two quantitative claims about the tail replicate: the top 0.01% by Fisher
contains no harmful weight on either family (the top 0.1%: none on tokens, 0.8% on features), and the most effective helpful weight out-measures
the most effective harmful one by 514× (tokens) and 43× (features), against "an order of
magnitude or more".

But the middle regime does not survive a change of ranking. Under Fisher the top 10% of the
token family is **46% harmful**; under MAttr it is **1.5%**, and under Expected Gradients 0.1%. The
regimes figure (`vw_audit_regimes.pdf`) shows why: Fisher's harmful arm starts at rank ~2×10⁴
and runs alongside the helpful arm to the end, because `w²` cannot see sign; MAttr's harmful
weights are almost all in the last decade of the ranking. The note's discussion draws a
conclusion about the model from this figure — "Harmful weights are a sign of the inherent
tradeoff between implementing more circuits and the interference caused between them … the
model contains a middle regime of helpful and harmful weights with marginal effectiveness." The
first half stands (60% of token-row weights are significantly harmful). The second half is a
statement about Fisher effectiveness: a signed score sorts the same weights into two regimes,
not three. Fisher's one advantage here is the very top: its first harmful weight comes at rank
18,688 against MAttr's 670, because a harmful weight with a large `w²` is rare while a harmful
weight MAttr scores highly is merely uncommon.

### 8, 9. The pruning figure

See `interference_vw.md` Results 6, 9, 11, 14, 15, 18 and `paper/tabs/vw_prune_*.tex`. The
summary, Δ held-out loss in nats with the family's total worth in brackets:

| setting | density 0.30 | 0.15 | 0.05 | 0.01 |
|---|---|---|---|---|
| tokens, full family, zero (the note's object) [1.176]: Fisher / MAttr best | +0.052 / **−0.002** | +1.027 / **+0.888** | +2.252 / **+0.982** | +1.893 / **+0.787** |
| tokens, token rows, mean (controlled) [0.569]: Fisher / MAttr best | **+0.002** / +0.034 | **+0.009** / +0.033 | **+0.036** / +0.041 | +0.101 / **+0.088** |
| features, mean [0.806]: Fisher / MAttr best | +0.018 / **−0.021** | +0.057 / **−0.017** | +0.125 / **+0.027** | +0.274 / **+0.140** |

The note's three published points (+0.00015 / +0.0107 / +0.0702 at density 0.55 / 0.30 / 0.15,
six families) sit within a factor of a few of our Fisher curve on one family, and Fisher beats
`|W|` at every density on both families, as the note says. "At 1% density the pruned model's
performance is significantly compromised" is true of Fisher (+0.10 of 0.57 on the controlled
token cut, +0.27 of 0.81 on features) and less true of MAttr (+0.09 and +0.14, i.e. 83–85% of
each family's value kept at 1% density).

### 10, 11. The density floor and the helpfulness-mass guide

The note argues from the fraction of helpful weights (47.6%, "tens of percent" even after a
significance test) to a floor on how sparse a filtered model can be, and then relaxes that to a
mass criterion (90% of positive mass at 2.43% density; 99% at 13.6%). What the best MAttr kept
set actually does at each density (`audit.json → along → at_density`):

| Tokens→Logits, MAttr best | positive mass kept | share of sig.-helpful weights kept | share of sig.-harmful kept | ΔL zero | ΔL mean |
|---|---|---|---|---|---|
| density 0.55 | 0.988 | 0.991 | 0.330 | −0.005 | +0.030 |
| 0.30 | 0.988 | 0.829 | 0.005 | −0.005 | +0.034 |
| **0.15** | 0.986 | **0.361** | 0.003 | **−0.003** | +0.033 |
| 0.05 | 0.975 | 0.117 | 0.001 | +0.008 | +0.041 |
| 0.02 | 0.957 | 0.058 | 0.000 | +0.026 | +0.059 |

| Features→Logits, MAttr best | positive mass kept | sig.-helpful kept | sig.-harmful kept | ΔL zero | ΔL mean |
|---|---|---|---|---|---|
| density 0.55 | 0.995 | 0.699 | 0.154 | −0.025 | −0.011 |
| 0.30 | 0.984 | 0.391 | 0.091 | −0.028 | −0.021 |
| **0.15** | 0.949 | **0.228** | 0.040 | **−0.024** | **−0.017** |
| 0.05 | 0.833 | 0.090 | 0.012 | +0.009 | +0.027 |
| 0.02 | 0.694 | 0.038 | 0.005 | +0.060 | +0.087 |

At density 0.15 the kept set discards **64% of the significantly-helpful token-row weights and
77% of the significantly-helpful feature weights**, and the pruned model is as good as or better
than the full one under both ablations on features and under zero ablation on tokens. The
count of individually-helpful weights is therefore not a floor on density for any joint
selection; Fisher's kept set at the same density keeps only 9.7% / 15.0% of the helpful
weights and costs +0.009 / +0.057, so it is not a floor for a saliency ranking either. What
tracks the loss is mass: every break-even kept set holds 95–99% of the family's positive
helpfulness mass, and the sets that hold 90% (MAttr at density ≈0.03–0.09) cost +0.03 to
+0.09 nats. So of the note's two mass guides, 99% is approximately right as a break-even
density and 90% is too aggressive — with the standing caveat from Result 3 that positive mass
summed over the token family overstates the family's actual worth by 78×, so the guide works as
a *relative* criterion and not as a nats budget.

The MAttr default arm illustrates a failure the note's mass analysis cannot see: it holds 84%
of positive mass from density 0.002 all the way to 0.55 (its curve is flat) because the 16% it
never captures are the large suppressive weights of Result 12, filed at the bottom of its
ranking. Its 99%-mass density is 1.0.

### 12. ERA versus Fisher

On this family `era = |W| · P(source active)` is constant in the target within a source row, so
ERA and `|W|` induce the same order inside every node (the per-node rows for the two are
identical above). Across the family ρ(Fisher, ERA) = 0.76 but the pruning curves are far apart
(ERA +0.116 vs Fisher +0.009 at density 0.15, controlled). The note's statement that the metrics
are "quite similar" is about its own ERA variant, which for logit-targeting weights uses linear
gradient attributions rather than the thresholded form; we did not implement that variant, so
this row is a caveat rather than a refutation.

### 0a. The motivating example as a membership question

The note's first figure decomposes the ACETYLCHOLINE completion into the direct, attention and
MLP paths. That is a statement about paths, not weights, so the MAttr version asks instead which
of the direct-path weights out of `IN` a learned mask keeps, and when (`audit.json →
in_targets`; "kept at" is the density at which the ranking first includes the weight):

| IN → target | W | followed | sig. | Fisher (row rank / kept at) | IG | MAttr default | **MAttr best** | oracle |
|---|---|---|---|---|---|---|---|---|
| E (the answer) | +2.95 | 3,395 | helpful | 3 / <0.001 | 2 / <0.001 | 2 / <0.001 | **2 / <0.001** | 2 |
| T | +2.83 | 3,817 | helpful | 2 / <0.001 | 1 | 1 | **1** | 1 |
| TER | +5.38 | 977 | helpful | 1 / <0.001 | 3 | 3 | 10 / 0.002 | 3 |
| ES | +3.37 | 1,522 | helpful | 4 / 0.001 | **4096 / 1.000** | **4095 / 1.000** | 32 / 0.006 | 7 |
| DE | +4.29 | 299 | helpful | 10 / 0.001 | **4092 / 1.000** | **4096 / 1.000** | 29 / 0.005 | 15 |
| vir | +4.25 | 0 | harmful | 155 / 0.037 | 3907 / 0.956 | 3822 / 0.928 | 3818 / 0.945 | 3951 |
| tig | +4.15 | 0 | harmful | 178 / 0.046 | 3810 / 0.930 | 3580 / 0.874 | 3724 / 0.929 | 3886 |
| ú | +4.11 | 0 | harmful | 582 / 0.140 | 3313 / 0.840 | 3362 / 0.837 | 2534 / 0.787 | 2968 |
| SA | +4.41 | 3 | n.s. | 39 / 0.005 | 4080 / 0.998 | 4066 / 0.998 | 3660 / 0.919 | 4081 |

Every non-magnitude ranking keeps the answer at the very top and drops the three zero-count
weights until the family is nearly complete. Fisher is the exception on the harmful side: it
includes ' vir ' and ' tig ' at density 0.04–0.05 (rank 155–178 within the row) where MAttr and
the oracle hold them out until density > 0.9. The path-linear scores show their blind spot in
the other direction: IG and the ε=1e-2 MAttr arm file ' ES ' and ' DE ', helpful weights with
1,522 and 299 co-occurrences, at the **very bottom** of the whole row; the ε=1e-8 arm does not.

### A2. How much data the sign of one weight needs, and how much a ranking needs

The note's single-weight histograms make the point that a weight's per-batch ΔL is two-sided,
so its mean needs a lot of tokens to be signed with confidence. With the exact per-weight
standard error at 97.7M positions, the token budget each weight would need for its 95% CI to
exclude zero is `n·(1.96·SE/|mean|)²`:

| family | median tokens needed | recoverable with 10⁶ | 10⁷ | 10⁸ | 10⁹ | 10¹⁰ |
|---|---|---|---|---|---|---|
| Tokens→Logits (token rows) | 6.7×10⁵ | 0.574 | 0.827 | 0.904 | 0.968 | 0.991 |
| Features→Logits | 1.7×10⁷ | 0.287 | 0.450 | 0.682 | 0.880 | 0.962 |

That is the ablation estimator's cost. A ranking pays it differently: it sees every weight at
every token. Keeping exactly as many weights as have positive helpfulness, and asking whether
the kept/dropped decision matches the sign of helpfulness, by how well-determined that sign is
(`audit.json → rank_agreement`):

| Tokens→Logits, agreement | all | z < 1 | 1–1.96 | 1.96–5 | 5–20 | z ≥ 20 |
|---|---|---|---|---|---|---|
| Fisher effectiveness | 0.508 | 0.625 | 0.712 | 0.762 | 0.593 | 0.416 |
| Virtual weight magnitude | 0.421 | 0.488 | 0.478 | 0.476 | 0.404 | 0.414 |
| Expected Gradients | 0.924 | 0.536 | 0.655 | 0.745 | 0.940 | 0.991 |
| MAttr (default) | 0.881 | 0.295 | 0.343 | 0.568 | 0.908 | 0.995 |
| **MAttr (best)** | **0.930** | 0.389 | 0.529 | 0.854 | **0.981** | **0.996** |

| Features→Logits, agreement | all | z < 1 | 1–1.96 | 1.96–5 | 5–20 | z ≥ 20 |
|---|---|---|---|---|---|---|
| Fisher effectiveness | 0.492 | 0.510 | 0.516 | 0.542 | 0.510 | 0.424 |
| Expected Gradients | 0.808 | 0.520 | 0.672 | 0.835 | 0.933 | 0.958 |
| MAttr (default) | 0.747 | 0.471 | 0.577 | 0.736 | 0.883 | 0.920 |
| **MAttr (best)** | 0.713 | 0.521 | 0.641 | 0.785 | 0.812 | 0.757 |

For weights whose sign is well determined (z ≥ 5) the learned ranking agrees with it 98–100% of
the time on tokens, from 393M sampled tokens, and IG 94–99%. Fisher is at chance on this
question at every z, which is the sign-blindness of `w²` stated as a classification. Two
honest notes: MAttr's agreement on the weakly-determined weights (z < 1.96) is *below* chance on
tokens (0.39–0.53), i.e. for weights whose helpfulness is essentially noise the mask's decision
is anti-correlated with the noisy sign, presumably because those decisions are driven by the
weight's set-level role rather than its marginal one; and on features MAttr's agreement for the
best-determined weights (0.76 at z ≥ 20) is below IG's (0.96), which is the same "keeps
individually non-significant or even harmful weights for the set" behaviour that the pruning
curve rewards on that family.

### A3, A5, A7. The distributions

| | Tokens→Logits (token rows) | Features→Logits | note (six families) |
|---|---|---|---|
| positive / negative by count | 36.3% / 63.4% | 57.8% / 42.2% | 47.6% positive |
| positive / negative helpfulness mass | 1.815 / 0.035 (52 : 1) | 0.616 / 0.094 (6.5 : 1) | 1.44e-4 / 1.12e-5 (13 : 1), sample |
| median \|h\| among positive / negative | 7.4e-10 / 1.8e-10 | 1.3e-9 / 1.8e-9 | — |
| weights with non-zero Fisher | 99.8% | 100% | 95.2% |
| Fisher mass on W > 0 / W < 0 | 3.965 / **0.014** | 0.523 / 0.052 | 3.62 / 1.59 |

The mean-helpfulness histogram (`vw_audit_hist.pdf`) matches the note's description on the
feature family. On the token family it does not: harmful weights are the majority by count and
helpful weights carry 52× the mass, with a tail that extends three decades further. The Fisher
mass split is the one number in the note's appendix that fails to replicate on the token family
in *direction*: the note finds 30% of effectiveness mass on negative weights, we find 0.35%.
Result 12 explains it: a weight that fully suppresses its target has `p_j ≈ 0`, so
`p_j(1 − p_j) ≈ 0` and its Fisher effectiveness vanishes however large `w` is, while its true
ablation effect is exponential in `w`. The note's Fisher appendix describes exactly this
failure for feature-targeting weights and patches it with a counterfactual variant; on the
Tokens→Logits family it does not, and on our model the negative token→logit weights are where
98% of the positive helpfulness mass lives.

### A12. The ROPE yardstick

Classifying every weight against a region of practical equivalence (−ε, ε), with the note's
per-family yardstick ε = (ln 4096 − 3.38) / N_family = 2.9×10⁻⁷ nats:

| ε | tokens: positive / negative / practically zero / uncertain | features |
|---|---|---|
| budget / N | 1.0% / 0.0% / **98.1%** / 1.0% | 1.1% / 0.1% / 97.8% / 1.0% |
| budget × 10⁻⁶ | 0.2% / 0.0% / 99.7% / 0.1% | 0.1% / 0.0% / 99.9% / 0.0% |
| budget × 10⁻⁵ and 10⁻⁴ | 0 / 0 / 100% / 0 | 0 / 0 / 100% / 0 |

By the note's own yardstick, about 1% of each family is practically non-zero, and the fraction
of the kept set that is practically zero is then a property of the density more than of the
ranking (at density 0.15 every ranking is ≥ 85% practically zero, because at most 1.9% of the
family is not). At density 0.01 it does discriminate:

| practically-zero share of the kept set at density 0.01 | Fisher | \|W\| | IG | MAttr default | MAttr best | oracle |
|---|---|---|---|---|---|---|
| Tokens→Logits | **0.108** | 0.947 | 0.116 | 0.305 | 0.171 | 0 |
| Features→Logits | **0.058** | 0.928 | 0.139 | 0.209 | 0.229 | 0 |

Fisher's 1% is the cleanest by this yardstick and MAttr's contains 2–4× as many practically-zero
weights, yet on features MAttr's 1% is the better model (+0.14 vs +0.27 nats, mean ablation).
That is the same fact as #10 from the other side: individually practically-zero weights carry
loss jointly, and a per-weight region of practical equivalence cannot see it.

### A11. The per-sign split (stage K, `scripts/vw/launch/vw_stageK_cpu.sh`)

The note's appendix repeats the pruning curve per family and per sign of the weight, and the
main text summarises it as "Fisher performs far better than raw virtual weight magnitude for
every density and individual weight family (except for negative Features→Logits weights)".
Zero ablation as in the note, 256 held-out sequences, weights of the other sign held on.

First the marginal picture, split by sign (all 97.7M positions):

| token rows | n | significantly helpful | significantly harmful | Σ positive helpfulness | joint removal costs |
|---|---|---|---|---|---|
| **positive** weights (W > 0) | 11,573,081 | 3.3% | **86.6%** | 1.786 | **+0.581** |
| **negative** weights (W < 0) | 5,204,135 | **89.7%** | 0.7% | 0.030 | **+0.011** |

| features | n | sig. helpful | sig. harmful | Σ positive helpfulness | joint removal costs |
|---|---|---|---|---|---|
| positive weights | 7,109,617 | 16.2% | 52.0% | 0.519 | **+0.447** |
| negative weights | 9,667,599 | 61.4% | 6.5% | 0.097 | **+0.120** |

On the token rows the sign of a weight's marginal helpfulness points the *wrong way* for its
joint value. Nine in ten negative (suppressive) token→logit weights are individually,
significantly helpful, and deleting all 5.2M of them at once costs 0.011 nats, 1.9% of what
the token rows are worth. Nearly nine in ten positive (excitatory) weights are individually
harmful, and deleting them costs 0.58 nats. The closed form says why: an excitatory weight to a
non-target lowers the loss when removed, so any weight that votes for a continuation is
"harmful" at every position where that continuation is wrong, which is most of them; jointly,
the excitatory weights are the model's votes. The suppressive weights are each helpful because
each one alone keeps a wrong token down, but removing all of them shifts every logit in a row
upward together and the softmax normalises most of that away. This is the note's footnote,
"up to nonlinear effects in removing multiple weights at once", at its largest: the count of
helpful weights (#10) not merely fails to bound the density, it identifies the wrong half of
the family.

The pruning curves per sign (`scripts/vw/vw_sign_table.py`; token tables rerun with the stage-J arm):

**Tokens→Logits (token rows), positive weights only** (11,573,081 prunable; all removed: +0.5809)

| ranking | d=0.55 | d=0.3 | d=0.15 | d=0.05 | d=0.02 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|---|
| Fisher | +0.0002 | +0.0014 | +0.0049 | +0.0179 | +0.0398 | +0.0733 | +0.2647 |
| |W| | +0.0564 | +0.1971 | +0.3101 | +0.4203 | +0.4808 | +0.5074 | +0.5495 |
| oracle | -0.0165 | -0.0163 | -0.0163 | -0.0120 | +0.0094 | +0.0471 | +0.2553 |
| MAttr b32 12k | -0.0031 | -0.0028 | -0.0027 | -0.0008 | +0.0176 | +0.0601 | +0.4704 |
| MAttr best (b64 24k) | -0.0065 | -0.0062 | -0.0060 | -0.0046 | +0.0118 | +0.0742 | +0.4815 |

**Tokens→Logits (token rows), negative weights only** (5,204,135 prunable; all removed: +0.0108)

| ranking | d=0.55 | d=0.3 | d=0.15 | d=0.05 | d=0.02 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|---|
| Fisher | +0.0001 | +0.0004 | +0.0013 | +0.0030 | +0.0044 | +0.0055 | +0.0077 |
| |W| | +0.0029 | +0.0066 | +0.0088 | +0.0105 | +0.0107 | +0.0108 | +0.0108 |
| oracle | -0.0030 | -0.0026 | -0.0019 | +0.0002 | +0.0023 | +0.0036 | +0.0068 |
| MAttr b32 12k | -0.0002 | +0.0002 | +0.0008 | +0.0020 | +0.0031 | +0.0045 | +0.0076 |
| MAttr best (b64 24k) | -0.0011 | -0.0009 | -0.0003 | +0.0009 | +0.0028 | +0.0046 | +0.0077 |

**Features→Logits, positive weights only** (7,109,617 prunable; all removed: +0.4467)

| ranking | d=0.55 | d=0.3 | d=0.15 | d=0.05 | d=0.02 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|---|
| Fisher | -0.0001 | +0.0028 | +0.0172 | +0.0792 | +0.1739 | +0.2504 | +0.3808 |
| |W| | +0.0150 | +0.0540 | +0.0968 | +0.1751 | +0.2503 | +0.2942 | +0.3718 |
| oracle | +0.0519 | +0.0522 | +0.0587 | +0.1094 | +0.1880 | +0.2558 | +0.3811 |
| MAttr best | -0.0187 | -0.0190 | -0.0170 | +0.0081 | +0.0480 | +0.0860 | +0.1926 |

**Features→Logits, negative weights only** (9,667,599 prunable; all removed: +0.1199)

| ranking | d=0.55 | d=0.3 | d=0.15 | d=0.05 | d=0.02 | d=0.01 | d=0.002 |
|---|---|---|---|---|---|---|---|
| Fisher | +0.0010 | +0.0080 | +0.0298 | +0.0855 | +0.1165 | +0.1293 | +0.1365 |
| |W| | +0.0211 | +0.0498 | +0.0721 | +0.0944 | +0.1076 | +0.1145 | +0.1191 |
| oracle | +0.0101 | +0.0138 | +0.0259 | +0.0606 | +0.0905 | +0.1060 | +0.1217 |
| MAttr best | -0.0066 | -0.0063 | +0.0008 | +0.0204 | +0.0388 | +0.0518 | +0.0774 |

Fisher does beat |W| at every density on both signs of the token family, as the note says. On
the positive token weights, which are all of the family's value, the best MAttr arm is
*negative* (better than the full model) from density 0.55 to 0.05, beats Fisher down to
density 0.02 (+0.012 vs +0.040), ties it at 0.01 and loses at 0.002; the oracle is negative to
0.05 as well. On the negative
token weights every curve is within 0.011 nats of the full model, so the "except negative
Features→Logits" exception has a token-family analogue in the trivial sense that there is
nothing to win. On the feature family the note's exception **replicates**: on negative Features→Logits weights Fisher loses to |W| from density 0.02 down and at density 0.002 is worse than deleting all of them (+0.137 against +0.120). MAttr is the best ranking on both signs of the feature family at every density, negative (better than the full model) to density 0.15 on positive weights and to 0.30 on negative ones, and it beats the exact oracle throughout; the oracle is the *worst* non-magnitude ranking on positive feature weights above density 0.05 (+0.052 at 0.55 against Fisher's −0.0001). Where Fisher has its one documented failure, a learned ranking does not share it.

### Not audited

The newline-feature vignette (Tokens→Features and Tokens→OV→Features position inputs cancelling
to a constant), the QK examples, the per-family split of the pruning curve into positive and
negative weights, and the "Fisher effectiveness for all paths" appendix all need weight families
this replication did not build. The per-sign pruning split is the one of these that would be
cheap to add on our two families.

## What changes in the appendix

`paper/sections/interference-lm.tex` gains a subsection with `tabs/vw_audit.tex`, the two audit
figures, and the one-line verdicts above. The summary sentence of that appendix — that the
note's conjecture is half right — is unchanged by the audit; what the audit adds is that two of
the note's *qualitative* claims about the model (effectiveness "does not isolate" helpful
weights; the model "contains a middle regime" of mixed weights) are properties of the
sign-blind Fisher score and largely disappear under a signed learned or gradient score, on the
token family entirely and on the feature family by half.
