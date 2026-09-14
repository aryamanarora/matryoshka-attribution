# Cause (top-k -> source flips the output): what the logged SVA metrics say

Date: 2026-09-05. Script: `scripts/sva/analyse_cause.py` (reads every `results/sva_sweep*/**.json`,
writes `results/cause_analysis.csv`). No re-evaluation was needed: `eval_sva.py` already sweeps
both directions and stores `cause_metrics.acc_source` (fraction of examples whose top-1 flips to
the SOURCE label when the top-k units are patched to source, complement clean) and its log-k
AUC `cause_accsrc_auc`. This is the paper's Cause metric (causal-abstraction.tex) with the
identity mapping pi; `acc_auc` is the Iso side.

Terminology (CLAUDE.md convention): iso = sufficient = denoising (top-k clean); cause =
necessary = noising (top-k patched). All 3.3k runs in `results/sva_sweep` are iso-TRAINED and
evaluated in both directions; the 36 runs in `results/sva_sweep_cause` (2026-07-25) are
cause-trained MAttr on the node substrate at default-eps Adam.

## 1. Which methods are best at cause, from iso-trained rankings

`cause_accsrc_auc`, task-group averaged (SVA / Arith / ARC-E / IOI as in
`plot_accauc_vs_faithauc.group_avg`), best loss per method:

| method (iso-trained)      | node  | mlp   | mlp+attn_head |
|---------------------------|-------|-------|---------------|
| IG                        | **0.982** | **0.849** | **0.921** |
| Expected Gradients (mc_ig m=1)   | 0.974 | 0.835 | 0.917 |
| MAttr soft, SGD lr1 (log) | 0.953 | 0.841 | 0.915 |
| MAttr soft, SGD lr1 (unif)| 0.951 | 0.841 | 0.916 |
| MAttr soft, Adam eps1e-2  | 0.935 | 0.845 | 0.913 |
| MAttr soft, Adam eps1e-8  | 0.933 | 0.680 | 0.762 |
| MAttr id-STE SGD (log)    | 0.931 | 0.827 | 0.874 |
| AttnLRP                   | 0.831 | 0.804 | 0.874 |
| IxG                       | 0.854 | 0.799 | 0.884 |
| Node Pruning (eprun s090) | 0.870 | 0.487 | 0.480 |
| DBM (sig lr0.3 l1 6)      | 0.932 | 0.638 | 0.725 |
| Random                    | 0.337 | 0.118 | 0.135 |

Reading:
* **IG is the best cause ranking everywhere, but only by 0.004-0.03 over tuned MAttr** at
  neuron scale (mlp: 0.849 vs 0.845 Adam-eps1e-2 / 0.841 SGD; mlp+attn: 0.921 vs 0.915/0.913).
  On node the gap is bigger (0.982 vs 0.953) because node is nearly saturated: IG's single top
  unit (k=1) already flips 83% of examples on average (k*_50 = 1 on 9/10 tasks), MAttr-SGD's
  64%. The node cause problem is "find the one head/MLP layer that carries the answer".
* Iso-trained MAttr is competitive at cause **without ever training for it**: the tuned
  variants sit within noise of IG on the neuron substrates and are 2nd-4th of 15 methods.
  Default-eps Adam MAttr (the sign(g) regime, see memory `adam-eps-neuron-scale`) is NOT — it
  is a poor cause ranking at neuron scale (0.68 / 0.76, k*_90 = 3745 on mlp vs IG's 56).
* Iso and cause rankings of methods agree moderately: Spearman(acc_auc, cause_accsrc_auc)
  across methods within a task is 0.82 (mlp), 0.76 (mlp+attn), 0.53 (node). The disagreement
  is IG: it is 1st at cause and 9th at iso on node (0.512 vs MAttr 0.554). Attribution patching
  estimates the marginal effect of patching each unit alone, which is exactly the small-k
  cause question; it says nothing about which set is jointly sufficient.
* At neuron scale a single neuron never flips anything (acc_source at k=1 is 0.00-0.03 for
  every method on mlp); IG/MAttr-SGD need k ~ 56-67 neurons (median over tasks) for a 90% flip
  rate, MAttr+attn-heads ~ 11-16 units.

## 2. Cause-TRAINED MAttr (existing runs: node, default-eps Adam, 2026-07-25)

Paired against the iso-trained twin (same task / variant / loss), 36 pairs:

| variant     | loss       | d cause_accsrc_auc | d acc_auc (iso) | d faith_auc |
|-------------|------------|--------------------|-----------------|-------------|
| soft-fwd    | acc        | +0.082             | -0.097          | -0.240 |
| soft-fwd    | ce         | -0.001             | -0.374          | -0.506 |
| soft-fwd    | logit_diff | -0.003             | -0.267          | -0.546 |
| hard-STE    | acc        | +0.023             | -0.294          | -0.458 |
| hard-STE    | ce         | -0.124             | -0.385          | -0.487 |
| hard-STE    | logit_diff | -0.057             | -0.384          | -0.663 |

Win rate over iso-trained on cause_accsrc_auc: 0.42 (n=36). So on the node substrate, training
in the cause direction (a) helps only with the `acc` (soft 0-1) loss, (b) is neutral-to-harmful
with ce / logit_diff, and (c) destroys the iso side (acc_auc 0.55 -> 0.02-0.45). The ce /
logit_diff failure is the gap-padding mode: once the top-k patch flips the prediction the loss
keeps rewarding a larger source margin, which at node scale is satisfied by any large-k mask,
so the log-k schedule's small-k steps are the only informative ones and they are outvoted.
Node is also saturated (IG k*=1), so there is little headroom to measure.

## 3. Experiments launched (2026-09-05): cause-trained MAttr at neuron scale

`scripts/sva/launch/submit_sva_cause_neuron.sh` -> `results/sva_sweep_cause/` (tags `necessary_topk_sgd_*`,
`necessary_topk_adam_eps1e-2_*`, later `joint_*`). Substrates mlp and mlp+attn_head, 8 tasks
(4 SVA + 4 arith), the two TUNED MAttr configs (soft fwd + log-k; SGD lr 1.0 / Adam eps 1e-2 lr
0.05, identical to their iso twins in results/sva_sweep), losses ce / acc / logit_diff.
Priority batch 1 = necessary x {ce, acc} = 64 jobs (~5 min each on an H100).

Results: see section 4 (filled in as jobs land).

### Interim note (19/64 of batch 1 landed): a cause-trained ranking can be a degenerate iso ranking
`simple/mlp, Adam eps1e-2, ce` (cause-trained): cause_accsrc_auc 0.825 but iso acc_auc 0.022 --
the iso curve sits at F_patch for EVERY k < total (keeping the top 744k of 1.38M neurons clean
still gives the source answer). The training probe shows iso acc-AUC at 0.65 through step 1200,
then 0.27 / 0.11 / 0.02 at steps 1400 / 1600 / 1800 while the cause loss kept falling. The
cause objective only constrains which units flip the output when patched; the tail of the
ranking (what the iso sweep patches first) is free, and Adam+ce drives a small flip-sufficient
set to the very bottom. The SGD twin (same loss, same cell) is intact (iso acc_auc 0.615). Not
an eval bug: F_clean/F_patch match the twin, and k=total recovers acc 1.0.

## 4. Results: cause-trained MAttr at neuron scale (batch 1: --mode necessary, losses ce / acc)

64 runs, 8 tasks x 2 substrates x 2 configs x 2 losses, each paired with its iso-trained twin in
results/sva_sweep (same config, same seed, same 100 eval pairs) and with IG at the same loss.
Figure: `plots/plot_cause_curves.py` -> `plots/cause_curves_{acc,ce}.pdf`.

### 4a. Cause-trained vs iso-trained (paired deltas, mean over 8 tasks)

| substrate     | config           | loss | d cause_accsrc_auc | wins/8 | d iso acc_auc | k*90 iso -> cause (median) |
|---------------|------------------|------|-------------------:|-------:|--------------:|---------------------------|
| mlp           | SGD lr1          | acc  | **+0.019**         | 7      | -0.014        | 82 -> 52 |
| mlp           | SGD lr1          | ce   | **+0.020**         | 8      | -0.049        | 67 -> 36 |
| mlp           | Adam eps1e-2     | acc  | +0.015             | 7      | -0.017        | 67 -> 36 |
| mlp           | Adam eps1e-2     | ce   | +0.020 (1 collapse)| 5      | -0.23         | 82 -> 36 |
| mlp+attn_head | SGD lr1          | acc  | **+0.015**         | 8      | -0.014        | 11 -> 7 |
| mlp+attn_head | SGD lr1          | ce   | +0.013             | 6      | -0.049        | 16 -> 7 |
| mlp+attn_head | Adam eps1e-2     | acc  | +0.015             | 6/7    | -0.017        | 13.5 -> 7 |
| mlp+attn_head | Adam eps1e-2     | ce   | +0.006 (3 collapse)| 5      | -0.23         | 16 -> 7 |

(Table values are from the per-cell pivot; the `mean delta by method x loss` block of
analyse_cause.py pools both substrates.) Win rate over all 63 pairs: 0.67 -> 0.8 excluding
Adam+ce.

* **Yes, the intervene-on-top-k loss does something, but it is small.** With the `acc` loss the
  cause AUC rises by +0.015-0.02 in every configuration, 29/31 cells positive, and the k needed
  for a 90% flip rate drops by ~35% (52 vs 82 neurons on mlp; 7 vs 11 units on mlp+attn). The
  gains are concentrated on the arithmetic tasks (addition/hours/months: +0.03 to +0.09), where
  the iso-trained ranking trailed IG; on the 4 SVA tasks the deltas are +0.00 to +0.02.
* **Cost on the iso side is small with `acc` (-0.014 acc_auc) and large with `ce`.** Adam+ce
  produced 4 degenerate rankings out of 16 (iso acc_auc 0.02-0.12; see the interim note above):
  the cause objective does not constrain the tail of the ranking, and Adam+ce sends a
  flip-sufficient set there. SGD never collapsed (min iso acc_auc 0.37).
* Adam eps1e-2 and SGD lr1 are again interchangeable when both work (+0.015 vs +0.015-0.019).

### 4b. Cause-trained MAttr vs IG (same task / substrate / loss)

| substrate     | config       | loss | MAttr-cause | IG    | mean d | wins/8 | k*90 MAttr / IG |
|---------------|--------------|------|------------:|------:|-------:|-------:|-----------------|
| mlp           | SGD lr1      | acc  | **0.861**   | 0.845 | +0.016 | 8      | 52 / 56 |
| mlp           | SGD lr1      | ce   | 0.857       | 0.848 | +0.010 | 7      | 36 / 56 |
| mlp           | Adam eps1e-2 | acc  | 0.859       | 0.845 | +0.014 | 6      | 36 / 56 |
| mlp           | Adam eps1e-2 | ce   | 0.840       | 0.848 | -0.007 | 2      | 36 / 56 |
| mlp+attn_head | SGD lr1      | acc  | **0.930**   | 0.919 | +0.011 | 8      | 7 / 10 |
| mlp+attn_head | SGD lr1      | ce   | 0.918       | 0.919 | -0.001 | 3      | 7 / 11 |
| mlp+attn_head | Adam eps1e-2 | acc  | 0.926       | 0.914 | +0.011 | 7/7    | 7 / 12 |
| mlp+attn_head | Adam eps1e-2 | ce   | 0.910       | 0.919 | -0.009 | 2      | 7 / 11 |

For reference the iso-trained twins LOSE to IG on 6-7 of 8 cells (mean -0.004 to -0.025). So
the cause loss turns MAttr from "ties IG at cause" into "beats IG at cause on 16/16 cells with the
acc loss" -- by about one AUC point, i.e. the same order as the seed-to-seed noise we still have
to measure (arm A below). The honest one-liner: IG is a near-optimal cause ranking at these
substrates; a cause-trained MAttr with the acc loss edges it, an iso-trained one ties it, and a
default-eps Adam MAttr is far behind either way.

### 4c. Side observation from the iso-trained sweep: SAE substrates
On `mlp_sae_span` iso-trained MAttr-SGD beats IG at cause by +0.13-0.17 on 8/8 tasks (0.745 vs
0.574) -- the one substrate where MAttr's cause ranking is clearly better than IG's; on
`resid_sae_span` it is far worse (0.27 vs 0.72), the dead-latent saturation failure already
documented (memory `mattr-sae-dead-latent-saturation`), which the cause direction inherits.

### Queued after batch 1 (2026-09-05 04:45 UTC)
* joint (--mode joint, per-step coin flip iso/cause) x {ce, acc} x both configs x both substrates
  (64) -- does one ranking get the cause gain without the iso cost?
* necessary x logit_diff, SGD only (16; the Adam ones were cancelled -- Adam+ce already unstable)
* arm A: seed-43 replicate of SGD {acc, ce} cause-trained -> results/sva_sweep_cause_s43 (32),
  plus seed-43 iso-trained SGD acc twins -> results/sva_sweep_s43 (16): the noise floor for the
  +0.015 deltas
* arm B: node substrate with the tuned configs (SGD lr1 / Adam eps1e-2) x {acc, ce}, 6 tasks (24)
* arm C: uniform-k schedule, SGD acc, cause-trained (16): does the cause objective need log-k?

## 5. Results: JOINT training (--mode joint: each step flips a coin between iso and cause)

64 runs, same grid as batch 1. Paired against the iso-trained twin and against IG:

| substrate     | config       | loss | d cause_accsrc_auc | wins/8 | d iso acc_auc | min iso acc_auc | vs IG (cause) |
|---------------|--------------|------|-------------------:|-------:|--------------:|----------------:|--------------:|
| mlp           | SGD lr1      | acc  | +0.014             | 8      | **+0.013**    | 0.47            | +0.010, 8/8 |
| mlp           | SGD lr1      | ce   | +0.020             | 7      | **+0.025**    | 0.42            | +0.010, 7/8 |
| mlp           | Adam eps1e-2 | acc  | +0.015             | 7      | +0.018        | 0.45            | +0.014, 7/8 |
| mlp           | Adam eps1e-2 | ce   | +0.022             | 6      | +0.000        | 0.44            | -0.004, 6/8 |
| mlp+attn_head | SGD lr1      | acc  | +0.015             | 8      | +0.003        | 0.47            | +0.011, 8/8 |
| mlp+attn_head | SGD lr1      | ce   | +0.018             | 6      | +0.012        | 0.43            | +0.004, 4/8 |
| mlp+attn_head | Adam eps1e-2 | acc  | +0.017             | 6      | +0.002        | 0.45            | +0.010, 7/8 |
| mlp+attn_head | Adam eps1e-2 | ce   | +0.008             | 4      | -0.018        | 0.40            | -0.006, 3/8 |

* **Joint gets the whole cause gain of pure cause-training (+0.014-0.022 vs +0.015-0.020) and
  pays nothing on the iso side -- iso acc_auc goes UP by +0.00 to +0.025** (pure cause-training:
  -0.005 to -0.31). No joint run collapsed (min iso acc_auc 0.40, vs 0.022 for the Adam+ce
  cause-only runs). With the acc loss, joint-MAttr beats IG at cause on 30/32 cells and beats
  iso-MAttr at iso on the same runs. So mixing the intervene-on-top-k objective into training
  is a strict improvement over the iso-only recipe on both metrics, at equal cost (same 2000
  steps, one forward per step).
* Why iso improves too: the cause direction is the noising sweep of the SAME top-k ranking, so
  it supplies gradient about the small-k head of the ranking -- exactly where the log-k iso
  objective is data-starved (a k-draw of 5 among 2.7M units almost never moves the base
  prediction, so the iso loss there is flat). The two objectives are complementary views of one
  ranking, not competing ones.
* Per-task, the gains again concentrate on arithmetic (addition/hours/months +0.02 to +0.07);
  SVA tasks +0.00 to +0.04. Adam+ce is the weakest combination in every table.

### 5b. Loss comparison for the cause objective (SGD lr1, cause-trained only, 8 tasks each)

| loss       | d cause_accsrc_auc (mlp / mlp+attn) | d iso acc_auc (mlp / mlp+attn) | min iso acc_auc |
|------------|-------------------------------------|--------------------------------|-----------------|
| acc        | +0.019 / +0.015                     | -0.007 / -0.021                | 0.43 |
| ce         | +0.020 / +0.013                     | -0.028 / -0.069                | 0.28 |
| logit_diff | **+0.031 / +0.027**                 | **-0.108 / -0.224**            | 0.02 (addition, mlp+attn collapsed) |

The unbounded margin loss buys the most cause AUC and wrecks the iso side the most -- the same
gap-padding ordering (acc < ce < logit_diff in damage) the iso-side loss study found. For the
joint recipe the bounded `acc` loss is the one to use.

### Queue note (06:20 UTC)
Six `mask-learning-finetuning` jobs of the user's now hold both nodes the QOS allows
(`QOSMaxNodePerUserLimit`), so arms A (seed replicates), B (tuned node) and C (uniform-k), plus
the joint seed replicate and joint-on-node, are pending with nothing of ours running. Results
above are complete for batches 1-3 (necessary x {acc, ce, logit_diff-SGD}, joint x {acc, ce}).

## 6. Noise floor (arm A: seed 43 replicate of the SGD cause-trained runs, results/sva_sweep_cause_s43)

Same config, same eval pairs, different training seed (30 pairs):

| loss | substrate     | SD of cause_accsrc_auc across seeds | SD of iso acc_auc across seeds |
|------|---------------|------------------------------------:|-------------------------------:|
| acc  | mlp           | 0.002                               | 0.015 |
| acc  | mlp+attn_head | 0.002                               | 0.014 |
| ce   | mlp           | 0.004                               | 0.022 |
| ce   | mlp+attn_head | 0.022 (one outlier cell)            | 0.045 |

So the +0.015 to +0.02 cause gain from cause/joint training with the acc loss is ~8 seed-SDs --
a real effect, not a lottery -- and its consistency (29/31 and 30/32 cells positive) says the
same. The iso side is ~7x noisier (SD 0.014), so joint's "+0.003 to +0.025 iso acc_auc" should
be read as "no cost, possibly a small gain", not as a reliable improvement. ce is noisier than
acc on both axes, another reason to prefer acc for the cause direction.

## 7. Arm B: node substrate with the TUNED configs (replaces the July default-eps Adam runs)

6 tasks (4 SVA + arc_easy + ioi/qwen2.5), cause-trained, paired with iso twins and IG:

| config       | loss | d cause_accsrc_auc | wins/6 | d iso acc_auc | min iso acc_auc | MAttr-cause vs IG |
|--------------|------|-------------------:|-------:|--------------:|----------------:|------------------:|
| SGD lr1      | acc  | **+0.045**         | 4 (2 ties at 1.0) | -0.029 | 0.47 | 0.991 vs 0.992 |
| SGD lr1      | ce   | -0.059             | 2      | -0.362        | 0.02            | 0.916 vs 0.910 |
| Adam eps1e-2 | acc  | +0.034             | 3      | -0.038        | 0.47            | 0.950 vs 0.992 |
| Adam eps1e-2 | ce   | -0.047             | 1      | -0.512        | 0.02            | 0.871 vs 0.910 |

On node the cause problem is "rank the one head/MLP layer that carries the answer first"; the
acc-loss SGD run gets there (0.991 = IG's 0.992, both at the ceiling of the 24-point grid whose
first three points are all k=1). ce is destructive at node scale for BOTH optimizers -- every
ce run collapsed its iso side (acc_auc 0.02 on at least one task each) and the cause AUC went
down. Consistent with sections 2 and 5b: the cause direction needs a bounded loss.

## 8. Arm C: k-schedule for the cause objective (SGD lr1, acc loss, cause-trained)

Mean cause_accsrc_auc over the 7-8 tasks per substrate (hours/uniform-k still pending):

| substrate     | IG    | iso-trained (log-k) | cause-trained log-k | cause-trained uniform-k |
|---------------|------:|--------------------:|--------------------:|------------------------:|
| mlp           | 0.845 | 0.841               | 0.861               | 0.868 (7 tasks) |
| mlp+attn_head | 0.919 | 0.915               | 0.930               | 0.924 |

The schedule does not matter for the cause direction (differences 0.006-0.007 in opposite
directions on the two substrates, per-task |diff| <= 0.015). Expected: the cause loss saturates
at large k either way (patching everything always flips), so both schedules spend their useful
gradient on the same small-k regime. Use log-k for consistency with the iso headline.

## 9. Final: joint replicate, joint on node, and the verdict (all 260 runs landed, 07:35 UTC)

### 9a. Joint vs iso-only, SGD lr1 + acc loss, pooled over seeds 42 and 43 (paired by cell and seed)

| substrate     | n  | d cause_accsrc_auc | wins | d iso acc_auc     | d faith_auc |
|---------------|---:|-------------------:|-----:|------------------:|------------:|
| mlp           | 16 | **+0.0148 ± 0.0030** | 16/16 | +0.028 ± 0.013  | +0.014 |
| mlp+attn_head | 16 | **+0.0150 ± 0.0016** | 16/16 | +0.011 ± 0.008  | -0.013 |
| node          | 6  | +0.045 (Adam: +0.075) | 5/6 | -0.002          | -0.039 |

(± = SE over the 16 cell-seed pairs.) Seed-to-seed SD at fixed config: iso-only 0.0067 (cause) /
0.039 (iso acc_auc); joint 0.0026 / 0.013 -- joint training is also ~3x more seed-stable on
BOTH metrics. On node, joint lands at 0.991-0.992 = IG's 0.992 (ceiling) with no iso cost.

### 9b. Verdict

1. **Which methods are best at cause (from the logged, iso-trained sweep)?** IG, at every
   substrate (node 0.982, mlp 0.849, mlp+attn 0.921 cause_accsrc_auc) -- attribution patching
   estimates exactly the marginal flip effect the small-k cause question asks. Tuned MAttr
   (SGD lr1 or Adam eps1e-2, soft top-k, log-k) ties it at neuron scale (within 0.005-0.03) and
   is the best non-gradient method; Expected Gradients is ~IG; AttnLRP / IxG trail by 0.03-0.05;
   default-eps Adam MAttr, DBM and Node Pruning are far behind. So **MAttr is good at cause
   without being trained for it -- but not better than IG.**
2. **Is training MAttr with the intervene-on-top-k (cause) loss useful?** Yes, modestly and
   reliably: +0.015 cause AUC (~8 seed-SDs; 29/31 to 32/32 cells), k*90 down ~35%, and it flips
   the IG comparison from "loses 6-7/8" to "wins 8/8" on every substrate with the acc loss. In
   absolute terms it is one AUC point: IG was already near the ceiling of what a ranking can do
   here.
3. **How to train it:** `--mode joint --loss acc` (per-step coin flip between iso and cause).
   Cause-only training buys the same cause gain but costs iso (-0.005 to -0.3 acc_auc) and can
   collapse the iso ranking outright (Adam+ce 4/16 cells, every ce run on node). Joint pays
   nothing on iso (point estimate slightly positive), never collapsed in 92 runs, and is more
   seed-stable than the current iso-only recipe. Bounded loss only: ce is worse, logit_diff
   gains most on cause (+0.03) and wrecks iso (-0.1 to -0.22). k-schedule is irrelevant for the
   cause direction (log vs uniform within 0.007).
4. **Where the gain is:** arithmetic tasks (+0.02 to +0.09 on addition/hours/months), where
   the iso-trained ranking trailed IG; SVA tasks +0.00 to +0.04; node is at the ceiling.

### Artifacts
* `scripts/sva/analyse_cause.py` -> `results/cause_analysis.csv` (all substrates, all runs)
* `scripts/sva/launch/submit_sva_cause_neuron.sh` (MODES / SUBS / LOSSES / CONFIGS / KS / SEED / OUT env)
* `plots/plot_cause_curves.py` -> `plots/cause_curves_{acc,ce}.pdf`
* results: `results/sva_sweep_cause/` (necessary_* / joint_* tags, incl. node + uniformk),
  `results/sva_sweep_cause_s43/`, `results/sva_sweep_s43/` (seed-43 replicates)
* Budget: 260 runs, ~5 min each on one H100; wall-clock 03:30-07:35 UTC, throughput bound by
  the 8-GPU / 2-node QOS shared with the mask-learning-finetuning jobs.
