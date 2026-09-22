# Project notes for Claude

## CRITICAL: `sufficient` vs `necessary` convention (don't re-flip it)

Unified rule across the whole repo: **`sufficient` = top-k stays CLEAN, complement
corrupted (denoising)**; **`necessary` = top-k corrupted, complement clean (noising)**.
This matches standard interp terminology (denoising tests sufficiency, noising tests
necessity) and is what MIB's CPR measures — so all our MIB runs are the *sufficient*
(denoising) intervention.

- MIB scripts (`eval_mib.py`, `eval_mib_edge.py`): `--mode sufficient` (now the default)
  = denoising = our runs. `--mode necessary` = noising.
- DAS: `sigmoid_das.RotateLayer.intervene`'s `sufficient=` param uses the OPPOSITE (legacy)
  sense (`True` = top-k get CF / noising); callers invert once at the boundary. Don't "fix" it.
  (The CausalGym `attribute.py` scripts that documented this were removed 2026-09-22; they are
  in git history.)

Both `--mode`/`sufficient:` labels were originally flipped and were corrected (commits on
2026-06-15). Behavior of all existing runs was preserved (config values flipped to match).
If you add a new config/script, follow the unified rule above; if a number looks like the
wrong intervention, check this first.

## CRITICAL: which results dir is the "MAttr" / "Ours" headline

**Defined in ONE place: `scripts/mib/mattr_variants.py`** (`HEADLINE`, `label()`, `mark_k()`,
`OPT_ORDER`). Every MIB table (`make_mib_table.py`, `make_mib_accauc_table.py`,
`make_mib_test_table.py`), the SVA+ tables (`scripts/sva/make_sva_table.py`), the bar chart
(`plots/plot_mib_test_avg.py`) and the hparams tables import it. Change the headline there, not
in a generator. The `OUR_METHODS` names are RELATIVE labels ("$+$ hard", "$-$ $c_k$"); the
k-schedule mark and the optimizer header are added at emission.

**As of 2026-09-15 (user decision) the headline is SOFT top-k forward, UNIFORM k, ADAM.** On the
SVA+ substrates, where Adam's epsilon is swept, the headline is eps = 1e-2 and the default-eps
(1e-8) run is a marked variant. The bar chart had drawn this run as the plain method since
2026-09-11; the tables now agree with it. Label grammar: `$+$ log $k$`, `$+$ SGD`, `$+$ hard`,
`$\epsilon{=}10^{-8}$`, comma-joined in that order; bare `\ourmethod{}` = headline.

| Results dir (`results/...`)              | Variant                            | Paper role (2026-09-15) |
|-------------------------------------------|------------------------------------|---------------------|
| `mib_node_topk_uniform_lr05` / `test_node_topk_uniform_lr05` | soft fwd, uniform k, Adam (node) | **MAttr headline** |
| `mib_edge_topk_uniform_lr05` / `test_edge_topk_uniform_lr05` | soft fwd, uniform k, Adam (edge) | **MAttr headline** |
| `topklog_lr_0.05` / `test_node_topk_log_lr05`; `mib_edge_topk_log_lr05` / `test_edge_topk_log_lr05` | soft fwd, log k, Adam | "$+$ log $k$" |
| `softuni_sgd_lr_3.0` / `test_node_softuni_sgd_lr_3.0`; `mib_edge_softuni_sgd_lr_3.0` / `test_edge_softuni_sgd_lr_3.0` | soft fwd, uniform k, SGD | "$+$ SGD" |
| `softlog_sgd_lr_1.0` / `test_node_softlog_sgd_lr_1.0`; `mib_edge_softlog_sgd_lr_3.0` / `test_edge_softlog_sgd_lr_3.0` | soft fwd, log k, SGD | "$+$ log $k$, $+$ SGD" (headline 2026-08-24 to 2026-09-15) |
| `mib_edge_softlog_sgd_lr_1.0` / `test_edge_softlog_sgd_lr_1.0` | edge SGD at the imported NODE lr | superseded, still on disk — do not quote |
| `htk_lr_0.05` (+ edge twins)               | hard STE fwd, uniform k, Adam      | "$+$ hard" |
| `htklog_lr_0.05` (+test/edge twins)        | hard STE fwd, log k, Adam          | "$+$ log $k$, $+$ hard" |
| `final_node`                               | soft fwd, uniform k, Adam, lr 0.01 | superseded by the lr05 dir |

History of the headline, for reading old tables: 2026-06-17 uniform k Adam (hard fwd era);
2026-07-21 soft fwd log k Adam; 2026-08-24 soft fwd log k SGD (edge lr 3.0 from 2026-09-04);
2026-09-15 soft fwd uniform k Adam. The SGD-vs-Adam reasoning below is kept as the record of why
each dir sits at the LR it does; it no longer describes the layout.

**Not flipped (2026-09-15): `plots/plot_mib_accauc_cpr_scatter.py`.** Its main-text panel is a
hand-curated set keyed on the SGD rows' exact labels (`(G_MSGD, "MAttr")`, LR_ANCHOR paths) and
its guards raise on any relabel, so it still draws the SGD log-k point as "MAttr". It needs its
own curation pass before it goes next to the flipped tables; `plot_method_corr_heatmap.py` uses
descriptive names ("MAttr SGD (log)") and is unaffected.

**Know what this costs before you quote it.** `submit_mib_edge_lr_sweep.sh` brackets BOTH
optimizers at edge scale; paired over its 4 cells (ioi/gpt2, ioi/llama3, mcqa/llama3,
arithmetic_subtraction/llama3), CPR AUC:

| Adam lr=0.1 | Adam lr=0.05 | SGD lr=3.0 (SGD's own edge optimum; the dir we ship) | SGD lr=1.0 (the old dir) |
|---|---|---|---|
| **7.65** | 7.58 | **6.89** | 4.72 |

Adam wins TUNED-vs-TUNED at edge scale — the node-level tie does not transfer. **Never write
prose saying SGD is the better edge optimizer** — the sweep says the opposite. Until 2026-09-04
the shipped SGD edge dir was `*_lr_1.0`, the imported NODE optimum, which made the bare edge
row read **4.96** against the Adam row's **6.23** — a wrong-LR artefact, not a finding.

**Fixed 2026-09-04 by repointing, not by flipping back**: `make_mib_test_table.OUR_EDGE_METHODS`
and `make_mib_table.OUR_METHODS` now name `mib_edge_softlog_sgd_lr_3.0` /
`test_edge_softlog_sgd_lr_3.0` (all 22 cells landed 2026-09-04 via
`scripts/mib/launch/submit_edge_sgd_lr3.sh`; gemma2 trained in the MIB venv, so no reeval stamp). Full-row
effect, 11 cells: validation avg 4.99 -> 6.04 (Adam 6.37), test avg 4.96 -> 5.93 (Adam 6.23).
10/11 cells improve (ioi/gemma2 is saturated for every method); SGD now beats Adam on the three
gemma2 cells and arith_sub/llama3, Adam wins every other llama3 cell by 1-2. So the bare edge row
trails the `$+$ Adam` row by ~0.3, not ~1.3, and that residual IS an optimizer result. The
4-cell `mib_edge_lrsweep_sgd_log_lr_3.0` is NOT a substitute (gpt2 there is scored on 200
examples, not the full split).

Do NOT use uniform-k dirs as the headline — their CPR averages look strong (esp. edge)
but acc-AUC is the worst of the three and they collapse on IOI/Qwen test; using them
as "MAttr" makes ablations look deceptively good. (Pre-2026-07-21 history/artifacts
used the hard log-k `htklog`/`mib_node_hard_topk_log` as headline — beware stale labels.)

### Verification anchor
`softlog_sgd_lr_1.0` `area_under` matches the bare `\ourmethod{}` row of
`paper/tabs/mib_results.tex` cell-for-cell (node section); `topklog_lr_0.05` matches the
`\ourmethod{}$+$Adam` row instead. If your "headline" numbers don't match the row you expect,
check you're not reading the pre-2026-08-24 (Adam-default) convention out of memory or an old
note. Compare against the table as it is on disk — do NOT hardcode expected values here or in
a script. Re-evaluations overwrite pkls in place (e.g. the 2026-07-24 Gemma TL 2.15.4 pass
moved every gemma cell), so any number copied out of the table goes stale silently and then
reads as "you're in the wrong dir" when the dir is fine.

## Reading CPR AUC apples-to-apples

The official "CPR AUC" is `area_under` inside each `results/<dir>/<task>_<model>_validation.pkl`
(also `faithfulnesses` = the 10-point CPR-vs-sparsity curve at
0.1/0.2/0.5/1/2/5/10/20/50/100%). Read AUC from the pkl for all methods so the
comparison is self-consistent — do NOT mix pkl numbers with stale `mib_results.tex`
values, and do NOT compare a fresh run's log "CPR AUC=..." against the table unless
both used the same eval. Eval was changed to average over `n_eval` examples (commit
45db054), so older pkls/table values may differ from a fresh run.

### Never evaluate a gemma2 cell in the L2A venv (TL 3.2.1 Gemma-2 forward bug)
`.venv` (TL 3.2.1) computes a **wrong Gemma-2 forward** — proved against an HF reference in
`525673a`; patching itself is faithful, the forward is not. Any gemma2 evaluation, training or
scoring runs in the TL 2.15.4 stack, which since 2026-09-16 is the **`tl2` dependency group**
of `pyproject.toml` (torch 2.5.1, transformers 4.46.3, transformer-lens 2.15.4 — the exact pins
of the Tilde venv every gemma2 number was produced with), locked in `uv.lock` and synced by the
job into its own env, like the ViT group:

    UV_PROJECT_ENVIRONMENT=.venv-tl2 uv run --no-default-groups --group tl2 python scripts/mib/eval_mib.py --model gemma2 ...

No venv inside the fork checkout and no `PYTHONPATH` (the fork's modules come in via
`deps.find_mib_path`); `submit_mib_node_mode_ablation_sc.sh` is the reference launcher. The
older `MIB-circuit-track/.venv` on Tilde is the same stack, hand-built. The L2A venv is fine for
gpt2/qwen2.5/llama3 (the bug is Gemma-2-specific), though that scoping rests on `525673a`'s
diagnosis rather than a per-model cross-check.

All gemma2 cells of the LR sweep + MAttr node dirs were re-evaluated under TL 2.15.4 on
2026-07-24 by `scripts/mib/reeval_gemma_mib.py` (which asserts TL 2.x and overwrites the pkls
in place), so `paper/tabs/lr_sweep.tex` and the MAttr rows of `mib_results.tex` are clean.
`submit_lr_sweep_*.sh` still runs every model in the default env — re-running one of those
scripts would silently reintroduce the bad Gemma numbers.

### llama3 cells are evaluated on 200 examples — match it
`MIB-circuit-track/run_variants.sh` scores every **llama3** cell with `--head 200`
(full validation crawls/OOMs on 8B); gpt2/qwen2.5/gemma2 cells run full validation.
That cap is what the `$^{\dagger}$` daggers in the appendix tables mean. Any NEW method or
baseline must pass `--head 200` for llama3 in `run_evaluation.py`, or its llama3 numbers
sit in a column next to numbers computed on a 200-example subset — not apples-to-apples.
(The Edge Pruning runner shipped without it and had to be fixed in `1a0216f`; uncapped
llama3 eval is also ~50× slower, ~9 h/job vs ~15 min.)
The cap is **validation-only**. Test splits are ≤1188 examples (ioi/arith 1000, arc-e 1188,
arc-c 586, mcqa 50) and both the MIB paper's test numbers and `submit_test_lr05.sh` are
full-split, so capping a test cell would make it the only subset-scored row in that table.

### "Node Pruning" vs "Edge Pruning" is a display name, not a different method
Same recipe and same code (`src/learning_to_attribute/edge_pruning.py`); the paper labels the
rows by the granularity actually pruned, mapped in `scripts/mib/make_mib_table.py:EPRUN_NAME`
(`node` → "Node Pruning", `edge` → "Edge Pruning"). Everything on disk keeps the original
name — `results/eprun_*` and the `EdgePruning_patching_<level>` subfolder MIB's
`run_evaluation.py --method EdgePruning` writes. Don't rename those; it would orphan the pkls.

### Removed on 2026-09-22 (pre-release cleanup)
The hard-topk-era rank/score scatter and CPR-curve plots, `compare_ranks.py` and the other
`rank_correlations` / `cpr_summary` / `hybrid` / DCM generators, the `scripts/mib/configs/` tree
(nothing passed `--config`), and the causalgym / global_kl / toy / impossibility substrates were
deleted; all are in git history before that date. Still on the pre-2026-09-15 headline: the
transfer chain (`plot_task_corr_heatmap.METHOD_DIR` -> `prep_transfer_sources.py` ->
`results/transfer_src/mattr`) and `plot_mib_curves.py`'s MAttr arms.

## Cluster (Stanford NLP `sc`) submission gotchas

**Canonical checkout since 2026-09-14: `/juice3/scr3/nlp/interp/learning-to-attribute`** (1 TB
volume). `results/` (all 435 dirs, 145 GB), `logs/`, `wandb/` and the MIB fork's own
`results/` (the `/home/guests/aryaman/MIB-circuit-track/results` the table scripts hardcode;
now `deps/MIB-circuit-track/results`) were rsync'd there from Tilde that day, plus the
arithmetic_addition cells that only ever ran on sc. The older sc clone at
`/nlp/scr/aryaman/learning-to-attribute` (juice2, 200 GB quota, chronically near full) and the
Tilde checkout at `/home/guests/aryaman/learning-to-attribute` are superseded — do not write new
results into either. Launchers with a hardcoded `ABS=` should point at the juice3 path
(`submit_arith_add_headline_sc.sh` does; the `/home/guests/...` sbatch scripts are Tilde-era).
On sc, submit with `nlprun` inside tmux and use `uv run python` in the job command: the login
node's NFS makes even a torch import crawl (nfs_wait_bit_killable for 20+ min), while the
compute nodes sync the env in seconds.

MIB eval reloads the model for the eval phase (peak host RAM ≈ 2× model), and the
averaged-sparsity eval is GPU-heavy. Per model size, submit via `nlprun` (in tmux):
- Small (gpt2, qwen2.5-0.5B): defaults are fine.
- Gemma2 (2B): `-d a6000 -c 3 -r 64G --batch-size 4`
- Llama3 (8B): `-d a6000 -c 4 -r 96G --batch-size 2` (ARC tasks need batch 2; others batch 4 ok)
- Always prepend `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the big models.
- `-r` must keep mem/cpu ≤ MaxMemPerCPU (~30G on jag): e.g. 64G needs `-c 3`, 96G needs
  `-c 4`. Mismatch triggers `srun: fatal: cpus-per-task set by two different env vars`.
- `eval_mib.py` still accepts `--config <path>` (relative to CWD first, then `scripts/mib/`),
  but every launcher passes CLI flags; task names must use underscores (`arc_easy`).
- `--mib-path` is resolved by `src/learning_to_attribute/deps.py:find_mib_path()`: explicit flag,
  `$L2A_MIB_PATH`, then `deps/MIB-circuit-track` (what `bash scripts/setup.sh` clones). **It must be OUR FORK with submodules**
  (`aryamanarora/MIB-circuit-track`, pinned in `scripts/setup.sh`; EAP-IG submodule `41e9b9c`).
  Upstream's `evaluate_area_under_curve` returns 5 values, ours 7 (`accuracies`, `acc_auc`), and
  `eval_mib.py` / `eval_mib_edge.py` unpack 7 -- an upstream or stale clone trains for hours and
  dies with `ValueError: not enough values to unpack (expected 7, got 5)` AFTER eval, before
  `scores.pt` is written (2026-09-14, sc: lost a 1.5 h llama3 node eval + a 25 min edge run). Both
  scripts import that function at startup, so fixing the clone does not rescue a running job --
  cancel and resubmit. `setup.sh`'s smoke check asserts the 7-value return without loading a model.

## ViT teaser: optimizer / LR / Adam-eps grid (2026-09-02)

`scripts/vit/vit_teaser_optim.py` + `scripts/vit/launch/vit_optim.sbatch` -> `results/vit_optim/seed{42,43,44}`,
figure `plots/plot_vit_optim.py` -> `paper/figs/vit_optim.pdf`. Same substrate as the teaser
(ViT-B/16, 196 patch tokens, resampled pixelate corruption, basenji-vs-Siamese logit diff, 2000
steps); metric is the held-out sufficiency AUC (`vit_teaser_faith.py` protocol), mean of 3 seeds.
Adam 7 LRs x 6 eps, SGD 9 LRs, Expected Gradients (patch-embedding path, alpha ~ U(0,1), 2000 draws).

- **Tuned Adam and tuned SGD tie within seed spread**: Adam lr 0.3 / eps 1e-2 = 5.62 [5.34, 5.84],
  SGD lr 1 = 5.42 [5.27, 5.64]. The teaser's shipped cell (Adam lr 0.05, eps 1e-8) re-measures
  at 4.95-5.05 -- not the optimum, but the same ranking beats every baseline either way.
- **eps does NOT decide the ranking at 196 units, unlike at 2.3M neurons.** At every LR the six eps
  values sit within ~0.5 AUC (default 1e-8 vs best 1e-2 at lr 0.3: 5.10 vs 5.62, against a seed
  range of ~0.5). The only structure is the expected diagonal: at eps >= 1e-1 the optimum shifts
  to larger LRs (Adam ~ SGD at lr/eps). The |s| p99/p50 ratio is 3-7 at every eps, i.e. no
  sign(g) collapse -- with 196 logits the per-step gradients are well above 1e-8.
- **Expected Gradients is far behind (3.55 [3.45, 3.70]), below AttnLRP (3.98) and KernelSHAP (4.51)**,
  and its probe trace plateaus by ~300 draws, so it is not under-sampled -- the gradient-path
  ranking is simply a worse sufficiency ranking on this image. Top-20 overlap of Adam's ranking
  with IG's is 0.27-0.37 at every eps, i.e. raising eps does not pull MAttr toward IG here.
- SGD collapses above lr 10 (100: 3.66); Adam above lr 1 at small eps. Both optimizers' curves
  are unimodal in LR with the peak one decade apart (0.3 vs 1), so "the optimizer doesn't
  matter once tuned" holds on this substrate too.
