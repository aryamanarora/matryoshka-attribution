# Project notes for Claude

## CRITICAL: `sufficient` vs `necessary` convention (don't re-flip it)

Unified rule across the whole repo: **`sufficient` = top-k stays CLEAN, complement
corrupted (denoising)**; **`necessary` = top-k corrupted, complement clean (noising)**.
This matches standard interp terminology (denoising tests sufficiency, noising tests
necessity) and is what MIB's CPR measures — so all our MIB runs are the *sufficient*
(denoising) intervention.

- MIB scripts (`eval_mib.py`, `eval_mib_edge.py`): `--mode sufficient` (now the default)
  = denoising = our runs. `--mode necessary` = noising.
- DAS / CausalGym (`scripts/attribute.py`): config key / flag `sufficient: true` = denoising
  = top-k clean, matching MIB. **Gotcha:** the *internal* legacy flag (and
  `sigmoid_das.intervene`'s `sufficient=` param) use the OPPOSITE sense (`True` = top-k get
  CF / noising); `attribute.py` inverts once right after `parse_args` (`args.sufficient =
  not args.sufficient`). Don't "fix" that inversion — it's load-bearing.

Both `--mode`/`sufficient:` labels were originally flipped and were corrected (commits on
2026-06-15). Behavior of all existing runs was preserved (config values flipped to match).
If you add a new config/script, follow the unified rule above; if a number looks like the
wrong intervention, check this first.

## CRITICAL: which results dir is the "MAttr" / "Ours" headline

**As of 2026-07-21 the headline MAttr is the SOFT top-k forward, log-k schedule, lr=0.05
variant** (best test CPR avg, best acc-AUC, no IOI/Qwen 0.25-floor collapse). The
hard sigmoid-STE forward is now the "$+$ hard" ablation; uniform-k rows are "+ unif k".

| Results dir (`results/...`)        | Variant                     | Paper role            |
|------------------------------------|-----------------------------|-----------------------|
| `topklog_lr_0.05`                  | soft fwd, log k (node, val) | **MAttr headline**    |
| `test_node_topk_log_lr05`          | soft fwd, log k (node, test)| **MAttr headline**    |
| `mib_edge_topk_log_lr05`           | soft fwd, log k (edge, val) | **MAttr headline**    |
| `test_edge_topk_log_lr05`          | soft fwd, log k (edge, test)| **MAttr headline**    |
| `htklog_lr_0.05` (+test/edge twins)| hard STE fwd, log k         | "$+$ hard" ablation   |
| `htk_lr_0.05`                      | hard STE fwd, uniform k     | "+ unif k, + hard"    |
| `final_node`                       | soft fwd, uniform k         | "+ unif k"            |

Do NOT use uniform-k dirs as the headline — their CPR averages look strong (esp. edge)
but acc-AUC is the worst of the three and they collapse on IOI/Qwen test; using them
as "MAttr" makes ablations look deceptively good. (Pre-2026-07-21 history/artifacts
used the hard log-k `htklog`/`mib_node_hard_topk_log` as headline — beware stale labels.)

### Verification anchor
`topklog_lr_0.05` `area_under` matches the `\ourmethod{}` row of `paper/tabs/mib_results.tex`
cell-for-cell. If your "headline" numbers don't match that row, you're reading the wrong dir.
Compare against the table as it is on disk — do NOT hardcode expected values here or in a
script. Re-evaluations overwrite pkls in place (e.g. the 2026-07-24 Gemma TL 2.15.4 pass moved
every gemma cell), so any number copied out of the table goes stale silently and then reads as
"you're in the wrong dir" when the dir is fine.

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
`525673a`; patching itself is faithful, the forward is not. Use
`MIB-circuit-track/.venv` (TL 2.15.4) for any gemma2 evaluation, training or scoring.
The L2A venv is fine for gpt2/qwen2.5/llama3 (the bug is Gemma-2-specific), though that
scoping rests on `525673a`'s diagnosis rather than a per-model cross-check.

All gemma2 cells of the LR sweep + MAttr node dirs were re-evaluated under TL 2.15.4 on
2026-07-24 by `scripts/reeval_gemma_mib.py` (which asserts TL 2.x and overwrites the pkls
in place), so `paper/tabs/lr_sweep.tex` and the MAttr rows of `mib_results.tex` are clean.
`submit_lr_sweep_*.sh` still points at `$ABS/.venv/bin/python` — re-running one of those
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
rows by the granularity actually pruned, mapped in `scripts/make_mib_table.py:EPRUN_NAME`
(`node` → "Node Pruning", `edge` → "Edge Pruning"). Everything on disk keeps the original
name — `results/eprun_*` and the `EdgePruning_patching_<level>` subfolder MIB's
`run_evaluation.py --method EdgePruning` writes. Don't rename those; it would orphan the pkls.

### Known-still-wrong artifacts (as of 2026-06-09)
These compare NAP-IG against the **uniform-k** `mib_node_hard_topk` instead of the
log-k `mib_node_hard_topk_log`, so they're inconsistent with the paper's L2A:
- `plots/plot_rank_scatter_all.py` (`LOCAL_OURS`/`CLUSTER_OURS`)
- `plots/plot_score_scatter_all.py` (`LOCAL_OURS`/`CLUSTER_OURS`)
- `scripts/compare_ranks.py` → `paper/tabs/rank_correlations.tex` (`compare_node_methods("mib_node_hard_topk", ...)`)

The natural-k comparison scripts (`plot_cpr_curves_naturalk.py`,
`plot_rank_scatter_naturalk.py`, `plot_score_scatter_naturalk.py`) were fixed to use
`mib_node_hard_topk_log`.

## Cluster (Stanford NLP `sc`) submission gotchas

MIB eval reloads the model for the eval phase (peak host RAM ≈ 2× model), and the
averaged-sparsity eval is GPU-heavy. Per model size, submit via `nlprun` (in tmux):
- Small (gpt2, qwen2.5-0.5B): defaults are fine.
- Gemma2 (2B): `-d a6000 -c 3 -r 64G --batch-size 4`
- Llama3 (8B): `-d a6000 -c 4 -r 96G --batch-size 2` (ARC tasks need batch 2; others batch 4 ok)
- Always prepend `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the big models.
- `-r` must keep mem/cpu ≤ MaxMemPerCPU (~30G on jag): e.g. 64G needs `-c 3`, 96G needs
  `-c 4`. Mismatch triggers `srun: fatal: cpus-per-task set by two different env vars`.
- `eval_mib.py` reads `--config <path>` relative to CWD first, then `scripts/`; task
  names in configs must use underscores (`arc_easy`, not `arc-easy`).
- `mib-path` defaults to `./MIB-circuit-track` (gitignored symlink to the cloned repo).
