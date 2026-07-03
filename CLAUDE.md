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

## CRITICAL: which results dir is the "L2A" / "Ours" baseline

The paper's headline method (**L2A**, the bold "Ours" row in `paper/tabs/mib_results.tex`)
is **log-uniform `k`**, NOT uniform `k`. The results dirs are named misleadingly:

| Results dir (`results/...`)      | `k-schedule` | Paper role                              |
|----------------------------------|--------------|-----------------------------------------|
| `mib_node_hard_topk_log`         | **log**      | **L2A / "Ours" (main method, line 11)** |
| `mib_node_hard_topk`             | uniform      | "Ours (uniform `k`)" ablation row       |
| `mib_node_natural_k`             | log + natural-k 0.2 | natural-k ablation (worse than L2A) |
| `mib_node_natural_k10`           | log + natural-k 0.1 | natural-k ablation (worse than L2A) |

**When comparing anything to "L2A", use `mib_node_hard_topk_log`.** The bare
`mib_node_hard_topk` dir is the *uniform-k* ablation and is weaker on several tasks
(esp. MCQA/Gemma 1.53, MCQA/Llama 1.56, IOI/Gemma 1.26 vs the log-k baseline's
2.17 / 2.34 / 1.70). Using it as "L2A" makes ablations look deceptively good.

### Verification anchor
`mib_node_hard_topk_log` `area_under` values match `mib_results.tex` line 11 exactly
(ioi/gpt2 1.84, ioi/gemma 1.70, mcqa/gemma 2.17, mcqa/llama 2.34, arc-c/llama 1.85, ...).
If your "baseline" numbers don't match that table, you're reading the wrong dir.

## Reading CPR AUC apples-to-apples

The official "CPR AUC" is `area_under` inside each `results/<dir>/<task>_<model>_validation.pkl`
(also `faithfulnesses` = the 10-point CPR-vs-sparsity curve at
0.1/0.2/0.5/1/2/5/10/20/50/100%). Read AUC from the pkl for all methods so the
comparison is self-consistent — do NOT mix pkl numbers with stale `mib_results.tex`
values, and do NOT compare a fresh run's log "CPR AUC=..." against the table unless
both used the same eval. Eval was changed to average over `n_eval` examples (commit
45db054), so older pkls/table values may differ from a fresh run.

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
