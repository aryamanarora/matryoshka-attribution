# Edge Pruning (Bhaskar et al., 2024) as a MIB mask-learning baseline

MIB ships exactly one mask-*learning* baseline, UGS, and it covers at most 3 of the 11
table cells (edge level, gpt2/qwen only — see [ugs_baseline.md](ugs_baseline.md)). Edge
Pruning is the other obvious mask-learning comparison, and unlike UGS there is nothing
model- or level-specific about it: the gates are ours, so it runs on all four MIB models
and at both edge and node level.

## What it is

Learn one hard-concrete gate per unit (edge or node), trained with KL to the unmasked
model plus a Lagrangian L0 penalty whose sparsity target is annealed. Implemented in
`src/learning_to_attribute/edge_pruning.py` (a port of the recipe from
`princeton-nlp/Edge-Pruning`: latents init N(10, 0.01) so the circuit starts dense,
stretched hard concrete with T=2/3 on [-0.1, 1.1], multipliers trained by gradient
*ascent*, AdamW lr 0.8 with linear warmup/decay, 3000 steps). The final log-alphas are
the attribution scores; MIB ranks units by them.

`scripts/eval_mib_edge_pruning.py` supplies the patching environment:

- `--level edge`: gates every real edge; a gated-off edge adds
  `(corrupted - clean)` of its source into its destination's input (same
  differentiable-adjacency trick as `eval_mib_edge.py`).
- `--level node`: one gate per attention head / MLP, interpolating that node's output
  between clean and corrupted (`z*clean + (1-z)*corrupted`). The input embedding is not
  gated by default (`--include-input` to gate it too); an ungated node keeps a NaN score,
  which is how MIB marks "always keep".

Denoising only (`mask=1` keeps clean), i.e. the *sufficient* intervention MIB CPR
measures — Edge Pruning is inherently this intervention. `--loss logit_diff` swaps the
task loss for MAttr's objective if you want to isolate objective from parameterization.

## Running it

```bash
# <model> <task> [level=node] [steps=3000] [split=validation]
sbatch scripts/run_edge_pruning.sbatch gpt2   ioi   node
sbatch scripts/run_edge_pruning.sbatch llama3 arc_challenge node
```

The runner trains, dumps the scored circuit as a MIB `graph.json`, then scores it with
MIB's `run_evaluation.py`:

```
results/eprun_node/graph_<task>_<model>.json          # circuit + scores
results/eprun_node/<task>_<model>_scores.pt           # log-alphas, loss/k logs, timing
results/eprun_eval/EdgePruning_patching_node/<task-dash>_<model>_validation_abs-False.pkl
```

The last path is the layout `make_mib_table.py` / `make_mib_accauc_table.py` read for
baselines, so the numbers land in the same harness as every other row. **Do not** use the
script's own in-process `--split`-eval numbers in the tables: `eval_mib.py`-style
in-process evaluation and `run_evaluation.py` disagree on the same circuit (notably on
Gemma), so mixing them is not apples-to-apples. `--skip-eval` (what the runner passes)
trains and dumps only.

Both training and evaluation run in the **MIB venv** (`MIB-circuit-track/.venv`,
TL 2.15.4): the L2A venv's TL 3.x has a Gemma-2 forward bug.

## Cost

Node level is cheap on every model (one multiply per node per forward; three forwards per
step — corrupted cache, clean reference for the KL, patched). Edge level is the expensive
one: each destination hook materializes a `[batch, pos, n_prev_sources, d_model]` diff
stack that is kept for backward, which is ~0.5 GB per hook at Llama-3.1-8B's last layers
(1057 sources, d_model 4096) and loops over sources in Python. Expect edge level to be
practical on gpt2/qwen, tight on gemma2, and slow on llama3.

## Caveat, same as UGS

Mask learning optimizes a *single* operating point (one sparsity target), while MIB sweeps
a circuit over 10 sparsity budgets. Away from the trained budget the log-alpha ranking is
much less informative, which is why a converged mask can score *worse* on CPR-AUC than a
half-trained one (we saw exactly that with UGS on gpt2/ioi). This is a property of the
baseline, not a bug — but it is worth saying out loud in the paper rather than presenting
the AUC gap as pure method advantage.
