# UGS: the mask-learning baseline on MIB

**UGS (Uniform Gradient Sampling)** is the only *mask-learning* baseline in the MIB circuit track —
every other MIB baseline (NAP-IG, Conductance, I×G, RelP, RelP+QK, AttnRLP, GIM, exact patching,
IFR) is a gradient/patching attribution method. It is therefore the closest published comparison
to MAttr: it also learns a continuous per-edge mask by gradient descent, but with an L0-style
sparsity penalty instead of a top-$k$ constraint, and it trains one mask per sparsity level
(sweeping `reg_lamb`) instead of one score vector for all sparsities.

MIB's README (`MIB-circuit-track/README.md`, "Uniform Gradient Sampling") points at a script in a
*different* repo (Stolfo et al.'s optimal-ablation code), then at `convert_mask_to_graph.py` in MIB
to turn the learned mask into a graph whose edge score is the mask value $\alpha$.

## Coverage: only 3 of our 11 MIB cells

`convert_mask_to_graph.py` hard-restricts `--model` to `gpt2-small` / `qwen`, and UGS is
**edge-level only**. Intersecting with our columns, UGS can cover:

| task | model | trainable? |
|------|-------|-----------|
| ioi  | gpt2  | yes (`-d ioi`, 9500 train examples, bs 5 → 1900 steps) |
| ioi  | qwen2.5 | yes |
| mcqa | qwen2.5 | yes (only 100 examples, upstream duplicates the set 6× → 300 steps at bs 2) |

Gemma-2 and Llama-3 cells are out of reach without extending the converter (the theta → edge-name
mapping assumes non-parallel attn/MLP and would need GQA handling), so UGS can only ever be a
partial-coverage row.

## Setup

The UGS code is `alestolfo/optimalablation`, cloned at `~/optimalablation` (not a submodule; it is
a baseline checkout, so the local patches below are captured in `baselines/optimalablation.patch`).
It is written against transformer_lens 2.x, so it runs in the **MIB venv**
(`MIB-circuit-track/.venv`, TL 2.15.4 + transformers 4.46.3) — the same venv we use for all MIB
eval. Its only missing dependency there was `seaborn` (installed; nothing else changed).

Four patches were needed to make the published script run at all:

1. `from utils.training_utils import ... load_data ...` — **`load_data` does not exist** anywhere in
   the repo, so `edge_pruning_unif_mib.py` failed at import. Dropped from the import; the one call
   site was in the InterpBench branch we don't use.
2. **No `--model` flag.** `load_args` has no model argument, and an unknown flag makes it swallow
   the `SystemExit` and silently reset *every* argument to its default (dataset `ioi`, model
   `interp-bench`). The model is now read from the `UGS_MODEL` env var.
3. `Qwen/Qwen2.5-0.5B-Instruct` → `Qwen/Qwen2.5-0.5B` in both the training script and
   `convert_mask_to_graph.py`, because MIB evaluates the **base** model
   (`MODEL_NAME_TO_FULLNAME` in `MIB_circuit_track/utils.py`). Training the mask on -Instruct and
   evaluating it on the base model would have been an unfair, silently-wrong baseline.
4. Skipped `load_model_data`'s 100k-document OpenWebText download — `owt_iter` is unused by the
   counterfactual (`cf`) objective that the MIB runs use.

Plus, in our MIB fork, `convert_mask_to_graph.py` gained `--lambdas` (default = upstream's six) and
now skips a missing snapshot instead of crashing, so we can convert just the `reg_lamb=0.001` run
that MIB's README specifies.

## Pipeline

```bash
# 1. train the mask (one job per cell; ~1 s/it, 1900 steps for ioi)
sbatch scripts/run_ugs.sbatch gpt2-small ioi 0.001
sbatch scripts/run_ugs.sbatch qwen ioi 0.001
sbatch scripts/run_ugs.sbatch qwen mcqa 0.001
# -> ~/optimalablation/results/pruning/<task>/cf/ugs_mib_<model>/0.001/snapshot.pth

# 2. convert the mask to a MIB graph (writes graph.json next to the snapshot)
cd ~/MIB-circuit-track && PYTHONPATH=.:EAP-IG/src .venv/bin/python convert_mask_to_graph.py \
    --path ~/optimalablation/results/pruning --task ioi --ablation cf \
    --model gpt2-small --lambdas 0.001

# 3. evaluate with MIB's own harness, into the layout make_mib_table.py reads for baselines
cd ~/MIB-circuit-track && PYTHONPATH=.:EAP-IG/src .venv/bin/python run_evaluation.py \
    --models gpt2 --tasks ioi --level edge --ablation patching --split validation \
    --method UGS --circuit-files <.../graph.json> \
    --output-dir ~/learning-to-attribute/results/ugs_eval
# -> results/ugs_eval/UGS_patching_edge/<task-dash>_<model>_validation_abs-False.pkl
```

Because the mask is trained per sparsity level, a single `reg_lamb` gives one mask that is then
swept over all 10 sparsity points by MIB's evaluation — the same curve treatment every other
method gets, but the mask was only optimised for one point on it. Training the full `reg_lamb`
sweep (six values) and stitching the best mask per sparsity would be the generous reading of UGS;
MIB's README specifies the single `0.001` run, so that is what we reproduce.
