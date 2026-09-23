<div align="center">
    <!-- Put the logo files in assets/ in the repo, then paste this at the top of README.md.
         GitHub swaps in the dark version automatically for dark-mode readers. -->
    <p align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/mattr-logo-dark.svg">
        <img alt="Matryoshka Attribution" src="assets/mattr-logo-light.svg" width="440">
      </picture>
    </p>
    
    <!-- Social preview: Settings → General → Social preview → upload assets/mattr-social-preview.png (1280×640). -->
  <a href="https://arxiv.org/abs/2609.25518"><strong>Read our paper »</strong></a>
</div>
<br>

**MAttr** (Matryoshka Attribution) learns an attribution ordering over a model's internal variables by optimising a mask with the *sigmoid top-k* operator, sampling a random sparsity level each step.

- **Parameter attribution**: [`aryamanarora/matryoshka-attribution-parameters`](https://github.com/aryamanarora/matryoshka-attribution-parameters) (we release this as a sibling repo to simplify the codebase; it includes RL and SFT training code and configs, along with attribution code)
- 🤗 **HuggingFace**: [**MIB circuit-track submission**](https://huggingface.co/aryaman/mattr-mib-circuits) (node and edge circuits for every MIB subtask)


## Highlights

1. **One training run, every sparsity**: the learned scores are a ranking, not a circuit at a fixed budget (unlike other mask learning methods); the whole CPR-vs-sparsity curve comes from one training run.
2. **Universal API for representations, weights, etc.**: `learn_scores` takes a `loss_fn(mask)` and a node count; the model adapters in `src/matryoshka_attribution/models` supply the hooks for nodes, edges, neurons, SAE latents and ViT patches.
3. **Strong baselines, same harness**: Node/Edge Pruning, UGS, EAP-IG, AtP*, RelP, AttnLRP, IxG, IG and Expected Gradients are all scored by MIB's own `evaluate_area_under_curve`, never a reimplementation.


## Repo layout

| Path | Contents |
|---|---|
| `src/matryoshka_attribution/` | The installed package: `trainer.learn_scores` (the algorithm), `sigmoid_topk`, `masks` (every masking ablation), `schedules`, `losses`, `evaluate`; baselines (`edge_pruning`, `grad_attribution` for RelP / AttnLRP, `sigmoid_das`); model and data adapters. |
| `scripts/` | Entry points, sweeps, analysis and table generation, one subdirectory per experiment substrate (see below and `scripts/README.md`). |
| `plots/` | One `plot_*.py` per paper figure, shared `palette.py`. |
| `tests/` | Pure-torch unit tests: `uv run python -m pytest tests`. |
| `baselines/` | Our patch to `alestolfo/optimalablation` for the UGS baseline (setup in `scripts/mib/launch/run_ugs.sbatch`). |
| `deps/MIB-circuit-track/` | Clone of **our fork** `aryamanarora/MIB-circuit-track` (with the EAP-IG submodule), cloned at the pinned commit by `scripts/setup.sh` or on first use by `deps.find_mib_path`; gitignored. |
| `results/` | One directory per run, `<variant>_<optim>_lr_<lr>/<task>_<model>_<split>.pkl`; gitignored. |


## Experiments

Every group has python entry points at its top level, cluster launchers (`submit_*.sh`, `*.sbatch`) under `launch/`, and is run from the repo root.

| Directory | Experiment set | Main entry points |
|---|---|---|
| `scripts/mib/` | MIB circuit track, node and edge level: MAttr runs, LR sweeps, ablations, Node/Edge Pruning, UGS and gradient baselines, gemma re-evaluation, every `mib_*`, `lr_sweep`, `hparams_*`, `paired_tests` table. | `eval_mib.py`, `eval_mib_edge.py`, `eval_mib_edge_pruning.py`, `make_mib_table.py`, `make_mib_test_table.py` |
| `scripts/sva/` | SVA+ and arithmetic sweeps over node, MLP-neuron and SAE substrates; optimiser / Adam-eps / cause-direction / zero-ablation / compute-matched follow-ups; top-neuron tables. | `eval_sva.py`, `launch/submit_sva_sweep.sh`, `make_sva_table.py`, `make_sva_neuron_table.py` |
| `scripts/transfer/` | Cross-task transfer of llama3 attribution scores (13x13 matrix). | `eval_transfer_mib.py`, `prep_transfer_sources.py`, `collect_transfer.py` |
| `scripts/vit/` | ViT-B/16 teaser: patch attribution, held-out faithfulness, sparsity ladder, optimiser x LR x eps grid. | `vit_teaser_attr.py`, `vit_teaser_faith.py`, `vit_teaser_optim.py` |
| `scripts/vw/` | Interference-weights replication on a tiny LM: training, scores, MAttr, pruning, transcoder features, audit. | `vw_train.py`, `vw_mattr.py`, `vw_prune.py`, `make_vw_tables.py` |
| `scripts/arith_formats/` | Arithmetic-format circuits: probes, MAttr ablations, per-format tables. | `run_model.py`, `analyse.py`, `make_tables.py` |

---

## Instructions

### Installation

We use `uv` for environments. `scripts/setup.sh` clones the pinned MIB fork into `deps/`, syncs the default environment and runs a no-model smoke check.

```bash
git clone git@github.com:aryamanarora/matryoshka-attribution.git
cd matryoshka-attribution
bash scripts/setup.sh
```

`meta-llama/*` and `google/gemma-2*` are gated, so set `HF_TOKEN` before running those models.

Three environments exist because their torch / transformers / transformer-lens stacks conflict; each is a `uv` dependency group synced on first use:

| Group | Used for | Run with |
|---|---|---|
| `lm` (default) | every LM experiment | `uv run python ...` |
| `tl2` | any **gemma2** cell, and the Node/Edge Pruning and UGS baselines (TL 2.15.4; TL 3.x has a Gemma-2 forward bug) | `UV_PROJECT_ENVIRONMENT=.venv-tl2 uv run --no-default-groups --group tl2 python ...` |
| `vit` | the ViT teaser | `UV_PROJECT_ENVIRONMENT=.venv-vit uv run --no-default-groups --group vit python ...` |

### Run MAttr on a MIB cell

Node level, the paper's headline recipe (soft top-k forward, uniform k, Adam lr 0.05, 500 steps):

```bash
uv run python scripts/mib/eval_mib.py --model gpt2 --task ioi \
  --steps 500 --k-schedule uniform --masking topk --mode iso --lr 0.05 \
  --split validation --train-split train --include-input \
  --output results/mib_node_topk_uniform_lr05
```

Edge level:

```bash
uv run python scripts/mib/eval_mib_edge.py --model gpt2 --task ioi \
  --steps 5000 --k-schedule uniform --masking topk --mode iso --lr 0.05 \
  --split validation --train-split train \
  --output results/mib_edge_topk_uniform_lr05
```

Each run writes `<task>_<model>_<split>.pkl` (the CPR curve and its AUC, from MIB's own evaluator), `<task>_<model>_scores.pt` and `<task>_<model>_importances.json` (the leaderboard format). `scripts/mib/launch/submit_softuni_lr05.sh` submits all 11 cells at both levels; `scripts/mib/make_mib_submission.py` packs a leaderboard submission.

### Run MAttr on SVA+ / arithmetic

```bash
uv run python scripts/sva/eval_sva.py --model llama3 --task nounpp --nodes mlp --method mattr
```

`--nodes` picks the substrate (`node`, `mlp`, `mlp+attn_head`, `mlp_sae_span`, ...), `--method` the attribution method (`mattr`, `ig`, `ixg`, `relp`, `attnlrp`, `mc_ig`, `edge_pruning`, ...). `scripts/sva/launch/submit_sva_sweep.sh` is the full sweep.

### Baselines

- **Node / Edge Pruning** (Bhaskar et al., 2024): `scripts/mib/eval_mib_edge_pruning.py` via `scripts/mib/launch/run_edge_pruning.sbatch` (`tl2` group).
- **UGS**: `scripts/mib/launch/run_ugs.sbatch` then `eval_ugs.sbatch`; the header of the first explains the external checkout and `baselines/optimalablation.patch`.
- **Gradient methods** (EAP-IG, AtP*, AttnLRP, IG, ...): run through the MIB fork's own `run_attribution.py` / `run_evaluation.py`; the launchers under `scripts/mib/launch/*_sc.sh` show the exact calls.

### Tables and figures

Every table under `paper/tabs/` is regenerated by one `scripts/*/make_*_table.py`, and every figure by one `plots/plot_*.py`, all reading `results/` directly:

```bash
uv run python scripts/mib/make_mib_table.py
uv run python plots/plot_mib_test_avg.py
```

Which `results/` directory is the paper's headline is defined once, in `scripts/mib/mattr_variants.py`.


## Citation

```bibtex
@article{arora2026matryoshkaattributionlearningattribute,
      title={Matryoshka attribution: Learning to attribute language model outputs to representations and weights}, 
      author={Aryaman Arora and Kirill Acharya and Nathan Hu and Yanzhe Zhang and Noah Goodman and Dan Jurafsky and Christopher Potts},
      year={2026},
      journal={arXiv:2609.25518},
      url={https://arxiv.org/abs/2609.25518}, 
}
```
