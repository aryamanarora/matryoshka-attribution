# scripts/ layout

One directory per experiment substrate. Inside each: python entry points and analysis /
table scripts at the top level, cluster launchers (`submit_*.sh`, `*.sbatch`, pipeline
stage scripts) in `launch/`, YAML configs in `configs/`. Everything is run from the repo
root (`python scripts/<group>/<script>.py`, `bash scripts/<group>/launch/<script>.sh`);
results/log paths inside the scripts are root-relative and unchanged.

| Group | What lives there |
|---|---|
| `mib/` | MIB circuit track, node + edge level: `eval_mib.py`, `eval_mib_edge.py`, the Node/Edge Pruning baseline runner (`eval_mib_edge_pruning.py`), gemma re-evals, cross-method rank comparisons, and every `make_*_table.py` feeding `paper/tabs/{mib,lr,edge_lr,cpr}_*.tex`. `launch/` holds all MIB sweeps plus `mib_node*.sbatch` / `mib_edge.sbatch`; `configs/` the per-cell YAMLs. |
| `sva/` | SVA+ / arithmetic sweeps over node, neuron and SAE substrates (`eval_sva.py`), the optimizer / eps / cause / zero-ablation follow-ups, fingerprint + neuron tables, `sva_sweep.sbatch`. |
| `transfer/` | Cross-task transfer of llama3 scores (`eval_transfer_mib.py`, `prep_transfer_sources.py`, `collect_transfer.py`). Imports `eval_mib` from `mib/`. |
| `vit/` | ViT-B/16 teaser: attribution, faithfulness, sparsity ladder, optimizer grid; deps are the `vit` uv group, run via `UV_PROJECT_ENVIRONMENT=.venv-vit uv run --no-default-groups --group vit python ...`. |
| `vw/` | Interference-weights tiny-LM replication (`vw_*.py`, staged `launch/vw_stage*.sh`, `vw.sbatch`). |
| `arith_formats/` | Arithmetic-format circuits: `run_model.py`, probes, MAttr ablations, `make_tables.py`. |
| `tools/` | Repo utilities: `lock_paper.sh`, `test_attnlrp_hf.py`. |

Sibling imports within a group go through `sys.path.insert(0, <own dir>)`; `plots/*.py`
that reuse a group's module insert `../scripts/<group>` explicitly.
