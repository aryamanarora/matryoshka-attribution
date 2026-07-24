"""Recompute the CPR anchors (baseline B, corrupted C) for the 3 gemma2 cells under the
CORRECT transformer_lens (MIB venv, TL 2.15.4); the originals in results/anchors/ were computed
under the buggy TL 3.2.1. Overwrites results/anchors/{task}_gemma2.json (used by the raw
logit-diff curve plot). Self-contained (no eval_mib / learning_to_attribute import).
Run in MIB venv:  PYTHONPATH=MIB-circuit-track:MIB-circuit-track/EAP-IG/src \
  MIB-circuit-track/.venv/bin/python scripts/reeval_gemma_anchors.py
"""
import sys, json
from functools import partial
from pathlib import Path
import torch

mib = Path("./MIB-circuit-track").resolve()
sys.path.insert(0, str(mib)); sys.path.insert(0, str(mib / "EAP-IG" / "src"))
import importlib.metadata as M
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.evaluate import evaluate_baseline, evaluate_graph
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.dataset import HFEAPDataset

assert M.version("transformer_lens").startswith("2."), "run in MIB venv (TL 2.x)"
HFTASK = {"ioi": "ioi", "mcqa": "copycolors_mcqa", "arc_easy": "arc_easy"}
HEAD = {"ioi": 200, "mcqa": 0, "arc_easy": 0}   # match reeval_gemma_mib caps
OUT = Path("results/anchors")

tl = HookedTransformer.from_pretrained("google/gemma-2-2b", attn_implementation="eager", torch_dtype=torch.bfloat16)
tl.cfg.use_split_qkv_input = True; tl.cfg.use_attn_result = True
tl.cfg.use_hook_mlp_in = True; tl.cfg.ungroup_grouped_query_attention = True

for task in ["ioi", "mcqa", "arc_easy"]:
    g = Graph.from_model(tl, node_scores=True)
    g.nodes_scores = torch.zeros(g.n_forward)
    ds = HFEAPDataset(f"mib-bench/{HFTASK[task]}", tl.tokenizer, split="validation", task=task, model_name="gemma2")
    if HEAD[task]: ds.head(HEAD[task])
    dl = ds.to_dataloader(batch_size=8)
    am = partial(get_metric("logit_diff", task, tl.tokenizer, tl), mean=False, loss=False)
    baseline = evaluate_baseline(tl, dl, am).mean().item()
    g.apply_topn(0, True, level="node", prune=True)
    corrupted = evaluate_graph(tl, g, dl, am, intervention="patching").mean().item()
    rec = {"task": task, "model": "gemma2", "split": "validation", "n_eval": len(ds),
           "baseline": baseline, "corrupted": corrupted}
    (OUT / f"{task}_gemma2.json").write_text(json.dumps(rec, indent=2))
    print(f"{task}/gemma2: B={baseline:.3f} C={corrupted:.3f} (n={len(ds)}) TL {M.version('transformer_lens')}")
