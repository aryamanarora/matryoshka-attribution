"""Isolate numerics: run MIB's node-level NAP-IG (EAP-IG-inputs) + CPR eval on ioi/qwen2.5
in bf16 (as the benchmark does) vs fp32, and print the CPR-AUC for each. If the bf16 floor
(~0.25) lifts in fp32, the IOI-Qwen gradient-method collapse is a precision artifact.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EAP-IG", "src"))
from functools import partial

import numpy as np
import torch
from transformer_lens import HookedTransformer

from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from eap.graph import Graph
from eap.attribute_node import attribute_node

MODEL, FULL, TASK = "qwen2.5", "Qwen/Qwen2.5-0.5B", "ioi"
HFT = "mib-bench/ioi"


def run(dtype, method="EAP-IG-inputs"):
    model = HookedTransformer.from_pretrained(FULL, attn_implementation="eager", torch_dtype=dtype)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    graph = Graph.from_model(model, node_scores=True)
    train = HFEAPDataset(HFT, model.tokenizer, split="train", task=TASK, model_name=MODEL, num_examples=100)
    dl = train.to_dataloader(batch_size=20)
    metric = get_metric("logit_diff", TASK, model.tokenizer, model)
    attribute_node(model, graph, dl, partial(metric, mean=True, loss=True),
                   method, "patching", neuron=False, ig_steps=5,
                   optimal_ablation_path=None, intervention_dataloader=dl)

    node_scores = np.array([float(n.score) for name, n in graph.nodes.items()
                            if name != "logits" and hasattr(n, "score")])

    val = HFEAPDataset(HFT, model.tokenizer, split="validation", task=TASK, model_name=MODEL)
    vdl = val.to_dataloader(batch_size=20)
    out = evaluate_area_under_curve(model, graph, vdl,
                                    partial(metric, mean=False, loss=False), level="node")
    area_under = out[1]
    au = float(np.mean(area_under)) if hasattr(area_under, "__len__") else float(area_under)
    finite = node_scores[np.isfinite(node_scores)]
    print(f"[{method} | {dtype}]  CPR-AUC = {au:.4f}   "
          f"nodes={len(finite)}  |score| mean={np.abs(finite).mean():.3e} "
          f"std={finite.std():.3e}  #|s|<1e-4={(np.abs(finite) < 1e-4).sum()}")
    del model
    torch.cuda.empty_cache()
    return au, finite


if __name__ == "__main__":
    print("=== MIB node NAP-IG (EAP-IG-inputs) on ioi/qwen2.5: bf16 vs fp32 ===")
    au_bf, s_bf = run(torch.bfloat16)
    au_f32, s_f32 = run(torch.float32)
    print(f"\nSUMMARY  bf16 CPR-AUC={au_bf:.4f}   fp32 CPR-AUC={au_f32:.4f}   "
          f"lift={au_f32 - au_bf:+.4f}")
