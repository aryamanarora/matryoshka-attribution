"""Decompose the GIM / AttnLRP agreement on a single MIB cell.

Reading the code, GIM and AttnLRP apply the SAME attention rule -- q,k gradients x1/4 and
v x1/2 -- reached two different ways (AttnLRP composes HalfGrad on the QK and OV matmul
outputs; GIM applies ScaleGrad constants directly to hook_q/k/v). Both also freeze the norm
scale. That leaves exactly two differences:

  (1) softmax backward: AttnLRP uses the ordinary softmax Jacobian (T=1); GIM evaluates the
      same JVP at a tempered softmax(x/T), T=2.
  (2) MLP activation: AttnLRP secant-linearizes (act(z)/z detached); GIM keeps the true
      derivative. On gpt2 (non-gated MLP) GIM therefore installs no MLP hook at all.

This script runs four configs and correlates the resulting node scores, so each difference
can be attributed separately:

  A  AttnLRP                    as shipped
  B  GIM                        as shipped
  D  GIM + secant act           = A but for the softmax temperature   -> corr(A,D) isolates (1)
  E  GIM + secant act, T=1      predicted to be IDENTICAL to A        -> corr(A,E) is the test
                                                                          of the whole reading

corr(B,D) isolates (2). If corr(A,E) ~ 1.0 the two methods provably differ only in (1)+(2).
"""
import argparse
from functools import partial

import torch

from transformer_lens import HookedTransformer
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES
from eap.graph import Graph
import eap.attribute_node as AN

# build_relp_fwd_hooks takes gim_T but get_scores_relp does not forward it, so inject it here.
_orig_build = AN.build_relp_fwd_hooks
_GIM_T = [2.0]


def _patched_build(*a, **kw):
    kw['gim_T'] = _GIM_T[0]
    return _orig_build(*a, **kw)


AN.build_relp_fwd_hooks = _patched_build

CONFIGS = [
    ("A AttnLRP",            2.0, dict(detach_qk=False, shapley_attn=True, softmax_rule=False, linearize_act=True)),
    ("B GIM",                2.0, dict(detach_qk=False, gim_attn=True, linearize_act=False)),
    ("D GIM+secant",         2.0, dict(detach_qk=False, gim_attn=True, linearize_act=True)),
    ("E GIM+secant,T=1",     1.0, dict(detach_qk=False, gim_attn=True, linearize_act=True)),
]


def spearman(x, y):
    """Rank correlation without scipy (the MIB venv has no scipy)."""
    def rank(v):
        order = v.argsort()
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(len(v), dtype=torch.float64)
        return r
    a, b = rank(x.double()), rank(y.double())
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--task", default="ioi")
    p.add_argument("--num-examples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=20)
    args = p.parse_args()

    if args.model in ("qwen2.5", "gemma2", "llama3"):
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[args.model],
                                                  attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[args.model])
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    graph = Graph.from_model(model, node_scores=True)
    dataset = HFEAPDataset(f'mib-bench/{TASKS_TO_HF_NAMES[args.task]}', model.tokenizer, split='train',
                           task=args.task, model_name=args.model, num_examples=args.num_examples)
    dataloader = dataset.to_dataloader(batch_size=args.batch_size)
    metric = partial(get_metric('logit_diff', args.task, model.tokenizer, model), mean=True, loss=True)

    out = {}
    for name, T, cfg in CONFIGS:
        _GIM_T[0] = T
        g = Graph.from_model(model, node_scores=True)
        out[name] = AN.get_scores_relp(model, g, dataloader, metric, quiet=True, **cfg).float().cpu()
        print(f"ran {name}", flush=True)

    names = [n for n, _, _ in CONFIGS]
    print(f"\n{args.task}/{args.model}, n={args.num_examples}, {len(out[names[0]])} nodes")
    print(f"{'':22s}" + "".join(f"{n:>22s}" for n in names))
    for i in names:
        print(f"{i:22s}" + "".join(f"{spearman(out[i], out[j]):>22.4f}" for j in names))


if __name__ == "__main__":
    main()
