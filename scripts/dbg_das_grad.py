"""Decisive check: does the DAS rotation receive gradient in train_step?
Builds the exact multitask DAS setup for one task, runs a few train_steps, and
prints the grad norm of the rotation params vs the scores."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from learning_to_attribute.data.arithmetic_wild import ArithmeticWildDataset
from learning_to_attribute.sigmoid_das import make_rotate_layer
from attribute_multitask import sample_k, train_step, get_hooks_classes

DATA = "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B"
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16, device_map="auto").eval()
for p in model.parameters():
    p.requires_grad_(False)
hidden, nL = model.config.hidden_size, model.config.num_hidden_layers
das_dim = 32
args = SimpleNamespace(mask="das", das_dim=das_dim, T=0.5, n_iters=30, k_schedule="uniform",
                       hard_fwd=True, loss="ce", pos_strategy="last", seed=42)
_, SpanHooks = get_hooks_classes(model)
rot = {li: make_rotate_layer(hidden, das_dim).to(dev) for li in range(nL)}
ds = ArithmeticWildDataset("months", DATA, seed=42)
hooker = SpanHooks(model, "das", ds.num_spans, pos_strategy="last", sufficient=False)
hooker.set_das_dim(das_dim)
for li in range(nL):
    hooker.R[li] = rot[li]
scores = nn.Parameter(torch.zeros(hooker.total, device=dev))
rot_params = [p for rl in rot.values() for p in rl.parameters()]
opt = torch.optim.Adam([{"params": [scores], "lr": 0.03}, {"params": rot_params, "lr": 0.02}])

for step in range(5):
    opt.zero_grad()
    loss, hooker = train_step(model, ds, hooker, scores, tok, dev, args, legacy_sufficient=False)
    loss.backward()
    hooker.remove_hooks()
    sg = scores.grad.norm().item() if scores.grad is not None else None
    rg0 = rot_params[0].grad
    rg18 = rot_params[18].grad
    rgtot = sum((p.grad.norm().item()**2 for p in rot_params if p.grad is not None)) ** 0.5
    n_with_grad = sum(1 for p in rot_params if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"step{step} loss={loss.item():.3f} | scores.grad={sg:.4e} | "
          f"rot.grad_total={rgtot:.4e} | rot[0].grad={'None' if rg0 is None else f'{rg0.norm():.3e}'} | "
          f"rot[18].grad={'None' if rg18 is None else f'{rg18.norm():.3e}'} | rot params w/ nonzero grad={n_with_grad}/{len(rot_params)}")
    opt.step()
print("rotation weight changed after steps:", (rot[18].weight.detach() - torch.eye(hidden, das_dim, device=dev)).abs().sum().item() > 0)
