"""Isolation test: does the multitask DAS intervention actually move the logits?"""
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent))
from learning_to_attribute import make_rotate_layer  # noqa
from attribute_multitask import build_task

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b", dtype=torch.bfloat16,
                                             device_map="auto")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

hidden = model.config.hidden_size
nL = model.config.num_hidden_layers
das_dim = 64
rots = {li: make_rotate_layer(hidden, das_dim).to(device) for li in range(nL)}

ds, hooker, scores = build_task(model, "syntaxgym/agr_sv_num_subj-relc", das_dim,
                                "last", False, rots, device)  # legacy_sufficient=False (denoising)
pair = ds.sample_pair()
t = ds.tokenize_pair(pair, tok, device=str(device))
hooker.set_alignment(t.base_alignment, t.src_alignment,
                     t.base_input_ids.shape[1], t.src_input_ids.shape[1])
print("base_span_to_pos:", hooker.base_span_to_pos)
print("num cf_acts before cache:", len(hooker.cf_acts_resid))
hooker.cache_cf_activations(t.src_input_ids)
print("num cf_acts after cache:", len(hooker.cf_acts_resid))
hooker.register_hooks()
print("num hooks registered:", len(hooker._hooks))

with torch.no_grad():
    hooker.mask = None
    clean = model(t.base_input_ids).logits[0, -1].float()
    hooker.mask = torch.ones(hooker.total, device=device)
    allon = model(t.base_input_ids).logits[0, -1].float()
    hooker.mask = torch.zeros(hooker.total, device=device)
    alloff = model(t.base_input_ids).logits[0, -1].float()

print("max|allon - clean| :", (allon - clean).abs().max().item())
print("max|alloff - clean|:", (alloff - clean).abs().max().item())
# also check base vs cf differ
print("cf_acts[0] vs base norm diff (layer0):",
      (hooker.cf_acts_resid[0].float().mean()).item())
