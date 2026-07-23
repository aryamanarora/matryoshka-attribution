"""Ground-truth referee: run HuggingFace's own gemma-2-2b (no transformer_lens) on the
arc_easy counterfactual inputs and compute the same logit_diff metric. Compare to the two TL
versions' cf_forward: TL3.2.1 gave +1.47 (acc .585), TL2.15.4 gave -3.04 (acc .015).
Whichever HF agrees with is the correct Gemma-2 forward.
"""
import sys
from pathlib import Path
import torch

mib = Path("./MIB-circuit-track").resolve()
sys.path.insert(0, str(mib)); sys.path.insert(0, str(mib / "EAP-IG" / "src"))
import importlib.metadata as M
from transformers import AutoModelForCausalLM, AutoTokenizer
from MIB_circuit_track.dataset import HFEAPDataset

print(f"transformers=={M.version('transformers')}")
name = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager", device_map="cuda")
model.eval()

ds = HFEAPDataset("mib-bench/arc_easy", tok, split="validation", task="arc_easy", model_name="gemma2")
ds.head(200)
dl = ds.to_dataloader(batch_size=4)

diffs = []
for clean, corrupted, label in dl:
    enc = tok(list(corrupted), return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
    with torch.inference_mode():
        logits = model(**enc).logits  # (b, pos, vocab)
    last = enc["attention_mask"].sum(1) - 1  # last real token idx per example
    lab = torch.as_tensor(label, device="cuda")  # (b,2): [correct, incorrect]
    for i in range(logits.size(0)):
        lg = logits[i, last[i]].float()
        diffs.append((lg[lab[i, 0]] - lg[lab[i, 1]]).item())

d = torch.tensor(diffs)
print(f"HF gemma-2-2b counterfactual: acc={ (d>0).float().mean():.3f}  mean logit_diff={d.mean():.3f}")
print("  TL3.2.1 gave acc .585 / +1.47 ; TL2.15.4 gave acc .015 / -3.04")
