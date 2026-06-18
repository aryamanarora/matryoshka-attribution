"""Generate an arithmetic-wild-format addition dataset for Llama-3.1-8B base:
sample a+b (a,b in 1..100), keep pairs the model gets right with a single-token sum.
"""
import json, random, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B/addition"
TARGET = 1600
random.seed(1265)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.bfloat16, device_map="auto").eval()

# candidate (a,b) with single-token sum
cands = []
seen = set()
while len(cands) < 6000:
    a, b = random.randint(1, 100), random.randint(1, 100)
    if (a, b) in seen:
        continue
    seen.add((a, b))
    s = a + b
    if len(tok.encode(str(s), add_special_tokens=False)) == 1:
        cands.append((a, b, s))

correct = []
with torch.no_grad():
    for i in range(0, len(cands), 64):
        batch = cands[i:i + 64]
        prompts = [f"{a}+{b}=" for a, b, _ in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        # last real token per row (left pad? default right pad -> use attention mask sum-1)
        lengths = enc["attention_mask"].sum(1) - 1
        logits = model(**enc).logits
        pred = logits[torch.arange(len(batch)), lengths].argmax(-1)
        for j, (a, b, s) in enumerate(batch):
            if pred[j].item() == tok.encode(str(s), add_special_tokens=False)[0]:
                correct.append((a, b, s))
        if len(correct) >= TARGET + 50:
            break
print(f"{len(correct)} model-correct single-token addition examples")

def ex(a, b, s):
    return {"input": str(a), "offset": str(b), "output": str(s),
            "raw_input": f"{a}+{b}=", "raw_output": str(s), "premod": s}

random.shuffle(correct)
correct = correct[:TARGET]
cf_pool = correct[:]
inputs, cfs = [], []
for (a, b, s) in correct:
    c = random.choice(cf_pool)
    while c == (a, b, s):
        c = random.choice(cf_pool)
    inputs.append(ex(a, b, s))
    cfs.append([ex(*c)])
os.makedirs(OUT, exist_ok=True)
json.dump({"input": inputs, "counterfactual_inputs": cfs}, open(f"{OUT}/filtered_dataset.json", "w"))
print(f"wrote {OUT}/filtered_dataset.json ({len(inputs)} pairs)")
