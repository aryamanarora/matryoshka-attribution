"""Diagnostic: no-intervention metrics + top-10 next-token preds for NPI subj-relc pairs."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from learning_to_attribute.data.causalgym import CausalGymDataset

NAME = "meta-llama/Llama-3.1-8B"
TASK = "npi_any_subj-relc"
N = 3

tok = AutoTokenizer.from_pretrained(NAME)
hf = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16,
                                          attn_implementation="eager").cuda().eval()

cg = CausalGymDataset(f"syntaxgym/{TASK}", seed=1)
for n in range(N):
    p = cg.sample_pair()
    clean = "".join(p.base_spans).replace("<|endoftext|>", "").lstrip()
    corr  = "".join(p.src_spans).replace("<|endoftext|>", "").lstrip()
    bid = tok(p.base_label).input_ids[-1]
    sid = tok(p.src_label).input_ids[-1]

    print("=" * 90)
    print(f"[{n}] CLEAN : {clean!r}")
    print(f"    CORRUPT: {corr!r}")
    print(f"    base_label={p.base_label!r} -> id {bid} ({tok.decode([bid])!r})")
    print(f"    src_label ={p.src_label!r} -> id {sid} ({tok.decode([sid])!r})")

    ids = tok(clean, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits = hf(ids).logits.float()[0, -1]
    probs = logits.softmax(-1)
    print(f"    last input tok: {tok.decode(ids[0,-1:].tolist())!r}  (n_tok={ids.shape[1]})")
    print(f"    p(base)={probs[bid].item():.4f}  p(source)={probs[sid].item():.4f}  "
          f"logit_diff={logits[bid].item()-logits[sid].item():+.3f}  "
          f"argmax={'base' if logits[bid]>logits[sid] else 'source'}")
    top = probs.topk(10)
    print("    top-10 next-token preds:")
    for rank, (pv, ti) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        mark = "  <-- base" if ti == bid else ("  <-- source" if ti == sid else "")
        print(f"      {rank+1:2d}. {tok.decode([ti])!r:<14} p={pv:.4f}{mark}")
