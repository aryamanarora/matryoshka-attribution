"""Is the fully-patched (topn=0) 'corrupted' state a faithful reproduction of the model's
behaviour on the counterfactual input? Run in each venv.

Computes, on arc_easy/gemma2:
  clean_acc      : accuracy of model on CLEAN tokens        (fraction logit_diff>0)
  cf_forward_acc : accuracy of model on COUNTERFACTUAL tokens (plain forward)  <- ground truth
  patched_acc    : accuracy of the topn=0 fully-ablated graph (EAP reconstruction)  <- what CPR/acc uses

If patching is faithful, patched_acc == cf_forward_acc. A clean forward (clean_acc, cf_forward_acc)
should agree across TL versions; only the reconstruction (patched_acc) can diverge.
"""
import sys
from functools import partial
from pathlib import Path
import torch

mib = Path("./MIB-circuit-track").resolve()
sys.path.insert(0, str(mib)); sys.path.insert(0, str(mib / "EAP-IG" / "src"))
import importlib.metadata as M
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.utils import tokenize_plus
from eap.evaluate import evaluate_baseline, evaluate_graph
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.dataset import HFEAPDataset

print(f"transformer_lens=={M.version('transformer_lens')}")
tl = HookedTransformer.from_pretrained("google/gemma-2-2b", attn_implementation="eager", torch_dtype=torch.bfloat16)
tl.cfg.use_split_qkv_input = True; tl.cfg.use_attn_result = True
tl.cfg.use_hook_mlp_in = True; tl.cfg.ungroup_grouped_query_attention = True

g = Graph.from_json("results/topklog_lr_0.05/arc_easy_gemma2_importances.json")
ds = HFEAPDataset("mib-bench/arc_easy", tl.tokenizer, split="validation", task="arc_easy", model_name="gemma2")
ds.head(200)
dl = ds.to_dataloader(batch_size=4)
metric = get_metric("logit_diff", "arc_easy", tl.tokenizer, tl)
am = partial(metric, mean=False, loss=False)

# clean forward accuracy
clean_ex = evaluate_baseline(tl, dl, am)
clean_acc = (clean_ex > 0).float().mean().item()

# direct counterfactual forward: run model on the CORRUPTED tokens, apply same metric
cf_vals = []
for clean, corrupted, label in dl:
    ct, mask, ilen, npos = tokenize_plus(tl, corrupted)
    with torch.inference_mode():
        logits = tl(ct, attention_mask=mask)
    r = am(logits, None, ilen, label).cpu()
    cf_vals.append(r if r.dim() else r.unsqueeze(0))
cf_ex = torch.cat(cf_vals)
cf_acc = (cf_ex > 0).float().mean().item()

# fully-patched (topn=0) via EAP reconstruction
g.apply_topn(0, True, level="node", prune=True)
patched_ex = evaluate_graph(tl, g, dl, am, intervention="patching")
patched_acc = (patched_ex > 0).float().mean().item()

print(f"clean_acc       = {clean_acc:.3f}  (mean logit_diff {clean_ex.float().mean():.3f})")
print(f"cf_forward_acc  = {cf_acc:.3f}  (mean logit_diff {cf_ex.float().mean():.3f})   <- true counterfactual")
print(f"patched_acc     = {patched_acc:.3f}  (mean logit_diff {patched_ex.float().mean():.3f})   <- EAP reconstruction")
