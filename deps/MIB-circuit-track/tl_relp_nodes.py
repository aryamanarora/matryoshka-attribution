"""TL-side RelP node scores (my port) on Llama-3.2-1B for the same pair as hf_relp_nodes.py.
Saved to tl_nodes.json for cross-check."""
import json, torch, sys
MODE = sys.argv[1] if len(sys.argv)>1 else "relp"
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.attribute_node import attribute_node

MODEL = "meta-llama/Llama-3.2-1B"
CLEAN     = "When John and Mary went to the store, John gave a drink to"
CORRUPTED = "When Alice and Bob went to the shop, Carol gave a drink to"
ANS_POS, ANS_NEG = " Mary", " John"

model = HookedTransformer.from_pretrained_no_processing(MODEL, dtype=torch.float32)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
tok = model.tokenizer
pos_id = tok(ANS_POS, add_special_tokens=False).input_ids[0]
neg_id = tok(ANS_NEG, add_special_tokens=False).input_ids[0]

def metric(logits, clean_logits, input_lengths, labels, mean=True, loss=False):
    idx = torch.arange(logits.size(0), device=logits.device)
    last = logits[idx, input_lengths - 1]
    val = last[:, pos_id] - last[:, neg_id]
    return val.mean() if mean else val

class DS(Dataset):
    def __len__(self): return 1
    def __getitem__(self, i): return CLEAN, CORRUPTED, 0
def collate(xs):
    c, k, l = zip(*xs); return list(c), list(k), list(l)
dl = DataLoader(DS(), batch_size=1, collate_fn=collate)

graph = Graph.from_model(model, node_scores=True)
attribute_node(model, graph, dl, partial(metric, mean=True), method=("RelP" if MODE=="relp" else "RelP-norules"),
               intervention="patching", quiet=True, neuron=False)

scores = {}
for name, node in graph.nodes.items():
    if name == "logits":
        continue
    try:
        scores[name] = float(node.score)
    except Exception:
        pass
json.dump(scores, open(f"tl_nodes_{MODE}.json", "w"))
print(f"wrote tl_nodes_{MODE}.json with", len(scores), "node scores")
print("sample:", dict(list(scores.items())[:4]))
