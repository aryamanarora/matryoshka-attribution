"""Top-10 MLP neurons per NPI method, decoded to (layer,pos,neuron) + Transluce descriptions."""
import json, time, urllib.request
import torch

API = "https://transluce--neuron-data-server-fastapi-app.modal.run/read_specific_file"
RES = "results/sva"
SEQ, INTER = 9, 14336
CACHE_PATH = "results/transluce_cache.json"
B = "npi_any_subj-relc_llama3_mlp_sufficient_hard_topk_adam"
METHODS = [
    ("logit-diff log", f"{B}_bs1"),
    ("logit-diff unif", f"{B}_uniformk_bs1"),
    ("logit-diff adapt", f"{B}_adaptivek_bs1"),
    ("logit log", f"{B}_logit_bs1"),
    ("CE log", f"{B}_ce_bs1"),
    ("hinge log", f"{B}_hinge_bs1"),
    ("hinge adapt", f"{B}_hinge_adaptivek_bs1"),
    ("IG", "npi_any_subj-relc_llama3_mlp_ig"),
    ("RelP", "npi_any_subj-relc_llama3_mlp_relp"),
    ("IxG", "npi_any_subj-relc_llama3_mlp_ixg"),
]

try:
    cache = json.load(open(CACHE_PATH))
except FileNotFoundError:
    cache = {}

def describe(layer, neuron, sign=1):
    key = f"{layer}:{neuron}:{sign}"
    if key in cache:
        return cache[key]
    url = f"{API}?layer={layer}&neuron={neuron}&sign={sign}"
    try:
        with urllib.request.urlopen(url, timeout=40) as f:
            d = json.load(f)
        exps = d.get("explanation_summary") or []
        cache[key] = exps[0][0] if exps else "(no description)"
    except Exception as e:
        cache[key] = f"(fetch error: {e})"
    time.sleep(0.3)
    return cache[key]

def decode(idx):
    layer = idx // (SEQ * INTER)
    rem = idx % (SEQ * INTER)
    return layer, rem // INTER, rem % INTER  # layer, pos, neuron

for lab, tag in METHODS:
    s = torch.load(f"{RES}/{tag}.scores.pt", map_location="cpu").float()
    top = torch.topk(s, 10)
    print(f"\n### {lab}")
    for rank, (sc, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        L, pos, N = decode(idx)
        desc = describe(L, N)
        print(f"  {rank:2d}. L{L:>2d} N{N:<5d} pos{pos}  (score {sc:.3g})  {desc}")
    json.dump(cache, open(CACHE_PATH, "w"))

json.dump(cache, open(CACHE_PATH, "w"))
print(f"\n[cached {len(cache)} descriptions -> {CACHE_PATH}]")
