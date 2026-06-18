"""Fetch Transluce neuron descriptions for the top L18 last-token MLP neurons per task."""
import pickle, json, sys, time, urllib.request
import numpy as np

API = "https://transluce--neuron-data-server-fastapi-app.modal.run/read_specific_file"
L18, SPAN_LAST, TOPK = 18, 2, 6
r = pickle.load(open("results/arith_mlp.pkl", "rb"))
paper = json.load(open("/home/guests/aryaman/arithmetic-wild/src/neurons_per_task.json"))
L, I = r["n_layers"], r["intermediate_size"]

# top-K L18 neurons per task + the union to fetch
top = {}
for t in r["tasks"]:
    S = r["num_spans"][t]
    v = r["scores"][t].view(L, S, I)[L18, SPAN_LAST, :].numpy()
    top[t] = [int(n) for n in np.argsort(-v)[:TOPK]]
union = sorted(set(n for ns in top.values() for n in ns))

cache = {}
def fetch(neuron, sign=1):
    if neuron in cache:
        return cache[neuron]
    url = f"{API}?layer={L18}&neuron={neuron}&sign={sign}"
    try:
        with urllib.request.urlopen(url, timeout=40) as f:
            d = json.load(f)
        exps = d.get("explanation_summary") or []
        cache[neuron] = exps[0][0] if exps else "(no description)"
    except Exception as e:
        cache[neuron] = f"(fetch error: {e})"
    time.sleep(0.3)
    return cache[neuron]

for n in union:
    fetch(n)

for t in r["tasks"]:
    pn = set(paper[t])
    print(f"\n### {t}  — top-{TOPK} L18 last-token neurons")
    for n in top[t]:
        tag = " [PAPER]" if n in pn else ""
        print(f"  L18:{n}{tag}\n      {cache[n]}")
