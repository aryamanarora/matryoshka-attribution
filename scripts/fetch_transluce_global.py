"""Fetch Transluce descriptions for the global top-6 MLP neurons per task (any layer, last-token)."""
import pickle, json, time, urllib.request
import numpy as np

API = "https://transluce--neuron-data-server-fastapi-app.modal.run/read_specific_file"
SPAN_LAST, TOPK = 2, 6
r = pickle.load(open("results/arith_mlp.pkl", "rb"))
paper = json.load(open("/home/guests/aryaman/arithmetic-wild/src/neurons_per_task.json"))
L, I = r["n_layers"], r["intermediate_size"]

def fetch(layer, neuron, sign=1):
    url = f"{API}?layer={layer}&neuron={neuron}&sign={sign}"
    try:
        with urllib.request.urlopen(url, timeout=40) as f:
            d = json.load(f)
        exps = d.get("explanation_summary") or []
        time.sleep(0.3)
        return exps[0][0] if exps else "(no description)"
    except Exception as e:
        return f"(fetch error: {e})"

for t in r["tasks"]:
    S = r["num_spans"][t]
    flat = r["scores"][t].view(L, S, I)[:, SPAN_LAST, :].numpy()   # [L, I]
    idx = np.argsort(-flat, axis=None)[:TOPK]
    pn = set(paper[t])
    print(f"\n### {t}  — global top-{TOPK} (any layer, last-token)")
    for i in idx:
        layer, neuron = int(i // I), int(i % I)
        tag = " [PAPER L18]" if (layer == 18 and neuron in pn) else ""
        print(f"  L{layer}:{neuron}{tag}\n      {fetch(layer, neuron)}")
