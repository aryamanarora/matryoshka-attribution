"""Neuronpedia feature lookups for SAE-latent tables and figures (a JSON cache next to the
results, one request per uncached feature). Used by plots/plot_sva_top_units.py."""
import json
import os
import time
import urllib.request

NP_API = "https://www.neuronpedia.org/api/feature/{model}/{sae}/{idx}"
NP_URL = "https://www.neuronpedia.org/{model}/{sae}/{idx}"


def load_cache(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def fetch_feature(model, sae, idx, cache):
    """Return dict {desc, pos, neg} for a feature; cache full info (re-fetch stale
    string-only cache entries from the description-only version of this helper)."""
    key = f"{model}/{sae}/{idx}"
    if isinstance(cache.get(key), dict):
        return cache[key]
    url = NP_API.format(model=model, sae=sae, idx=idx)
    info = {"desc": "(no description)", "pos": [], "neg": []}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(req, timeout=40) as f:
            d = json.load(f)
        exps = d.get("explanations") or []
        if exps:
            info["desc"] = exps[0].get("description", info["desc"]).strip()
        info["pos"] = d.get("pos_str") or []   # top (promoted) logit tokens
        info["neg"] = d.get("neg_str") or []   # bottom (suppressed) logit tokens
    except Exception as e:
        info["desc"] = f"(fetch error: {e})"
    cache[key] = info
    time.sleep(0.3)
    return info
