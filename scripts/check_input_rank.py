"""Report the rank/percentile of a target node under MAttr vs NAP-IG per task."""
import json
import sys
from pathlib import Path

RB = Path("results")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "input"

TASKS = [
    ("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
    ("arithmetic_subtraction", "llama3"),
    ("mcqa", "qwen2.5"), ("mcqa", "gemma2"), ("mcqa", "llama3"),
    ("arc_easy", "gemma2"), ("arc_easy", "llama3"), ("arc_challenge", "llama3"),
]


def load(path):
    if not path.exists():
        return None
    d = json.load(open(path))
    nodes = d.get("nodes", d)
    return {n: info["score"] for n, info in nodes.items()
            if n != "logits" and "score" in info}


def input_rank(scores):
    if scores is None or TARGET not in scores:
        return None
    ordered = sorted(scores, key=lambda n: scores[n], reverse=True)
    r = ordered.index(TARGET) + 1
    return r, len(ordered)


def mattr(task, model):
    return load(RB / "mib_node_hard_topk" / f"{task}_{model}_importances.json")


def napig(task, model):
    st = task.replace("_", "-")
    for nm in (f"{st}_{model}", f"{task}_{model}"):
        p = RB / "napig_repro" / "EAP-IG-inputs_patching_node" / nm / "importances.json"
        if p.exists():
            return load(p)
    return None


print(f"target node: {TARGET}")
print(f"{'task/model':28} | {'MAttr rank':>16} | {'NAP-IG rank':>16}")
print("-" * 68)
for task, model in TASKS:
    m = input_rank(mattr(task, model))
    n = input_rank(napig(task, model))
    def fmt(x):
        if x is None:
            return "n/a"
        r, tot = x
        return f"{r}/{tot} ({100*r/tot:.0f}%)"
    print(f"{task+'/'+model:28} | {fmt(m):>16} | {fmt(n):>16}")
