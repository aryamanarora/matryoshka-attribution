"""Emit one line per Node Pruning logit-diff (_ld) cell as its MIB eval lands.

Each line pairs the new _ld CPR AUC with the KL run's AUC on the same cell, so the
objective ablation is readable while it fills in rather than only at 11/11. Exits at 11.

  uv run python scripts/watch_ld_scores.py
"""
import pickle, sys, time
from pathlib import Path

LD = Path("results/eprun_eval_s0.9_ld/EdgePruning_patching_node")
KL = Path("results/eprun_eval/EdgePruning_patching_node")
# MAttr's NODE headline (CLAUDE.md). Node Pruning here is node-level, so the edge-level
# \ourmethod{} row is NOT its comparator -- mixing them overstates the gap by ~5x.
MATTR = Path("results/topklog_lr_0.05")
CELLS = [("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
         ("arithmetic-subtraction", "llama3"), ("mcqa", "qwen2.5"), ("mcqa", "gemma2"),
         ("mcqa", "llama3"), ("arc-easy", "gemma2"), ("arc-easy", "llama3"),
         ("arc-challenge", "llama3")]


def auc(base, task, model):
    p = base / f"{task}_{model}_validation_abs-False.pkl"
    if not p.exists():                              # MAttr dir uses the eval_mib layout
        p = base / f"{task.replace('-', '_')}_{model}_validation.pkl"
    if not p.exists():
        return None
    try:
        return pickle.load(open(p, "rb")).get("area_under")
    except Exception:
        return None            # mid-write; picked up on the next poll


seen, gaps = set(), []
while len(seen) < len(CELLS):
    for task, model in CELLS:
        if (task, model) in seen:
            continue
        a = auc(LD, task, model)
        if a is None:
            continue
        seen.add((task, model))
        k, mt = auc(KL, task, model), auc(MATTR, task, model)
        # "closed" = how much of the KL->MAttr gap the objective swap accounts for. This is
        # the number the objective-confound question actually turns on.
        if k is not None and mt is not None and mt != k:
            frac = f"closes {(a - k) / (mt - k) * 100:4.0f}% of gap"
        else:
            frac = "n/a"
        gaps.append((a - k) / (mt - k) if k is not None and mt is not None and mt != k else None)
        print(f"[{len(seen):2d}/11] {task}/{model:8s} ld {a:5.2f}  KL {k if k is None else round(k,2)}"
              f"  MAttr-node {mt if mt is None else round(mt,2)}   {frac}", flush=True)
    if len(seen) < len(CELLS):
        time.sleep(60)
ok = [g for g in gaps if g is not None]
print(f"ALL 11 done. mean fraction of KL->MAttr gap closed by logit-diff: "
      f"{sum(ok)/len(ok)*100:.0f}% (n={len(ok)})", flush=True)
