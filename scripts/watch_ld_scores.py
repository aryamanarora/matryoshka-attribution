"""Emit one line per Node Pruning logit-diff (_ld) cell as its MIB eval lands, across budgets.

Each line pairs the new _ld CPR AUC with the KL run at the SAME budget and with MAttr's
node headline, so the objective ablation is readable while it fills in rather than only at
11/11. Exits once every budget dir that exists on disk is complete.

Node Pruning here is node-level, so MAttr's edge-level \\ourmethod{} row is NOT its
comparator -- reading across granularities overstates the gap ~5x (1.79 vs 7.17 on
mcqa/qwen2.5). MATTR below is the node headline per CLAUDE.md.

  uv run python scripts/watch_ld_scores.py
"""
import pickle, time
from pathlib import Path

SUB = "EdgePruning_patching_node"
# (budget label, _ld eval dir, KL eval dir at the same budget). The s=0.9 KL runs live in the
# UNSUFFIXED dir -- run_edge_pruning.sbatch only appends _s<S> when sparsity is passed.
# Every budget that make_mib_table.py's EPRUN_SPARSITIES registers must appear here, or its
# cells land in the table without ever being emitted as an event -- and, worse, the exit check
# below ignores that dir, so the watch ends while it is still filling. s=0.5 was missing.
BUDGETS = [("0.1", f"results/eprun_eval_s0.1_ld/{SUB}", f"results/eprun_eval_s0.1/{SUB}"),
           ("0.25", f"results/eprun_eval_s0.25_ld/{SUB}", f"results/eprun_eval_s0.25/{SUB}"),
           ("0.5", f"results/eprun_eval_s0.5_ld/{SUB}", f"results/eprun_eval_s0.5/{SUB}"),
           ("0.8", f"results/eprun_eval_s0.8_ld/{SUB}", f"results/eprun_eval_s0.8/{SUB}"),
           ("0.9", f"results/eprun_eval_s0.9_ld/{SUB}", f"results/eprun_eval/{SUB}"),
           ("0.95", f"results/eprun_eval_s0.95_ld/{SUB}", f"results/eprun_eval_s0.95/{SUB}"),
           ("0.99", f"results/eprun_eval_s0.99_ld/{SUB}", f"results/eprun_eval_s0.99/{SUB}")]
MATTR = Path("results/topklog_lr_0.05")
CELLS = [("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
         ("arithmetic-subtraction", "llama3"), ("mcqa", "qwen2.5"), ("mcqa", "gemma2"),
         ("mcqa", "llama3"), ("arc-easy", "gemma2"), ("arc-easy", "llama3"),
         ("arc-challenge", "llama3")]


def auc(base, task, model):
    p = Path(base) / f"{task}_{model}_validation_abs-False.pkl"
    if not p.exists():                              # MAttr dir uses the eval_mib layout
        p = Path(base) / f"{task.replace('-', '_')}_{model}_validation.pkl"
    if not p.exists():
        return None
    try:
        return pickle.load(open(p, "rb")).get("area_under")
    except Exception:
        return None            # mid-write; picked up on the next poll


seen, deltas = set(), {}
while True:
    for label, ld_dir, kl_dir in BUDGETS:
        for task, model in CELLS:
            key = (label, task, model)
            if key in seen:
                continue
            a = auc(ld_dir, task, model)
            if a is None:
                continue
            seen.add(key)
            k, mt = auc(kl_dir, task, model), auc(MATTR, task, model)
            n = sum(1 for s in seen if s[0] == label)
            bits = [f"[s={label} {n:2d}/11] {task}/{model:8s} ld {a:5.2f}"]
            if k is not None:
                bits.append(f"KL {k:.2f} ({a - k:+.2f})")
                deltas.setdefault(label, []).append(a - k)
            if mt is not None:
                bits.append(f"MAttr-node {mt:.2f}")
            print("  ".join(bits), flush=True)
    # Done when EVERY budget listed above has all 11 cells. Do not gate this on the dir
    # existing: a budget whose jobs are still queued has no dir yet, and treating that as
    # "nothing to wait for" ends the watch on the first pass whenever the already-finished
    # budgets are complete -- which is exactly what happened when 0.25/0.1 were added.
    if all(sum(1 for s in seen if s[0] == lab) >= len(CELLS) for lab, _, _ in BUDGETS):
        break
    time.sleep(60)
for lab, ds in sorted(deltas.items()):
    print(f"ALL s={lab}: mean delta vs KL {sum(ds)/len(ds):+.2f} over {len(ds)} cells", flush=True)
