"""Collect acc-AUC (and CPR-AUC) for the MAttr (L2A) MIB variants.

Reads results/mattr_accauc/<variant>_patching_node/<task>-<model>_validation_abs-False.pkl
(patched eval). Prints per-cell tables, rows = task/model, cols = variant, + per-variant means.
Run:  .venv/bin/python collect_accauc_mattr.py
"""
import pickle, os

CELLS = [
    ("gpt2", "ioi"), ("qwen2.5", "ioi"), ("qwen2.5", "mcqa"),
    ("gemma2", "ioi"), ("gemma2", "mcqa"), ("gemma2", "arc_easy"),
    ("llama3", "ioi"), ("llama3", "mcqa"),
    ("llama3", "arithmetic_subtraction"), ("llama3", "arc_easy"), ("llama3", "arc_challenge"),
]
VARIANTS = [
    ("hard_topk_log", "mib_node_hard_topk_log"),      # L2A headline
    ("hard_topk", "mib_node_hard_topk"),              # uniform-k ablation
    ("htk_gumbel", "mib_node_hard_topk_gumbel"),
    ("topk_log", "mib_node_topk_log"),
    ("bern", "mib_node_bernoulli_reinforce"),
    ("bern_log", "mib_node_bernoulli_reinforce_log"),
    ("dtau", "mib_node_detached_tau"),
    ("dtau_log", "mib_node_detached_tau_log"),
    ("id_sgd", "mib_node_identity_sgd"),
    ("id_sgd_log", "mib_node_identity_sgd_log"),
    ("id_gum_log", "mib_node_identity_gumbel_sgd_log"),
    ("id_gum_unif", "mib_node_identity_gumbel_sgd_uniform"),
]
BASE = "results/mattr_accauc"


def load(variant, task, model):
    stask = task.replace("_", "-")
    p = f"{BASE}/{variant}_patching_node/{stask}_{model}_validation_abs-False.pkl"
    if not os.path.exists(p):
        return None
    try:
        return pickle.load(open(p, "rb"))
    except Exception:
        return None


def table(metric, label):
    hdr = f"{label + ' | cell':26}" + "".join(f"{t:>12}" for t, _ in VARIANTS)
    print(hdr); print("-" * len(hdr))
    rows = []
    for model, task in CELLS:
        vals = {t: (None if (d := load(v, task, model)) is None else d.get(metric))
                for t, v in VARIANTS}
        rows.append(vals)
        s = f"{task + '/' + model:26}" + "".join(
            (f"{vals[t]:>12.4f}" if vals[t] is not None else f"{'--':>12}") for t, _ in VARIANTS)
        print(s)
    print("-" * len(hdr))
    for t, _ in VARIANTS:
        done = [r[t] for r in rows if r[t] is not None]
        mean = sum(done) / len(done) if done else float("nan")
        print(f"  {t:14} mean over {len(done):2}/{len(CELLS)} = {mean:.4f}")
    print()


if __name__ == "__main__":
    table("acc_auc", "acc-AUC")
    print("=" * 70)
    table("area_under", "CPR-AUC")
