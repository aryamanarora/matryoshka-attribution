"""Collect CPR AUC (area_under) for the 3 node methods across the 12 cells into a table."""
import pickle, glob, os

CELLS = [
    ("gpt2","ioi"),("qwen2.5","ioi"),("qwen2.5","mcqa"),
    ("gemma2","ioi"),("gemma2","mcqa"),("gemma2","arc_easy"),
    ("llama3","ioi"),("llama3","mcqa"),("llama3","arithmetic_addition"),
    ("llama3","arithmetic_subtraction"),("llama3","arc_easy"),("llama3","arc_challenge"),
]
# tag -> (output_dir, method_name_saveable)
METHODS = {
    "NAP-IG(5)":   ("napig_ref_eval",  "EAP-IG-inputs_patching_node"),
    "local(5)":    ("napig_local_eval","EAP-IG-inputs-local_patching_node"),
    "inputxgrad":  ("ig1_eval",        "EAP-IG-inputs_patching_node"),
}
BASE = "results"

def auc(odir, mdir, task, model):
    stask = task.replace("_", "-")
    p = f"{BASE}/{odir}/{mdir}/{stask}_{model}_validation_abs-False.pkl"
    if not os.path.exists(p):
        return None
    try:
        return pickle.load(open(p, "rb"))["area_under"]
    except Exception:
        return None

hdr = f"{'cell':28}" + "".join(f"{m:>13}" for m in METHODS)
print(hdr); print("-"*len(hdr))
rows = []
for model, task in CELLS:
    cell = f"{task}/{model}"
    vals = {m: auc(od, md, task, model) for m, (od, md) in METHODS.items()}
    rows.append((cell, vals))
    cells_s = f"{cell:28}" + "".join((f"{vals[m]:>13.4f}" if vals[m] is not None else f"{'--':>13}") for m in METHODS)
    print(cells_s)

# means over completed cells
print("-"*len(hdr))
for m in METHODS:
    done = [v[m] for _, v in rows if v[m] is not None]
    mean = sum(done)/len(done) if done else float("nan")
    print(f"  {m:24} mean over {len(done):2}/12 cells = {mean:.4f}")
