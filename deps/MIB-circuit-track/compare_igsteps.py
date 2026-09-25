"""NAP-IG step-count control: 5 steps (napig_ref) vs 10 steps (napig10), node level, MIB val.

Both circuit sets come from run_variants.sh / run_napig10.sh, which differ ONLY in
--ig-steps, so any gap here is the integration grid and nothing else. 1-step (input x grad)
is the degenerate end of the same family.

Which 5-step dir is the reference depends on the metric, and they are NOT interchangeable:
  * CPR-AUC  -> napig_ref_eval.   Same eval sizing as run_napig10.sh (gemma2/ioi full val,
                gemma2 eval batch 10), so it is harness-identical to the 10-step column.
  * acc-AUC  -> napig_ref_accauc. The only 5-step dir that stores acc_auc, but run_accauc.sh
                re-evaluated with DIFFERENT sizing on two cells (:27-28 caps gemma2/ioi at
                --head 200 and drops gemma2 eval batch to 4). Those cells are marked '~' and
                excluded from the delta mean -- their gap mixes step count with eval sizing.
                Confirmed real, not hypothetical: --check-ref shows ioi/gemma2 1.3854 vs
                1.3778 and mcqa/gemma2 1.2077 vs 1.2067 for the SAME circuits.
"""
import pickle, os, sys

CELLS = [
    ("gpt2","ioi"),("qwen2.5","ioi"),("qwen2.5","mcqa"),
    ("gemma2","ioi"),("gemma2","mcqa"),("gemma2","arc_easy"),
    ("llama3","ioi"),("llama3","mcqa"),("llama3","arithmetic_addition"),
    ("llama3","arithmetic_subtraction"),("llama3","arc_easy"),("llama3","arc_challenge"),
]
MDIR = "EAP-IG-inputs_patching_node"
BASE = "results"
NEW = "napig10_eval"
# metric -> (1-step dir, 5-step dir, cells whose 5-step eval sizing differs from the 10-step run)
REF = {
    "area_under": ("ig1_eval",   "napig_ref_eval",   set()),
    "acc_auc":    ("ig1_accauc", "napig_ref_accauc", {("gemma2","ioi"), ("gemma2","mcqa")}),
}
LABEL = {"area_under": "CPR-AUC", "acc_auc": "acc-AUC"}

def load(odir, task, model):
    p = f"{BASE}/{odir}/{MDIR}/{task.replace('_','-')}_{model}_validation_abs-False.pkl"
    if not os.path.exists(p):
        return None
    try:
        return pickle.load(open(p, "rb"))
    except Exception:
        return None

if "--check-ref" in sys.argv:
    print("napig_ref_eval vs napig_ref_accauc (same circuits, different eval sizing):")
    for model, task in CELLS:
        a, b = load("napig_ref_eval", task, model), load("napig_ref_accauc", task, model)
        if a and b and abs(a["area_under"] - b["area_under"]) > 1e-9:
            print(f"  {task}/{model:<10} eval={a['area_under']:.6f}  accauc={b['area_under']:.6f}"
                  f"  (delta {b['area_under']-a['area_under']:+.6f})")
    print()

W = max(30, max(len(f"{t}/{m}") for m, t in CELLS) + 2)
for metric in ("area_under", "acc_auc"):
    d1, d5, caveat = REF[metric]
    cols = [("IxG(1)", d1), ("NAP-IG(5)", d5), ("NAP-IG(10)", NEW)]
    hdr = f"{LABEL[metric]:<{W}}" + "".join(f"{n:>13}" for n, _ in cols) + f"{'d(10-5)':>11}"
    print(hdr); print("-" * len(hdr))
    got = {n: [] for n, _ in cols}
    deltas, skipped = [], 0
    for model, task in CELLS:
        vals = {}
        for n, od in cols:
            d = load(od, task, model)
            vals[n] = d.get(metric) if d else None
            if vals[n] is not None:
                got[n].append(vals[n])
        mark = "~" if (model, task) in caveat else " "
        line = f"{mark}{task+'/'+model:<{W-1}}"
        line += "".join(f"{vals[n]:>13.4f}" if vals[n] is not None else f"{'--':>13}" for n, _ in cols)
        v5, v10 = vals["NAP-IG(5)"], vals["NAP-IG(10)"]
        if v5 is not None and v10 is not None:
            line += f"{v10 - v5:>+11.4f}"
            if (model, task) in caveat:
                skipped += 1
            else:
                deltas.append(v10 - v5)
        else:
            line += f"{'--':>11}"
        print(line)
    print("-" * len(hdr))
    n_cmp = len(CELLS) - len(caveat)
    foot = f"{'mean':<{W}}"
    foot += "".join(f"{sum(g)/len(g):>13.4f}" if len(g) == len(CELLS) else f"{f'{len(g)}/12':>13}"
                    for n, g in got.items())
    foot += f"{sum(deltas)/len(deltas):>+11.4f}" if len(deltas) == n_cmp else f"{'--':>11}"
    print(foot)
    note = []
    if len(deltas) < n_cmp:
        note.append(f"delta over {len(deltas)}/{n_cmp} comparable cells (10-step still running)")
    if skipped:
        note.append(f"{skipped} '~' cell(s) excluded: 5-step eval sizing differs")
    if note:
        print("  " + "; ".join(note))
    print()
