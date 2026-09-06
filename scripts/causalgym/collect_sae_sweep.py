"""Summarize the per-task node-mode SAE sweep (layer 12): sufficiency + error-node rank."""
import json, glob, os
D = "results/sae_node_sweep"
rows = []
for p in sorted(glob.glob(f"{D}/*/results.json")):
    task = os.path.basename(os.path.dirname(p))
    try:
        d = json.load(open(p))
    except Exception:
        continue
    c = d.get("curve", {})
    ks, accs = c.get("k", []), c.get("learned_acc", [])
    rand = c.get("random_acc", [])
    mink = next((k for k, a in zip(ks, accs) if a >= 0.9), None)
    plateau = max(accs) if accs else float("nan")
    randmax = max(rand) if rand else float("nan")
    en = d.get("error_node") or {}
    rank = en.get("error_node_rank")
    top3 = d.get("top50_features", [])[:3]
    rows.append((task, mink, plateau, randmax, rank, top3))

print(f"{'task':28} {'min-k@.9':>9} {'plateau':>8} {'rand-max':>9} {'err-rank':>9}  top3")
print("-" * 100)
for t, mk, pl, rm, rk, t3 in rows:
    mks = str(mk) if mk is not None else ">4096"
    rks = str(rk) if rk is not None else "-"
    t3s = ",".join("ERR" if x == 16384 else str(x) for x in t3)
    print(f"{t:28} {mks:>9} {pl:>8.3f} {rm:>9.3f} {rks:>9}  {t3s}")
print(f"\n{len(rows)}/29 tasks complete")
