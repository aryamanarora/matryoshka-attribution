"""Print headline results: multitask DAS (full vs lean) + hybrid CPR table. Run on sc."""
import pickle
from pathlib import Path
import numpy as np

R = Path("results")


def load(p):
    return pickle.load(open(p, "rb")) if Path(p).exists() else None


def mt_summary(tag, path):
    d = load(path)
    if d is None:
        print(f"[{tag}] MISSING {path}"); return
    ll = d.get("loss_log", [])
    final = ll[-1][1] if ll else float("nan")
    pte = d.get("per_task_eval", {})
    # CE@5% per task (index 4 in sparsities list)
    ces = {t: ev["eval_learned_ce"][4] for t, ev in pte.items() if ev.get("eval_learned_ce")}
    arr = np.array(list(ces.values())) if ces else np.array([np.nan])
    print(f"[{tag}] final_loss={final:.3f}  tasks={len(pte)}  "
          f"CE@5%: mean={arr.mean():.3f} min={arr.min():.3f} max={arr.max():.3f}")
    return d, ces


def main():
    print("=" * 70)
    full = mt_summary("FULL 3000", R / "pythia1b_multitask_das.pkl")
    lean = mt_summary("LEAN 1500", R / "pythia1b_multitask_das_lean.pkl")

    # lean vs full per-task CE@5% delta
    if full and lean:
        df, cf = full; dl, cl = lean
        common = sorted(set(cf) & set(cl))
        deltas = [(t, cl[t] - cf[t]) for t in common]
        worse = sorted(deltas, key=lambda x: -x[1])[:5]
        print("\nLEAN worse than FULL (CE@5% lean-full, top 5):")
        for t, dd in worse:
            print(f"   {t:42s} +{dd:.3f}")
        print(f"mean lean-full CE@5% delta = {np.mean([d for _,d in deltas]):+.3f} "
              f"(>0 => lean worse)")

        # overlap matrix headline (from full run)
        M = df.get("overlap_matrix"); tasks = df.get("overlap_tasks")
        if M is not None:
            M = np.array(M); n = len(tasks)
            pairs = [(tasks[i], tasks[j], M[i, j]) for i in range(n) for j in range(i+1, n)]
            top = sorted(pairs, key=lambda x: -x[2])[:8]
            print("\nTop cross-task feature overlap (Jaccard, full run):")
            for a, b, v in top:
                print(f"   {v:.2f}  {a.split('/')[-1]:28s} <-> {b.split('/')[-1]}")

    # Hybrid CPR table
    print("\n" + "=" * 70 + "\nHYBRID CPR AUC (validation):")
    COLS = [("ioi","gpt2"),("ioi","qwen2.5"),("ioi","gemma2"),("ioi","llama3"),
            ("arithmetic_subtraction","llama3"),("mcqa","qwen2.5"),("mcqa","gemma2"),
            ("mcqa","llama3"),("arc_easy","gemma2"),("arc_easy","llama3"),
            ("arc_challenge","llama3")]
    H = R / "hybrid_eval"
    print(f"{'task/model':28s} {'napig+MAttrMLP':>15s} {'MAttr+napigMLP':>15s}")
    for task, model in COLS:
        a = load(H / f"{task}_{model}_napig_ours_mlp_validation.pkl")
        b = load(H / f"{task}_{model}_ours_napig_mlp_validation.pkl")
        av = f"{a['area_under']:.2f}" if a else "--"
        bv = f"{b['area_under']:.2f}" if b else "--"
        print(f"{task+'/'+model:28s} {av:>15s} {bv:>15s}")


if __name__ == "__main__":
    main()
