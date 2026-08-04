"""Submit every acc-AUC backfill that needs only re-evaluation (circuit already trained).

Discovers the gaps rather than hardcoding them: for each (dir, level, split) below it looks for
cells that have a saved circuit but no acc_auc, and submits scripts/reeval_mib_accauc.py.

Jobs are grouped by (level, split, model, task), NOT by dir: loading an 8B HookedTransformer
costs minutes and dominates a single cell's eval, so one job evaluates every dir for that cell
against one loaded model. That turns ~70 cell-jobs into ~20.

A dir/cell with no circuit is reported and skipped -- that is a training gap, which this script
deliberately cannot paper over (see the printed TRAIN lines; scripts/submit_edge_arc_llama3.sh
covers the edge ones).

RUN IT AGAIN after any training wave lands. Each job bakes its --dirs list in at submit time,
so a cell that had no circuit when this ran is a TRAIN line, not a job -- and stays missing
until a second pass discovers it. Wait for the current wave to DRAIN first: gaps are detected
by "no acc_auc on disk", which is still true for cells an already-queued job is about to fill,
so re-running early duplicates work rather than extending it.

  uv run python scripts/submit_accauc_backfill.py            # submit
  DRYRUN=1 uv run python scripts/submit_accauc_backfill.py   # print the plan only
"""
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
import make_mib_table as M  # noqa: E402  COLUMNS

L2A = Path("/home/guests/aryaman/learning-to-attribute")
MIBV = "/home/guests/aryaman/MIB-circuit-track/.venv/bin/python"
R = L2A / "results"
DRYRUN = os.environ.get("DRYRUN", "0") == "1"

# (level, split, [dirs]). Only dirs that a paper artifact actually reads; one-off probe dirs are
# left alone. The lr05 / mattr_accauc dirs are already 11/11 and are not listed.
TARGETS = [
    # node LR sweeps: CPR is complete (they are rows of lr_sweep.tex) but acc-AUC was only ever
    # run on the three cheap cells, which is what keeps them out of the acc-AUC scatter.
    ("node", "validation", ["htk_lr_0.005", "htk_lr_0.1", "htk_lr_0.3",
                            "bern_lr_0.01", "bern_lr_0.05", "bern_lr_0.3", "bern_lr_0.1_2k"]),
    # edge ablations: acc-AUC exists for the 3 headline edge dirs only.
    ("edge", "validation", ["mib_edge_detached_tau", "mib_edge_bernoulli_reinforce",
                            "mib_edge_identity_sgd_log", "mib_edge_identity_sgd_uniform"]),
    # superseded-by-_lr05 test dirs, which have CPR but no acc-AUC at all.
    ("node", "test", ["test_node_hard_topk_uniform"]),
    ("edge", "test", ["test_edge_hard_topk_uniform"]),
]

# from run_accauc_mattr.sh; edge eval walks a much longer sparsity sweep, so it gets more time.
RES = {"llama3": dict(mem="96G", cpus=4, bs=1), "gemma2": dict(mem="64G", cpus=3, bs=4),
       "qwen2.5": dict(mem="32G", cpus=4, bs=10), "gpt2": dict(mem="32G", cpus=4, bs=20)}


def has_acc(d, t, m, split):
    p = R / d / f"{t}_{m}_{split}.pkl"
    if not p.exists():
        return False
    try:
        return pickle.load(open(p, "rb")).get("acc_auc") is not None
    except Exception:
        return False


def has_circuit(d, t, m, level):
    stem = "importances.json" if level == "node" else "scores.pt"
    return (R / d / f"{t}_{m}_{stem}").exists()


def main():
    jobs = defaultdict(list)   # (level, split, model, task) -> [dirs]
    train_gaps = []
    for level, split, dirs in TARGETS:
        for d in dirs:
            if not (R / d).is_dir():
                print(f"  MISSING dir {d}", file=sys.stderr)
                continue
            for t, m, _ in M.COLUMNS:
                if has_acc(d, t, m, split):
                    continue
                if not has_circuit(d, t, m, level):
                    train_gaps.append(f"{level}/{split} {d} {t}/{m}")
                    continue
                jobs[(level, split, m, t)].append(d)

    n = 0
    for (level, split, model, task), dirs in sorted(jobs.items()):
        r = RES[model]
        bs = 1 if (model == "gemma2" and task.startswith(("arc_", "arithmetic_"))) else r["bs"]
        hrs = 24 if model == "llama3" else (12 if level == "edge" else 6)
        name = f"acc-{level[0]}{split[0]}-{task}-{model}"
        cmd = (f"cd {L2A} && PYTHONPATH=MIB-circuit-track:MIB-circuit-track/EAP-IG/src "
               f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {MIBV} "
               f"scripts/reeval_mib_accauc.py --level {level} --split {split} "
               f"--model {model} --task {task} --batch-size {bs} --skip-done "
               f"--dirs {','.join(sorted(set(dirs)))}")
        if DRYRUN:
            print(f"DRY {name}: {len(set(dirs))} dirs -> {sorted(set(dirs))}")
        else:
            (L2A / "logs").mkdir(exist_ok=True)
            subprocess.run(["sbatch", "--partition=main", "--gres=gpu:1",
                            f"--cpus-per-task={r['cpus']}", f"--mem={r['mem']}",
                            f"--time={hrs}:00:00", f"--job-name={name}",
                            f"--output={L2A}/logs/{name}.out", f"--wrap={cmd}"],
                           check=True, stdout=subprocess.DEVNULL)
            print(f"submitted {name} ({len(set(dirs))} dirs)")
        n += 1

    print(f"== {'DRY ' if DRYRUN else ''}total {n} acc-AUC backfill jobs ==")
    if train_gaps:
        print(f"\n== {len(train_gaps)} cells need TRAINING, not re-eval (not submitted here) ==")
        for g in train_gaps:
            print(f"  TRAIN {g}")


if __name__ == "__main__":
    main()
