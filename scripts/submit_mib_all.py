"""Generate and submit sbatch jobs for all MIB task/model combos."""

import subprocess
import tempfile
from pathlib import Path

MIB_PATH = "/nlp/scr/aryaman/MIB-circuit-track"
WORK_DIR = "/nlp/scr/aryaman/learning-to-attribute"

COMBOS = [
    ("gpt2", "ioi", "32G"),
    ("qwen2.5", "ioi", "32G"),
    ("qwen2.5", "mcqa", "32G"),
    ("gemma2", "ioi", "64G"),
    ("gemma2", "mcqa", "64G"),
    ("gemma2", "arc_easy", "64G"),
    ("llama3", "ioi", "96G"),
    ("llama3", "mcqa", "96G"),
    ("llama3", "arithmetic_addition", "96G"),
    ("llama3", "arithmetic_subtraction", "96G"),
    ("llama3", "arc_easy", "96G"),
    ("llama3", "arc_challenge", "96G"),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output", type=str, default="results/mib")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for model, task, mem in COMBOS:
        stask = task.replace("_", "-")
        name = f"mib-{stask}-{model}"

        script = f"""#!/bin/bash
#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --job-name={name}
#SBATCH --mem={mem}
#SBATCH --output={name}.out
#SBATCH --partition=jag-standard
#SBATCH --time=14-0

unset SLURM_CPUS_PER_TASK
unset SLURM_TRES_PER_TASK
cd {WORK_DIR}

uv run python scripts/eval_mib.py \
  --mib-path {MIB_PATH} \
  --model {model} --task {task} \
  --steps {args.steps} --split {args.split} \
  --output {args.output}
"""
        if args.dry_run:
            print(f"[DRY RUN] {name}: {model} / {task}")
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            f.flush()
            r = subprocess.run(["sbatch", f.name], capture_output=True, text=True)
            print(f"{name}: {r.stdout.strip()}")


if __name__ == "__main__":
    main()
