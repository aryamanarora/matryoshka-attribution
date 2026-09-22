#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=8:00:00
#SBATCH --output=logs_arith/%x_%j.out
#SBATCH --error=logs_arith/%x_%j.err
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
i=0; for m in olmo3-7b gemma2-9b llama3.1-8b; do CUDA_VISIBLE_DEVICES=$i bash scripts/arith_formats/launch/mattr_ablate.sh $m > logs_arith/mattrabl_${m}_$SLURM_JOB_ID.log 2>&1 & i=$((i+1)); done; wait
