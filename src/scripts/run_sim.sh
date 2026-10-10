#!/bin/bash
#
# One seed of a simulation setting (Normal or Beta covariates), all 14 overlap values s = 0,...,13.
#
# Submit from the repository root, one array task per seed:
#   sbatch --array=0-99 src/scripts/run_sim.sh <norm|beta> <5|20>
# or run a single seed directly:
#   bash src/scripts/run_sim.sh beta 5 0
#
# Requires precomputed/distances_<dist>_d<d>.pkl (see src/scripts/run_dist.sh).
# Output: results/<dist>_shift/seeds/<d>d_<dist>_quadratic_linear_seed<seed>.csv
#
#SBATCH --job-name=ste_sim
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --account=stats_dept1      # edit for your cluster
#SBATCH --partition=standard       # edit for your cluster

set -eo pipefail

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-ste}"
fi

DIST=${1:?usage: run_sim.sh <norm|beta> <5|20> [seed]}
D=${2:?usage: run_sim.sh <norm|beta> <5|20> [seed]}
SEED=${3:-${SLURM_ARRAY_TASK_ID:?pass a seed or submit with --array}}

if [[ ! "$DIST" =~ ^(norm|beta)$ ]]; then echo "dist must be norm or beta"; exit 1; fi
if [[ ! "$D" =~ ^(5|20)$ ]]; then echo "d must be 5 or 20"; exit 1; fi

OUTCOME_TYPE=quadratic
OUTCOME_MODEL_TYPE=linear
STEM="${D}d_${DIST}_${OUTCOME_TYPE}_${OUTCOME_MODEL_TYPE}_seed${SEED}"
OUT_DIR="results/${DIST}_shift/seeds"
mkdir -p "$OUT_DIR" logs

echo "STE simulation: dist=$DIST d=$D seed=$SEED on $(hostname) at $(date)"

python -u -m src.att_main \
    --dist "$DIST" \
    --d "$D" \
    --n 10000 \
    --r 0.6 \
    --K 5 \
    --sigma 0.01 \
    --outcome_type "$OUTCOME_TYPE" \
    --outcome_model_type "$OUTCOME_MODEL_TYPE" \
    --ps_order 1 \
    --overlap_grid "0,13,14" \
    --seed_start "$SEED" \
    --seed_end "$((SEED + 1))" \
    --n_jobs -1 \
    --precomputed_distances "precomputed/distances_${DIST}_d${D}.pkl" \
    --out_csv "${OUT_DIR}/${STEM}.csv" \
    --out_summary "${OUT_DIR}/${STEM}_summary.csv"

echo "Finished at $(date)"
