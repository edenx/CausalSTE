#!/bin/bash
#
# One seed (one cross-fitting split) of a real-data study: lalonde, rhc or meps.
# Outcome model and data paths default to the paper's choices in src/att_main.py.
#
# Submit from the repository root, one array task per seed:
#   sbatch --array=0-99 src/scripts/run_data.sh <lalonde|rhc|meps>
# or run a single seed directly:
#   bash src/scripts/run_data.sh rhc 0
#
# Requires precomputed/distances_<dist>.pkl (see src/scripts/run_dist.sh).
# Output: results/<dist>/seeds/<dist>_<model>_seed<seed>.csv
#
#SBATCH --job-name=ste_data
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
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

DIST=${1:?usage: run_data.sh <lalonde|rhc|meps> [seed]}
SEED=${2:-${SLURM_ARRAY_TASK_ID:?pass a seed or submit with --array}}

case "$DIST" in
    lalonde) MODEL=ridge ;;
    rhc|meps) MODEL=quadratic ;;
    *) echo "dist must be lalonde, rhc or meps"; exit 1 ;;
esac

STEM="${DIST}_${MODEL}_seed${SEED}"
OUT_DIR="results/${DIST}/seeds"
mkdir -p "$OUT_DIR" logs

echo "STE real data: dist=$DIST model=$MODEL seed=$SEED on $(hostname) at $(date)"

python -u -m src.att_main \
    --dist "$DIST" \
    --outcome_model_type "$MODEL" \
    --K 5 \
    --ps_order 1 \
    --seed_start "$SEED" \
    --seed_end "$((SEED + 1))" \
    --n_jobs -1 \
    --out_csv "${OUT_DIR}/${STEM}.csv" \
    --out_summary "${OUT_DIR}/${STEM}_summary.csv"

echo "Finished at $(date)"
