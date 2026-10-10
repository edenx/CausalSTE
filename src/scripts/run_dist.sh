#!/bin/bash
#
# Precompute the distances used for the data-driven choice of eta.
#
# Submit from the repository root:
#   sbatch src/scripts/run_dist.sh <norm|beta> <5|20>
#   sbatch src/scripts/run_dist.sh <lalonde|rhc|meps>
# or run directly:
#   bash src/scripts/run_dist.sh beta 5
#
# Output: precomputed/distances_<dist>_d<d>.pkl (simulations) or precomputed/distances_<dist>.pkl
#
#SBATCH --job-name=ste_dist
#SBATCH --output=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --account=stats_dept1      # edit for your cluster
#SBATCH --partition=standard       # edit for your cluster

set -eo pipefail
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-ste}"
fi

DIST=${1:?usage: run_dist.sh <dist> [d]}
D=${2:-5}
mkdir -p precomputed logs

echo "Precomputing distances: dist=$DIST d=$D on $(hostname) at $(date)"

python -u -m src.dist_main \
    --dist "$DIST" \
    --d "$D" \
    --n 8000 \
    --r 0.6 \
    --output_dir precomputed

echo "Finished at $(date)"
