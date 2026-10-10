#!/bin/bash
#
# Precompute distances for a given (dist, d) combination
#
#SBATCH --job-name=precompute
#SBATCH --output=logs/precompute-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1        # Lightweight job
#SBATCH --mem=20G
#SBATCH --time=00:20:00          # 30 minutes should be enough
#SBATCH --account=stats_dept1
#SBATCH --partition=standard

# Usage: sbatch run_dist.sh <dist> <d>
# Example: sbatch runn_dist.sh norm 5

# ===== Environment =====
export PYTHONUNBUFFERED=1

# ===== Load Python / activate env =====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate causal

# ===== Create directories =====
mkdir -p logs
mkdir -p precomputed

# ===== Get parameters =====
SCRIPT="/home/yidanxu/causal/src/dist_main.py"
OUTPUT_DIR="/home/yidanxu/causal/src/precomputed"
DIST=${1}
D=${2}

N=8000
R=0.6        # r=3/5

if [[ -z "$DIST" ]] || [[ -z "$D" ]]; then
    echo "Error: Missing arguments"
    echo "Usage: sbatch precompute_job.sh <dist> <d>"
    exit 1
fi

echo "========================================"
echo "Precomputing Distances"
echo "========================================"
echo "Configuration:"
echo "  dist: $DIST"
echo "  d: $D"
echo "  Started: $(date)"
echo "  Node: $(hostname)"
echo "========================================"

# Run precomputation
python -u "${SCRIPT}" \
    --dist "$DIST" \
    --d "$D" \
    --n "$N" \
    --r "$R" \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo "========================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE