#!/bin/bash
#
# Single Seed Version - Fast to queue, runs 1 seed with all deltas
#
# ===== Slurm submission options =====
#SBATCH --job-name=att_seed
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=8            # Small allocation, fast to schedule
#SBATCH --mem=40G                    
#SBATCH --time=02:00:00              # Plenty for 1 seed × 14 deltas
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
# No --exclusive (sharing node is fine)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=yidanxu@umich.edu

# ===== Usage =====
# This script runs ONE seed for a given (dist, d) combination.
# Submit with: sbatch run_att_single_seed.sh <dist> <d> <seed>
# Example: sbatch run_sim_single_seed.sh norm 5 42
#
# Or use submit_all_seeds.sh to submit all 100 seeds at once

# ===== Environment =====
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ===== Load Python / activate env =====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate causal

# ===== Create logs directory =====
mkdir -p logs

# ===== Get parameters from command line =====
DIST=${1}
SEED=${2}

# Validate inputs
if [[ -z "$DIST" ]] || [[ -z "$SEED" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch run_att_single_seed.sh <dist> <seed>"
    echo "Example: sbatch run_att_single_seed.sh lalonde 42"
    exit 1
fi

# ===== Config =====
SCRIPT="/home/yidanxu/causal/src/ATT_main.py"
# Path to precomputed distances
PRECOMPUTED_FILE="/home/yidanxu/causal/src/precomputed/distances_${DIST}.pkl"
# Check if precomputed file exists
if [ ! -f "$PRECOMPUTED_FILE" ]; then
    echo "ERROR: Precomputed distances file not found: $PRECOMPUTED_FILE"
    echo "Please run precomputation first!"
    exit 1
fi

# Other parameters
N=10000
R=0.6
K=5
SEED_START=${SEED}
SEED_END=$((SEED + 1))    # Process only this one seed
SIGMA=0.01
OUTCOME_MODEL_TYPE="tree"
OUTCOME_TYPE="quadratic"
PS_ORDER=1
N_JOBS=-1    # Use 8 of 8 cores

# Overlap parameter grid (all 14 deltas for this seed)
OVERLAP_GRID="0,1,1"

# Create output directory structure
OUTPUT_DIR="/home/yidanxu/causal/src/results/${DIST}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/seeds"

# Output files - one per seed
OUT_CSV="${OUTPUT_DIR}/seeds/${DIST}_${OUTCOME_MODEL_TYPE}_seed${SEED}.csv"
OUT_SUMMARY="${OUTPUT_DIR}/seeds/${DIST}_${OUTCOME_MODEL_TYPE}_seed${SEED}_summary.csv"
DATA_PATH="home/yidanxu/causal/src/data/rhc_processed.csv"
SYNTHETIC_PATH="home/yidanxu/causal/src/data/rhc_nightmare.csv"

# ===== Run =====
echo "========================================"
echo "Single Seed Job"
echo "========================================"
echo "Configuration:"
echo "  dist: $DIST"
echo "  d: $D"
echo "  seed: $SEED"
echo "  Processing 14 delta values"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Node: $(hostname)"
echo "  Started: $(date)"
echo "========================================"

python -u "${SCRIPT}" \
    --dist "${DIST}" \
    --d "${D}" \
    --n "${N}" \
    --r "${R}" \
    --K "${K}" \
    --seed_start "${SEED_START}" \
    --seed_end "${SEED_END}" \
    --sigma "${SIGMA}" \
    --outcome_model_type "${OUTCOME_MODEL_TYPE}" \
    --outcome_type "${OUTCOME_TYPE}" \
    --ps_order "${PS_ORDER}" \
    --n_jobs "${N_JOBS}" \
    --overlap_grid "${OVERLAP_GRID}" \
    --precomputed_distances "${PRECOMPUTED_FILE}" \
    --out_csv "${OUT_CSV}" \
    --out_summary "${OUT_SUMMARY}" \
    --data_path "${DATA_PATH}" \
    --synthetic_path "${SYNTHETIC_PATH}"

EXIT_CODE=$?

echo "========================================"
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results saved to:"
echo "  ${OUT_CSV}"
echo "========================================"

exit $EXIT_CODE