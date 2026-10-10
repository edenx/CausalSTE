#!/bin/bash
#
# ===== Slurm submission options =====
#SBATCH --job-name=att_analysis
#SBATCH --output=logs/%x-%j.log   
#SBATCH --nodes=1
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=8
#SBATCH --exclusive            
#SBATCH --mem=80G                    
#SBATCH --time=08:00:00              
#SBATCH --account=stats_dept1
#SBATCH --partition=standard         
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=yidanxu@umich.edu

# ===== Usage =====
# This script runs a single combination of parameters.
# Submit with: sbatch run_att_single.sh <dist> <d>
# Example: sbatch run_att_single.sh norm 5
# Example: sbatch run_att_single.sh beta 20

# ===== Environment (set before importing NumPy in Python) =====
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ===== Load Python / activate env =====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate causal

# ===== Create logs directory if it doesn't exist =====
mkdir -p logs

# ===== Get parameters from command line or use defaults =====
DIST=${1:-"norm"}  # First argument: 'norm' or 'beta'
D=${2:-5}          # Second argument: 5 or 20

# Validate inputs
if [[ ! "$DIST" =~ ^(norm|beta)$ ]]; then
    echo "Error: dist must be 'norm' or 'beta'"
    exit 1
fi

if [[ ! "$D" =~ ^(5|20)$ ]]; then
    echo "Error: d must be 5 or 20"
    exit 1
fi

# ===== Config =====
SCRIPT="/home/yidanxu/causal/src/ATT_main.py"
# Path to precomputed distances
PRECOMPUTED_FILE="/home/yidanxu/causal/src/precomputed/distances_${DIST}_d${D}.pkl"
# Check if precomputed file exists
if [ ! -f "$PRECOMPUTED_FILE" ]; then
    echo "ERROR: Precomputed distances file not found: $PRECOMPUTED_FILE"
    echo "Please run precomputation first!"
    exit 1
fi

# Other parameters (adjust as needed)
N=10000
R=0.6        # r=3/5
K=5
SEED_START=0
SEED_END=1
SIGMA=0.01
OUTCOME_MODEL_TYPE="linear"
OUTCOME_TYPE="quadratic"
PS_ORDER=1
N_JOBS=-1

# Overlap parameter grid (corresponds to delta values)
# This will be passed to your Python script's overlap_grid parameter
OVERLAP_GRID="0,13,14"

# Create output directory structure
OUTPUT_DIR="/home/yidanxu/causal/src/results/${DIST}_shift"
mkdir -p "${OUTPUT_DIR}"

# Output files
OUT_CSV="${OUTPUT_DIR}/${D}d_${DIST}_${OUTCOME_TYPE}_${OUTCOME_MODEL_TYPE}_${SEED_END}.csv"
OUT_SUMMARY="${OUTPUT_DIR}/${D}d_${DIST}_${OUTCOME_TYPE}_${OUTCOME_MODEL_TYPE}_summary_${SEED_END}.csv"


# ===== Run =====
echo "========================================"
echo "Running ATT analysis with:"
echo "  dist: $DIST"
echo "  d: $D"
echo "  n: $N"
echo "  r: $R"
echo "  K: $K"
echo "  seeds: ${SEED_START} to ${SEED_END}"
echo "Starting job on $(hostname) at $(date)"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "========================================"

# Note: Adjust the command line arguments based on how your ATT_main.py
# is set up to receive parameters. This assumes it follows a similar
# pattern to your main block.

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
    --out_summary "${OUT_SUMMARY}"

echo "Finished at $(date)"
echo "Results saved to:"
echo "  CSV: ${OUT_CSV}"
echo "  Summary: ${OUT_SUMMARY}"

