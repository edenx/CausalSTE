#!/bin/bash
#
# Submit 100 seed jobs for each (dist, ) combination
# Total: 200 jobs (100 seeds × 2 combinations)
#
# Usage: bash submit_all_seeds.sh [dist]
#   - No arguments: Submit all 400 jobs (100 seeds × 4 combos)
#   - With dist only: Submit 100 jobs (100 seeds values)

# Get parameters or default to all
DIST_ARG=${1:-"all"}

# Determine what to submit
if [ "$DIST_ARG" = "all" ]; then
    DISTS=("lalonde" "rhc")
else
    DISTS=("$DIST_ARG")
fi

# Calculate total jobs
END_ID=100
START_ID=0
NUM_SEEDS=$(END_ID-START_ID)
TOTAL_JOBS=$((NUM_SEEDS * ${#DISTS[@]}))

echo "======================================================"
echo "Submitting Single Seed Jobs"
echo "======================================================"
echo "Configuration:"
echo "  Script: run_rhc_single_seed.sh"
echo "  Cores per job: 8"
echo "  Time per job: 2 hours"
echo "  Seeds: 0-99 (100 total)"
echo ""
echo "Jobs to submit:"
echo "  Distributions: ${DISTS[@]}"
echo "  Total jobs: $TOTAL_JOBS"
echo "======================================================"
echo ""

# Ask for confirmation if submitting many jobs
if [ $TOTAL_JOBS -gt 50 ]; then
    read -p "Submit $TOTAL_JOBS jobs? This will use significant resources. (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Track job IDs
JOB_COUNT=0
FAILED_COUNT=0

# Submit jobs
for dist in "${DISTS[@]}"; do
    echo ""
    echo "Submitting jobs for dist=$dist"
    echo "----------------------------------------"
    
    for seed in $(seq $START_ID $END_ID); do
        SCRIPT_PATH="/home/yidanxu/causal/src/run_${dist}_single_seed.sh"

        # Check if script exists
        if [ ! -f "$SCRIPT_PATH" ]; then
            echo "Error: Script not found: $SCRIPT_PATH"
            exit 1
        fi

        # Make executable
        chmod +x "${SCRIPT_PATH}"

        JOB_ID=$(sbatch --parsable "${SCRIPT_PATH}" "${dist}" "${seed}" 2>&1)
        
        if [[ $JOB_ID =~ ^[0-9]+$ ]]; then
            # Success
            ((JOB_COUNT++))
            if [ $((seed % 10)) -eq 0 ]; then
                echo "  Submitted seeds 0-$seed... (latest Job ID: $JOB_ID)"
            fi
        else
            # Failed
            ((FAILED_COUNT++))
            echo "  WARNING: Failed to submit seed $seed: $JOB_ID"
        fi
        
        # Small delay every 20 jobs to avoid overwhelming scheduler
        if [ $((seed % 20)) -eq 19 ]; then
            sleep 1
        fi
    done
    
    echo "  Completed: 100 seeds for dist=$dist"
done

echo ""
echo "======================================================"
echo "Submission Complete!"
echo "======================================================"
echo "Successfully submitted: $JOB_COUNT jobs"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "Failed submissions: $FAILED_COUNT jobs"
fi
echo ""
echo "Useful commands:"
echo "  Check status:       squeue -u \$USER"
echo "  Count running:      squeue -u \$USER -t RUNNING | wc -l"
echo "  Count pending:      squeue -u \$USER -t PENDING | wc -l"
echo "  Cancel all:         scancel -u \$USER"
echo "  Monitor progress:   watch -n 10 'squeue -u \$USER | head -20'"
echo ""
echo "Output structure:"
echo "  results/<dist>/seeds/<dist>_<model>_seed<N>.csv"
echo ""
echo "After completion, merge results with:"
echo "  python merge_seed_results.py"
echo "======================================================"