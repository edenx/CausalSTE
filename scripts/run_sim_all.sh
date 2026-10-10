#!/bin/bash
#
# Helper script to submit jobs for all combinations of dist and d
# Usage: bash submit_all_combinations.sh

echo "Submitting jobs for all combinations of dist and d..."
echo "======================================================"

SCRIPT_PATH="/home/yidanxu/causal/src/run_sim.sh"

# Make sure the script is executable
chmod +x "${SCRIPT_PATH}"

# Submit jobs for all 4 combinations
for dist in norm beta; do
    for d in 5 20; do
        echo "Submitting: dist=${dist}, d=${d}"
        sbatch "${SCRIPT_PATH}" "${dist}" "${d}"
        sleep 1  # Small delay to avoid overwhelming the scheduler
    done
done

echo "======================================================"
echo "All jobs submitted!"
echo ""
echo "Check job status with: squeue -u $USER"
echo "Check logs in: logs/"
echo "Results will be saved in: results/<dist>_shift/"