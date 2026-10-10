#!/bin/bash
#
# Submit every experiment in the paper to SLURM. Run from the repository root:
#   bash src/scripts/submit_all.sh            # simulations and real data
#   bash src/scripts/submit_all.sh sim        # simulations only
#   bash src/scripts/submit_all.sh data       # real data only
#
# Each setting first precomputes its distances, then runs one array task per seed
# once that job succeeds. Override the seed range with SEEDS, e.g. SEEDS=0-49.
# The real-data inputs must already exist under data/ (see README).

set -eo pipefail

WHAT=${1:-all}
SEEDS=${SEEDS:-0-99}
mkdir -p logs

submit() {  # submit <dist_args...> -- <run_script> <run_args...>
    local dist_args=()
    while [[ "$1" != "--" ]]; do dist_args+=("$1"); shift; done; shift
    local dep
    dep=$(sbatch --parsable src/scripts/run_dist.sh "${dist_args[@]}")
    sbatch --dependency=afterok:"$dep" --array="$SEEDS" "$@"
    echo "  submitted ${dist_args[*]} (distances job $dep, seeds $SEEDS)"
}

if [[ "$WHAT" == all || "$WHAT" == sim ]]; then
    for dist in norm beta; do
        for d in 5 20; do
            submit "$dist" "$d" -- src/scripts/run_sim.sh "$dist" "$d"
        done
    done
fi

if [[ "$WHAT" == all || "$WHAT" == data ]]; then
    for dist in lalonde rhc meps; do
        submit "$dist" -- src/scripts/run_data.sh "$dist"
    done
fi

echo "Check status with: squeue -u $USER   (logs in logs/)"
