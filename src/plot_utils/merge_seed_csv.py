#!/usr/bin/env python
"""
Merge per-seed results (one SLURM array task per seed) into combined CSV files.

Naming convention (shared with src/scripts/run_sim.sh and run_data.sh):
    simulations : results/{dist}_shift/seeds/{d}d_{dist}_{outcome_type}_{model}_seed{s}.csv
                  -> results/{dist}_shift/{d}d_{dist}_{outcome_type}_{model}.csv
    real data   : results/{dist}/seeds/{dist}_{model}_seed{s}.csv
                  -> results/{dist}/{dist}_{model}.csv

Usage (from the repository root):
    python -m src.plot_utils.merge_seed_csv                    # all simulation settings
    python -m src.plot_utils.merge_seed_csv --dist beta --d 5  # one simulation setting
    python -m src.plot_utils.merge_seed_csv --dist rhc         # one real dataset
"""

import re
import pandas as pd
from pathlib import Path
import argparse
import sys

SIM_DISTS = ['norm', 'beta']
DATA_DISTS = ['lalonde', 'rhc', 'meps']
# Outcome models used in the paper (same defaults as src/att_main.py).
DEFAULT_MODEL = {'norm': 'linear', 'beta': 'linear',
                 'lalonde': 'ridge', 'rhc': 'quadratic', 'meps': 'quadratic'}


def result_paths(dist, d, outcome_type, outcome_model_type):
    """Return (base_dir, stem) for a setting."""
    if dist in SIM_DISTS:
        return Path(f"results/{dist}_shift"), f"{d}d_{dist}_{outcome_type}_{outcome_model_type}"
    return Path(f"results/{dist}"), f"{dist}_{outcome_model_type}"


def merge_seeds(dist, d, outcome_type='quadratic', outcome_model_type='linear', verbose=True):
    """Merge all per-seed CSV files of one setting into a combined CSV and a summary CSV."""
    base_dir, stem = result_paths(dist, d, outcome_type, outcome_model_type)
    seeds_dir = base_dir / "seeds"

    if not seeds_dir.exists():
        if verbose:
            print(f"Warning: Directory not found: {seeds_dir}")
        return False

    # Match {stem}_seed{s}.csv only (not the per-seed *_summary.csv files)
    pattern = re.compile(rf"^{re.escape(stem)}_seed(\d+)\.csv$")
    seed_files = sorted(
        (int(m.group(1)), f) for f in seeds_dir.iterdir() if (m := pattern.match(f.name))
    )

    if not seed_files:
        if verbose:
            print(f"Warning: No seed files found matching: {seeds_dir}/{stem}_seed*.csv")
        return False

    if verbose:
        print(f"\nMerging: {base_dir}/{stem}")
        print(f"Found {len(seed_files)} seed files")

    dfs = []
    for seed, seed_file in seed_files:
        try:
            dfs.append(pd.read_csv(seed_file))
        except Exception as e:
            if verbose:
                print(f"  Error reading seed {seed}: {e}")

    if not dfs:
        if verbose:
            print(f"  Error: No valid data files found")
        return False

    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by seed and delta
    if r'$\delta$' in combined_df.columns:
        combined_df = combined_df.sort_values(['seed', r'$\delta$']).reset_index(drop=True)
    elif 'delta' in combined_df.columns:
        combined_df = combined_df.sort_values(['seed', 'delta']).reset_index(drop=True)

    output_csv = base_dir / f"{stem}.csv"
    combined_df.to_csv(output_csv, index=False)

    if verbose:
        print(f"  Combined {len(dfs)} seed files")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Saved to: {output_csv}")

    create_summary(combined_df, base_dir / f"{stem}_summary.csv", verbose)

    return True


def create_summary(df, output_summary, verbose=True):
    """Create summary statistics across seeds."""

    # Determine delta column name
    delta_col = r'$\delta$' if r'$\delta$' in df.columns else 'delta'

    # Group by delta and strategy, compute statistics
    group_cols = [delta_col, 'strategy']

    agg_dict = {
        'mean_eta': ['mean', 'std', 'count'],
        'ATT_estimate': ['mean', 'std', 'count'],
        'bias': ['mean', 'std', 'count'],
    }

    # Add other numeric columns if they exist
    for col in ['variance', 'prediction_MSE', 'max_weight', 'avg_weight']:
        if col in df.columns:
            agg_dict[col] = ['mean', 'std', 'count']

    summary = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns.values]

    summary.to_csv(output_summary, index=False)

    if verbose:
        print(f"  Summary saved to: {output_summary}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge individual seed results into combined CSV files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dist', type=str, default='all',
                        choices=['all'] + SIM_DISTS + DATA_DISTS,
                        help='Setting to merge ("all" = both simulation settings)')
    parser.add_argument('--d', type=int, default=0,
                        choices=[0, 5, 20],
                        help='Dimensionality to merge (simulations only; 0 for all)')
    parser.add_argument('--outcome_type', type=str, default='quadratic',
                        help='Outcome type (simulations only)')
    parser.add_argument('--outcome_model_type', type=str, default=None,
                        help='Outcome model type (default: the one used in the paper)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    dists = SIM_DISTS if args.dist == 'all' else [args.dist]
    verbose = not args.quiet

    if verbose:
        print("="*60)
        print("Merging Seed Results")
        print("="*60)

    jobs = []
    for dist in dists:
        model = args.outcome_model_type or DEFAULT_MODEL[dist]
        d_values = ([5, 20] if args.d == 0 else [args.d]) if dist in SIM_DISTS else [None]
        jobs += [(dist, d, model) for d in d_values]

    success_count = sum(
        merge_seeds(dist, d, args.outcome_type, model, verbose) for dist, d, model in jobs
    )

    if verbose:
        print("\n" + "="*60)
        print(f"Merging Complete: {success_count}/{len(jobs)} successful")
        print("="*60)

    return 0 if success_count == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
