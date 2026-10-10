#!/usr/bin/env python
"""
Merge individual seed results into combined CSV files.

After running 100 separate seed jobs, this script combines them into
the final output format expected by your analysis pipeline.

Usage:
    python merge_seed_results.py [--dist DIST] [--d D]
    
Examples:
    python merge_seed_results.py                    # Merge all
    python merge_seed_results.py --dist norm        # Merge all norm results
    python merge_seed_results.py --dist beta --d 5  # Merge beta, d=5 only
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

def merge_seeds(dist, d, outcome_type='quadratic', outcome_model_type='linear', verbose=True):
    """
    Merge individual seed CSV files into combined results.
    
    Parameters:
    -----------
    dist : str
        Distribution type ('norm' or 'beta')
    d : int
        Dimensionality (5 or 20)
    outcome_type : str
        Outcome type (default: 'quadratic')
    outcome_model_type : str
        Outcome model type (default: 'linear')
    verbose : bool
        Print progress messages
    """
    start = 0
    end = 100
    
    # Setup paths
    base_dir = Path(f"/results/{dist}_shift")
    seeds_dir = base_dir / "seeds"
    
    if not seeds_dir.exists():
        if verbose:
            print(f"Warning: Directory not found: {seeds_dir}")
        return False
    
    # Pattern for seed files
    pattern = f"{d}d_{dist}_{outcome_type}_{outcome_model_type}_seed*.csv"
    seed_files = sorted(seeds_dir.glob(pattern))
    
    if not seed_files:
        if verbose:
            print(f"Warning: No seed files found matching: {pattern}")
        return False
    
    if verbose:
        print(f"\nMerging: dist={dist}, d={d}")
        print(f"Found {len(seed_files)} seed files")
    
    # Read and combine all seed files
    dfs = []
    missing_seeds = []
    
    for seed in range(start, end):
        seed_file = seeds_dir / f"{d}d_{dist}_{outcome_type}_{outcome_model_type}_seed{seed}.csv"
        
        if seed_file.exists():
            try:
                df = pd.read_csv(seed_file)
                dfs.append(df)
            except Exception as e:
                if verbose:
                    print(f"  Error reading seed {seed}: {e}")
                missing_seeds.append(seed)
        else:
            missing_seeds.append(seed)
    
    if not dfs:
        if verbose:
            print(f"  Error: No valid data files found")
        return False
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by seed and delta
    if r'$\delta$' in combined_df.columns:
        combined_df = combined_df.sort_values(['seed', r'$\delta$']).reset_index(drop=True)
    elif 'delta' in combined_df.columns:
        combined_df = combined_df.sort_values(['seed', 'delta']).reset_index(drop=True)
    
    # Save combined results
    output_csv = base_dir / f"{d}d_{dist}_{outcome_type}_{outcome_model_type}.csv"
    combined_df.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"  Combined {len(dfs)} seed files")
        print(f"  Total rows: {len(combined_df)}")
        if missing_seeds:
            print(f"  Missing seeds: {missing_seeds[:10]}{'...' if len(missing_seeds) > 10 else ''}")
            print(f"  Total missing: {len(missing_seeds)}")
        print(f"  Saved to: {output_csv}")
    
    # Also create summary statistics
    create_summary(combined_df, base_dir, d, dist, outcome_type, outcome_model_type, verbose)
    
    return True

def create_summary(df, base_dir, d, dist, outcome_type, outcome_model_type, verbose=True):
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
    
    # Save summary
    output_summary = base_dir / f"{d}d_{dist}_{outcome_type}_{outcome_model_type}_summary.csv"
    summary.to_csv(output_summary, index=False)
    
    if verbose:
        print(f"  Summary saved to: {output_summary}")

def main():
    parser = argparse.ArgumentParser(
        description='Merge individual seed results into combined CSV files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dist', type=str, default='all',
                        choices=['all', 'norm', 'beta'],
                        help='Distribution to merge (or "all")')
    parser.add_argument('--d', type=int, default=0,
                        choices=[0, 5, 20],
                        help='Dimensionality to merge (0 for all)')
    parser.add_argument('--outcome_type', type=str, default='quadratic',
                        help='Outcome type')
    parser.add_argument('--outcome_model_type', type=str, default='linear',
                        help='Outcome model type')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Determine what to merge
    if args.dist == 'all':
        dists = ['norm', 'beta']
    else:
        dists = [args.dist]
    
    if args.d == 0:
        d_values = [5, 20]
    else:
        d_values = [args.d]
    
    verbose = not args.quiet
    
    if verbose:
        print("="*60)
        print("Merging Seed Results")
        print("="*60)
    
    # Merge each combination
    success_count = 0
    total_count = len(dists) * len(d_values)
    
    for dist in dists:
        for d in d_values:
            if merge_seeds(dist, d, args.outcome_type, args.outcome_model_type, verbose):
                success_count += 1
    
    if verbose:
        print("\n" + "="*60)
        print(f"Merging Complete: {success_count}/{total_count} successful")
        print("="*60)
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())