import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
import pickle
import logging
import sys
from pathlib import Path
from datetime import datetime
import traceback
from tqdm import tqdm
import warnings
import time

from .models.att_est import *
from .models.att_other import *
from .models.crossfit import *
from .data_utils.gen_data import *
from .data_utils.gen_lalonde import *
from .data_utils.gen_rhc import *
from .data_utils.gen_meps import load_meps_data

from .models.estimator import MultidimRatioEstimator
import argparse

def setup_logger(seed, overlap_param, log_dir="logs"):
    """
    Set up a logger for each worker process.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"worker_seed{seed}_delta{overlap_param:.2f}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler
    log_file = Path(log_dir) / f"seed{seed}_delta{overlap_param:.2f}.log"
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Console handler (will be captured by main process)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger





def process_single_run(
    seed, overlap_param, r, n, K, d=1, 
    log_dir="logs", dist="norm",
    outcome_model_type='linear',
    outcome_type='quadratic',
    sigma=0.01,
    max_avg_weight=1.1,
    ps_order=1,
    distances=None,
    data_path="data/rhc_processed.csv",
    synthetic_path="data/rhc_nightmare.csv",
    **kwargs
    ):
    """
    Process a single combination of seed and overlap parameter.
    This function will be called in parallel.
    """
    logger = setup_logger(seed, overlap_param, log_dir)
    rng = np.random.default_rng(seed)
    try:
        logger.info(f"Starting job: seed={seed}, delta={overlap_param:.4f}")
        
        eta_strategies = ['none', 'global(bias)', 'ratio', 'ipm(w1)', 'ipm(w2)']
        # eta_strategies = ['oracle', 'ratio', 'ipm(w1)']
        strategies = ['ipw', 'sipw', 'aipw', 'or', 'did']
        if dist == "rhc":
            # print("debug-----", data_path)
            # print("debug-----", synthetic_path)
            data = load_rhc_data(
                path=data_path,
                standardize=True,
                semisynthetic=True,
                semisynthetic_path=synthetic_path
            )
            # print("data", data)
            outcome_type = data['outcome_info']['type']
        elif dist == "lalonde":
            data = load_lalonde_data(
                path=data_path,
                version='psid', 
                standardize=True
            )
            outcome_type = data['outcome_info']['type']
        elif dist == "meps":
            data = load_meps_data(
                processed_path=data_path,
                semisynthetic=True,
                semisynthetic_path=synthetic_path
            )
            outcome_type = data['outcome_info']['type']
        else:
            config = {
                'rng': rng,
                'n0': int(n*(1-r)),
                'n1': int(n*r),
                'd': d,
                'outcome_type': outcome_type,
                # 'quadratic_strength': .5,
                # 'treatment_effect': 2.0,
                'overlap_param': overlap_param,
                'sigma': sigma
            }
            outcome_type = config['outcome_type']
        
        if outcome_type == "binary":
            binary_outcome = True
        else:
            binary_outcome = False

        if outcome_type == "positive":
            use_log_link = True
        else:
            use_log_link = False
        
        # Generate data
        # data = generate_data(**config)
        if dist == "norm":
            data = gen_norm_data(**config)
        elif dist == "beta":
            data = gen_beta_data(**config)
        elif dist in ['lalonde', 'rhc', 'meps']:
            # bootstrap
            data = bootstrap(
                rng, data, bootstrap_type='standard', stratified=True)

        logger.info(f"Generating data with n0={data['n0']}, n1={data['n1']}")
        logger.debug(f"Data generated successfully")
        
        # Create crossfit data
        crossfit_data = CrossFitData(
            data['X_control'], data['X_treatment'], 
            data['Y_control'], data['Y_treatment'], 
            data['Y0_treatment'], data['true_ATT'], 
            K=K, random_state=seed,
            binary_outcome=binary_outcome,
            # use_log_link=use_log_link
            )
        logger.debug(f"CrossFit data created with K={K} folds")
        
        # Fit models
        logger.info("Fitting transport maps...")
        crossfit_data.fit_transport_maps(
            verbose=False, method='exact', oos_method='auto')
        
        logger.info("Fitting outcome models...")
        crossfit_data.fit_outcome_models(
            outcome_model_type=outcome_model_type, **kwargs)
        
        estimator = ATTEstimatorWithOT(seed=seed)
        
        # Get results for each strategy
        results_dict = {}
        if distances is None:
            max_avg_weight_ipm_w1 = max_avg_weight
            max_avg_weight_ipm_w2 = max_avg_weight
            max_avg_weight_ratio = max_avg_weight
        else:
            max_avg_weight_ipm_w1 = distances['w1']
            max_avg_weight_ipm_w2 = distances['w2']
            max_avg_weight_ratio = distances['chisq']
        # print("max_avg_weight_ipm_w1", max_avg_weight_ipm_w1)
        # print("max_avg_weight_ipm_w2", max_avg_weight_ipm_w2)
        # print("max_avg_weight_ratio", max_avg_weight_ratio)

        for strategy in eta_strategies:
            logger.info(f"Processing strategy: {strategy}")
            
            try:
                if strategy == 'none':
                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy=None, verbose=False
                    )
                elif strategy == 'individual':
                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy='optimal',
                        individual_etas=None, verbose=False
                    )
                elif strategy == 'oracle':
                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy='oracle',
                        verbose=False
                    )
                elif strategy == 'global(bias)':
                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy='global', metric='bias',
                        global_eta=None, verbose=False
                    )
                elif strategy == 'global(mse)':
                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy='global', metric='mse',
                        global_eta=None, verbose=False
                    )
                # elif strategy == "ratio":
                #     res = estimator.estimate_ATT_with_eta_crossfit(
                #         crossfit_data, eta_strategy='ratio',
                #         global_eta=None, verbose=False, 
                #         max_avg_weight = max_avg_weight,
                #         gaussian=True if dist == "norm" else False
                #         )
                elif strategy == "ratio":
                    # if dist in ['norm', 'beta']:
                    #     if d == 20:
                    #         max_avg_weight_ratio = 3.5 
                    #     elif d == 5:
                    #         max_avg_weight_ratio = 1.1 
                    print("max_avg_weight_ratio", max_avg_weight_ratio)
                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy='ratio',
                        global_eta=None, verbose=False, 
                        max_avg_weight = max_avg_weight_ratio,
                        gaussian=True if dist == "norm" else False
                    )
                    # if overlap_param == 0 and dist != "norm":
                    #     res["mean_eta"] = 0.25
                    #     max_avg_weight_ratio = res.get('dist_01')
                    # print("max_avg_weight_ratio", max_avg_weight_ratio)

                elif strategy == "ipm(w2)":
                    # if dist in ['norm', 'beta']:
                    #     if d == 20:
                    #         max_avg_weight_ipm_w2 = 20 
                    #     elif d == 5:
                    #         max_avg_weight_ipm_w2 = 2.6
                    
                    # if dist == "lalonde":
                    #     max_avg_weight_ipm_w2 = 1.3

                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy='ipm',
                        global_eta=None, verbose=False, 
                        max_avg_weight = max_avg_weight_ipm_w2,
                        metric='w2',
                        gaussian=True if dist == "norm" else False
                        )
                    # if overlap_param == 0:
                    #     res["mean_eta"] = 0.25
                    #     max_avg_weight_ipm_w2 = res.get('dist_01')

                elif strategy == "ipm(w1)":
                    # if dist in ['norm', 'beta']:
                    #     if d == 20:
                    #         max_avg_weight_ipm_w1 = 11.4
                    #     elif d == 5:
                    #         max_avg_weight_ipm_w1 = 2.2
                    
                    # if dist == "lalonde":
                    #     max_avg_weight_ipm_w1 = 1.

                    res = estimator.estimate_ATT_with_eta_crossfit(
                        crossfit_data, eta_strategy='ipm',
                        global_eta=None, verbose=False, 
                        max_avg_weight = max_avg_weight_ipm_w1,
                        metric='w1',
                        gaussian=True if dist == "norm" else False
                        )
                    # if overlap_param == 0:
                    #     res["mean_eta"] = 0.25
                    #     max_avg_weight_ipm_w1 = res.get('dist_01')
                    # print("max_avg_weight_ipm", max_avg_weight_ipm)

                logger.info(f"  -> eta={res.get('mean_eta', 'N/A')}, "
                          f"bias={res.get('bias', 'N/A')}, "
                          f"MSE={res.get('prediction_MSE', 'N/A')}")
                
                results_dict[strategy] = res
                
            except Exception as e:
                logger.error(f"Error in strategy {strategy}: {str(e)}")
                logger.debug(traceback.format_exc())
                # Store error info in results
                results_dict[strategy] = {
                    'error': str(e),
                    'mean_eta': np.nan,
                    'bias': np.nan,
                }
        for strategy in strategies:
            try:
                res = estimate_ATT_crossfit(
                    crossfit_data, method=strategy, ps_order=ps_order, verbose=False)
                res["mean_eta"] = None
                res['etas'] = None
                res['n_mapped'] = None

                results_dict[strategy] = res

                logger.info(f"  -> bias={res.get('bias', 'N/A'):.4f}, "
                          f"MSE={res.get('prediction_MSE', 'N/A'):.4f}")

            except Exception as e:
                logger.error(f"Error in strategy {strategy}: {str(e)}")
                logger.debug(traceback.format_exc())
                # Store error info in results
                results_dict[strategy] = {
                    'error': str(e),
                    'mean_eta': np.nan,
                    'bias': np.nan,
                }

        logger.info(f"Completed successfully: seed={seed}, delta={overlap_param:.4f}")
        return (seed, overlap_param, results_dict, None)
        
    except Exception as e:
        error_msg = f"FAILED: seed={seed}, delta={overlap_param:.4f}\n{str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return (seed, overlap_param, None, error_msg)


def main_parallel(
    seeds=range(42, 52), r=3/5, n=5000, K=5, d=1,
    overlap_grid=np.linspace(0, -8, 10), n_jobs=-1, 
    sigma=0.01,
    max_avg_weight=1.1,
    ps_order=1,
    log_dir="logs", 
    dist="norm",
    outcome_model_type='linear',
    outcome_type='quadratic',
    precomputed_distances="precomputed/distances_norm_d5.pkl",
    path="results/normal_shift/results.csv",
    path_summary="results/normal_shift/summary.csv",
    data_path="data/rhc_processed.csv",
    synthetic_path="data/rhc_nightmare.csv",
    save_intermediate=True,
    **kwargs):
    """
    Run main function in parallel over both seeds and overlap parameters.
    
    Parameters:
    -----------
    seeds : list or range
        List of seeds to run (for repetition)
    r, n, K : as before
    overlap_grid : array-like
        Grid of overlap parameters
    n_jobs : int
        Number of parallel jobs. -1 means use all processors
    log_dir : str
        Directory to save log files
    save_intermediate : bool
        Whether to save intermediate results as they complete
    """
    with open(precomputed_distances, 'rb') as f:
        distances = pickle.load(f)
        print("precomputed distances", distances)
    # Setup main logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.handlers.clear()
    
    main_log_file = Path(log_dir) / f"main_{timestamp}.log"
    fh = logging.FileHandler(main_log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    main_logger.addHandler(fh)
    main_logger.addHandler(ch)
    
    # Create all combinations of (seed, overlap_param)
    param_combinations = list(product(seeds, overlap_grid))
    total_jobs = len(param_combinations)
    
    main_logger.info("="*80)
    main_logger.info(f"Starting parallel execution")
    main_logger.info(f"Total jobs: {total_jobs} ({len(seeds)} seeds × {len(overlap_grid)} overlap values)")
    main_logger.info(f"Number of workers: {n_jobs if n_jobs > 0 else 'all available cores'}")
    main_logger.info(f"Log directory: {log_dir}")
    main_logger.info("="*80)
    
    # Run in parallel with progress bar
    results = []
    failed_jobs = []

    try:
        with tqdm(total=total_jobs, desc="Processing jobs", 
                  unit="job", ncols=100, file=sys.stdout) as pbar:
            
            for result in Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(process_single_run)(
                    seed, overlap_param, r, n, K, d, log_dir,
                    dist, outcome_model_type, outcome_type, 
                    sigma, max_avg_weight, ps_order, distances,
                    data_path, synthetic_path, **kwargs)

                for seed, overlap_param in param_combinations
            ):
                seed, overlap_param, results_dict, error = result
                
                if error is not None:
                    failed_jobs.append({
                        'seed': seed,
                        'overlap_param': overlap_param,
                        'error': error
                    })
                    pbar.set_postfix_str(f"✗ FAILED: seed={seed}, δ={overlap_param:.2f}")
                    main_logger.error(f"Job failed: seed={seed}, delta={overlap_param:.4f}")
                else:
                    results.append(result)
                    pbar.set_postfix_str(f"✓ seed={seed}, δ={overlap_param:.2f}")
                
                pbar.update(1)
        
        main_logger.info(f"\nCompleted: {len(results)}/{total_jobs} jobs successful")
        
        if failed_jobs:
            main_logger.warning(f"Failed jobs: {len(failed_jobs)}")
            for job in failed_jobs:
                main_logger.warning(f"  - seed={job['seed']}, delta={job['overlap_param']:.4f}")
            
            # Save failed jobs info
            failed_df = pd.DataFrame(failed_jobs)
            failed_df.to_csv(Path(log_dir) / f"failed_jobs_{timestamp}.csv", index=False)
        
    except KeyboardInterrupt:
        main_logger.warning("\n\nInterrupted by user! Saving partial results...")
    except Exception as e:
        main_logger.error(f"Unexpected error in main parallel execution: {str(e)}")
        main_logger.debug(traceback.format_exc())
    
    # Process results into dataframe
    if not results:
        main_logger.error("No successful results to process!")
        return None, None
    
    main_logger.info("\nProcessing results...")
    fields = ["mean_eta", "bias", "ATT_estimate", "true_ATT"]
    flat_records = []
    
    for seed, overlap_param, results_dict, _ in results:
        for strategy, res in results_dict.items():
            record = {
                "seed": seed,
                r"$\delta$": overlap_param,
                "strategy": strategy
            }
            record.update({f: res.get(f, np.nan) for f in fields})
            
            # Mark if this strategy had an error
            if 'error' in res:
                record['has_error'] = True
                record['error_msg'] = res.get('error', '')
            else:
                record['has_error'] = False
                record['error_msg'] = ''
            
            flat_records.append(record)
    
    df_long = pd.DataFrame(flat_records)
    
    # Save combined results
    output_file = path # f"results/normal_shift/{d}d_example_crossfit_all_seeds_{timestamp}.csv"
    df_long.to_csv(output_file, index=False)
    main_logger.info(f"Saved detailed results to: {output_file}")
    
    # Compute summary statistics across seeds (excluding errors)
    df_clean = df_long[~df_long['has_error']]
    
    if len(df_clean) > 0:
        summary_df = df_clean.groupby([r"$\delta$", "strategy"]).agg({
            'mean_eta': ['mean', 'std', 'count'],
            'bias': ['mean', 'std'],
            'ATT_estimate': ['mean', 'std'],
            'true_ATT': ['mean', 'std']
            # 'variance': ['mean', 'std'],
            # 'prediction_MSE': ['mean', 'std']
        }).reset_index()
        
        summary_file = path_summary # f"results/normal_shift/{d}d_example_crossfit_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        main_logger.info(f"Saved summary statistics to: {summary_file}")
        
        main_logger.info("\n" + "="*80)
        main_logger.info("Summary Statistics:")
        main_logger.info("="*80)
        print(summary_df.to_string())
    else:
        main_logger.warning("No clean data available for summary statistics")
        summary_df = None
    
    # Report on errors
    error_count = df_long['has_error'].sum()
    if error_count > 0:
        main_logger.warning(f"\nTotal strategy-level errors: {error_count}")
        error_summary = df_long[df_long['has_error']].groupby('strategy').size()
        main_logger.warning("Errors by strategy:")
        for strategy, count in error_summary.items():
            main_logger.warning(f"  {strategy}: {count} errors")
    
    main_logger.info("\n" + "="*80)
    main_logger.info(f"All done! Logs saved to: {log_dir}")
    main_logger.info("="*80)
    
    return df_long, summary_df


def plot_with_error_bars(df_long, error_type='std', figsize=(20, 6), 
                         save_path="figs/normal_shift/d1_example_parallel.png"):
    """
    Plot results from parallel runs with error bars showing variation across seeds.
    
    Parameters:
    -----------
    df_long : pd.DataFrame
        Long-format dataframe with columns: seed, $\delta$, strategy, mean_eta, 
        bias, variance, prediction_MSE, has_error
    error_type : str
        'std' for standard deviation, 'sem' for standard error of mean, 
        'ci95' for 95% confidence interval
    figsize : tuple
        Figure size
    save_path : str
        Path to save the figure
    """
    
    # Filter out errors
    df_clean = df_long[~df_long['has_error']].copy()
    
    # Compute summary statistics across seeds
    agg_dict = {
        'mean_eta': ['mean', 'std', 'count'],
        'ATT_estimate': ['mean', 'std', 'count'],
        'bias': ['mean', 'std', 'count'],
    }
    
    summary = df_clean.groupby([r'$\delta$', 'strategy']).agg(agg_dict).reset_index()
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in summary.columns.values]
    
    # Calculate error bars based on error_type
    for metric in ['mean_eta', 'ATT_estimate', 'bias']:
        if error_type == 'std':
            summary[f'{metric}_error'] = summary[f'{metric}_std']
        elif error_type == 'sem':
            summary[f'{metric}_error'] = summary[f'{metric}_std'] / np.sqrt(summary[f'{metric}_count'])
        elif error_type == 'ci95':
            # 95% CI: mean ± 1.96 * SEM
            summary[f'{metric}_error'] = 1.96 * summary[f'{metric}_std'] / np.sqrt(summary[f'{metric}_count'])
    
    # Create plot
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    markers = ['o', '*', '>', 'X', '^', 'P', 's', 'd', 'p', 'v', 'D'] # ['o', '*', '>', 'x', '^']
    # colors = plt.cm.tab20(np.linspace(0, 0.9, 12))
    
    # Get unique strategies
    strategies = summary['strategy'].unique()
    
    # Plot 1: Mean eta
    for i, strategy in enumerate(strategies):
        # Clean up label
        if strategy in ["oracle"]:
            lab = strategy + "(1st order)"
        else:
            lab = strategy
            
        subset = summary[summary['strategy'] == strategy].sort_values(r'$\delta$')
        
        ax[0].errorbar(
            np.abs(subset[r'$\delta$']), 
            np.abs(subset['mean_eta_mean']),
            yerr=subset['mean_eta_error'],
            marker=markers[i], 
            label=lab,
            # color=colors[i],
            capsize=3,
            capthick=1,
            alpha=0.8
        )
    
    ax[0].set_xlabel(r"|$\delta$|", fontsize=12)
    ax[0].set_ylabel(r"$\bar{\eta}$", fontsize=12)
    # ax[0].legend(title="Strategy", loc='upper left', fontsize=10)
    ax[0].grid(True, alpha=0.3)
    
    # Plot 2: Bias
    for i, strategy in enumerate(strategies):
        if strategy in ["oracle"]:
            lab = strategy + "(1st order)"
        else:
            lab = strategy
            
        subset = summary[summary['strategy'] == strategy].sort_values(r'$\delta$')
        
        ax[1].errorbar(
            np.abs(subset[r'$\delta$']), 
            np.abs(subset['bias_mean']),
            yerr=subset['bias_error'],
            marker=markers[i], 
            label=lab,
            # color=colors[i],
            capsize=3,
            capthick=1,
            alpha=0.8
        )
    
    ax[1].set_xlabel(r"|$\delta$|", fontsize=12)
    ax[1].set_ylabel("|Bias|", fontsize=12)
    # ax[1].set_yscale('log')
    # ax[1].legend(title="Strategy", loc='upper left', fontsize=10)
    ax[1].grid(True, alpha=0.3, which='both')
    
    # Plot 3: Variance
    for i, strategy in enumerate(strategies):
        if strategy in ["oracle"]:
            lab = strategy + "(1st order)"
        else:
            lab = strategy
            
        subset = summary[summary['strategy'] == strategy].sort_values(r'$\delta$')
        
        ax[2].errorbar(
            np.abs(subset[r'$\delta$']), 
            subset['ATT_estimate_mean'],
            yerr=subset['ATT_estimate_error'],
            marker=markers[i], 
            label=lab,
            # color=colors[i],
            capsize=3,
            capthick=1,
            alpha=0.8
        )
    
    ax[2].set_xlabel(r"|$\delta$|", fontsize=12)
    ax[2].set_ylabel("Estimator Value", fontsize=12)
    # ax[2].set_yscale('log')
    # ax[2].legend(title="Strategy", loc='upper left', fontsize=10)
    ax[2].grid(True, alpha=0.3, which='both')
    
    # # Plot 4: MSE
    # for i, strategy in enumerate(strategies):
    #     if strategy in ["oracle"]:
    #         lab = strategy + "(1st order)"
    #     else:
    #         lab = strategy
            
    #     subset = summary[summary['strategy'] == strategy].sort_values(r'$\delta$')
        
    #     ax[3].errorbar(
    #         np.abs(subset[r'$\delta$']), 
    #         subset['prediction_MSE_mean'],
    #         yerr=subset['prediction_MSE_error'],
    #         marker=markers[i], 
    #         label=lab,
    #         color=colors[i],
    #         capsize=3,
    #         capthick=1,
    #         alpha=0.8
    #     )
    
    # ax[3].set_xlabel(r"|$\delta$|", fontsize=12)
    # ax[3].set_ylabel("MSE", fontsize=12)
    # ax[3].set_yscale('log')
    # ax[3].legend(title="Strategy", loc='upper left', fontsize=10)
    # ax[3].grid(True, alpha=0.3, which='both')
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=len(labels), frameon=True, fontsize=10)

    # Add title with info about error bars
    error_label = {
        'std': 'Standard Deviation',
        'sem': 'Standard Error',
        'ci95': '95% Confidence Interval'
    }
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.show()
    plt.close()
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run ATT estimation with cross-fitting across different overlap parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation parameters
    parser.add_argument('--dist', type=str, default='norm', 
                        choices=['norm', 'beta', 'lalonde', 'rhc', 'meps'],
                        help='Distribution type for generating data')
    parser.add_argument('--d', type=int, default=5, choices=[5,20],
                        help='Dimensionality of covariates')
    parser.add_argument('--n', type=int, default=10000,
                        help='Total sample size')
    parser.add_argument('--r', type=float, default=0.6,
                        help='Proportion of treated units (n1 = n*r)')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='Noise level in outcome generation')
    
    # Model parameters
    parser.add_argument('--outcome_model_type', type=str, default=None,
                        help='Type of outcome model (default: linear for norm/beta, '
                             'ridge for lalonde, quadratic for rhc/meps)')
    parser.add_argument('--outcome_type', type=str, default='quadratic',
                        help='Type of outcome function')
    parser.add_argument('--ps_order', type=int, default=1,
                        help='Order of propensity score model')
    
    # Cross-fitting parameters
    parser.add_argument('--K', type=int, default=5,
                        help='Number of folds for cross-fitting')
    
    # Simulation parameters
    parser.add_argument('--seed_start', type=int, default=0,
                        help='Starting seed value')
    parser.add_argument('--seed_end', type=int, default=1,
                        help='Ending seed value (exclusive)')
    
    # Overlap parameter grid
    parser.add_argument('--overlap_grid', type=str, default=None,
                        help='Overlap parameter grid. Format: "start,stop,num" for linspace, '
                             'or "val1,val2,val3,..." for explicit values '
                             '(default: 0,13,14 for norm/beta, 0 for real data)')
    parser.add_argument('--degree', type=int, default=3,
                        help='degrees of polynomial outcome regression model.')
    # Computational parameters
    parser.add_argument('--n_jobs', type=int, default=-2,
                        help='Number of parallel jobs. -1 uses all cores, -2 leaves one free')
    parser.add_argument('--max_avg_weight', type=float, default=float('inf'),
                        help='Maximum average weight for weight trimming')
    
    # Output parameters
    parser.add_argument('--precomputed_distances', type=str, 
                        help='Path to saved results to precomputed distances')
    parser.add_argument('--out_csv', type=str, 
                        help='Path to save results CSV file')
    parser.add_argument('--out_summary', type=str, 
                        help='Path to save summary CSV file')
    parser.add_argument('--data_path', type=str, 
                        help='Path to data file')
    parser.add_argument('--synthetic_path', type=str, 
                        help='Path to semi-synthetic data file')
    parser.add_argument('--log_dir', type=str, default='logs/parallel_run',
                        help='Directory for log files')
    
    # Optional flags
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate results during computation')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    

    # Per-dataset defaults used in the paper; any option passed explicitly wins.
    # All paths are relative to the repository root.
    DATASET_DEFAULTS = {
        'rhc': dict(outcome_model_type='quadratic', overlap_grid='0',
                    data_path='data/rhc_processed.csv',
                    synthetic_path='data/rhc_nightmare.csv'),
        'lalonde': dict(outcome_model_type='ridge', overlap_grid='0',
                        data_path='data/lalonde_psid.csv'),
        'meps': dict(outcome_model_type='quadratic', overlap_grid='0',
                     data_path='data/meps_data/processed/meps_medicare_att.csv',
                     synthetic_path='data/meps_data/processed/meps_medicare_att_semisynthetic_severe.csv'),
    }
    SIM_DEFAULTS = dict(outcome_model_type='linear', overlap_grid='0,13,14')
    for key, val in DATASET_DEFAULTS.get(args.dist, SIM_DEFAULTS).items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.dist in ['norm', 'beta']:
        stem = f"results/{args.dist}_shift/{args.d}d_{args.dist}_{args.outcome_type}_{args.outcome_model_type}"
        default_dist_file = f"precomputed/distances_{args.dist}_d{args.d}.pkl"
    else:
        stem = f"results/{args.dist}/{args.dist}_{args.outcome_model_type}"
        default_dist_file = f"precomputed/distances_{args.dist}.pkl"
    if args.out_csv is None:
        args.out_csv = f"{stem}.csv"
    if args.out_summary is None:
        args.out_summary = f"{stem}_summary.csv"
    if args.precomputed_distances is None:
        args.precomputed_distances = default_dist_file

    # Parse overlap_grid
    overlap_grid_str = args.overlap_grid.strip()
    overlap_parts = overlap_grid_str.split(',')
    
    if len(overlap_parts) == 3:
        # Format: "start,stop,num" -> use linspace
        try:
            start = float(overlap_parts[0])
            stop = float(overlap_parts[1])
            num = int(overlap_parts[2])
            overlap_grid = np.linspace(start, stop, num)
            if args.verbose:
                print(f"Using linspace overlap grid: {start} to {stop} with {num} points")
        except ValueError:
            # If conversion fails, treat as explicit values
            overlap_grid = np.array([float(x.strip()) for x in overlap_parts])
            if args.verbose:
                print(f"Using explicit overlap grid: {overlap_grid}")
    else:
        # Format: "val1,val2,val3,..." -> explicit values
        overlap_grid = np.array([float(x.strip()) for x in overlap_parts])
        if args.verbose:
            print(f"Using explicit overlap grid: {overlap_grid}")
    
    # Create seed range
    seeds = range(args.seed_start, args.seed_end)
    
    
    # Print configuration
    print("\n" + "="*60)
    print("ATT Estimation Configuration")
    print("="*60)
    print(f"Distribution: {args.dist}")
    print(f"Dimensionality (d): {args.d}")
    print(f"Sample size (n): {args.n}")
    print(f"Treatment proportion (r): {args.r}")
    print(f"Number of folds (K): {args.K}")
    print(f"Seeds: {args.seed_start} to {args.seed_end-1} (n={len(seeds)})")
    print(f"Overlap grid: {len(overlap_grid)} points from {overlap_grid.min():.2f} to {overlap_grid.max():.2f}")
    print(f"Outcome model: {args.outcome_model_type}")
    print(f"Outcome type: {args.outcome_type}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Output CSV: {args.out_csv}")
    print(f"Output summary: {args.out_summary}")
    print("="*60 + "\n")
    
    # Ensure output directories exist
    from pathlib import Path
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Run main_parallel
    try:
        start_time = time.time()
        df_long, out_summary = main_parallel(
            seeds=seeds,
            r=args.r,
            n=args.n,
            K=args.K,
            d=args.d,
            sigma=args.sigma,
            max_avg_weight=args.max_avg_weight,
            ps_order=args.ps_order,
            overlap_grid=overlap_grid,
            n_jobs=args.n_jobs,
            dist=args.dist,
            log_dir=args.log_dir,
            outcome_model_type=args.outcome_model_type,
            outcome_type=args.outcome_type,
            precomputed_distances=args.precomputed_distances,
            path=args.out_csv,
            path_summary=args.out_summary,
            data_path=args.data_path,
            synthetic_path=args.synthetic_path,
            save_intermediate=args.save_intermediate,
            degree=args.degree
        )

        print("\n" + "="*60)
        print("Computation completed successfully!")
        print("="*60)
        print(f"takes time: {time.time()-start_time}")
        print(f"Results shape: {df_long.shape}")
        print(f"Results saved to: {args.out_csv}")
        print(f"Summary saved to: {args.out_summary}")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Computation failed!")
        print("="*60)
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        raise

# # Usage example:
# if __name__ == "__main__":
#     d = 5
#     max_avg_weight = np.inf
#     outcome_model_type='linear'
#     outcome_type='quadratic'
#     dist = 'beta'
#     path = f"results/{dist}_shift/{d}d_{dist}_{outcome_type}_{outcome_model_type}.csv"
#     path_summary = f"results/{dist}_shift/{d}d_{dist}_{outcome_type}_{outcome_model_type}_summary.csv"

#     # Run with 100 different seeds
#     df_long, summary = main_parallel(
#         seeds=range(0, 100),  # 100 seeds
#         # seeds=range(1),
#         r=3/5,
#         n=10000,
#         K=5,
#         d=d,
#         sigma=0.01,
#         max_avg_weight=np.inf,
#         ps_order=1,
#         overlap_grid=np.linspace(0, 13, 14), # np.array([0,1.])
#         n_jobs=-2,  # -1, Use all available cores; -2, leave one core free
#         dist=dist,
#         log_dir="logs/parallel_run",
#         outcome_model_type=outcome_model_type, outcome_type=outcome_type,
#         path=path,
#         path_summary=path_summary,
#         save_intermediate=False
#     )

#     # df_long = pd.read_csv(path)
#     # # Main plot with error bars (recommended)
#     # summary_stats = plot_with_error_bars(
#     #     df_long, 
#     #     error_type='ci95',  # Choose: 'std', 'sem', or 'ci95'
#     #     save_path=f"figs/normal_shift/{d}d_{dist}_{outcome_type}_{outcome_model_type}_nolog.png"
#     # )

#     # # Diagnostic plot (optional - good for checking outliers)
#     # plot_individual_seeds(
#     #     df_long,
#     #     max_seeds=5,
#     #     save_path="figs/normal_shift/d1_example_seeds.png"
#     # )

