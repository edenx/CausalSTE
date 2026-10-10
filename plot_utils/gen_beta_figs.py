"""Generate beta_shift quadratic/linear figures (log + eta) for 5d and 20d.

Run from the src/ directory:  python -m plot_utils.gen_beta_figs
"""
import matplotlib
matplotlib.use("Agg")  # headless: makes plt.show() a no-op

import pandas as pd

from plot_utils.plot import (
    rename_strategies,
    compute_statistics,
    plot_bias_variance_mse,
    plot_methods_with_eta,
)

DISTS = ("beta", "norm")
outcome_type = "quadratic"
outcome_model_type = "linear"

# Tunable-eta methods (after renaming) we want in the eta figure.
ETA_STRATEGIES = [
    r"STE($\eta_{gs}$)",
    r"STE($\hat{\eta}_{chisq}$)",
    r"STE($\hat{\eta}_{w1}$)",
    r"STE($\hat{\eta}_{w2}$)",
]

for dist in DISTS:
  for d in (5, 20):
    # Full merged data (all methods) lives under figs/, not results/
    csv_path = f"figs/{dist}_shift/{d}d_{dist}_{outcome_type}_{outcome_model_type}.csv"
    output_dir = f"figs/{dist}_shift/"
    stem = f"{d}d_{dist}_{outcome_type}_{outcome_model_type}"

    print("=" * 70)
    print(f"[{dist} {d}d] loading {csv_path}")

    df = pd.read_csv(csv_path)
    # Drop the convex-hull projection method
    df = df[df["strategy"] != "individual"]
    # Paper naming convention (baselines -> caps/DiM, ours -> STE)
    df = rename_strategies(df)
    # eta is reported as (1 - mean_eta) in these runs
    df["mean_eta"] = 1 - df["mean_eta"].values

    stats = compute_statistics(df)
    print(f"[{dist} {d}d] stats for {len(stats)} (strategy, delta) combos; "
          f"methods: {sorted(stats['strategy'].unique())}")

    # --- log figure: |Bias|, Variance (log), MSE (log) ---
    plot_bias_variance_mse(
        stats,
        figsize=(18, 5),
        use_log_scale={"bias": False, "variance": True, "mse": True},
        use_abs_bias=True,
        error_bars="se",
        save_path=output_dir + f"{stem}_log.pdf",
    )

    # --- eta figure: only methods with a tunable eta that are present ---
    present_eta = [s for s in ETA_STRATEGIES if s in set(stats["strategy"].unique())]
    if present_eta:
        stats_eta = stats[stats["strategy"].isin(present_eta)].copy()
        print(f"[{dist} {d}d] eta methods: {present_eta}")
        plot_methods_with_eta(
            stats_eta,
            figsize=(18, 5),
            save_path=output_dir + f"{stem}_etas.pdf",
            one_minus_eta=True,
        )
    else:
        print(f"[{dist} {d}d] no tunable-eta methods present; skipping eta figure")

print("=" * 70)
print("DONE")
