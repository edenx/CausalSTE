# CausalSTE

Code for the experiments in the paper on the **Stable Transport Estimator (STE)**. The STE estimates the average treatment effect on the treated (ATT) by mapping treated covariates onto the control distribution with an optimal transport map, instead of reweighting with a density ratio. EC-STE, its extrapolation-controlled variant, interpolates between the identity and the transport map through a parameter eta.

## Repository structure

```
src/
├── att_main.py           # runs all estimators over seeds and overlap levels
├── dist_main.py          # precomputes the distances used to choose eta
├── models/               # STE / EC-STE, baselines (IPW, AIPW, OR, ...), cross-fitting, OT maps
├── data_utils/           # simulation designs and the RHC, MEPS and Lalonde data builders
├── plot_utils/           # merging per-seed results, figures and LaTeX tables
└── scripts/              # SLURM job scripts
```

## Setup

```bash
conda env create -f environment.yml
conda activate ste
```

Run everything from the repository root as `python -m src.<module>`. Inputs and outputs go to `data/`, `precomputed/`, `results/` and `figs/`, which are created on first use.

## Running the experiments

Each experiment follows the same four steps.

1. `src/scripts/run_dist.sh` precomputes the distances used by the data-driven choice of eta.
2. `src/scripts/run_sim.sh` or `src/scripts/run_data.sh` runs one seed per SLURM array task.
3. `src.plot_utils.merge_seed_csv` combines the per-seed results.
4. `src.plot_utils.gen_sim_figs` makes the simulation figures, and `src.plot_utils.plot` makes the real-data tables.

Figure scripts (run from the repository root; each writes a colour PDF and a
`_gray` greyscale PDF for print, with panel letters in the upper-left corners):

| Paper figure | Script | Output |
|---|---|---|
| Sec. 6 Normal 20D (panels A to C) and appendix 5D/20D figures (panels A to F) | `python -m src.plot_utils.gen_sim_figs` | `figs/{dist}_shift/{d}d_{dist}_quadratic_linear_log.pdf`, `..._combined.pdf` |
| Overlap examples (appendix) | `python -m src.plot_utils.examples_plot` | `figs/density_ratio_plots.pdf` |
| Covariate distributions of the two designs (appendix) | `python -m src.plot_utils.sim_density_figure` | `figs/sim_covariate_distributions.pdf` |
| MEPS age discontinuity (introduction) | `python -m src.plot_utils.meps_figure` | `figs/meps/meps_age_discontinuity.pdf` |

`gen_sim_figs` reads the merged 100-seed result files `results/{dist}_shift/{d}d_{dist}_quadratic_linear.csv`.

`src/scripts/submit_all.sh` submits steps 1 and 2 for every setting. Each script's header documents its usage.

| Experiment | Prepare data | Run |
|---|---|---|
| Simulations (Normal and Beta, d = 5 and 20) | none (`src/data_utils/gen_data.py`) | `run_sim.sh` |
| RHC semi-synthetic | `python -m src.data_utils.gen_rhc` | `run_data.sh rhc` |
| MEPS semi-synthetic | `src.data_utils.download_meps`, then `src.data_utils.gen_meps --process` and `--create-semisynthetic` | `run_data.sh meps` |
| Lalonde (NSW + PSID) | `load_lalonde_data(..., download=True)` in `src/data_utils/gen_lalonde.py` | `run_data.sh lalonde` |

When an option is not given, `python -m src.att_main` uses the settings from the paper. These cover the outcome model, the overlap grid and the data paths. See `--help` for the full list.
