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

Run everything from the repository root as `python -m src.<module>`. Inputs and outputs go to `data/`, `precomputed/`, `results/` and `figs/`, which are created on first use. When an option is not given, `python -m src.att_main` uses the settings of the paper (outcome model, overlap grid, data paths); see `--help` for the full list.

## How each experiment runs

Every study follows the same four steps, and each section below gives the exact commands.

1. Precompute the distances used for the data-driven choice of eta: `src/scripts/run_dist.sh`.
2. Run one seed per job: `src/scripts/run_sim.sh` (simulations) or `src/scripts/run_data.sh` (real data). The scripts carry SLURM headers (edit the account and partition lines for your cluster) and also run directly with `bash` when a seed is given as the last argument. Without SLURM, loop over seeds, for example `for s in $(seq 0 99); do bash src/scripts/run_sim.sh norm 20 $s; done`.
3. Merge the per-seed files: `python -m src.plot_utils.merge_seed_csv`.
4. Make the figure or table with the script named in the section.

`bash src/scripts/submit_all.sh` submits steps 1 and 2 for every setting on SLURM. The paper uses seeds 0 to 99 (100 seeds) everywhere.

### Section 6.1: Gaussian simulation (Figure 2; Normal 5D figure in the appendix)

Covariates `X | A=1 ~ N(-s, 5^2+1)^d`, `X | A=0 ~ N(s, 5^2+1)^d`, `s = 0, ..., 13`, `d = 20` in the main text and `d = 5` in the appendix (`src/data_utils/gen_data.py`, no data to prepare).

```bash
bash src/scripts/run_dist.sh norm 20                  # -> precomputed/distances_norm_d20.pkl
sbatch --array=0-99 src/scripts/run_sim.sh norm 20    # -> results/norm_shift/seeds/20d_norm_quadratic_linear_seed{s}.csv
python -m src.plot_utils.merge_seed_csv --dist norm --d 20   # -> results/norm_shift/20d_norm_quadratic_linear.csv
python -m src.plot_utils.gen_sim_figs                 # -> figs/norm_shift/20d_norm_quadratic_linear_log.pdf (Figure 2, panels A-C)
                                                      #    figs/norm_shift/20d_norm_quadratic_linear_combined.pdf (panels A-F)
```

Repeat with `5` in place of `20` for the appendix figure (`figs/norm_shift/5d_norm_quadratic_linear_combined.pdf`). `gen_sim_figs` makes all four simulation settings at once from whatever merged files exist.

### Section 6.2: Lalonde job-training data (Table 1)

NSW treated units and the PSID comparison group of Dehejia and Wahba, downloaded from Rajeev Dehejia's page at https://users.nber.org/~rdehejia/data/. Outcome model: ridge regression.

```bash
python -c "from src.data_utils.gen_lalonde import load_lalonde_data as f; f(path='data/lalonde_psid.csv', download=True, version='psid', include_categorical=False, compute_true_ATT=True, standardize=True)"
bash src/scripts/run_dist.sh lalonde                  # -> precomputed/distances_lalonde.pkl
sbatch --array=0-99 src/scripts/run_data.sh lalonde   # -> results/lalonde/seeds/lalonde_ridge_seed{s}.csv
python -m src.plot_utils.merge_seed_csv --dist lalonde       # -> results/lalonde/lalonde_ridge.csv
python -m src.plot_utils.plot --dist lalonde          # -> figs/lalonde/table_bias_variance_mse_ridge.tex (Table 1)
```

### Section 6.3: Medical Expenditure Panel Survey (Table 2; Figure 1)

Full Year Consolidated public use files from AHRQ (https://meps.ahrq.gov); the paper uses the 2022 file HC-243. The ATT sample restricts to ages 58 to 72 with Medicare status from `MCREV`; outcomes are regenerated semi-synthetically so that the ATT is known. Outcome model: quadratic.

```bash
python -m src.data_utils.download_meps                # -> data/meps_data/h243.dta (and earlier years), meps_2019_2022_processed.csv
python -m src.data_utils.gen_meps --process           # -> data/meps_data/processed/meps_medicare_att.csv
python -m src.data_utils.gen_meps --create-semisynthetic   # -> data/meps_data/processed/meps_medicare_att_semisynthetic_severe.csv
bash src/scripts/run_dist.sh meps                     # -> precomputed/distances_meps.pkl
sbatch --array=0-99 src/scripts/run_data.sh meps      # -> results/meps/seeds/meps_quadratic_seed{s}.csv
python -m src.plot_utils.merge_seed_csv --dist meps   # -> results/meps/meps_quadratic.csv
python -m src.plot_utils.plot --dist meps             # -> figs/meps/table_bias_variance_mse_quadratic.tex (Table 2)
python -m src.plot_utils.meps_figure                  # -> figs/meps/meps_age_discontinuity.pdf (Figure 1)
```

### Appendix: Beta simulation (Beta 5D and 20D figures)

Covariates `X | A=1 ~ Beta(1.3, 1.55 - 0.1 s; 50, -25)^d`, `X | A=0 ~ Beta(1.55 - 0.1 s, 2; 50, -25)^d`, same grid of `s`, `d = 5` and `20`.

```bash
bash src/scripts/run_dist.sh beta 5
sbatch --array=0-99 src/scripts/run_sim.sh beta 5     # -> results/beta_shift/seeds/5d_beta_quadratic_linear_seed{s}.csv
python -m src.plot_utils.merge_seed_csv --dist beta --d 5
python -m src.plot_utils.gen_sim_figs                 # -> figs/beta_shift/5d_beta_quadratic_linear_combined.pdf
```

Repeat with `20` for the 20D figure.

### Appendix: Right Heart Catheterization semi-synthetic study (Table 5)

Original RHC data of Connors et al. (1996) from https://hbiostat.org/data/repo/rhc.csv; real covariates and outcomes with regenerated treatment assignment (`src/data_utils/gen_rhc.py`). Outcome model: quadratic.

```bash
python -m src.data_utils.gen_rhc                      # -> data/rhc_processed.csv, data/rhc_nightmare.csv
bash src/scripts/run_dist.sh rhc                      # -> precomputed/distances_rhc.pkl
sbatch --array=0-99 src/scripts/run_data.sh rhc       # -> results/rhc/seeds/rhc_quadratic_seed{s}.csv
python -m src.plot_utils.merge_seed_csv --dist rhc    # -> results/rhc/rhc_quadratic.csv
python -m src.plot_utils.plot --dist rhc              # -> figs/rhc/table_bias_variance_mse_quadratic.tex (Table 5)
```

### Appendix: illustrative figures (no simulation runs needed)

```bash
python -m src.plot_utils.examples_plot                # -> figs/density_ratio_plots.pdf (overlap examples, Figure 3)
python -m src.plot_utils.sim_density_figure           # -> figs/sim_covariate_distributions.pdf (Figure 4)
```

## Figure conventions

Every figure script writes a colour PDF and a `_gray` greyscale PDF for print, with panel letters in the upper-left corners. The table scripts write LaTeX `table` environments that are pasted into the manuscript.

## Where things come from

| Paper item | Produced by | File |
|---|---|---|
| Figure 1 (MEPS age discontinuity) | `src.plot_utils.meps_figure` | `figs/meps/meps_age_discontinuity.pdf` |
| Figure 2 (Normal 20D, panels A-C) | `src.plot_utils.gen_sim_figs` | `figs/norm_shift/20d_norm_quadratic_linear_log.pdf` |
| Table 1 (Lalonde) | `src.plot_utils.plot --dist lalonde` | `figs/lalonde/table_bias_variance_mse_ridge.tex` |
| Table 2 (MEPS) | `src.plot_utils.plot --dist meps` | `figs/meps/table_bias_variance_mse_quadratic.tex` |
| Figure 3 (overlap examples) | `src.plot_utils.examples_plot` | `figs/density_ratio_plots.pdf` |
| Figure 4 (covariate distributions) | `src.plot_utils.sim_density_figure` | `figs/sim_covariate_distributions.pdf` |
| Figures 5-7 (Beta 5D, Beta 20D, Normal 5D) | `src.plot_utils.gen_sim_figs` | `figs/{beta,norm}_shift/{5,20}d_*_combined.pdf` |
| Table 5 (RHC) | `src.plot_utils.plot --dist rhc` | `figs/rhc/table_bias_variance_mse_quadratic.tex` |
