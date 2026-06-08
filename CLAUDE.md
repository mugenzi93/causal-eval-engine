# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

This project automates end-to-end evaluation of an observational study. The objective is a pipeline that takes raw observational data and a study config, then runs the full causal inference workflow — data validation, diagnostics, DAG construction, estimation, sensitivity analysis, and report generation — without manual intervention.

The user wants the ability to choose **Python or R** for any given step, and to run a broad set of causal methods including (but not limited to):
- Propensity Score Matching (PSM)
- Difference-in-Differences (DiD)
- Instrumental Variables / 2SLS (IV)
- Regression Discontinuity Design (RDD)
- Fixed Effects Regression (Two-Way FE)
- Augmented Inverse Probability Weighting (AIPW)

New methods should be addable without touching the core pipeline.

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_eval.py --config config/study.yaml

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_ingestion.py::test_balance_summary_shape

# Install companion R packages (MatchIt, WeightIt, rdrobust, fixest, ivreg, did, cobalt, tableone)
Rscript install.R
```

## Architecture

**Entry point:** `run_eval.py` — CLI that orchestrates the six-step pipeline. It resolves the data path relative to the project root and `os.chdir`s there so all relative output paths work.

**Config schema (`config/study.yaml`):**
- `data.path`, `data.id_col`, `data.time_col` — data source; `time_col` and `id_col` required for DiD and Fixed Effects
- `outcome`, `treatment`, `covariates` — core identifiers
- `instrument` — optional, required for IV
- `running_variable` / `cutoff` — optional, required for RDD
- `methods` — list from `["psm", "did", "iv", "rdd", "fixed_effects", "aipw"]`
- `sensitivity: true` — enables E-value analysis

**`src/` modules:**

| Module | Responsibility |
|---|---|
| `ingestion.py` | `load_data` (CSV or parquet), `validate_columns`, `missingness_report`, `balance_summary`, `ingest` (convenience wrapper) |
| `dag.py` | Builds a `networkx.DiGraph` where every covariate points to both treatment and outcome; computes minimal adjustment set via ancestor/descendant logic; renders PNG |
| `diagnostics.py` | `table_one`, `love_plot`, `density_plots`, `ecdf_plots`, `run_diagnostics` (wrapper). All figures saved to `output/figures/` |
| `estimators/` | Plugin registry — `REGISTRY` dict maps method name → function |
| `sensitivity.py` | E-value computation (VanderWeele & Ding 2017 / Haneuse 2019 continuous-outcome approximation) |
| `report.py` | Renders `src/templates/report.html.j2` via Jinja2; all images embedded as base64 so `output/report.html` is self-contained |

**Estimator details:**
- `psm.py` — Logistic regression propensity score → 1:1 nearest-neighbour matching (`ball_tree`). Returns `propensity_scores` in result dict (consumed by `run_eval.py` for post-matching diagnostics, then popped before report).
- `did.py` — Two-period DiD using first vs. last time point; OLS SE via manual `lstsq`. Generates an event-study plot when >2 periods exist.
- `iv.py` — Manual 2SLS (`lstsq`). Reports first-stage F-stat; sets `weak_instrument_warning: true` when F < 10.
- `rdd.py` — Regression discontinuity via local linear regression at the cutoff. Tries R's `rdrobust` first (via `rpy2`); falls back to a pure-Python IK-bandwidth local linear estimator. Saves `output/figures/rdd_plot.png`. Requires `running_variable` and `cutoff` in config.
- `fixed_effects.py` — Two-way fixed effects (entity + time). Tries `linearmodels.PanelOLS` with clustered SEs first; falls back to manual within-transformation (demeaning) with sandwich SE. Always saves `output/figures/parallel_trends.png`. Requires `id_col` and `time_col` in config.
- `aipw.py` — Doubly-robust Augmented IPW with 5-fold cross-fitting for both the propensity model (logistic + `StandardScaler`) and the outcome models (Ridge + `StandardScaler`). Propensity scores are clipped to [0.01, 0.99] for positivity.

**Adding a new estimator (Python):** create `src/estimators/<name>.py` with a `run_<name>(df, config) -> dict` function and register it in `src/estimators/__init__.py`'s `REGISTRY`. The result dict must include `"method"` and should include `"ate"` (or `"late"` / `"tau"`) plus `"ci_lower"` / `"ci_upper"` for sensitivity analysis to pick it up automatically.

**Adding an R-backed estimator:** `rpy2` is in `requirements.txt` and `install.R` pre-installs the relevant R packages (MatchIt, WeightIt, rdrobust, fixest, ivreg, did, cobalt). Wrap R calls inside the same `run_<name>(df, config) -> dict` interface so the pipeline treats Python and R estimators identically. See `rdd.py`'s `_run_rdrobust` for the established pattern: try R first, catch all exceptions, return `None` to trigger the Python fallback.

**Output:** `output/report.html` (self-contained) and `output/figures/*.png`. Both are regenerated on every run.

## Key constraints

- The DAG structure is fixed: covariates → treatment, covariates → outcome, treatment → outcome. It is not user-configurable from the YAML.
- `run_eval.py` uses `sys.path.insert` to put `src/` on the path; tests do the same. There is no installed package.
- Data supports `.parquet` and `.csv`; format is detected by file suffix in `ingestion.load_data`.
- `sensitivity.py` looks for `"ate"` first, then `"late"` — RDD results use `"tau"` and will be skipped by sensitivity analysis unless the key is renamed.
- Fixed Effects and DiD both require panel data (`id_col` + `time_col`). If either column is absent the estimator returns an `"error"` key and the pipeline skips it gracefully.
