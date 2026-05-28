# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

This project automates end-to-end evaluation of an observational study. The objective is a pipeline that takes raw observational data and a study config, then runs the full causal inference workflow — data validation, diagnostics, DAG construction, estimation, sensitivity analysis, and report generation — without manual intervention.

The user wants the ability to choose **Python or R** for any given step, and to run a broad set of causal methods including (but not limited to):
- Propensity Score Matching (PSM)
- Difference-in-Differences (DiD)
- Instrumental Variables / 2SLS (IV)
- Regression Discontinuity Design (RDD)
- Fixed Effects Regression

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
- `data.path`, `data.id_col`, `data.time_col` — data source; `time_col` required for DiD
- `outcome`, `treatment`, `covariates` — core identifiers
- `instrument` — optional, required for IV
- `running_variable` / `cutoff` — optional, required for RDD (estimator not yet implemented)
- `methods` — list from `["psm", "did", "iv"]`
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

**Adding a new estimator (Python):** create `src/estimators/<name>.py` with a `run_<name>(df, config) -> dict` function and register it in `src/estimators/__init__.py`'s `REGISTRY`. The result dict must include `"method"` and should include `"ate"` (or `"late"`) plus `"ci_lower"` / `"ci_upper"` for sensitivity analysis to pick it up automatically.

**Adding an R-backed estimator:** the project has `rpy2` in `requirements.txt` and `install.R` pre-installs the relevant R packages (MatchIt, WeightIt, rdrobust, fixest, ivreg, did, cobalt). The convention is to wrap R calls inside the same `run_<name>(df, config) -> dict` interface so the pipeline treats Python and R estimators identically.

**Output:** `output/report.html` (self-contained) and `output/figures/*.png`. Both are regenerated on every run.

## Key constraints

- The DAG structure is currently fixed: covariates → treatment, covariates → outcome, treatment → outcome. It is not user-configurable from the YAML.
- `run_eval.py` uses `sys.path.insert` to put `src/` on the path; tests do the same. There is no installed package.
- Data supports `.parquet` and `.csv`; format is detected by file suffix in `ingestion.load_data`.
- RDD config keys (`running_variable`, `cutoff`) are parsed by `ingestion.validate_columns` but no RDD estimator exists yet — it is the next method to implement.
