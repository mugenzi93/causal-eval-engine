# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project goal

This project automates end-to-end evaluation of observational studies. The objective is a pipeline that takes raw observational data and a study config, then runs the full causal inference workflow — data validation, diagnostics, DAG construction, estimation, sensitivity analysis, and report generation — without manual intervention.

The user wants the ability to choose **Python or R** for any given step, and to run a broad set of causal methods. New methods should be addable without touching the core pipeline.

## Critical: path encoding issue

The project lives inside a folder called `Desktop - Clement's MacBook Pro` where the apostrophe is a **Unicode RIGHT SINGLE QUOTATION MARK (U+2019, `\xe2\x80\x99`)**, not a standard ASCII apostrophe (`'`, U+0027).

macOS APFS treats these as distinct directory names, so two folders with the same visible name can coexist on disk. The Claude Code Read/Write tools normalize the path to ASCII (`\x27`), causing them to write to a **ghost sibling directory** that is invisible to the shell, VS Code, and git.

**Workaround:** Always use `python3` with explicit byte-string path construction, or use `Bash` tool commands (shell, git, pytest) for any file operations — never rely on the Read/Write tools to write source files in this repo. If files written by Read/Write tools are not visible in VS Code, run the copy script below to migrate them from the ghost path to the real one:

```python
import os, shutil

desktop = '/Users/clementmugenzi/Desktop'
ghost_name = b'Desktop - Clement\x27s MacBook Pro'   # ASCII apostrophe (wrong)
real_name  = b'Desktop - Clement\xe2\x80\x99s MacBook Pro'  # U+2019 (correct)

ghost = os.path.join(desktop.encode(), ghost_name).decode()
real  = os.path.join(desktop.encode(), real_name).decode()

src = os.path.join(ghost, 'Python', 'causal-eval-engine')
dst = os.path.join(real,  'Python', 'causal-eval-engine')

for dirpath, dirnames, filenames in os.walk(src):
    dirnames[:] = [d for d in dirnames if d not in {'.git', '__pycache__', '.pytest_cache'}]
    rel = os.path.relpath(dirpath, src)
    os.makedirs(os.path.join(dst, rel), exist_ok=True)
    for f in filenames:
        if not f.endswith('.pyc'):
            shutil.copy2(os.path.join(dirpath, f), os.path.join(dst, rel, f))
```

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install R packages (optional — all methods have Python fallbacks)
Rscript install.R

# Run the full pipeline
python run_eval.py --config config/study.yaml

# Run tests
pytest tests/

# Run a single test
pytest tests/test_ingestion.py::test_balance_summary_shape
```

## Architecture

**Entry point:** `run_eval.py` — CLI that orchestrates the six-step pipeline. Resolves the data path relative to the project root (where `run_eval.py` lives) and `os.chdir`s there so all relative output paths (`output/`, `output/figures/`) resolve correctly regardless of where the script is invoked from.

**Config schema (`config/study.yaml`):**

| Key | Required | Notes |
|---|---|---|
| `data.path` | yes | CSV or Parquet; relative paths resolved from project root |
| `data.id_col` | yes for FE | Entity identifier for panel data |
| `data.time_col` | yes for DiD/FE | Time period column |
| `outcome` | yes | Outcome variable name |
| `treatment` | yes | Binary treatment indicator (0/1) |
| `covariates` | yes | List of covariate column names |
| `instrument` | for IV | Instrument variable name |
| `running_variable` | for RDD | Running variable name |
| `cutoff` | for RDD | Threshold value |
| `methods` | yes | Any subset of: `psm`, `did`, `iv`, `rdd`, `fixed_effects`, `aipw` |
| `sensitivity` | no | Set `true` to compute E-values |

**`src/` modules:**

| Module | Responsibility |
|---|---|
| `ingestion.py` | `load_data` (CSV or Parquet), `validate_columns`, `missingness_report`, `balance_summary`, `ingest` (wrapper) |
| `diagnostics.py` | `table_one`, `love_plot`, `density_plots`, `ecdf_plots`, `run_diagnostics`. All figures saved to `output/figures/` using paths anchored to `__file__` (not CWD) |
| `dag.py` | Builds a `networkx.DiGraph`; covariates → treatment, covariates → outcome, treatment → outcome. Computes minimal adjustment set. Renders PNG |
| `estimators/` | Plugin registry — `REGISTRY` dict maps method name → function |
| `sensitivity.py` | E-value computation (VanderWeele & Ding 2017 / Haneuse 2019 continuous-outcome approximation) |
| `report.py` | Renders `src/templates/report.html.j2` via Jinja2; all images embedded as base64 so `output/report.html` is fully self-contained. Registers a `b64img` filter for per-result figure embedding |

**Estimator details:**

| File | Method | Notes |
|---|---|---|
| `psm.py` | Propensity Score Matching | Logistic regression PS → 1:1 nearest-neighbour (`ball_tree`). Returns `propensity_scores` in result dict (used for post-matching diagnostics in `run_eval.py`, then popped before report) |
| `did.py` | Difference-in-Differences | Two-period DiD (first vs. last time point); OLS SE via `lstsq`. Generates event-study plot when >2 periods |
| `iv.py` | Instrumental Variables | Manual 2SLS. First-stage F-stat; `weak_instrument_warning: true` when F < 10 |
| `rdd.py` | Regression Discontinuity | Tries R `rdrobust` via `rpy2` first; falls back to local linear with IK bandwidth. Widens bandwidth progressively if too few obs near cutoff |
| `fixed_effects.py` | Two-Way Fixed Effects | Tries `linearmodels.PanelOLS` with entity + time effects and clustered SEs; falls back to manual within-transformation. Generates parallel trends plot |
| `aipw.py` | Doubly-Robust AIPW | 5-fold cross-fitted propensity (LogisticRegression) and outcome (Ridge) models; propensity trimmed to [0.01, 0.99]. Semiparametrically efficient |

**Adding a new estimator:**
1. Create `src/estimators/<name>.py` with `run_<name>(df: pd.DataFrame, config: dict) -> dict`
2. Result dict must include `"method"`. Include `"ate"` (or `"late"`) + `"ci_lower"` / `"ci_upper"` for sensitivity analysis to pick it up automatically
3. Register in `src/estimators/__init__.py` `REGISTRY`
4. Add to `config/study.yaml` `methods` list

**R-backed estimators:** `rpy2` is in `requirements.txt`; `install.R` pre-installs MatchIt, WeightIt, rdrobust, fixest, ivreg, did, cobalt, tableone. Wrap R calls inside the same `run_<name>(df, config) -> dict` interface.

**Output:** `output/report.html` (self-contained, all figures base64-embedded) and `output/figures/*.png`. Both regenerated on every run.

## Key constraints

- The DAG structure is fixed: covariates → treatment, covariates → outcome, treatment → outcome. Not user-configurable from YAML.
- `run_eval.py` uses `sys.path.insert` to put `src/` on the path; tests do the same. There is no installed package.
- All output paths in `src/` modules must be anchored to `Path(__file__).resolve().parent...` — never `Path("output/...")` relative to CWD, because modules are imported before `os.chdir` takes effect.
- Data: `.parquet` and `.csv` supported; format detected by suffix in `ingestion.load_data`.
