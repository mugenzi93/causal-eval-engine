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


## How to use this pipeline: step-by-step

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

For R-backed estimators (RDD via `rdrobust`, optional upgraded DiD/FE via `fixest`), also run:

```bash
Rscript install.R
```

This is optional. Every method has a pure Python fallback, so you can skip this step if R is not available.

---

### Step 2 — Prepare your data

Place your dataset in `data/raw/`. Accepted formats: `.csv` or `.parquet`.

Your data must have:
- One row per observation (cross-sectional) **or** one row per entity-period (panel)
- A binary treatment column (0 = control, 1 = treated)
- A numeric outcome column
- One or more numeric covariate columns

**For DiD and Fixed Effects specifically**, the data must be in long panel format with a subject ID column and a time period column. Example:

| subject_id | year | treatment | age | outcome |
|---|---|---|---|---|
| 1 | 2018 | 0 | 34 | 30000 |
| 1 | 2019 | 0 | 34 | 31500 |
| 2 | 2018 | 1 | 28 | 35000 |

**For IV**, you need an additional column that serves as the instrument (a variable that affects treatment assignment but has no direct effect on the outcome).

**For RDD**, you need a running variable (a continuous score) and a known cutoff value above/below which treatment is assigned.

---

### Step 3 — Configure your study

Edit `config/study.yaml` to match your data. At minimum, set:

```yaml
data:
  path: data/raw/your_data.csv
  id_col: your_subject_id_column     # required for Fixed Effects
  time_col: your_time_column         # required for DiD and Fixed Effects

outcome: your_outcome_column
treatment: your_treatment_column
covariates:
  - covariate_1
  - covariate_2
  - covariate_3

methods:
  - psm          # include only the methods your data supports
  - did
  - iv
  - rdd
  - fixed_effects
  - aipw

sensitivity: true
```

**Method eligibility checklist:**

| Method | What your data needs |
|---|---|
| `psm` | Binary treatment, numeric covariates |
| `did` | Panel data (`id_col` + `time_col`), at least 2 time periods |
| `iv` | An `instrument` column specified in the config |
| `rdd` | A `running_variable` and `cutoff` specified in the config |
| `fixed_effects` | Panel data (`id_col` + `time_col`), multiple entities and periods |
| `aipw` | Binary treatment, numeric covariates (same as PSM) |

Only list methods your data actually supports. The pipeline will return an error message (not crash) for any method with missing required fields, but it is cleaner to exclude them upfront.

---

### Step 4 — Run the pipeline

From the project root:

```bash
python run_eval.py --config config/study.yaml
```

You will see progress printed for each of the 6 stages:

```
[1/6] Loading and validating data...
[2/6] Running diagnostics (Table 1, love plot, density, ECDF)...
[3/6] Building causal DAG...
[4/6] Running estimators: ['psm', 'did', ...]
[5/6] Running sensitivity analysis (E-values)...
[6/6] Building HTML report...

Done. Report saved to: output/report.html
```

---

### Step 5 — Interpret the report

Open `output/report.html` in any browser. It is fully self-contained (no internet required). Work through it in order:

1. **Causal DAG** — verify the assumed causal structure looks right for your study. The adjustment set shown is what the pipeline used to control for confounding.

2. **Table 1** — check baseline balance between treated and control groups. Standardized Mean Differences (SMD) above 0.1 in absolute value flag potential confounding; these covariates need adjustment.

3. **Love plot** — shows SMDs before and after matching/weighting. After PSM, all SMDs should fall inside the dashed threshold line (|SMD| < 0.1). If they don't, the matching failed to achieve balance.

4. **Density / overlap plots** — verify there is sufficient overlap in the propensity score distribution between treated and control. If the distributions don't overlap, the positivity assumption is violated and IPW-based estimates will be unreliable.

5. **ECDF plots** — a second look at covariate overlap for individual variables.

6. **Causal effect estimates** — review the ATE (or LATE for IV/RDD) from each method. If estimates are consistent across methods, that strengthens the causal interpretation. Large disagreements across methods suggest unmet assumptions in one or more of them.

7. **Sensitivity analysis (E-values)** — for each estimate, the E-value is the minimum strength of association an unmeasured confounder would need (with both treatment and outcome) to fully explain away the result. A larger E-value means the result is more robust. An E-value close to 1.0 means a very weak confounder could explain it away entirely.

---

### Step 6 — Iterate

Common iteration patterns:

- **Add or remove covariates** — edit the `covariates` list in `study.yaml` and re-run. Watch how Table 1 SMDs and estimates change.
- **Try a different method** — add or remove entries from `methods` in `study.yaml`.
- **Swap datasets** — change `data.path`; everything else adapts automatically as long as column names match.
- **Add a new estimator** — see the *Adding a new estimator* section under Architecture.

All outputs are overwritten on each run, so there is no cleanup needed between iterations.
