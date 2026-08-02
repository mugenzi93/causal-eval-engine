# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project goal

This project automates end-to-end evaluation of observational studies. The objective is a pipeline that takes raw observational data and a study config, then runs the full causal inference workflow — data validation, diagnostics, DAG construction, estimation, sensitivity analysis, and report generation — without manual intervention.

The user wants the ability to choose **Python or R** for any given step, and to run a broad set of causal methods. New methods should be addable without touching the core pipeline.

---

## Critical: path encoding issue

The project lives inside a folder called `Desktop - Clement\u2019s MacBook Pro` where the apostrophe is a **Unicode RIGHT SINGLE QUOTATION MARK (U+2019, `\xe2\x80\x99`)**, not a standard ASCII apostrophe (`'`, U+0027).

macOS APFS treats these as distinct directory names, so two folders with the same visible name can coexist on disk. The Claude Code Read/Write tools and shell `sed` commands normalize the path to ASCII (`\x27`), causing them to write to a **ghost sibling directory** that is invisible to the shell, VS Code, and git.

**Workaround — always write files using Python byte-string path construction:**

```python
import os

desktop = '/Users/clementmugenzi/Desktop'
real_name = b'Desktop - Clement\xe2\x80\x99s MacBook Pro'
real = os.path.join(desktop.encode(), real_name).decode()
path = os.path.join(real, 'Python', 'causal-eval-engine', 'src', 'some_file.py')
open(path, 'w').write(content)
```

**Workaround — always run the pipeline using the real Unicode path via subprocess:**

```python
import os, subprocess

desktop = '/Users/clementmugenzi/Desktop'
real_name = b'Desktop - Clement\xe2\x80\x99s MacBook Pro'
real = os.path.join(desktop.encode(), real_name).decode()
project = os.path.join(real, 'Python', 'causal-eval-engine')
config  = os.path.join(project, 'config', 'study_one.yaml')
script  = os.path.join(project, 'run_eval.py')
subprocess.run(['/opt/anaconda3/bin/python', script, '--config', config], cwd=project)
```

For `config/study.yaml` (the default), `python run_eval.py --config config/study.yaml` still works if run from the real CWD. For `config/study_one.yaml` (which only exists in the real directory), always use the subprocess approach above.

**Note:** The ghost directory (`Desktop - Clement's MacBook Pro` with ASCII apostrophe) has been permanently deleted. The copy script is no longer needed.

---

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install R packages (optional — all methods have Python fallbacks)
Rscript install.R

# The AI interpretation step (agent.interpret: true) requires an API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run the full pipeline with the default config (relative path — safe from real CWD)
python run_eval.py --config config/study.yaml

# Run with the injury dataset config (use subprocess workaround — see path encoding above)
python3 -c "
import os, subprocess
desktop = '/Users/clementmugenzi/Desktop'
real_name = b'Desktop - Clement\xe2\x80\x99s MacBook Pro'
real = os.path.join(desktop.encode(), real_name).decode()
project = os.path.join(real, 'Python', 'causal-eval-engine')
config  = os.path.join(project, 'config', 'study_one.yaml')
script  = os.path.join(project, 'run_eval.py')
subprocess.run(['/opt/anaconda3/bin/python', script, '--config', config], cwd=project)
"

# Run tests
pytest tests/
```

---

## Architecture

**Entry point:** `run_eval.py` — CLI that orchestrates the six-step pipeline (plus an optional AI interpretation step, 5.5). Resolves the data path relative to the project root and `os.chdir`s there so all relative output paths (`output/`, `output/figures/`) resolve correctly.

**Config files:**

| File | Dataset | Notes |
|---|---|---|
| `config/study.yaml` | `data/raw/sample_data.csv` | 1000-row synthetic panel data; all 7 methods; AI interpretation enabled |
| `config/study_one.yaml` | `data/raw/injury.csv` | Workers\u2019 compensation injury dataset, PSM + DiD + DR-DiD + AIPW; AI interpretation enabled |

**Config schema:**

| Key | Required | Notes |
|---|---|---|
| `data.path` | yes | CSV or Parquet; relative paths resolved from project root |
| `data.id_col` | FE only | Entity identifier for panel data |
| `data.time_col` | DiD / FE | Time period column |
| `outcome` | yes | Outcome variable name |
| `treatment` | yes | Binary treatment indicator (0/1) |
| `covariates` | yes | List of covariate column names |
| `instrument` | IV only | Instrument variable name |
| `running_variable` | RDD only | Running variable name |
| `cutoff` | RDD only | Threshold value |
| `methods` | yes | Any subset of: `psm`, `did`, `drdid`, `iv`, `rdd`, `fixed_effects`, `aipw` |
| `sensitivity` | no | Set `true` to compute E-values |
| `agent.interpret` | no | Set `true` to run the AI interpretation agent (step 5.5). Requires `ANTHROPIC_API_KEY` |

**`src/` modules:**

| Module | Responsibility |
|---|---|
| `ingestion.py` | `load_data`, `validate_columns`, `missingness_report`, `impute_covariates`, `balance_summary`, `ingest` |
| `diagnostics.py` | `table_one`, `love_plot`, `density_plots`, `ecdf_plots`, `run_diagnostics`. All figures saved to `output/figures/` anchored to `__file__` |
| `dag.py` | Builds `networkx.DiGraph`; covariates → treatment, covariates → outcome, treatment → outcome. Computes minimal adjustment set. Renders PNG |
| `estimators/` | Plugin registry — `REGISTRY` maps method name → function |
| `sensitivity.py` | E-value computation (VanderWeele & Ding 2017 / Haneuse 2019) |
| `agent/interpreter.py` | AI interpretation agent — optional step 5.5. Single Claude API call (`claude-sonnet-4-6`) with structured outputs; turns estimator + sensitivity results into plain-language findings. Gated on `agent.interpret` |
| `report.py` | Renders `src/templates/report.html.j2` via Jinja2; all images base64-embedded so `output/report.html` is self-contained |

---

## Missing value handling

`run_eval.py` runs `missingness_report()` immediately after loading. If any gaps are found:

1. A warning is printed listing each column and its missing count/percentage
2. `impute_covariates()` is called automatically on all columns used by the estimators (covariates, outcome, treatment, instrument, running variable)
3. Numeric columns → **median imputation**; categorical columns → **mode imputation**
4. A summary of what was imputed is printed before any estimators run

**Propensity score estimation always uses complete cases only** (see below) — imputation covers non-PS columns (outcome, DiD time indicators, etc.).

---

## Propensity score estimation: complete cases only

Both `diagnostics.py` (`_fit_propensity`) and `psm.py` (`_estimate_propensity`) restrict PS model fitting and prediction to rows with no missing values in `[treatment] + covariates`. This is enforced via `dropna()` before fitting and returning a `pd.Series` indexed by the complete-case row positions.

**Why this matters:** overlap weights have an exact finite-sample balance property (SMD → 0) that only holds when the same observations appear in both the PS estimation and the balance check. Imputing missing values before PS estimation breaks this guarantee. By restricting to complete cases, the property is restored.

**Downstream effects:**
- The love plot balance check is restricted to the complete-case subset
- PSM matching happens within complete cases; `matched_weights` is aligned back to the full df index (0 for incomplete/unmatched rows)
- AIPW already uses `dropna()` internally — unaffected

---

## Diagnostics: multi-scheme love plot

`run_diagnostics()` produces a single love plot showing covariate balance across all available adjustment schemes:

| Scheme | Color | Marker | When shown |
|---|---|---|---|
| Unadjusted | Red | Circle | Always |
| IPW | Blue | Diamond | When PS can be estimated |
| Overlap weights | Green | Square | When PS can be estimated |
| Matching weights | Purple | Triangle | After PSM runs |

Faint horizontal lines connect each covariate\u2019s unadjusted dot to its adjusted counterparts. A vertical dashed line marks the |SMD| = 0.1 balance threshold.

`run_eval.py` calls `run_diagnostics()` twice:
1. Before estimators run: shows Unadjusted + IPW + Overlap weights
2. After PSM runs: regenerates with Matching weights added (replaces the first love plot in the report)

---

## Estimator details

All estimators use `solver="saga"` and `max_iter=2000` for any internal `LogisticRegression`, wrapped in a `StandardScaler` pipeline to ensure convergence with many covariates.

All estimators return a `p_value` field (two-sided) in addition to the estimate and CI:

| File | Method | P-value approach |
|---|---|---|
| `psm.py` | PSM | Two-sided t-test on matched means, df = 2n − 2 |
| `did.py` | DiD | Two-sided t-test on OLS interaction coefficient |
| `iv.py` | IV / 2SLS | Two-sided t-test on LATE, df = n − k |
| `rdd.py` | RDD | Extracted from `rdrobust`; normal approx for Python fallback |
| `fixed_effects.py` | Two-Way FE | Extracted from `linearmodels`; t-test for numpy fallback |
| `aipw.py` | AIPW | Two-sided z-test (normal approx — large sample) |
| `drdid.py` | DR-DiD | Two-sided z-test (normal approx — large sample) |

The report shows significance stars: `***` p < 0.001, `**` p < 0.01, `*` p < 0.05.

**Method-specific notes:**

- **PSM**: PS estimated on complete cases; 1:1 nearest-neighbour matching via `ball_tree`. Returns `propensity_scores` and `matched_weights` (consumed by `run_eval.py`, popped before report)
- **DiD**: Raw (unadjusted) two-period DiD — no covariate adjustment, no PS weighting. Uses first vs. last time point; OLS interaction coefficient at index 3 is the ATT. Relies on unconditional parallel trends. For covariate-adjusted DiD, use `drdid`. Event-study plot generated when >2 periods exist
- **IV**: Manual 2SLS via `lstsq`. First-stage F-stat reported; `weak_instrument_warning: true` when F < 10
- **RDD**: Tries R `rdrobust` via `rpy2` first; falls back to local linear with IK bandwidth. Deduplicates panel data per subject before running. Covariates passed to both estimators
- **Fixed Effects**: Tries `linearmodels.PanelOLS` with entity + time effects and clustered SEs; falls back to manual within-transformation. Covariates within-demeaned. Requires `id_col` and `time_col`
- **AIPW**: 5-fold cross-fitted propensity (LogisticRegression + StandardScaler) and outcome (Ridge + StandardScaler) models; propensity clipped to [0.01, 0.99]
- **DR-DiD**: Doubly-robust DiD (Sant'Anna & Zhao 2020). Fits a logistic PS model and two Ridge outcome models (control group, pre- and post-period). Efficient influence function combines IPW-DiD and outcome regression DiD. Consistent if either model is correctly specified. Works for panel and repeated cross-section designs. Use instead of `did` when there is meaningful baseline imbalance between treatment and control groups

**Adding a new estimator:**
1. Create `src/estimators/<name>.py` with `run_<name>(df: pd.DataFrame, config: dict) -> dict`
2. Result dict must include `"method"`. Include `"ate"` (or `"late"`) + `"ci_lower"` / `"ci_upper"` for sensitivity analysis and `"p_value"` for report display
3. Register in `src/estimators/__init__.py` `REGISTRY` (current keys: `psm`, `did`, `drdid`, `iv`, `rdd`, `fixed_effects`, `aipw`)
4. Add to the `methods` list in the relevant config YAML

---

## AI interpretation agent

`src/agent/interpreter.py` adds an optional step **5.5**, between sensitivity analysis and report generation. It is a single Claude API call (not a tool-using loop): it never touches the estimation, only translates the numeric results into plain-language findings for the report.

- **Gating:** runs only when `agent.interpret: true` in the config — `run_eval.py` checks `config["agent"]["interpret"]`. Both `study.yaml` and `study_one.yaml` have it enabled.
- **Input:** `interpret_results(estimator_results, sensitivity_results, config)` builds a JSON payload (`outcome`, `treatment`, `estimator_results`, `sensitivity_results`). `_clean_results` strips bulky/non-serializable fields via `_STRIP_KEYS` (`propensity_scores`, `matched_weights`, `rdd_plot`, `parallel_trends_plot`, `event_study_plot`).
- **Model call:** `claude-sonnet-4-6`, `max_tokens=4096`, a methodologist system prompt, and **structured outputs** (`output_config.format` + `_RESULT_SCHEMA`) that force a fixed 7-key JSON object: `consensus`, `summary`, `actionable_insights`, `conflicts`, `robustness`, `caveats`, `conclusion`.
- **Parsing:** `_parse_json_response` extracts the JSON (tolerates ```` ```json ```` fences, falls back to slicing the outermost `{...}`) before `json.loads`.
- **Output:** the returned dict is passed to `build_report`, which renders it in the green "AI interpretation" section of `report.html`.
- **Requirements & failure handling:** needs the `anthropic` package (in `requirements.txt`) and `ANTHROPIC_API_KEY` set. The call is wrapped in try/except — a missing/invalid key or any error prints `WARNING: ... skipping` and returns `None`, and the pipeline still finishes and builds the report without the interpretation section.

---

## Key constraints

- The DAG structure is fixed: covariates → treatment, covariates → outcome, treatment → outcome. Not user-configurable from YAML
- `run_eval.py` uses `sys.path.insert` to put `src/` on the path; tests do the same. There is no installed package
- All output paths in `src/` modules must be anchored to `Path(__file__).resolve().parent...` — never `Path("output/...")` relative to CWD
- Data: `.parquet` and `.csv` supported; format detected by suffix in `ingestion.load_data`
- `sensitivity.py` looks for `"ate"` first, then `"late"` — RDD results use `"tau"` and are skipped unless the key is renamed
- Python bytecode is ignored via `.gitignore` (`__pycache__/`, `*.pyc`); `output/` (report + figures) is intentionally tracked and versioned

---

## User guide

For a full walkthrough — data preparation, method selection, assumptions, and report interpretation — see [USER_GUIDE.md](USER_GUIDE.md).
