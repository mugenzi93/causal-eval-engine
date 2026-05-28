# causal-eval-engine

A config-driven Python/R pipeline that automates end-to-end evaluation of observational studies. Point it at a dataset, declare your outcome, treatment, covariates, and methods in a YAML file, and it produces a self-contained HTML report with diagnostics, causal effect estimates, and sensitivity analysis.

---

## Goal

Observational studies are the workhorse of real-world causal inference — they are used when randomized controlled trials are impractical, unethical, or simply unavailable. But evaluating them rigorously requires executing a long, error-prone chain of steps: checking covariate balance, selecting an appropriate identification strategy, running the estimator, stress-testing the result against unmeasured confounding, and documenting everything clearly enough that others can reproduce it.

The goal of this project is to automate that entire chain. Given a dataset and a study specification, `causal-eval-engine` should be able to:

- Construct and validate the assumed causal structure (DAG)
- Produce publication-quality diagnostics (Table 1, love plots, overlap checks)
- Run multiple causal estimators and surface their estimates side-by-side
- Quantify robustness to hidden bias via sensitivity analysis
- Package all of the above into a single reproducible report

The intended users are researchers and analysts who need to evaluate treatment effects from observational data quickly and credibly — without having to re-implement standard methods from scratch every time.

## Quickstart

```bash
pip install -r requirements.txt
Rscript install.R          # install R packages (optional — Python fallbacks exist)

python run_eval.py --config config/study.yaml
# → output/report.html
```

---

## Project Structure

```
causal-eval-engine/
├── config/
│   └── study.yaml              # Declarative study specification
├── src/
│   ├── ingestion.py            # Data loading, validation, missingness report
│   ├── diagnostics.py          # Table 1, love plots, density plots, ECDFs
│   ├── dag.py                  # Causal DAG construction + adjustment set
│   ├── sensitivity.py          # E-values for unmeasured confounding
│   ├── report.py               # Self-contained HTML report builder
│   ├── templates/
│   │   └── report.html.j2      # Jinja2 HTML template
│   └── estimators/
│       ├── psm.py              # Propensity Score Matching
│       ├── did.py              # Difference-in-Differences + event study
│       ├── iv.py               # Instrumental Variables (2SLS)
│       ├── rdd.py              # Regression Discontinuity Design
│       ├── fixed_effects.py    # Two-Way Fixed Effects
│       └── aipw.py             # Doubly-Robust AIPW
├── data/raw/                   # Place your CSV or Parquet data here
├── notebooks/
│   └── ex.ipynb                # Sandbox notebook (R kernel)
├── output/
│   ├── report.html             # Generated report
│   └── figures/                # All diagnostic and method figures
├── tests/
│   └── test_ingestion.py
├── requirements.txt
├── install.R
└── run_eval.py                 # CLI entry point
```

---

## Configuration

Edit `config/study.yaml` to describe your study:

```yaml
data:
  path: data/raw/your_data.csv
  id_col: subject_id
  time_col: year            # required for DiD and Fixed Effects

outcome: earnings
treatment: program_participation
covariates: [age, education, prior_earnings]
instrument: distance_to_office   # required for IV
running_variable: score          # required for RDD
cutoff: 50                       # required for RDD

methods:
  - psm
  - did
  - iv
  - rdd
  - fixed_effects
  - aipw

sensitivity: true
```

---

## Supported Methods

| Method | Key | Notes |
|---|---|---|
| Propensity Score Matching | `psm` | 1:1 nearest-neighbor; produces pre/post love plot and density overlap |
| Difference-in-Differences | `did` | Two-period OLS DiD; event study plot for multi-period data |
| Instrumental Variables | `iv` | 2SLS with Cragg-Donald weak instrument F-test |
| Regression Discontinuity | `rdd` | Local linear with IK bandwidth; uses R `rdrobust` if available |
| Two-Way Fixed Effects | `fixed_effects` | Entity + time FE, cluster-robust SEs; uses `linearmodels` if available |
| Doubly-Robust AIPW | `aipw` | Cross-fitted (5-fold) propensity + outcome models; semiparametrically efficient |

---

## Report Contents

The generated `output/report.html` is a single self-contained file with all figures embedded as base64. It includes:

- **Causal DAG** with computed adjustment set
- **Table 1** — baseline characteristics (mean, SD, SMD) for treated vs. control
- **Love plot** — covariate SMDs before and after adjustment
- **Density / overlap plots** — propensity score and covariate distributions
- **ECDF plots** — empirical CDFs for each covariate
- **Causal effect estimates** — ATE/LATE with 95% CIs for each selected method
- **Sensitivity analysis** — E-values for unmeasured confounding per estimator

---

## Dependencies

**Python:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `pyyaml`, `jinja2`, `matplotlib`, `seaborn`, `tableone`, `linearmodels` (optional), `rpy2` (optional)

**R (optional):** `MatchIt`, `WeightIt`, `rdrobust`, `fixest`, `ivreg`, `did`, `cobalt`, `tableone`

R packages are only used when `rpy2` is installed and R is available. All methods have pure Python fallbacks.

---

## Running Tests

```bash
pytest tests/
```
