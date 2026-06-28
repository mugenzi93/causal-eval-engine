# Project Structure

```
causal-eval-engine/
├── run_eval.py
├── requirements.txt
├── install.R
├── CLAUDE.md
├── README.md
├── USER_GUIDE.md
├── config/
│   ├── study.yaml
│   └── study_one.yaml
├── data/
│   └── raw/
│       ├── sample_data.csv
│       └── injury.csv
├── src/
│   ├── ingestion.py
│   ├── diagnostics.py
│   ├── dag.py
│   ├── sensitivity.py
│   ├── report.py
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── psm.py
│   │   ├── did.py
│   │   ├── drdid.py
│   │   ├── iv.py
│   │   ├── rdd.py
│   │   ├── fixed_effects.py
│   │   └── aipw.py
│   └── templates/
│       └── report.html.j2
├── output/
│   ├── report.html
│   └── figures/
├── tests/
│   └── test_ingestion.py
└── notebooks/
    └── ex.ipynb
```

---

## Root-level files

| File | Purpose |
|---|---|
| `run_eval.py` | Single entry point — orchestrates the full pipeline from data loading to report generation |
| `requirements.txt` | Python dependencies |
| `install.R` | R dependencies (optional — every method has a Python fallback) |
| `CLAUDE.md` | Instructions for Claude Code; documents architecture, constraints, and the Unicode path workaround |
| `README.md` | Project overview and quick-start |
| `USER_GUIDE.md` | Full end-user walkthrough — data prep, method selection, assumptions, report interpretation |

---

## `config/`

Study definitions. Each file describes one study: which dataset to use, which columns map to outcome/treatment/covariates, and which causal methods to run.

| File | Dataset | Notes |
|---|---|---|
| `study.yaml` | `data/raw/sample_data.csv` | Synthetic 1000-row panel data; all methods enabled |
| `study_one.yaml` | `data/raw/injury.csv` | Workers' compensation injury study; PSM + DiD + DR-DiD + AIPW |

---

## `data/raw/`

Raw input datasets. Never modified by the pipeline — all outputs go to `output/`.

| File | Description |
|---|---|
| `sample_data.csv` | Synthetic panel data for testing and demonstration |
| `injury.csv` | Real workers' compensation dataset used in `study_one.yaml` |

---

## `src/`

All pipeline logic. Imported by `run_eval.py` via `sys.path`.

| Module | Responsibility |
|---|---|
| `ingestion.py` | Loads data, validates columns, reports and imputes missing values |
| `diagnostics.py` | Table 1, love plot (multi-scheme: unadjusted, IPW, overlap, matching weights), density plots, ECDF plots |
| `dag.py` | Builds and renders the causal DAG; computes the minimal adjustment set |
| `sensitivity.py` | E-value computation for robustness to unmeasured confounding |
| `report.py` | Renders the final self-contained HTML report via Jinja2 |
| `estimators/` | One file per causal method; `__init__.py` holds the registry mapping method names to functions |
| `templates/report.html.j2` | HTML template the report is rendered from |

### `src/estimators/`

| File | Method | Notes |
|---|---|---|
| `psm.py` | Propensity Score Matching | Complete-case PS estimation; 1:1 nearest-neighbour matching |
| `did.py` | Difference-in-Differences | Raw unadjusted two-period DiD; no covariate adjustment or weighting |
| `drdid.py` | Doubly-Robust DiD | Sant'Anna & Zhao (2020); adjusts for baseline imbalance via IPW + outcome regression |
| `iv.py` | Instrumental Variables (2SLS) | Manual two-stage least squares; first-stage F-stat reported |
| `rdd.py` | Regression Discontinuity | Tries R `rdrobust` first; falls back to local linear with IK bandwidth |
| `fixed_effects.py` | Two-Way Fixed Effects | Tries `linearmodels.PanelOLS`; falls back to manual within-transformation |
| `aipw.py` | Doubly-Robust AIPW | 5-fold cross-fitted propensity and outcome models; semiparametrically efficient |

---

## `output/`

Everything the pipeline writes. Overwritten on each run — no cleanup needed between iterations.

| Path | Contents |
|---|---|
| `report.html` | Self-contained HTML report with all figures embedded as base64 |
| `figures/` | All plots saved as PNGs: love plots, density, ECDF, DAG, RDD plot, event study |

---

## `tests/`

| File | What it tests |
|---|---|
| `test_ingestion.py` | Data loading, column validation, and missingness reporting |

---

## `notebooks/`

| File | Purpose |
|---|---|
| `ex.ipynb` | Scratch notebook used during early development; not part of the pipeline |
