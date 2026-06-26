# Causal Eval Engine — User Guide

## Overview

`causal-eval-engine` is a config-driven pipeline that automates end-to-end evaluation of observational studies. You describe your study in a single YAML file, run one command, and receive a self-contained HTML report containing diagnostic plots, causal effect estimates from one or more methods, and sensitivity analysis.

This guide covers:
1. How to set up and run the pipeline
2. Which causal methods are available, when to use each, and what assumptions must hold
3. How to interpret the output report
4. How to iterate

---

## Part 1 — Getting Started

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
- A binary treatment column (0 = control, 1 = treated)
- A numeric outcome column
- One or more numeric covariate columns

**For DiD and Fixed Effects**, the data must be in long panel format with a subject ID column and a time period column:

| subject_id | year | treatment | age | outcome |
|---|---|---|---|---|
| 1 | 2018 | 0 | 34 | 30000 |
| 1 | 2019 | 0 | 34 | 31500 |
| 2 | 2018 | 1 | 28 | 35000 |

**For IV**, you need an additional instrument column — a variable that affects treatment assignment but has no direct effect on the outcome.

**For RDD**, you need a continuous running variable and a known cutoff value above or below which treatment is assigned.

---

### Step 3 — Configure your study

Edit `config/study.yaml` to match your data:

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

instrument: distance_to_office       # required for IV; omit otherwise
running_variable: score              # required for RDD; omit otherwise
cutoff: 50                           # required for RDD; omit otherwise

methods:
  - psm
  - did
  - iv
  - rdd
  - fixed_effects
  - aipw

sensitivity: true
```

> **Example study (`config/study_one.yaml`):** The injury dataset (`data/raw/injury.csv`) is drawn from a workers’ compensation study examining the effect of higher earnings (`highearn`) on injury duration (`ldurat`). The DiD methodology used for this dataset is explained in detail at:
> [https://evalsp21.classes.andrewheiss.com/example/diff-in-diff/](https://evalsp21.classes.andrewheiss.com/example/diff-in-diff/)

**Method eligibility checklist:**

| Method | What your data needs |
|---|---|
| `psm` | Binary treatment, numeric covariates |
| `did` | Panel data (`id_col` + `time_col`), at least 2 time periods |
| `iv` | An `instrument` column specified in the config |
| `rdd` | A `running_variable` and `cutoff` specified in the config |
| `fixed_effects` | Panel data (`id_col` + `time_col`), multiple entities and periods |
| `aipw` | Binary treatment, numeric covariates |
| `drdid` | Panel data (`id_col` + `time_col`), at least 2 time periods, numeric covariates |

Only list methods your data supports. The pipeline returns a descriptive error (without crashing) for any method with missing required fields, but it is cleaner to exclude them upfront.

---

### Step 4 — Run the pipeline

From the project root:

```bash
python run_eval.py --config config/study.yaml
```

Terminal output:

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

### Step 5 — Open the report

```bash
open output/report.html
```

Or drag `output/report.html` into any browser. The file is fully self-contained — all figures are embedded — so it can be shared or archived without any supporting files.

---

### Step 6 — Interpret the report

Work through the report sections in order:

1. **Causal DAG** — verify the assumed causal structure. The adjustment set shown is what the pipeline used to control for confounding. If important arrows are missing (e.g. a known confounder is not in your covariates list), add it and re-run.

2. **Table 1** — baseline balance between treated and control groups. Standardized Mean Differences (SMD) above 0.1 in absolute value flag potential confounders that need adjustment.

3. **Love plot** — SMDs before and after matching/weighting. After PSM, all SMDs should fall inside the dashed threshold (|SMD| < 0.1). Failure here means matching did not achieve balance.

4. **Density / overlap plots** — check that propensity score distributions overlap between treated and control. Non-overlapping distributions indicate a positivity violation; IPW-based estimates will be unreliable.

5. **ECDF plots** — a per-covariate view of distributional overlap.

6. **Causal effect estimates** — review the ATE (or LATE for IV/RDD) from each method. Consistent estimates across methods strengthen the causal interpretation. Large disagreements suggest unmet assumptions in one or more methods.

7. **Sensitivity analysis (E-values)** — the E-value is the minimum strength of association an unmeasured confounder would need (with both treatment and outcome) to fully explain away the observed effect. A larger E-value means greater robustness. An E-value close to 1.0 means a very weak confounder could nullify the result.

---

### Step 7 — Iterate

Common patterns:

- **Adjust covariate set** — edit `covariates` in `study.yaml` and re-run. Watch how Table 1 SMDs and estimates respond.
- **Try a different method** — add or remove entries from `methods`.
- **Swap datasets** — change `data.path`; everything else adapts automatically as long as column names match.
- **Add a new estimator** — see the Architecture section in `CLAUDE.md`.

All outputs are overwritten on each run; no cleanup is needed between iterations.

---

## Part 2 — Causal Methods: When to Use Each and Key Assumptions

### Method 1: Propensity Score Matching (PSM)

**What it does**
Estimates the treatment effect by matching each treated unit to a control unit with a similar probability of receiving treatment (the propensity score), then comparing outcomes within matched pairs. This balances measured covariates between groups, mimicking a randomized experiment on the observed confounders.

**When to use it**
Use PSM when:
- Treatment assignment is observational (not randomized)
- You believe all important confounders are measured and included as covariates
- Treated and control groups have reasonable overlap in covariate distributions
- You want a transparent, intuitive method with interpretable diagnostics (love plot, overlap)

PSM is a good default first method for cross-sectional observational studies.

**Key assumptions**

| Assumption | What it means | How to check |
|---|---|---|
| Unconfoundedness (ignorability) | All confounders are measured and included — there are no unmeasured variables that affect both treatment and outcome | Cannot be tested directly; defend with theory and domain knowledge |
| Overlap (positivity) | Every unit has a non-zero probability of receiving either treatment or control | Inspect density/overlap plots in the report; extreme propensity scores near 0 or 1 signal a violation |
| SUTVA | One unit's treatment does not affect another unit's outcome; treatment is well-defined | Assess by study design; problematic in networks or when treatment spills over |

**What the output means**
PSM estimates the **Average Treatment Effect on the Treated (ATT)** — the effect of treatment for those who actually received it. The `ate` field in the report is this quantity.

---

### Method 2: Difference-in-Differences (DiD)

**What it does**
Compares the change in outcomes over time for the treated group to the change over time for a control group. The key insight is that common time trends cancel out, isolating the treatment effect from any pre-existing level differences between groups.

**When to use it**
Use DiD when:
- You have panel data (multiple observations per unit over time)
- A treatment began at a known point in time for some units but not others
- You cannot assume all confounders are measured, but you are willing to assume treated and control groups would have trended similarly in the absence of treatment

DiD is appropriate for policy evaluations, natural experiments, and any setting where an intervention was rolled out at a specific time.

**Key assumptions**

| Assumption | What it means | How to check |
|---|---|---|
| Parallel trends | In the absence of treatment, the treated group's outcome would have changed by the same amount as the control group's | Inspect the event-study plot in the report; pre-treatment coefficients should be near zero and flat |
| No anticipation | Units do not change behavior before the treatment officially starts | Check for pre-treatment trends in the event study |
| Stable composition | The composition of treated and control groups does not change over time in ways that affect outcomes | Assess by study design |
| SUTVA | No spillover between units | Assess by study design |

**What the output means**
DiD estimates the **Average Treatment Effect on the Treated (ATT)**. The `ate` field is the difference-in-differences estimate. The event-study plot (when >2 time periods are present) shows dynamic treatment effects over time.

**Further reading**
The DiD example used in `config/study_one.yaml` (workers’ compensation injury data) is fully explained at:
[https://evalsp21.classes.andrewheiss.com/example/diff-in-diff/](https://evalsp21.classes.andrewheiss.com/example/diff-in-diff/)

---

### Method 3: Instrumental Variables (IV / 2SLS)

**What it does**
Uses a third variable — the instrument — that affects treatment assignment but has no direct effect on the outcome. By exploiting only the variation in treatment that is driven by the instrument, IV estimates a causal effect even in the presence of unmeasured confounders.

**When to use it**
Use IV when:
- You suspect important confounders are unmeasured (so PSM and AIPW are not credible)
- You have a plausible instrument — a variable that shifts treatment assignment but is otherwise unrelated to the outcome

Classic instruments: distance to a facility (for access-based treatments), lottery assignment, policy changes that affect only some groups, geographic or institutional variation.

**Key assumptions**

| Assumption | What it means | How to check |
|---|---|---|
| Relevance | The instrument is meaningfully correlated with treatment | Inspect the first-stage F-statistic in the report; F < 10 is a weak instrument warning |
| Exclusion restriction | The instrument affects the outcome **only** through its effect on treatment — no direct path | Cannot be tested; must be defended on theoretical grounds |
| Independence | The instrument is as good as randomly assigned — uncorrelated with unmeasured confounders | Cannot be tested directly; defend with study design and falsification tests |
| Monotonicity | The instrument moves all units in the same direction (no defiers) | Assess by context; usually plausible for binary instruments |

**What the output means**
IV estimates the **Local Average Treatment Effect (LATE)** — the effect for "compliers," the subset of units whose treatment status is changed by the instrument. LATE may not generalize to the full population. The `late` field in the report is this quantity. A weak instrument (F < 10) produces unreliable estimates with wide confidence intervals.

---

### Method 4: Regression Discontinuity Design (RDD)

**What it does**
Exploits a sharp threshold in a continuous "running variable" that determines treatment eligibility. Units just above and just below the cutoff are assumed to be similar in all ways except their treatment status, creating a local natural experiment at the boundary.

**When to use it**
Use RDD when:
- Treatment is assigned based on whether a score or index crosses a known threshold (e.g., test score cutoffs, income thresholds, age eligibility, geographic boundaries)
- You can obtain data on the running variable for all units
- There are enough observations near the cutoff to estimate local effects

RDD is one of the most credible non-experimental designs when its assumptions hold, because it does not require measuring all confounders.

**Key assumptions**

| Assumption | What it means | How to check |
|---|---|---|
| No manipulation | Units cannot precisely control whether they fall just above or just below the cutoff | Inspect the density of the running variable around the cutoff (McCrary density test); a spike just above or below suggests sorting |
| Continuity | The relationship between the running variable and the outcome is continuous at the cutoff in the absence of treatment | Cannot be directly tested; check using placebo cutoffs and alternative outcomes |
| Local randomization | Near the cutoff, treatment assignment is effectively random | Supported when manipulation is absent and covariates are balanced near the threshold |

**What the output means**
RDD estimates the **LATE at the cutoff** — the treatment effect for units at the threshold. This is a local estimate and may not generalize to units far from the cutoff. The `tau` field in the report is this quantity. The bandwidth determines how far from the cutoff observations are used; narrower bandwidths are more credible but less precise.

---

### Method 5: Two-Way Fixed Effects (Fixed Effects Regression)

**What it does**
Controls for all time-invariant unobserved differences between units (entity fixed effects) and all common time shocks shared across units (time fixed effects), by demeaning the data within each entity and each time period. What remains after demeaning is within-unit, within-period variation.

**When to use it**
Use Fixed Effects when:
- You have panel data with multiple observations per unit across time
- You are worried about unobserved, time-invariant confounders (e.g., innate ability, geographic characteristics, firm culture)
- Treatment varies within units over time (some units switch from control to treated, or treatment intensity changes)

Fixed effects is the standard approach for panel data in economics and policy research.

**Key assumptions**

| Assumption | What it means | How to check |
|---|---|---|
| Strict exogeneity | Treatment is uncorrelated with past and future errors — no feedback from outcomes to future treatment | Difficult to test; violated if past outcomes affect future treatment decisions |
| No time-varying confounders | Only time-invariant confounders are removed; unmeasured variables that change over time and affect both treatment and outcome remain a threat | Assess by domain knowledge; add time-varying covariates to the model if available |
| Parallel trends | In the absence of treatment, units would have followed parallel paths over time | Inspect the parallel trends plot in the report |
| Sufficient within-unit variation | Treatment must vary within units over time; purely between-unit variation is absorbed by fixed effects | Check that not all treated units are always treated (no time variation = no identification) |

**What the output means**
Fixed Effects estimates the **Average Treatment Effect (ATE)** within units over time. The `ate` field is this estimate with cluster-robust standard errors (clustered by entity). Standard errors account for serial correlation within units.

---

### Method 6: Doubly-Robust AIPW (Augmented Inverse Probability Weighting)

**What it does**
Combines two models: a propensity score model (probability of treatment given covariates) and an outcome model (expected outcome given treatment and covariates). The AIPW estimator is called "doubly robust" because it gives consistent estimates if **either** model is correctly specified — not both. Both models are fit using cross-validation to avoid overfitting bias.

**When to use it**
Use AIPW when:
- Treatment assignment is observational and you believe all confounders are measured (same setting as PSM)
- You want semiparametric efficiency — the smallest possible variance among doubly-robust estimators
- You want robustness against misspecification of either the propensity or outcome model
- You are comfortable with a slightly more complex method in exchange for better statistical properties

AIPW is generally preferred over plain PSM when sample sizes are moderate to large, as it is more efficient and more robust.

**Key assumptions**

| Assumption | What it means | How to check |
|---|---|---|
| Unconfoundedness (ignorability) | All confounders are measured and included — same as PSM | Cannot be tested; defend with theory and domain knowledge |
| Overlap (positivity) | Every unit has a non-zero probability of receiving either treatment | Propensity scores are trimmed to [0.01, 0.99] automatically; check overlap plots |
| Correct specification of at least one model | Either the propensity model or the outcome model (or both) is correctly specified | Use flexible model classes; inspect both model fits if possible |
| SUTVA | No interference between units | Assess by study design |

**What the output means**
AIPW estimates the **Average Treatment Effect (ATE)** — the effect of treatment for the entire population (not just the treated). It is semiparametrically efficient, meaning it achieves the lowest possible variance given the assumptions. The `ate` field is this quantity. Confidence intervals are computed from the influence function (standard errors are analytic, not bootstrap).

---

### Method 7: Doubly-Robust Difference-in-Differences (DR-DiD)

**What it does**
Extends plain DiD by combining two adjustments: a propensity score model (probability of treatment given covariates) and an outcome regression model for the control group. The estimator uses the efficient influence function from Sant'Anna & Zhao (2020), which is doubly robust — consistent if **either** the propensity model **or** the outcome model is correctly specified. DR-DiD produces better estimates than plain DiD when treated and control units differ meaningfully on observed covariates at baseline, because it adjusts for those differences rather than relying solely on the parallel-trends assumption.

**When to use it over plain DiD**
Use DR-DiD instead of `did` when:
- Baseline covariate imbalance between treated and control groups is substantial (SMD > 0.1 on key covariates)
- You want robustness against model misspecification — DR-DiD is consistent even if one of the two adjustment models is wrong
- Your sample is large enough for the asymptotic normal approximation to be reliable
- You want semiparametric efficiency gains over plain DiD

If treated and control groups are well-balanced at baseline and you trust the parallel-trends assumption, plain `did` and `drdid` should give similar results. Consistent estimates across both methods strengthen credibility.

**Key assumptions**

| Assumption | What it means | How to check |
|---|---|---|
| Conditional parallel trends | After adjusting for covariates, the treated group's outcome would have changed by the same amount as the control group in the absence of treatment | Inspect the event-study plot from `did` as a proxy; covariate adjustment makes this more plausible than unconditional parallel trends |
| No anticipation | Units do not change behavior before the treatment officially starts | Check for pre-treatment trends in the event study |
| Overlap (positivity) | Every unit has a non-zero probability of being in the treated or control group given covariates | Propensity scores are clipped to [0.01, 0.99] automatically; check overlap plots |
| Correct specification of at least one model | Either the propensity model or the outcome regression model (or both) is correctly specified | Double robustness means you get one "free" misspecification; use flexible model classes for both |
| SUTVA | No spillover between units | Assess by study design |

**What the output means**
DR-DiD estimates the **Average Treatment Effect on the Treated (ATT)** — the effect of treatment for those who actually received it, at the post-treatment time period. The `ate` field is this quantity. Standard errors are computed from the empirical variance of the efficient influence function (analytic, not bootstrap), and inference uses a normal approximation. The `engine` field shows whether Python (default) or R was used.

**Reference**
Sant'Anna, P. H. C. & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *Journal of Econometrics*, 219(1), 101–122.

---

## Part 3 — Choosing the Right Method

The table below summarizes which method is appropriate given your study's data structure and identification strategy:

| Your situation | Recommended method(s) |
|---|---|
| Cross-sectional data, all confounders measured | PSM, AIPW |
| Panel data, treatment varies over time | DiD, DR-DiD, Fixed Effects |
| Panel data with baseline imbalance between groups | DR-DiD over plain DiD |
| Unmeasured confounders, valid instrument available | IV |
| Treatment assigned by a score threshold | RDD |
| Want to compare methods for robustness | Run all applicable methods and check consistency |
| Large sample, want maximum efficiency | AIPW over PSM |
| Small sample near a cutoff | RDD (if threshold exists) |

As a general rule: **use the method whose identification assumptions are most plausible given your study design**, not simply the one that produces the estimate you prefer. When multiple methods are applicable, running them all and checking consistency is a form of sensitivity analysis in itself.
