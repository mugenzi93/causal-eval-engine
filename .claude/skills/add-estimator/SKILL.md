---
name: add-estimator
description: Scaffold and register a new causal estimator plugin in causal-eval-engine, following the required 4-step procedure.
argument-hint: [estimator-name]
disable-model-invocation: true
allowed-tools: Bash(python:*), Bash(python3:*)
---

Add a new estimator named `$ARGUMENTS` (a short snake_case key, e.g. `survival`, `synthetic_control`). If no name is given, ask for one. Follow all four steps — a new method must not require touching the core pipeline beyond the registry and config.

## Path-encoding trap (must obey)

Write every file with the Python byte-string path construction below — the Read/Write tools and shell `sed` normalize the U+2019 apostrophe to ASCII and write to an invisible ghost directory.

```python
import os
desktop = '/Users/clementmugenzi/Desktop'
real_name = b'Desktop - Clement\xe2\x80\x99s MacBook Pro'
real = os.path.join(desktop.encode(), real_name).decode()
project = os.path.join(real, 'Python', 'causal-eval-engine')
path = os.path.join(project, 'src', 'estimators', '<name>.py')
open(path, 'w').write(content)
```

## Step 1 — create `src/estimators/<name>.py`

Define `def run_<name>(df: pd.DataFrame, config: dict) -> dict:`. Read inputs from `config` (`config["outcome"]`, `config["treatment"]`, `config["covariates"]`, and any method-specific keys like `instrument`, `running_variable`/`cutoff`, `id_col`/`time_col`). Match the conventions of the existing estimators:

- Use `solver="saga"`, `max_iter=2000` inside a `StandardScaler` pipeline for any internal `LogisticRegression`.
- Restrict to complete cases with `.dropna()` before fitting where a propensity/overlap balance property matters.

## Step 2 — return the required result dict

The dict **must** include `"method"`. For sensitivity analysis include `"ate"` (or `"late"`) plus `"ci_lower"`/`"ci_upper"`, and include `"p_value"` (two-sided) for report display. Example shape (from `aipw.py`):

```python
return {
    "method": "<name>",
    "ate": round(float(ate), 4),
    "ci_lower": round(float(ci_lower), 4),
    "ci_upper": round(float(ci_upper), 4),
    "n_obs": int(n),
    "p_value": round(p_value, 4) if not np.isnan(p_value) else None,
    "note": "One-line description of the method.",
}
```

Note: `sensitivity.py` looks for `"ate"` then `"late"`. If the natural estimand key is something else (RDD uses `"tau"`), it is skipped by sensitivity unless renamed.

## Step 3 — register in `src/estimators/__init__.py`

Add both the import and the `REGISTRY` entry (current keys: `psm`, `did`, `drdid`, `iv`, `rdd`, `fixed_effects`, `aipw`), and add the name to `__all__`:

```python
from .<name> import run_<name>
# ...
REGISTRY = { ..., "<name>": run_<name> }
```

## Step 4 — enable it in a config

Add `<name>` to the `methods:` list in the relevant `config/*.yaml`, and confirm that config supplies any method-specific keys the estimator reads.

## Verify

Run the pipeline (see the `run-study` skill) on a config that includes the new method and confirm the estimator appears in `output/report.html` with an estimate, CI, and p-value. Report failures verbatim.
