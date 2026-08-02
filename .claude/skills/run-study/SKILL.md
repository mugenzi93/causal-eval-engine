---
name: run-study
description: Run the causal-eval-engine pipeline for a given study config, handling the Unicode path-encoding workaround automatically.
model: claude-haiku-4-5-20251001
argument-hint: [config-name]
disable-model-invocation: true
allowed-tools: Bash(/opt/anaconda3/bin/python:*), Bash(python:*), Bash(python3:*)
---

Run the full six-step pipeline in `run_eval.py` for the config the user names in `$ARGUMENTS` (e.g. `study`, `study_one`, or a full path). If no config is given, ask which one; default is `study`.

## Critical: path-encoding trap

This project lives inside a directory whose apostrophe is U+2019 (`\xe2\x80\x99`), not ASCII `'`. Never build the path with a plain string literal or shell `cd` into a typed path — it silently resolves to a deleted ghost sibling. Always construct the real path with Python byte-strings.

## How to run

**Default config (`config/study.yaml`)** — the relative-path form is safe when CWD is already the real project root:

```bash
python run_eval.py --config config/study.yaml
```

**Any other config, including `config/study_one.yaml`** — use the subprocess + byte-string form, which is always correct regardless of CWD:

```bash
/opt/anaconda3/bin/python3 -c "
import os, subprocess
desktop = '/Users/clementmugenzi/Desktop'
real_name = b'Desktop - Clement\xe2\x80\x99s MacBook Pro'
real = os.path.join(desktop.encode(), real_name).decode()
project = os.path.join(real, 'Python', 'causal-eval-engine')
config  = os.path.join(project, 'config', 'CONFIG_FILE')   # e.g. study_one.yaml
script  = os.path.join(project, 'run_eval.py')
subprocess.run(['/opt/anaconda3/bin/python', script, '--config', config], cwd=project)
"
```

Replace `CONFIG_FILE` with `<config-name>.yaml` (add the `.yaml` if the user gave a bare name). Prefer this subprocess form for anything other than the default `study.yaml`.

## After the run

- Report whether it succeeded and surface any errors verbatim — do not claim success if the process errored.
- Point the user to the generated artifacts: `output/report.html` (self-contained) and `output/figures/`.
- If missing-value warnings or a weak-instrument warning appeared in the output, mention them.
