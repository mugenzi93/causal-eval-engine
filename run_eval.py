#!/usr/bin/env python3
"""
CLI entry point for the causal evaluation engine.

Usage:
    python run_eval.py --config config/study.yaml
"""
import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import ingest
from diagnostics import run_diagnostics
from dag import build_and_render
from estimators import REGISTRY
from sensitivity import run_sensitivity
from report import build_report


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Resolve relative data path against the project root (directory containing run_eval.py)
    project_root = Path(__file__).resolve().parent
    data_path = Path(config["data"]["path"])
    if not data_path.is_absolute():
        config["data"]["path"] = str(project_root / data_path)
    return config


def main():
    parser = argparse.ArgumentParser(description="Causal Eval Engine")
    parser.add_argument("--config", required=True, help="Path to study.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    # Run from the project root so output/ is created there
    import os
    os.chdir(Path(__file__).resolve().parent)
    print(f"[1/6] Loading and validating data from: {config['data']['path']}")
    df, miss, _ = ingest(config)

    if not miss.empty:
        print(f"      Missingness detected:\n{miss.to_string()}")

    print("[2/6] Running diagnostics (Table 1, love plot, density, ECDF)...")
    diagnostics = run_diagnostics(
        df,
        treatment=config["treatment"],
        covariates=config["covariates"],
        label="pre",
    )

    print("[3/6] Building causal DAG...")
    dag_info = build_and_render(config)
    print(f"      Adjustment set: {dag_info['adjustment_set']}")

    print("[4/6] Running estimators:", config["methods"])
    estimator_results = []
    for method in config.get("methods", []):
        if method not in REGISTRY:
            print(f"      WARNING: unknown method '{method}', skipping.")
            continue
        print(f"      → {method}")
        result = REGISTRY[method](df, config)
        estimator_results.append(result)

        # Regenerate love plot with matching weights after PSM
        if method == "psm":
            result.pop("propensity_scores", None)
            mw = result.pop("matched_weights", None)
            if mw is not None:
                post_diag = run_diagnostics(
                    df,
                    treatment=config["treatment"],
                    covariates=config["covariates"],
                    matching_weights=mw,
                    label="post_psm",
                )
                diagnostics["love_plot"] = post_diag["love_plot"]

    sensitivity_results = []
    if config.get("sensitivity", False):
        print("[5/6] Running sensitivity analysis (E-values)...")
        sensitivity_results = run_sensitivity(estimator_results)

    print("[6/6] Building HTML report...")
    report_path = build_report(config, diagnostics, dag_info, estimator_results, sensitivity_results)
    print(f"\nDone. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
