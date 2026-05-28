import base64
from pathlib import Path
from datetime import datetime
import jinja2
import pandas as pd


TEMPLATE_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _img_to_b64(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode()


def _df_to_html(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"
    return df.to_html(index=False, classes="table", border=0)


def build_report(
    config: dict,
    diagnostics: dict,
    dag_info: dict,
    estimator_results: list[dict],
    sensitivity_results: list[dict],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    env.filters["b64img"] = lambda path: _img_to_b64(path) or ""
    template = env.get_template("report.html.j2")

    # Encode images as base64 for self-contained HTML
    love_b64 = _img_to_b64(diagnostics.get("love_plot"))
    dag_b64 = _img_to_b64(dag_info.get("dag_figure"))

    density_b64 = [
        _img_to_b64(p) for p in diagnostics.get("density_plots", []) if _img_to_b64(p)
    ]
    ecdf_b64 = [
        _img_to_b64(p) for p in diagnostics.get("ecdf_plots", []) if _img_to_b64(p)
    ]

    event_study_b64 = None
    for r in estimator_results:
        if r.get("method") == "did" and r.get("event_study_plot"):
            event_study_b64 = _img_to_b64(r["event_study_plot"])

    rendered = template.render(
        title="Causal Evaluation Report",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        config=config,
        table_one_html=_df_to_html(diagnostics.get("table_one")),
        love_plot_b64=love_b64,
        dag_b64=dag_b64,
        density_plots_b64=density_b64,
        ecdf_plots_b64=ecdf_b64,
        event_study_b64=event_study_b64,
        estimator_results=estimator_results,
        sensitivity_results=sensitivity_results,
        adjustment_set=dag_info.get("adjustment_set", []),
    )

    out = OUTPUT_DIR / "report.html"
    out.write_text(rendered, encoding="utf-8")
    return out
