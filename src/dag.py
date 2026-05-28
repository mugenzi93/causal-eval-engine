import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def build_dag(treatment: str, outcome: str, covariates: list) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node(treatment)
    G.add_node(outcome)
    for cov in covariates:
        G.add_edge(cov, treatment)
        G.add_edge(cov, outcome)
    G.add_edge(treatment, outcome)
    return G


def adjustment_set(G: nx.DiGraph, treatment: str, outcome: str) -> list:
    """Return the minimal adjustment set: all ancestors of treatment or outcome
    that are not descendants of treatment (i.e., the covariates layer)."""
    ancestors_t = nx.ancestors(G, treatment)
    ancestors_o = nx.ancestors(G, outcome)
    descendants_t = nx.descendants(G, treatment)
    candidates = (ancestors_t | ancestors_o) - descendants_t - {treatment, outcome}
    return sorted(candidates)


def render_dag(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    save_path: Path | None = None,
) -> Path:
    pos = nx.spring_layout(G, seed=42)
    colors = []
    for node in G.nodes():
        if node == treatment:
            colors.append("#e07b54")
        elif node == outcome:
            colors.append("#4c72b0")
        else:
            colors.append("#b0b0b0")

    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors, node_size=1800,
        font_size=9, font_color="white", font_weight="bold",
        arrows=True, arrowsize=20, edge_color="#555",
    )
    ax.set_title("Causal DAG")
    ax.axis("off")
    plt.tight_layout()

    out = save_path or Path(__file__).resolve().parent.parent / "output" / "figures" / "dag.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def build_and_render(config: dict) -> dict:
    treatment = config["treatment"]
    outcome = config["outcome"]
    covariates = config["covariates"]

    G = build_dag(treatment, outcome, covariates)
    adj_set = adjustment_set(G, treatment, outcome)
    dag_path = render_dag(G, treatment, outcome)

    return {
        "graph": G,
        "adjustment_set": adj_set,
        "dag_figure": str(dag_path),
    }
