from __future__ import annotations

# Home page - Mini-Grid LV Toolkit.
# Three stacked sections (one per page) sharing the same visual format:
# tagline, "What it does", then Key inputs / Key outputs side by side.

from typing import List

import streamlit as st

from config.settings import PathManager

st.set_page_config(
    page_title="Mini-Grid LV Toolkit",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Shared section renderer: guarantees the same format for the 3 descriptions.
# Sections are stacked vertically so different lengths never break the layout.
# ---------------------------------------------------------------------------
def _section(
    *,
    number: int,
    title: str,
    tagline: str,
    what: str,
    inputs: List[str],
    outputs: List[str],
    page_path: str,
    page_label: str,
) -> None:
    st.header(f"{number}) {title}")
    st.markdown(f"*{tagline}*")
    st.markdown(what)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Key inputs**")
        st.markdown("\n".join(f"- {x}" for x in inputs))
    with c2:
        st.markdown("**Key outputs**")
        st.markdown("\n".join(f"- {x}" for x in outputs))

    try:
        st.page_link(page_path, label=page_label, icon="➡️")
    except Exception:
        st.caption(f"Open **{page_label}** from the sidebar.")

    st.divider()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Mini-Grid LV Toolkit")
st.markdown(
    """
    A lightweight, interactive toolkit for planning and checking **mini-grid
    distribution networks**, from customer locations to a validated hybrid
    MV/LV design. It is a **planning and assessment aid** — transparent,
    explainable and reproducible — not a detailed engineering design software.

    The workflow has three steps, one per page: design the LV topology,
    validate it with an AC power flow, and reinforce it with MV/LV
    transformers when a single LV network is not electrically feasible.
    Both validation and reinforcement include a **distribution cost analysis**
    based on unit costs from real mini-grid projects (DRC and Uganda).
    """
)


st.divider()

# ---------------------------------------------------------------------------
# 1) Grid Topology
# ---------------------------------------------------------------------------
_section(
    number=1,
    title="Grid Topology",
    tagline="Heuristic LV network design: poles, customer association, MST backbone.",
    what=(
        "Candidate poles are sampled along roads (or placed freely by clustering "
        "buildings), customers are associated to poles under a connection radius "
        "and a max-users-per-pole cap, and the poles are connected into a single "
        "radial LV network with a **Minimum Spanning Tree**. Spans longer than the "
        "engineering cap are subdivided with support poles. Isolated buildings can "
        "be left unconnected as **standalone candidates** using an **economic "
        "differential-cost criterion** — a cluster is connected only if its marginal "
        "connection cost stays below the budget `(c_standalone − c_generation) × E` "
        "per building — or the legacy minimum-cluster-size rule."
    ),
    inputs=[
        "Users file (`.gpkg` / `.xlsx`), optional roads file (`.gpkg`)",
        "Pole spacing, connection radius, max users per pole, max LV span",
        "Standalone criterion: economic (c_sa, c_gen, kWh/year, horizon) or topological",
    ],
    outputs=[
        "Summary metrics: backbone and service-drop length, poles, coverage",
        "Interactive map (served / standalone buildings, poles, LV network)",
        "GeoJSON nodes & edges, associations CSV, building-metadata template",
    ],
    page_path="pages/1_Grid_Topology.py",
    page_label="Open Grid Topology",
)

# ---------------------------------------------------------------------------
# 2) Grid Validation
# ---------------------------------------------------------------------------
_section(
    number=2,
    title="Grid Validation",
    tagline="Single-snapshot AC power flow (PyPSA) on the LV topology.",
    what=(
        "The LV topology — from the current session, imported from "
        "**OffGridPlanner** or **OMG (OnSSET for Mini-Grids)**, or uploaded "
        "manually — is turned into a PyPSA network. Hourly demand is aggregated "
        "to poles from building metadata and category load profiles, line "
        "parameters come from global defaults or a cable catalog, and a "
        "**Newton-Raphson power flow** checks voltages and line loadings. "
        "A collapsed **cost analysis** panel after the power flow maps the "
        "topology drivers onto distribution unit costs, with sliders for "
        "site-specific differential analysis."
    ),
    inputs=[
        "Topology source: session results, OffGridPlanner / OMG import, or manual files",
        "`building_metadata.csv` + `category_profiles.csv` (demand)",
        "Slack pole, voltage limits, line parameters (global or catalog)",
    ],
    outputs=[
        "Summary metrics and bus / line result tables with violations",
        "Voltage map and current map (interactive)",
        "Distribution cost breakdown (LV + last mile) with CSV download",
    ],
    page_path="pages/2_Grid_Validation.py",
    page_label="Open Grid Validation",
)

# ---------------------------------------------------------------------------
# 3) Grid Reinforcement
# ---------------------------------------------------------------------------
_section(
    number=3,
    title="Grid Reinforcement",
    tagline="Hybrid MV/LV design: k transformers, k+1 LV subnetworks, one power flow.",
    what=(
        "When a single LV network cannot meet the voltage-drop cap, the settlement "
        "is partitioned into **k+1 LV subnetworks** fed by **k MV/LV transformers** "
        "plus the plant. Two criteria drive the iteration on k: a **distance cap** "
        "(complete-linkage clustering on the cluster diameter) or a **voltage cap** "
        "(k-means with smart seeding and backtracking, verified by a power flow on "
        "the full hybrid network). Transformers are auto-sized on standard ratings, "
        "the MV backbone is routed as an MST over plant and transformer sites, and "
        "the results include per-subnetwork reports, three maps (connections, "
        "voltage, current) and a **cost analysis** with a differential comparison "
        "across the explored k values (LV-only vs hybrid designs)."
    ),
    inputs=[
        "Topology and demand: results from pages 1-2 or manual uploads",
        "Partition criterion (distance / voltage cap) and voltage-drop limit",
        "MV voltage (11 / 33 kV), transformer sizes and margins, max transformers",
    ],
    outputs=[
        "Iteration history and summary metrics (LV / MV poles, lengths, worst ΔV)",
        "Per-subnetwork detail and transformer summary",
        "Maps + distribution cost breakdown (LV, MV, transformers) per iteration",
    ],
    page_path="pages/3_Grid_Reinforcement.py",
    page_label="Open Grid Reinforcement",
)

st.caption(
    "Based on original work by Edoardo Silvestri, Marcello Fossa and Alessandro Onori — contact: "
    "alessandro.onori@polimi.it — EUPL v1.1."
)
