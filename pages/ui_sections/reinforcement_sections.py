from __future__ import annotations

"""
UI sections for the Grid Reinforcement page (hybrid MV/LV, MicroGridsPy only).

Rendering-only module: all decisions live in core.mv_reinforcement_service.
Maps 2 and 3 reuse the Grid Validation builders on a combined view of the
hybrid network, built with synthetic global pole ids gid = sid * 100000 + pid
(pole ids are only unique within a subnetwork). The MV backbone is drawn on
top of the returned folium maps: black in the overview map, dark orange
(#d95f02, ColorBrewer) in the current map to contrast with the purple LV
bands without vanishing on the OpenStreetMap basemap.
"""

from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import Element
from folium import CircleMarker, PolyLine
from streamlit_folium import st_folium

from core.costs import mv_pole_spacing_m
from core.pipeline_state import get_mv_reinforcement_request
from config.settings import (
    DEFAULT_MAX_ASSOCIATIONS,
    DEFAULT_SAMPLING_DISTANCE_M,
    DEFAULT_USER_DISTANCE_M,
    MV_DV_CAP_PU,
    MV_LINE_I_MAX_A,
    MV_LINE_R_OHM_PER_KM,
    MV_LINE_X_OHM_PER_KM,
    MV_MAX_CLUSTER_DIAMETER_M,
    MV_MAX_TRANSFORMERS,
    MV_TR_SIZING_MARGIN,
    MV_TR_STANDARD_SIZES_KVA,
    MV_TR_VSC_PCT,
    MV_TR_VSCR_PCT,
    MV_V_NOM_KV_DEFAULT,
    MV_V_NOM_KV_OPTIONS,
)
from core.contracts import (
    MvIterationResult,
    MvLineParams,
    MvReinforcementRequest,
    MvReinforcementResult,
    MvTopologyParams,
    MvTransformerParams,
    ValidationInputs,
)
from core.powerflow_io import make_map_lv_current_branches, make_map_lv_voltage_nodes

# subnet 0 (slack) first; up to MV_MAX_TRANSFORMERS + 1 entries
SUBNET_PALETTE = [
    "#1f77b4", "#2ca02c", "#e377c2", "#17becf", "#bcbd22",
    "#8c564b", "#ff7f0e", "#9467bd", "#7f7f7f", "#d62728", "#aec7e8",
]
MV_COLOR_OVERVIEW = "#000000"
MV_COLOR_CURRENT = "#d95f02"

_GID_BASE = 100_000


def _gid(sid: int, pid: int) -> int:
    return int(sid) * _GID_BASE + int(pid)


def _subnet_color(sid: int) -> str:
    return SUBNET_PALETTE[int(sid) % len(SUBNET_PALETTE)]


# =============================================================================
# Header and inputs
# =============================================================================
def render_page_header() -> None:
    st.title("Grid Reinforcement — hybrid MV/LV")
    st.markdown(
        "Use this page when the **pure-LV** results of *Grid Validation* are infeasible "
        "(voltage-drop violations). The settlement is partitioned into **k + 1 LV "
        "subnetworks** fed by an **MV backbone** through k step-down transformers; "
        "k is found iteratively against the selected criterion.  \n"
        "*Available for the MicroGridsPy workflow only.*"
    )
    st.markdown("---")


def render_topology_input_section(page1_available: bool) -> Dict[str, Any]:
    """Section '1) Upload topology data'.

    Two independent sources:
    - 'Use results from Grid Topology (Page 1)': inherits buildings, roads,
      geometric params AND the standalone (coverage) policy exactly as run
      on Page 1 (same TopologyResult.gdf_served_4326 / gdf_unserved_4326 —
      no re-derivation, so a Page-1 run with standalone enabled/disabled
      genuinely changes what Grid Reinforcement partitions).
    - 'Manually upload data': reuses topology_sections' own upload + params
      widgets VERBATIM (same functions, same keys as Page 1) so this section
      is, by construction, a live copy of Page 1 — including the standalone
      checkbox, shown here (unlike before) since standalone buildings are
      meaningful again via the one-shot pre-pass.
    """
    from pages.ui_sections.topology_sections import render_params_section as _t_params
    from pages.ui_sections.topology_sections import render_upload_section as _t_upload

    st.subheader("1) Upload topology data")
    source = st.radio(
        "Buildings / roads source",
        options=("Use results from Grid Topology (Page 1)", "Manually upload data"),
        index=(0 if page1_available else 1),
        key="gr_topo_source",
        help="Reuses the buildings, roads and topology parameters (including the "
             "standalone policy) exactly as run in Grid Topology during this session.",
    )
    use_page1 = source.startswith("Use results")

    users_file: Optional[Any] = None
    roads_file: Optional[Any] = None
    follow_roads_mode = "Free placement"
    manual_topo_params: Optional[Dict[str, Any]] = None

    if use_page1:
        if page1_available:
            st.success("Buildings, roads and topology parameters (standalone policy "
                       "included) will be taken from the Grid Topology results in "
                       "this session.")
        else:
            st.error("No Grid Topology results in this session: run Page 1 first, "
                     "or switch to **Manually upload data**.")
    else:
        users_file, follow_roads_mode, roads_file = _t_upload()
        manual_topo_params = _t_params(
            default_sampling_distance_m=int(DEFAULT_SAMPLING_DISTANCE_M),
            default_user_distance_m=int(DEFAULT_USER_DISTANCE_M),
            default_max_associations=int(DEFAULT_MAX_ASSOCIATIONS),
        )

    st.markdown("---")
    return {
        "use_page1": use_page1,
        "users_file": users_file,
        "roads_file": roads_file,
        "follow_roads_mode": follow_roads_mode,
        "manual_topo_params": manual_topo_params,
    }


def render_pf_input_section(page2_result: Optional[Any]) -> Dict[str, Any]:
    """Section '2) Power flow setup' — manual-upload path only shows widgets;
    the Page-2 path inherits everything (demand, voltage limits, cosφ,
    voltage-base convention, LV cable) from the Grid Validation run.
    """
    from pages.ui_sections.validation_sections import (
        render_demand_upload_section,
        render_electrical_assumptions_section,
        render_line_params_section,
    )

    st.subheader("2) Power flow setup")
    source = st.radio(
        "Power flow setup source",
        options=("Use results from Grid Validation (Page 2)", "Manually upload data"),
        index=(0 if page2_result is not None else 1),
        key="gr_pf_source",
        help="Reuses the demand data, voltage limits, power factor and LV cable "
             "exactly as run in Grid Validation during this session.",
    )
    use_page2 = source.startswith("Use results")

    meta_file: Optional[Any] = None
    profiles_file: Optional[Any] = None
    manual_pf_params: Optional[Dict[str, Any]] = None
    manual_line_params: Optional[Dict[str, Any]] = None
    manual_hour: Optional[Dict[str, Any]] = None

    if use_page2:
        if page2_result is not None:
            p = page2_result.params or {}
            st.success(
                f"Using Grid Validation inputs: V limits [{p.get('v_min_pu', 0.9):.2f}, "
                f"{p.get('v_max_pu', 1.1):.2f}] p.u., cosφ = {p.get('pf_load', 0.95):.2f}, "
                f"cable R = {p.get('r_ohm_per_km', 0.641):.3f} Ω/km, "
                f"hour = {page2_result.hour}."
            )
            st.caption("Note: if Grid Validation used a per-line cable catalog override, "
                       "that detail does not carry over — Grid Reinforcement applies a "
                       "single uniform LV cable, taken from Grid Validation's base parameters.")
        else:
            st.error("No Grid Validation results in this session: run Page 2 "
                     "(including the power flow) first, or switch to "
                     "**Manually upload data**.")
    else:
        meta_file, profiles_file = render_demand_upload_section()
        manual_pf_params = render_electrical_assumptions_section()
        manual_line_params = render_line_params_section()

        st.markdown("**Snapshot hour**")
        auto_hour = st.checkbox(
            "Use the system peak hour (auto)", value=True, key="gr_auto_hour",
            help="If unchecked, the snapshot is solved at the selected hour.",
        )
        hour = None if auto_hour else st.slider("Snapshot hour", 0, 23, 19, key="gr_hour")
        manual_hour = {"auto_hour": auto_hour, "hour": hour}

    st.markdown("---")
    return {
        "use_page2": use_page2,
        "meta_file": meta_file,
        "profiles_file": profiles_file,
        "manual_pf_params": manual_pf_params,
        "manual_line_params": manual_line_params,
        "manual_hour": manual_hour,
    }


def render_mv_cycle_section(gdf_buildings_preview_4326: Optional[gpd.GeoDataFrame]) -> Dict[str, Any]:
    """Section '3) MV iterative cycle setup': MV cables -> LV/MV transformers
    -> partition criterion -> slack (plant) position. The manual-coordinates
    default is the northmost building of the currently selected settlement
    (Page 1 or manually uploaded), computed once here and only used to seed
    the widget on its first render — subsequent edits by the user persist."""
    st.subheader("3) MV iterative cycle setup")

    with st.expander("MV cables (backbone conductor, bare overhead ACSR)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            r_mv = st.number_input("R [Ω/km]", 0.01, 5.0, float(MV_LINE_R_OHM_PER_KM),
                                   0.01, key="gr_r_mv")
        with c2:
            x_mv = st.number_input("X [Ω/km]", 0.0, 2.0, float(MV_LINE_X_OHM_PER_KM),
                                   0.01, key="gr_x_mv")
        with c3:
            i_mv = st.number_input("I max [A]", 10.0, 1000.0, float(MV_LINE_I_MAX_A),
                                   5.0, key="gr_i_mv")

    with st.expander("LV/MV transformers", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            mv_kv = st.selectbox(
                "MV nominal voltage [kV]",
                options=list(MV_V_NOM_KV_OPTIONS),
                index=list(MV_V_NOM_KV_OPTIONS).index(MV_V_NOM_KV_DEFAULT),
                key="gr_mv_kv",
            )
        with c2:
            auto_size = st.checkbox(
                "Auto-size on cluster peak", value=True, key="gr_tr_auto",
                help=f"Peak / pf × {MV_TR_SIZING_MARGIN} → next standard size "
                     f"{tuple(int(s) for s in MV_TR_STANDARD_SIZES_KVA)} kVA. "
                     "The plant step-up is sized on the system peak.",
            )
        s_tr = None
        if not auto_size:
            s_tr = st.selectbox(
                "Transformer size [kVA] (all transformers)",
                options=[float(s) for s in MV_TR_STANDARD_SIZES_KVA],
                index=3, key="gr_tr_kva",
            )
        c1, c2 = st.columns(2)
        with c1:
            vsc = st.number_input("uk — short-circuit voltage [%]", 1.0, 15.0,
                                  float(MV_TR_VSC_PCT), 0.1, key="gr_vsc")
        with c2:
            vscr = st.number_input("uk resistive component [%]", 0.1, 5.0,
                                   float(MV_TR_VSCR_PCT), 0.1, key="gr_vscr")
        st.caption("Tap ratio fixed at 1.0 for a fair comparison with the pure-LV baseline. "
                   "T model with the no-load branch neglected (g = b = 0).")

    st.markdown("---")
    st.subheader("Partition criterion (drives the k-iteration)")

    crit_label = st.radio(
        "Criterion",
        options=(
            "Voltage cap — every subnetwork must respect the max voltage drop",
            "Distance cap — every cluster diameter must respect the max distance",
        ),
        index=0,
        key="gr_criterion",
    )
    criterion = "voltage_cap" if crit_label.startswith("Voltage") else "distance_cap"

    c1, c2, c3 = st.columns(3)
    with c1:
        dv_cap_pct = st.number_input(
            "Voltage-drop cap [%]", min_value=1.0, max_value=20.0,
            value=float(MV_DV_CAP_PU * 100.0), step=0.5,
            disabled=(criterion != "voltage_cap"), key="gr_dv_cap",
        )
    with c2:
        diam_cap_m = st.number_input(
            "Max cluster diameter [m]", min_value=100.0, max_value=5000.0,
            value=float(MV_MAX_CLUSTER_DIAMETER_M), step=50.0,
            disabled=(criterion != "distance_cap"), key="gr_diam_cap",
        )
    with c3:
        max_tr = st.number_input(
            "Max transformers (safety stop)", min_value=1, max_value=30,
            value=int(MV_MAX_TRANSFORMERS), step=1, key="gr_max_tr",
        )

    default_lat, default_lon = 0.0, 0.0
    if gdf_buildings_preview_4326 is not None and len(gdf_buildings_preview_4326):
        # Buildings may be polygons (footprints) or points depending on the
        # source file. Centroids are computed in a projected (metric) CRS —
        # geopandas warns (correctly) that centroids on a geographic CRS are
        # distorted — then the single northmost point is converted back to
        # WGS84 for the lat/lon default.
        from core.distribution_io import derive_utm_epsg
        proj = gdf_buildings_preview_4326.to_crs(epsg=derive_utm_epsg(gdf_buildings_preview_4326))
        cent_proj = proj.geometry.centroid
        north_idx = cent_proj.y.idxmax()
        north_point_4326 = (
            gpd.GeoSeries([cent_proj.loc[north_idx]], crs=proj.crs).to_crs(epsg=4326).iloc[0]
        )
        default_lat = float(north_point_4326.y)
        default_lon = float(north_point_4326.x)

    plant_mode = st.radio(
        "Plant (slack) position",
        options=("Settlement centroid (auto)", "Manual coordinates"),
        index=0, key="gr_plant_mode", horizontal=True,
    )
    plant_lat = plant_lon = None
    if plant_mode.startswith("Manual"):
        st.caption(
            "Pre-filled with the northmost building of the current settlement "
            "on first use; edit freely afterwards."
        )
        c1, c2 = st.columns(2)
        with c1:
            plant_lat = st.number_input("Plant latitude", value=default_lat,
                                        format="%.6f", key="gr_plant_lat")
        with c2:
            plant_lon = st.number_input("Plant longitude", value=default_lon,
                                        format="%.6f", key="gr_plant_lon")

    st.markdown("---")
    return {
        "mv_r_ohm_per_km": float(r_mv), "mv_x_ohm_per_km": float(x_mv), "mv_i_max_a": float(i_mv),
        "mv_v_nom_kv": float(mv_kv), "tr_s_nom_kva": (None if auto_size else float(s_tr)),
        "vsc_pct": float(vsc), "vscr_pct": float(vscr),
        "clustering_criterion": criterion,
        "dv_cap_pu": float(dv_cap_pct) / 100.0,
        "max_cluster_diameter_m": float(diam_cap_m),
        "max_transformers": int(max_tr),
        "plant_auto": plant_mode.startswith("Settlement"),
        "plant_lat": plant_lat, "plant_lon": plant_lon,
    }


def render_run_controls(ready: bool) -> tuple[bool, bool]:
    c1, c2 = st.columns([1, 1])
    with c1:
        run = st.button("Run Grid Reinforcement", type="primary",
                        disabled=not ready, key="gr_run")
    with c2:
        clear = st.button("Clear results", key="gr_clear")
    if not ready:
        st.info("Resolve the topology and power-flow sources above (see the "
                "messages in sections 1 and 2) to enable the run.")
    return run, clear


def build_request(
    topo_geom: Dict[str, Any],
    mv_inputs: Dict[str, Any],
    pf_electrical: Dict[str, Any],
    cable_params: Dict[str, Any],
) -> MvReinforcementRequest:
    pf_params = ValidationInputs(
        schema_version=1,
        mode="mgp",
        gdf_nodes_4326=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        associations_df=pd.DataFrame(),
        pole_id_col="pole_id",
        center=(0.0, 0.0),
        selected_hour=pf_electrical.get("selected_hour"),
        v_min_pu=pf_electrical["v_min_pu"], v_max_pu=pf_electrical["v_max_pu"],
        pf_load=pf_electrical["pf_load"], v_nom_kv=pf_electrical.get("v_nom_kv", 0.4),
        r_ohm_per_km=cable_params["r_ohm_per_km"], x_ohm_per_km=cable_params["x_ohm_per_km"],
        s_nom_kva=cable_params["s_nom_kva"],
    )
    return MvReinforcementRequest(
        schema_version=1,
        topo_params=MvTopologyParams(
            follow_roads_mode="roads",
            road_pole_spacing_m=topo_geom["road_pole_spacing_m"],
            max_user_connection_radius_m=topo_geom["max_user_connection_radius_m"],
            max_users_per_pole=topo_geom["max_users_per_pole"],
            max_pole_span_m=topo_geom["max_pole_span_m"],
            clustering_criterion=mv_inputs["clustering_criterion"],
            max_cluster_diameter_m=mv_inputs["max_cluster_diameter_m"],
            max_transformers=mv_inputs["max_transformers"],
            allow_unserved_isolated=bool(topo_geom.get("allow_unserved_isolated", False)),
            min_cluster_size=int(topo_geom.get("min_cluster_size", 1)),
        ),
        transformer_params=MvTransformerParams(
            mv_v_nom_kv=mv_inputs["mv_v_nom_kv"],
            s_nom_kva=mv_inputs["tr_s_nom_kva"],
            vsc_pct=mv_inputs["vsc_pct"],
            vscr_pct=mv_inputs["vscr_pct"],
        ),
        mv_line_params=MvLineParams(
            r_ohm_per_km=mv_inputs["mv_r_ohm_per_km"],
            x_ohm_per_km=mv_inputs["mv_x_ohm_per_km"],
            i_max_a=mv_inputs["mv_i_max_a"],
        ),
        pf_params=pf_params,
    )


# =============================================================================
# Results
# =============================================================================
def render_results(result: MvReinforcementResult, dv_cap_pu: float) -> None:
    final = result.final
    views = _combined_views(final)
    standalone = result.gdf_standalone_4326
    st.markdown("---")
    st.header("Results")

    if final.converged and final.n_transformers == 0:
        st.success(
            "Max voltage drop respects the fixed limit with the **pure LV network**: "
            "no MV layer is needed. Eventual current issues may be fixed by changing "
            "cables in the Grid Validation section."
        )
    elif final.converged:
        st.success(
            f"Converged with **{final.n_transformers} transformer(s)** "
            f"({len(final.subnetworks)} subnetworks) — criterion: `{result.criterion}`."
        )
    else:
        st.warning(
            f"NOT converged within the transformer cap: showing the best attempt with "
            f"{final.n_transformers} transformer(s). Consider raising the cap or "
            f"revisiting the inputs."
        )
    if result.criterion == "distance_cap":
        n_viol = sum(1 for s in final.subnetworks if s.dv_cap_ok is False)
        if n_viol:
            st.warning(
                f"The distance criterion is met, but {n_viol} subnetwork(s) still violate "
                f"the {dv_cap_pu:.0%} voltage-drop cap — consider re-running with the "
                f"voltage-cap criterion."
            )

    _render_iteration_history(result)
    _render_overall_summary(result, views, standalone)
    _render_subnetwork_expanders(final, result.criterion)
    _render_maps(final, views, standalone)
    _render_downloads(result)


def _render_iteration_history(result: MvReinforcementResult) -> None:
    st.subheader("Iteration history")
    rows = []
    for i, it in enumerate(result.iterations):
        diam = (max(it.cluster_diameters_m.values())
                if it.cluster_diameters_m else np.nan)
        worst = (max((s.worst_dv_pu or np.nan) for s in it.subnetworks)
                 if it.subnetworks else np.nan)
        note = it.note or ""
        stage = "power flow" if it.pf_executed else "topology check"
        if note.startswith("diverged"):
            stage = "power flow (diverged)"
        rows.append({
            "iteration": i + 1,
            "k (transformers)": it.n_transformers,
            "stage": stage,
            "max cluster diameter [m]": (round(diam, 1) if np.isfinite(diam) else None),
            "worst ΔV [%]": (round(worst * 100.0, 2) if np.isfinite(worst) else None),
            "converged": bool(it.converged),
            "time [s]": round(it.solve_seconds, 2),
            "note": (note[:80] if note else None),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_overall_summary(
    result: MvReinforcementResult, views: Dict[str, Any],
    standalone: Optional[gpd.GeoDataFrame],
) -> None:
    final = result.final
    st.subheader("Summary metrics")
    tot_lv_km = sum(float(s.topology.metrics.get("total_network_length_km", 0.0))
                    for s in final.subnetworks)
    worst = max((s.worst_dv_pu or 0.0) for s in final.subnetworks)
    n_viol = sum(int(s.validation.summary.get("num_voltage_violations", 0))
                 for s in final.subnetworks if s.validation is not None)
    lv_poles = sum(int(s.topology.metrics.get("num_poles_total", 0))
                   for s in final.subnetworks)
    # MV poles: equispaced along the backbone (60 m @ 11 kV, 120 m @ 33 kV).
    _req = get_mv_reinforcement_request(st.session_state)
    _mv_kv = MV_V_NOM_KV_DEFAULT
    if _req is not None:
        try:
            _mv_kv = float(_req.transformer_params.mv_v_nom_kv)
        except Exception:
            pass
    _spacing_m = mv_pole_spacing_m(_mv_kv)
    mv_poles = (
        int(np.ceil(final.mv_backbone_length_km * 1000.0 / _spacing_m)) + 1
        if final.mv_backbone_length_km > 0
        else 0
    )
    n_standalone = 0 if standalone is None else len(standalone)
    c = st.columns(9)
    c[0].metric("Subnetworks", len(final.subnetworks))
    c[1].metric("Transformers", final.n_transformers)
    c[2].metric("LV poles", lv_poles)
    c[3].metric("MV poles", mv_poles,
                help=f"MV backbone / {_spacing_m:.0f} m spacing @ {_mv_kv:.0f} kV, plus one.")
    c[4].metric("MV backbone [km]", f"{final.mv_backbone_length_km:.2f}")
    c[5].metric("Total LV length [km]", f"{tot_lv_km:.2f}")
    c[6].metric("Worst ΔV [%]", f"{worst * 100.0:.2f}")
    c[7].metric("Voltage violations", n_viol)
    c[8].metric("Standalone candidates", n_standalone,
                help="Buildings excluded before partitioning by the standalone "
                     "(coverage) policy — shown as red points on the overview map.")
    st.caption(f"Total solve time: {result.total_solve_seconds:.1f} s "
               f"({len(result.iterations)} iterations).")

    st.markdown("**Summary metrics — total network**")
    _render_total_network_summary(final, views)

    if final.transformer_summary is not None and len(final.transformer_summary):
        with st.expander("Transformer summary", expanded=False):
            cols = ["transformer", "mv_bus", "s_nom_kva", "s0_kVA",
                    "loading_pct", "v_mv_pu", "v_lv_pu", "dv_internal_pu"]
            df = final.transformer_summary
            st.dataframe(df[[c for c in cols if c in df.columns]].round(4),
                         use_container_width=True, hide_index=True)


def _render_subnetwork_expanders(final: MvIterationResult, criterion: str) -> None:
    st.subheader("Per-subnetwork detail")
    diams = final.cluster_diameters_m or {}
    for s in sorted(final.subnetworks, key=lambda x: x.subnet_id):
        m = s.topology.metrics
        label = (f"Subnetwork {s.subnet_id} — "
                 f"{'plant / slack' if s.root_kind == 'slack' else f'transformer {s.subnet_id}'}")
        with st.expander(label, expanded=False):
            st.markdown("**Topology**")
            c = st.columns(4)
            c[0].metric("Total network length [km]", f"{m.get('total_network_length_km', 0.0):.2f}")
            c[1].metric("LV backbone length [km]", f"{m.get('backbone_length_km', 0.0):.2f}")
            c[2].metric("Service drop length [km]", f"{m.get('service_drop_length_km', 0.0):.2f}")
            diam = diams.get(int(s.subnet_id))
            c[3].metric(
                "Subnetwork diameter [m]",
                f"{diam:.0f}" if diam is not None else "n/a",
                help="Maximum distance between any two buildings of this subnetwork. "
                     "Constrained to stay below the cap only under the distance criterion.",
                delta=(None if (diam is None or criterion != "distance_cap") else "≤ cap"),
            )
            st.caption("Number of poles in the LV network. Serving poles supply buildings "
                       "directly, while support poles are added only to limit span lengths.")
            c = st.columns(3)
            c[0].metric("Total poles", int(m.get("num_poles_total", 0)))
            c[1].metric("Serving poles", int(m.get("num_poles_serving", 0)))
            c[2].metric("Support poles", int(m.get("num_poles_support", 0)))
            st.caption("Coverage of the settlement by this subnetwork "
                       "(no standalone candidates in hybrid MV/LV mode).")
            c = st.columns(2)
            c[0].metric("Total buildings", int(m.get("num_buildings", 0)))
            c[1].metric("Grid-served buildings", int(m.get("num_served", 0)))

            st.markdown("**Power flow**")
            summ = s.validation.summary if s.validation is not None else {}
            c = st.columns(3)
            c[0].metric("Worst ΔV [%]",
                        f"{(s.worst_dv_pu or 0.0) * 100.0:.2f}",
                        delta=("OK" if s.dv_cap_ok else "cap violated"),
                        delta_color=("normal" if s.dv_cap_ok else "inverse"))
            c[1].metric("Voltage violations", int(summ.get("num_voltage_violations", 0)))
            max_i_a = None
            if s.validation is not None and "I_A" in s.validation.line_results.columns:
                _ia = pd.to_numeric(s.validation.line_results["I_A"], errors="coerce")
                if _ia.notna().any():
                    max_i_a = float(_ia.max())
            c[2].metric("Max line current [A]", f"{max_i_a:.1f}" if max_i_a is not None else "n/a")
            c = st.columns(3)
            c[0].metric("Peak load [kW]", f"{s.peak_load_kw:.1f}")
            if s.root_kind == "transformer":
                c[1].metric("Transformer [kVA]",
                            f"{s.tr_s_nom_kva:.0f}" if s.tr_s_nom_kva else "n/a")
                c[2].metric("Transformer loading [%]",
                            f"{s.tr_loading_pct:.1f}" if s.tr_loading_pct is not None else "n/a")


# =============================================================================
# Maps
# =============================================================================
def _combined_views(final: MvIterationResult) -> Dict[str, Any]:
    poles_recs: List[Dict[str, Any]] = []
    edges_latlon: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    bus_v_pu: Dict[int, float] = {}
    edge_recs: List[Dict[str, Any]] = []
    loading: Dict[Tuple[int, int], float] = {}
    snom: Dict[Tuple[int, int], float] = {}
    slack_gid: Optional[int] = None

    for s in final.subnetworks:
        sid = int(s.subnet_id)
        gdfp = s.topology.gdf_poles_4326
        geom_by_pid = {}
        for _, r in gdfp.iterrows():
            pid = int(r["pole_id"])
            g = r.geometry
            if g is None or g.is_empty:
                continue
            geom_by_pid[pid] = g
            poles_recs.append({"pole_id": _gid(sid, pid), "geometry": g})
        edges_latlon.extend(s.topology.mst_edges_latlon or [])

        if s.root_kind == "slack":
            root = int(s.validation.summary.get("slack_bus")) if s.validation else None
            if root is not None:
                slack_gid = _gid(sid, root)

        if s.validation is None:
            continue
        for _, r in s.validation.bus_results.iterrows():
            bus_v_pu[_gid(sid, int(r["bus"]))] = float(r["v_pu"])
        for _, r in s.validation.line_results.iterrows():
            try:
                u, v = int(r["bus0"]), int(r["bus1"])
            except (TypeError, ValueError):
                continue
            gu, gv = geom_by_pid.get(u), geom_by_pid.get(v)
            if gu is None or gv is None:
                continue
            gid_u, gid_v = _gid(sid, u), _gid(sid, v)
            edge_recs.append({"u": gid_u, "v": gid_v, "geometry": _linestring(gu, gv)})
            lp = r.get("loading_pu")
            loading[(gid_u, gid_v)] = float(lp) if pd.notna(lp) else np.nan
            sk = r.get("s_nom_kva")
            if pd.notna(sk):
                snom[(gid_u, gid_v)] = float(sk)

    poles_gdf = gpd.GeoDataFrame(poles_recs, crs="EPSG:4326")
    edges_gdf = gpd.GeoDataFrame(edge_recs, crs="EPSG:4326") if edge_recs else None
    lats = [g.y for g in poles_gdf.geometry]
    lons = [g.x for g in poles_gdf.geometry]
    center = (float(np.mean(lats)), float(np.mean(lons)))
    return {
        "poles_gdf": poles_gdf, "edges_latlon": edges_latlon, "bus_v_pu": bus_v_pu,
        "edges_gdf": edges_gdf, "loading": loading, "snom": snom,
        "center": center, "slack_gid": slack_gid,
    }


def _linestring(g1, g2):
    from shapely.geometry import LineString
    return LineString([(g1.x, g1.y), (g2.x, g2.y)])


def _fix_leaflet_sizing(m: folium.Map) -> None:
    """Force Leaflet to recompute its tile grid after the Streamlit iframe
    reaches its final size. Without this, tiles are only requested for the
    viewport measured at init time and late-laid-out regions stay grey
    (the classic bottom/left grey rectangle in streamlit-folium)."""
    name = m.get_name()
    # The injected element can be rendered BEFORE the map-init script, so the
    # map variable must only be dereferenced at window 'load' / event time.
    m.get_root().script.add_child(Element(
        f"window.addEventListener('load', function() {{\n"
        f"  setTimeout(function() {{ {name}.invalidateSize(true); }}, 250);\n"
        f"  setTimeout(function() {{ {name}.invalidateSize(true); }}, 1000);\n"
        f"}});\n"
        f"window.addEventListener('resize', function() {{\n"
        f"  if (typeof {name} !== 'undefined') {{ {name}.invalidateSize(true); }}\n"
        f"}});"
    ))


def _ensure_osm_basemap(m: folium.Map) -> None:
    """Force an OpenStreetMap tile layer on the map (defensive: some folium /
    streamlit-folium version combinations render an empty basemap when the
    map carries many child elements)."""
    folium.TileLayer("OpenStreetMap", control=False).add_to(m)


def _render_total_network_summary(final: MvIterationResult, views: Dict[str, Any]) -> None:
    """Reuse the Grid Validation 'Summary metrics' (B1/B2 framework) on the
    combined hybrid network: per-subnet bus/line results are concatenated on
    the synthetic global ids so slack distances and loading stats span the
    whole network."""
    from pages.ui_sections.validation_sections import _render_summary_metrics

    bus_frames, line_frames = [], []
    for s in final.subnetworks:
        if s.validation is None:
            continue
        sid = int(s.subnet_id)
        b = s.validation.bus_results.copy()
        b["bus"] = b["bus"].astype(int).map(lambda p, _sid=sid: _gid(_sid, p))
        bus_frames.append(b)
        l = s.validation.line_results.copy()
        l["bus0"] = pd.to_numeric(l["bus0"], errors="coerce").astype("Int64").map(
            lambda p, _sid=sid: _gid(_sid, int(p)) if pd.notna(p) else None)
        l["bus1"] = pd.to_numeric(l["bus1"], errors="coerce").astype("Int64").map(
            lambda p, _sid=sid: _gid(_sid, int(p)) if pd.notna(p) else None)
        line_frames.append(l)
    if not bus_frames or not line_frames:
        st.info("Summary metrics not available for this result.")
        return

    res = {
        "bus_results": pd.concat(bus_frames, ignore_index=True),
        "line_results": pd.concat(line_frames, ignore_index=True),
        "params": {"v_nom_kv": 0.4},  # LV nominal voltage for I_nom ratings
    }
    pf_map = {
        "line_loading_pu": views["loading"],
        "gdf_poles_4326": views["poles_gdf"],
        "gdf_edges_4326": views["edges_gdf"],
        "slack_pole_id": views["slack_gid"],
    }
    _render_summary_metrics(res, pf_map)
    st.caption(
        "Note: the ΔV-per-km-to-slack metric is defined along the LV graph only; "
        "it reads n/a when the worst node lies beyond a transformer (the electrical "
        "path crosses the MV layer)."
    )


def _add_mv_overlay(m: folium.Map, final: MvIterationResult, *,
                    color: str, weight: float, show_current: bool = False) -> None:
    gdf = final.mv_backbone_edges_4326
    if gdf is None or gdf.empty:
        return
    for _, r in gdf.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        if show_current and pd.notna(r.get("I_A")):
            tip = f"MV {r.get('line')} — ~{float(r['I_A']):.1f} A ({float(r.get('length_km', 0.0)):.2f} km)"
        else:
            tip = f"MV {r.get('line')} ({float(r.get('length_km', 0.0)):.2f} km)"
        PolyLine([(lat, lon) for lon, lat in geom.coords],
                 color=color, weight=weight, opacity=0.95, tooltip=tip).add_to(m)


def _mv_site_markers(m: folium.Map, final: MvIterationResult) -> None:
    """Plant + transformer positions in black (Figure 26 style)."""
    tr_kva = {int(s.subnet_id): s.tr_s_nom_kva for s in final.subnetworks
              if s.root_kind == "transformer"}
    for s in final.subnetworks:
        gdfp = s.topology.gdf_poles_4326.set_index("pole_id")
        root = (int(s.validation.summary.get("slack_bus"))
                if s.validation is not None else None)
        if root is None or root not in gdfp.index:
            continue
        g = gdfp.loc[root].geometry
        if s.root_kind == "slack":
            tip = "Plant (slack)"
        else:
            kva = tr_kva.get(int(s.subnet_id))
            tip = f"Transformer {s.subnet_id}" + (f" — {kva:.0f} kVA" if kva else "")
        CircleMarker(location=[float(g.y), float(g.x)], radius=6, color="#000000",
                     weight=1, fill=True, fill_color="#000000", fill_opacity=1.0,
                     tooltip=tip).add_to(m)


def _render_maps(
    final: MvIterationResult, views: Dict[str, Any],
    standalone: Optional[gpd.GeoDataFrame],
) -> None:
    center, slack_gid = views["center"], views["slack_gid"]

    # ---------------- Map 1: overview (pre-PF, Figure 26 style) ----------------
    st.subheader("Overview map — subnetworks and MV backbone")
    st.caption(
        "Buildings colored by their subnetwork (transformer service area); LV branches thin, "
        "same color; **MV backbone and transformer/plant positions in black**."
    )
    hl_options: List[Optional[Tuple[int, int]]] = [None]
    for s in sorted(final.subnetworks, key=lambda x: x.subnet_id):
        for pid in sorted(int(p) for p in s.topology.gdf_poles_4326["pole_id"].tolist()):
            hl_options.append((int(s.subnet_id), pid))
    hl_enabled = st.checkbox("Highlight a pole on the map", value=False, key="gr_hl_enabled")
    hl = st.selectbox(
        "Pole to highlight",
        options=hl_options,
        format_func=lambda x: "None" if x is None else f"Subnetwork {x[0]} — pole {x[1]}",
        disabled=not hl_enabled, key="gr_hl_pole",
    )
    hl_latlon = None
    if hl_enabled and hl is not None:
        sid, pid = hl
        s = [x for x in final.subnetworks if x.subnet_id == sid][0]
        gdfp = s.topology.gdf_poles_4326.set_index("pole_id")
        if pid in gdfp.index:
            g = gdfp.loc[pid].geometry
            hl_latlon = (float(g.y), float(g.x))

    m1 = folium.Map(
        location=list(hl_latlon or center),
        zoom_start=(18 if hl_latlon else 15),
        tiles="OpenStreetMap", control_scale=True,
    )
    for s in sorted(final.subnetworks, key=lambda x: x.subnet_id):
        color = _subnet_color(s.subnet_id)
        for (lat1, lon1), (lat2, lon2) in (s.topology.mst_edges_latlon or []):
            PolyLine([(lat1, lon1), (lat2, lon2)], color=color,
                     weight=2, opacity=0.8).add_to(m1)
        for _, r in s.topology.gdf_buildings_4326.iterrows():
            g = r.geometry
            if g is None or g.is_empty:
                continue
            p = g if g.geom_type == "Point" else g.representative_point()
            CircleMarker(location=[float(p.y), float(p.x)], radius=2, color=color,
                         weight=1, fill=True, fill_color=color, fill_opacity=0.9).add_to(m1)
    _ensure_osm_basemap(m1)
    _fix_leaflet_sizing(m1)
    _add_mv_overlay(m1, final, color=MV_COLOR_OVERVIEW, weight=4)
    _mv_site_markers(m1, final)
    if standalone is not None and len(standalone):
        for _, r in standalone.iterrows():
            g = r.geometry
            if g is None or g.is_empty:
                continue
            p = g if g.geom_type == "Point" else g.representative_point()
            CircleMarker(location=[float(p.y), float(p.x)], radius=3, color="#d62728",
                         weight=1, fill=True, fill_color="#d62728", fill_opacity=0.9,
                         tooltip="Standalone candidate (excluded before partitioning)").add_to(m1)
    if hl_latlon is not None:
        CircleMarker(location=list(hl_latlon), radius=11, color="#FF3333", weight=3,
                     fill=False, tooltip=f"Highlighted: subnetwork {hl[0]}, pole {hl[1]}").add_to(m1)
    legend = " · ".join(
        f"S{s.subnet_id}" for s in sorted(final.subnetworks, key=lambda x: x.subnet_id)
    )
    st_folium(m1, height=650, width=1300,
              key=f"gr_map1_{final.n_transformers}_{hl}")
    st.caption(f"Subnetworks on the map: {legend} (colors in subnetwork order).")

    # ---------------- Map 2: voltage ----------------
    st.subheader("Voltage map")
    st.caption("Nodes colored by voltage drop (green ≤ 10% < orange ≤ 20% < red); "
               "purple = slack pole. MV backbone in dark gray for context.")
    m2 = make_map_lv_voltage_nodes(
        center=center,
        gdf_poles_4326=views["poles_gdf"],
        pole_id_col="pole_id",
        mst_edges_latlon=views["edges_latlon"],
        zoom_start=15,
        slack_pole_id=slack_gid,
        bus_v_pu=views["bus_v_pu"],
    )
    _ensure_osm_basemap(m2)
    _fix_leaflet_sizing(m2)
    _add_mv_overlay(m2, final, color="#555555", weight=3)
    st_folium(m2, height=650, width=1300,
              key=f"gr_map2_{final.n_transformers}")

    # ---------------- Map 3: current ----------------
    st.subheader("Current map")
    st.caption("LV branches colored by estimated current (purple bands, red = overcurrent); "
               "**MV backbone in dark orange** for contrast.")
    m3 = make_map_lv_current_branches(
        center=center,
        gdf_poles_4326=views["poles_gdf"],
        pole_id_col="pole_id",
        gdf_edges_4326=views["edges_gdf"],
        zoom_start=15,
        slack_pole_id=slack_gid,
        line_loading_pu=views["loading"],
        line_s_nom_kva=views["snom"],
        v_nom_kv=0.4,
    )
    _ensure_osm_basemap(m3)
    _fix_leaflet_sizing(m3)
    _add_mv_overlay(m3, final, color=MV_COLOR_CURRENT, weight=5, show_current=True)
    st_folium(m3, height=650, width=1300,
              key=f"gr_map3_{final.n_transformers}")


def _render_downloads(result: MvReinforcementResult) -> None:
    final = result.final
    with st.expander("Downloads", expanded=False):
        rows = []
        for i, it in enumerate(result.iterations):
            diam = max(it.cluster_diameters_m.values()) if it.cluster_diameters_m else np.nan
            worst = (max((s.worst_dv_pu or np.nan) for s in it.subnetworks)
                     if it.subnetworks else np.nan)
            rows.append({"iteration": i + 1, "k": it.n_transformers,
                         "pf_executed": it.pf_executed, "converged": it.converged,
                         "max_cluster_diameter_m": diam, "worst_dv_pu": worst,
                         "solve_seconds": it.solve_seconds})
        st.download_button(
            "Iteration history (CSV)",
            pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
            file_name="gr_iteration_history.csv", mime="text/csv", key="gr_dl_hist",
        )
        bus_frames, line_frames = [], []
        for s in final.subnetworks:
            if s.validation is None:
                continue
            b = s.validation.bus_results.copy(); b.insert(0, "subnet_id", s.subnet_id)
            l = s.validation.line_results.copy(); l.insert(0, "subnet_id", s.subnet_id)
            bus_frames.append(b); line_frames.append(l)
        if bus_frames:
            st.download_button(
                "Bus results — all subnetworks (CSV)",
                pd.concat(bus_frames, ignore_index=True).to_csv(index=False).encode("utf-8"),
                file_name="gr_bus_results.csv", mime="text/csv", key="gr_dl_bus",
            )
        if line_frames:
            st.download_button(
                "Line results — all subnetworks (CSV)",
                pd.concat(line_frames, ignore_index=True).to_csv(index=False).encode("utf-8"),
                file_name="gr_line_results.csv", mime="text/csv", key="gr_dl_line",
            )
        if final.transformer_summary is not None and len(final.transformer_summary):
            st.download_button(
                "Transformer summary (CSV)",
                final.transformer_summary.drop(columns=[c for c in ["geometry"]
                                                        if c in final.transformer_summary],
                                               errors="ignore")
                .to_csv(index=False).encode("utf-8"),
                file_name="gr_transformers.csv", mime="text/csv", key="gr_dl_tr",
            )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            """
            1. **Choose the topology source** (Grid Distribution results or manual upload)  
            2. **Choose the power-flow source** (Grid Validation results or manual inputs)  
            3. Set the **MV cycle parameters**: partition criterion, voltage-drop cap, MV voltage, max transformers, plant location  
            4. Click **Run Grid Reinforcement**  
            5. Review **iteration history**, summary metrics, per-subnetwork detail and the three maps  
            6. Open **Cost analysis** for the LV + MV + transformer cost breakdown and the comparison across k
            """
        )
        st.markdown("---")
        st.markdown("**Example data**: see the `examples/` folder in this project.")
