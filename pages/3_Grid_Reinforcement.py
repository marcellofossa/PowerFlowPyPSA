from __future__ import annotations

# Page 3 — Grid Reinforcement (hybrid MV/LV, MicroGridsPy workflow only).
# Thin orchestration: inputs -> core.mv_reinforcement_service.run_grid_reinforcement
# -> results rendering. All UI widgets live in pages/ui_sections/reinforcement_sections.py.
# NOTE: kept as comments (not a bare string) because Streamlit "magic" would
# render a module-level string literal at the top of the page.

import numpy as np
import streamlit as st

from core.distribution_io import derive_utm_epsg, load_and_transform_data
from core.mv_reinforcement_service import classify_standalone, run_grid_reinforcement
from core.pipeline_state import (
    clear_mv_reinforcement_state,
    ensure_session_domains,
    get_mv_reinforcement_fingerprint,
    get_mv_reinforcement_result,
    get_topology_inputs,
    get_topology_result,
    get_validation_demand,
    get_validation_result,
    set_mv_reinforcement_fingerprint,
    set_mv_reinforcement_request,
    set_mv_reinforcement_result,
)
from core.powerflow_io import read_building_metadata_csv, read_category_profiles_csv
from pages.ui_sections.cost_sections import render_reinforcement_cost_section
from pages.ui_sections.reinforcement_sections import (
    build_request,
    render_mv_cycle_section,
    render_page_header,
    render_pf_input_section,
    render_results,
    render_run_controls,
    render_sidebar,
    render_topology_input_section,
)

st.set_page_config(page_title="Grid Reinforcement", layout="wide")
ensure_session_domains(st.session_state)


def _seek0(f):
    """Defensive rewind: an UploadedFile's cursor persists across reruns of
    the same script, so reading it more than once (a preview parse here, then
    the real parse in the Run block) silently returns nothing the second
    time unless we rewind first."""
    try:
        f.seek(0)
    except Exception:
        pass
    return f


def _load_buildings_preview_4326(users_file):
    """Cheap WGS84 parse of the uploaded buildings file, used only to seed
    the northmost-point default for the manual slack coordinates. Returns
    None if there is no file yet or it fails to parse."""
    if users_file is None:
        return None
    try:
        _seek0(users_file)
        gdf = load_and_transform_data(users_file, target_crs=4326)
        _seek0(users_file)
        return gdf
    except Exception:
        return None


render_page_header()
render_sidebar()

# ---------------------------------------------------------------------------
# Sections 1-3
# ---------------------------------------------------------------------------
page1_result = get_topology_result(st.session_state)
page1_inputs = get_topology_inputs(st.session_state)
page2_result = get_validation_result(st.session_state)
p2_meta, p2_profiles = get_validation_demand(st.session_state)

topo_inputs = render_topology_input_section(page1_available=(page1_result is not None))

if topo_inputs["use_page1"]:
    buildings_preview = page1_result.gdf_buildings_4326 if page1_result is not None else None
else:
    buildings_preview = _load_buildings_preview_4326(topo_inputs["users_file"])

pf_inputs = render_pf_input_section(page2_result)
mv_inputs = render_mv_cycle_section(gdf_buildings_preview_4326=buildings_preview)

# ---------------------------------------------------------------------------
# Readiness — two independent sources, each with its own alert (already
# rendered inside the section functions when the chosen source is missing)
# ---------------------------------------------------------------------------
if topo_inputs["use_page1"]:
    topo_ready = page1_result is not None
else:
    needs_roads = topo_inputs["follow_roads_mode"].startswith("Follow roads")
    topo_ready = (
        topo_inputs["users_file"] is not None
        and (topo_inputs["roads_file"] is not None or not needs_roads)
    )

if pf_inputs["use_page2"]:
    pf_ready = page2_result is not None
else:
    pf_ready = bool(pf_inputs["meta_file"]) and bool(pf_inputs["profiles_file"])

ready = topo_ready and pf_ready

# ---------------------------------------------------------------------------
# Three-way logic from the Grid Validation state (voltage criterion only):
# - page 2 ran and is feasible  -> warning, no run
# - page 2 ran and is infeasible -> start from max(1, seed): skip_k0=True
# - page 2 not run               -> iteration 0 is the pure-LV network
# ---------------------------------------------------------------------------
dv_cap_pu = float(mv_inputs["dv_cap_pu"])
page2_feasible = None
page2_worst_dv = None
# The Grid Validation result speaks for THIS settlement only when the user
# declared Page 2 as the power-flow source. With manual PF inputs, a stored
# GV result may belong to a different settlement (e.g. GV run on Mafa, GR
# run manually on Keana), so it must not drive the feasibility bypass.
if pf_inputs["use_page2"] and page2_result is not None:
    v_min = (page2_result.summary or {}).get("v_min_pu_observed")
    if v_min is not None:
        page2_worst_dv = float(1.0 - float(v_min))
        page2_feasible = bool(page2_worst_dv <= dv_cap_pu)

if mv_inputs["clustering_criterion"] == "voltage_cap":
    if page2_feasible is True:
        st.info(
            f"Grid Validation (pure LV) reports a max voltage drop of "
            f"{page2_worst_dv * 100.0:.2f}% — within the {dv_cap_pu * 100.0:.0f}% cap."
        )
    elif page2_feasible is False:
        st.caption(
            f"Grid Validation (pure LV) is infeasible "
            f"(max ΔV = {page2_worst_dv * 100.0:.2f}% > {dv_cap_pu * 100.0:.0f}%): "
            "the iteration will start directly from the hybrid network (k ≥ 1)."
        )
    elif pf_inputs["use_page2"]:
        st.caption(
            "Grid Validation has not been run in this session: iteration 0 will "
            "test the pure-LV network first."
        )
    else:
        st.caption(
            "Manual power-flow inputs: any Grid Validation result in this session "
            "is ignored for the feasibility bypass — iteration 0 will test the "
            "pure-LV network first."
        )

run_clicked, clear_clicked = render_run_controls(ready)

if clear_clicked:
    clear_mv_reinforcement_state(st.session_state)
    st.success("Grid Reinforcement results cleared.")


def _fingerprint() -> tuple:
    return (
        topo_inputs["use_page1"],
        getattr(topo_inputs["users_file"], "name", None),
        getattr(topo_inputs["roads_file"], "name", None),
        tuple(sorted((topo_inputs["manual_topo_params"] or {}).items())),
        pf_inputs["use_page2"],
        getattr(pf_inputs["meta_file"], "name", None),
        getattr(pf_inputs["profiles_file"], "name", None),
        tuple(sorted((pf_inputs["manual_pf_params"] or {}).items())),
        tuple(sorted((pf_inputs["manual_line_params"] or {}).items())),
        tuple(sorted((pf_inputs["manual_hour"] or {}).items())),
        tuple(sorted((k, v) for k, v in mv_inputs.items())),
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_clicked and ready:
    if mv_inputs["clustering_criterion"] == "voltage_cap" and page2_feasible is True:
        st.warning(
            "Max voltage drop respects the fixed limit — eventual current issues "
            "may be fixed by changing cables in the Grid Validation section. "
            "No hybrid MV/LV network is needed for this settlement."
        )
    else:
        progress = st.empty()

        def _say(msg: str) -> None:
            progress.info(msg)

        try:
            with st.spinner("Running Grid Reinforcement (this may take several minutes)..."):
                _say("Loading input data...")

                # ---- topology + standalone split ----
                if topo_inputs["use_page1"]:
                    b4326 = page1_result.gdf_buildings_4326
                    utm = derive_utm_epsg(b4326)
                    gdf_served = page1_result.gdf_served_4326.to_crs(epsg=utm)
                    gdf_standalone = page1_result.gdf_unserved_4326
                    r4326 = page1_result.gdf_roads_4326
                    gdf_roads = (
                        r4326.to_crs(epsg=utm) if (r4326 is not None and len(r4326)) else None
                    )
                    if page1_inputs is None:
                        st.warning(
                            "Grid Topology inputs were not found in this session "
                            "(older run?) — using default geometric parameters."
                        )
                    geom_params = (page1_inputs or {}).get("params", {})
                    topo_geom = {
                        "road_pole_spacing_m": geom_params.get("road_pole_spacing_m", 40.0),
                        "max_user_connection_radius_m": geom_params.get(
                            "max_user_connection_radius_m", 35.0),
                        "max_users_per_pole": geom_params.get("max_users_per_pole", 16),
                        "max_pole_span_m": geom_params.get("max_pole_span_m", 40.0),
                        "allow_unserved_isolated": geom_params.get(
                            "allow_unserved_isolated", False),
                        "min_cluster_size": geom_params.get("min_cluster_size", 1),
                    }
                else:
                    _seek0(topo_inputs["users_file"])
                    gdf_buildings_raw = load_and_transform_data(topo_inputs["users_file"])
                    if gdf_buildings_raw is None or gdf_buildings_raw.empty:
                        raise ValueError("Buildings file could not be loaded or is empty.")
                    needs_roads = topo_inputs["follow_roads_mode"].startswith("Follow roads")
                    gdf_roads = None
                    if needs_roads and topo_inputs["roads_file"] is not None:
                        _seek0(topo_inputs["roads_file"])
                        gdf_roads = load_and_transform_data(topo_inputs["roads_file"])

                    topo_geom = topo_inputs["manual_topo_params"]
                    if topo_geom["allow_unserved_isolated"]:
                        gdf_served, gdf_standalone_proj = classify_standalone(
                            gdf_buildings_raw, gdf_roads,
                            road_pole_spacing_m=topo_geom["road_pole_spacing_m"],
                            max_user_connection_radius_m=topo_geom["max_user_connection_radius_m"],
                            max_users_per_pole=topo_geom["max_users_per_pole"],
                            max_pole_span_m=topo_geom["max_pole_span_m"],
                            min_cluster_size=topo_geom["min_cluster_size"],
                        )
                        gdf_standalone = gdf_standalone_proj.to_crs(epsg=4326)
                    else:
                        gdf_served = gdf_buildings_raw
                        gdf_standalone = gdf_buildings_raw.iloc[0:0].to_crs(epsg=4326)

                # ---- demand + electrical assumptions + LV cable ----
                if pf_inputs["use_page2"]:
                    building_meta, category_profiles = p2_meta, p2_profiles
                    if building_meta is None or category_profiles is None:
                        raise ValueError(
                            "Grid Validation demand data not found in this session."
                        )
                    p = page2_result.params or {}
                    pf_electrical = {
                        "v_min_pu": p.get("v_min_pu", 0.9),
                        "v_max_pu": p.get("v_max_pu", 1.1),
                        "pf_load": p.get("pf_load", 0.95),
                        "v_nom_kv": p.get("v_nom_kv", 0.4),
                        "selected_hour": page2_result.hour,
                    }
                    cable_params = {
                        "r_ohm_per_km": p.get("r_ohm_per_km", 0.641),
                        "x_ohm_per_km": p.get("x_ohm_per_km", 0.083),
                        "s_nom_kva": p.get("s_nom_kva", 114.3),
                    }
                else:
                    _seek0(pf_inputs["meta_file"])
                    _seek0(pf_inputs["profiles_file"])
                    building_meta = read_building_metadata_csv(pf_inputs["meta_file"])
                    category_profiles = read_category_profiles_csv(pf_inputs["profiles_file"])
                    mp = pf_inputs["manual_pf_params"]
                    v_nom_kv = (
                        0.4 if mp["v_base_mode"].startswith("3-phase") else (0.4 / np.sqrt(3))
                    )
                    mh = pf_inputs["manual_hour"] or {"auto_hour": True, "hour": None}
                    pf_electrical = {
                        "v_min_pu": mp["v_min_pu"], "v_max_pu": mp["v_max_pu"],
                        "pf_load": mp["pf_load"], "v_nom_kv": v_nom_kv,
                        "selected_hour": (None if mh["auto_hour"] else mh["hour"]),
                    }
                    cable_params = pf_inputs["manual_line_params"]

                if mv_inputs["plant_auto"]:
                    c = gdf_served.to_crs(epsg=4326).geometry.union_all().centroid
                    plant_latlon = (float(c.y), float(c.x))
                else:
                    plant_latlon = (
                        float(mv_inputs["plant_lat"]), float(mv_inputs["plant_lon"])
                    )

                request = build_request(topo_geom, mv_inputs, pf_electrical, cable_params)
                set_mv_reinforcement_request(st.session_state, request)

                result = run_grid_reinforcement(
                    gdf_buildings=gdf_served,
                    gdf_roads=gdf_roads,
                    building_meta=building_meta,
                    category_profiles=category_profiles,
                    request=request,
                    plant_latlon=plant_latlon,
                    dv_cap_pu=dv_cap_pu,
                    skip_k0=(page2_feasible is False),
                    gdf_standalone_4326=gdf_standalone,
                    progress_cb=_say,
                )
                set_mv_reinforcement_result(st.session_state, result)
                set_mv_reinforcement_fingerprint(st.session_state, _fingerprint())
            progress.empty()
            st.success(f"Run completed in {result.total_solve_seconds:.1f} s.")
        except Exception as e:
            progress.empty()
            st.error(f"Grid Reinforcement failed: {e}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = get_mv_reinforcement_result(st.session_state)
if result is not None:
    stored_fp = get_mv_reinforcement_fingerprint(st.session_state)
    if stored_fp is not None and stored_fp != _fingerprint():
        st.info("Inputs changed since the last run — the results below may be stale. "
                "Re-run to refresh them.")
    render_results(result, dv_cap_pu=dv_cap_pu)

    # Distribution cost analysis (Task 1) - collapsed by default
    st.divider()
    st.subheader("Cost analysis")
    render_reinforcement_cost_section(result)
else:
    st.caption("No results yet. Configure the inputs above and press "
               "**Run Grid Reinforcement**.")
