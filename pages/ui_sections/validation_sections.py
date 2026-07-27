from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.powerflow_io import (make_map_lv_with_load_bubbles, make_map_lv_with_pf_violations,
    make_map_lv_voltage_nodes, make_map_lv_current_branches)


def render_page_header() -> None:
    with st.sidebar:
        st.markdown("**Example data**: see the `examples/` folder in this project.")

    st.title("Grid Validation (Power Flow)")
    st.markdown(
        """
This page implements the **minimum electrical validation step** for an LV topology:

1) select a topology source (reuse Page 1 results *or* load external files)  
2) load/assign hourly building demands and aggregate to poles  
3) choose line-parameter assumptions (global or catalog-driven)  
4) run a single-snapshot power flow with a single slack/generation bus  
"""
    )
    st.divider()


def render_topology_source_section() -> tuple[str, Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """
    Render the topology source selector.

    Returns
    -------
    topology_source  : str
        One of:
          "Use results from Grid Topology (Page 1)"
          "Load external files (stand-alone)"
          "Import from OffGridPlanner"
          "Import from OMG (OnSSET for Mini-Grids)"
    nodes_file   : uploaded file or None
    edges_file   : uploaded file or None
    assoc_file   : uploaded file or None
    offgrid_file : uploaded .xlsx or None  (OffGridPlanner)
    omg_poles_file     : uploaded .geojson or None  (OMG)
    omg_secondary_file : uploaded .geojson or None  (OMG)
    omg_trunks_file    : uploaded .geojson or None  (OMG)
    omg_buildings_file : uploaded .geojson/.gpkg or None  (OMG)
    """
    st.subheader("Grid Topology")

    topology_source = st.radio(
        "Choose how to provide topology data",
        options=(
            "Use results from Grid Topology (Page 1)",
            "Load external files (stand-alone)",
            "Import from OffGridPlanner",
            "Import from OMG (OnSSET for Mini-Grids)",
        ),
        index=1,
        help=(
            "Four options are supported:\n"
            "- **Page 1**: reuse in-session topology from Grid Topology.\n"
            "- **External files**: upload nodes.geojson, edges.geojson, associations.csv.\n"
            "- **OffGridPlanner**: upload the Excel export from OffGridPlanner.\n"
            "- **OMG**: upload the four GeoJSON outputs from the OMG notebook "
            "(poles, secondary lines, trunk lines, buildings)."
        ),
    )

    nodes_file: Optional[Any] = None
    edges_file: Optional[Any] = None
    assoc_file: Optional[Any] = None
    offgrid_file: Optional[Any] = None
    # ------------------------------------------------------------------
    # Option A: External files (original behaviour)
    # ------------------------------------------------------------------
    if topology_source == "Load external files (stand-alone)":
        st.markdown("Upload the **topology outputs** (nodes + edges + associations).")

        c1, c2 = st.columns(2)
        with c1:
            nodes_file = st.file_uploader(
                "Nodes (poles) file (.geojson/.gpkg)",
                type=["geojson", "json", "gpkg"],
                key="pf_nodes_file",
            )
        with c2:
            edges_file = st.file_uploader(
                "Edges (LV network) file (.geojson/.gpkg)",
                type=["geojson", "json", "gpkg"],
                key="pf_edges_file",
            )

        assoc_file = st.file_uploader(
            "Associations CSV (building_id, pole_id)",
            type=["csv"],
            key="pf_assoc_file",
        )

    # ------------------------------------------------------------------
    # Option B: OffGridPlanner Excel import
    # ------------------------------------------------------------------
    elif topology_source == "Import from OffGridPlanner":
        st.markdown(
            "Upload the **Excel export** downloaded from OffGridPlanner "
            "(*Step 6 → Download → Excel*)."
        )
        st.caption(
            "The converter extracts poles (nodes), distribution links (edges), "
            "and consumer-to-pole associations automatically. "
            "A building metadata template (category = *Dummy*) is also generated "
            "- download it from the sidebar and edit it before running the power flow."
        )

        offgrid_file = st.file_uploader(
            "OffGridPlanner export (.xlsx)",
            type=["xlsx"],
            key="pf_offgrid_file",
        )

        if offgrid_file is not None:
            _run_offgrid_conversion(offgrid_file)

    # ------------------------------------------------------------------
    # Option C: OMG (OnSSET for Mini-Grids) — single GeoPackage only
    # ------------------------------------------------------------------
    elif topology_source == "Import from OMG (OnSSET for Mini-Grids)":
        st.caption(
            "Upload the **single .gpkg** produced by the OMG notebook "
            "(e.g. *MAFA distribution_grid.gpkg*) plus your buildings file."
        )
        c1, c2 = st.columns(2)
        with c1:
            omg_gpkg_file = st.file_uploader(
                "OMG distribution grid (.gpkg)", type=["gpkg"], key="pf_omg_gpkg",
            )
        with c2:
            omg_buildings_file = st.file_uploader(
                "Buildings file (.gpkg / .geojson)",
                type=["gpkg", "geojson", "json"], key="pf_omg_buildings_gpkg",
            )

        st.markdown("**Post-processing: span length control**")
        apply_span_cap_str = st.radio(
            "Maximum span between poles",
            options=["Keep original OMG lines",
                     "Apply distance cap (insert intermediate poles)"],
            index=0, key="pf_omg_span_mode",
            help="Apply cap: split long lines. Isolated sub-graphs are always reconnected.",
        )
        do_apply_span_cap = (apply_span_cap_str == "Apply distance cap (insert intermediate poles)")
        max_span_m_omg = 40.0
        if do_apply_span_cap:
            max_span_m_omg = st.slider(
                "Max LV span between poles [m]",
                min_value=20, max_value=150, value=40, step=5, key="pf_omg_max_span",
            )

        if omg_gpkg_file is not None and omg_buildings_file is not None:
            _run_omg_gpkg_conversion(
                omg_gpkg_file, omg_buildings_file,
                apply_span_cap=do_apply_span_cap,
                max_span_m=float(max_span_m_omg),
            )

    return (topology_source,
            nodes_file, edges_file, assoc_file,
            offgrid_file)

def _run_offgrid_conversion(offgrid_file) -> None:
    """
    Convert the uploaded OffGridPlanner Excel file and cache results in
    st.session_state["_offgrid_converted"].

    Re-runs only when the uploaded file changes (tracked by file name + size).
    """
    from core.offgrid_converter import convert_offgridplanner_excel

    cache_key = f"{offgrid_file.name}_{offgrid_file.size}"
    existing = st.session_state.get("_offgrid_converted", {})

    if existing.get("_cache_key") == cache_key:
        s = existing["_summary"]
        _show_conversion_summary(s, existing)
        return

    with st.spinner("Converting OffGridPlanner export…"):
        try:
            result = convert_offgridplanner_excel(offgrid_file.read())
        except Exception as exc:
            st.error(f"Conversion failed: {exc}")
            return

    result["_cache_key"] = cache_key
    st.session_state["_offgrid_converted"] = result
    s = result["_summary"]
    _show_conversion_summary(s, result)


def _show_conversion_summary(s: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Display conversion summary as a rich metric table matching Grid Topology style."""
    st.success("OffGridPlanner export converted successfully.")

    if s["n_unmatched_fallback"] > 0:
        st.info(
            f"{s['n_unmatched_fallback']} building(s) assigned to nearest pole by distance fallback."
        )

    st.text(
        "Estimated total line length of the LV system, split between the main backbone "
        "(pole-to-pole feeder network) and the final connections from poles to individual buildings."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total network length [km]", f"{s.get('total_length_km', 0.0):.2f}")
    c2.metric("LV backbone length [km]", f"{s.get('backbone_length_km', 0.0):.2f}")
    c3.metric("Service drop length [km]", f"{s.get('service_drop_length_km', 0.0):.2f}")

    st.text(
        "Number of poles in the LV network. Serving poles supply buildings directly, "
        "while support poles are added only to limit span lengths."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total poles", s["n_poles"])
    c2.metric("Serving poles", s.get("n_serving_poles", s["n_poles"]))
    c3.metric("Support poles", s.get("n_support_poles", 0))

    st.text(
        "Coverage of the settlement by the LV network. "
        "Standalone candidates are buildings left unconnected."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total buildings", s["n_buildings"])
    c2.metric("Grid-served buildings", s["n_assoc"])
    c3.metric("Standalone candidates", s["n_buildings"] - s["n_assoc"])

    st.markdown("**Download converted files** (optional - used automatically in validation below):")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("nodes.geojson", data=result["nodes_geojson"],
            file_name="offgrid_poles.geojson", mime="application/geo+json", key="dl_offgrid_nodes")
    with col2:
        st.download_button("edges.geojson", data=result["edges_geojson"],
            file_name="offgrid_edges.geojson", mime="application/geo+json", key="dl_offgrid_edges")
    with col3:
        st.download_button("associations.csv", data=result["associations_csv"],
            file_name="offgrid_associations.csv", mime="text/csv", key="dl_offgrid_assoc")
    with col4:
        st.download_button("building_metadata_template.csv",
            data=result["building_metadata_template_xlsx"],
            file_name="building_metadata_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_offgrid_meta",
            help="Pre-filled with all building IDs and category = 'Dummy'.")


def _run_omg_gpkg_conversion(
    gpkg_file, buildings_file,
    apply_span_cap: bool = True,
    max_span_m: float = 40.0,
) -> None:
    """Convert a single OMG GeoPackage and cache results."""
    from core.omg_converter import convert_omg_gpkg

    buildings_driver = "GPKG" if buildings_file.name.endswith(".gpkg") else "GeoJSON"
    cache_key = (f"{gpkg_file.name}_{gpkg_file.size}_{buildings_file.name}_"
                 f"{buildings_file.size}_{apply_span_cap}_{max_span_m}")
    existing = st.session_state.get("_omg_converted", {})
    if existing.get("_cache_key") == cache_key:
        _show_omg_summary(existing["_summary"], existing)
        return

    with st.spinner("Converting OMG GeoPackage and reconnecting topology…"):
        try:
            result = convert_omg_gpkg(
                gpkg_bytes=gpkg_file.read(),
                buildings_bytes=buildings_file.read(),
                buildings_driver=buildings_driver,
                apply_span_cap=apply_span_cap,
                max_span_m=max_span_m,
            )
        except Exception as exc:
            st.error(f"OMG conversion failed: {exc}")
            return

    result["_cache_key"] = cache_key
    st.session_state["_omg_converted"] = result
    _show_omg_summary(result["_summary"], result)


def _show_omg_summary(s: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Display OMG conversion summary matching Grid Topology style."""
    st.success("OMG outputs converted successfully.")

    if s.get("n_poles_synthetic", 0) > 0:
        st.info(f"{s['n_poles_synthetic']} synthetic pole(s) inserted to snap edge endpoints.")
    if s.get("n_reconnection_edges", 0) > 0:
        st.info(
            f"{s['n_reconnection_edges']} reconnection edge(s) added to connect isolated "
            "sub-graphs to the main network (Navarro-Espinosa & Ochoa, CIRED 2015)."
        )
    if s.get("n_reassigned", 0) > 0:
        st.info(
            f"{s['n_reassigned']} building(s) reassigned to nearest connected pole for "
            "power flow. Original assignments preserved in associations.csv."
        )

    st.text(
        "Estimated total line length of the LV system, split between the main backbone "
        "(pole-to-pole feeder network) and the final connections from poles to individual buildings."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total network length [km]", f"{s.get('total_length_km', 0.0):.2f}")
    c2.metric("LV backbone length [km]", f"{s.get('backbone_length_km', 0.0):.2f}")
    c3.metric("Service drop length [km]", f"{s.get('service_drop_length_km', 0.0):.2f}")

    st.text(
        "Number of poles in the LV network. Serving poles supply buildings directly, "
        "while support poles are added only to limit span lengths."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total poles", s["n_poles"])
    c2.metric("Serving poles", s.get("n_serving_poles", 0))
    c3.metric("Support poles", s.get("n_support_poles", 0))

    st.text(
        "Coverage of the settlement by the LV network. "
        "Standalone candidates are buildings left unconnected."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total buildings", s["n_buildings"])
    c2.metric("Grid-served buildings", s["n_assoc"])
    c3.metric("Standalone candidates", s["n_buildings"] - s["n_assoc"])

    if s.get("apply_span_cap", True):
        st.caption(
            f"Post-processed with max span cap {s.get('cap_value_m', 40):.0f} m. "
            f"Mean span: {s.get('mean_span_m', 0):.1f} m — Max span: {s.get('max_span_m', 0):.1f} m."
        )
    else:
        st.caption(
            f"Original OMG lines used without span cap. "
            f"Mean span: {s.get('mean_span_m', 0):.1f} m — Max span: {s.get('max_span_m', 0):.1f} m."
        )

    st.markdown("**Download converted files** (optional — used automatically in validation below):")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("nodes.geojson", data=result["nodes_geojson"],
            file_name="omg_nodes.geojson", mime="application/geo+json", key="dl_omg_nodes")
    with col2:
        st.download_button("edges.geojson", data=result["edges_geojson"],
            file_name="omg_edges.geojson", mime="application/geo+json", key="dl_omg_edges")
    with col3:
        st.download_button("associations.csv", data=result["associations_csv"],
            file_name="omg_associations.csv", mime="text/csv", key="dl_omg_assoc",
            help="Original assignments (all poles). For map display.")
    with col4:
        st.download_button("building_metadata_template.xlsx",
            data=result["building_metadata_template_xlsx"],
            file_name="omg_building_metadata_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_omg_meta", help="Pre-filled with all building IDs, category = 'Dummy'.")
    if "pf_associations_csv" in result:
        st.download_button("pf_associations.csv", data=result["pf_associations_csv"],
            file_name="omg_pf_associations.csv", mime="text/csv", key="dl_omg_pf_assoc",
            help="PF-ready: buildings reassigned to nearest connected pole.")


def get_offgrid_converted_files() -> Optional[Dict[str, Any]]:
    """Return the cached OffGridPlanner conversion result from session state, or None."""
    return st.session_state.get("_offgrid_converted", None)


def get_omg_converted_files() -> Optional[Dict[str, Any]]:
    """Return the cached OMG conversion result from session state, or None."""
    return st.session_state.get("_omg_converted", None)

def render_demand_upload_section() -> tuple[Optional[Any], Optional[Any]]:
    c1, c2 = st.columns(2)
    with c1:
        meta_file = st.file_uploader(
            "Building metadata (Excel .xlsx or CSV)",
            type=["xlsx", "csv"],
            key="pf_building_meta_file",
        )
    with c2:
        profiles_file = st.file_uploader(
            "Category profiles (Excel .xlsx or CSV - hour + one column per category, W per building)",
            type=["xlsx", "csv"],
            key="pf_category_profiles_file",
        )

    return meta_file, profiles_file


def render_demand_controls(pole_loads_kW: pd.DataFrame) -> Dict[str, Any]:
    col_1, col_2 = st.columns([2.5, 1])
    with col_2:
        scaling_mode = st.selectbox(
            "Bubble scaling mode",
            options=("Absolute (fixed over the year)", "Relative (rescaled each hour)"),
            index=0,
            key="pf_scaling_mode_dropdown",
        )

    year_max_pole_kW = float(np.nanmax(pole_loads_kW.to_numpy())) if pole_loads_kW.size else 0.0
    pmax_ref_kW = year_max_pole_kW if scaling_mode.startswith("Absolute") else None

    with col_1:
        total_load = pole_loads_kW.sum(axis=1)
        h_min = int(pole_loads_kW.index.min())
        h_max = int(pole_loads_kW.index.max())
        peak_hour = int(total_load.idxmax())
        peak_hour = int(np.clip(peak_hour, h_min, h_max))
        hour = st.slider(
            "Select hour for visualization",
            min_value=h_min,
            max_value=h_max,
            value=peak_hour,
            step=1,
            key="pf_vis_hour_slider",
            help=f"Hours {h_min}-{h_max} as defined in category_profiles. Peak demand at hour {peak_hour}.",
        )

    return {
        "scaling_mode": scaling_mode,
        "year_max_pole_kW": year_max_pole_kW,
        "pmax_ref_kW": pmax_ref_kW,
        "hour": int(hour),
    }


def render_load_visualization(
    *,
    vis: Optional[Dict[str, Any]],
    gdf_nodes_4326,
    slack_pole_id: Optional[int],
) -> None:
    if vis is None:
        st.info("No saved visualization yet. Upload building metadata + category profiles to create it.")
        return

    st.markdown(f"**Aggregated pole load mapping** (hour {vis['hour']})")

    pole_col = str(vis["pole_col"])
    pole_ids = (
        pd.to_numeric(gdf_nodes_4326[pole_col], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    load_dict = vis.get("pole_load_dict", {}) or {}
    default_pid = None
    if load_dict:
        try:
            default_pid = int(max(load_dict.items(), key=lambda kv: float(kv[1]))[0])
        except Exception:
            default_pid = None

    highlight_enabled = st.checkbox(
        "Highlight a pole on the map",
        value=False,
        key="pf_highlight_enabled",
    )
    highlight_pole_id = st.selectbox(
        "Pole to highlight",
        options=[None] + pole_ids,
        index=(0 if default_pid is None else (pole_ids.index(default_pid) + 1)) if pole_ids else 0,
        format_func=lambda x: "None" if x is None else f"Pole {x}",
        disabled=not highlight_enabled,
        key="pf_highlight_pole_id",
    )

    m = make_map_lv_with_load_bubbles(
        center=tuple(vis["center"]),
        gdf_poles_4326=gdf_nodes_4326,
        pole_id_col=pole_col,
        pole_load_kW_at_hour=vis["pole_load_dict"],
        mst_edges_latlon=vis.get("mst_edges_latlon"),
        gdf_edges_4326=vis.get("gdf_edges_4326"),
        gdf_roads_4326=vis.get("gdf_roads_4326"),
        zoom_start=15,
        pmax_ref_kW=vis.get("pmax_ref_kW"),
        show_legend=True,
        slack_pole_id=slack_pole_id,
        highlight_pole_id=(int(highlight_pole_id) if highlight_pole_id is not None else None),
        zoom_to_highlight=bool(highlight_enabled and highlight_pole_id is not None),
    )

    map_key = (
        f"pf_map_{int(vis['hour'])}_"
        f"{'abs' if vis.get('pmax_ref_kW') is not None else 'rel'}_"
        f"hl_{highlight_pole_id}"
    )
    scale_txt = "absolute (year-fixed)" if vis.get("pmax_ref_kW") is not None else "relative (per-hour)"
    pmax_txt = f"{(vis.get('pmax_ref_kW') or 0.0)*1000:.1f} W"
    st.caption(
        f"Black dot = pole node; Purple dot = slack / plant pole; "
        f"Orange bubble = aggregated load at pole (size proportional to W); "
        f"Dark gray line = LV distribution cable. "
        f"Bubble scaling: {scale_txt} | Reference max load: {pmax_txt}"
    )
    st_folium(m, height=650, use_container_width=True, key=map_key)

    with st.expander("Preview aggregated pole loads - top 20 (W)", expanded=False):
        df_preview = (
            pd.Series(vis["pole_load_dict"])
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )
        df_preview["value_W"] = df_preview.iloc[:, 1] * 1000
        df_preview = df_preview[["index", "value_W"]]
        df_preview.columns = ["pole_id", "p_W"]
        st.dataframe(df_preview, use_container_width=True)

    if vis.get("resolved_line_params_df") is not None:
        with st.expander("Preview final merged line parameters (top 20)", expanded=False):
            st.dataframe(vis["resolved_line_params_df"].head(20), use_container_width=True)


def render_electrical_assumptions_section() -> Dict[str, Any]:
    """Minimum electrical assumptions (voltage limits, power factor, voltage
    base convention) — extracted from render_pf_setup_section so Grid
    Reinforcement can reuse the exact same widgets/keys without a slack-pole
    dropdown (GR's plant position is a lat/lon, not a pole_id from an
    already-built topology)."""
    p1, p2, p3 = st.columns(3)
    with p1:
        v_min_pu = st.number_input("Min voltage limit (p.u.)", 0.70, 1.00, 0.90, 0.01, format="%.2f", key="pf_v_min_pu")
    with p2:
        v_max_pu = st.number_input("Max voltage limit (p.u.)", 1.00, 1.30, 1.10, 0.01, format="%.2f", key="pf_v_max_pu")
    with p3:
        pf_load = st.number_input("Assumed load power factor (lagging)", 0.50, 1.00, 0.95, 0.01, format="%.2f", key="pf_pf_load")

    v_base_mode = st.selectbox(
        "Voltage base convention",
        options=("3-phase LV (0.4 kV line-to-line)", "Per-phase equivalent (0.230 kV L-N)"),
        index=0,
        key="pf_vbase_mode",
    )

    return {
        "v_min_pu": float(v_min_pu),
        "v_max_pu": float(v_max_pu),
        "pf_load": float(pf_load),
        "v_base_mode": v_base_mode,
    }


def render_pf_setup_section(*, pole_ids: list[int], suggested_slack: int) -> Dict[str, Any]:
    st.divider()
    st.subheader("Power flow setup")
    st.markdown(
        "Select the **plant / slack pole** and the **minimum electrical assumptions** used to build the network."
    )

    slack_pole_id = st.selectbox(
        "Slack / plant connection pole (pole_id)",
        options=pole_ids,
        index=pole_ids.index(suggested_slack),
        key="pf_slack_pole_dropdown",
    )
    st.caption(f"Suggested based on load-weighted centroid (current map hour): pole_id = {suggested_slack}")

    electrical = render_electrical_assumptions_section()

    return {"slack_pole_id": int(slack_pole_id), **electrical}


def render_line_params_section() -> Dict[str, Any]:
    # ---------------------------------------------------------------------------
    # ABC LV aluminium cable catalog — Caledonian Cables, BS 7870 / 0.6/1 kV
    # R: DC resistance at 20 °C (Ω/km)
    # X: reactance (Ω/km) — standard value for LV ABC aluminium bundles
    # I_max: current rating in still air at 30 °C (A)
    # S_nom: apparent power capacity at 0.4 kV three-phase (kVA) = √3 × 0.4 × I_max
    # Source: caledoniancable.com/English/product/abc-cables/al/lv.html
    # ---------------------------------------------------------------------------
    _ABC_CATALOG: list[dict] = [
        {"label": "ABC 16 mm²  — 72 A",  "section_mm2": 16,  "r": 1.910, "x": 0.083, "i_max_A": 72,  "s_nom_kva": round(1.732 * 0.4 * 72,  1)},
        {"label": "ABC 25 mm²  — 107 A", "section_mm2": 25,  "r": 1.200, "x": 0.083, "i_max_A": 107, "s_nom_kva": round(1.732 * 0.4 * 107, 1)},
        {"label": "ABC 35 mm²  — 132 A", "section_mm2": 35,  "r": 0.868, "x": 0.083, "i_max_A": 132, "s_nom_kva": round(1.732 * 0.4 * 132, 1)},
        {"label": "ABC 50 mm²  — 165 A", "section_mm2": 50,  "r": 0.641, "x": 0.083, "i_max_A": 165, "s_nom_kva": round(1.732 * 0.4 * 165, 1)},
        {"label": "ABC 70 mm²  — 205 A", "section_mm2": 70,  "r": 0.443, "x": 0.083, "i_max_A": 205, "s_nom_kva": round(1.732 * 0.4 * 205, 1)},
        {"label": "ABC 95 mm²  — 240 A", "section_mm2": 95,  "r": 0.320, "x": 0.083, "i_max_A": 240, "s_nom_kva": round(1.732 * 0.4 * 240, 1)},
        {"label": "ABC 120 mm² — 290 A", "section_mm2": 120, "r": 0.253, "x": 0.083, "i_max_A": 290, "s_nom_kva": round(1.732 * 0.4 * 290, 1)},
        {"label": "ABC 150 mm² — 334 A", "section_mm2": 150, "r": 0.206, "x": 0.083, "i_max_A": 334, "s_nom_kva": round(1.732 * 0.4 * 334, 1)},
        {"label": "Custom",               "section_mm2": None, "r": None,  "x": None,  "i_max_A": None, "s_nom_kva": None},
    ]
    _DEFAULT_CABLE_IDX = 3  # ABC 50 mm² — matches previous default

    with st.expander("Line parameters", expanded=True):
        st.markdown(
            "Select a standard **ABC LV aluminium cable** (Caledonian BS 7870, 0.6/1 kV, still air 30 °C) "
            "or enter custom values."
        )

        cable_labels = [c["label"] for c in _ABC_CATALOG]
        selected_label = st.selectbox(
            "Cable type",
            options=cable_labels,
            index=_DEFAULT_CABLE_IDX,
            key="pf_cable_selector",
        )
        selected = next(c for c in _ABC_CATALOG if c["label"] == selected_label)
        is_custom = selected["section_mm2"] is None

        # Show catalog values or editable fields
        if not is_custom:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Section", f"{selected['section_mm2']} mm²")
            col2.metric("R (Ω/km)", f"{selected['r']:.3f}")
            col3.metric("X (Ω/km)", f"{selected['x']:.3f}")
            col4.metric("I_max (A)", f"{selected['i_max_A']}")
            st.caption(
                f"S_nom at 0.4 kV three-phase: **{selected['s_nom_kva']:.1f} kVA** "
                f"(= √3 × 0.4 kV × {selected['i_max_A']} A). "
                "Source: Caledonian Cables catalog, BS 7870."
            )
            r_ohm_per_km = selected["r"]
            x_ohm_per_km = selected["x"]
            s_nom_kva    = selected["s_nom_kva"]

        else:
            st.caption("Enter custom electrical parameters.")
            col1, col2, col3 = st.columns(3)
            with col1:
                r_ohm_per_km = st.number_input(
                    "R (Ω/km)", min_value=0.0001, max_value=5.0,
                    value=0.641, step=0.001, format="%.4f", key="pf_custom_r",
                )
            with col2:
                x_ohm_per_km = st.number_input(
                    "X (Ω/km)", min_value=0.0001, max_value=5.0,
                    value=0.083, step=0.001, format="%.4f", key="pf_custom_x",
                )
            with col3:
                s_nom_kva = st.number_input(
                    "S_nom (kVA)", min_value=1.0, max_value=2000.0,
                    value=114.3, step=1.0, key="pf_custom_s_nom",
                )

        return {
            "mode": "global",
            "r_ohm_per_km": float(r_ohm_per_km),
            "x_ohm_per_km": float(x_ohm_per_km),
            "s_nom_kva": float(s_nom_kva),
            "line_types_file": None,
            "lines_meta_file": None,
        }


def render_pf_run_controls(
    *,
    runner,
    runner_version: str,
    hour_min: int,
    hour_max: int,
    selected_hour: int,
) -> Dict[str, Any]:
    st.divider()
    st.subheader("Run Power Flow (PyPSA)")

    cdbg1, cdbg2, cdbg3 = st.columns(3)
    with cdbg1:
        pf_min_len_m = st.number_input("PF: min edge length filter (m)", 0.0, 50.0, 5.0, 1.0, key="pf_min_len_m")
    with cdbg2:
        pf_sn_mva = st.number_input("PF: system base (sn_mva)", 0.01, 100.0, 1.0, 0.1, key="pf_sn_mva")
    with cdbg3:
        pf_fail_on_nonsense = st.checkbox(
            "PF: fail if voltages non-physical", value=True, key="pf_fail_on_nonsense",
        )

    with st.expander("Runner identity (sanity check)", expanded=False):
        st.write("RUNNER_VERSION:", runner_version)
        st.write("runner type:", type(runner))
        st.write("runner module:", getattr(type(runner), "__module__", None))
        st.write("run_snapshot defaults:", getattr(runner.run_snapshot, "__defaults__", None))

    st.markdown(
        "Choose the exact demand hour used for the electrical check. This is independent from the load-bubble map hour."
    )
    st.caption(f"Recommended hour: {int(selected_hour)} (highest total aggregated load).")
    h_min = int(hour_min)
    h_max = int(hour_max)
    h_sel = int(np.clip(int(selected_hour), h_min, h_max))
    pf_hour = st.slider(
        "Hour to run power flow",
        min_value=h_min, max_value=h_max,
        value=h_sel, step=1, key="pf_run_hour_slider",
        help=f"Hours {h_min}-{h_max} as defined in category_profiles. Peak demand at hour {h_sel}.",
    )

    run_clicked = st.button("Run power flow for selected hour", type="primary", key="pf_run_pf_btn")

    return {
        "pf_min_len_m": float(pf_min_len_m),
        "pf_sn_mva": float(pf_sn_mva),
        "pf_fail_on_nonsense": bool(pf_fail_on_nonsense),
        "pf_hour": int(pf_hour),
        "run_clicked": bool(run_clicked),
    }


def _render_summary_metrics(res: dict, pf_map: Optional[dict]) -> None:
    """
    Compute and display the B1 (normalised) and B2 (electrical) summary metrics
    defined in the comparison framework (grandezze_confronto.docx).
    """
    import math

    df_bus  = res.get("bus_results")
    df_line = res.get("line_results")

    if df_bus is None or df_line is None:
        st.info("Summary metrics not available for this result.")
        return

    # ── Pre-processing ────────────────────────────────────────────────────────
    dv = pd.to_numeric(df_bus.get("deltaV_pct"), errors="coerce").dropna()
    v_pu = pd.to_numeric(df_bus.get("v_pu"), errors="coerce").dropna()
    n_nodes = len(dv)

    I_A       = pd.to_numeric(df_line.get("I_A"),       errors="coerce")
    length_km = pd.to_numeric(df_line.get("length_km"), errors="coerce")
    s_nom_kva = pd.to_numeric(df_line.get("s_nom_kva"), errors="coerce")
    # Nominal voltage from the PF params carried by the result (kV).
    # BUGFIX: this previously read pf_map["v_min_pu"] (the 0.9 p.u. voltage
    # LIMIT, dimensionless) as if it were a voltage in kV, shrinking I_nom by
    # 0.9/0.4 = 2.25x and flagging healthy branches as overloaded.
    _params = res.get("params") or {}
    v_nom_kv = float(_params.get("v_nom_kv", 0.4) or 0.4)
    # True I_nom per line from s_nom_kva
    I_nom_series = s_nom_kva * 1000.0 / (math.sqrt(3) * v_nom_kv * 1000.0)

    # ── B1: Normalised metrics ────────────────────────────────────────────────
    backbone_km   = float(length_km.sum()) if not length_km.empty else float("nan")
    n_edges       = len(df_line)
    # Serving poles = poles with positive load in pf_map
    load_pu_dict  = pf_map.get("line_loading_pu", {}) if pf_map else {}
    n_poles_total = len(pf_map["gdf_poles_4326"]) if pf_map and "gdf_poles_4326" in pf_map else float("nan")
    # Buildings served ≈ poles with load (each association mapped to one pole)
    n_served_bld  = len(set(k[0] for k in load_pu_dict) | set(k[1] for k in load_pu_dict)) if load_pu_dict else float("nan")

    backbone_per_bld = backbone_km * 1000.0 / n_served_bld if n_served_bld and n_served_bld > 0 else float("nan")
    poles_per_bld    = n_poles_total / n_served_bld if n_served_bld and n_served_bld > 0 else float("nan")

    # ── B2: Electrical metrics ────────────────────────────────────────────────
    dv_abs   = dv.abs()
    pct_gt5  = float((dv_abs > 5.0).sum())  / n_nodes * 100 if n_nodes else float("nan")
    pct_gt10 = float((dv_abs > 10.0).sum()) / n_nodes * 100 if n_nodes else float("nan")
    dv_max   = float(dv_abs.max()) if not dv_abs.empty else float("nan")

    # ΔV normalised by electrical distance to slack
    # Use graph shortest path weighted by length_km
    if pf_map and "gdf_edges_4326" in pf_map and pf_map["gdf_edges_4326"] is not None and not df_line.empty:
        try:
            import networkx as nx
            G = nx.Graph()
            for _, r in df_line.iterrows():
                b0 = int(pd.to_numeric(r.get("bus0"), errors="coerce"))
                b1 = int(pd.to_numeric(r.get("bus1"), errors="coerce"))
                ln = float(pd.to_numeric(r.get("length_km"), errors="coerce") or 0)
                G.add_edge(b0, b1, length=ln)
            slack = int(pf_map.get("slack_pole_id", -1))
            # Find node with max ΔV
            worst_bus = int(dv_abs.idxmax()) if not dv_abs.empty else slack
            worst_bus_id = int(df_bus.iloc[worst_bus]["bus"]) if "bus" in df_bus.columns else worst_bus
            if G.has_node(slack) and G.has_node(worst_bus_id):
                d_km = nx.shortest_path_length(G, slack, worst_bus_id, weight="length")
                dv_norm = dv_max / d_km if d_km > 0 else float("nan")
            else:
                dv_norm = float("nan")
        except Exception:
            dv_norm = float("nan")
    else:
        dv_norm = float("nan")

    # % branches with I > I_nom
    valid_mask = I_A.notna() & I_nom_series.notna()
    if valid_mask.any():
        overcurrent = (I_A[valid_mask] > I_nom_series[valid_mask]).sum()
        pct_overcurrent = float(overcurrent) / valid_mask.sum() * 100
    else:
        pct_overcurrent = float("nan")

    # Current mean weighted by length
    valid_I = I_A.notna() & length_km.notna()
    if valid_I.any():
        I_mean = float((I_A[valid_I] * length_km[valid_I]).sum() / length_km[valid_I].sum())
    else:
        I_mean = float("nan")

    I_max = float(I_A.max()) if not I_A.dropna().empty else float("nan")

    # ── Display ───────────────────────────────────────────────────────────────
    def _fmt(v, decimals=2, suffix=""):
        return f"{v:.{decimals}f}{suffix}" if not math.isnan(v) else "n/a"

    b1_rows = {
        "Backbone / building [m/building]":  _fmt(backbone_per_bld, 1),
        "Poles / building":                  _fmt(poles_per_bld, 2),
    }
    b2_rows = {
        "ΔVmax [%]":                         _fmt(dv_max, 2),
        "% nodes |ΔV| > 5%":                _fmt(pct_gt5, 1, " %"),
        "% nodes |ΔV| > 10% (EN 50160)":    _fmt(pct_gt10, 1, " %"),
        "ΔVmax / elec. distance [%/km]":     _fmt(dv_norm, 3),
        "I_max [A]":                         _fmt(I_max, 1),
        "I_mean weighted [A]":               _fmt(I_mean, 1),
        "% branches I > I_nom":              _fmt(pct_overcurrent, 1, " %"),
    }

    c1, c2 = st.columns(2)
    with c1:
        st.caption("**B1 — Normalised metrics**")
        st.dataframe(
            pd.DataFrame(b1_rows.items(), columns=["Metric", "Value"]),
            hide_index=True, use_container_width=True,
        )
    with c2:
        st.caption("**B2 — Electrical metrics (toolkit contribution)**")
        st.dataframe(
            pd.DataFrame(b2_rows.items(), columns=["Metric", "Value"]),
            hide_index=True, use_container_width=True,
        )


def render_pf_results(
    res: Optional[Dict[str, Any]],
    pf_map: Optional[Dict[str, Any]] = None,
) -> None:
    if res is None:
        return

    st.markdown(f"**Snapshot hour:** `{res['hour']}`")

    with st.expander("PF debug (quick checks)", expanded=False):
        st.json(res.get("debug", {}) or {})

    # ---- Bus voltage table ----
    st.markdown("**Bus voltages**")
    df_bus = res["bus_results"].copy()
    # Select and rename columns
    bus_cols = {
        "bus": "bus",
        "V_V": "V [V]",
        "deltaV_V": "ΔV [V]",
        "deltaV_pct": "ΔV [%]",
        "violates_limits": "violates limit",
    }
    bus_show = df_bus[[c for c in bus_cols if c in df_bus.columns]].rename(columns=bus_cols)
    fmt_bus = {}
    if "V [V]" in bus_show.columns:
        fmt_bus["V [V]"] = "{:.2f}"
    if "ΔV [V]" in bus_show.columns:
        fmt_bus["ΔV [V]"] = "{:.2f}"
    if "ΔV [%]" in bus_show.columns:
        fmt_bus["ΔV [%]"] = "{:.2f}"
    st.dataframe(bus_show.style.format(fmt_bus), use_container_width=True)

    # ---- Line flows table ----
    st.markdown("**Line flows**")
    df_line = res["line_results"].copy()
    # p0_W / q0_VAr / s0_VA / I_A are pre-computed in powerflow_network.py
    line_cols_map = {
        "line": "line",
        "bus0": "bus0",
        "bus1": "bus1",
        "length_km": "length [km]",
        "r_ohm": "R [Ω]",
        "x_ohm": "X [Ω]",
        "p0_W": "P0 [W]",
        "q0_VAr": "Q0 [VAr]",
        "s0_VA": "S0 [VA]",
        "I_A": "I [A]",
    }
    line_show = df_line[[c for c in line_cols_map if c in df_line.columns]].rename(columns=line_cols_map)
    fmt_line = {}
    for col in ["length [km]", "R [Ω]", "X [Ω]", "P0 [W]", "Q0 [VAr]", "S0 [VA]", "I [A]"]:
        if col in line_show.columns:
            fmt_line[col] = "{:.3f}"
    st.dataframe(line_show.style.format(fmt_line), use_container_width=True)

    # ---- Summary metrics table (B1 + B2) ----
    st.markdown("**Summary metrics**")
    _render_summary_metrics(res, pf_map)

    if pf_map is not None:
        v_min = float(pf_map["v_min_pu"])
        v_max = float(pf_map["v_max_pu"])

        # --- Map 1: voltage nodes ---
        st.markdown("**Voltage map**")
        st.caption(
            f"Nodes colored by voltage drop from nominal (1.0 p.u.): "
            f"green = drop < 5%; teal = 5-10%; orange = 10-20%; red = > 20%. "
            f"Purple = slack / plant pole. Branches: thin black lines. "
            f"Voltage limits: [{v_min:.2f}, {v_max:.2f}] p.u."
        )
        m1 = make_map_lv_voltage_nodes(
            center=tuple(pf_map["center"]),
            gdf_poles_4326=pf_map["gdf_poles_4326"],
            pole_id_col=str(pf_map["pole_id_col"]),
            gdf_edges_4326=pf_map.get("gdf_edges_4326"),
            mst_edges_latlon=pf_map.get("mst_edges_latlon"),
            gdf_roads_4326=pf_map.get("gdf_roads_4326"),
            zoom_start=15,
            slack_pole_id=pf_map.get("slack_pole_id"),
            bus_v_pu=pf_map.get("bus_v_pu"),
            v_min_pu=v_min,
            v_max_pu=v_max,
        )
        st_folium(m1, height=600, use_container_width=True, key=f"pf_vmap_{int(res['hour'])}")

        # --- Map 2: current branches ---
        st.markdown("**Current map**")
        st.caption(
            "Branches colored by estimated current: "
            "gray = 0 A (no load); current bands 0-10 / 10-20 / 20-30 / 30-40 / 40-50 / 50+ A "
            "(light → dark blue). Nodes: small black dots (purple = slack pole)."
        )
        m2 = make_map_lv_current_branches(
            center=tuple(pf_map["center"]),
            gdf_poles_4326=pf_map["gdf_poles_4326"],
            pole_id_col=str(pf_map["pole_id_col"]),
            gdf_edges_4326=pf_map.get("gdf_edges_4326"),
            mst_edges_latlon=pf_map.get("mst_edges_latlon"),
            gdf_roads_4326=pf_map.get("gdf_roads_4326"),
            zoom_start=15,
            slack_pole_id=pf_map.get("slack_pole_id"),
            line_loading_pu=pf_map.get("line_loading_pu"),
            line_s_nom_kva=pf_map.get("line_s_nom_kva"),
        )
        st_folium(m2, height=600, use_container_width=True, key=f"pf_imap_{int(res['hour'])}")

    if res["summary"]["num_voltage_violations"] > 0:
        st.warning("Voltage violations detected.")
    else:
        st.success("No voltage violations for this snapshot.")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            """
            1. **Select the topology source** (Grid Topology results, OffGridPlanner, OMG, or manual files)  
            2. Upload **building metadata** and **category profiles**, then aggregate demand  
            3. Inspect the **load map** and pick the reference hour  
            4. Set the **slack pole**, voltage limits and **line parameters** (global or catalog)  
            5. Click **Run power flow**  
            6. Inspect summary metrics, **voltage** and **current** maps  
            7. Open **Cost analysis** for the distribution cost breakdown
            """
        )
        st.markdown("---")
        st.markdown("**Example data**: see the `examples/` folder in this project.")
