from __future__ import annotations

"""Distribution cost analysis UI (Task 1).

One collapsed-by-default expander per page (Grid Validation and Grid
Reinforcement) containing:
- unit-cost sliders (defaults from core.costs.DistributionUnitCosts);
- the cost breakdown computed from the physical drivers already produced by
  the code (lengths, pole counts, connections, transformer sizes);
- a CSV download of the breakdown.
"""

from typing import Any, Optional

import pandas as pd
import streamlit as st

from core.costs import (
    DistributionUnitCosts,
    breakdown_totals,
    combine_breakdowns,
    last_mile_cost,
    lv_cable_cost_from_line_params,
    lv_network_cost,
    mv_network_cost,
    mv_pole_spacing_m,
)


# ---------------------------------------------------------------------------
# Unit-cost sliders
# ---------------------------------------------------------------------------


def _unit_cost_sliders(prefix: str, *, include_mv: bool) -> DistributionUnitCosts:
    """Sliders for every unit cost; returns the resulting DistributionUnitCosts."""
    d = DistributionUnitCosts()

    st.markdown("**LV network unit costs**")
    c1, c2, c3 = st.columns(3)
    with c1:
        pole = st.slider("LV pole material [$/pole]", 50.0, 400.0,
                         d.lv_pole_material_usd, 5.0, key=f"{prefix}_lv_pole")
        cable3 = st.slider("ABC 3-phase cable [$/m]", 1.0, 12.0,
                           d.lv_cable_3ph_usd_per_m, 0.1, key=f"{prefix}_c3")
        install = st.slider("Line stringing [$/km]", 500.0, 5000.0,
                            d.lv_install_usd_per_km, 50.0, key=f"{prefix}_inst")
    with c2:
        hw = st.slider("LV pole hardware [$/pole]", 0.0, 150.0,
                       d.lv_pole_hardware_usd, 5.0, key=f"{prefix}_lv_hw")
        cable1 = st.slider("ABC 1-phase cable [$/m]", 0.5, 8.0,
                           d.lv_cable_1ph_usd_per_m, 0.1, key=f"{prefix}_c1")
        earth = st.slider("Network earthing [$/km]", 0.0, 3000.0,
                          d.lv_earthing_usd_per_km, 50.0, key=f"{prefix}_earth")
    with c3:
        found = st.slider("LV pole foundation [$/pole]", 0.0, 200.0,
                          d.lv_pole_foundation_usd, 2.0, key=f"{prefix}_lv_found")
        share3 = st.slider("Share of 3-phase backbone [-]", 0.0, 1.0,
                           d.lv_share_3ph, 0.05, key=f"{prefix}_share3")
        transport = st.slider("Transport lump [$/network]", 0.0, 15000.0,
                              d.transport_lump_usd, 250.0, key=f"{prefix}_transp")

    st.markdown("**Last-mile unit costs (per connection)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        drop = st.slider("Service-drop cable [$/m]", 0.05, 2.0,
                         d.drop_cable_usd_per_m, 0.05, key=f"{prefix}_drop")
        board = st.slider("Ready board + box [$/conn]", 0.0, 150.0,
                          d.ready_board_usd, 2.0, key=f"{prefix}_board")
    with c2:
        meter = st.slider("Smart meter [$/conn]", 0.0, 200.0,
                          d.meter_usd, 2.0, key=f"{prefix}_meter")
        chw = st.slider("Connection hardware [$/conn]", 0.0, 80.0,
                        d.conn_hardware_usd, 1.0, key=f"{prefix}_chw")
    with c3:
        cearth = st.slider("Customer earthing [$/conn]", 0.0, 60.0,
                           d.conn_earthing_usd, 1.0, key=f"{prefix}_cearth")
        cinst = st.slider("Connection labour [$/conn]", 0.0, 100.0,
                          d.conn_install_usd, 1.0, key=f"{prefix}_cinst")

    kwargs: dict[str, Any] = dict(
        lv_pole_material_usd=pole,
        lv_pole_hardware_usd=hw,
        lv_pole_foundation_usd=found,
        lv_cable_3ph_usd_per_m=cable3,
        lv_cable_1ph_usd_per_m=cable1,
        lv_share_3ph=share3,
        lv_earthing_usd_per_km=earth,
        lv_install_usd_per_km=install,
        transport_lump_usd=transport,
        drop_cable_usd_per_m=drop,
        ready_board_usd=board,
        meter_usd=meter,
        conn_hardware_usd=chw,
        conn_earthing_usd=cearth,
        conn_install_usd=cinst,
    )

    if include_mv:
        st.markdown("**MV network and transformer unit costs**")
        c1, c2, c3 = st.columns(3)
        with c1:
            mv_pole = st.slider("MV pole [$/pole]", 50.0, 800.0,
                                d.mv_pole_usd, 10.0, key=f"{prefix}_mv_pole")
            mv_cable = st.slider("MV cable [$/km]", 500.0, 4000.0,
                                 d.mv_cable_usd_per_km, 50.0, key=f"{prefix}_mv_cable")
        with c2:
            mv_hw = st.slider("MV pole hardware [$/pole]", 0.0, 150.0,
                              d.mv_pole_hardware_usd, 5.0, key=f"{prefix}_mv_hw")
            mv_stay = st.slider("MV stay + accessories [$/km]", 0.0, 2000.0,
                                d.mv_stay_usd_per_km, 50.0, key=f"{prefix}_mv_stay")
        with c3:
            mv_found = st.slider("MV pole foundation [$/pole]", 0.0, 200.0,
                                 d.mv_pole_foundation_usd, 2.0, key=f"{prefix}_mv_found")
            mv_inst = st.slider("MV line stringing [$/km]", 500.0, 6000.0,
                                d.mv_install_usd_per_km, 50.0, key=f"{prefix}_mv_inst")
        c1, c2, c3 = st.columns(3)
        with c1:
            tr_c0 = st.slider("Transformer base cost @25 kVA [$]", 1000.0, 8000.0,
                              d.tr_c0_usd, 50.0, key=f"{prefix}_tr_c0")
        with c2:
            tr_alpha = st.slider("Transformer scaling exponent [-]", 0.1, 1.0,
                                 d.tr_alpha, 0.01, key=f"{prefix}_tr_alpha",
                                 help="C(S) = C0 x (S / 25 kVA)^alpha, fitted on the "
                                      "Uganda V2B step-down points (25 and 500 kVA).")
        with c3:
            tr_struct = st.slider("Transformer structure [$]", 0.0, 2000.0,
                                  d.tr_structure_usd, 25.0, key=f"{prefix}_tr_struct")
        kwargs.update(
            mv_pole_usd=mv_pole,
            mv_pole_hardware_usd=mv_hw,
            mv_pole_foundation_usd=mv_found,
            mv_cable_usd_per_km=mv_cable,
            mv_stay_usd_per_km=mv_stay,
            mv_install_usd_per_km=mv_inst,
            tr_c0_usd=tr_c0,
            tr_alpha=tr_alpha,
            tr_structure_usd=tr_struct,
        )

    return DistributionUnitCosts(**kwargs)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_breakdown(df: pd.DataFrame, *, key: str) -> None:
    if df is None or df.empty:
        st.info("No cost drivers available yet.")
        return

    totals = breakdown_totals(df)
    cols = st.columns(len(totals))
    for col, (name, value) in zip(cols, totals.items()):
        col.metric(f"{name} [k$]", f"{value / 1000.0:,.1f}")

    show = df.copy()
    show["total_kusd"] = (show["total_usd"] / 1000.0).round(1)
    show = show.drop(columns=["total_usd"])
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.download_button(
        "Download cost breakdown (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="distribution_cost_breakdown.csv",
        mime="text/csv",
        key=f"{key}_dl",
    )


def _drivers_from_validation_inputs(validation_inputs: Any) -> Optional[dict]:
    """Fallback drivers when the topology comes from an external source
    (OffGridPlanner / OMG / manual files): backbone length from the edge
    geometries, counts from nodes/associations. The service-drop length is
    not part of external topologies, so it is estimated from a mean drop
    length slider."""
    try:
        gdf_edges = validation_inputs.gdf_edges_4326
        gdf_nodes = validation_inputs.gdf_nodes_4326
        assoc = validation_inputs.associations_df
        if gdf_nodes is None or assoc is None:
            return None

        backbone_km = 0.0
        if gdf_edges is not None and len(gdf_edges) > 0:
            from core.distribution_io import derive_utm_epsg

            g = gdf_edges
            if g.crs is None:
                g = g.set_crs(epsg=4326, allow_override=True)
            utm = derive_utm_epsg(g.to_crs(epsg=4326))
            backbone_km = float(g.to_crs(epsg=utm).length.sum()) / 1000.0

        n_served = int(assoc["building_id"].nunique()) if "building_id" in assoc.columns else int(len(assoc))
        avg_drop_m = st.slider(
            "Mean service-drop length (external topology) [m]",
            5.0, 80.0, 32.0, 1.0, key="cost_avg_drop_m",
            help="External topologies do not carry drop geometries; 32 m is the "
                 "DRC-Idjwi observed mean (81 km / 2540 connections).",
        )
        return {
            "backbone_length_km": backbone_km,
            "service_drop_length_km": n_served * avg_drop_m / 1000.0,
            "num_poles_total": int(len(gdf_nodes)),
            "num_served": n_served,
        }
    except Exception as exc:  # defensive: cost section must never break the page
        st.warning(f"Cannot derive cost drivers from the loaded topology: {exc}")
        return None


# ---------------------------------------------------------------------------
# Page 2 — Grid Validation
# ---------------------------------------------------------------------------


def render_validation_cost_section(
    *,
    validation_inputs: Any,
    topology_result: Any | None,
) -> None:
    with st.expander("Distribution cost analysis", expanded=False):
        st.caption(
            "Unit costs synthesised from four real mini-grid cost workbooks "
            "(DRC-Idjwi, Uganda V2A/V2B, Bugarula/Prolasa, portfolio model). "
            "Adjust the sliders for site-specific differential analysis."
        )
        costs = _unit_cost_sliders("gvcost", include_mv=False)

        if topology_result is not None:
            metrics = dict(topology_result.metrics)
        else:
            metrics = _drivers_from_validation_inputs(validation_inputs)
            if metrics is None:
                st.info(
                    "Load a topology (session or external) to compute the cost breakdown."
                )
                return

        lv_df = lv_network_cost(metrics, costs)

        # Per-cable-type cost from the line catalog (cost_usd_per_m column in
        # line_types.csv): replaces the generic backbone-cable row with exact
        # per-edge costs whenever the catalog mode resolved them.
        catalog_df = lv_cable_cost_from_line_params(
            getattr(validation_inputs, "resolved_line_params_df", None)
        )
        if catalog_df is not None:
            lv_df = lv_df[~lv_df["item"].str.startswith("Backbone cable")]
            lv_df = combine_breakdowns([lv_df, catalog_df])
            st.caption(
                "Backbone cable cost taken from the line catalog "
                "(`cost_usd_per_m` in line_types.csv) — the ABC cable sliders "
                "above are ignored for the backbone."
            )

        df = combine_breakdowns([lv_df, last_mile_cost(metrics, costs)])
        n_served = float(metrics.get("num_served", 0) or 0)
        if n_served > 0:
            st.caption(
                f"Total / connection: "
                f"**{float(df['total_usd'].sum()) / n_served:,.0f} $/conn** "
                f"({int(n_served)} grid-served buildings)."
            )
        _render_breakdown(df, key="gvcost")


# ---------------------------------------------------------------------------
# Page 3 — Grid Reinforcement
# ---------------------------------------------------------------------------


def render_reinforcement_cost_section(result: Any) -> None:
    """Cost analysis of the final iteration + differential comparison across k.

    `result` is a core.contracts.MvReinforcementResult.
    """
    with st.expander("Distribution cost analysis", expanded=False):
        st.caption(
            "Unit costs synthesised from four real mini-grid cost workbooks. "
            "MV poles are equispaced along the backbone: 60 m at 11 kV, 120 m at 33 kV."
        )
        costs = _unit_cost_sliders("grcost", include_mv=True)

        mv_kv = float(
            st.selectbox(
                "MV nominal voltage [kV]",
                options=(11.0, 33.0),
                index=0,
                key="grcost_mv_kv",
                help=f"Sets the MV pole spacing: "
                     f"{mv_pole_spacing_m(11.0):.0f} m @ 11 kV, "
                     f"{mv_pole_spacing_m(33.0):.0f} m @ 33 kV.",
            )
        )

        final = result.final

        # --- final iteration: LV subnetworks + MV backbone + transformers ---
        lv_frames = []
        subnet_rows = []
        tr_kvas: list[float] = []
        for sub in final.subnetworks:
            m = dict(sub.topology.metrics)
            sub_df = combine_breakdowns(
                [lv_network_cost(m, costs), last_mile_cost(m, costs)]
            )
            lv_frames.append(sub_df)
            if sub.tr_s_nom_kva is not None:
                tr_kvas.append(float(sub.tr_s_nom_kva))
            subnet_rows.append(
                {
                    "subnetwork": f"Subnetwork {int(sub.subnet_id) + 1} ({sub.root_kind})",
                    "backbone_km": round(float(m.get("backbone_length_km", 0.0)), 3),
                    "drops_km": round(float(m.get("service_drop_length_km", 0.0)), 3),
                    "poles": int(m.get("num_poles_total", 0)),
                    "served": int(m.get("num_served", 0)),
                    "tr_kva": (None if sub.tr_s_nom_kva is None else float(sub.tr_s_nom_kva)),
                    "lv_cost_kusd": round(float(sub_df["total_usd"].sum()) / 1000.0, 1),
                }
            )

        lv_df = combine_breakdowns(lv_frames)
        # aggregate the per-subnetwork frames into single LV/last-mile rows
        lv_df = (
            lv_df.groupby(["category", "item", "unit", "unit_cost_usd"], sort=False, as_index=False)
            .agg(quantity=("quantity", "sum"), total_usd=("total_usd", "sum"))
            [["category", "item", "quantity", "unit", "unit_cost_usd", "total_usd"]]
        )

        mv_df = mv_network_cost(
            float(final.mv_backbone_length_km), tr_kvas, costs, mv_v_nom_kv=mv_kv
        )
        df = combine_breakdowns([lv_df, mv_df])

        st.markdown(
            f"**Final iteration: k = {int(final.n_transformers)} transformer(s), "
            f"MV backbone = {float(final.mv_backbone_length_km):.2f} km**"
        )
        _render_breakdown(df, key="grcost")

        st.markdown("**Per-subnetwork LV costs**")
        st.dataframe(
            pd.DataFrame(subnet_rows), use_container_width=True, hide_index=True
        )

        # --- differential comparison across the k-iterations -----------------
        if len(result.iterations) > 1:
            st.markdown("**Differential comparison across iterations**")
            comp_rows = []
            for it in result.iterations:
                it_lv = 0.0
                it_tr_kvas: list[float] = []
                for sub in it.subnetworks:
                    m = dict(sub.topology.metrics)
                    it_lv += float(
                        combine_breakdowns(
                            [lv_network_cost(m, costs), last_mile_cost(m, costs)]
                        )["total_usd"].sum()
                    )
                    if sub.tr_s_nom_kva is not None:
                        it_tr_kvas.append(float(sub.tr_s_nom_kva))
                it_mv_df = mv_network_cost(
                    float(it.mv_backbone_length_km), it_tr_kvas, costs, mv_v_nom_kv=mv_kv
                )
                it_mv = float(
                    it_mv_df[it_mv_df["category"] == "MV network"]["total_usd"].sum()
                )
                it_tr = float(
                    it_mv_df[it_mv_df["category"] == "Transformers"]["total_usd"].sum()
                )
                comp_rows.append(
                    {
                        "k (transformers)": int(it.n_transformers),
                        "converged": bool(it.converged),
                        "MV backbone [km]": round(float(it.mv_backbone_length_km), 2),
                        "LV + last mile [k$]": round(it_lv / 1000.0, 1),
                        "MV network [k$]": round(it_mv / 1000.0, 1),
                        "Transformers [k$]": round(it_tr / 1000.0, 1),
                        "Total distribution [k$]": round((it_lv + it_mv + it_tr) / 1000.0, 1),
                    }
                )
            st.dataframe(
                pd.DataFrame(comp_rows), use_container_width=True, hide_index=True
            )
            st.caption(
                "Differential analysis: the LV-only iteration (k = 0, when present) "
                "is the baseline against which the hybrid MV/LV designs are compared."
            )
