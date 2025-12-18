"""
Power Flow (PF) module that consumes the *outputs* of the distribution part:

Inputs:
- mst_nodes.geojson  (points)  -> poles/nodes
    * expected columns: "id" (node id) OR "pole_id"
- mst_edges.geojson  (lines)   -> LV network segments
    * expected columns: "source" and "target" (node ids) OR other endpoint pairs
- associations.csv   (table)   -> building-to-pole mapping
    * expected columns: "pole_id" and "building_id"

This version is specifically adapted for Lusanga (DRC) load, the next version taking a load profile and sun availability profiles will come soon
Morover with big grids, length > 20km infeasibility of the PF may occur, the next version will integrate a stand alone choice part and/or MV net 

import math
import os
import io
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pypsa


# ======================================================================================
# HEADER
# ======================================================================================
st.title("Power Flow (PF) - from Routing Outputs (Nodes/Edges + Associations)")
st.caption("Uploads: mst_nodes.geojson, mst_edges.geojson, associations.csv")


# ======================================================================================
# DEFAULT 24h PROFILES (Lusanga)
# ======================================================================================
DEFAULT_HOURLY_TOTAL_KW = {
    1:  14.5,  2:  12.1,  3:  11.4,  4:  7.7,  5:  7.9,  6:  13.0,
    7:  22.0,  8:  28.3,  9:  35.1,  10: 34.6,  11: 29.8,  12: 27.0,
    13: 62.6,  14: 64.2,  15: 63.3,  16: 66.7,  17: 29.9,  18: 32.1,
    19: 38.7,  20: 48.6,  21: 41.1,  22: 30.6,  23: 21.5,  24: 16.7,
}

DEFAULT_HOURLY_PV_PU = {
    1: 0.0,  2: 0.0,  3: 0.0,  4: 0.0,  5: 0.0,  6: 0.0,
    7: 0.073,  8: 0.213,  9: 0.341,  10: 0.437,  11: 0.494,  12: 0.516,
    13: 0.493,  14: 0.419,  15: 0.325,  16: 0.207,  17: 0.095,  18: 0.013,
    19: 0.0,  20: 0.0,  21: 0.0,  22: 0.0,  23: 0.0,  24: 0.0,
}

SNAPSHOTS = list(range(1, 25))


# ======================================================================================
# IO HELPERS
# ======================================================================================
def _read_geofile(uploaded_file) -> gpd.GeoDataFrame:
    """Read GeoJSON/GPKG uploads into a GeoDataFrame."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    data = uploaded_file.getvalue()

    if suffix in [".geojson", ".json"]:
        return gpd.read_file(io.BytesIO(data))

    if suffix == ".gpkg":
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gpkg") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return gpd.read_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    raise ValueError(f"Unsupported geofile extension: {suffix}")


def load_and_transform_geodata(uploaded_file, target_crs_epsg: int = 32633) -> gpd.GeoDataFrame:
    """
    Load and project to a metric CRS (meters) so we can compute lengths consistently.
    """
    gdf = _read_geofile(uploaded_file)

    if gdf.crs is None:
        raise ValueError("Input GeoJSON has no CRS. Re-export with CRS (e.g., EPSG:4326) then retry.")

    if gdf.crs.to_epsg() != target_crs_epsg:
        gdf = gdf.to_crs(epsg=target_crs_epsg)

    gdf = gdf[gdf.is_valid].copy()
    return gdf


def load_associations_csv(uploaded_file) -> pd.DataFrame:
    """
    Load associations.csv produced by the routing app.

    Accepted schemas:
    - pole_id, building_id     (recommended)
    - pole_id, building        (legacy)
    - pole,   building_id      (legacy)
    - pole,   building         (legacy)

    We normalize to columns: pole_id, building_id.
    """
    df = pd.read_csv(uploaded_file)

    cols_l = {c.lower(): c for c in df.columns}

    if "pole_id" in cols_l:
        pole_col = cols_l["pole_id"]
    elif "pole" in cols_l:
        pole_col = cols_l["pole"]
    else:
        raise ValueError("Associations CSV must contain a pole column: pole_id (preferred) or pole")

    if "building_id" in cols_l:
        bld_col = cols_l["building_id"]
    elif "building" in cols_l:
        bld_col = cols_l["building"]
    else:
        raise ValueError("Associations CSV must contain a building column: building_id (preferred) or building")

    out = df[[pole_col, bld_col]].rename(columns={pole_col: "pole_id", bld_col: "building_id"}).copy()
    return out


def infer_line_endpoints_columns(gdf_lines: gpd.GeoDataFrame) -> tuple[str, str]:
    """
    Detect which columns encode line endpoints.

    Supports your routing export ("source"/"target") and a few common variants.
    """
    candidates = [
        ("source", "target"),
        ("bus0", "bus1"),
        ("from", "to"),
        ("u", "v"),
    ]
    cols = set([c.lower() for c in gdf_lines.columns])
    for a, b in candidates:
        if a in cols and b in cols:
            real_a = next(c for c in gdf_lines.columns if c.lower() == a)
            real_b = next(c for c in gdf_lines.columns if c.lower() == b)
            return real_a, real_b

    raise ValueError("Edges GeoJSON must contain endpoints columns (e.g., source/target).")


def ensure_pole_id_column(gdf_poles: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    """
    Ensure the nodes GeoDataFrame has a standardized 'pole_id' attribute.

    Adaptation for your routing export:
    - nodes file contains 'id' (0..N-1). We use it as pole_id if pole_id is missing.
    """
    cols_l = [c.lower() for c in gdf_poles.columns]

    if "pole_id" in cols_l:
        pole_id_col = next(c for c in gdf_poles.columns if c.lower() == "pole_id")
        return gdf_poles, pole_id_col

    if "id" in cols_l:
        id_col = next(c for c in gdf_poles.columns if c.lower() == "id")
        gdf_poles = gdf_poles.copy()
        gdf_poles["pole_id"] = gdf_poles[id_col]
        return gdf_poles, "pole_id"

    gdf_poles = gdf_poles.copy()
    gdf_poles["pole_id"] = gdf_poles.index
    return gdf_poles, "pole_id"


# ======================================================================================
# DISPATCH
# ======================================================================================
def simulate_dispatch_24h(
    load_kW: pd.Series,
    pv_pu: pd.Series,
    pv_nom_kW: float,
    batt_e_kWh: float,
    batt_p_kW: float,
    genset_kW: float,
    eta_ch: float,
    eta_dis: float,
    dod: float,
    soc_init_pct: float,
    soc_low_pct: float,
    soc_setpoint_pct: float,
    allow_preventive_genset: bool,
) -> pd.DataFrame:
    snaps = load_kW.index

    e_max = float(batt_e_kWh)
    e_min = e_max * (1.0 - float(dod))
    e = e_max * (float(soc_init_pct) / 100.0) if e_max > 0 else 0.0

    out = {
        "Load_kW": [],
        "PV_gross_kW": [],
        "PV_to_load_kW": [],
        "PV_to_batt_kW": [],
        "PV_used_kW": [],
        "Curtailment_kW": [],
        "Battery_discharge_kW": [],
        "Battery_to_load_kW": [],
        "Battery_charge_kW": [],
        "Genset_out_kW": [],
        "Unmet_kW": [],
        "SOC_pct": [],
        "Preventive_genset_flag": [],
    }

    for t in snaps:
        L = float(load_kW.loc[t])
        pv_gross = float(pv_nom_kW * pv_pu.loc[t])

        pv_to_load = min(L, pv_gross)
        remaining = L - pv_to_load
        pv_left = pv_gross - pv_to_load

        batt_to_load = 0.0
        batt_dis = 0.0
        batt_ch = 0.0
        genset = 0.0
        unmet = 0.0
        prev_flag = 0.0

        if remaining > 1e-9:
            if e_max > 0 and batt_p_kW > 0:
                e_avail = max(0.0, e - e_min)
                batt_dis_max = min(batt_p_kW, e_avail)
                batt_to_load_max = batt_dis_max * eta_dis

                batt_to_load = min(remaining, batt_to_load_max)
                batt_dis = batt_to_load / max(eta_dis, 1e-9)
                remaining -= batt_to_load

            if remaining > 1e-9 and genset_kW > 0:
                genset = min(genset_kW, remaining)
                remaining -= genset

            if remaining > 1e-9:
                unmet = remaining

        else:
            surplus = pv_left
            if e_max > 0 and batt_p_kW > 0:
                e_room = max(0.0, e_max - e)
                batt_ch_max = min(batt_p_kW, e_room)
                pv_for_batt_max = batt_ch_max / max(eta_ch, 1e-9)

                pv_to_batt = min(surplus, pv_for_batt_max)
                batt_ch = pv_to_batt * eta_ch
                surplus -= pv_to_batt

            pv_left = surplus

        if allow_preventive_genset and e_max > 0 and batt_p_kW > 0:
            soc_after = 100.0 * ((e - batt_dis + batt_ch) / e_max)
            if soc_after < soc_low_pct:
                prev_flag = 1.0
                e_target = e_max * (soc_setpoint_pct / 100.0)
                e_need = max(0.0, e_target - (e - batt_dis + batt_ch))

                batt_ch_room = min(batt_p_kW, max(0.0, e_max - (e - batt_dis + batt_ch)))
                batt_ch_from_genset = min(batt_ch_room, e_need)

                genset_extra = batt_ch_from_genset / max(eta_ch, 1e-9)
                genset_extra = min(genset_kW - genset, max(0.0, genset_extra)) if genset_kW > 0 else 0.0

                batt_ch += genset_extra * eta_ch
                genset += genset_extra

        e = e - batt_dis + batt_ch
        if e_max > 0:
            e = min(max(e, e_min), e_max)
            soc = 100.0 * (e / e_max)
        else:
            soc = 0.0

        pv_available_for_batt = max(0.0, pv_gross - pv_to_load)
        pv_to_batt_pv = min(pv_available_for_batt, batt_ch / max(eta_ch, 1e-9))

        pv_used = pv_to_load + pv_to_batt_pv
        curtail = max(0.0, pv_gross - pv_used)

        out["Load_kW"].append(L)
        out["PV_gross_kW"].append(pv_gross)
        out["PV_to_load_kW"].append(pv_to_load)
        out["PV_to_batt_kW"].append(pv_to_batt_pv)
        out["PV_used_kW"].append(pv_used)
        out["Curtailment_kW"].append(curtail)
        out["Battery_discharge_kW"].append(batt_dis)
        out["Battery_to_load_kW"].append(batt_to_load)
        out["Battery_charge_kW"].append(batt_ch)
        out["Genset_out_kW"].append(genset)
        out["Unmet_kW"].append(unmet)
        out["SOC_pct"].append(soc)
        out["Preventive_genset_flag"].append(prev_flag)

    return pd.DataFrame(out, index=snaps)


# ======================================================================================
# PYPSA NETWORK BUILD
# ======================================================================================
def build_pypsa_pf_network(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_edges: gpd.GeoDataFrame,
    associations_df: pd.DataFrame,
    hourly_total_kW: dict,
    power_factor: float,
    v_nom_kV: float,
    dispatch_df_kW: pd.DataFrame,
    line_r_ohm_per_km: float,
    line_x_ohm_per_km: float,
    line_s_nom_MVA: float,
) -> pypsa.Network:
    net = pypsa.Network()
    net.set_snapshots(SNAPSHOTS)

    gdf_nodes, pole_id_col = ensure_pole_id_column(gdf_nodes)

    # Buses from nodes
    pole_to_bus = {}
    for _, row in gdf_nodes.iterrows():
        pole_id = row[pole_id_col]
        bus_name = f"bus_{pole_id}"
        net.add("Bus", name=bus_name, v_nom=v_nom_kV, carrier="AC")
        pole_to_bus[pole_id] = bus_name

    # Lines from edges
    a_col, b_col = infer_line_endpoints_columns(gdf_edges)

    if "length_m" in [c.lower() for c in gdf_edges.columns]:
        length_col = next(c for c in gdf_edges.columns if c.lower() == "length_m")
        lengths_m = gdf_edges[length_col].astype(float).values
    else:
        lengths_m = gdf_edges.geometry.length.astype(float).values

    for i, row in gdf_edges.reset_index(drop=True).iterrows():
        u = row[a_col]
        v = row[b_col]

        if u not in pole_to_bus or v not in pole_to_bus:
            raise ValueError(
                f"Edge #{i} references missing node id: ({u}, {v}). "
                "Ensure 'source'/'target' match nodes 'id'."
            )

        length_km = float(lengths_m[i]) / 1000.0
        r = float(line_r_ohm_per_km) * length_km
        x = float(line_x_ohm_per_km) * length_km

        net.add(
            "Line",
            name=f"line_{u}_{v}_{i}",
            bus0=pole_to_bus[u],
            bus1=pole_to_bus[v],
            r=r,
            x=x,
            s_nom=line_s_nom_MVA,
            length=length_km,
            carrier="AC",
        )

    # Loads from associations counts per pole
    buildings_per_pole = associations_df.groupby("pole_id").size()

    total_buildings = float(buildings_per_pole.sum())
    if total_buildings <= 0:
        raise ValueError("Associations results in 0 buildings. Check associations.csv.")

    cosphi = float(power_factor)
    sinphi = math.sqrt(max(0.0, 1.0 - cosphi**2))
    tanphi = (sinphi / cosphi) if cosphi > 0 else 0.0

    for pole_id, count in buildings_per_pole.items():
        if pole_id not in pole_to_bus:
            raise ValueError(f"Associations references pole_id not in nodes: {pole_id}")

        bus_name = pole_to_bus[pole_id]
        p_profile_MW = pd.Series(
            [(float(count) * float(hourly_total_kW[h]) / total_buildings) / 1000.0 for h in SNAPSHOTS],
            index=SNAPSHOTS,
        )
        q_profile_MVAr = p_profile_MW * tanphi

        net.add("Load", name=f"Load_{pole_id}", bus=bus_name, p_set=p_profile_MW, q_set=q_profile_MVAr)

    # Slack bus at first node
    first_pole_id = list(pole_to_bus.keys())[0]
    slack_bus = pole_to_bus[first_pole_id]

    pv_used_MW = (dispatch_df_kW["PV_used_kW"] / 1000.0).reindex(SNAPSHOTS)
    diesel_MW = (dispatch_df_kW["Genset_out_kW"] / 1000.0).reindex(SNAPSHOTS)

    net.add("Generator", name="PV_used", bus=slack_bus, control="PQ",
            p_nom=max(1e-6, float(pv_used_MW.max())), p_set=pv_used_MW)
    net.add("Generator", name="Diesel", bus=slack_bus, control="PQ",
            p_nom=max(1e-6, float(diesel_MW.max())), p_set=diesel_MW)

    batt_to_load_MW = (dispatch_df_kW["Battery_to_load_kW"] / 1000.0).reindex(SNAPSHOTS)
    batt_charge_MW = (dispatch_df_kW["Battery_charge_kW"] / 1000.0).reindex(SNAPSHOTS)

    net.add("Generator", name="Battery_discharge", bus=slack_bus, control="PQ",
            p_nom=max(1e-6, float(batt_to_load_MW.max())), p_set=batt_to_load_MW)
    net.add("Load", name="Battery_charge_load", bus=slack_bus,
            p_set=batt_charge_MW, q_set=pd.Series(0.0, index=SNAPSHOTS))

    net.add("Generator", name="Slack", bus=slack_bus, control="Slack", p_nom=1e3)

    return net


# ======================================================================================
# RESULTS TABLES
# ======================================================================================
def build_bus_table_pf(net: pypsa.Network, snapshot: int) -> pd.DataFrame:
    bus_idx = net.buses.index
    v_pu = net.buses_t.v_mag_pu.loc[snapshot].reindex(bus_idx).fillna(np.nan)
    V_nom_V = (net.buses.v_nom.reindex(bus_idx) * 1000.0)
    V_V = v_pu * V_nom_V
    DeltaV_pct = (V_nom_V - V_V) / V_nom_V * 100.0

    P_MW = pd.Series(0.0, index=bus_idx)
    Q_MVAr = pd.Series(0.0, index=bus_idx)

    if hasattr(net, "generators_t") and hasattr(net.generators_t, "p") and not net.generators_t.p.empty:
        gp = net.generators_t.p.loc[snapshot]
        for g, p in gp.items():
            b = net.generators.at[g, "bus"]
            P_MW[b] += float(p)

    if hasattr(net, "loads_t") and hasattr(net.loads_t, "p") and not net.loads_t.p.empty:
        lp = net.loads_t.p.loc[snapshot]
        for l, p in lp.items():
            b = net.loads.at[l, "bus"]
            P_MW[b] -= float(p)

    if hasattr(net, "loads_t") and hasattr(net.loads_t, "q") and not net.loads_t.q.empty:
        lq = net.loads_t.q.loc[snapshot]
        for l, q in lq.items():
            b = net.loads.at[l, "bus"]
            Q_MVAr[b] -= float(q)

    df = pd.DataFrame({
        "Bus": bus_idx,
        "V_V": (V_V.values),
        "DeltaV_%": (DeltaV_pct.values),
        "P_kW": (P_MW.values * 1000.0),
        "Q_kVAr": (Q_MVAr.values * 1000.0),
    })
    df["S_kVA"] = np.sqrt(df["P_kW"] ** 2 + df["Q_kVAr"] ** 2)
    return df.sort_values("Bus").reset_index(drop=True)


def build_line_table_pf(net: pypsa.Network, snapshot: int) -> pd.DataFrame:
    has_p0 = hasattr(net.lines_t, "p0") and not net.lines_t.p0.empty
    has_p1 = hasattr(net.lines_t, "p1") and not net.lines_t.p1.empty
    has_q0 = hasattr(net.lines_t, "q0") and not net.lines_t.q0.empty
    has_q1 = hasattr(net.lines_t, "q1") and not net.lines_t.q1.empty

    rows = []
    for ln in net.lines.index:
        b0 = net.lines.at[ln, "bus0"]
        b1 = net.lines.at[ln, "bus1"]
        length_m = float(net.lines.at[ln, "length"] * 1000.0) if "length" in net.lines.columns else np.nan

        p0 = float(net.lines_t.p0.loc[snapshot, ln]) if has_p0 else 0.0
        p1 = float(net.lines_t.p1.loc[snapshot, ln]) if has_p1 else 0.0
        q0 = float(net.lines_t.q0.loc[snapshot, ln]) if has_q0 else 0.0
        q1 = float(net.lines_t.q1.loc[snapshot, ln]) if has_q1 else 0.0

        if np.isnan(p0): p0 = 0.0
        if np.isnan(p1): p1 = 0.0
        if np.isnan(q0): q0 = 0.0
        if np.isnan(q1): q1 = 0.0

        loss_MW_raw = p0 + p1
        loss_W = max(0.0, loss_MW_raw) * 1e6

        S0_MVA = math.sqrt(p0**2 + q0**2)
        S1_MVA = math.sqrt(p1**2 + q1**2)
        Sflow = max(S0_MVA, S1_MVA)
        loss_pct = (100.0 * (max(0.0, loss_MW_raw) / Sflow)) if Sflow > 1e-12 else 0.0

        rows.append({
            "Line": ln,
            "Bus0": b0,
            "Bus1": b1,
            "Length_m": length_m,
            "P0_kW": p0 * 1000.0,
            "Q0_kVAr": q0 * 1000.0,
            "P1_kW": p1 * 1000.0,
            "Q1_kVAr": q1 * 1000.0,
            "Loss_W": loss_W,
            "Loss_%_of_S": loss_pct,
        })

    df = pd.DataFrame(rows)
    for c in ["P0_kW", "Q0_kVAr", "P1_kW", "Q1_kVAr", "Loss_W", "Loss_%_of_S", "Length_m"]:
        if c in df.columns:
            df[c] = df[c].astype(float).round(2)
    return df


# ======================================================================================
# UI INPUTS
# ======================================================================================
st.subheader("Upload routing outputs")
nodes_file = st.file_uploader("Nodes (mst_nodes.geojson)", type=["geojson", "json", "gpkg"])
edges_file = st.file_uploader("Edges (mst_edges.geojson)", type=["geojson", "json", "gpkg"])
assoc_csv_file = st.file_uploader("Associations (associations.csv)", type=["csv"])

st.subheader("Electrical settings")
c1, c2, c3 = st.columns(3)
v_nom_kV = c1.number_input("LV nominal voltage (kV)", min_value=0.1, max_value=1.0, value=0.4, step=0.01)
power_factor = c2.slider("Load power factor (lagging)", 0.7, 1.0, 0.9)
hour_to_inspect = c3.slider("Hour to inspect (1-24)", 1, 24, 19)

st.subheader("System sizes at slack bus")
s1, s2, s3, s4 = st.columns(4)
pv_p_nom_kW = s1.number_input("PV size (kW)", min_value=0.0, value=250.0, step=10.0)
batt_e_kWh = s2.number_input("Battery energy (kWh)", min_value=0.0, value=500.0, step=50.0)
batt_p_kW = s3.number_input("Battery power (kW)", min_value=0.0, value=250.0, step=10.0)
genset_kW = s4.number_input("Genset size (kW)", min_value=0.0, value=90.0, step=10.0)

st.subheader("Battery & dispatch parameters")
d1, d2, d3, d4, d5 = st.columns(5)
eta_ch = d1.number_input("Battery charge eff", min_value=0.50, max_value=1.0, value=0.95, step=0.01)
eta_dis = d2.number_input("Battery discharge eff", min_value=0.50, max_value=1.0, value=0.95, step=0.01)
DOD = d3.number_input("DOD (usable fraction)", min_value=0.10, max_value=1.0, value=0.80, step=0.05)
soc_init_pct = d4.number_input("Initial SOC (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
soc_low_pct = d5.number_input("SOC low threshold (%)", min_value=0.0, max_value=100.0, value=25.0, step=5.0)

d6, d7 = st.columns(2)
soc_setpoint_pct = d6.number_input("Preventive genset SOC setpoint (%)", min_value=0.0, max_value=100.0, value=45.0, step=5.0)
allow_preventive_genset = d7.checkbox("Enable preventive genset charging", value=True)

st.subheader("Line electrical parameters (simplified)")
l1, l2, l3 = st.columns(3)
line_r = l1.number_input("R (ohm/km)", min_value=0.0, value=0.642, step=0.001, format="%.3f")
line_x = l2.number_input("X (ohm/km)", min_value=0.0, value=0.083, step=0.001, format="%.3f")
line_s_nom = l3.number_input("Line rating s_nom (MVA)", min_value=0.01, value=0.2, step=0.01, format="%.2f")

show_debug = st.checkbox("Show debug tables", value=False)


# ======================================================================================
# RUN PF
# ======================================================================================
st.subheader("Run (Dispatch -> PF)")

if st.button("Run PF", type="primary"):
    if nodes_file is None or edges_file is None or assoc_csv_file is None:
        st.error("Please upload Nodes, Edges, and Associations CSV.")
        st.stop()

    try:
        gdf_nodes = load_and_transform_geodata(nodes_file)
        gdf_edges = load_and_transform_geodata(edges_file)
        associations_df = load_associations_csv(assoc_csv_file)
    except Exception as e:
        st.error(f"Input loading error: {e}")
        st.stop()

    load_series = pd.Series([DEFAULT_HOURLY_TOTAL_KW[h] for h in SNAPSHOTS], index=SNAPSHOTS, dtype=float)
    pvpu_series = pd.Series([DEFAULT_HOURLY_PV_PU[h] for h in SNAPSHOTS], index=SNAPSHOTS, dtype=float)

    dispatch_df = simulate_dispatch_24h(
        load_kW=load_series,
        pv_pu=pvpu_series,
        pv_nom_kW=pv_p_nom_kW,
        batt_e_kWh=batt_e_kWh,
        batt_p_kW=batt_p_kW,
        genset_kW=genset_kW,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
        dod=DOD,
        soc_init_pct=soc_init_pct,
        soc_low_pct=soc_low_pct,
        soc_setpoint_pct=soc_setpoint_pct,
        allow_preventive_genset=allow_preventive_genset,
    )

    st.success("Dispatch completed ✔")
    st.dataframe(dispatch_df.round(3))

    try:
        net = build_pypsa_pf_network(
            gdf_nodes=gdf_nodes,
            gdf_edges=gdf_edges,
            associations_df=associations_df,
            hourly_total_kW=DEFAULT_HOURLY_TOTAL_KW,
            power_factor=power_factor,
            v_nom_kV=v_nom_kV,
            dispatch_df_kW=dispatch_df,
            line_r_ohm_per_km=line_r,
            line_x_ohm_per_km=line_x,
            line_s_nom_MVA=line_s_nom,
        )
    except Exception as e:
        st.error(f"Network build error: {e}")
        st.stop()

    if show_debug:
        st.write("Buses:", net.buses.shape, "Lines:", net.lines.shape, "Loads:", net.loads.shape, "Generators:", net.generators.shape)
        st.write("Loads:")
        st.dataframe(net.loads)
        st.write("Generators:")
        st.dataframe(net.generators)
        st.write("Lines:")
        st.dataframe(net.lines)

    st.write("Running AC power flow (pf) for 24h ...")
    net.pf()
    st.success("PF completed ✔")

    snapshot = int(hour_to_inspect)

    st.subheader(f"Bus results (hour {snapshot})")
    bus_df = build_bus_table_pf(net, snapshot).round(2)
    st.dataframe(bus_df)

    st.subheader(f"Line flows & losses (hour {snapshot})")
    line_df = build_line_table_pf(net, snapshot)
    st.dataframe(line_df)

    total_loss_W = float(line_df["Loss_W"].sum()) if "Loss_W" in line_df.columns else 0.0
    total_loss_kW = total_loss_W / 1000.0

    total_load_kW = (
        float(net.loads_t.p.loc[snapshot].sum() * 1000.0)
        if hasattr(net.loads_t, "p") and not net.loads_t.p.empty
        else float(load_series.loc[snapshot])
    )

    gen_p_kW = (
        float(net.generators_t.p.loc[snapshot].sum() * 1000.0)
        if hasattr(net.generators_t, "p") and not net.generators_t.p.empty
        else np.nan
    )

    # Loss percentage relative to total generation at this snapshot
    loss_pct_of_gen = (100.0 * total_loss_kW / gen_p_kW) if (gen_p_kW is not None and not np.isnan(gen_p_kW) and gen_p_kW > 1e-9) else np.nan

    st.subheader(f"Totals (hour {snapshot})")
    a, b, c, d = st.columns(4)
    a.metric("Total load (kW)", f"{total_load_kW:.3f}")
    b.metric("Total line losses (kW)", f"{total_loss_kW:.3f}")
    c.metric("Losses / Generation (%)", f"{loss_pct_of_gen:.2f}" if not np.isnan(loss_pct_of_gen) else "n/a")
    d.metric("Total generation (kW)", f"{gen_p_kW:.3f}" if not np.isnan(gen_p_kW) else "n/a")
    st.subheader("Download PF outputs")
    st.download_button("Download dispatch CSV", dispatch_df.to_csv().encode("utf-8"), "dispatch_24h.csv")
    st.download_button("Download bus PF CSV", bus_df.to_csv(index=False).encode("utf-8"), "bus_pf.csv")
    st.download_button("Download line PF CSV", line_df.to_csv(index=False).encode("utf-8"), "lines_pf.csv")
