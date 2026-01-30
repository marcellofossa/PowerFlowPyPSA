"""
pftoolv2 - Power Flow tool (Streamlit)

Inputs (uploads):
- PF Excel workbook (.xlsx) with sheets:
    - Network: global electrical parameters (v_nom, power_factor, line_r, line_x, slack_pole_id, ph_at_slack, crs_epsg, ...)
      Format: columns [Parameter, Value].
    - Dispatch: hourly time series (Preliminary Sizning Tool-style columns supported).
- Nodes GeoJSON: poles (points). Must contain an id column (preferred: "id" or "pole_id"), Distribution Tool output supperted.
- Edges GeoJSON: LV network segments (lines) with endpoints (preferred: "source"/"target"), Distribution Tool output supperted.
- Associations CSV: building-to-pole mapping (building_id, pole_id), Distribution Tool output supperted.

Outputs:
- Nodal voltages table:
    V_abs_V, DeltaV_V (V - Vnom), DeltaV_pct
- Branch currents table:
    I0_A, I1_A, Imax_A (computed from line-end apparent power and bus voltage)

Notes:
- This tool runs an AC power flow for a SINGLE selected snapshot (hour) to keep runtime reasonable in Streamlit.
- Loads are allocated across poles proportionally to the number of buildings mapped to each pole_id (from associations.csv).
- PV/Genset/Battery are aggregated at the slack bus (pole_id = slack_pole_id).
- Households and businesses power loads are evenly distribuited among ALL buildings, PH load can be allocated at the slack bus through the input excel.
"""

import io
import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import pypsa


# ======================================================================================
# Helpers for robust Excel parsing
# ======================================================================================

def _is_nan(x) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _to_float(x, *, name: str) -> float:
    """Convert Excel values to float (supports decimal comma like '0,415')."""
    if x is None or _is_nan(x):
        raise ValueError(f"Network parameter '{name}' is missing.")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"Network parameter '{name}' cannot be parsed as float: {x}") from e


def _to_int(x, *, name: str) -> int:
    """Convert Excel values to int (accepts '0', 0.0, etc.)."""
    return int(round(_to_float(x, name=name)))


def _to_bool(x, *, name: str, default: bool = False) -> bool:
    """Convert Excel values to bool (supports TRUE/FALSE and VERO/FALSO)."""
    if x is None or _is_nan(x):
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "vero")


# ======================================================================================
# Basic file readers
# ======================================================================================

def read_geojson(uploaded_file) -> gpd.GeoDataFrame:
    """Read a GeoJSON file uploaded via Streamlit into a GeoDataFrame."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix not in [".geojson", ".json"]:
        raise ValueError("Please upload a .geojson (or .json) file.")
    return gpd.read_file(io.BytesIO(uploaded_file.getvalue()))



def load_and_project_geodata(uploaded_file, target_epsg: int) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame and project it to a metric CRS for length calculations."""
    gdf = read_geojson(uploaded_file)
    if gdf.crs is None:
        raise ValueError("Geo file has no CRS. Re-export with a CRS (e.g., EPSG:32633).")
    if gdf.crs.to_epsg() != int(target_epsg):
        gdf = gdf.to_crs(epsg=int(target_epsg))
    gdf = gdf[gdf.is_valid].copy()
    return gdf


def load_associations_csv(uploaded_file) -> pd.DataFrame:
    """
    Load associations.csv and normalize columns to: pole_id, building_id
    Accepted variants: pole_id/pole and building_id/building.
    """
    df = pd.read_csv(uploaded_file)
    cols_l = {c.lower(): c for c in df.columns}

    pole_col = cols_l.get("pole_id", cols_l.get("pole", None))
    bld_col = cols_l.get("building_id", cols_l.get("building", None))

    if pole_col is None or bld_col is None:
        raise ValueError("associations.csv must have columns pole_id (or pole) and building_id (or building).")

    out = df[[pole_col, bld_col]].rename(columns={pole_col: "pole_id", bld_col: "building_id"}).copy()
    out["pole_id"] = pd.to_numeric(out["pole_id"], errors="raise").astype(int)
    return out


def ensure_pole_id_column(gdf_nodes: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    """Ensure nodes have a 'pole_id' column; fall back to 'id' or index."""
    cols_l = [c.lower() for c in gdf_nodes.columns]
    if "pole_id" in cols_l:
        pole_col = next(c for c in gdf_nodes.columns if c.lower() == "pole_id")
        return gdf_nodes, pole_col

    gdf_nodes = gdf_nodes.copy()
    if "id" in cols_l:
        id_col = next(c for c in gdf_nodes.columns if c.lower() == "id")
        gdf_nodes["pole_id"] = gdf_nodes[id_col]
    else:
        gdf_nodes["pole_id"] = gdf_nodes.index

    return gdf_nodes, "pole_id"


def infer_edge_endpoints(gdf_edges: gpd.GeoDataFrame) -> tuple[str, str]:
    """Detect endpoint columns for edges (supports source/target and common variants)."""
    candidates = [("source", "target"), ("bus0", "bus1"), ("from", "to"), ("u", "v")] # source and target are currently used
    cols = set([c.lower() for c in gdf_edges.columns])
    for a, b in candidates:
        if a in cols and b in cols:
            a_real = next(c for c in gdf_edges.columns if c.lower() == a)
            b_real = next(c for c in gdf_edges.columns if c.lower() == b)
            return a_real, b_real
    raise ValueError("Edges file must contain endpoints columns (e.g., source/target).")


# ======================================================================================
# Excel reader (Network + Dispatch)
# ======================================================================================

def read_pf_workbook(uploaded_xlsx) -> tuple[dict, pd.DataFrame]:
    """Read Network parameters and Dispatch time series from the PF Excel workbook."""
    xls = pd.ExcelFile(uploaded_xlsx)
    if "Network" not in xls.sheet_names or "Dispatch" not in xls.sheet_names:
        raise ValueError("Excel workbook must contain sheets named 'Network' and 'Dispatch'.")

    net_df = pd.read_excel(xls, "Network")
    disp_raw = pd.read_excel(xls, "Dispatch")

    if not {"Parameter", "Value"}.issubset(set(net_df.columns)):
        raise ValueError("Sheet 'Network' must contain columns: Parameter, Value (Unit optional).")

    # Raw mapping: Parameter -> Value
    # Build a raw dictionary mapping network parameter names to their Excel values.
    # The Excel sheet is expected to have columns like: Parameter | Value | Unit.
    # Each row is read and stored as:
    #   raw[Parameter] = Value
    # Whitespace around parameter names is stripped.
    # Empty parameter cells are ignored.
    # If the same parameter appears multiple times, the last occurrence overwrites the previous one.             
    raw = {}
    for _, r in net_df.iterrows():
        key = str(r["Parameter"]).strip()
        if key:
            raw[key] = r["Value"]

    # Require these keys to exist (no silent defaults)
    required = ["v_nom", "power_factor", "line_r", "line_x", "slack_pole_id", "crs_epsg"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing required Network parameters in Excel: {missing}")

    cfg = {
        "v_nom_kV": _to_float(raw["v_nom"], name="v_nom"),
        "power_factor": _to_float(raw["power_factor"], name="power_factor"),
        "line_r_ohm_per_km": _to_float(raw["line_r"], name="line_r"),
        "line_x_ohm_per_km": _to_float(raw["line_x"], name="line_x"),
        "slack_pole_id": _to_int(raw["slack_pole_id"], name="slack_pole_id"),
        "ph_at_slack": _to_bool(raw.get("ph_at_slack", None), name="ph_at_slack", default=True),
        "crs_epsg": _to_int(raw["crs_epsg"], name="crs_epsg"),
    }
    return cfg, disp_raw


def normalize_dispatch(disp_raw: pd.DataFrame, reduce_load_by_unmet: bool = True) -> pd.DataFrame:
    """
    Produce a minimal dispatch dataframe with kW columns used by PF:
      - Load_kW (served)
      - PV_used_kW
      - Genset_out_kW
      - Battery_to_load_kW
      - Battery_charge_kW

    Supports MicroGridsPy-style columns (your case):
      hour,
      Load HHs + Buzs, Load PH,
      PV Net Production or PV to Load (+ PV to Battery),
      Genset to Load (+ Genset to Battery),
      Battery to Load,
      PV to Battery, Genset to Battery,
      Unmet Demand (optional)
    """
    df = disp_raw.copy()
    # Normalize column lookup: lowercase trimmed -> real name
    cols_map = {str(c).lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            key = str(n).lower().strip()
            if key in cols_map:
                return cols_map[key]
        return None

    # --- Snapshots (hours) ---
    hour_col = pick("hour")
    if hour_col is not None:
        df = df.sort_values(hour_col).reset_index(drop=True)
        snapshots = df[hour_col].astype(int).tolist()
    else:
        snapshots = list(range(1, len(df) + 1))
        df = df.reset_index(drop=True)

    out = pd.DataFrame(index=snapshots)

    # --- Load total (kW) ---
    vill_col = pick("load hhs + buzs")
    ph_col = pick("load ph")
    total_load = None

    if vill_col is not None:
        total_load = df[vill_col].astype(float).fillna(0.0)
        if ph_col is not None:
            total_load = total_load + df[ph_col].astype(float).fillna(0.0)
    else:
        # Fallback to a generic "Load" column if present
        load_col = pick("load")
        if load_col is None:
            raise ValueError("Dispatch must include either 'Load HHs + Buzs' (and optionally 'Load PH') or a 'Load' column.")
        total_load = df[load_col].astype(float).fillna(0.0)

    # Served load adjustment
    if reduce_load_by_unmet:
        unmet_col = pick("unmet demand", "unmet_demand")
        if unmet_col is not None:
            total_load = (total_load - df[unmet_col].astype(float).fillna(0.0)).clip(lower=0.0)

    out["Load_kW"] = pd.Series(total_load.values, index=out.index)

    # --- PV used (kW) ---
    pv_net = pick("pv net production")
    if pv_net is not None:
        pv_used = df[pv_net].astype(float).fillna(0.0)
    else:
        pv2l = pick("pv to load")
        pv2b = pick("pv to battery")
        if pv2l is None and pv2b is None:
            pv_used = pd.Series(0.0, index=df.index)
        else:
            pv_used = (df[pv2l].astype(float).fillna(0.0) if pv2l else 0.0) + (df[pv2b].astype(float).fillna(0.0) if pv2b else 0.0)

    out["PV_used_kW"] = pd.Series(pv_used.values, index=out.index)

    # --- Genset out (kW) ---
    g2l = pick("genset to load")
    g2b = pick("genset to battery")
    if g2l is None and g2b is None:
        gen = pd.Series(0.0, index=df.index)
    else:
        gen = (df[g2l].astype(float).fillna(0.0) if g2l else 0.0) + (df[g2b].astype(float).fillna(0.0) if g2b else 0.0)
    out["Genset_out_kW"] = pd.Series(gen.values, index=out.index)

    # --- Battery discharge (kW) ---
    b2l = pick("battery to load")
    if b2l is None:
        raise ValueError("Dispatch must include 'Battery to Load'")
    out["Battery_to_load_kW"] = pd.Series(df[b2l].astype(float).fillna(0.0).values, index=out.index)

    # --- Battery charge (kW): PV to Battery + Genset to Battery ---
    pv2b = pick("pv to battery")
    g2b = pick("genset to battery")
    pv_part = df[pv2b].astype(float).fillna(0.0) if pv2b is not None else 0.0
    gen_part = df[g2b].astype(float).fillna(0.0) if g2b is not None else 0.0
    out["Battery_charge_kW"] = pd.Series((pv_part + gen_part).values, index=out.index)

    return out


# ======================================================================================
# PyPSA network build + PF
# ======================================================================================

def build_network(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_edges: gpd.GeoDataFrame,
    associations: pd.DataFrame,
    dispatch_kW: pd.DataFrame,
    cfg: dict,
) -> pypsa.Network:
    """Build a PyPSA AC network for PF using node/edge geometry and dispatch time series."""
    net = pypsa.Network()
    snapshots = dispatch_kW.index.tolist()
    net.set_snapshots(snapshots)

    # Standardize IDs
    gdf_nodes, pole_col = ensure_pole_id_column(gdf_nodes)
    u_col, v_col = infer_edge_endpoints(gdf_edges)

    # Create buses
    pole_to_bus = {}
    for _, r in gdf_nodes.iterrows():
        pole_id = int(r[pole_col])
        bus_name = f"bus_{pole_id}"
        net.add("Bus", name=bus_name, v_nom=cfg["v_nom_kV"], carrier="AC")
        pole_to_bus[pole_id] = bus_name

    # Compute edge lengths in meters (geometry), then convert to km
    gdf_edges = gdf_edges.copy()
    gdf_edges["length_m"] = gdf_edges.geometry.length.astype(float)

    # Drop zero-length edges (these behave like short circuits and can break PF)
    gdf_edges = gdf_edges[gdf_edges["length_m"] > 1e-6].copy()

    # Create lines
    for i, r in gdf_edges.reset_index(drop=True).iterrows():
        u = int(r[u_col])
        v = int(r[v_col])
        if u not in pole_to_bus or v not in pole_to_bus:
            raise ValueError(f"Edge #{i} references missing node id: ({u}, {v})")

        length_km = float(r["length_m"]) / 1000.0
        r_ohm = cfg["line_r_ohm_per_km"] * length_km
        x_ohm = cfg["line_x_ohm_per_km"] * length_km

        net.add(
            "Line",
            name=f"line_{u}_{v}_{i}",
            bus0=pole_to_bus[u],
            bus1=pole_to_bus[v],
            r=float(r_ohm),
            x=float(x_ohm),
            length=float(length_km),
            carrier="AC",
        )

    # Allocate served load across poles proportionally to number of buildings per pole
    bcount = associations.groupby("pole_id").size()
    total_buildings = float(bcount.sum())
    if total_buildings <= 0:
        raise ValueError("associations.csv results in 0 mapped buildings.")

    cosphi = float(cfg["power_factor"])
    sinphi = math.sqrt(max(0.0, 1.0 - cosphi**2))
    tanphi = (sinphi / cosphi) if cosphi > 0 else 0.0

    for pole_id, n_bld in bcount.items():
        pole_id = int(pole_id)
        if pole_id not in pole_to_bus:
            raise ValueError(f"associations.csv references pole_id not in nodes: {pole_id}")

        share = float(n_bld) / total_buildings
        p_MW = (dispatch_kW["Load_kW"] * share) / 1000.0
        q_MVAr = p_MW * tanphi

        net.add(
            "Load",
            name=f"Load_{pole_id}",
            bus=pole_to_bus[pole_id],
            p_set=p_MW,
            q_set=q_MVAr,
        )

    # Slack bus and aggregated sources (PV, genset, battery discharge) + battery charge load
    slack_pole_id = int(cfg["slack_pole_id"])
    if slack_pole_id not in pole_to_bus:
        raise ValueError(f"slack_pole_id={slack_pole_id} not found among node pole_id values.")
    slack_bus = pole_to_bus[slack_pole_id]

    pv_MW = dispatch_kW["PV_used_kW"] / 1000.0
    gen_MW = dispatch_kW["Genset_out_kW"] / 1000.0
    bdis_MW = dispatch_kW["Battery_to_load_kW"] / 1000.0
    bch_MW = dispatch_kW["Battery_charge_kW"] / 1000.0

    def add_pq_generator(name: str, p_series_MW: pd.Series):
        p_nom = max(1e-6, float(p_series_MW.max()))
        net.add("Generator", name=name, bus=slack_bus, control="PQ", p_nom=p_nom, p_set=p_series_MW)

    add_pq_generator("PV_used", pv_MW)
    add_pq_generator("Genset", gen_MW)
    add_pq_generator("Battery_discharge", bdis_MW)

    # Battery charge is modeled as an extra load at slack
    net.add("Load", name="Battery_charge", bus=slack_bus, p_set=bch_MW, q_set=pd.Series(0.0, index=snapshots))

    # Slack element to balance the network (ensures PF solvability)
    net.add("Generator", name="Slack", bus=slack_bus, control="Slack", p_nom=1e3)

    return net


# ======================================================================================
# Results tables
# ======================================================================================

def nodal_voltage_table(net: pypsa.Network, snapshot) -> pd.DataFrame:
    """Return nodal voltage results (absolute and deviations from nominal)."""
    buses = net.buses.index
    v_nom_V = net.buses.v_nom.reindex(buses).astype(float) * 1000.0  # kV -> V (line-to-line)
    v_pu = net.buses_t.v_mag_pu.loc[snapshot].reindex(buses).astype(float)

    v_abs_V = v_pu * v_nom_V
    delta_V = v_abs_V - v_nom_V
    delta_pct = 100.0 * delta_V / v_nom_V

    df = pd.DataFrame({
        "Bus": buses,
        "V_abs_V": v_abs_V.values,
        "DeltaV_V": delta_V.values,
        "DeltaV_pct": delta_pct.values,
    })
    return df.sort_values("Bus").reset_index(drop=True)


def branch_current_table(net: pypsa.Network, snapshot) -> pd.DataFrame:
    """
    Compute branch currents from line-end apparent power and bus voltage:
      I = |S| / (sqrt(3) * V_LL)
    (assumes 3-phase line-to-line voltage in net.buses.v_nom)
    """
    rows = []
    sqrt3 = math.sqrt(3.0)

    v_pu = net.buses_t.v_mag_pu.loc[snapshot]
    v_LL = (net.buses.v_nom * 1000.0) * v_pu  # V

    for ln in net.lines.index:
        b0 = net.lines.at[ln, "bus0"]
        b1 = net.lines.at[ln, "bus1"]

        p0 = float(net.lines_t.p0.loc[snapshot, ln])  # MW
        q0 = float(net.lines_t.q0.loc[snapshot, ln])  # MVAr
        p1 = float(net.lines_t.p1.loc[snapshot, ln])
        q1 = float(net.lines_t.q1.loc[snapshot, ln])

        s0_MVA = math.sqrt(p0**2 + q0**2)
        s1_MVA = math.sqrt(p1**2 + q1**2)

        v0 = float(v_LL.loc[b0])
        v1 = float(v_LL.loc[b1])

        i0 = (s0_MVA * 1e6) / (sqrt3 * v0) if v0 > 1e-6 else np.nan
        i1 = (s1_MVA * 1e6) / (sqrt3 * v1) if v1 > 1e-6 else np.nan

        rows.append({
            "Line": ln,
            "Bus0": b0,
            "Bus1": b1,
            "I0_A": i0,
            "I1_A": i1,
            "Imax_A": np.nanmax([i0, i1]),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Line").reset_index(drop=True)


# ======================================================================================
# Streamlit App
# ======================================================================================

st.set_page_config(page_title="pftoolv2", layout="wide")
st.title("pftoolv2 - Slim Power Flow (Excel + GeoJSON + Associations)")

st.subheader("1) Upload inputs")
pf_excel = st.file_uploader("PF Excel workbook (.xlsx) with sheets: Network, Dispatch", type=["xlsx", "xls"])
nodes_file = st.file_uploader("Nodes (mst_nodes.geojson / .gpkg)", type=["geojson", "json", "gpkg"])
edges_file = st.file_uploader("Edges (mst_edges.geojson / .gpkg)", type=["geojson", "json", "gpkg"])
assoc_file = st.file_uploader("Associations (associations.csv)", type=["csv"])

reduce_load_by_unmet = st.checkbox("Use served load = Load - Unmet Demand (if present)", value=True)

if pf_excel and nodes_file and edges_file and assoc_file:
    try:
        cfg, disp_raw = read_pf_workbook(pf_excel)
        dispatch_kW = normalize_dispatch(disp_raw, reduce_load_by_unmet=reduce_load_by_unmet)

        gdf_nodes = load_and_project_geodata(nodes_file, cfg["crs_epsg"])
        gdf_edges = load_and_project_geodata(edges_file, cfg["crs_epsg"])
        associations = load_associations_csv(assoc_file)

        st.success(f"Inputs loaded. Snapshots: {len(dispatch_kW)}. slack_pole_id={cfg['slack_pole_id']}")
        st.write("Network config (from Excel):", cfg)
    except Exception as e:
        st.error(f"Input error: {repr(e)}")
        st.stop()

    st.subheader("2) Select snapshot and run PF")
    snap = st.selectbox("Snapshot (hour index)", dispatch_kW.index.tolist(), index=0)

    if st.button("Run Power Flow", type="primary"):
        try:
            # Build network for ALL snapshots (required by PyPSA), but solve PF only for the selected one.
            net = build_network(gdf_nodes, gdf_edges, associations, dispatch_kW, cfg)
            net.pf(snapshots=[snap])
        except Exception as e:
            st.error(f"PF error: {repr(e)}")
            st.stop()

        st.success("PF completed.")

        st.subheader("Nodal voltages")
        vtab = nodal_voltage_table(net, snap).round(6)
        st.dataframe(vtab, use_container_width=True)

        st.subheader("Branch currents")
        itab = branch_current_table(net, snap).round(6)
        st.dataframe(itab, use_container_width=True)

        st.subheader("Downloads")
        st.download_button("Download nodal voltages CSV", vtab.to_csv(index=False).encode("utf-8"), "nodal_voltages.csv")
        st.download_button("Download branch currents CSV", itab.to_csv(index=False).encode("utf-8"), "branch_currents.csv")
else:
    st.info("Upload the PF Excel workbook, nodes, edges, and associations to enable the PF run.")
