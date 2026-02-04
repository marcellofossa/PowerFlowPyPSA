"""
pftoolV8 - Power Flow tool (PyPSA + Streamlit)

Inputs (uploads):
- PF Excel workbook (.xlsx) with sheets:
    - Network: global electrical parameters (v_nom, power_factor, line_r, line_x, slack_pole_id, ph_at_slack, crs_epsg, ...)
      Format: columns [Parameter, Value].
    - Dispatch: hourly time series (Preliminary Sizning Tool-style columns supported).
- Nodes GeoJSON/GPKG: poles (points). Must contain an id column (preferred: "id" or "pole_id").
- Edges GeoJSON/GPKG: LV network segments (lines) with endpoints (preferred: "source"/"target").
- Associations CSV: building-to-pole mapping (building_id, pole_id).

Outputs:
- Nodal voltages table:
    V_abs_V, DeltaV_V (V - Vnom), DeltaV_pct
- Branch currents table:
    I0_A, I1_A, Imax_A (computed from line-end apparent power and bus voltage)

Notes:
- This tool runs an AC power flow for a SINGLE selected snapshot (hour) to keep runtime reasonable in Streamlit.
- Loads are allocated across poles proportionally to the number of buildings mapped to each pole_id (from associations.csv).
- PV/Genset/Battery are aggregated at the slack bus (pole_id = slack_pole_id).
"""

import io
import os
import tempfile
import math
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

# ======================================================================================
# Helpers: robust bus sorting + snapshot resolution
# ======================================================================================

_bus_num_re = re.compile(r"(\d+)$")

def _bus_sort_key(label) -> int:
    """
    Aim
        Provide a stable *numeric* sorting key for bus labels such as 'bus_12'.

    How it works
        Extracts the trailing integer from the label using a regular expression.
        If no trailing number exists, it returns a very large value so those labels
        are pushed to the end of the sorted list.

    Why it is useful
        Streamlit tables (and pandas) default to lexicographic sorting for strings,
        which produces the undesired order: bus_1, bus_10, bus_100, ...
        Using this key yields the human-expected order: bus_1, bus_2, ... bus_10.
    """
    s = str(label)
    m = _bus_num_re.search(s)
    return int(m.group(1)) if m else 10**18

def _sorted_bus_index(bus_index):
    return sorted(list(bus_index), key=_bus_sort_key)

def _resolve_snapshot(net: pypsa.Network, snap):
    """
    Aim
        Ensure the snapshot label requested by the user matches the type/labels
        stored inside `net.snapshots`.

    How it works
        Tries:
          1) direct membership (e.g., 19 in snapshots)
          2) string cast (e.g., "19" in snapshots)
          3) int(float(...)) for Excel-like values such as 19.0

    Why it is useful
        A very common source of 'all NaN' (main issue with a big net) results is indexing time-series tables with
        a snapshot key of the wrong type. This helper prevents that class of errors.
    """
    if snap in net.snapshots:
        return snap
    s = str(snap)
    if s in net.snapshots:
        return s
    # try int conversion
    try:
        i = int(float(snap))
        if i in net.snapshots:
            return i
        si = str(i)
        if si in net.snapshots:
            return si
    except Exception:
        pass
    raise ValueError(
        f"Selected snapshot {snap!r} not found in net.snapshots. " 
        f"Example snapshots: {list(net.snapshots[:10])}"
    )
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
    """
    Aim
        Parse a numeric value coming from Excel into a Python float.

    How it works
        - Rejects None/NaN with a clear error message naming the parameter.
        - Accepts numeric types directly.
        - Accepts strings and supports the European decimal comma (e.g., '0,415').

    Why it is useful
        Prevents silent coercions and makes configuration errors obvious early,
        before building the network and running the power flow.
    """
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
    """
    Aim
        Parse an Excel value into an integer.

    How it works
        Uses `_to_float` and rounds before casting to int (handles 0, 0.0, "0").

    Why it is useful
        Parameters such as `slack_pole_id` and `crs_epsg` are logically integers;
        keeping them as int avoids mismatches and indexing bugs.
    """
    return int(round(_to_float(x, name=name)))


def _to_bool(x, *, name: str, default: bool = False) -> bool:
    """
    Aim
        Parse Excel-like boolean fields into a Python bool.

    How it works
        Supports TRUE/FALSE, 1/0, yes/no, and Italian VERO/FALSO.
        Missing values default to the provided `default`.

    Why it is useful
        Makes optional toggles resilient to the different ways spreadsheets encode booleans.
    """
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
    """
    Aim
        Read an uploaded geographic vector file into a GeoDataFrame.

    How it works
        - For .geojson/.json: reads directly from bytes. NB: should be .geojson
        - For .gpkg: writes to a temporary file (GeoPackage drivers usually require a path)
          and then reads it with GeoPandas.

    Why it is useful
        The tool supports both GeoJSON and GeoPackage exports from GIS tools,
        reducing friction for different workflows.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    data = uploaded_file.getvalue()
    if suffix in [".geojson", ".json"]:
        return gpd.read_file(io.BytesIO(data))
    if suffix == ".gpkg":
        # GeoPackage usually requires a real file path
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return gpd.read_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    raise ValueError("Please upload a .geojson/.json or .gpkg file.")


def load_and_project_geodata(uploaded_file, target_epsg: int) -> gpd.GeoDataFrame:
    """
    Aim
        Load a geodata file and ensure it is in a metric CRS suitable for distance/length.

    How it works
        - Loads the file with `read_geojson`.
        - Verifies a CRS is present.
        - Reprojects to `target_epsg` when needed.
        - Drops invalid geometries.

    Why it is useful
        Line lengths are used to compute electrical impedances. If the CRS is not metric
        (or is inconsistent), computed lengths become wrong and the PF can diverge.
    """
    gdf = read_geojson(uploaded_file)
    if gdf.crs is None:
        raise ValueError("Geo file has no CRS. Re-export with a CRS (e.g., EPSG:32633).")
    if gdf.crs.to_epsg() != int(target_epsg):
        gdf = gdf.to_crs(epsg=int(target_epsg))
    gdf = gdf[gdf.is_valid].copy()
    return gdf


def load_associations_csv(uploaded_file) -> pd.DataFrame:
    """
    Aim
        Load the building-to-pole mapping (associations.csv) and normalize its schema.

    How it works
        - Accepts common column variants (pole_id/pole, building_id/building).
        - Coerces pole_id to integer and rejects missing/non-numeric rows with examples.

    Why it is useful
        The load allocation step relies on integer pole IDs. Early validation prevents
        hidden NaNs that later cause IntCasting errors or misallocated loads.
    """
    df = pd.read_csv(uploaded_file)
    cols_l = {c.lower(): c for c in df.columns}

    pole_col = cols_l.get("pole_id", cols_l.get("pole", None))
    bld_col = cols_l.get("building_id", cols_l.get("building", None))

    if pole_col is None or bld_col is None:
        raise ValueError("associations.csv must have columns pole_id (or pole) and building_id (or building).")

    out = df[[pole_col, bld_col]].rename(columns={pole_col: "pole_id", bld_col: "building_id"}).copy()
    pid = pd.to_numeric(out["pole_id"], errors="coerce")
    bad = pid.isna() | ~np.isfinite(pid)
    if bad.any():
        example = out.loc[bad, ["pole_id", "building_id"]].head(10).to_dict(orient="records")
        raise ValueError(
            f"associations.csv: {int(bad.sum())} rows have missing/non-numeric pole_id. Examples (up to 10): {example}"
        )
    out["pole_id"] = pid.astype(int)
    return out


def ensure_pole_id_column(gdf_nodes: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    """
    Aim
        Guarantee that the nodes layer contains a usable pole identifier column.

    How it works
        - Uses an existing 'pole_id' column when present.
        - Otherwise falls back to an 'id' column (common in exports).
        - Otherwise creates pole_id from the GeoDataFrame index.

    Why it is useful
        The entire network construction (buses, load mapping, slack selection) depends on
        a consistent pole identifier. There were issues due to "bus_12" =! "12".
    """
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
    """
    Aim
        Detect which columns identify the endpoints of each edge/line segment.

    How it works
        Searches for common endpoint column pairs:
          (source,target), (bus0,bus1), (from,to), (u,v) N.B.: currently (source,target)

    Why it is useful
        Makes the tool robust to different naming conventions from GIS/export pipelines.
    """
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
    """
    Aim
        Parse the PF Excel workbook into:
        - `cfg`: a dictionary of network parameters
        - `disp_raw`: the raw Dispatch sheet as a DataFrame

    How it works
        - Validates the presence of 'Network' and 'Dispatch' sheets.
        - Reads Network rows formatted as Parameter/Value.
        - Validates required parameters and parses them with `_to_float/_to_int/_to_bool`.

    Why it is useful
        The possibulity to pass a dictionary instead of 8 parameters makes the code more compact and cleaner.
    """
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
    Aim
        Convert the raw Dispatch sheet into a standardized time-series table
        that the PF model consumes.

    How it works
        - Detects column names (case-insensitive) for the supported schema.
        - Builds an index of snapshots (hours).
        - Produces:
            Load_kW (served load, optionally reduced by Unmet Demand),
            PV_used_kW, Genset_out_kW, Battery_to_load_kW, Battery_charge_kW.

    Why it is useful
        Lets you reuse the same PF model across different simulation outputs
        (MicroGridsPy-like tables, or simplified tables).
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
        h = pd.to_numeric(df[hour_col], errors="coerce")
        bad = h.isna() | ~np.isfinite(h)
        if bad.any():
            # Common in Excel: "ghost" rows where some other column has a 0, but hour is blank.
            st.warning(f"Dispatch: dropping {int(bad.sum())} rows with missing/non-numeric hour values.")
            df = df.loc[~bad].copy()
            h = h.loc[~bad]
        df = df.assign(**{hour_col: h.astype(int)}).sort_values(hour_col).reset_index(drop=True)
        snapshots = df[hour_col].tolist()
    else:
        snapshots = list(range(1, len(df) + 1))
        df = df.reset_index(drop=True)

    out = pd.DataFrame(index=snapshots)

    # --- Load total (kW) ---
    build_load = pick("load hhs + buzs")
    ph_load = pick("load ph")
    total_load = None

    if build_load is not None and ph_load is not None:
        total_load = df[build_load].astype(float).fillna(0.0)
        if ph_load is not None:
            total_load = total_load + df[ph_load].astype(float).fillna(0.0)
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

    out["Load_build_kW"] = pd.Series(build_load, index=out.index)
    out["Load_ph_kW"]   = pd.Series(ph_load,   index=out.index)
    out["Load_total_kW"] = pd.Series(total_load.values, index=out.index)

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
    """
    Aim
        Build a PyPSA AC network from:
          nodes (poles), edges (LV segments), associations (load mapping),
          dispatch (time series), and configuration parameters.

    How it works
        1) Sets snapshots from the dispatch index.
        2) Standardizes IDs for nodes and edge endpoints.
        3) Creates one Bus per pole (named 'bus_{pole_id}') and assigns v_nom.
        4) Creates Lines from the edges geometry:
           - computes length (m -> km),
           - computes R and X from per-km values,
           - adds one PyPSA Line per segment.
        5) Allocates total served load to poles proportionally to the number of
           buildings mapped to each pole_id.
        6) Adds aggregated sources at the slack bus:
           PV, genset, battery discharge (as PQ generators),
           and battery charging as an additional Load.
        7) Adds one Slack generator (control='Slack') to ensure the power balance
           and provide a reference for the PF.

    Why it is useful
        This is the "model assembly" step: once the network is built correctly,
        PF runs become repeatable and comparable across scenarios and snapshots.
    """
    net = pypsa.Network()
    snapshots = dispatch_kW.index.tolist()
    net.set_snapshots(snapshots)

    # Standardize IDs
    gdf_nodes, pole_col = ensure_pole_id_column(gdf_nodes)
    u_col, v_col = infer_edge_endpoints(gdf_edges)

    # ------------------------------------------------------------------
    # Create buses (ONLY those actually used)
    # Rationale: nodes.geojson may contain poles that are not referenced
    # by any edge or load. Keeping them as buses creates electrical islands
    # of size 1, which breaks PF and/or yields NaN results.
    # ------------------------------------------------------------------
    slack_pole_id = int(cfg["slack_pole_id"])

    # Poles referenced by edges
    u_vals = pd.to_numeric(gdf_edges[u_col], errors="coerce")
    v_vals = pd.to_numeric(gdf_edges[v_col], errors="coerce")
    # set deletes duplicates, dropna deletes NaN, astype(int) converts to int, tolist lists them.
    edge_poles = set(pd.concat([u_vals, v_vals], ignore_index=True).dropna().astype(int).tolist())

    # Poles referenced by loads (associations)
    # some poles with an associated building may be disconnected from the grid, 
    # including them allows the code to track this phenomena after.
    assoc_poles = set(pd.to_numeric(associations["pole_id"], errors="coerce").dropna().astype(int).unique().tolist())

    used_poles = edge_poles | assoc_poles | {slack_pole_id}

    pole_to_bus: dict[int, str] = {}
    for _, r in gdf_nodes.iterrows():
        pole_id = int(r[pole_col])
        if pole_id in used_poles:
            bus_name = f"bus_{pole_id}"
            net.add("Bus", name=bus_name, v_nom=float(cfg["v_nom_kV"]), carrier="AC")
            pole_to_bus[pole_id] = bus_name

    # Ensure v_nom is set for ALL buses (missing v_nom leads to NaN per-unit voltages)
    net.buses["v_nom"] = net.buses["v_nom"].astype(float).fillna(float(cfg["v_nom_kV"]))

    # Validate that all required poles exist in nodes
    missing_used = [pid for pid in used_poles if pid not in pole_to_bus]
    if missing_used:
        # give a short helpful error
        example = sorted(missing_used)[:20]
        raise ValueError(
            f"Some required pole_ids are missing in nodes.geojson (showing up to 20): {example}. "
            "Fix: ensure nodes contains all pole endpoints used by edges and all pole_id referenced by associations."
        )

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
        r_ohm = float(cfg["line_r_ohm_per_km"]) * length_km
        x_ohm = float(cfg["line_x_ohm_per_km"]) * length_km

        net.add(
            "Line",
            name=f"line_{u}_{v}",
            bus0=pole_to_bus[u],
            bus1=pole_to_bus[v],
            r=r_ohm,
            x=x_ohm,
            length=length_km,
            carrier="AC",
        )

    # Connectivity sanity check: any pole that receives load must be connected by at least one edge
    connected_buses = set()
    if len(net.lines) > 0:
        connected_buses |= set(net.lines["bus0"].tolist())
        connected_buses |= set(net.lines["bus1"].tolist())

    load_poles = set(pd.to_numeric(associations["pole_id"], errors="coerce").dropna().astype(int).unique().tolist())
    orphan_load_poles = [pid for pid in load_poles if pole_to_bus.get(pid) not in connected_buses and pid != slack_pole_id]
    if orphan_load_poles:
        example = orphan_load_poles[:20]
        raise ValueError(
            "Some poles receive load (from associations.csv) but are not connected by any edge. "
            "This creates electrical islands (size=1) and PF cannot run. "
            f"Example pole_ids (up to 20): {example}"
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
        share = float(n_bld) / total_buildings
        p_MW = (dispatch_kW["Load_build_kW"] * share) / 1000.0
        q_MVAr = p_MW * tanphi

        net.add(
            "Load",
            name=f"Load_{pole_id}",
            bus=pole_to_bus[pole_id],
            p_set=p_MW,
            q_set=q_MVAr,
        )

    # Slack bus and aggregated sources (PV, genset, battery discharge) + battery charge load
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

def _pre_pf_diagnostics(net, slack_pole_id, snap):
    """
    Aim
        Run lightweight checks that catch the most common PF failure modes
        *before* running `net.pf(...)`.

    How it works
        - Resolves the slack bus label (supports bus_{id} naming).
        - Checks that v_nom exists and is non-missing for all buses.
        - Verifies at least one Slack generator exists and is attached to the slack bus.
        - Ensures the requested snapshot exists (via `_resolve_snapshot`).
        - Runs PyPSA's topology detection and raises if multiple islands exist.

    Why it is useful
        Without these checks you often get opaque errors (or NaN results). This function
        fails fast and points to the exact input inconsistency (IDs, missing v_nom, islands).
    """

    if slack_pole_id is None:
        raise ValueError("Slack pole id is None. Set slack_pole_id in the workbook 'Network' sheet.")

    # --- Resolve slack bus label (supports 'bus_{id}' naming) ---
    buses_index = net.buses.index
    candidates = [slack_pole_id, str(slack_pole_id)]
    try:
        sid_int = int(float(slack_pole_id))
        candidates.append(f"bus_{sid_int}")
    except Exception:
        sid_int = None
    candidates.append(f"bus_{slack_pole_id}")

    slack_bus = next((c for c in candidates if c in buses_index), None)
    if slack_bus is None:
        sample = list(buses_index[:10])
        raise ValueError(
            f"Slack bus not found in network buses. slack_pole_id={slack_pole_id!r}. "
            f"First bus labels: {sample}. "
            "Fix: if buses are named like 'bus_0', keep slack_pole_id=0 and let the tool map it, "
            "or ensure consistent mapping between nodes.geojson IDs and the bus naming."
        )

    # --- v_nom sanity (missing v_nom => NaN v_mag_pu) ---
    if "v_nom" not in net.buses.columns:
        raise ValueError("net.buses has no v_nom column. PF voltage results cannot be expressed in per-unit.")
    n_vnom_nan = int(net.buses["v_nom"].isna().sum())
    if n_vnom_nan > 0:
        examples = net.buses.index[net.buses["v_nom"].isna()].tolist()[:20]
        raise ValueError(
            f"{n_vnom_nan} buses have missing v_nom (nominal voltage). "
            f"Examples (up to 20): {examples}. "
            "Fix: set v_nom for all buses when adding them (net.add('Bus', ..., v_nom=...))."
        )

    # --- Slack generator exists and is at slack bus ---
    if getattr(net, "generators", None) is None or net.generators.empty:
        raise ValueError("No generators exist in the network. Add a generator with control='Slack' at the slack bus.")
    if "control" not in net.generators.columns:
        raise ValueError("net.generators has no 'control' column. Cannot verify slack generator.")

    slack_gens = net.generators.index[net.generators["control"].astype(str).str.lower().eq("slack")]
    if len(slack_gens) == 0:
        raise ValueError("No generator with control='Slack' found. PyPSA needs a slack generator.")
    # At least one slack gen on slack_bus
    slack_on_bus = net.generators.loc[slack_gens, "bus"].astype(str).eq(str(slack_bus)).any()
    if not slack_on_bus:
        ex = net.generators.loc[slack_gens, ["bus"]].head(10).to_dict(orient="records")
        raise ValueError(
            f"Slack generator(s) exist but none are connected to slack bus '{slack_bus}'. "
            f"Examples: {ex}. Fix: set Slack generator bus='{slack_bus}'."
        )

    # --- Snapshot sanity (handle int/str mismatch) ---
    _ = _resolve_snapshot(net, snap)

    # --- Topology / islands check (version-robust) ---
    try:
        net.determine_network_topology()
        if "sub_network" in net.buses.columns:
            sn_sizes = net.buses["sub_network"].value_counts(dropna=False)
            if len(sn_sizes) > 1:
                slack_sn = net.buses.at[slack_bus, "sub_network"]
                raise ValueError(
                    "Network has electrical islands (multiple sub_networks). PyPSA requires a slack generator in EACH island, "
                    "or you must remove islands / orphan buses before PF. "
                    f"Slack bus '{slack_bus}' is in sub_network: {slack_sn}. "
                    f"Sub-network sizes (top 10): {sn_sizes.head(10).to_dict()}"
                )
    except Exception as topo_e:
        raise ValueError(f"Topology check failed: {repr(topo_e)}")

    return slack_bus
def nodal_voltage_table(net: pypsa.Network, snapshot) -> pd.DataFrame:
    """
    Aim
        Produce a tidy table of nodal voltage results for a given snapshot.

    How it works
        - Reads per-unit magnitudes `net.buses_t.v_mag_pu` for the snapshot.
        - Converts to absolute voltage in Volts using v_nom (kV -> V).
        - Computes deviation from nominal (ΔV) and percentage deviation.

    Why it is useful
        It turns PyPSA internal result tables into a report-ready format that can be
        exported and compared across scenarios.
    """
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
    df = df.sort_values('Bus', key=lambda s: s.map(_bus_sort_key)).reset_index(drop=True)
    return df.sort_values("Bus").reset_index(drop=True)


def branch_current_table(net: pypsa.Network, snapshot) -> pd.DataFrame:
    """
    Aim
        Estimate branch currents for each line at a given snapshot.

    How it works
        - Uses line-end active/reactive power (p0,q0,p1,q1) from PyPSA.
        - Computes apparent power |S| at each end.
        - Estimates current via I = |S| / (sqrt(3) * V_LL) using the end-bus voltage.

    Why it is useful
        Distribution planning often needs currents for thermal limits and sizing;
        this provides a consistent approximation even when direct current results
        are not available in the installed PyPSA version.
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
    df = df.sort_values(['Bus0','Bus1'], key=lambda s: s.map(_bus_sort_key)).reset_index(drop=True)
    return df.sort_values("Line").reset_index(drop=True)


# ======================================================================================
# Streamlit App
# ======================================================================================

# --------------------------------------------------------------------------------------
# STREAMLIT UI / MAIN RUN FLOW
#
# The app is organized in three phases:
#   (1) Upload inputs (Excel + geodata + associations)
#   (2) Parse/validate inputs and build the PF-ready data structures
#   (3) Build the PyPSA network and run an AC power flow for ONE selected snapshot
#
# Why a single snapshot?
#   Running an AC PF for all hours can be slow in a Streamlit app. Most debugging and
#   engineering checks (voltage drops, current peaks, bottlenecks) can be performed on
#   representative hours selected from the dispatch (peak load, peak PV, worst-case).
#
# What happens when the user clicks "Run Power Flow"?
#   - We call build_network(...) to assemble buses, lines, loads, and generators.
#   - We call _pre_pf_diagnostics(...) to catch typical input issues early.
#   - We call net.pf(snapshots=[snap]) to solve the AC PF for that hour.
#   - We post-process results into human-friendly tables:
#         nodal_voltage_table(...) and branch_current_table(...)
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="pftoolV7", layout="wide")
st.title("pftoolV7 - Slim Power Flow (Excel + GeoJSON + Associations)")

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
        st.exception(e)
        st.stop()

    st.subheader("2) Select snapshot and run PF")
    snap = st.selectbox("Snapshot (hour index)", dispatch_kW.index.tolist(), index=0)

    if st.button("Run Power Flow", type="primary"):
        try:
            # Build network for ALL snapshots (required by PyPSA), but solve PF only for the selected one.
            net = build_network(gdf_nodes, gdf_edges, associations, dispatch_kW, cfg)
            # Debug: snapshot typing + v_nom completeness
            st.write('Selected snap:', snap, type(snap))
            st.write('Snapshots dtype:', net.snapshots.dtype)
            st.write('snap in net.snapshots?', snap in net.snapshots)
            st.write('str(snap) in net.snapshots?', str(snap) in net.snapshots)
            st.write('v_nom NaN buses:', int(net.buses.v_nom.isna().sum()))
            slack_bus = _pre_pf_diagnostics(net, cfg.get('slack_pole_id'), snap)
            snap_key = _resolve_snapshot(net, snap)
            net.pf(snapshots=[snap_key])
        except Exception as e:
            st.error(f"PF error: {repr(e)}")
            st.exception(e)
            st.stop()

        st.success("PF completed.")

        st.subheader("Nodal voltages")
        vtab = nodal_voltage_table(net, snap_key).round(6)
        st.dataframe(vtab, use_container_width=True)

        st.subheader("Branch currents")
        itab = branch_current_table(net, snap_key).round(6)
        st.dataframe(itab, use_container_width=True)

        st.subheader("Downloads")
        st.download_button("Download nodal voltages CSV", vtab.to_csv(index=False).encode("utf-8"), "nodal_voltages.csv")
        st.download_button("Download branch currents CSV", itab.to_csv(index=False).encode("utf-8"), "branch_currents.csv")
else:
    st.info("Upload the PF Excel workbook, nodes, edges, and associations to enable the PF run.")
