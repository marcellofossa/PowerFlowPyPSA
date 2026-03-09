"""
Power Flow tool (PyPSA + Streamlit)

Inputs (uploads):
- PF Excel workbook (.xlsx) with sheets:
    - Network: global electrical parameters (v_nom, power_factor, line_r, line_x, slack_pole_id, ph_at_slack, crs_epsg, ...)
      Format: columns [Parameter, Value].
    - Dispatch: hourly time series (Preliminary Sizning Tool-style columns supported).
- Nodes GeoJSON/GPKG: poles (points). Must contain an id column (preferred: "id" or "pole_id").
- Edges GeoJSON/GPKG: LV network segments (lines) with endpoints (preferred: "source"/"target").
- Associations CSV: building-to-pole mapping (building_id, pole_id).
- OnSSET friendly: distribution grid GPKG (assumptions reguarding the load distribution will be done)

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
from __future__ import annotations

import io
import os
import tempfile
import math
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import streamlit as st
import pypsa
# ======================================================================================
# Helpers: robust bus sorting + snapshot resolution
# ======================================================================================

_bus_num_re = re.compile(r"(\d+)$")

def _bus_sort_key(label) -> int:
    """Sort key for bus labels like 'bus_12' (numeric)."""
    s = str(label)
    m = _bus_num_re.search(s)
    return int(m.group(1)) if m else 10**18

def _sorted_bus_index(bus_index):
    return sorted(list(bus_index), key=_bus_sort_key)

def _resolve_snapshot(net: pypsa.Network, snap):
    """Return a snapshot label that exists in net.snapshots (handles int/str mismatch)."""
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
    """Read an uploaded vector file (.geojson/.json/.gpkg) into a GeoDataFrame."""
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
    Load associations.csv and normalize columns to: pole_id, building_id.
    Accepted variants: pole_id/pole and building_id/building.
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
    edge_poles = set(pd.concat([u_vals, v_vals], ignore_index=True).dropna().astype(int).tolist())

    # Poles referenced by loads (associations)
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
        if bool(cfg.get("drop_orphan_load_poles", False)):
            # Drop associations for orphan poles (keeps network solvable). This is useful for OnSSET,
            # where some exported poles may not be referenced by any line segment after snapping.
            st.warning(
                "Some poles receive load but are not connected by any edge (electrical islands). "
                f"Dropping their loads to allow PF. Example pole_ids (up to 20): {example}"
            )
            associations = associations.loc[~associations["pole_id"].isin(orphan_load_poles)].copy()
        else:
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
    """Run sanity checks before calling net.pf(). Raises ValueError with actionable messages."""
    import pandas as pd
    import numpy as np

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

    # --- Topology / islands handling (version-robust) ---
    # PyPSA requires a slack generator in EACH island. For OnSSET (and often for real GIS-derived graphs),
    # small islands/orphan buses can appear due to snapping / reduced export granularity.
    # Policy: keep ONLY the connected component (sub_network) that contains the chosen slack bus,
    # and drop all other islands before running PF. This is transparent and avoids silent wrong results.
    try:
        net.determine_network_topology()
        if "sub_network" in net.buses.columns:
            sn_sizes = net.buses["sub_network"].value_counts(dropna=False)
            if len(sn_sizes) > 1:
                slack_sn = net.buses.at[slack_bus, "sub_network"]
                # Inform user
                msg = (
                    "Network has electrical islands (multiple sub_networks). "
                    f"Keeping only the island containing slack bus '{slack_bus}' (sub_network={slack_sn}) "
                    "and dropping the others. "
                    f"Sub-network sizes (top 10): {sn_sizes.head(10).to_dict()}"
                )
                try:
                    import streamlit as st
                    st.warning(msg)
                except Exception:
                    print("WARNING:", msg)

                keep_buses = net.buses.index[net.buses["sub_network"].astype(str) == str(slack_sn)].tolist()
                drop_buses = [b for b in net.buses.index.tolist() if b not in set(keep_buses)]

                # Drop components attached to buses that will be removed
                def _remove_many(component_name, names):
                    names = list(names or [])
                    if not names:
                        return
                    # Older PyPSA versions do not have mremove; fall back to remove in a loop.
                    if hasattr(net, "mremove"):
                        net.mremove(component_name, names)
                    else:
                        for name in names:
                            net.remove(component_name, name)

                def _drop_by_bus(component_name, df, bus_cols=("bus", "bus0", "bus1")):
                    if df is None or df.empty:
                        return
                    cols = [c for c in bus_cols if c in df.columns]
                    if not cols:
                        return
                    keep_bus_set = set(map(str, keep_buses))
                    mask = pd.Series(False, index=df.index)
                    for c in cols:
                        mask = mask | (~df[c].astype(str).isin(keep_bus_set))
                    to_drop = df.index[mask].tolist()
                    if to_drop:
                        _remove_many(component_name, to_drop)

                _drop_by_bus("Load", net.loads, ("bus",))
                _drop_by_bus("Generator", net.generators, ("bus",))
                _drop_by_bus("Line", net.lines, ("bus0", "bus1"))
                _drop_by_bus("Transformer", getattr(net, "transformers", None), ("bus0", "bus1"))
                _drop_by_bus("ShuntImpedance", getattr(net, "shunt_impedances", None), ("bus",))
                _drop_by_bus("Store", getattr(net, "stores", None), ("bus",))
                _drop_by_bus("StorageUnit", getattr(net, "storage_units", None), ("bus",))

                # Finally drop buses
                if drop_buses:
                    _remove_many("Bus", drop_buses)

                # Recompute topology after pruning
                net.determine_network_topology()

    except Exception as topo_e:
        raise ValueError(f"Topology handling failed: {repr(topo_e)}")

    return slack_bus
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
        "V bus [V]": v_abs_V.values,
        "deltaV bus [V]": delta_V.values,
        "deltaV% bus [%]": delta_pct.values,
    })

    # Natural sort by bus number: bus_1, bus_2, ..., bus_10
    df["_busn"] = df["Bus"].map(_bus_sort_key)
    df = df.sort_values(["_busn", "Bus"]).drop(columns=["_busn"]).reset_index(drop=True)
    return df


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

    lengths_m = net.lines.length.astype(float) * 1000.0  # km -> m

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
            "Branch length [m]": float(lengths_m.loc[ln]),
            "I [A]": np.nanmax([i0, i1]),
            "S [kVA]": 1000.0 * np.nanmax([s0_MVA, s1_MVA]),
        })

    df = pd.DataFrame(rows)

    # Natural sort: first by Bus0 number, then Bus1 number, then line endpoints (e.g. line_0_17, line_0_18, ...)
    df["_bus0n"] = df["Bus0"].map(_bus_sort_key)
    df["_bus1n"] = df["Bus1"].map(_bus_sort_key)

    def _line_u_v_nums(name: str):
        nums = re.findall(r"\d+", str(name))
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        if len(nums) == 1:
            return int(nums[0]), 10**18
        return 10**18, 10**18

    uv = df["Line"].map(_line_u_v_nums)
    df["_lu"] = uv.map(lambda t: t[0])
    df["_lv"] = uv.map(lambda t: t[1])

    df = (
        df.sort_values(["_bus0n", "_bus1n", "_lu", "_lv", "Line"])
          .drop(columns=["_bus0n", "_bus1n", "_lu", "_lv"])
          .reset_index(drop=True)
    )
    return df




# ======================================================================================
# OnSSET GPKG adapter (Trunk/Laterals/Service_lines/Poles -> nodes/edges/associations)
# ======================================================================================

def _write_tmp_uploaded(uploaded_file, suffix: str) -> str:
    """Write Streamlit uploaded file bytes to a temp path and return it."""
    data = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def read_gpkg_layer(uploaded_file, layer: str) -> gpd.GeoDataFrame:
    """Read a specific layer from an uploaded GeoPackage."""
    tmp_path = _write_tmp_uploaded(uploaded_file, suffix=".gpkg")
    try:
        return gpd.read_file(tmp_path, layer=layer)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _explode_to_primitives(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode multi-geometries and keep only Points/LineStrings."""
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.is_valid].copy()
    # explode multiparts
    try:
        gdf = gdf.explode(index_parts=False, ignore_index=True)
    except TypeError:
        gdf = gdf.explode(index_parts=False)
        gdf = gdf.reset_index(drop=True)
    # keep supported
    gdf = gdf[gdf.geometry.type.isin(["Point", "LineString"])].copy()
    return gdf


def onsett_gpkg_to_graph(
    uploaded_gpkg,
    *,
    target_epsg: int,
    snap_tolerance_m: float = 50.0,
    drop_duplicate_edges: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Convert OnSSET mini-grid gpkg to:
      - nodes GeoDataFrame with 'pole_id' (0..N-1)
      - edges GeoDataFrame with 'source','target','length_m','net_type'

    Approximations (explicit, as per your methodology):
      - No explicit service poles beyond those in gpkg Poles layer.
      - Edge endpoints are inferred by snapping each segment endpoint to the nearest pole.
    """
    # --- Read layers ---
    poles = read_gpkg_layer(uploaded_gpkg, "Poles")
    trunk = read_gpkg_layer(uploaded_gpkg, "Trunk_line")
    lat = read_gpkg_layer(uploaded_gpkg, "Laterals")
    srv = read_gpkg_layer(uploaded_gpkg, "Service_lines")

    # --- Project to metric CRS ---
    for g in (poles, trunk, lat, srv):
        if g.crs is None:
            raise ValueError("OnSSET gpkg layer has no CRS. Re-export it with a CRS.")
    if poles.crs.to_epsg() != int(target_epsg):
        poles = poles.to_crs(epsg=int(target_epsg))
    if trunk.crs.to_epsg() != int(target_epsg):
        trunk = trunk.to_crs(epsg=int(target_epsg))
    if lat.crs.to_epsg() != int(target_epsg):
        lat = lat.to_crs(epsg=int(target_epsg))
    if srv.crs.to_epsg() != int(target_epsg):
        srv = srv.to_crs(epsg=int(target_epsg))

    # --- Explode to primitives ---
    poles = _explode_to_primitives(poles)
    trunk = _explode_to_primitives(trunk)
    lat = _explode_to_primitives(lat)
    srv = _explode_to_primitives(srv)

    # --- Nodes ---
    nodes = poles.copy()
    nodes = nodes.reset_index(drop=True)
    if len(nodes) == 0:
        raise ValueError("OnSSET gpkg: Poles layer has 0 points after explode.")
    nodes["pole_id"] = np.arange(len(nodes), dtype=int)

    # --- Edges: concat all types ---
    edges = pd.concat(
        [
            trunk.assign(net_type="trunk"),
            lat.assign(net_type="lateral"),
            srv.assign(net_type="service"),
        ],
        ignore_index=True,
    )
    edges = edges[edges.geometry.type.eq("LineString")].copy()
    if len(edges) == 0:
        raise ValueError("OnSSET gpkg: no LineString segments found in Trunk/Laterals/Service_lines.")

    # --- Build endpoints GeoDataFrame (two points per segment) ---
    starts = []
    ends = []
    for i, geom in enumerate(edges.geometry.tolist()):
        coords = list(geom.coords)
        starts.append(Point(coords[0]))
        ends.append(Point(coords[-1]))

    endpts = gpd.GeoDataFrame(
        {
            "seg_i": np.repeat(np.arange(len(edges), dtype=int), 2),
            "side": ["start", "end"] * len(edges),
        },
        geometry=starts + ends,
        crs=edges.crs,
    )

    # --- Snap endpoints to nearest pole ---
    # Use spatial join nearest; fallback to brute force if unavailable
    try:
        joined = gpd.sjoin_nearest(
            endpts,
            nodes[["pole_id", "geometry"]],
            how="left",
            distance_col="snap_dist_m",
            max_distance=float(snap_tolerance_m),
        )
    except Exception:
        # brute force (slower, but robust)
        node_xy = np.vstack([nodes.geometry.x.values, nodes.geometry.y.values]).T
        out_pole = []
        out_dist = []
        for pt in endpts.geometry:
            dx = node_xy[:, 0] - pt.x
            dy = node_xy[:, 1] - pt.y
            d = np.hypot(dx, dy)
            j = int(np.argmin(d))
            out_pole.append(int(nodes.iloc[j]["pole_id"]))
            out_dist.append(float(d[j]))
        joined = endpts.copy()
        joined["pole_id"] = out_pole
        joined["snap_dist_m"] = out_dist
        joined.loc[joined["snap_dist_m"] > float(snap_tolerance_m), "pole_id"] = np.nan

    n_missing = int(joined["pole_id"].isna().sum())
    if n_missing > 0:
        # Robust fallback for OnSSET exports:
        # if some segment endpoints cannot be snapped within the tolerance, we create
        # *virtual poles* exactly at those endpoints, so the reconstructed graph is
        # still electrically consistent for PF comparisons.
        missing = joined.loc[joined["pole_id"].isna()].copy()

        # Identify truly invalid endpoint geometries (empty or NaN coords). These indicate
        # corrupted geometries/CRS and should be fixed at source.
        invalid = missing.geometry.isna() | missing.geometry.is_empty
        invalid = invalid | (~np.isfinite(missing.geometry.x.values)) | (~np.isfinite(missing.geometry.y.values))
        if invalid.any():
            ex = missing.loc[invalid, ["seg_i", "side"]].head(10).to_dict(orient="records")
            raise ValueError(
                f"OnSSET gpkg: {int(invalid.sum())} endpoint geometries are invalid (empty/NaN). "
                f"Examples (up to 10): {ex}. Fix: re-export gpkg with valid geometries/CRS."
            )

        next_id = int(nodes["pole_id"].max()) + 1
        created = 0
        # Create one virtual pole per missing endpoint (simple + deterministic)
        for ridx, row in missing.iterrows():
            pt = row.geometry
            nodes = pd.concat(
                [
                    nodes,
                    gpd.GeoDataFrame(
                        [{"pole_id": next_id, "geometry": pt, "pole_origin": "onsett_virtual"}],
                        crs=nodes.crs,
                    ),
                ],
                ignore_index=True,
            )
            joined.loc[ridx, "pole_id"] = next_id
            joined.loc[ridx, "snap_dist_m"] = 0.0
            next_id += 1
            created += 1

        if "st" in globals():
            st.warning(
                f"OnSSET: {n_missing} segment endpoints could not be snapped within {snap_tolerance_m} m. "
                f"Created {created} virtual poles at those endpoints (assumption for methodology)."
            )

    # --- Assemble source/target ---
    joined["pole_id"] = joined["pole_id"].astype(int)
    piv = joined.pivot_table(index="seg_i", columns="side", values="pole_id", aggfunc="first")
    edges = edges.reset_index(drop=True)
    edges["source"] = piv["start"].reindex(range(len(edges))).astype(int).values
    edges["target"] = piv["end"].reindex(range(len(edges))).astype(int).values

    edges["length_m"] = edges.geometry.length.astype(float)
    edges = edges[edges["length_m"] > 1e-6].copy()

    # Optionally drop duplicate undirected edges (common when snapping)
    if drop_duplicate_edges:
        a = np.minimum(edges["source"].values, edges["target"].values)
        b = np.maximum(edges["source"].values, edges["target"].values)
        edges["_ab"] = list(zip(a.tolist(), b.tolist(), edges["net_type"].astype(str).tolist()))
        edges = edges.drop_duplicates(subset=["_ab"]).drop(columns=["_ab"]).reset_index(drop=True)

    # Keep only columns expected downstream
    edges_out = edges[["source", "target", "net_type", "length_m", "geometry"]].copy()
    nodes_out = nodes[["pole_id", "geometry"]].copy()
    return nodes_out, edges_out


def make_uniform_associations(pole_ids: list[int]) -> pd.DataFrame:
    """
    Create a synthetic associations table with the SAME number of buildings per pole (1 each).
    This implements your stated approximation when building-to-pole mapping is unavailable.
    """
    pole_ids = [int(p) for p in pole_ids]
    return pd.DataFrame({"building_id": np.arange(len(pole_ids), dtype=int), "pole_id": pole_ids})


def connected_poles_from_edges(gdf_edges: gpd.GeoDataFrame) -> list[int]:
    """Return sorted unique pole_ids that appear in edges endpoints (source/target)."""
    if gdf_edges is None or len(gdf_edges) == 0:
        return []
    pole_ids = pd.concat(
        [
            pd.to_numeric(gdf_edges.get("source"), errors="coerce"),
            pd.to_numeric(gdf_edges.get("target"), errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    try:
        pole_ids = pole_ids.astype(int)
    except Exception:
        pole_ids = pole_ids.astype(float).astype(int)
    return sorted(pole_ids.unique().tolist())




# ======================================================================================
# Streamlit App
# ======================================================================================

st.set_page_config(page_title="pftoolV8", layout="wide")
st.title("pftoolV8 - Slim Power Flow (Excel + GeoJSON + Associations)")

st.subheader("1) Upload inputs")

input_mode = st.radio(
    "Input mode",
    ["Custom tool (nodes+edges+associations)", "OnSSET mini-grid (single .gpkg)"],
    index=0,
    help=(
        "Use 'OnSSET' if you have a GeoPackage with layers: Poles, Trunk_line, Laterals, Service_lines. "
        "Endpoints are inferred by snapping segment endpoints to nearest poles."
    ),
)

pf_excel = st.file_uploader("PF Excel workbook (.xlsx) with sheets: Network, Dispatch", type=["xlsx", "xls"])

reduce_load_by_unmet = st.checkbox("Use served load = Load - Unmet Demand (if present)", value=True)

if input_mode.startswith("Custom"):
    nodes_file = st.file_uploader("Nodes (mst_nodes.geojson / .gpkg)", type=["geojson", "json", "gpkg"])
    edges_file = st.file_uploader("Edges (mst_edges.geojson / .gpkg)", type=["geojson", "json", "gpkg"])
    assoc_file = st.file_uploader("Associations (associations.csv)", type=["csv"])
    onsett_gpkg = None
else:
    onsett_gpkg = st.file_uploader("OnSSET distribution_grid.gpkg", type=["gpkg"])
    snap_tol = st.number_input("OnSSET snapping tolerance [m]", min_value=1.0, max_value=500.0, value=50.0, step=1.0)
    assoc_file = st.file_uploader(
        "Optional associations.csv (if you DO have building-to-pole mapping). If omitted, load is spread uniformly across poles.",
        type=["csv"],
    )
    nodes_file = None
    edges_file = None

def _inputs_ready():
    if pf_excel is None:
        return False
    if input_mode.startswith("Custom"):
        return nodes_file is not None and edges_file is not None and assoc_file is not None
    return onsett_gpkg is not None

if _inputs_ready():
    try:
        cfg, disp_raw = read_pf_workbook(pf_excel)
        dispatch_kW = normalize_dispatch(disp_raw, reduce_load_by_unmet=reduce_load_by_unmet)

        if input_mode.startswith("Custom"):
            gdf_nodes = load_and_project_geodata(nodes_file, cfg["crs_epsg"])
            gdf_edges = load_and_project_geodata(edges_file, cfg["crs_epsg"])
            associations = load_associations_csv(assoc_file)
        else:
            gdf_nodes, gdf_edges = onsett_gpkg_to_graph(
                onsett_gpkg,
                target_epsg=int(cfg["crs_epsg"]),
                snap_tolerance_m=float(snap_tol),
            )
            if assoc_file is not None:
                associations = load_associations_csv(assoc_file)
            else:
                # Uniform allocation across ALL poles present in OnSSET file
                # Uniform allocation across CONNECTED poles only (avoid 1-node electrical islands)
                connected = connected_poles_from_edges(gdf_edges)
                slack_pid = int(cfg.get("slack_pole_id", 0))
                if slack_pid not in connected:
                    connected = [slack_pid] + connected
                n_iso = int(len(gdf_nodes) - len(set(connected)))
                if n_iso > 0:
                    st.warning(
                        f"OnSSET: {n_iso} pole(s) are not referenced by any edge (isolated). "
                        "They will be excluded from load allocation to avoid 1-node islands."
                    )
                associations = make_uniform_associations(connected)

            cfg["drop_orphan_load_poles"] = True
            st.info(
                "OnSSET mode assumptions applied: "
                "segment endpoints snapped to nearest poles; if no associations.csv provided, "
                "loads are distributed uniformly across poles."
            )
            st.info(
                "Important: since OnSSET poles have no original IDs, pole_id are assigned as 0..N-1. "
                "Set slack_pole_id in the Excel 'Network' sheet accordingly (commonly 0)."
            )

        st.success(f"Inputs loaded. Snapshots: {len(dispatch_kW)}. slack_pole_id={cfg['slack_pole_id']}")
        st.write("Network config (from Excel):", cfg)
        st.write(f"Nodes: {len(gdf_nodes)} | Edges: {len(gdf_edges)} | Assoc rows: {len(associations)}")
    except Exception as e:
        st.error(f"Input error: {repr(e)}")
        st.exception(e)
        st.stop()

    st.subheader("2) Select snapshot and run PF")
    snap = st.selectbox("Snapshot (hour index)", dispatch_kW.index.tolist(), index=0)

    if st.button("Run Power Flow", type="primary"):
        try:
            net = build_network(gdf_nodes, gdf_edges, associations, dispatch_kW, cfg)
            slack_bus = _pre_pf_diagnostics(net, cfg.get("slack_pole_id"), snap)
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
    st.info("Upload the PF Excel workbook and the required network files to enable the PF run.")

