from __future__ import annotations
import io
import os
import tempfile
from typing import Optional, Tuple, Iterable, Dict

import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium import Map, PolyLine, CircleMarker
from branca.element import MacroElement, Template


def read_vector(uploaded_file) -> gpd.GeoDataFrame:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    data = uploaded_file.getvalue()

    if suffix in [".geojson", ".json"]:
        return gpd.read_file(io.BytesIO(data))

    if suffix == ".gpkg":
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

    raise ValueError("Unsupported file format.")


def find_column(df, *candidates: str) -> Optional[str]:
    cols_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.strip().lower()
        if k in cols_map:
            return cols_map[k]
    return None


def read_building_metadata_csv(uploaded_file) -> pd.DataFrame:
    """
    Expected columns: building_id, category, weight (optional).
    Accepts both Excel (.xlsx) and CSV (any separator, any decimal notation).
    """
    fname = getattr(uploaded_file, "name", "") or ""
    if fname.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, dtype=str)
    else:
        # Try comma first, then semicolon
        raw = uploaded_file.read() if hasattr(uploaded_file, "read") else open(uploaded_file, "rb").read()
        for sep in [",", ";"]:
            try:
                candidate = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str)
                if len(candidate.columns) > 1:
                    df = candidate
                    break
            except Exception:
                continue
        else:
            df = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str)

    bcol = find_column(df, "building_id", "building")
    ccol = find_column(df, "category", "user_category", "type")
    wcol = find_column(df, "weight", "multiplier", "scale")

    if bcol is None or ccol is None:
        raise ValueError("Building metadata must include columns: building_id and category (weight optional).")

    cols = {bcol: "building_id", ccol: "category"}
    if wcol is not None:
        cols[wcol] = "weight"
    out = df[[bcol, ccol] + ([wcol] if wcol else [])].rename(columns=cols)
    if wcol is None:
        out["weight"] = 1.0

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(1.0)
    out["building_id"] = out["building_id"].astype(str)
    out["category"] = out["category"].astype(str)

    return out


def read_category_profiles_csv(uploaded_file) -> pd.DataFrame:
    """
    Expected format (wide): hour, CAT_A, CAT_B, ... (kW per building of that category).
    Accepts both Excel (.xlsx) and CSV (any separator, European or standard decimals).
    Returns a DataFrame indexed by hour (int), columns = categories (str), values = kW (float).
    """
    fname = getattr(uploaded_file, "name", "") or ""
    if fname.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raw = uploaded_file.read() if hasattr(uploaded_file, "read") else open(uploaded_file, "rb").read()
        # Auto-detect separator
        for sep in [",", ";"]:
            try:
                candidate = pd.read_csv(io.BytesIO(raw), sep=sep, dtype=str)
                if len(candidate.columns) > 1:
                    df = candidate
                    break
            except Exception:
                continue
        else:
            df = pd.read_csv(io.BytesIO(raw), sep=";", dtype=str)

        # Fix European decimal commas in all columns
        for col in df.columns:
            converted = df[col].astype(str).str.replace(",", ".", regex=False)
            numeric = pd.to_numeric(converted, errors="coerce")
            if numeric.notna().sum() > 0:
                df[col] = numeric

    hcol = find_column(df, "hour", "t", "time", "snapshot")
    if hcol is None:
        raise ValueError("Category profiles must include an 'hour' column.")

    hours = pd.to_numeric(df[hcol], errors="coerce")
    bad = hours.isna()
    if bad.any():
        raise ValueError(f"Category profiles has non-numeric hour entries: {df.loc[bad].head(10)}")

    df = df.copy()
    df[hcol] = hours.astype(int)

    cats = [c for c in df.columns if c != hcol]
    if len(cats) == 0:
        raise ValueError("Category profiles must have at least one category column besides 'hour'.")

    out = df.set_index(hcol)[cats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out.index.name = "hour"

    if out.index.duplicated().any():
        raise ValueError("Category profiles contains duplicated hour values.")

    return out


def make_map_lv_with_load_bubbles(
    *,
    center: Tuple[float, float],
    gdf_poles_4326: gpd.GeoDataFrame,
    pole_id_col: str,
    pole_load_kW_at_hour,  # dict[int,float] preferred (or Series)
    mst_edges_latlon: Optional[Iterable[tuple[tuple[float, float], tuple[float, float]]]] = None,
    gdf_edges_4326: Optional[gpd.GeoDataFrame] = None,
    gdf_roads_4326: Optional[gpd.GeoDataFrame] = None,
    zoom_start: int = 15,
    max_bubble_radius: float = 18.0,
    min_bubble_radius: float = 3.0,
    pmax_ref_kW: Optional[float] = None,
    show_legend: bool = True,
    slack_pole_id: Optional[int] = None,
    highlight_pole_id: Optional[int] = None,          # NEW
    zoom_to_highlight: bool = False,                  # NEW
) -> Map:
    """
    Folium map with:
      - OSM basemap
      - optional roads layer (gray)
      - LV network edges (blue)
      - poles as black dots (ALWAYS with pole_id tooltip)
      - load bubbles (orange) at poles (tooltip includes pole_id + load)

    Scaling:
      - If pmax_ref_kW is None -> RELATIVE scaling (per hour): pmax = max(p_kW at this hour)
      - If pmax_ref_kW is provided -> ABSOLUTE scaling (fixed): pmax = pmax_ref_kW
    """

    # -------------------------------------------------------
    # 1) Normalize loads to dict[int, float]
    # -------------------------------------------------------
    load_dict: Dict[int, float] = {}
    if pole_load_kW_at_hour is not None:
        if isinstance(pole_load_kW_at_hour, pd.Series):
            load_dict = {int(k): float(v) for k, v in pole_load_kW_at_hour.to_dict().items() if v is not None}
        else:
            load_dict = {int(k): float(v) for k, v in dict(pole_load_kW_at_hour).items() if v is not None}

    # -------------------------------------------------------
    # 2) Base map
    # -------------------------------------------------------
    m = folium.Map(
        location=[float(center[0]), float(center[1])],
        zoom_start=int(zoom_start),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -------------------------------------------------------
    # 3) Roads layer
    # -------------------------------------------------------
    if gdf_roads_4326 is not None and not gdf_roads_4326.empty:
        for _, row in gdf_roads_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
                PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)

    # -------------------------------------------------------
    # 4) LV edges
    # -------------------------------------------------------
    edge_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    if mst_edges_latlon is not None:
        edge_pairs = list(mst_edges_latlon)
    elif gdf_edges_4326 is not None and not gdf_edges_4326.empty:
        for _, row in gdf_edges_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
                for i in range(len(coords) - 1):
                    edge_pairs.append((coords[i], coords[i + 1]))
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    for i in range(len(coords) - 1):
                        edge_pairs.append((coords[i], coords[i + 1]))

    for (lat1, lon1), (lat2, lon2) in edge_pairs:
        PolyLine(
            locations=[(lat1, lon1), (lat2, lon2)],
            color="blue",
            weight=2,
            opacity=0.9,
        ).add_to(m)

    # -------------------------------------------------------
    # 5) Prepare poles (pole_id + coords + load)
    # -------------------------------------------------------
    if pole_id_col not in gdf_poles_4326.columns:
        raise ValueError(f"Pole id column '{pole_id_col}' not found in poles GeoDataFrame.")

    gdfp = gdf_poles_4326.copy()
    gdfp["pole_id"] = pd.to_numeric(gdfp[pole_id_col], errors="coerce")
    gdfp = gdfp.dropna(subset=["pole_id"]).copy()
    gdfp["pole_id"] = gdfp["pole_id"].astype(int)

    # Load (0 if missing)
    gdfp["p_kW"] = gdfp["pole_id"].map(load_dict).fillna(0.0).astype(float)

    def _safe_latlon(geom):
        if geom is None or geom.is_empty:
            return None
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        return (float(pt.y), float(pt.x))

    latlons = gdfp.geometry.apply(_safe_latlon)
    gdfp["lat"] = [x[0] if x else np.nan for x in latlons]
    gdfp["lon"] = [x[1] if x else np.nan for x in latlons]
    gdfp = gdfp.dropna(subset=["lat", "lon"]).copy()

    # Optional: re-center map on highlighted pole
    if zoom_to_highlight and highlight_pole_id is not None:
        sel = gdfp.loc[gdfp["pole_id"] == int(highlight_pole_id)]
        if not sel.empty:
            center = (float(sel.iloc[0]["lat"]), float(sel.iloc[0]["lon"]))
            # Re-create map centered here
            m.location = [center[0], center[1]]

    # -------------------------------------------------------
    # 6) Scaling reference
    # -------------------------------------------------------
    p_values = gdfp["p_kW"].to_numpy(dtype=float)
    pmax_hour = float(np.nanmax(p_values)) if len(p_values) else 0.0
    pmax = float(pmax_ref_kW) if pmax_ref_kW is not None else float(pmax_hour)

    # -------------------------------------------------------
    # 7) Draw poles (ALWAYS with pole_id tooltip)
    #    Also highlight slack pole and an optional "highlight pole".
    # -------------------------------------------------------
    for _, r in gdfp.iterrows():
        pid = int(r["pole_id"])
        is_slack = (slack_pole_id is not None and pid == int(slack_pole_id))
        is_hl = (highlight_pole_id is not None and pid == int(highlight_pole_id))

        # base style
        base_radius = 2.5
        base_color = "black"

        # slack style
        if is_slack:
            base_radius = 4.5
            base_color = "purple"

        # draw base pole marker
        CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=base_radius,
            color=base_color,
            fill=True,
            fill_color=base_color,
            fill_opacity=1.0,
            tooltip=f"Pole {pid}" + (" (SLACK / PLANT)" if is_slack else ""),
        ).add_to(m)

        # highlight overlay (high-contrast ring) if requested
        if is_hl:
            # outer ring (cyan)
            CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=11.0,
                color="#00FFFF",      # cyan
                weight=5,
                fill=False,
                opacity=1.0,
                tooltip=f"Pole {pid} (HIGHLIGHTED)",
            ).add_to(m)

            # inner dot (white) to avoid blending with orange bubble
            CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=3.2,
                color="white",
                weight=2,
                fill=True,
                fill_color="white",
                fill_opacity=1.0,
                opacity=1.0,
            ).add_to(m)

    # -------------------------------------------------------
    # 8) Load bubbles (tooltip includes pole_id + load)
    # -------------------------------------------------------
    if pmax > 0:
        for _, r in gdfp.iterrows():
            pkW = float(r["p_kW"])
            if pkW <= 0:
                continue

            radius = min_bubble_radius + (pkW / pmax) * (max_bubble_radius - min_bubble_radius)
            radius = float(np.clip(radius, min_bubble_radius, max_bubble_radius))

            CircleMarker(
                location=[float(r["lat"]), float(r["lon"])],
                radius=radius,
                color="#ff7f0e",
                fill=True,
                fill_color="#ff7f0e",
                fill_opacity=0.35,
                opacity=0.8,
                tooltip=f"Pole {int(r['pole_id'])} — {pkW:.2f} kW",
            ).add_to(m)

    # -------------------------------------------------------
    # 9) Small fixed legend (pure HTML)
    # -------------------------------------------------------
    if show_legend:
        scale_txt = "ABS (year-fixed)" if pmax_ref_kW is not None else "REL (per-hour)"
        pmax_txt = f"{pmax:.2f} kW" if pmax > 0 else "0 kW"

        legend_js = f"""
        <script>
        (function addLegend() {{
            var map = null;
            // wait for Leaflet map to be ready
            var interval = setInterval(function() {{
                var maps = Object.values(window).filter(function(v) {{
                    return v && typeof v === 'object' && v._leaflet_id !== undefined;
                }});
                if (maps.length === 0) return;
                map = maps[0];
                clearInterval(interval);

                var legend = L.control({{position: 'bottomleft'}});
                legend.onAdd = function() {{
                    var div = L.DomUtil.create('div', '');
                    div.style.cssText = 'background:white;border:1px solid #999;border-radius:6px;padding:10px 12px;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,.2);line-height:1.6;';
                    div.innerHTML = '<b>Map legend</b><br>'
                      + '<span style="display:inline-block;width:10px;height:10px;background:black;border-radius:50%;margin-right:6px;"></span>Pole<br>'
                      + '<span style="display:inline-block;width:10px;height:10px;background:purple;border-radius:50%;margin-right:6px;"></span>Slack / plant pole<br>'
                      + '<span style="display:inline-block;width:10px;height:10px;background:#ff7f0e;border-radius:50%;margin-right:6px;opacity:.7;"></span>Load bubble<br>'
                      + '<span style="font-size:11px;">Scaling: <b>{scale_txt}</b><br>Reference max: <b>{pmax_txt}</b></span>';
                    return div;
                }};
                legend.addTo(map);
            }}, 200);
        }})();
        </script>
        """
        m.get_root().html.add_child(folium.Element(legend_js))

    return m

def _infer_edge_uv_cols(gdf_edges: gpd.GeoDataFrame) -> tuple[str, str]:
    """Try common edge endpoint column names."""
    cols = {c.lower(): c for c in gdf_edges.columns}
    u = cols.get("u") or cols.get("from") or cols.get("bus0") or cols.get("source")
    v = cols.get("v") or cols.get("to")   or cols.get("bus1") or cols.get("target")
    if u is None or v is None:
        raise ValueError(
            "Cannot infer edge endpoint columns. Expected one of: "
            "u/v, from/to, bus0/bus1, source/target."
        )
    return u, v


def make_map_lv_voltage_nodes(
    *,
    center,
    gdf_poles_4326,
    pole_id_col: str,
    gdf_edges_4326=None,
    mst_edges_latlon=None,
    gdf_roads_4326=None,
    zoom_start: int = 15,
    slack_pole_id=None,
    bus_v_pu=None,
    v_min_pu: float = 0.90,
    v_max_pu: float = 1.10,
) -> Map:
    """
    Map 1: nodes colored by voltage drop %, branches as thin black lines.
    drop < 10% -> green (#2ECC71)
    drop 10-20%-> orange
    drop > 20% -> red
    slack      -> purple
    """
    bus_v_pu = bus_v_pu or {}
    m = folium.Map(location=[float(center[0]), float(center[1])], zoom_start=int(zoom_start),
                   tiles="OpenStreetMap", control_scale=True)

    # Roads
    if gdf_roads_4326 is not None and not gdf_roads_4326.empty:
        for _, row in gdf_roads_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            if geom.geom_type == "LineString":
                PolyLine([(lat, lon) for lon, lat in geom.coords], color="#CCCCCC", weight=1.5, opacity=0.5).add_to(m)

    # Branches: thin black
    if gdf_edges_4326 is not None and not gdf_edges_4326.empty:
        for _, row in gdf_edges_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            def _draw_black(ls):
                PolyLine([(lat, lon) for lon, lat in ls.coords], color="#222222", weight=1.5, opacity=0.85).add_to(m)
            if geom.geom_type == "LineString": _draw_black(geom)
            elif geom.geom_type == "MultiLineString":
                for ls in geom.geoms: _draw_black(ls)
    elif mst_edges_latlon is not None:
        for (lat1, lon1), (lat2, lon2) in list(mst_edges_latlon):
            PolyLine([(lat1, lon1), (lat2, lon2)], color="#222222", weight=1.5, opacity=0.85).add_to(m)

    # Nodes: colored by voltage drop
    if pole_id_col not in gdf_poles_4326.columns: return m
    gdfp = gdf_poles_4326.copy()
    gdfp["_pid"] = pd.to_numeric(gdfp[pole_id_col], errors="coerce")
    gdfp = gdfp.dropna(subset=["_pid"]).copy()
    gdfp["_pid"] = gdfp["_pid"].astype(int)
    pts = gdfp.geometry.apply(lambda g: g if g.geom_type == "Point" else g.representative_point())
    gdfp["_lat"] = pts.y.astype(float)
    gdfp["_lon"] = pts.x.astype(float)
    gdfp["_vpu"] = gdfp["_pid"].map(bus_v_pu)

    for _, r in gdfp.iterrows():
        pid = int(r["_pid"])
        vpu = r["_vpu"]
        is_slack = slack_pole_id is not None and pid == int(slack_pole_id)
        v_drop_pct = max(0.0, (1.0 - float(vpu)) * 100.0) if (vpu is not None and np.isfinite(float(vpu))) else 0.0
        tip = f"Pole {pid}"
        if vpu is not None and np.isfinite(float(vpu)):
            tip += f" - V={float(vpu):.4f} p.u. (drop: {v_drop_pct:.2f}%)"
        if is_slack: color, fill_color = "purple", "purple"
        elif v_drop_pct > 20.0: color, fill_color = "#CC0000", "#CC0000"
        elif v_drop_pct > 10.0: color, fill_color = "#E07B00", "#F5A623"
        else: color, fill_color = "#1A8C3E", "#2ECC71"  # verde speranza, drop < 10%
        CircleMarker(location=[float(r["_lat"]), float(r["_lon"])], radius=1.7,
                     color=color, weight=1, fill=True, fill_color=fill_color,
                     fill_opacity=0.85, tooltip=tip).add_to(m)
    return m


def make_map_lv_current_branches(
    *,
    center,
    gdf_poles_4326,
    pole_id_col: str,
    gdf_edges_4326=None,
    mst_edges_latlon=None,
    gdf_roads_4326=None,
    zoom_start: int = 15,
    slack_pole_id=None,
    line_loading_pu=None,
    reinforced_line_pairs=None,
    line_s_nom_kva: Optional[Dict[Tuple[int, int], float]] = None,
    v_nom_kv: float = 0.4,
    s_nom_kva_fallback: float = 100.0,
) -> Map:
    """
    Map 2: only slack pole shown (purple); branches colored by estimated current.
    gray        = 0 A (no load)
    0-25 A      = viola #9B45C0
    25-50 A     = viola scuro #7B1FA2
    50+ A       = viola molto scuro #4A0072
    > I_nom     = red #CC0000 (overcurrent — overrides all)
    orange      = reinforced cable

    Current estimate per line: I = loading_pu * I_nom
    where I_nom = s_nom_kva * 1000 / (v_nom_kv * 1000 * sqrt(3)).
    If line_s_nom_kva is provided each line uses its own thermal rating;
    otherwise falls back to s_nom_kva_fallback (default 100 kVA).
    """
    line_loading_pu = line_loading_pu or {}
    reinforced_line_pairs = reinforced_line_pairs or set()
    line_s_nom_kva = line_s_nom_kva or {}

    def _i_nom_a(u: int, v: int) -> float:
        s_kva = line_s_nom_kva.get((u, v)) or line_s_nom_kva.get((v, u)) or s_nom_kva_fallback
        return float(s_kva) * 1000.0 / (float(v_nom_kv) * 1000.0 * 1.73205)
    m = folium.Map(location=[float(center[0]), float(center[1])], zoom_start=int(zoom_start),
                   tiles="OpenStreetMap", control_scale=True)

    # Roads
    if gdf_roads_4326 is not None and not gdf_roads_4326.empty:
        for _, row in gdf_roads_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            if geom.geom_type == "LineString":
                PolyLine([(lat, lon) for lon, lat in geom.coords], color="#CCCCCC", weight=1.5, opacity=0.5).add_to(m)

    def _cur_color(loading, i_nom: float = 144.0):
        if not np.isfinite(loading) or loading <= 0:
            return "#AAAAAA"
        i_est = float(loading) * i_nom
        if i_est <= 25.0:  return "#9B45C0"  # viola         0-25 A
        elif i_est <= 50.0: return "#7B1FA2"  # viola scuro  25-50 A
        else:               return "#4A0072"  # viola molto   50+ A

    # Branches: colored by current
    if gdf_edges_4326 is not None and not gdf_edges_4326.empty:
        u_col, v_col = _infer_edge_uv_cols(gdf_edges_4326)
        for _, row in gdf_edges_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty: continue
            u = int(pd.to_numeric(row[u_col], errors="coerce"))
            v = int(pd.to_numeric(row[v_col], errors="coerce"))
            key = (u, v) if (u, v) in line_loading_pu else (v, u)
            loading = float(line_loading_pu.get(key, np.nan))
            is_reinforced = (min(u, v), max(u, v)) in reinforced_line_pairs
            i_nom = _i_nom_a(u, v)
            i_est = float(loading) * i_nom if np.isfinite(loading) else 0.0
            if i_est > i_nom:
                color = "#CC0000"  # red — overcurrent (I > I_nom)
            elif is_reinforced:
                color = "#FF8C00"  # orange — reinforced cable
            else:
                color = _cur_color(loading, i_nom)
            tip = f"Line {u}-{v} - ~{i_est:.0f} A (I_nom={i_nom:.0f} A)"
            def _draw_col(ls):
                PolyLine([(lat, lon) for lon, lat in ls.coords], color=color, weight=2.5,
                         opacity=0.9, tooltip=tip).add_to(m)
            if geom.geom_type == "LineString": _draw_col(geom)
            elif geom.geom_type == "MultiLineString":
                for ls in geom.geoms: _draw_col(ls)
    elif mst_edges_latlon is not None:
        for (lat1, lon1), (lat2, lon2) in list(mst_edges_latlon):
            PolyLine([(lat1, lon1), (lat2, lon2)], color="#AAAAAA", weight=2, opacity=0.8).add_to(m)

    # Nodes: only slack pole shown (purple); all other nodes removed for clarity
    if slack_pole_id is not None and pole_id_col in gdf_poles_4326.columns:
        gdfp = gdf_poles_4326.copy()
        gdfp["_pid"] = pd.to_numeric(gdfp[pole_id_col], errors="coerce")
        gdfp = gdfp.dropna(subset=["_pid"]).copy()
        gdfp["_pid"] = gdfp["_pid"].astype(int)
        slack_row = gdfp[gdfp["_pid"] == int(slack_pole_id)]
        for _, r in slack_row.iterrows():
            pt = r.geometry if r.geometry.geom_type == "Point" else r.geometry.representative_point()
            CircleMarker(location=[float(pt.y), float(pt.x)], radius=5,
                         color="purple", weight=1, fill=True, fill_color="purple",
                         fill_opacity=1.0, tooltip=f"Slack pole {slack_pole_id}").add_to(m)
    return m


def make_map_lv_with_pf_violations(
    *,
    center: Tuple[float, float],
    gdf_poles_4326: gpd.GeoDataFrame,
    pole_id_col: str,
    gdf_edges_4326: Optional[gpd.GeoDataFrame] = None,
    mst_edges_latlon: Optional[Iterable[tuple[tuple[float, float], tuple[float, float]]]] = None,
    gdf_roads_4326: Optional[gpd.GeoDataFrame] = None,
    zoom_start: int = 15,
    slack_pole_id: Optional[int] = None,
    # PF results (recommended)
    bus_v_pu: Optional[Dict[int, float]] = None,
    line_loading_pu: Optional[Dict[Tuple[int, int], float]] = None,  # (u,v) -> loading in p.u.
    reinforced_line_pairs: Optional[set[Tuple[int, int]]] = None,
    v_min_pu: float = 0.90,
    v_max_pu: float = 1.10,
    line_loading_limit_pu: float = 1.00,  # 1.0 = 100%
    show_legend: bool = True,
) -> Map:
    """
    Map:
      - OSM basemap
      - roads (gray, optional)
      - edges colored by loading violation (red if > limit, else blue/gray)
      - poles colored by voltage violation (red if outside [v_min,v_max], else black)
      - tooltips ALWAYS show pole_id; if voltage available, show v_pu as well
    """

    bus_v_pu = bus_v_pu or {}
    line_loading_pu = line_loading_pu or {}
    reinforced_line_pairs = reinforced_line_pairs or set()

    m = folium.Map(
        location=[float(center[0]), float(center[1])],
        zoom_start=int(zoom_start),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -----------------------------
    # Roads layer
    # -----------------------------
    if gdf_roads_4326 is not None and not gdf_roads_4326.empty:
        for _, row in gdf_roads_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in geom.coords]
                PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = [(lat, lon) for lon, lat in line.coords]
                    PolyLine(coords, color="gray", weight=3, opacity=0.6).add_to(m)

    # -----------------------------
    # Poles dataframe with coords
    # -----------------------------
    if pole_id_col not in gdf_poles_4326.columns:
        raise ValueError(f"Pole id column '{pole_id_col}' not found in poles GeoDataFrame.")

    gdfp = gdf_poles_4326.copy()
    gdfp["_pid"] = pd.to_numeric(gdfp[pole_id_col], errors="coerce")
    gdfp = gdfp.dropna(subset=["_pid"]).copy()
    gdfp["_pid"] = gdfp["_pid"].astype(int)

    # representative point coords
    pts = gdfp.geometry.apply(lambda geom: geom if geom.geom_type == "Point" else geom.representative_point())
    gdfp["_lat"] = pts.y.astype(float)
    gdfp["_lon"] = pts.x.astype(float)

    # voltage attach
    gdfp["_vpu"] = gdfp["_pid"].map(bus_v_pu)

    # -----------------------------
    # Edges layer (prefer gdf_edges_4326 if available, else mst_edges_latlon)
    # -----------------------------
    if gdf_edges_4326 is not None and not gdf_edges_4326.empty:
        u_col, v_col = _infer_edge_uv_cols(gdf_edges_4326)

        def _loading_color(loading: float) -> str:
            if not np.isfinite(loading):
                return "#5B6B7A"
            ratio = float(np.clip(loading / max(line_loading_limit_pu, 1e-9), 0.0, 2.0))
            if ratio <= 1.0:
                g = int(round(166 + (1.0 - ratio) * 54))
                r = int(round(40 + ratio * 180))
                b = int(round(74 - ratio * 34))
                return f"#{r:02X}{g:02X}{b:02X}"
            excess = min(1.0, ratio - 1.0)
            r = 220
            g = int(round(166 - excess * 116))
            b = int(round(40 - excess * 24))
            return f"#{r:02X}{g:02X}{b:02X}"

        for _, row in gdf_edges_4326.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            u = int(pd.to_numeric(row[u_col], errors="coerce"))
            v = int(pd.to_numeric(row[v_col], errors="coerce"))
            key = (u, v) if (u, v) in line_loading_pu else (v, u)
            is_reinforced = (min(u, v), max(u, v)) in reinforced_line_pairs

            loading = float(line_loading_pu.get(key, np.nan))
            is_viol = (np.isfinite(loading) and loading > line_loading_limit_pu)

            if np.isfinite(loading):
                ratio = max(0.0, loading / max(line_loading_limit_pu, 1e-9))
                weight = 2.5 + 2.0 * min(1.0, ratio) + 4.0 * max(0.0, min(1.0, ratio - 1.0))
            else:
                weight = 2.5

            color = "#FF8C00" if is_reinforced else _loading_color(loading)
            opacity = 0.95 if is_viol else 0.85
            dash_array = None if not is_viol else "10, 8"

            def _draw_linestring(ls):
                coords = [(lat, lon) for lon, lat in ls.coords]
                tip = f"Line {u}-{v}"
                if np.isfinite(loading):
                    tip += f" — loading: {100*loading:.1f}%"
                if is_viol:
                    tip += " (OVERLOADED)"
                if is_reinforced:
                    tip += " (REINFORCED)"
                PolyLine(
                    coords,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    tooltip=tip,
                    dash_array=dash_array,
                ).add_to(m)

            if geom.geom_type == "LineString":
                _draw_linestring(geom)
            elif geom.geom_type == "MultiLineString":
                for ls in geom.geoms:
                    _draw_linestring(ls)

    elif mst_edges_latlon is not None:
        # fallback: no IDs, so just draw in blue (no violation logic possible)
        for (lat1, lon1), (lat2, lon2) in list(mst_edges_latlon):
            PolyLine([(lat1, lon1), (lat2, lon2)], color="blue", weight=2, opacity=0.8).add_to(m)

    # -----------------------------
    # Poles layer: color by voltage violation
    # -----------------------------
    for _, r in gdfp.iterrows():
        pid = int(r["_pid"])
        vpu = r["_vpu"]
        is_slack = slack_pole_id is not None and pid == int(slack_pole_id)

        violates_v = False
        if vpu is not None and np.isfinite(vpu):
            violates_v = (float(vpu) < float(v_min_pu)) or (float(vpu) > float(v_max_pu))

        # priority: slack highlight > violation > normal
        if is_slack:
            color = "purple"
            rad = 5.0
        elif violates_v:
            color = "red"
            rad = 4.5
        else:
            color = "black"
            rad = 2.8

        tip = f"Pole {pid}"
        if vpu is not None and np.isfinite(vpu):
            tip += f" — V={float(vpu):.3f} p.u."

        CircleMarker(
            location=[float(r["_lat"]), float(r["_lon"])],
            radius=rad,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=tip,
        ).add_to(m)

    # -----------------------------
    # Legend
    # -----------------------------
    if show_legend:
        legend_js = f"""
        <script>
        (function addLegend() {{
            var interval = setInterval(function() {{
                var maps = Object.values(window).filter(function(v) {{
                    return v && typeof v === 'object' && v._leaflet_id !== undefined;
                }});
                if (maps.length === 0) return;
                clearInterval(interval);
                var map = maps[0];

                var legend = L.control({{position: 'bottomleft'}});
                legend.onAdd = function() {{
                    var div = L.DomUtil.create('div', '');
                    div.style.cssText = 'background:white;border:1px solid #999;border-radius:6px;padding:10px 12px;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,.2);line-height:1.8;';
                    div.innerHTML = '<b>PF violations legend</b><br>'
                      + '<span style="display:inline-block;width:10px;height:10px;background:black;border-radius:50%;margin-right:6px;"></span>Bus OK<br>'
                      + '<span style="display:inline-block;width:10px;height:10px;background:red;border-radius:50%;margin-right:6px;"></span>Bus voltage violated<br>'
                      + '<span style="display:inline-block;width:10px;height:10px;background:purple;border-radius:50%;margin-right:6px;"></span>Slack / plant bus<br>'
                      + '<span style="display:inline-block;width:14px;height:3px;background:#28A745;vertical-align:middle;margin-right:6px;"></span>Lightly loaded<br>'
                      + '<span style="display:inline-block;width:14px;height:3px;background:#DCA628;vertical-align:middle;margin-right:6px;"></span>Near limit<br>'
                      + '<span style="display:inline-block;width:14px;height:3px;background:#DC3228;vertical-align:middle;margin-right:6px;"></span>Overloaded<br>'
                      + '<span style="font-size:11px;">V limits: <b>[{v_min_pu:.2f}, {v_max_pu:.2f}]</b> p.u. | '
                      + 'Loading: <b>{100*line_loading_limit_pu:.0f}%</b></span>';
                    return div;
                }};
                legend.addTo(map);
            }}, 200);
        }})();
        </script>
        """
        m.get_root().html.add_child(folium.Element(legend_js))

    return m
