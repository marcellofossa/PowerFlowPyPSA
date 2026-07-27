from __future__ import annotations

"""
offgrid_converter.py
--------------------
Converts an OffGridPlanner Excel export (.xlsx) into the three files
required by the Grid Validation page:

    nodes.geojson        — poles only (276 in a typical run)
    edges.geojson        — distribution links only (pole-to-pole backbone)
    associations.csv     — mapping: building_id -> pole_id
    building_metadata_template.csv  — pre-filled template (category = "Dummy")

OffGridPlanner Excel sheets used
---------------------------------
  nodes : all network nodes
          Node type == "pole"     -> grid buses  (-> nodes.geojson)
          Node type == "consumer" -> buildings   (-> associations + metadata)

  links : all cable segments
          Link type == "distribution" -> pole-to-pole backbone (-> edges.geojson)
          Link type == "connection"   -> service drops (pole<->building, ignored for edges)

Matching consumers to poles
----------------------------
OffGridPlanner "connection" links already encode the consumer<->pole mapping:
  - Lat from / Lon from : consumer coordinates
  - Lat to   / Lon to   : pole coordinates  (or vice-versa)
We identify which endpoint belongs to a consumer and which to a pole by
cross-referencing with the nodes sheet.

If a consumer cannot be matched via connection links (e.g. Length == 0 cases
where consumer and pole share the same coordinates), we fall back to a
nearest-pole spatial join.

Output column conventions (must match powerflow_validation.py)
---------------------------------------------------------------
  nodes.geojson  : pole_id (int), geometry (Point, EPSG:4326)
  edges.geojson  : source (int), target (int), length_m (float), geometry (LineString, EPSG:4326)
  associations   : pole_id (int), building_id (str)
  building_meta  : building_id (str), category (str), weight (float)
"""

import io
import json
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _auto_utm(gdf) -> str:
    """Detect the correct UTM CRS from the bounding box of a GeoDataFrame."""
    try:
        bounds = gdf.to_crs("EPSG:4326").total_bounds  # [minx, miny, maxx, maxy]
        lon = float((bounds[0] + bounds[2]) / 2)
        lat = float((bounds[1] + bounds[3]) / 2)
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"
    except Exception:
        return "EPSG:32633"


def _compute_summary(gdf_poles, gdf_edges, assoc_df, consumers_df, unmatched) -> dict:
    """
    Compute network summary statistics matching the Grid Topology metrics style.
    Lengths computed from GeoJSON geometry reprojected to auto-detected UTM.
    """
    utm_crs = _auto_utm(gdf_poles)

    # Backbone length: sum of distribution edge lengths
    backbone_m = 0.0
    try:
        gdf_edges_utm = gdf_edges.to_crs(utm_crs)
        backbone_m = float(gdf_edges_utm.geometry.length.sum())
    except Exception:
        if "length_m" in gdf_edges.columns:
            backbone_m = float(gdf_edges["length_m"].sum())

    # Service drop length: sum of pole-to-building distances
    service_drop_m = 0.0
    try:
        gdf_poles_utm = gdf_poles.to_crs(utm_crs)
        consumers_pts = consumers_df.copy()
        # consumers_df has Latitude/Longitude columns
        if "Latitude" in consumers_df.columns and "Longitude" in consumers_df.columns:
            import geopandas as gpd
            from shapely.geometry import Point
            geoms = [Point(float(r["Longitude"]), float(r["Latitude"])) for _, r in consumers_df.iterrows()]
            consumers_gdf = gpd.GeoDataFrame(consumers_df.copy(), geometry=geoms, crs="EPSG:4326").to_crs(utm_crs)
            pole_geom_by_id = gdf_poles_utm.set_index("pole_id").geometry.to_dict()
            for _, row in assoc_df.iterrows():
                pid = int(row["pole_id"])
                bid = int(row["building_id"])
                pg = pole_geom_by_id.get(pid)
                bg = consumers_gdf.geometry.iloc[bid] if bid < len(consumers_gdf) else None
                if pg and bg:
                    service_drop_m += float(pg.distance(bg))
    except Exception:
        pass

    total_m = backbone_m + service_drop_m

    # Serving vs support poles
    serving_ids = set(assoc_df["pole_id"].unique()) if not assoc_df.empty else set()
    n_serving = int(sum(1 for pid in gdf_poles["pole_id"] if int(pid) in serving_ids))
    n_support = int(len(gdf_poles)) - n_serving

    return {
        "n_poles":               int(len(gdf_poles)),
        "n_edges":               int(len(gdf_edges)),
        "n_buildings":           int(len(consumers_df)),
        "n_assoc":               int(len(assoc_df)),
        "n_unmatched_fallback":  int(len(unmatched)),
        "n_serving_poles":       n_serving,
        "n_support_poles":       n_support,
        "total_length_km":       round(total_m / 1000.0, 3),
        "backbone_length_km":    round(backbone_m / 1000.0, 3),
        "service_drop_length_km": round(service_drop_m / 1000.0, 3),
    }


def convert_offgridplanner_excel(
    excel_bytes: bytes,
    default_category: str = "Dummy",
    default_weight: float = 1.0,
    coord_tolerance_deg: float = 1e-5,
    fallback_tol_deg: float = 5e-4,
) -> Dict[str, bytes]:
    """
    Convert an OffGridPlanner Excel export to Grid Validation inputs.

    Parameters
    ----------
    excel_bytes        : raw bytes of the .xlsx file
    default_category   : category assigned to all buildings in the metadata template
    default_weight     : demand weight assigned to all buildings
    coord_tolerance_deg: tolerance (degrees) for the primary exact-grid coordinate match
    fallback_tol_deg   : tolerance (degrees, ~ fallback_tol_deg * 111000 m) for the
                         nearest-neighbor fallback used when a distribution-link
                         endpoint cannot be matched exactly (observed coordinate
                         precision drift in some OffGridPlanner exports, up to
                         a few tens of metres). Default 5e-4 deg ~= 55 m.

    Returns
    -------
    dict with keys:
        "nodes_geojson"                  : bytes (GeoJSON)
        "edges_geojson"                  : bytes (GeoJSON)
        "associations_csv"               : bytes (CSV)
        "building_metadata_template_csv" : bytes (CSV)
    """
    xl = pd.ExcelFile(io.BytesIO(excel_bytes))

    # ------------------------------------------------------------------
    # 1) Read sheets
    # ------------------------------------------------------------------
    nodes_df = pd.read_excel(xl, sheet_name="nodes")
    links_df = pd.read_excel(xl, sheet_name="links")

    _validate_sheets(nodes_df, links_df)

    # ------------------------------------------------------------------
    # 2) Split nodes into poles and consumers
    # ------------------------------------------------------------------
    # Normalise node type column (lowercase, strip whitespace)
    nodes_df["Node type"] = nodes_df["Node type"].astype(str).str.strip().str.lower()

    poles_df = nodes_df[nodes_df["Node type"] == "pole"].copy().reset_index(drop=True)
    consumers_df = nodes_df[nodes_df["Node type"] == "consumer"].copy().reset_index(drop=True)

    if poles_df.empty:
        raise ValueError("No poles found in the 'nodes' sheet (Node type == 'pole').")
    if consumers_df.empty:
        raise ValueError("No consumers found in the 'nodes' sheet (Node type == 'consumer').")

    # Assign integer pole_id (0-based index)
    poles_df["pole_id"] = poles_df.index.astype(int)

    # Assign integer building_id (0-based index, stored as string for compatibility)
    consumers_df["building_id"] = consumers_df.index.astype(int).astype(str)

    # ------------------------------------------------------------------
    # 3) Build nodes GeoDataFrame (poles only)
    # ------------------------------------------------------------------
    pole_geometries = [
        Point(float(row["Longitude"]), float(row["Latitude"]))
        for _, row in poles_df.iterrows()
    ]
    gdf_poles = gpd.GeoDataFrame(
        {"pole_id": poles_df["pole_id"].values},
        geometry=pole_geometries,
        crs="EPSG:4326",
    )

    # ------------------------------------------------------------------
    # 4) Build edges GeoDataFrame (distribution links = pole-to-pole)
    # ------------------------------------------------------------------
    dist_links = links_df[
        links_df["Link type"].astype(str).str.strip().str.lower() == "distribution"
    ].copy().reset_index(drop=True)

    if dist_links.empty:
        raise ValueError("No distribution links found in the 'links' sheet.")

    # Build a lookup: rounded (lat, lon) -> pole_id for fast exact matching,
    # plus a KDTree for nearest-neighbor fallback when the exact match fails
    # (handles small coordinate precision drift between 'nodes' and 'links' sheets).
    pole_coord_index = _build_coord_index(poles_df, tol=coord_tolerance_deg)
    pole_kdtree = _build_pole_kdtree(poles_df)

    edge_records = []
    edge_geometries = []
    fallback_matches: list[dict] = []
    unmatched_edges = 0

    for _, row in dist_links.iterrows():
        lat_from = float(row["Lat from"])
        lon_from = float(row["Lon from"])
        lat_to   = float(row["Lat to"])
        lon_to   = float(row["Lon to"])

        # Match endpoints to pole_ids (exact grid match first)
        u = _lookup_pole(lat_from, lon_from, pole_coord_index, tol=coord_tolerance_deg)
        v = _lookup_pole(lat_to,   lon_to,   pole_coord_index, tol=coord_tolerance_deg)

        # Fallback to nearest-neighbor when the exact match fails for an endpoint
        if u is None:
            u = _lookup_pole_nearest(lat_from, lon_from, pole_kdtree, max_dist_deg=fallback_tol_deg)
            if u is not None:
                fallback_matches.append({"endpoint": "from", "lat": lat_from, "lon": lon_from, "pole_id": u})
        if v is None:
            v = _lookup_pole_nearest(lat_to, lon_to, pole_kdtree, max_dist_deg=fallback_tol_deg)
            if v is not None:
                fallback_matches.append({"endpoint": "to", "lat": lat_to, "lon": lon_to, "pole_id": v})

        if u is None or v is None:
            # Still unmatched even with the fallback: skip this edge
            unmatched_edges += 1
            continue
        if u == v:
            # Skip self-loops
            continue

        length_m = float(row["Length"]) if not pd.isna(row["Length"]) else 0.0

        edge_records.append({
            "source": int(u),
            "target": int(v),
            "length_m": round(length_m, 4),
        })
        edge_geometries.append(
            LineString([(lon_from, lat_from), (lon_to, lat_to)])
        )

    if not edge_records:
        raise ValueError(
            "Could not match any distribution link endpoints to poles. "
            "Check coordinate precision in the Excel file."
        )

    # Deduplicate undirected edges (u-v and v-u are the same physical cable)
    edge_df = pd.DataFrame(edge_records)
    edge_df["_a"] = edge_df[["source", "target"]].min(axis=1)
    edge_df["_b"] = edge_df[["source", "target"]].max(axis=1)
    edge_df["_geom"] = edge_geometries
    edge_df = edge_df.drop_duplicates(subset=["_a", "_b"]).reset_index(drop=True)
    edge_geometries_clean = edge_df["_geom"].tolist()
    edge_df = edge_df.drop(columns=["_a", "_b", "_geom"])

    gdf_edges = gpd.GeoDataFrame(edge_df, geometry=edge_geometries_clean, crs="EPSG:4326")

    # ------------------------------------------------------------------
    # 5) Build associations (consumer -> pole)
    # ------------------------------------------------------------------
    # Primary strategy: use "connection" links to map consumer coords to pole coords
    conn_links = links_df[
        links_df["Link type"].astype(str).str.strip().str.lower() == "connection"
    ].copy().reset_index(drop=True)

    # Build consumer coord index for matching
    consumer_coord_index = _build_coord_index(consumers_df, tol=coord_tolerance_deg)

    assoc_records = []
    matched_consumer_ids = set()

    for _, row in conn_links.iterrows():
        lat_from = float(row["Lat from"])
        lon_from = float(row["Lon from"])
        lat_to   = float(row["Lat to"])
        lon_to   = float(row["Lon to"])

        # One endpoint is a consumer, the other is a pole.
        # Try to identify which is which.
        consumer_id_from = _lookup_consumer(lat_from, lon_from, consumer_coord_index, tol=coord_tolerance_deg)
        consumer_id_to   = _lookup_consumer(lat_to,   lon_to,   consumer_coord_index, tol=coord_tolerance_deg)

        pole_id_from = _lookup_pole(lat_from, lon_from, pole_coord_index, tol=coord_tolerance_deg)
        pole_id_to   = _lookup_pole(lat_to,   lon_to,   pole_coord_index, tol=coord_tolerance_deg)

        # Case A: from=consumer, to=pole
        if consumer_id_from is not None and pole_id_to is not None:
            building_id = consumers_df.loc[consumer_id_from, "building_id"]
            assoc_records.append({"pole_id": int(pole_id_to), "building_id": str(building_id)})
            matched_consumer_ids.add(consumer_id_from)

        # Case B: from=pole, to=consumer
        elif pole_id_from is not None and consumer_id_to is not None:
            building_id = consumers_df.loc[consumer_id_to, "building_id"]
            assoc_records.append({"pole_id": int(pole_id_from), "building_id": str(building_id)})
            matched_consumer_ids.add(consumer_id_to)

    # Fallback: consumers not matched via connection links -> nearest pole (spatial join)
    unmatched_mask = ~consumers_df.index.isin(matched_consumer_ids)
    unmatched = consumers_df[unmatched_mask]

    if not unmatched.empty:
        # Build arrays for vectorised nearest-neighbour
        pole_lats = poles_df["Latitude"].values.astype(float)
        pole_lons = poles_df["Longitude"].values.astype(float)

        for idx, row in unmatched.iterrows():
            clat = float(row["Latitude"])
            clon = float(row["Longitude"])
            # Euclidean distance in degree space (sufficient for small villages)
            dists = np.sqrt((pole_lats - clat) ** 2 + (pole_lons - clon) ** 2)
            nearest_local = int(np.argmin(dists))
            pole_id = int(poles_df.loc[nearest_local, "pole_id"])
            assoc_records.append({
                "pole_id": pole_id,
                "building_id": str(row["building_id"]),
            })

    assoc_df = (
        pd.DataFrame(assoc_records, columns=["pole_id", "building_id"])
        .drop_duplicates(subset=["building_id"])  # one pole per building
        .sort_values(["pole_id", "building_id"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 6) Building metadata template
    # ------------------------------------------------------------------
    meta_df = pd.DataFrame({
        "building_id": consumers_df["building_id"].values,
        "category": default_category,
        "weight": default_weight,
    })

    # ------------------------------------------------------------------
    # 7) Serialise to bytes
    # ------------------------------------------------------------------
    nodes_geojson   = _gdf_to_geojson_bytes(gdf_poles)
    edges_geojson   = _gdf_to_geojson_bytes(gdf_edges)
    assoc_csv       = assoc_df.to_csv(index=False, sep=";").encode("utf-8")

    # Generate building metadata template as Excel (.xlsx) — no separator ambiguity
    import io as _io
    _meta_buf = _io.BytesIO()
    meta_df.to_excel(_meta_buf, index=False)
    meta_xlsx = _meta_buf.getvalue()

    summary = _compute_summary(gdf_poles, gdf_edges, assoc_df, consumers_df, unmatched)
    # Report nearest-neighbor fallback usage so the UI can flag it to the user
    summary["n_edge_endpoints_fallback_matched"] = len(fallback_matches)
    summary["n_distribution_edges_unmatched"] = unmatched_edges
    if fallback_matches:
        max_drift_deg = max(
            (
                min(
                    ((p["Latitude"] - fm["lat"]) ** 2 + (p["Longitude"] - fm["lon"]) ** 2) ** 0.5
                    for p in [poles_df.iloc[fm["pole_id"]]]
                )
                for fm in fallback_matches
            ),
            default=0.0,
        )
        summary["max_fallback_match_drift_m"] = round(max_drift_deg * 111000, 1)

    return {
        "nodes_geojson":                   nodes_geojson,
        "edges_geojson":                   edges_geojson,
        "associations_csv":                assoc_csv,
        "building_metadata_template_xlsx": meta_xlsx,
        # Summary for UI feedback
        "_summary": summary,
    }


# ---------------------------------------------------------------------------
# Helper: coordinate lookup index
# ---------------------------------------------------------------------------

def _round_coord(lat: float, lon: float, tol: float) -> Tuple[int, int]:
    """Round lat/lon to a grid defined by tolerance for fast dict lookup."""
    factor = 1.0 / tol
    return (round(lat * factor), round(lon * factor))


def _build_coord_index(
    df: pd.DataFrame,
    tol: float,
) -> Dict[Tuple[int, int], int]:
    """
    Build a dict mapping rounded (lat, lon) -> row index in df.
    Used for O(1) coordinate matching between sheets.
    """
    index: Dict[Tuple[int, int], int] = {}
    for i, row in df.iterrows():
        key = _round_coord(float(row["Latitude"]), float(row["Longitude"]), tol)
        index[key] = int(i)
    return index


def _lookup_pole(
    lat: float,
    lon: float,
    pole_index: Dict[Tuple[int, int], int],
    tol: float,
) -> Optional[int]:
    """Return pole_id if (lat, lon) matches a known pole, else None."""
    key = _round_coord(lat, lon, tol)
    local_idx = pole_index.get(key)
    if local_idx is None:
        return None
    return local_idx  # pole_id == local index (assigned in step 2)


def _build_pole_kdtree(poles_df: pd.DataFrame):
    """Build a cKDTree over pole (lat, lon) for nearest-neighbor fallback matching."""
    from scipy.spatial import cKDTree
    coords = poles_df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    return cKDTree(coords)


def _lookup_pole_nearest(
    lat: float,
    lon: float,
    kdtree,
    max_dist_deg: float,
) -> Optional[int]:
    """
    Fallback pole lookup using nearest-neighbor search instead of an exact grid match.
    Some OffGridPlanner exports have a small coordinate precision drift between the
    'nodes' sheet and a 'links' endpoint for the same physical pole (observed up to
    ~25m). This fallback recovers those edges instead of silently dropping them.
    Returns None if the nearest pole is farther than max_dist_deg.
    """
    dist, idx = kdtree.query([lat, lon])
    if dist > max_dist_deg:
        return None
    return int(idx)


def _lookup_consumer(
    lat: float,
    lon: float,
    consumer_index: Dict[Tuple[int, int], int],
    tol: float,
) -> Optional[int]:
    """Return consumer local index if (lat, lon) matches a known consumer, else None."""
    key = _round_coord(lat, lon, tol)
    return consumer_index.get(key)


# ---------------------------------------------------------------------------
# Helper: GeoDataFrame -> GeoJSON bytes
# ---------------------------------------------------------------------------

def _gdf_to_geojson_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    """Serialise a GeoDataFrame to GeoJSON bytes (UTF-8)."""
    return gdf.to_json(show_bbox=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Helper: basic sheet validation
# ---------------------------------------------------------------------------

def _validate_sheets(nodes_df: pd.DataFrame, links_df: pd.DataFrame) -> None:
    required_node_cols = {"Latitude", "Longitude", "Node type"}
    required_link_cols = {"Link type", "Length", "Lat from", "Lon from", "Lat to", "Lon to"}

    missing_nodes = required_node_cols - set(nodes_df.columns)
    missing_links = required_link_cols - set(links_df.columns)

    if missing_nodes:
        raise ValueError(f"'nodes' sheet is missing columns: {missing_nodes}")
    if missing_links:
        raise ValueError(f"'links' sheet is missing columns: {missing_links}")
