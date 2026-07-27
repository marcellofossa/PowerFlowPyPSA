from __future__ import annotations

"""
omg_converter.py
----------------
Converts OMG (OnSSET for Mini-Grids) GeoPackage output into the three files
required by the Grid Validation page:

    nodes.geojson        — poles (full topology, for map display)
    edges.geojson        — distribution links (full topology, for map display)
    pf_nodes.geojson     — poles (reconnected topology, for power flow)
    pf_edges.geojson     — distribution links (reconnected topology, for power flow)
    associations.csv     — building_id -> nearest pole (full topology)
    pf_associations.csv  — building_id -> nearest CONNECTED pole (for power flow)
    building_metadata_template.xlsx

Only FORMAT A is supported (single .gpkg from the real OMG notebook):
    Trunk_line, Laterals, Service_lines, Poles

Processing steps
-----------------
1.  Load poles → assign sequential pole_id (0-based)
2.  Merge trunk + lateral lines → backbone edges
3.  Snap endpoints to nearest pole (or insert synthetic pole)
4.  Optionally split edges longer than max_span_m
5.  Deduplicate edges (undirected)
6.  RECONNECTION (Navarro-Espinosa & Ochoa, CIRED 2015):
      a. BFS from largest component → identify isolated sub-graphs
      b. Iteratively connect each isolated component to the growing
         main graph via shortest point-to-segment Euclidean distance
      c. Insert new pole at the projection point, split the host edge
7.  Dual associations:
      - assoc_df     : every building → nearest pole (any pole, for map)
      - assoc_pf_df  : every building → nearest pole IN reconnected topology
8.  Summary statistics

Output column conventions
--------------------------
  nodes / pf_nodes : pole_id (int), geometry (Point, EPSG:4326)
  edges / pf_edges : source (int), target (int), length_m (float),
                     geometry (LineString, EPSG:4326)
  associations     : pole_id (int), building_id (str)
"""

import io
from typing import Dict, List, Optional, Tuple
from collections import deque

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SPAN_M:       float = 40.0
SNAP_TOLERANCE_M: float = 5.0
_UTM_FALLBACK           = "EPSG:32633"


# ---------------------------------------------------------------------------
# Public API — FORMAT A only
# ---------------------------------------------------------------------------

def convert_omg_gpkg(
    gpkg_bytes: bytes,
    buildings_bytes: bytes,
    buildings_driver: str = "GPKG",
    max_span_m: float = MAX_SPAN_M,
    apply_span_cap: bool = True,
    default_category: str = "Dummy",
    default_weight: float = 1.0,
) -> Dict[str, bytes]:
    """
    Convert a real OMG notebook GeoPackage to Grid Validation inputs.

    Parameters
    ----------
    gpkg_bytes       : raw bytes of the OMG .gpkg (4 layers)
    buildings_bytes  : raw bytes of the buildings file
    buildings_driver : "GPKG" or "GeoJSON"
    max_span_m       : max edge length before span-splitting (apply_span_cap=True)
    apply_span_cap   : if False keep original OMG lines without modification
    default_category : category for building metadata template
    default_weight   : demand weight for building metadata template

    Returns
    -------
    dict with keys:
        nodes_geojson / edges_geojson          — full topology (map)
        pf_nodes_geojson / pf_edges_geojson    — reconnected topology (PF)
        associations_csv                        — all buildings → nearest pole
        pf_associations_csv                     — buildings → nearest PF pole
        building_metadata_template_xlsx
        _summary
    """
    buf = io.BytesIO(gpkg_bytes)

    gdf_poles_raw    = gpd.read_file(buf, layer="Poles");        buf.seek(0)
    gdf_trunks_raw   = gpd.read_file(buf, layer="Trunk_line");   buf.seek(0)
    gdf_laterals_raw = gpd.read_file(buf, layer="Laterals");     buf.seek(0)
    gdf_service_raw  = gpd.read_file(buf, layer="Service_lines")
    gdf_buildings    = gpd.read_file(io.BytesIO(buildings_bytes))

    if gdf_poles_raw.empty:
        raise ValueError("Poles layer is empty.")

    utm_crs = _detect_utm(gdf_poles_raw)

    # Explode MultiPoint → individual poles
    pole_points: List[Point] = []
    for geom in gdf_poles_raw.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPoint":
            pole_points.extend(list(geom.geoms))
        elif geom.geom_type == "Point":
            pole_points.append(geom)
    if not pole_points:
        raise ValueError("No pole points found in Poles layer.")

    poles_m = gpd.GeoDataFrame(geometry=pole_points, crs=gdf_poles_raw.crs).to_crs(utm_crs)
    pole_coords = np.array([[g.x, g.y] for g in poles_m.geometry])
    all_pole_coords = pole_coords.tolist()
    next_pole_id    = len(all_pole_coords)
    n_original_poles = len(all_pole_coords)

    # Explode lines (trunk + laterals)
    all_lines: List[LineString] = []
    for gdf_raw in [gdf_trunks_raw, gdf_laterals_raw]:
        for geom in gdf_raw.to_crs(utm_crs).geometry:
            if geom is None or geom.is_empty:
                continue
            segs = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            all_lines.extend([s for s in segs if not s.is_empty])
    if not all_lines:
        raise ValueError("No backbone lines found in Trunk_line / Laterals layers.")

    # Service drop length
    service_drop_m = 0.0
    try:
        for geom in gdf_service_raw.to_crs(utm_crs).geometry:
            if geom is None or geom.is_empty:
                continue
            segs = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            service_drop_m += sum(s.length for s in segs)
    except Exception:
        pass

    buildings_m = gdf_buildings.to_crs(utm_crs)

    return _build_output(
        all_lines=all_lines,
        all_pole_coords=all_pole_coords,
        next_pole_id=next_pole_id,
        n_original_poles=n_original_poles,
        buildings_m=buildings_m,
        service_drop_m=service_drop_m,
        utm_crs=utm_crs,
        apply_span_cap=apply_span_cap,
        max_span_m=max_span_m,
        default_category=default_category,
        default_weight=default_weight,
    )


# ---------------------------------------------------------------------------
# Shared processing core
# ---------------------------------------------------------------------------

def _build_output(
    all_lines: List[LineString],
    all_pole_coords: list,
    next_pole_id: int,
    n_original_poles: int,
    buildings_m: gpd.GeoDataFrame,
    service_drop_m: float,
    utm_crs: str,
    apply_span_cap: bool,
    max_span_m: float,
    default_category: str = "Dummy",
    default_weight: float = 1.0,
) -> Dict[str, bytes]:
    """Edge-building, reconnection, associations and serialisation."""

    # ── 1. Build edges ───────────────────────────────────────────────────────
    edge_records, edge_geoms = [], []
    for line in all_lines:
        sub_lines = _split_line(line, max_span_m) if apply_span_cap else [line]
        for seg in sub_lines:
            src_id, all_pole_coords, next_pole_id = _snap_or_insert(
                Point(seg.coords[0]), all_pole_coords, next_pole_id, SNAP_TOLERANCE_M)
            tgt_id, all_pole_coords, next_pole_id = _snap_or_insert(
                Point(seg.coords[-1]), all_pole_coords, next_pole_id, SNAP_TOLERANCE_M)
            if src_id == tgt_id:
                continue
            edge_records.append({"source": src_id, "target": tgt_id,
                                  "length_m": round(float(seg.length), 2)})
            edge_geoms.append(seg)

    if not edge_records:
        raise ValueError("No edges could be built from backbone lines.")

    # ── 2. Deduplicate edges ─────────────────────────────────────────────────
    edge_df = pd.DataFrame(edge_records)
    edge_df["_geom"] = edge_geoms
    edge_df["_a"]    = edge_df[["source", "target"]].min(axis=1)
    edge_df["_b"]    = edge_df[["source", "target"]].max(axis=1)
    edge_df = (edge_df.sort_values("length_m")
                      .drop_duplicates(subset=["_a", "_b"])
                      .reset_index(drop=True))
    edge_geoms_clean = edge_df["_geom"].tolist()
    edge_df = edge_df.drop(columns=["_geom", "_a", "_b"])

    # ── 3. Pre-reconnection edges GDF (original OMG lines — for map display) ──
    # Built here so the map shows original lines without reconnection artefacts.
    gdf_edges_m  = gpd.GeoDataFrame(edge_df, geometry=edge_geoms_clean, crs=utm_crs)
    gdf_edges    = gdf_edges_m.to_crs("EPSG:4326")

    # ── 4. Full associations: every building → nearest pole (any) ───────────
    bld_centroids = buildings_m.copy()
    bld_centroids["geometry"]    = buildings_m.geometry.apply(_safe_centroid)
    bld_centroids = bld_centroids[~bld_centroids.geometry.is_empty].reset_index(drop=True)
    bld_centroids["building_id"] = bld_centroids.index.astype(str)
    bld_coords   = np.array([[g.x, g.y] for g in bld_centroids.geometry])
    pole_arr_all = np.array(all_pole_coords)
    _, nn_all    = cKDTree(pole_arr_all).query(bld_coords, k=1)
    assoc_df = pd.DataFrame({
        "pole_id":     nn_all.astype(int),
        "building_id": bld_centroids["building_id"].values,
    })

    # ── 5. RECONNECTION (Navarro-Espinosa & Ochoa, CIRED 2015) ──────────────
    # Build graph, find connected components, iteratively reconnect isolated
    # sub-graphs to the growing main component (largest first).
    pf_edge_df, pf_edge_geoms, n_reconnection_edges = _reconnect_components(
        edge_df=edge_df,
        edge_geoms=edge_geoms_clean,
        all_pole_coords=all_pole_coords,
    )

    # ── 3b. Full topology nodes — built AFTER reconnection so that junction
    #         poles inserted during _reconnect_components are included.
    #         powerflow_network.py needs every bus referenced by an edge.
    pole_geoms_m = [Point(c[0], c[1]) for c in all_pole_coords]
    gdf_nodes_m  = gpd.GeoDataFrame(
        {"pole_id": list(range(len(pole_geoms_m)))},
        geometry=pole_geoms_m, crs=utm_crs)
    gdf_nodes    = gdf_nodes_m.to_crs("EPSG:4326")

    # PF poles: only those that appear in at least one PF edge
    pf_edge_poles = (
        set(pf_edge_df["source"].astype(int).tolist()) |
        set(pf_edge_df["target"].astype(int).tolist())
    )
    pf_pole_ids = sorted(pf_edge_poles)
    pf_n_poles  = len(all_pole_coords)  # may have grown during reconnection

    pf_nodes_m = gpd.GeoDataFrame(
        {"pole_id": pf_pole_ids},
        geometry=[Point(all_pole_coords[pid]) for pid in pf_pole_ids],
        crs=utm_crs,
    )
    pf_edges_m = gpd.GeoDataFrame(pf_edge_df, geometry=pf_edge_geoms, crs=utm_crs)
    gdf_pf_nodes = pf_nodes_m.to_crs("EPSG:4326")
    gdf_pf_edges = pf_edges_m.to_crs("EPSG:4326")

    # ── 6. PF associations: every building → nearest pole IN PF topology ────
    pole_arr_pf  = np.array([all_pole_coords[pid] for pid in pf_pole_ids])
    _, nn_pf_idx = cKDTree(pole_arr_pf).query(bld_coords, k=1)
    nn_pf_global = np.array([pf_pole_ids[i] for i in nn_pf_idx])
    assoc_pf_df  = pd.DataFrame({
        "pole_id":     nn_pf_global.astype(int),
        "building_id": bld_centroids["building_id"].values,
    })
    n_reassigned = int(np.sum(nn_all != nn_pf_global))

    # ── 7. Fallback service drop ─────────────────────────────────────────────
    if service_drop_m == 0.0:
        try:
            pg_by_id = {int(r.pole_id): r.geometry for _, r in gdf_nodes_m.iterrows()}
            bg_by_id = {int(r.building_id): r.geometry for _, r in bld_centroids.iterrows()}
            for _, row in assoc_df.iterrows():
                pg = pg_by_id.get(int(row["pole_id"]))
                bg = bg_by_id.get(int(row["building_id"]))
                if pg is not None and bg is not None:
                    service_drop_m += float(pg.distance(bg))
        except Exception:
            pass

    # ── 8. Building metadata template ───────────────────────────────────────
    meta_buf = io.BytesIO()
    pd.DataFrame({
        "building_id": bld_centroids["building_id"].values,
        "category":    default_category,
        "weight":      default_weight,
    }).to_excel(meta_buf, index=False)

    # ── 9. Summary ───────────────────────────────────────────────────────────
    n_synthetic = len(all_pole_coords) - n_original_poles
    summary = _compute_summary(
        gdf_nodes_m, gdf_edges_m, assoc_df,
        n_buildings=len(bld_centroids),
        n_synthetic=n_synthetic,
        n_reconnection_edges=n_reconnection_edges,
        n_reassigned=n_reassigned,
        service_drop_m=service_drop_m,
        apply_span_cap=apply_span_cap,
        max_span_m=max_span_m,
    )

    return {
        # Full topology — for map display
        "nodes_geojson":                   _gdf_to_geojson_bytes(gdf_nodes),
        "edges_geojson":                   _gdf_to_geojson_bytes(gdf_edges),
        # Reconnected topology — for power flow
        "pf_nodes_geojson":                _gdf_to_geojson_bytes(gdf_pf_nodes),
        "pf_edges_geojson":                _gdf_to_geojson_bytes(gdf_pf_edges),
        # Associations
        "associations_csv":                assoc_df.to_csv(index=False, sep=";").encode("utf-8"),
        "pf_associations_csv":             assoc_pf_df.to_csv(index=False, sep=";").encode("utf-8"),
        "building_metadata_template_xlsx": meta_buf.getvalue(),
        "_summary": summary,
    }


# ---------------------------------------------------------------------------
# Reconnection (Navarro-Espinosa & Ochoa, CIRED 2015 — Step 3)
# ---------------------------------------------------------------------------

def _reconnect_components(
    edge_df: pd.DataFrame,
    edge_geoms: List[LineString],
    all_pole_coords: list,
) -> Tuple[pd.DataFrame, List[LineString], int]:
    """
    Iteratively connect isolated sub-graphs to the largest component.

    Algorithm:
    1. Build undirected NetworkX graph from current edges.
    2. Find all connected components; largest = main component.
    3. For each isolated component (sorted largest-first):
       a. For every pole in the isolated component, compute distance to
          every edge segment in the current main component
          (point-to-segment Euclidean distance).
       b. Find the global minimum: (isolated_pole, host_edge, proj_point).
       c. If proj_point falls strictly inside the host edge:
            - Insert new pole at proj_point
            - Split host edge into two new edges
            - Add edge isolated_pole → new_pole
          Else:
            - Add edge isolated_pole → nearest endpoint of host edge
       d. Merge isolated component into main component and repeat.
    4. Return updated edge_df and edge_geoms.
    """
    # Work on mutable lists so we can append new poles
    pole_coords = list(all_pole_coords)
    next_id     = len(pole_coords)

    edges_list  = edge_df.to_dict("records")   # list of {source, target, length_m}
    geoms_list  = list(edge_geoms)

    n_reconnection = 0

    for _iteration in range(500):   # safety cap
        G = nx.Graph()
        for e in edges_list:
            G.add_edge(int(e["source"]), int(e["target"]))

        components = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(components) <= 1:
            break

        main_comp   = set(components[0])
        # Collect isolated components from largest to smallest
        iso_comps   = [set(c) for c in components[1:]]

        # Build spatial index for main-component edge segments
        main_edge_indices = [
            i for i, e in enumerate(edges_list)
            if int(e["source"]) in main_comp and int(e["target"]) in main_comp
        ]

        best_dist   = float("inf")
        best_iso_pid = None
        best_edge_i  = None
        best_proj    = None
        best_iso_comp = None

        for iso_comp in iso_comps:
            for iso_pid in iso_comp:
                iso_pt = Point(pole_coords[iso_pid])
                for ei in main_edge_indices:
                    seg = geoms_list[ei]
                    proj = seg.interpolate(seg.project(iso_pt))
                    dist = iso_pt.distance(proj)
                    if dist < best_dist:
                        best_dist    = dist
                        best_iso_pid = iso_pid
                        best_edge_i  = ei
                        best_proj    = proj
                        best_iso_comp = iso_comp
            # Connect one component per iteration (the closest one found so far)
            # to avoid O(n²) growth; break after first improvement
            if best_iso_pid is not None:
                break

        if best_iso_pid is None:
            break

        # Determine connection point
        host_seg  = geoms_list[best_edge_i]
        host_edge = edges_list[best_edge_i]
        t         = host_seg.project(Point(pole_coords[best_iso_pid]))
        eps       = 1e-3   # 1 mm tolerance for "strictly inside"

        if eps < t < host_seg.length - eps:
            # Project point is interior to the segment → split
            new_pid = next_id
            next_id += 1
            pole_coords.append([best_proj.x, best_proj.y])

            src, tgt = int(host_edge["source"]), int(host_edge["target"])
            seg_a = LineString([Point(pole_coords[src]), best_proj])
            seg_b = LineString([best_proj, Point(pole_coords[tgt])])
            conn_seg = LineString([Point(pole_coords[best_iso_pid]), best_proj])

            # Remove host edge and add three new ones
            edges_list.pop(best_edge_i)
            geoms_list.pop(best_edge_i)

            for new_src, new_tgt, new_seg in [
                (src, new_pid, seg_a),
                (new_pid, tgt, seg_b),
                (best_iso_pid, new_pid, conn_seg),
            ]:
                edges_list.append({"source": new_src, "target": new_tgt,
                                    "length_m": round(float(new_seg.length), 2)})
                geoms_list.append(new_seg)

        else:
            # Projection outside segment → connect to nearest endpoint
            src, tgt = int(host_edge["source"]), int(host_edge["target"])
            pt_src = Point(pole_coords[src])
            pt_tgt = Point(pole_coords[tgt])
            iso_pt = Point(pole_coords[best_iso_pid])
            conn_tgt = src if iso_pt.distance(pt_src) <= iso_pt.distance(pt_tgt) else tgt
            conn_seg = LineString([iso_pt, Point(pole_coords[conn_tgt])])
            edges_list.append({"source": best_iso_pid, "target": conn_tgt,
                                "length_m": round(float(conn_seg.length), 2)})
            geoms_list.append(conn_seg)

        n_reconnection += 1
        # Update pole_coords in the caller's namespace
        all_pole_coords.clear()
        all_pole_coords.extend(pole_coords)

    pf_edge_df = pd.DataFrame(edges_list)
    return pf_edge_df, geoms_list, n_reconnection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_utm(gdf: gpd.GeoDataFrame) -> str:
    try:
        bounds = gdf.to_crs("EPSG:4326").total_bounds
        lon    = float((bounds[0] + bounds[2]) / 2)
        lat    = float((bounds[1] + bounds[3]) / 2)
        zone   = int((lon + 180) / 6) + 1
        epsg   = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"
    except Exception:
        return _UTM_FALLBACK


def _split_line(line: LineString, max_len: float) -> List[LineString]:
    total = line.length
    if total <= max_len:
        return [line]
    n_segs  = int(np.ceil(total / max_len))
    seg_len = total / n_segs
    return [
        LineString([line.interpolate(i * seg_len),
                    line.interpolate(min((i + 1) * seg_len, total))])
        for i in range(n_segs)
    ]


def _snap_or_insert(point: Point, pole_coords: list, next_id: int, tol: float):
    arr   = np.array(pole_coords)
    dists = np.sqrt(((arr - [point.x, point.y]) ** 2).sum(axis=1))
    idx   = int(np.argmin(dists))
    if dists[idx] <= tol:
        return idx, pole_coords, next_id
    pole_coords = pole_coords + [[point.x, point.y]]
    return next_id, pole_coords, next_id + 1


def _safe_centroid(geom):
    if geom is None or geom.is_empty:
        return Point()
    return geom.centroid


def _gdf_to_geojson_bytes(gdf: gpd.GeoDataFrame) -> bytes:
    return gdf.to_json(show_bbox=False).encode("utf-8")


def _compute_summary(
    gdf_nodes_m, gdf_edges_m, assoc_df,
    n_buildings, n_synthetic,
    n_reconnection_edges=0,
    n_reassigned=0,
    service_drop_m=0.0,
    apply_span_cap=True,
    max_span_m=MAX_SPAN_M,
) -> dict:
    backbone_m  = float(gdf_edges_m.geometry.length.sum())
    total_m     = backbone_m + service_drop_m
    serving_ids = set(assoc_df["pole_id"].unique())
    n_serving   = int(sum(1 for pid in gdf_nodes_m["pole_id"] if int(pid) in serving_ids))
    lengths     = gdf_edges_m.geometry.length
    return {
        "n_poles":                int(len(gdf_nodes_m)),
        "n_poles_original":       int(len(gdf_nodes_m)) - n_synthetic,
        "n_poles_synthetic":      n_synthetic,
        "n_edges":                int(len(gdf_edges_m)),
        "n_buildings":            n_buildings,
        "n_assoc":                int(len(assoc_df)),
        "n_serving_poles":        n_serving,
        "n_support_poles":        int(len(gdf_nodes_m)) - n_serving,
        "n_reconnection_edges":   int(n_reconnection_edges),
        "n_reassigned":           int(n_reassigned),
        "total_length_km":        round(total_m / 1000.0, 3),
        "backbone_length_km":     round(backbone_m / 1000.0, 3),
        "service_drop_length_km": round(service_drop_m / 1000.0, 3),
        "max_span_m":             round(float(lengths.max()), 1) if len(lengths) else 0.0,
        "mean_span_m":            round(float(lengths.mean()), 1) if len(lengths) else 0.0,
        "apply_span_cap":         apply_span_cap,
        "cap_value_m":            max_span_m,
    }
