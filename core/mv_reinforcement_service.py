from __future__ import annotations

"""
Grid Reinforcement (hybrid MV/LV) orchestration service.

Pure orchestration on top of the existing stations:
- station 1 (core.distribution_service.run_low_voltage) builds each LV
  subnetwork from a cluster of buildings;
- station 2 (core.powerflow_network.HybridPyPSAPowerFlowRunner) solves ONE
  PyPSA network spanning all subnetworks, the MV backbone and the
  transformers.

The k-iteration lives here and only here (core.partitioning never decides k):
- distance cap: iterate partition -> diameter check (cheap, no station 1 /
  no PF inside the loop); topology + PF are built once at the final k.
  Behaviourally identical to iterating topology creation, since cluster
  diameters depend only on the partition.
- voltage cap: iterate partition -> topology -> PF -> per-subnet check of
  max voltage drop <= dv_cap_pu.
"""

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString, Point

from .contracts import (
    MvIterationResult,
    MvReinforcementRequest,
    MvReinforcementResult,
    MvSubnetworkResult,
    SESSION_SCHEMA_VERSION,
    TopologyResult,
    ValidationResult,
)
from .distribution_service import run_low_voltage
from .partitioning import (
    cluster_diameters_m,
    partition_buildings,
    partition_buildings_complete_linkage,
    suggest_k_start,
)
from .powerflow_network import (
    HybridPyPSAPowerFlowRunner,
    LvSubnetSpec,
    MvLayerSpec,
    MvTransformerSpec,
    PFScenarioParams,
    PFTopologyBundle,
    _approx_km_from_latlon,
)
from .powerflow_validation import aggregate_pole_loads


# =============================================================================
# Helpers
# =============================================================================
def auto_size_transformer_kva(
    peak_kw: float,
    pf_load: float,
    sizing_margin: float,
    standard_sizes_kva: Tuple[float, ...],
) -> float:
    """Cluster peak [kW] -> apparent power with margin -> next standard size.

    Above the largest standard size, round up to the next 50 kVA multiple
    (large plant step-up transformers are procured per project).
    """
    if peak_kw < 0:
        raise ValueError(f"peak_kw must be >= 0; got {peak_kw}")
    pf = float(np.clip(pf_load, 1e-6, 1.0))
    s_req = float(peak_kw) / pf * float(sizing_margin)
    if s_req <= 0.0:
        return float(min(standard_sizes_kva))
    for s in sorted(standard_sizes_kva):
        if s >= s_req:
            return float(s)
    return float(math.ceil(s_req / 50.0) * 50.0)


def _plant_xy_m(gdf_buildings: gpd.GeoDataFrame, plant_latlon: Tuple[float, float]) -> Tuple[float, float]:
    """(lat, lon) -> (x, y) in the projected CRS of the buildings."""
    lat, lon = float(plant_latlon[0]), float(plant_latlon[1])
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(gdf_buildings.crs).iloc[0]
    return (float(pt.x), float(pt.y))


def _building_coords_m(gdf_buildings: gpd.GeoDataFrame) -> np.ndarray:
    cent = gdf_buildings.geometry.centroid
    return np.column_stack([cent.x.to_numpy(dtype=float), cent.y.to_numpy(dtype=float)])


def _nearest_pole_id(gdf_poles_4326: gpd.GeoDataFrame, target_latlon: Tuple[float, float]) -> int:
    """Pole (pole_id) closest to a (lat, lon) target, approx planar km metric."""
    lat_t, lon_t = float(target_latlon[0]), float(target_latlon[1])
    best_pid, best_d = None, np.inf
    for _, r in gdf_poles_4326.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        d = _approx_km_from_latlon(lat_t, lon_t, float(g.y), float(g.x))
        if d < best_d:
            best_d, best_pid = d, int(r["pole_id"])
    if best_pid is None:
        raise ValueError("No valid poles to select a root from.")
    return best_pid


def _topology_result_from_service(out: Dict[str, Any]) -> TopologyResult:
    return TopologyResult(
        schema_version=SESSION_SCHEMA_VERSION,
        metrics=out["metrics"],
        gdf_buildings_4326=out["gdf_buildings_4326"],
        gdf_poles_4326=out["gdf_poles_4326"],
        gdf_edges_4326=None,
        gdf_roads_4326=out.get("gdf_roads_4326"),
        gdf_served_4326=out["gdf_served_4326"],
        gdf_unserved_4326=out["gdf_unserved_4326"],
        mst_edges_latlon=out["mst_edges_latlon"],
        mst_edges_pole_ids=out["mst_edges_pole_ids"],
        associations_df=out["associations_df"],
        center=out["center"],
    )


def _mv_backbone_mst(
    sites_latlon: Dict[str, Tuple[float, float]],
) -> Tuple[List[Tuple[str, str, str, float]], gpd.GeoDataFrame, float]:
    """
    Minimum spanning tree over MV sites (plant + transformer positions).

    Returns (edges for MvLayerSpec, GeoDataFrame of LineStrings in EPSG:4326,
    total length in km).
    """
    names = list(sites_latlon.keys())
    m = len(names)
    if m < 2:
        empty = gpd.GeoDataFrame({"line": [], "length_km": []}, geometry=[], crs="EPSG:4326")
        return [], empty, 0.0

    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            (la1, lo1), (la2, lo2) = sites_latlon[names[i]], sites_latlon[names[j]]
            D[i, j] = _approx_km_from_latlon(la1, lo1, la2, lo2)
    mst = minimum_spanning_tree(D).tocoo()

    edges: List[Tuple[str, str, str, float]] = []
    recs, geoms = [], []
    total = 0.0
    for idx, (i, j, w) in enumerate(zip(mst.row, mst.col, mst.data)):
        u, v = names[int(i)], names[int(j)]
        name = f"MV{idx}"
        L = float(w)
        edges.append((name, u, v, L))
        total += L
        (la1, lo1), (la2, lo2) = sites_latlon[u], sites_latlon[v]
        recs.append({"line": name, "bus0": u, "bus1": v, "length_km": L})
        geoms.append(LineString([(lo1, la1), (lo2, la2)]))
    gdf = gpd.GeoDataFrame(recs, geometry=geoms, crs="EPSG:4326")
    return edges, gdf, total


def classify_standalone(
    gdf_buildings: gpd.GeoDataFrame,
    gdf_roads: Optional[gpd.GeoDataFrame],
    *,
    road_pole_spacing_m: float,
    max_user_connection_radius_m: float,
    max_users_per_pole: int,
    max_pole_span_m: float,
    min_cluster_size: int,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    ONE-SHOT standalone classification on the WHOLE settlement.

    Runs station 1 once with the page-1 coverage policy
    (allow_unserved_isolated=True) and splits the projected buildings into
    (served, standalone) by index. Grid Reinforcement then partitions ONLY
    the served set: a building's standalone status is decided once, before
    the k-iteration, and can never depend on k (same semantics as Page 1).

    Returns (gdf_served, gdf_standalone), both in the CRS of gdf_buildings,
    with the original indices preserved.
    """
    out = run_low_voltage(
        gdf_buildings,
        gdf_roads,
        sampling_distance=float(road_pole_spacing_m),
        user_distance=float(max_user_connection_radius_m),
        max_associations=int(max_users_per_pole),
        allow_unserved_isolated=True,
        min_cluster_size=int(min_cluster_size),
        max_pole_span_m=float(max_pole_span_m),
    )
    unserved_idx = set(out["gdf_unserved_4326"].index.tolist()) if out["gdf_unserved_4326"] is not None else set()
    served_mask = ~gdf_buildings.index.isin(unserved_idx)
    return gdf_buildings.loc[served_mask], gdf_buildings.loc[~served_mask]


# =============================================================================
# Kernel
# =============================================================================
def build_and_validate(
    *,
    gdf_buildings: gpd.GeoDataFrame,
    gdf_roads: Optional[gpd.GeoDataFrame],
    building_meta: pd.DataFrame,
    category_profiles: pd.DataFrame,
    request: MvReinforcementRequest,
    plant_latlon: Tuple[float, float],
    labels: np.ndarray,
    k: int,
    dv_cap_pu: float,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> MvIterationResult:
    """Build k+1 LV subnetworks + MV backbone + transformers, run ONE PF."""
    t0 = time.perf_counter()
    topo = request.topo_params
    trp = request.transformer_params
    mvp = request.mv_line_params
    pfp = request.pf_params

    def _say(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    coords = _building_coords_m(gdf_buildings)
    diams = cluster_diameters_m(coords, labels)

    # ---- station 1 per cluster -------------------------------------------
    subnet_topos: Dict[int, TopologyResult] = {}
    subnet_loads: Dict[int, pd.DataFrame] = {}
    for sid in range(k + 1):
        sub = gdf_buildings.loc[labels == sid]
        if sub.empty:
            raise ValueError(f"Subnet {sid}: empty cluster (k={k}).")
        _say(f"Station 1 — subnetwork {sid}/{k} ({len(sub)} buildings)")
        try:
            out = run_low_voltage(
                sub,
                gdf_roads,
                sampling_distance=float(topo.road_pole_spacing_m),
                user_distance=float(topo.max_user_connection_radius_m),
                max_associations=int(topo.max_users_per_pole),
                allow_unserved_isolated=False,
                min_cluster_size=1,
                max_pole_span_m=float(topo.max_pole_span_m),
            )
        except Exception as e:
            raise RuntimeError(f"Station 1 failed on subnetwork {sid} (k={k}): {e}") from e
        subnet_topos[sid] = _topology_result_from_service(out)
        subnet_loads[sid] = aggregate_pole_loads(
            out["associations_df"], building_meta, category_profiles
        )

    # ---- hour selection ----------------------------------------------------
    total_by_hour = None
    for df in subnet_loads.values():
        s = df.sum(axis=1)
        total_by_hour = s if total_by_hour is None else total_by_hour.add(s, fill_value=0.0)
    hour = int(pfp.selected_hour) if pfp.selected_hour is not None else int(total_by_hour.idxmax())

    # ---- roots, transformer sizing, MV sites -------------------------------
    subnet_specs: List[LvSubnetSpec] = []
    tr_specs: List[MvTransformerSpec] = []
    sites_latlon: Dict[str, Tuple[float, float]] = {}
    peaks_kw: Dict[int, float] = {}
    roots: Dict[int, int] = {}

    for sid in range(k + 1):
        tr_topo = subnet_topos[sid]
        loads = subnet_loads[sid]
        peaks_kw[sid] = float(loads.sum(axis=1).max()) if len(loads) else 0.0

        if sid == 0:
            target = plant_latlon
        else:
            c = tr_topo.gdf_buildings_4326.geometry.union_all().centroid
            target = (float(c.y), float(c.x))
        root = _nearest_pole_id(tr_topo.gdf_poles_4326, target)
        roots[sid] = root

        if hour in loads.index:
            pole_p_kw = {int(p): float(v) for p, v in loads.loc[hour].items() if float(v) > 0.0}
        else:
            pole_p_kw = {}

        bundle = PFTopologyBundle(
            gdf_nodes_4326=tr_topo.gdf_poles_4326,
            pole_id_col="pole_id",
            mst_edges_pole_ids=tr_topo.mst_edges_pole_ids,
        )
        subnet_specs.append(
            LvSubnetSpec(
                subnet_id=sid,
                bundle=bundle,
                root_pole_id=root,
                pole_p_kw=pole_p_kw,
                line_params_df=None,  # cable catalog applied via global PF params
            )
        )

        root_geom = tr_topo.gdf_poles_4326.set_index("pole_id").loc[root].geometry
        if k > 0:
            if sid == 0:
                sites_latlon["mv_plant"] = (float(root_geom.y), float(root_geom.x))
            else:
                sites_latlon[f"mv_tr{sid}"] = (float(root_geom.y), float(root_geom.x))

    # transformers (only in hybrid mode, k > 0):
    # step-up on system peak, step-downs on cluster peaks
    if k > 0:
        total_peak_kw = float(sum(peaks_kw.values()))
        stepup_kva = (
            float(trp.s_nom_kva)
            if trp.s_nom_kva is not None
            else auto_size_transformer_kva(total_peak_kw, pfp.pf_load, trp.sizing_margin, trp.standard_sizes_kva)
        )
        tr_specs.append(
            MvTransformerSpec(
                name="tr_stepup",
                mv_bus="mv_plant",
                lv_bus=HybridPyPSAPowerFlowRunner.lv_bus_name(0, roots[0]),
                s_nom_kva=stepup_kva,
                vsc_pct=float(trp.vsc_pct),
                vscr_pct=float(trp.vscr_pct),
                tap_ratio=float(trp.tap_ratio),
            )
        )
    for sid in range(1, k + 1):
        s_kva = (
            float(trp.s_nom_kva)
            if trp.s_nom_kva is not None
            else auto_size_transformer_kva(peaks_kw[sid], pfp.pf_load, trp.sizing_margin, trp.standard_sizes_kva)
        )
        tr_specs.append(
            MvTransformerSpec(
                name=f"tr_{sid}",
                mv_bus=f"mv_tr{sid}",
                lv_bus=HybridPyPSAPowerFlowRunner.lv_bus_name(sid, roots[sid]),
                s_nom_kva=s_kva,
                vsc_pct=float(trp.vsc_pct),
                vscr_pct=float(trp.vscr_pct),
                tap_ratio=float(trp.tap_ratio),
            )
        )

    mv_edges, mv_gdf, mv_len_km = _mv_backbone_mst(sites_latlon)
    mv_layer = MvLayerSpec(
        v_nom_kv=float(trp.mv_v_nom_kv),
        nodes=sites_latlon,
        edges=mv_edges,
        r_ohm_per_km=float(mvp.r_ohm_per_km),
        x_ohm_per_km=float(mvp.x_ohm_per_km),
        i_max_a=float(mvp.i_max_a),
    )

    # ---- station 2: one hybrid PF ------------------------------------------
    _say(f"Station 2 — hybrid power flow (k = {k})")
    params = PFScenarioParams(
        slack_pole_id=roots[0],
        v_min_pu=float(pfp.v_min_pu),
        v_max_pu=float(pfp.v_max_pu),
        pf_load=float(pfp.pf_load),
        v_nom_kv=float(pfp.v_nom_kv),
        r_ohm_per_km=float(pfp.r_ohm_per_km),
        x_ohm_per_km=float(pfp.x_ohm_per_km),
        s_nom_kva=float(pfp.s_nom_kva),
    )
    runner = HybridPyPSAPowerFlowRunner(subnet_specs, mv_layer, tr_specs)
    pf_out = runner.run_snapshot(params=params, debug=True)

    # attach MV power-flow quantities to the backbone GeoDataFrame (map tooltips)
    df_mv_pf = pf_out.get("mv_line_results")
    if mv_gdf is not None and len(mv_gdf) and df_mv_pf is not None and len(df_mv_pf):
        mv_gdf = mv_gdf.merge(
            df_mv_pf[["line", "I_A", "loading_pu", "v0_pu", "v1_pu"]], on="line", how="left"
        )

    # ---- assemble contract objects ------------------------------------------
    df_tr = pf_out["transformer_results"]
    tr_kva_by_sid: Dict[int, float] = {}
    tr_load_by_sid: Dict[int, float] = {}
    if df_tr is not None and len(df_tr):  # empty in pure-LV mode (k = 0)
        for name, row in df_tr.set_index("transformer").iterrows():
            if name.startswith("tr_") and name != "tr_stepup":
                sid_key = int(name.split("_")[1])
                tr_kva_by_sid[sid_key] = float(row["s_nom_kva"])
                tr_load_by_sid[sid_key] = float(row["loading_pct"])

    subnetworks: List[MvSubnetworkResult] = []
    all_ok = True
    for sid in range(k + 1):
        res = pf_out["subnet_results"][sid]
        worst = float(res["summary"]["worst_dv_pu"])
        ok = bool(worst <= float(dv_cap_pu))
        all_ok = all_ok and ok
        validation = ValidationResult(
            schema_version=SESSION_SCHEMA_VERSION,
            hour=hour,
            params=params.__dict__.copy(),
            summary=res["summary"],
            bus_results=res["bus_results"],
            line_results=res["line_results"],
            debug=None,
        )
        subnetworks.append(
            MvSubnetworkResult(
                subnet_id=sid,
                root_kind=("slack" if sid == 0 else "transformer"),
                topology=subnet_topos[sid],
                peak_load_kw=peaks_kw[sid],
                validation=validation,
                tr_s_nom_kva=(None if sid == 0 else tr_kva_by_sid.get(sid)),
                tr_loading_pct=(None if sid == 0 else tr_load_by_sid.get(sid)),
                worst_dv_pu=worst,
                dv_cap_ok=ok,
            )
        )

    return MvIterationResult(
        n_transformers=k,
        subnetworks=subnetworks,
        mv_backbone_edges_4326=mv_gdf,
        mv_backbone_length_km=float(mv_len_km),
        converged=bool(all_ok),
        pf_executed=True,
        solve_seconds=float(time.perf_counter() - t0),
        transformer_summary=df_tr,
        cluster_diameters_m=diams,
    )


# =============================================================================
# Public entry: the two criterion loops
# =============================================================================
def run_grid_reinforcement(
    *,
    gdf_buildings: gpd.GeoDataFrame,
    gdf_roads: Optional[gpd.GeoDataFrame],
    building_meta: pd.DataFrame,
    category_profiles: pd.DataFrame,
    request: MvReinforcementRequest,
    plant_latlon: Tuple[float, float],
    dv_cap_pu: float = 0.10,
    seed: int = 42,
    skip_k0: bool = False,
    seed_diameter_m: float = 2000.0,
    gdf_standalone_4326: Optional[gpd.GeoDataFrame] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> MvReinforcementResult:
    """
    Grid Reinforcement entry point.

    Voltage-cap branch (three-way start, decided by the caller/UI):
    - Grid Validation ran and is feasible -> the UI does not call this at all;
    - Grid Validation ran and is infeasible -> skip_k0=True: start from
      max(1, seed) where seed = complete-linkage cluster count at
      `seed_diameter_m` minus 1 (clusters wider than ~2 km cannot respect the
      cap at village-scale loads);
    - Grid Validation not run -> skip_k0=False: iteration 0 is the pure-LV
      network; if it meets the cap the result has 0 transformers.
    Newton-Raphson divergence is treated as "this k is not enough" (iteration
    recorded with a note, loop continues), and when the first attempted k
    converges the loop BACKTRACKS to k-1, k-2, ... so the returned k is
    always the minimum that respects the cap regardless of the seed.

    Distance-cap branch: one-shot complete-linkage at the diameter cap
    (diameter guarantee by construction, minimal k, no iteration), followed
    by a single topology + power-flow build.
    """
    if gdf_buildings is None or gdf_buildings.empty:
        raise ValueError("gdf_buildings is empty.")
    if getattr(gdf_buildings.crs, "is_geographic", False):
        raise ValueError("gdf_buildings must be in a projected CRS (meters).")

    t0 = time.perf_counter()
    topo = request.topo_params
    criterion = str(topo.clustering_criterion)
    max_k = int(topo.max_transformers)
    coords = _building_coords_m(gdf_buildings)
    plant_xy = _plant_xy_m(gdf_buildings, plant_latlon)

    def _attempt(k: int) -> MvIterationResult:
        """One k-attempt; PF divergence/failures become a not-converged iteration."""
        t_it = time.perf_counter()
        if k == 0:
            labels = np.zeros(coords.shape[0], dtype=int)
        else:
            labels = partition_buildings(coords, plant_xy, k, seed=seed)
        try:
            return build_and_validate(
                gdf_buildings=gdf_buildings, gdf_roads=gdf_roads,
                building_meta=building_meta, category_profiles=category_profiles,
                request=request, plant_latlon=plant_latlon,
                labels=labels, k=k, dv_cap_pu=dv_cap_pu, progress_cb=progress_cb,
            )
        except RuntimeError as e:
            return MvIterationResult(
                n_transformers=k,
                subnetworks=[],
                mv_backbone_edges_4326=None,
                mv_backbone_length_km=0.0,
                converged=False,
                pf_executed=True,
                solve_seconds=float(time.perf_counter() - t_it),
                transformer_summary=None,
                cluster_diameters_m=cluster_diameters_m(coords, labels),
                note=(f"diverged/failed: {e}")[:300],
            )

    iterations: List[MvIterationResult] = []
    final: Optional[MvIterationResult] = None

    if criterion == "distance_cap":
        cap_m = float(topo.max_cluster_diameter_m)
        labels = partition_buildings_complete_linkage(coords, plant_xy, cap_m)
        k = int(labels.max())
        if k > max_k:
            raise ValueError(
                f"The {cap_m:.0f} m diameter cap requires k = {k} transformers "
                f"(complete-linkage), above the safety cap of {max_k}. "
                f"Raise 'Max transformers' or relax the diameter cap."
            )
        final = build_and_validate(
            gdf_buildings=gdf_buildings, gdf_roads=gdf_roads,
            building_meta=building_meta, category_profiles=category_profiles,
            request=request, plant_latlon=plant_latlon,
            labels=labels, k=k, dv_cap_pu=dv_cap_pu, progress_cb=progress_cb,
        )
        final.converged = True  # diameter criterion satisfied by construction
        final.note = "complete-linkage one-shot (diameter cap guaranteed by construction)"
        iterations = [final]

    elif criterion == "voltage_cap":
        if not skip_k0:
            it0 = _attempt(0)
            iterations.append(it0)
            if it0.converged:
                it0.note = "pure LV meets the voltage cap: no MV layer needed"
                final = it0

        if final is None:
            k_start = max(1, min(suggest_k_start(coords, seed_diameter_m), max_k))
            it = _attempt(k_start)
            iterations.append(it)
            if it.converged:
                final = it
                kk = k_start - 1
                while kk >= 1:  # backtrack: guarantee the minimum k
                    it2 = _attempt(kk)
                    iterations.append(it2)
                    if it2.converged:
                        final = it2
                        kk -= 1
                    else:
                        break
            else:
                final = it
                kk = k_start + 1
                while kk <= max_k:
                    it = _attempt(kk)
                    iterations.append(it)
                    final = it
                    if it.converged:
                        break
                    kk += 1
    else:
        raise ValueError(f"Unknown clustering_criterion: {criterion!r}")

    assert final is not None
    return MvReinforcementResult(
        schema_version=SESSION_SCHEMA_VERSION,
        criterion=criterion,
        iterations=iterations,
        final=final,
        total_solve_seconds=float(time.perf_counter() - t0),
        gdf_standalone_4326=gdf_standalone_4326,
    )
