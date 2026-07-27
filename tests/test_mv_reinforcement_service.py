"""
Integration tests for core/mv_reinforcement_service.py.

Geometry-designed oracles (projected EPSG:32633 meters, loads chosen so the
outcome is forced by construction):

- Two clumps ~1.5 km apart, light loads: with skip_k0=False the pure-LV
  iteration (k = 0) already meets the cap (~4.8% drop on the 1.5 km run)
  -> 0 transformers; with skip_k0=True the loop starts at max(1, seed) = 1
  and stops there (backtracking floor).
- Three clumps mutually ~2 km apart, 15 kW on the remote ones: pure LV and
  k = 1 both leave ~2 km LV runs (dV ~ 12% > 10%); the seed (complete
  linkage @2 km) starts the loop at k = 2, which converges, and the
  backtracking to k = 1 fails -> minimum k = 2 is guaranteed.
- Distance cap: complete-linkage one-shot at 1 km -> k = 2 by construction,
  single iteration, then one topology + PF build.
- Divergence handling: a monkeypatched build_and_validate raising
  RuntimeError for low k must yield "diverged" iterations and a converged
  final at the first healthy k.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from core.contracts import (
    MvLineParams,
    MvReinforcementRequest,
    MvTopologyParams,
    MvTransformerParams,
    ValidationInputs,
)
from core.mv_reinforcement_service import (
    auto_size_transformer_kva,
    run_grid_reinforcement,
)

CRS_M = 32633
X0, Y0 = 300_000.0, 1_000_000.0


def _clump_gdf(cx, cy, n, spread_m, seed):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(cx - spread_m / 2, cx + spread_m / 2, n)
    ys = rng.uniform(cy - spread_m / 2, cy + spread_m / 2, n)
    return [Point(x, y) for x, y in zip(xs, ys)]


def _buildings(clumps):
    """clumps: list of (cx, cy, n, spread, seed) -> gdf with global integer index."""
    pts = []
    for (cx, cy, n, spread, seed) in clumps:
        pts.extend(_clump_gdf(cx, cy, n, spread, seed))
    return gpd.GeoDataFrame({"geometry": pts}, crs=f"EPSG:{CRS_M}")


def _demand(gdf, w_per_building=500.0):
    meta = pd.DataFrame(
        {"building_id": [str(i) for i in gdf.index], "category": "hh", "weight": 1.0}
    )
    profiles = pd.DataFrame({"hh": [w_per_building] * 24}, index=pd.RangeIndex(24, name="hour"))
    return meta, profiles


def _plant_latlon(gdf, x, y):
    pt = gpd.GeoSeries([Point(x, y)], crs=f"EPSG:{CRS_M}").to_crs(epsg=4326).iloc[0]
    return (float(pt.y), float(pt.x))


def _request(criterion, max_transformers=6):
    pf = ValidationInputs(
        schema_version=1,
        mode="mgp",
        gdf_nodes_4326=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
        associations_df=pd.DataFrame(),
        pole_id_col="pole_id",
        center=(0.0, 0.0),
        selected_hour=12,
        r_ohm_per_km=0.641,
        x_ohm_per_km=0.083,
        s_nom_kva=114.0,
    )
    return MvReinforcementRequest(
        schema_version=1,
        topo_params=MvTopologyParams(
            follow_roads_mode="none",
            road_pole_spacing_m=40.0,
            max_user_connection_radius_m=30.0,
            max_users_per_pole=8,
            max_pole_span_m=50.0,
            clustering_criterion=criterion,
            max_cluster_diameter_m=1000.0,
            max_transformers=max_transformers,
        ),
        transformer_params=MvTransformerParams(),
        mv_line_params=MvLineParams(),
        pf_params=pf,
    )


def test_auto_size_transformer():
    sizes = (25.0, 50.0, 100.0, 200.0, 315.0, 500.0)
    # 30 kW / 0.95 * 1.25 = 39.5 kVA -> 50 kVA
    assert auto_size_transformer_kva(30.0, 0.95, 1.25, sizes) == 50.0
    # 600 kW -> 789 kVA, beyond the largest standard -> next 50 kVA multiple
    assert auto_size_transformer_kva(600.0, 0.95, 1.25, sizes) == 800.0
    assert auto_size_transformer_kva(0.0, 0.95, 1.25, sizes) == 25.0
    with pytest.raises(ValueError):
        auto_size_transformer_kva(-1.0, 0.95, 1.25, sizes)


def _two_clumps():
    gdf = _buildings([
        (X0, Y0, 20, 100.0, 1),                 # clump A (plant)
        (X0 + 1500.0, Y0, 20, 100.0, 2),        # clump B
    ])
    meta, profiles = _demand(gdf, w_per_building=400.0)  # 8 kW per clump
    return gdf, meta, profiles


def test_two_clumps_pure_lv_suffices_k0():
    gdf, meta, profiles = _two_clumps()
    res = run_grid_reinforcement(
        gdf_buildings=gdf, gdf_roads=None,
        building_meta=meta, category_profiles=profiles,
        request=_request("voltage_cap"), plant_latlon=_plant_latlon(gdf, X0, Y0),
        skip_k0=False,
    )
    # pure LV (~4.8% drop) meets the cap: no MV layer at all
    assert res.final.n_transformers == 0
    assert len(res.iterations) == 1
    assert res.final.converged and res.final.pf_executed
    assert len(res.final.subnetworks) == 1
    assert "no MV layer needed" in (res.final.note or "")


def test_two_clumps_skip_k0_starts_and_stays_at_k1():
    gdf, meta, profiles = _two_clumps()
    res = run_grid_reinforcement(
        gdf_buildings=gdf, gdf_roads=None,
        building_meta=meta, category_profiles=profiles,
        request=_request("voltage_cap"), plant_latlon=_plant_latlon(gdf, X0, Y0),
        skip_k0=True,
    )
    # seed = 0 (compact @2 km) -> start at max(1, 0) = 1; backtracking floor is 1
    assert res.final.n_transformers == 1
    assert res.final.converged
    assert len(res.final.subnetworks) == 2
    assert all(it.n_transformers >= 1 for it in res.iterations)


def test_three_clumps_voltage_cap_needs_k2():
    gdf = _buildings([
        (X0, Y0, 20, 100.0, 1),                       # A (plant), 10 kW
        (X0 + 2000.0, Y0, 30, 100.0, 2),              # B, 15 kW
        (X0 + 1000.0, Y0 + 1800.0, 30, 100.0, 3),     # C, 15 kW
    ])
    meta, profiles = _demand(gdf, w_per_building=500.0)
    res = run_grid_reinforcement(
        gdf_buildings=gdf, gdf_roads=None,
        building_meta=meta, category_profiles=profiles,
        request=_request("voltage_cap"), plant_latlon=_plant_latlon(gdf, X0, Y0),
        skip_k0=True,
    )
    assert res.criterion == "voltage_cap"
    # seed (CL @2 km) = 2 -> k=2 converges -> backtrack to k=1 fails -> min k = 2
    ks = [it.n_transformers for it in res.iterations]
    assert ks == [2, 1]
    assert res.final.n_transformers == 2
    assert res.final.converged
    assert not res.iterations[-1].converged  # the k=1 backtrack attempt
    assert len(res.final.subnetworks) == 3
    for s in res.final.subnetworks:
        assert s.validation is not None
        assert s.worst_dv_pu is not None and s.dv_cap_ok
    assert set(res.final.cluster_diameters_m.keys()) == {0, 1, 2}


def test_three_clumps_voltage_cap_with_k0_first():
    gdf = _buildings([
        (X0, Y0, 20, 100.0, 1),
        (X0 + 2000.0, Y0, 30, 100.0, 2),
        (X0 + 1000.0, Y0 + 1800.0, 30, 100.0, 3),
    ])
    meta, profiles = _demand(gdf, w_per_building=500.0)
    res = run_grid_reinforcement(
        gdf_buildings=gdf, gdf_roads=None,
        building_meta=meta, category_profiles=profiles,
        request=_request("voltage_cap"), plant_latlon=_plant_latlon(gdf, X0, Y0),
        skip_k0=False,
    )
    ks = [it.n_transformers for it in res.iterations]
    assert ks[0] == 0 and not res.iterations[0].converged  # pure LV fails (~12%)
    assert res.final.n_transformers == 2 and res.final.converged


def test_three_clumps_distance_cap():
    gdf = _buildings([
        (X0, Y0, 20, 100.0, 1),
        (X0 + 2000.0, Y0, 30, 100.0, 2),
        (X0 + 1000.0, Y0 + 1800.0, 30, 100.0, 3),
    ])
    meta, profiles = _demand(gdf, w_per_building=500.0)
    res = run_grid_reinforcement(
        gdf_buildings=gdf, gdf_roads=None,
        building_meta=meta, category_profiles=profiles,
        request=_request("distance_cap"), plant_latlon=_plant_latlon(gdf, X0, Y0),
    )
    assert res.criterion == "distance_cap"
    # complete-linkage one-shot: single iteration, k emerges from the cut
    assert len(res.iterations) == 1 and res.iterations[0] is res.final
    assert res.final.pf_executed
    assert max(res.final.cluster_diameters_m.values()) <= 1000.0
    assert res.final.n_transformers == 2
    assert res.final.converged and res.final.pf_executed
    assert len(res.final.subnetworks) == 3
    # transformer summary: step-up + 2 step-downs; MV backbone spans the clumps
    assert len(res.final.transformer_summary) == 3
    assert 2.0 < res.final.mv_backbone_length_km < 8.0
    # slack subnet has no transformer attributes
    s0 = [s for s in res.final.subnetworks if s.subnet_id == 0][0]
    assert s0.root_kind == "slack" and s0.tr_s_nom_kva is None
    s1 = [s for s in res.final.subnetworks if s.subnet_id == 1][0]
    assert s1.tr_s_nom_kva in (25.0, 50.0)  # ~15 kW peak -> small standard size


def test_unknown_criterion_raises():
    gdf = _buildings([(X0, Y0, 10, 80.0, 1)])
    meta, profiles = _demand(gdf)
    req = _request("voltage_cap")
    req.topo_params.clustering_criterion = "nope"
    with pytest.raises(ValueError):
        run_grid_reinforcement(
            gdf_buildings=gdf, gdf_roads=None,
            building_meta=meta, category_profiles=profiles,
            request=req, plant_latlon=_plant_latlon(gdf, X0, Y0),
        )


def test_distance_cap_exceeding_max_transformers_raises():
    gdf = _buildings([
        (X0, Y0, 20, 100.0, 1),
        (X0 + 2000.0, Y0, 30, 100.0, 2),
        (X0 + 1000.0, Y0 + 1800.0, 30, 100.0, 3),
    ])
    meta, profiles = _demand(gdf)
    req = _request("distance_cap", max_transformers=1)  # CL needs k = 2 > 1
    with pytest.raises(ValueError):
        run_grid_reinforcement(
            gdf_buildings=gdf, gdf_roads=None,
            building_meta=meta, category_profiles=profiles,
            request=req, plant_latlon=_plant_latlon(gdf, X0, Y0),
        )


def test_voltage_loop_treats_divergence_as_k_plus_1(monkeypatch):
    """A build that blows up at low k must be recorded as a diverged
    iteration and the loop must continue upward to the first healthy k."""
    import core.mv_reinforcement_service as svc

    gdf, meta, profiles = _two_clumps()
    real = svc.build_and_validate

    def flaky(**kwargs):
        if kwargs["k"] < 2:
            raise RuntimeError("Hybrid PF returned non-physical voltages (test)")
        return real(**kwargs)

    monkeypatch.setattr(svc, "build_and_validate", flaky)
    res = svc.run_grid_reinforcement(
        gdf_buildings=gdf, gdf_roads=None,
        building_meta=meta, category_profiles=profiles,
        request=_request("voltage_cap"), plant_latlon=_plant_latlon(gdf, X0, Y0),
        skip_k0=True,
    )
    ks = [it.n_transformers for it in res.iterations]
    assert ks == [1, 2]
    assert (res.iterations[0].note or "").startswith("diverged")
    assert not res.iterations[0].converged and res.iterations[0].pf_executed
    assert res.final.n_transformers == 2 and res.final.converged


def test_classify_standalone_isolates_far_building():
    """A lone building ~3 km from everything must be classified standalone by
    the ONE-SHOT pre-pass (page-1 semantics), leaving the served set intact
    with original indices preserved."""
    from core.mv_reinforcement_service import classify_standalone

    gdf = _buildings([
        (X0, Y0, 20, 100.0, 1),
        (X0 + 1500.0, Y0, 20, 100.0, 2),
    ])
    # append one isolated building far away, custom index preserved by concat
    import geopandas as gpd
    from shapely.geometry import Point
    lone = gpd.GeoDataFrame({"geometry": [Point(X0 + 3000.0, Y0 + 3000.0)]},
                            crs=gdf.crs, index=[9999])
    gdf_all = gpd.GeoDataFrame(
        __import__("pandas").concat([gdf, lone]), crs=gdf.crs
    )
    served, standalone = classify_standalone(
        gdf_all, None,
        road_pole_spacing_m=40.0, max_user_connection_radius_m=35.0,
        max_users_per_pole=16, max_pole_span_m=40.0, min_cluster_size=2,
    )
    assert 9999 in standalone.index and 9999 not in served.index
    # served + standalone must partition the input; besides the lone far
    # building, sparse buildings inside the clumps may legitimately end up
    # standalone too (no neighbour within user_distance -> singleton cluster,
    # exactly the page-1 semantics), so no strict count equality here.
    assert set(served.index) | set(standalone.index) == set(gdf_all.index)
    assert set(served.index) & set(standalone.index) == set()
    assert len(standalone) >= 1 and len(served) >= 30
    assert served.crs == gdf_all.crs


def test_result_carries_standalone_set():
    gdf, meta, profiles = _two_clumps()
    import geopandas as gpd
    from shapely.geometry import Point
    fake_standalone = gpd.GeoDataFrame({"geometry": [Point(8.0, 9.0)]}, crs="EPSG:4326")
    res = run_grid_reinforcement(
        gdf_buildings=gdf, gdf_roads=None,
        building_meta=meta, category_profiles=profiles,
        request=_request("voltage_cap"), plant_latlon=_plant_latlon(gdf, X0, Y0),
        skip_k0=False, gdf_standalone_4326=fake_standalone,
    )
    assert res.gdf_standalone_4326 is not None and len(res.gdf_standalone_4326) == 1
