import geopandas as gpd
import pytest
from shapely.geometry import Point

from core.costs import (
    DistributionUnitCosts,
    StandaloneEconomics,
    build_standalone_gate,
)
from core.distribution_algos import place_poles_for_unassociated_buildings

CRS = "EPSG:32633"


def _gdf(points):
    return gpd.GeoDataFrame({"geometry": [Point(*p) for p in points]}, crs=CRS)


def _gate(**eco_kwargs):
    eco = StandaloneEconomics(**eco_kwargs)
    return build_standalone_gate(eco, DistributionUnitCosts(), max_pole_span_m=40.0)


def test_far_single_building_goes_standalone():
    """One building 800 m from the network: extension cost >> threshold."""
    existing = _gdf([(0.0, 0.0)])
    buildings = _gdf([(800.0, 0.0)])
    gate = _gate()  # threshold 1872 $; 800 m extension alone > 7000 $

    poles, assoc, remaining = place_poles_for_unassociated_buildings(
        buildings, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, standalone_gate=gate,
        gdf_existing_poles=existing,
    )
    assert len(remaining) == 1
    assert len(poles) == 0
    assert len(assoc) == 0


def test_shared_extension_connects_cluster():
    """Five buildings 400 m out: the shared extension passes the gate
    (~7.7 k$ < 5 x 1872 $), while a lone building at the same distance
    fails it (~6.9 k$ > 1872 $). Cost sharing is what flips the decision."""
    existing = _gdf([(0.0, 0.0)])
    pts = [(400.0, float(y)) for y in range(0, 50, 10)]
    cluster = _gdf(pts)
    lone = _gdf([(400.0, 0.0)])
    gate = _gate()

    poles, assoc, remaining = place_poles_for_unassociated_buildings(
        cluster, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, standalone_gate=gate,
        gdf_existing_poles=existing,
    )
    assert len(remaining) == 0
    assert len(poles) == 1
    assert len(assoc) == 5

    _, assoc_lone, rem_lone = place_poles_for_unassociated_buildings(
        lone, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, standalone_gate=gate,
        gdf_existing_poles=existing,
    )
    assert len(rem_lone) == 1
    assert len(assoc_lone) == 0


def test_nearby_building_always_connected():
    """A building 30 m from the network costs ~ fixed last-mile only."""
    existing = _gdf([(0.0, 0.0)])
    buildings = _gdf([(30.0, 0.0)])
    gate = _gate()

    poles, assoc, remaining = place_poles_for_unassociated_buildings(
        buildings, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, standalone_gate=gate,
        gdf_existing_poles=existing,
    )
    assert len(remaining) == 0
    assert len(assoc) == 1


def test_threshold_monotonicity():
    """Raising c_sa (or E) can only move buildings from standalone to grid."""
    existing = _gdf([(0.0, 0.0)])
    buildings = _gdf([(300.0, 0.0)])

    cheap_sa = _gate(standalone_cost_usd_per_kwh=0.45)   # low threshold
    dear_sa = _gate(standalone_cost_usd_per_kwh=2.0)     # high threshold

    _, _, rem_cheap = place_poles_for_unassociated_buildings(
        buildings, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, standalone_gate=cheap_sa,
        gdf_existing_poles=existing,
    )
    _, _, rem_dear = place_poles_for_unassociated_buildings(
        buildings, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, standalone_gate=dear_sa,
        gdf_existing_poles=existing,
    )
    assert len(rem_cheap) >= len(rem_dear)


def test_legacy_criterion_unchanged():
    """standalone_gate=None keeps the historical min_cluster_size behaviour."""
    buildings = _gdf([(0.0, 0.0), (5.0, 0.0), (500.0, 500.0)])

    poles, assoc, remaining = place_poles_for_unassociated_buildings(
        buildings, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, min_cluster_size=2,
    )
    # the pair is served, the isolated building is left once the largest
    # remaining cluster (size 1) falls below min_cluster_size
    assert len(remaining) == 1
    assert len(assoc) == 2


def test_no_gate_full_coverage():
    """allow_unserved_isolated=False serves everything (baseline behaviour)."""
    buildings = _gdf([(0.0, 0.0), (5.0, 0.0), (500.0, 500.0)])

    poles, assoc, remaining = place_poles_for_unassociated_buildings(
        buildings, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=False,
    )
    assert len(remaining) == 0
    assert len(assoc) == 3


def test_economic_gate_first_pole_of_empty_network():
    """With no existing poles the first cluster pays no extension cost."""
    buildings = _gdf([(0.0, 0.0), (5.0, 0.0)])
    gate = _gate()

    poles, assoc, remaining = place_poles_for_unassociated_buildings(
        buildings, user_distance=35.0, max_associations=16,
        allow_unserved_isolated=True, standalone_gate=gate,
        gdf_existing_poles=None,
    )
    assert len(remaining) == 0
    assert len(assoc) == 2
