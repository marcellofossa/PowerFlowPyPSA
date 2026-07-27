"""
Unit tests for core/partitioning.py.

Oracle: a synthetic settlement built by design — three compact clumps of
buildings (~100 m wide) whose centers are ~2 km apart, slack in clump A.
With a 1 km distance cap the outcome is forced by construction:
- k = 1 (2 clusters over 3 clumps): at least one cluster must contain two
  clumps -> diameter ~2 km > 1 km.
- k = 2 (3 clusters): each clump on its own -> all diameters ~100 m <= 1 km.
"""

import numpy as np
import pytest

from core.partitioning import (
    cluster_diameters_m,
    cluster_sizes,
    partition_buildings,
    partition_buildings_complete_linkage,
    suggest_k_start,
)


def _clump(cx: float, cy: float, n: int, spread_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(cx - spread_m / 2, cx + spread_m / 2, n),
         rng.uniform(cy - spread_m / 2, cy + spread_m / 2, n)]
    )


@pytest.fixture()
def three_clumps():
    a = _clump(0.0, 0.0, 40, 100.0, seed=1)       # clump A (slack here)
    b = _clump(2000.0, 0.0, 35, 100.0, seed=2)    # clump B, 2 km east
    c = _clump(1000.0, 1800.0, 30, 100.0, seed=3)  # clump C, ~2 km from both
    coords = np.vstack([a, b, c])
    slack_xy = (0.0, 0.0)
    return coords, slack_xy, (len(a), len(b), len(c))


def test_labels_shape_and_range(three_clumps):
    coords, slack_xy, _ = three_clumps
    labels = partition_buildings(coords, slack_xy, k=2, seed=42)
    assert labels.shape == (coords.shape[0],)
    assert set(np.unique(labels)) == {0, 1, 2}


def test_k0_single_cluster(three_clumps):
    coords, slack_xy, _ = three_clumps
    labels = partition_buildings(coords, slack_xy, k=0, seed=42)
    assert set(np.unique(labels)) == {0}


def test_k1_violates_1km_cap(three_clumps):
    coords, slack_xy, _ = three_clumps
    labels = partition_buildings(coords, slack_xy, k=1, seed=42)
    diams = cluster_diameters_m(coords, labels)
    assert len(diams) == 2
    assert max(diams.values()) > 1000.0  # some cluster spans two clumps


def test_k2_respects_1km_cap_and_recovers_clumps(three_clumps):
    coords, slack_xy, (na, nb, nc) = three_clumps
    labels = partition_buildings(coords, slack_xy, k=2, seed=42)
    diams = cluster_diameters_m(coords, labels)
    assert len(diams) == 3
    assert max(diams.values()) <= 1000.0  # each clump alone (~100 m wide)
    sizes = sorted(cluster_sizes(labels).values())
    assert sizes == sorted([na, nb, nc])


def test_slack_cluster_is_label_0(three_clumps):
    coords, slack_xy, (na, _, _) = three_clumps
    labels = partition_buildings(coords, slack_xy, k=2, seed=42)
    # buildings of clump A (first na rows) must carry label 0
    assert set(labels[:na]) == {0}


def test_determinism(three_clumps):
    coords, slack_xy, _ = three_clumps
    l1 = partition_buildings(coords, slack_xy, k=2, seed=42)
    l2 = partition_buildings(coords, slack_xy, k=2, seed=42)
    assert np.array_equal(l1, l2)


def test_diameter_exact_value():
    # hand-checkable oracle: 3-4-5 triangle -> diameter 5
    pts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    labels = np.zeros(3, dtype=int)
    diams = cluster_diameters_m(pts, labels)
    assert diams[0] == pytest.approx(5.0)


def test_diameter_hull_path_matches_pdist():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 500, size=(800, 2))  # > _HULL_MIN_POINTS -> hull path
    labels = np.zeros(len(pts), dtype=int)
    from scipy.spatial.distance import pdist

    diams = cluster_diameters_m(pts, labels)
    assert diams[0] == pytest.approx(float(pdist(pts).max()))


def test_too_many_clusters_raises():
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    with pytest.raises(ValueError):
        partition_buildings(pts, (0.0, 0.0), k=5)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        partition_buildings(np.zeros((4, 3)), (0.0, 0.0), k=1)


def test_complete_linkage_three_clumps(three_clumps):
    coords, slack_xy, (na, nb, nc) = three_clumps
    labels = partition_buildings_complete_linkage(coords, slack_xy, max_diameter_m=1000.0)
    assert int(labels.max()) == 2  # 3 clusters -> k = 2, emerging from the cut
    diams = cluster_diameters_m(coords, labels)
    assert max(diams.values()) <= 1000.0  # guaranteed by construction
    assert set(labels[:na]) == {0}        # slack clump -> label 0
    sizes = sorted(cluster_sizes(labels).values())
    assert sizes == sorted([na, nb, nc])


def test_complete_linkage_single_compact_settlement():
    pts = _clump(0.0, 0.0, 30, 300.0, seed=9)
    labels = partition_buildings_complete_linkage(pts, (0.0, 0.0), max_diameter_m=1000.0)
    assert set(np.unique(labels)) == {0}  # one cluster -> k = 0, pure LV


def test_complete_linkage_deterministic(three_clumps):
    coords, slack_xy, _ = three_clumps
    l1 = partition_buildings_complete_linkage(coords, slack_xy, 1000.0)
    l2 = partition_buildings_complete_linkage(coords, slack_xy, 1000.0)
    assert np.array_equal(l1, l2)


def test_suggest_k_start(three_clumps):
    coords, _, _ = three_clumps
    # clumps mutually ~2 km apart with ~100 m spread: max inter-clump pair
    # distance > 2000 m -> no merge at the threshold -> 3 clusters -> k = 2
    assert suggest_k_start(coords, hopeless_diameter_m=2000.0) == 2
    # a single compact clump is never doomed by geometry
    assert suggest_k_start(_clump(0, 0, 20, 100.0, seed=4), 2000.0) == 0


def test_complete_linkage_bad_inputs():
    with pytest.raises(ValueError):
        partition_buildings_complete_linkage(np.zeros((4, 3)), (0, 0), 1000.0)
    with pytest.raises(ValueError):
        partition_buildings_complete_linkage(np.zeros((0, 2)), (0, 0), 1000.0)
    with pytest.raises(ValueError):
        partition_buildings_complete_linkage(np.zeros((4, 2)), (0, 0), -5.0)
