from __future__ import annotations

"""
Spatial partitioning of buildings for Grid Reinforcement (hybrid MV/LV).

This module is pure geometry: it never decides k. The k-iteration lives in
core/mv_reinforcement_service.py, which calls partition_buildings(k) and then
checks the active criterion (cluster diameters for the distance cap, per-subnet
voltage drops for the voltage cap).

Conventions
-----------
- Coordinates are planar meters (projected CRS, as used by station 1).
- k = number of step-down transformers -> the partition has k + 1 clusters.
- Cluster label 0 is, by convention, the slack subnetwork: after k-means the
  labels are remapped so that the cluster whose centroid is closest to the
  slack (plant) position gets label 0. Transformer subnetworks are 1..k.
- Diameter of a cluster = max Euclidean distance between any two of its
  buildings (the distance-cap check quantity).
"""

from typing import Dict, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

_KMEANS_RETRIES = 5
_HULL_MIN_POINTS = 500  # above this size, reduce to hull vertices before pdist


def _slack_first_labels(
    coords: np.ndarray, labels: np.ndarray, slack_xy_m: Tuple[float, float]
) -> np.ndarray:
    """Remap labels so the cluster whose centroid (member mean) is closest
    to the slack becomes label 0; the others keep a stable order."""
    uniq = sorted(int(u) for u in np.unique(labels))
    slack = np.asarray(slack_xy_m, dtype=float).reshape(1, 2)
    cent = {u: coords[labels == u].mean(axis=0) for u in uniq}
    slack_cluster = min(uniq, key=lambda u: float(np.linalg.norm(cent[u] - slack)))
    remap = {slack_cluster: 0}
    nxt = 1
    for u in uniq:
        if u != slack_cluster:
            remap[u] = nxt
            nxt += 1
    return np.array([remap[int(l)] for l in labels], dtype=int)


def partition_buildings(
    coords_m: np.ndarray,
    slack_xy_m: Tuple[float, float],
    k: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Partition buildings into k + 1 spatially compact clusters.

    Parameters
    ----------
    coords_m : (N, 2) array of building coordinates in meters.
    slack_xy_m : (x, y) of the slack / plant position in the same CRS.
    k : number of transformers (k >= 0). k = 0 returns a single cluster
        (pure-LV degenerate case, label 0 everywhere).
    seed : deterministic k-means initialisation.

    Returns
    -------
    labels : (N,) int array with values in {0, ..., k}; label 0 is the
        slack subnetwork (centroid closest to the plant).

    Raises
    ------
    ValueError on invalid inputs or if k-means cannot produce k + 1
    non-empty clusters (e.g. k + 1 > number of distinct locations).
    """
    coords = np.asarray(coords_m, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_m must be (N, 2); got {coords.shape}")
    n = coords.shape[0]
    if n == 0:
        raise ValueError("coords_m is empty")
    if k < 0:
        raise ValueError(f"k must be >= 0; got {k}")

    n_clusters = k + 1
    if n_clusters == 1:
        return np.zeros(n, dtype=int)
    if n_clusters > n:
        raise ValueError(f"k + 1 = {n_clusters} clusters requested but only {n} buildings")

    labels = None
    centroids = None
    for attempt in range(_KMEANS_RETRIES):
        rng_seed = int(seed) + attempt
        np.random.seed(rng_seed)
        try:
            centroids, labels = kmeans2(coords, n_clusters, minit="++", seed=rng_seed)
        except Exception:
            labels = None
            continue
        if len(np.unique(labels)) == n_clusters:
            break
        labels = None

    if labels is None or centroids is None:
        raise ValueError(
            f"k-means failed to produce {n_clusters} non-empty clusters after "
            f"{_KMEANS_RETRIES} attempts (n = {n}). The settlement may have too few "
            f"distinct building locations for this k."
        )

    # Label 0 = slack subnetwork (cluster centroid closest to the plant).
    return _slack_first_labels(coords, labels, slack_xy_m)


def cluster_diameters_m(coords_m: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    """
    Diameter (max intra-cluster pairwise distance, meters) of each cluster.

    For large clusters the point set is first reduced to its convex-hull
    vertices (the diameter of a planar point set is attained by hull points),
    with a plain pdist fallback for degenerate geometries.
    """
    coords = np.asarray(coords_m, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if coords.shape[0] != labels.shape[0]:
        raise ValueError("coords_m and labels must have the same length")

    out: Dict[int, float] = {}
    for lab in np.unique(labels):
        pts = coords[labels == lab]
        if pts.shape[0] < 2:
            out[int(lab)] = 0.0
            continue
        if pts.shape[0] >= _HULL_MIN_POINTS:
            try:
                pts = pts[ConvexHull(pts).vertices]
            except QhullError:
                pass  # collinear/degenerate: fall back to full pdist
        out[int(lab)] = float(pdist(pts).max())
    return out


def cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """Number of buildings per cluster label."""
    labels = np.asarray(labels, dtype=int)
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}


def partition_buildings_complete_linkage(
    coords_m: np.ndarray,
    slack_xy_m: Tuple[float, float],
    max_diameter_m: float,
) -> np.ndarray:
    """
    One-shot partition with a hard diameter guarantee (distance-cap branch).

    Complete-linkage hierarchical clustering merges, at every step, the two
    clusters whose merge yields the smallest possible diameter; cutting the
    dendrogram at `max_diameter_m` therefore returns clusters whose diameter
    is <= the threshold BY CONSTRUCTION, with the number of clusters emerging
    from the cut (no k to guess, no iteration, deterministic).

    Returns labels in {0, ..., k}; label 0 is the slack subnetwork.
    """
    coords = np.asarray(coords_m, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords_m must be (N, 2); got {coords.shape}")
    n = coords.shape[0]
    if n == 0:
        raise ValueError("coords_m is empty")
    if max_diameter_m <= 0:
        raise ValueError(f"max_diameter_m must be > 0; got {max_diameter_m}")
    if n == 1:
        return np.zeros(1, dtype=int)

    Z = linkage(coords, method="complete")
    raw = fcluster(Z, t=float(max_diameter_m), criterion="distance") - 1
    return _slack_first_labels(coords, raw.astype(int), slack_xy_m)


def suggest_k_start(coords_m: np.ndarray, hopeless_diameter_m: float = 2000.0) -> int:
    """
    Physically-motivated starting k for the voltage-cap iteration.

    A cluster of diameter D implies LV feeders of order D; above ~2 km a
    feeder cannot stay within a 10% drop at village-scale loads, so any k
    whose partition necessarily contains such clusters cannot converge.
    Returns (number of complete-linkage clusters at the threshold) - 1,
    i.e. the smallest k not doomed by geometry. The voltage loop still
    backtracks below this seed when the first attempt converges, so the
    returned k never prevents finding the true minimum.
    """
    coords = np.asarray(coords_m, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        raise ValueError(f"coords_m must be a non-empty (N, 2) array; got {coords.shape}")
    if coords.shape[0] == 1:
        return 0
    Z = linkage(coords, method="complete")
    ncl = len(np.unique(fcluster(Z, t=float(hopeless_diameter_m), criterion="distance")))
    return max(0, int(ncl) - 1)
