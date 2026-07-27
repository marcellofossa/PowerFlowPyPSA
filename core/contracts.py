from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import geopandas as gpd
import pandas as pd


SESSION_SCHEMA_VERSION = 1


class DomainEnvelope(TypedDict, total=False):
    version: int


class ProjectDomain(DomainEnvelope, total=False):
    topology_request: Dict[str, Any]
    validation_request: Dict[str, Any]


class TopologyDomain(DomainEnvelope, total=False):
    result: Optional["TopologyResult"]
    solve_seconds: Optional[float]
    inputs: Optional[Dict[str, Any]]   # run-time inputs (follow_roads_mode + params)


class ValidationDomain(DomainEnvelope, total=False):
    inputs: Optional["ValidationInputs"]
    result: Optional["ValidationResult"]
    runner: Any
    topo_fingerprint: Optional[Tuple[Any, ...]]
    building_meta: Optional[pd.DataFrame]
    category_profiles: Optional[pd.DataFrame]


class UIDomain(DomainEnvelope, total=False):
    flags: Dict[str, Any]


class MvReinforcementDomain(DomainEnvelope, total=False):
    request: Optional["MvReinforcementRequest"]
    result: Optional["MvReinforcementResult"]
    inputs_fingerprint: Optional[Tuple[Any, ...]]


@dataclass
class TopologyResult:
    schema_version: int
    metrics: Dict[str, float]
    gdf_buildings_4326: gpd.GeoDataFrame
    gdf_poles_4326: gpd.GeoDataFrame
    gdf_edges_4326: Optional[gpd.GeoDataFrame]
    gdf_roads_4326: Optional[gpd.GeoDataFrame]
    gdf_served_4326: gpd.GeoDataFrame
    gdf_unserved_4326: gpd.GeoDataFrame
    mst_edges_latlon: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    mst_edges_pole_ids: List[Tuple[int, int]]
    associations_df: pd.DataFrame
    center: Tuple[float, float]


@dataclass
class ValidationInputs:
    schema_version: int
    mode: str
    gdf_nodes_4326: gpd.GeoDataFrame
    associations_df: pd.DataFrame
    pole_id_col: str
    center: Tuple[float, float]
    mst_edges_latlon: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    mst_edges_pole_ids: Optional[List[Tuple[int, int]]] = None
    gdf_edges_4326: Optional[gpd.GeoDataFrame] = None
    gdf_roads_4326: Optional[gpd.GeoDataFrame] = None
    edge_u_col: Optional[str] = None
    edge_v_col: Optional[str] = None
    pole_loads_kW: Optional[pd.DataFrame] = None
    selected_hour: Optional[int] = None
    pole_load_dict: Dict[int, float] = field(default_factory=dict)
    scaling_mode: str = "Absolute (fixed over the year)"
    pmax_ref_kW: Optional[float] = None
    year_max_pole_kW: float = 0.0
    slack_pole_id: Optional[int] = None
    v_min_pu: float = 0.90
    v_max_pu: float = 1.10
    pf_load: float = 0.95
    v_nom_kv: float = 0.4
    v_base_mode: str = "3-phase LV (0.4 kV line-to-line)"
    r_ohm_per_km: float = 0.642
    x_ohm_per_km: float = 0.083
    s_nom_kva: float = 100.0
    line_params_mode: str = "global"
    default_line_type: Optional[str] = None
    line_types_df: Optional[pd.DataFrame] = None
    lines_meta_df: Optional[pd.DataFrame] = None
    resolved_line_params_df: Optional[pd.DataFrame] = None


@dataclass
class ValidationResult:
    schema_version: int
    hour: int
    params: Dict[str, Any]
    summary: Dict[str, Any]
    bus_results: pd.DataFrame
    line_results: pd.DataFrame
    debug: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Grid Reinforcement (hybrid MV/LV) contracts
# ---------------------------------------------------------------------------


@dataclass
class MvTopologyParams:
    """Topology-stage inputs for Grid Reinforcement.

    Mirrors the "Heuristic pole placement and customer association" inputs of
    Grid Topology, minus the standalone-user options (no standalone users in
    hybrid MV/LV mode). Adds the partition criterion driving the k-iteration.
    """

    follow_roads_mode: str
    road_pole_spacing_m: float
    max_user_connection_radius_m: float
    max_users_per_pole: int
    max_pole_span_m: float
    clustering_criterion: str = "voltage_cap"  # "distance_cap" | "voltage_cap"
    # Standalone (coverage) policy — applied ONCE on the whole settlement
    # before partitioning (never per-cluster; a building's standalone status
    # must not depend on k):
    allow_unserved_isolated: bool = False
    min_cluster_size: int = 1
    max_cluster_diameter_m: float = 1000.0     # distance-cap check: max intra-cluster distance
    max_transformers: int = 10                 # safety stop for the k-iteration


@dataclass
class MvTransformerParams:
    """Electrical parameters of the MV/LV step-down transformers.

    Applied to PyPSA via direct per-unit attributes (type=""), T model with
    g = b = 0 (no-load branch neglected): r_pu = vscr/100,
    x_pu = sqrt(vsc^2 - vscr^2)/100 on the s_nom base.
    """

    mv_v_nom_kv: float = 11.0
    s_nom_kva: Optional[float] = None          # None = auto-size from cluster peak
    sizing_margin: float = 1.25
    standard_sizes_kva: Tuple[float, ...] = (25.0, 50.0, 100.0, 200.0, 315.0, 500.0)
    vsc_pct: float = 4.0                       # short-circuit voltage uk [%]
    vscr_pct: float = 1.1                      # resistive component of uk [%]
    tap_ratio: float = 1.0                     # kept at 1.0 for fair comparison vs pure LV
    model: str = "t"


@dataclass
class MvLineParams:
    """MV backbone conductor (single default, e.g. ACSR ~50 mm2 overhead)."""

    r_ohm_per_km: float = 0.54
    x_ohm_per_km: float = 0.37
    i_max_a: float = 185.0


@dataclass
class MvReinforcementRequest:
    schema_version: int
    topo_params: MvTopologyParams
    transformer_params: MvTransformerParams
    mv_line_params: MvLineParams
    pf_params: ValidationInputs                # reused as-is for the LV subnetworks (v_nom 0.4 kV)


@dataclass
class MvSubnetworkResult:
    """One LV subnetwork rooted at the slack (plant) or at a step-down transformer."""

    subnet_id: int
    root_kind: str                             # "slack" | "transformer"
    topology: TopologyResult
    peak_load_kw: float
    validation: Optional[ValidationResult] = None   # None for topology-only iterations
    tr_s_nom_kva: Optional[float] = None            # None for the slack subnetwork
    tr_loading_pct: Optional[float] = None
    worst_dv_pu: Optional[float] = None
    dv_cap_ok: Optional[bool] = None


@dataclass
class MvIterationResult:
    """State of one k-iteration (k transformers -> k+1 subnetworks)."""

    n_transformers: int
    subnetworks: List[MvSubnetworkResult]
    mv_backbone_edges_4326: Optional[gpd.GeoDataFrame]
    mv_backbone_length_km: float
    converged: bool                            # criterion check of this iteration passed
    pf_executed: bool                          # False for topology-only iterations (distance branch)
    solve_seconds: float
    transformer_summary: Optional[pd.DataFrame] = None  # per-transformer: s_nom, loading, internal dV
    cluster_diameters_m: Optional[Dict[int, float]] = None
    note: Optional[str] = None                 # e.g. "diverged: ..." or method annotations


@dataclass
class MvReinforcementResult:
    schema_version: int
    criterion: str                             # copied from the request, for rendering
    iterations: List[MvIterationResult]        # full history of the k-iteration
    final: MvIterationResult
    total_solve_seconds: float
    gdf_standalone_4326: Optional[gpd.GeoDataFrame] = None  # buildings excluded
                                               # by the standalone pre-pass
