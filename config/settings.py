from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent


class PathManager:
    """
    Simple centralised path helper.

    Usage:
        from config.settings import PathManager
        PathManager.ASSETS / "distribution_methodology.png"
    """

    ROOT = ROOT_DIR
    ASSETS = ROOT / "config" / "assets"
    EXAMPLES = ROOT / "examples"


# ---------------------------------------------------------------------
# App defaults
# ---------------------------------------------------------------------

# Target projected CRS for all calculations (UTM 33N as placeholder)
TARGET_CRS = 32633

# Cost defaults
DEFAULT_COST_PER_KM_LV = 3000.0      # USD/km
DEFAULT_FIXED_COSTS_LV = 0.0         # USD

# Heuristic defaults
DEFAULT_SAMPLING_DISTANCE_M = 40     # m between candidate poles
DEFAULT_USER_DISTANCE_M = 35         # max distance user–pole
DEFAULT_MAX_ASSOCIATIONS = 16        # max users per pole


# ---------------------------------------------------------------------
# Grid Reinforcement (hybrid MV/LV) defaults
# ---------------------------------------------------------------------

# Partition criterion / k-iteration
MV_MAX_TRANSFORMERS = 10                 # safety stop for the k-iteration
MV_MAX_CLUSTER_DIAMETER_M = 1000.0       # distance-cap: max intra-cluster distance [m]
MV_DV_CAP_PU = 0.10                      # voltage-cap: max voltage drop per subnetwork [p.u.]
MV_PARTITION_SEED = 42                   # deterministic k-means initialisation

# MV backbone voltage options [kV] (common rural distribution levels, e.g. Nigeria)
MV_V_NOM_KV_OPTIONS = (11.0, 33.0)
MV_V_NOM_KV_DEFAULT = 11.0

# Step-down transformer electrical defaults (distribution class)
MV_TR_STANDARD_SIZES_KVA = (25.0, 50.0, 100.0, 200.0, 315.0, 500.0)
MV_TR_SIZING_MARGIN = 1.25               # auto-sizing: cluster peak x margin -> next standard size
MV_TR_VSC_PCT = 4.0                      # short-circuit voltage uk [%] (<= 630 kVA class)
MV_TR_VSCR_PCT = 1.1                     # resistive component of uk [%]
MV_TR_TAP_RATIO = 1.0                    # fixed at 1.0 for fair comparison vs pure LV

# MV backbone conductor default (bare overhead ACSR ~50 mm2 class)
MV_LINE_R_OHM_PER_KM = 0.54
MV_LINE_X_OHM_PER_KM = 0.37
MV_LINE_I_MAX_A = 185.0

# MV pole spacing along the backbone (pole-count estimate, as in Silvestri)
MV_POLE_SPACING_M = 40.0
