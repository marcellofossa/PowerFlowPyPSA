from __future__ import annotations

"""Distribution cost model (Task 1) and standalone economics (Task 2).

Unit costs are synthesised from four real mini-grid cost workbooks
(DRC-Idjwi 4-site cost breakdown; Uganda Lake Victoria islands V2A/V2B BoQ;
Bugarula/Prolasa CAPEX; portfolio financial model). Reference values and
observed ranges are documented in `tabella_costi_unitari_distribuzione.xlsx`.

Design rules:
- Pure module: no Streamlit imports, safe to use from core services and tests.
- Every function maps *physical drivers already produced by the code*
  (lengths, pole counts, connections, transformer sizes) onto unit costs,
  so a differential analysis only requires changing site or site parameters.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Unit costs (reference values from the 4 cost workbooks)
# ---------------------------------------------------------------------------


@dataclass
class DistributionUnitCosts:
    """Distribution-only unit costs [USD].

    Defaults are the reference values of the unit-cost table; all fields can
    be overridden from the UI (sliders in the cost-analysis expanders).
    """

    # --- LV network -------------------------------------------------------
    lv_pole_material_usd: float = 175.0       # 9 m pole (DRC 175.8, Uganda 175)
    lv_pole_hardware_usd: float = 45.0        # accessories + suspension/stay kit
    lv_pole_foundation_usd: float = 78.0      # pole pit & casting (DRC)
    lv_cable_3ph_usd_per_m: float = 5.5       # ABC 4-core 35-70 mm2 (4.72-7.0)
    lv_cable_1ph_usd_per_m: float = 3.0       # ABC 2-core (2.40-3.5)
    lv_share_3ph: float = 1.0                 # share of backbone built 3-phase
    lv_earthing_usd_per_km: float = 1200.0
    lv_install_usd_per_km: float = 1950.0     # line stringing labour (1899-2000)
    transport_lump_usd: float = 5000.0        # per network (3000-8000, Uganda)

    # --- Service drop / last mile (per connection) --------------------------
    drop_cable_usd_per_m: float = 0.30        # 2-core 10-16 mm2 (0.287-1.32)
    ready_board_usd: float = 60.0             # ready board + distribution box
    meter_usd: float = 62.0                   # single-phase smart meter
    conn_hardware_usd: float = 25.0           # brackets, clamps, piercing conn.
    conn_earthing_usd: float = 18.0           # rod + Cu earth cable
    conn_install_usd: float = 20.0            # last-mile labour

    # --- MV network (Grid Reinforcement) ------------------------------------
    mv_pole_usd: float = 250.0                # between Kwenge BoQ (280) and V2A/V2B-era estimate (200)
    mv_pole_hardware_usd: float = 45.0        # same as LV (checkpoint decision)
    mv_pole_foundation_usd: float = 78.0      # same as LV (checkpoint decision)
    mv_cable_usd_per_km: float = 2500.0       # ~ACSR 50 mm2, consistent with MvLineParams
                                              # (r=0.54 ohm/km, 185 A); observed range:
                                              # 1373 (7x4.26 mm2, Uganda V2B) - 3870 (Kwenge)
    mv_stay_usd_per_km: float = 700.0         # stay + accessories (440-1410)
    mv_earthing_usd_per_km: float = 1200.0
    mv_install_usd_per_km: float = 2000.0

    # --- MV/LV transformers: scaling law C(S) = c0 * (S / s0)^alpha ---------
    # Fitted on the two step-down points of the Uganda V2B BoQ:
    # 25 kVA -> 3390 $ and 500 kVA -> 9087 $ (the 250 kVA @ 3933 $ entry is a
    # plant STEP-UP transformer, excluded from the law).
    tr_c0_usd: float = 3390.0
    tr_s0_kva: float = 25.0
    tr_alpha: float = 0.329
    tr_structure_usd: float = 400.0

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def lv_pole_installed_usd(self) -> float:
        return (
            float(self.lv_pole_material_usd)
            + float(self.lv_pole_hardware_usd)
            + float(self.lv_pole_foundation_usd)
        )

    def mv_pole_installed_usd(self) -> float:
        return (
            float(self.mv_pole_usd)
            + float(self.mv_pole_hardware_usd)
            + float(self.mv_pole_foundation_usd)
        )

    def lv_cable_mix_usd_per_m(self) -> float:
        s = min(max(float(self.lv_share_3ph), 0.0), 1.0)
        return s * float(self.lv_cable_3ph_usd_per_m) + (1.0 - s) * float(
            self.lv_cable_1ph_usd_per_m
        )

    def lv_backbone_usd_per_m(self) -> float:
        """Cable + stringing + earthing per metre of LV backbone (no poles)."""
        return (
            self.lv_cable_mix_usd_per_m()
            + (float(self.lv_install_usd_per_km) + float(self.lv_earthing_usd_per_km))
            / 1000.0
        )

    def connection_fixed_usd(self) -> float:
        """Fixed last-mile cost per connection (excluding the drop cable)."""
        return (
            float(self.ready_board_usd)
            + float(self.meter_usd)
            + float(self.conn_hardware_usd)
            + float(self.conn_earthing_usd)
            + float(self.conn_install_usd)
        )

    def transformer_cost_usd(self, s_nom_kva: float) -> float:
        """MV/LV step-down transformer cost via scaling law + mounting structure."""
        s = max(float(s_nom_kva), 1e-9)
        return float(self.tr_c0_usd) * (s / float(self.tr_s0_kva)) ** float(
            self.tr_alpha
        ) + float(self.tr_structure_usd)


def mv_pole_spacing_m(mv_v_nom_kv: float) -> float:
    """MV poles are equispaced: 60 m at 11 kV, 120 m at 33 kV (checkpoint)."""
    return 60.0 if float(mv_v_nom_kv) <= 20.0 else 120.0


# ---------------------------------------------------------------------------
# Standalone economics (Task 2)
# ---------------------------------------------------------------------------


@dataclass
class StandaloneEconomics:
    """Differential-cost standalone criterion parameters.

    A building stays on the grid when::

        C_conn + c_gen * E  <=  c_sa * E
        C_conn <= (c_sa - c_gen) * E      with E = energy_kwh_per_year * horizon_years
    """

    standalone_cost_usd_per_kwh: float = 0.90   # c_sa
    gen_cost_usd_per_kwh: float = 0.38          # c_gen
    energy_kwh_per_year: float = 180.0          # E_y per building
    horizon_years: float = 20.0                 # N (checkpoint decision)

    def threshold_usd_per_building(self) -> float:
        margin = float(self.standalone_cost_usd_per_kwh) - float(
            self.gen_cost_usd_per_kwh
        )
        return max(margin, 0.0) * float(self.energy_kwh_per_year) * float(
            self.horizon_years
        )


@dataclass(frozen=True)
class StandaloneGate:
    """Precomputed scalar gate consumed by the placement algorithm.

    Keeps `distribution_algos` decoupled from the full cost model: the service
    layer builds this once per run via :func:`build_standalone_gate`.
    """

    threshold_usd: float       # (c_sa - c_gen) * E, per building
    fixed_conn_usd: float      # last-mile fixed cost per connection
    drop_usd_per_m: float      # service-drop cable
    pole_usd: float            # one installed LV pole
    ext_usd_per_m: float       # backbone extension (cable+labour+support poles)


def build_standalone_gate(
    economics: StandaloneEconomics,
    unit_costs: Optional[DistributionUnitCosts] = None,
    *,
    max_pole_span_m: float = 0.0,
) -> StandaloneGate:
    uc = unit_costs or DistributionUnitCosts()
    pole_usd = uc.lv_pole_installed_usd()
    ext_per_m = uc.lv_backbone_usd_per_m()
    if max_pole_span_m and float(max_pole_span_m) > 0.0:
        # a support pole every max_pole_span_m along the extension
        ext_per_m += pole_usd / float(max_pole_span_m)
    return StandaloneGate(
        threshold_usd=economics.threshold_usd_per_building(),
        fixed_conn_usd=uc.connection_fixed_usd(),
        drop_usd_per_m=float(uc.drop_cable_usd_per_m),
        pole_usd=pole_usd,
        ext_usd_per_m=ext_per_m,
    )


# ---------------------------------------------------------------------------
# Cost breakdowns (Task 1)
# ---------------------------------------------------------------------------

_COLS = ["category", "item", "quantity", "unit", "unit_cost_usd", "total_usd"]


def _row(cat: str, item: str, qty: float, unit: str, unit_cost: float) -> Dict:
    return {
        "category": cat,
        "item": item,
        "quantity": round(float(qty), 3),
        "unit": unit,
        "unit_cost_usd": round(float(unit_cost), 3),
        "total_usd": float(qty) * float(unit_cost),
    }


def lv_network_cost(
    metrics: Mapping[str, float],
    costs: DistributionUnitCosts,
    *,
    n_networks: int = 1,
) -> pd.DataFrame:
    """LV network cost from the topology `metrics` dict of `run_low_voltage`."""
    bb_km = float(metrics.get("backbone_length_km", 0.0))
    n_poles = float(metrics.get("num_poles_total", 0))
    rows = [
        _row("LV network", "Poles (material + hardware + foundation)",
             n_poles, "pole", costs.lv_pole_installed_usd()),
        _row("LV network", "Backbone cable (3F/1F mix)",
             bb_km * 1000.0, "m", costs.lv_cable_mix_usd_per_m()),
        _row("LV network", "Line stringing (labour)",
             bb_km, "km", costs.lv_install_usd_per_km),
        _row("LV network", "Network earthing",
             bb_km, "km", costs.lv_earthing_usd_per_km),
        _row("LV network", "Transport (lump per network)",
             float(n_networks), "network", costs.transport_lump_usd),
    ]
    return pd.DataFrame(rows, columns=_COLS)


def last_mile_cost(
    metrics: Mapping[str, float],
    costs: DistributionUnitCosts,
) -> pd.DataFrame:
    """Service-drop + last-mile cost from the topology `metrics` dict."""
    drop_km = float(metrics.get("service_drop_length_km", 0.0))
    n_conn = float(metrics.get("num_served", 0))
    rows = [
        _row("Last mile", "Service-drop cable",
             drop_km * 1000.0, "m", costs.drop_cable_usd_per_m),
        _row("Last mile", "Ready board + distribution box",
             n_conn, "conn", costs.ready_board_usd),
        _row("Last mile", "Smart meter (single-phase)",
             n_conn, "conn", costs.meter_usd),
        _row("Last mile", "Connection hardware (clamps, brackets)",
             n_conn, "conn", costs.conn_hardware_usd),
        _row("Last mile", "Customer earthing (rod + Cu cable)",
             n_conn, "conn", costs.conn_earthing_usd),
        _row("Last mile", "Connection labour",
             n_conn, "conn", costs.conn_install_usd),
    ]
    return pd.DataFrame(rows, columns=_COLS)


def mv_network_cost(
    mv_backbone_length_km: float,
    transformer_kvas: Sequence[float],
    costs: DistributionUnitCosts,
    *,
    mv_v_nom_kv: float = 11.0,
) -> pd.DataFrame:
    """MV backbone + transformer cost (Grid Reinforcement).

    MV poles are equispaced along the backbone (60 m @ 11 kV, 120 m @ 33 kV).
    """
    mv_km = float(mv_backbone_length_km)
    spacing = mv_pole_spacing_m(mv_v_nom_kv)
    n_mv_poles = int(math.ceil(mv_km * 1000.0 / spacing)) + (1 if mv_km > 0 else 0)
    rows = [
        _row("MV network", f"MV poles (spacing {spacing:.0f} m @ {mv_v_nom_kv:.0f} kV)",
             n_mv_poles, "pole", costs.mv_pole_installed_usd()),
        _row("MV network", "MV cable",
             mv_km, "km", costs.mv_cable_usd_per_km),
        _row("MV network", "Stay + accessories",
             mv_km, "km", costs.mv_stay_usd_per_km),
        _row("MV network", "MV earthing",
             mv_km, "km", costs.mv_earthing_usd_per_km),
        _row("MV network", "MV line stringing (labour)",
             mv_km, "km", costs.mv_install_usd_per_km),
    ]
    for s_nom in transformer_kvas:
        if s_nom is None:
            continue
        rows.append(
            _row("Transformers", f"MV/LV transformer {float(s_nom):.0f} kVA "
                 "(scaling law + structure)",
                 1.0, "unit", costs.transformer_cost_usd(float(s_nom)))
        )
    return pd.DataFrame(rows, columns=_COLS)


def lv_cable_cost_from_line_params(
    resolved_line_params_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Per-cable-type backbone cost from the resolved line-parameter table.

    Returns one row per line_type (quantity = total length in m, unit cost =
    cost_usd_per_m from line_types.csv). Returns None when the table is
    missing, has no `cost_usd_per_m` column, or the column is entirely empty —
    callers then fall back to the generic cable slider.
    """
    df = resolved_line_params_df
    if df is None or df.empty:
        return None
    if "cost_usd_per_m" not in df.columns or "length_km" not in df.columns:
        return None
    d = df.copy()
    d["cost_usd_per_m"] = pd.to_numeric(d["cost_usd_per_m"], errors="coerce")
    if d["cost_usd_per_m"].notna().sum() == 0:
        return None
    if d["cost_usd_per_m"].isna().any():
        # partial costs are ambiguous: refuse and fall back to the slider
        return None
    if "line_type" not in d.columns:
        d["line_type"] = "catalog"
    rows = []
    for lt, sub in d.groupby("line_type", sort=False):
        length_m = float(pd.to_numeric(sub["length_km"], errors="coerce").sum()) * 1000.0
        rows.append(
            _row("LV network", f"Backbone cable [{lt}] (from catalog)",
                 length_m, "m", float(sub["cost_usd_per_m"].iloc[0]))
        )
    return pd.DataFrame(rows, columns=_COLS)


def combine_breakdowns(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=_COLS)
    return pd.concat(frames, ignore_index=True)


def breakdown_totals(df: pd.DataFrame) -> Dict[str, float]:
    """Per-category totals + grand total, for rendering."""
    out: Dict[str, float] = {}
    if df is None or df.empty:
        return {"Total": 0.0}
    for cat, sub in df.groupby("category", sort=False):
        out[str(cat)] = float(sub["total_usd"].sum())
    out["Total"] = float(df["total_usd"].sum())
    return out
