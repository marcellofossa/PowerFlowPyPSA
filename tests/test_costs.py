import math

import pytest

from core.costs import (
    DistributionUnitCosts,
    StandaloneEconomics,
    breakdown_totals,
    build_standalone_gate,
    combine_breakdowns,
    last_mile_cost,
    lv_network_cost,
    mv_network_cost,
    mv_pole_spacing_m,
)


# ---------------------------------------------------------------------------
# Transformer scaling law
# ---------------------------------------------------------------------------

def test_transformer_scaling_law_anchors():
    uc = DistributionUnitCosts()
    # anchored on the two Uganda V2B step-down points (+ structure 400 $)
    assert uc.transformer_cost_usd(25.0) == pytest.approx(3390.0 + 400.0, rel=1e-3)
    assert uc.transformer_cost_usd(500.0) == pytest.approx(9087.0 + 400.0, rel=2e-3)


def test_transformer_scaling_law_monotonic():
    uc = DistributionUnitCosts()
    sizes = [25.0, 50.0, 100.0, 200.0, 315.0, 500.0]
    vals = [uc.transformer_cost_usd(s) for s in sizes]
    assert all(b > a for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# MV pole spacing (60 m @ 11 kV, 120 m @ 33 kV)
# ---------------------------------------------------------------------------

def test_mv_pole_spacing():
    assert mv_pole_spacing_m(11.0) == 60.0
    assert mv_pole_spacing_m(33.0) == 120.0


def test_mv_network_pole_count():
    uc = DistributionUnitCosts()
    df = mv_network_cost(3.0, [], uc, mv_v_nom_kv=11.0)
    poles_row = df[df["item"].str.startswith("MV poles")].iloc[0]
    # 3 km / 60 m = 50 spans -> 51 poles
    assert int(poles_row["quantity"]) == 51
    assert poles_row["unit_cost_usd"] == pytest.approx(250.0 + 45.0 + 78.0)

    df33 = mv_network_cost(3.0, [], uc, mv_v_nom_kv=33.0)
    poles_row33 = df33[df33["item"].str.startswith("MV poles")].iloc[0]
    assert int(poles_row33["quantity"]) == 26  # 3 km / 120 m -> 25 spans + 1


# ---------------------------------------------------------------------------
# LV + last-mile breakdowns
# ---------------------------------------------------------------------------

METRICS = {
    "backbone_length_km": 2.0,
    "service_drop_length_km": 0.5,
    "num_poles_total": 52,
    "num_served": 100,
    "num_unserved": 5,
}


def test_lv_network_cost_total():
    uc = DistributionUnitCosts()
    df = lv_network_cost(METRICS, uc)
    expected = (
        52 * (175.0 + 45.0 + 78.0)          # poles
        + 2000.0 * 5.5                      # cable (share_3ph = 1)
        + 2.0 * 1950.0                      # stringing
        + 2.0 * 1200.0                      # earthing
        + 5000.0                            # transport
    )
    assert df["total_usd"].sum() == pytest.approx(expected)


def test_last_mile_cost_total():
    uc = DistributionUnitCosts()
    df = last_mile_cost(METRICS, uc)
    expected = 500.0 * 0.30 + 100 * (60.0 + 62.0 + 25.0 + 18.0 + 20.0)
    assert df["total_usd"].sum() == pytest.approx(expected)


def test_cable_mix():
    uc = DistributionUnitCosts(lv_share_3ph=0.5)
    assert uc.lv_cable_mix_usd_per_m() == pytest.approx(0.5 * 5.5 + 0.5 * 3.0)


def test_breakdown_totals_and_combine():
    uc = DistributionUnitCosts()
    df = combine_breakdowns([lv_network_cost(METRICS, uc), last_mile_cost(METRICS, uc)])
    totals = breakdown_totals(df)
    assert totals["Total"] == pytest.approx(
        totals["LV network"] + totals["Last mile"]
    )


# ---------------------------------------------------------------------------
# Standalone economics (Task 2)
# ---------------------------------------------------------------------------

def test_threshold_default():
    eco = StandaloneEconomics()
    # (0.90 - 0.38) * 180 * 20 = 1872 $
    assert eco.threshold_usd_per_building() == pytest.approx(1872.0)


def test_threshold_never_negative():
    eco = StandaloneEconomics(
        standalone_cost_usd_per_kwh=0.2, gen_cost_usd_per_kwh=0.5
    )
    assert eco.threshold_usd_per_building() == 0.0


def test_gate_extension_includes_support_poles():
    eco = StandaloneEconomics()
    uc = DistributionUnitCosts()
    g_no_span = build_standalone_gate(eco, uc, max_pole_span_m=0.0)
    g_span = build_standalone_gate(eco, uc, max_pole_span_m=40.0)
    assert g_span.ext_usd_per_m == pytest.approx(
        g_no_span.ext_usd_per_m + uc.lv_pole_installed_usd() / 40.0
    )
    assert g_no_span.fixed_conn_usd == pytest.approx(185.0)  # 60+62+25+18+20


# ---------------------------------------------------------------------------
# Per-cable-type cost from the line catalog
# ---------------------------------------------------------------------------

import pandas as pd

from core.costs import lv_cable_cost_from_line_params


def test_cable_cost_from_catalog():
    resolved = pd.DataFrame(
        {
            "line_id": ["a", "b", "c"],
            "line_type": ["ABC_4x50", "ABC_4x50", "ABC_4x70"],
            "length_km": [1.0, 0.5, 2.0],
            "cost_usd_per_m": [3.5, 3.5, 4.7],
        }
    )
    df = lv_cable_cost_from_line_params(resolved)
    assert df is not None and len(df) == 2
    assert df["total_usd"].sum() == pytest.approx(1500 * 3.5 + 2000 * 4.7)


def test_cable_cost_fallback_cases():
    # no table / no column / partially missing costs -> None (slider fallback)
    assert lv_cable_cost_from_line_params(None) is None
    no_col = pd.DataFrame({"line_type": ["x"], "length_km": [1.0]})
    assert lv_cable_cost_from_line_params(no_col) is None
    partial = pd.DataFrame(
        {"line_type": ["x", "y"], "length_km": [1.0, 1.0],
         "cost_usd_per_m": [3.5, None]}
    )
    assert lv_cable_cost_from_line_params(partial) is None
