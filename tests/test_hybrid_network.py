"""
Tests for the hybrid MV/LV extension of core/powerflow_network.py.

Synthetic two-subnet case with hand-checkable physics:
- Subnet 0 (plant): slack pole + one loaded pole (10 kW), ~110 m of LV cable.
- Subnet 1 (remote, ~2.2 km away): transformer root pole + one loaded pole
  (50 kW), ~220 m of LV cable.
- MV backbone (11 kV): plant -> remote, 2.2 km, via step-up (200 kVA) and
  step-down (100 kVA) transformers with uk = 4%, vscr = 1.1%.

Analytic oracles:
- step-down internal drop ~ (r_pu*cos(phi) + x_pu*sin(phi)) * loading
- MV drop over 2.2 km at ~53 kVA is negligible (< 0.2%)
- LV drop on subnet-1 branch ~ (R*P + X*Q) / V_LL^2
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from core.powerflow_network import (
    HybridPyPSAPowerFlowRunner,
    LvSubnetSpec,
    MvLayerSpec,
    MvTransformerSpec,
    PFScenarioParams,
    PFTopologyBundle,
    _transformer_pu_impedance,
)

DEG = 0.001 / 0.111  # ~0.001 km in degrees of latitude -> 1 m ~ 9.009e-6 deg
LAT0, LON0 = 9.0, 8.0  # somewhere in Nigeria


def _nodes_gdf(points):
    """points: list of (pole_id, dx_m, dy_m) offsets from a local origin."""
    recs = []
    for pid, dx, dy, lat0, lon0 in points:
        lat = lat0 + dy / 1000.0 * 0.009009
        lon = lon0 + dx / 1000.0 * 0.009009 / np.cos(np.deg2rad(lat0))
        recs.append({"pole_id": pid, "geometry": Point(lon, lat)})
    return gpd.GeoDataFrame(recs, crs="EPSG:4326")


@pytest.fixture()
def hybrid_case():
    # subnet 0 at the plant: poles 0 (slack) and 1 (10 kW), ~110 m apart
    g0 = _nodes_gdf([(0, 0.0, 0.0, LAT0, LON0), (1, 110.0, 0.0, LAT0, LON0)])
    b0 = PFTopologyBundle(gdf_nodes_4326=g0, pole_id_col="pole_id", mst_edges_pole_ids=[(0, 1)])
    s0 = LvSubnetSpec(subnet_id=0, bundle=b0, root_pole_id=0, pole_p_kw={1: 10.0})

    # subnet 1 remote (~2.2 km east): poles 0 (transformer root) and 1 (50 kW), ~220 m
    lat1, lon1 = LAT0, LON0 + 2.2 / 111.0 / np.cos(np.deg2rad(LAT0))
    g1 = _nodes_gdf([(0, 0.0, 0.0, lat1, lon1), (1, 220.0, 0.0, lat1, lon1)])
    b1 = PFTopologyBundle(gdf_nodes_4326=g1, pole_id_col="pole_id", mst_edges_pole_ids=[(0, 1)])
    s1 = LvSubnetSpec(subnet_id=1, bundle=b1, root_pole_id=0, pole_p_kw={1: 50.0})

    mv = MvLayerSpec(
        v_nom_kv=11.0,
        nodes={"mv_plant": (LAT0, LON0), "mv_tr1": (lat1, lon1)},
        edges=[("MV0", "mv_plant", "mv_tr1", 2.2)],
        r_ohm_per_km=0.54,
        x_ohm_per_km=0.37,
        i_max_a=185.0,
    )
    trs = [
        MvTransformerSpec(name="tr_stepup", mv_bus="mv_plant",
                          lv_bus=HybridPyPSAPowerFlowRunner.lv_bus_name(0, 0),
                          s_nom_kva=200.0, vsc_pct=4.0, vscr_pct=1.1),
        MvTransformerSpec(name="tr_1", mv_bus="mv_tr1",
                          lv_bus=HybridPyPSAPowerFlowRunner.lv_bus_name(1, 0),
                          s_nom_kva=100.0, vsc_pct=4.0, vscr_pct=1.1),
    ]
    params = PFScenarioParams(
        slack_pole_id=0, v_min_pu=0.90, v_max_pu=1.10, pf_load=0.95,
        v_nom_kv=0.4, r_ohm_per_km=0.641, x_ohm_per_km=0.083, s_nom_kva=114.0,
    )
    return s0, s1, mv, trs, params


def _run(hybrid_case):
    s0, s1, mv, trs, params = hybrid_case
    runner = HybridPyPSAPowerFlowRunner([s0, s1], mv, trs)
    return runner.run_snapshot(params=params, debug=True)


def test_pu_impedance_conversion():
    r_pu, x_pu = _transformer_pu_impedance(4.0, 1.1)
    assert r_pu == pytest.approx(0.011)
    assert x_pu == pytest.approx(np.sqrt(0.04**2 - 0.011**2))
    with pytest.raises(ValueError):
        _transformer_pu_impedance(4.0, 5.0)   # vscr >= vsc
    with pytest.raises(ValueError):
        _transformer_pu_impedance(0.0, 0.0)


def test_converges_and_voltages_sane(hybrid_case):
    out = _run(hybrid_case)
    assert out["summary"]["n_subnets"] == 2
    assert out["summary"]["n_transformers"] == 2
    for sid in (0, 1):
        v = out["subnet_results"][sid]["bus_results"]["v_pu"]
        assert v.between(0.85, 1.01).all()
    # slack exactly 1.0
    b0 = out["subnet_results"][0]["bus_results"].set_index("bus")
    assert b0.loc[0, "v_pu"] == pytest.approx(1.0, abs=1e-6)


def test_namespacing_returns_original_pole_ids(hybrid_case):
    out = _run(hybrid_case)
    for sid in (0, 1):
        buses = set(out["subnet_results"][sid]["bus_results"]["bus"].tolist())
        assert buses == {0, 1}  # same local ids in both subnets, no collision


def test_stepdown_loading_and_internal_drop_match_analytics(hybrid_case):
    out = _run(hybrid_case)
    tr = out["transformer_results"].set_index("transformer").loc["tr_1"]
    # S at the MV side = load S plus downstream LV-line losses (active and
    # reactive), so it must sit between the pure load S and load S + ~15%.
    s_load_kva = 50.0 / 0.95  # ~52.6 kVA
    assert s_load_kva <= tr["s0_kVA"] <= s_load_kva * 1.15
    assert tr["loading_pct"] == pytest.approx(100.0 * tr["s0_kVA"] / 100.0, rel=1e-6)

    # internal drop oracle evaluated at the MEASURED loading/power factor
    r_pu, x_pu = _transformer_pu_impedance(4.0, 1.1)
    p0, q0 = float(tr["p0_MW"]), float(tr["q0_MVAr"])
    s0 = float(np.hypot(p0, q0))
    cosphi, sinphi = p0 / s0, q0 / s0
    loading = tr["s0_kVA"] / 100.0
    dv_expected = (r_pu * cosphi + x_pu * sinphi) * loading  # ~0.013 pu
    assert tr["dv_internal_pu"] == pytest.approx(dv_expected, rel=0.20, abs=0.003)


def test_mv_drop_negligible(hybrid_case):
    out = _run(hybrid_case)
    mv = out["mv_line_results"].iloc[0]
    assert abs(mv["v0_pu"] - mv["v1_pu"]) < 0.002  # 1/V^2 scaling at 11 kV
    assert mv["I_A"] < 5.0  # ~53 kVA at 11 kV -> ~2.8 A


def test_lv_branch_drop_matches_analytics(hybrid_case):
    out = _run(hybrid_case)
    b1 = out["subnet_results"][1]["bus_results"].set_index("bus")
    # branch: ~0.22 km * (0.641, 0.083) ohm/km, P = 50 kW, Q = P*tan(phi)
    R, X = 0.22 * 0.641, 0.22 * 0.083
    P, Q = 50e3, 50e3 * np.tan(np.arccos(0.95))
    dv_expected = (R * P + X * Q) / (400.0**2)  # pu, ~0.046
    dv_pf = float(b1.loc[0, "v_pu"] - b1.loc[1, "v_pu"])
    assert dv_pf == pytest.approx(dv_expected, rel=0.15)


def test_worst_dv_by_subnet_reported(hybrid_case):
    out = _run(hybrid_case)
    worst = out["summary"]["worst_dv_pu_by_subnet"]
    assert set(worst.keys()) == {0, 1}
    assert worst[1] > worst[0]  # remote subnet has trafo + MV + bigger LV drop
    assert out["summary"]["worst_dv_pu"] == pytest.approx(max(worst.values()))


def test_missing_subnet0_raises(hybrid_case):
    s0, s1, mv, trs, params = hybrid_case
    with pytest.raises(ValueError):
        HybridPyPSAPowerFlowRunner([s1], mv, trs)


def test_bad_transformer_bus_raises(hybrid_case):
    s0, s1, mv, trs, params = hybrid_case
    bad = [MvTransformerSpec(name="bad", mv_bus="nope", lv_bus="s0_0",
                             s_nom_kva=100.0, vsc_pct=4.0, vscr_pct=1.1)]
    runner = HybridPyPSAPowerFlowRunner([s0, s1], mv, bad)
    with pytest.raises(ValueError):
        runner.run_snapshot(params=params, debug=False)
