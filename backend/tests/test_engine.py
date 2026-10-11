"""
Tests for the Geo4Irr engine.

These are written as assertions about agronomic behaviour, not just code paths.
A scheduler can pass a unit test and still tell a grower something ruinous, so
the checks below encode what the answers have to look like: peak corn ET in a
Delaware July, the size of the profile on Evesboro sand, the direction the
filter moves when a probe says the field is drier than the model thought.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from app.engine.advisory import (
    DECISION_IRRIGATE, DECISION_RAIN_HOLD, DECISION_WET, ForecastDay,
    SystemSpec, build_advisory, yield_loss_estimate,
)
from app.engine.bayesian import (
    BayesianAssimilator, EnsembleAssimilator, Observation, SENSOR_CLASSES,
    depletion_from_vwc,
)
from app.engine.crops import get_crop
from app.engine.economics import PumpSpec, application_cost, energy_per_acre_inch
from app.engine.fao56 import (
    BalanceState, WeatherDay, kcb_from_canopy_cover, kcb_from_ndvi,
    reference_et, run_season, step_water_balance,
)
from app.engine.nitrogen import (
    build_nitrogen_plan, manure_available_n, nitrate_leaching_estimate,
)
from app.engine.sensors import Candidate, design_network, grid_candidates
from app.engine.soils import curve_number_for, get_soil

DE_LAT = 38.70


def july_day(rain=0.0) -> WeatherDay:
    """A hot, clear mid-July day in Sussex County."""
    return WeatherDay(
        day=date(2026, 7, 15), tmax_f=91.0, tmin_f=69.0,
        rh_max=92.0, rh_min=48.0, wind_mph=7.0,
        solar_rad_mj_m2=25.0, rain_in=rain,
    )


# ---------------------------------------------------------------------------
# Reference ET
# ---------------------------------------------------------------------------

def test_peak_july_eto_is_in_delaware_range():
    eto = reference_et(july_day(), DE_LAT)
    assert 0.17 <= eto <= 0.30, f"July ETo of {eto} in/day is outside Delmarva range"


def test_cool_april_eto_is_much_lower_than_july():
    april = WeatherDay(day=date(2026, 4, 10), tmax_f=61.0, tmin_f=41.0,
                       rh_max=85.0, rh_min=45.0, wind_mph=8.0,
                       solar_rad_mj_m2=16.0)
    assert reference_et(april, DE_LAT) < reference_et(july_day(), DE_LAT) * 0.75


def test_hargreaves_fallback_when_radiation_missing():
    w = WeatherDay(day=date(2026, 7, 15), tmax_f=91.0, tmin_f=69.0)
    eto = reference_et(w, DE_LAT)
    assert 0.10 <= eto <= 0.35
    # The fallback should be in the same neighbourhood as the full equation.
    assert abs(eto - reference_et(july_day(), DE_LAT)) < 0.12


def test_provider_supplied_eto_is_used_verbatim():
    w = july_day()
    w.eto_in = 0.213
    assert reference_et(w, DE_LAT) == 0.213


# ---------------------------------------------------------------------------
# Soil profiles
# ---------------------------------------------------------------------------

def test_evesboro_holds_far_less_water_than_matapeake():
    sand = get_soil("evesboro").taw_in(30)
    silt = get_soil("matapeake").taw_in(30)
    assert silt > sand * 2.4
    assert 2.0 <= sand <= 2.3, "30 in of Evesboro should hold about 2.1 in"


def test_restrictive_layer_caps_the_root_zone():
    s = get_soil("sassafras")
    s.restrictive_layer_in = 20.0
    assert s.taw_in(36) == pytest.approx(s.taw_in(20))


def test_shallow_water_table_supplies_upflux_only_when_dry():
    s = get_soil("pocomoke")
    wet = s.upflux_in(root_depth_in=24, depletion_in=0.05)
    dry = s.upflux_in(root_depth_in=24, depletion_in=2.5)
    assert dry > wet
    assert get_soil("evesboro").upflux_in(24, 2.5) == 0.0


def test_curve_number_drops_with_residue():
    assert curve_number_for("matapeake", 0) > curve_number_for("matapeake", 80)


# ---------------------------------------------------------------------------
# Water balance
# ---------------------------------------------------------------------------

def test_peak_corn_water_use_matches_delaware_expectation():
    crop = get_crop("corn_grain")
    soil = get_soil("sassafras")
    state = BalanceState(dr_in=0.4, root_depth_in=30)
    r = step_water_balance(state, july_day(), crop, soil, dap=75, lat_deg=DE_LAT)
    assert 0.20 <= r.etc_adj_in <= 0.32, (
        f"Peak corn ETc of {r.etc_adj_in} in/day is outside what Delaware corn does")


def test_irrigation_beyond_depletion_becomes_deep_percolation():
    crop = get_crop("corn_grain")
    soil = get_soil("evesboro")
    state = BalanceState(dr_in=0.5, root_depth_in=30)
    r = step_water_balance(state, july_day(), crop, soil, dap=75, lat_deg=DE_LAT,
                           irrigation_in=1.5)
    assert r.deep_perc_in > 0.5, "Over-application on sand must show as percolation"


def test_stress_engages_past_the_allowed_depletion():
    crop = get_crop("corn_grain")
    soil = get_soil("evesboro")
    state = BalanceState(dr_in=soil.taw_in(30) * 0.85, root_depth_in=30)
    r = step_water_balance(state, july_day(), crop, soil, dap=75, lat_deg=DE_LAT)
    assert r.ks < 1.0 and r.stress


def test_mad_tightens_during_pollination():
    crop = get_crop("corn_grain")
    assert crop.mad(70) < crop.mad(30)


def test_season_run_conserves_water():
    crop = get_crop("corn_grain")
    soil = get_soil("sassafras")
    plant = date(2026, 5, 1)
    days = [
        WeatherDay(day=plant + timedelta(days=i), tmax_f=82.0, tmin_f=61.0,
                   rh_max=90.0, rh_min=50.0, wind_mph=6.0,
                   solar_rad_mj_m2=22.0, rain_in=0.15 if i % 5 == 0 else 0.0)
        for i in range(100)
    ]
    res = run_season(days, crop, soil, plant, DE_LAT)
    assert len(res) == 100
    assert all(r.depletion_in <= r.taw_in + 1e-6 for r in res)
    assert all(r.depletion_in >= 0 for r in res)


# ---------------------------------------------------------------------------
# Canopy transfers
# ---------------------------------------------------------------------------

def test_canopy_cover_to_kcb_is_monotonic_and_bounded():
    vals = [kcb_from_canopy_cover(fc) for fc in (0.0, 0.2, 0.5, 0.8, 1.0)]
    assert vals == sorted(vals)
    assert vals[0] >= 0.14 and vals[-1] <= 1.16


def test_bare_soil_ndvi_gives_initial_kcb():
    assert kcb_from_ndvi(0.12) == 0.15


# ---------------------------------------------------------------------------
# Bayesian assimilation
# ---------------------------------------------------------------------------

def test_depletion_from_vwc_ignores_readings_below_the_root_front():
    soil = get_soil("sassafras")
    shallow = depletion_from_vwc({6: 0.12, 12: 0.12}, soil, root_depth_in=14)
    deep = depletion_from_vwc({6: 0.12, 12: 0.12, 36: 0.12}, soil, root_depth_in=14)
    assert shallow == deep


def test_filter_moves_toward_a_drier_probe_reading():
    crop, soil = get_crop("corn_grain"), get_soil("sassafras")
    a = BayesianAssimilator(soil, crop, DE_LAT)
    a.fs.dr_in = 0.8
    obs = Observation(day=july_day().day, kind="soil_moisture", value=2.6,
                      sensor_key="tdr_research", depths_reported=5)
    r = a.assimilate_day(july_day(), dap=75, observations=[obs])
    assert r.n_observations == 1
    # The posterior must sit between the model's prior and the probe reading,
    # and strictly closer to the probe when the probe is the accurate one.
    assert r.prior_depletion_in < r.posterior_depletion_in <= 2.6
    assert abs(r.posterior_depletion_in - 2.6) < abs(r.prior_depletion_in - 2.6)


def test_accurate_sensor_pulls_harder_than_a_cheap_one():
    crop, soil = get_crop("corn_grain"), get_soil("sassafras")
    moves = {}
    for key in ("tdr_research", "capacitance_basic"):
        a = BayesianAssimilator(soil, crop, DE_LAT)
        a.fs.dr_in, a.fs.var_dr = 0.8, 0.25
        obs = Observation(day=july_day().day, kind="soil_moisture", value=2.8,
                          sensor_key=key, depths_reported=2)
        r = a.assimilate_day(july_day(), dap=75, observations=[obs])
        moves[key] = r.posterior_depletion_in - r.prior_depletion_in
    assert moves["tdr_research"] > moves["capacitance_basic"], moves


def test_gross_outlier_is_gated_out():
    crop, soil = get_crop("corn_grain"), get_soil("sassafras")
    a = BayesianAssimilator(soil, crop, DE_LAT)
    a.fs.dr_in, a.fs.var_dr = 0.5, 0.02
    obs = Observation(day=july_day().day, kind="soil_moisture", value=8.0,
                      sensor_key="tdr_research", depths_reported=5)
    r = a.assimilate_day(july_day(), dap=75, observations=[obs])
    assert r.n_observations == 0
    assert any("rejected" in n for n in r.notes)


def test_uncertainty_grows_without_observations():
    crop, soil = get_crop("corn_grain"), get_soil("evesboro")
    a = BayesianAssimilator(soil, crop, DE_LAT)
    first = a.assimilate_day(july_day(), dap=70).depletion_sd_in
    for i in range(1, 8):
        w = july_day()
        w.day = date(2026, 7, 15) + timedelta(days=i)
        last = a.assimilate_day(w, dap=70 + i).depletion_sd_in
    assert last > first, "Open-loop uncertainty must widen"


def test_canopy_ahead_of_schedule_raises_the_kcb_bias():
    crop, soil = get_crop("corn_grain"), get_soil("sassafras")
    a = BayesianAssimilator(soil, crop, DE_LAT)
    obs = Observation(day=july_day().day, kind="canopy_cover", value=0.95,
                      sensor_key="uav_multispectral")
    a.assimilate_day(july_day(), dap=45, observations=[obs])
    assert a.fs.kcb_bias > 1.0


def test_ensemble_reports_a_spread():
    crop, soil = get_crop("corn_grain"), get_soil("evesboro")
    e = EnsembleAssimilator(soil, crop, DE_LAT, n_particles=120, seed=3)
    out = e.step(july_day(rain=0.4), dap=70, irrigation_in=0.0)
    assert out["p95_in"] >= out["p50_in"] >= out["p05_in"]


# ---------------------------------------------------------------------------
# Advisory
# ---------------------------------------------------------------------------

def _forecast(pop=0.1, qpf=0.1, days=7, tmax=90.0):
    return [ForecastDay(day=date(2026, 7, 15) + timedelta(days=i + 1),
                        tmax_f=tmax, tmin_f=68.0, pop=pop, qpf_in=qpf)
            for i in range(days)]


def _advisory(depletion, soil="evesboro", forecast=None, **kw):
    return build_advisory(
        field_id="t", field_name="Test pivot", as_of=date(2026, 7, 15),
        crop_key="corn_grain", planting=date(2026, 5, 1),
        soil=get_soil(soil), system=SystemSpec(acres=130, capacity_gpm=800),
        depletion_in=depletion, depletion_sd_in=0.25,
        forecast=forecast or _forecast(), **kw)


def test_dry_field_says_irrigate_today():
    a = _advisory(1.8)
    assert a.decision == DECISION_IRRIGATE
    assert a.recommended_net_in > 0
    assert a.run_hours > 0


def test_wet_field_recommends_nothing():
    a = _advisory(0.05)
    assert a.decision == DECISION_WET
    assert a.recommended_net_in == 0


def test_large_forecast_rain_holds_the_pivot():
    a = _advisory(0.95, forecast=_forecast(pop=0.9, qpf=1.2))
    assert a.decision == DECISION_RAIN_HOLD
    assert a.recommended_net_in == 0


def test_recommendation_never_exceeds_system_capability():
    a = build_advisory(
        field_id="t", field_name="Small pivot", as_of=date(2026, 7, 15),
        crop_key="corn_grain", planting=date(2026, 5, 1),
        soil=get_soil("matapeake"),
        system=SystemSpec(acres=130, capacity_gpm=500, max_application_in=0.6),
        depletion_in=3.5, depletion_sd_in=0.3, forecast=_forecast())
    assert a.recommended_net_in <= 0.6
    assert any("split" in c.lower() for c in a.constraints)


def test_allocation_limit_trims_the_recommendation():
    a = build_advisory(
        field_id="t", field_name="Limited", as_of=date(2026, 7, 15),
        crop_key="corn_grain", planting=date(2026, 5, 1),
        soil=get_soil("evesboro"),
        system=SystemSpec(acres=130, capacity_gpm=800,
                          allocation_ac_in_remaining=26.0),
        depletion_in=1.9, depletion_sd_in=0.3, forecast=_forecast())
    assert a.recommended_gross_in * 130 <= 26.1


def test_trajectory_carries_prediction_intervals():
    a = _advisory(0.9)
    assert len(a.trajectory) == 7
    for t in a.trajectory:
        assert t["p05_in"] <= t["p50_in"] <= t["p95_in"]
        assert 0.0 <= t["prob_past_mad"] <= 1.0


def test_sand_reaches_mad_sooner_than_silt_loam():
    sand = _advisory(0.9, soil="evesboro").days_to_mad
    silt = _advisory(0.9, soil="matapeake").days_to_mad
    assert sand is not None
    assert silt is None or silt > sand


def test_confidence_flag_reflects_sensor_presence():
    assert _advisory(1.0, has_sensors=True).confidence == "sensor-assimilated"
    assert _advisory(1.0).confidence == "modeled"


# ---------------------------------------------------------------------------
# Nitrogen
# ---------------------------------------------------------------------------

def test_surface_applied_litter_loses_more_ammonium():
    quick = manure_available_n("broiler_litter", 3.0, "immediate")
    slow = manure_available_n("broiler_litter", 3.0, "surface_no_incorporation")
    assert quick["available_year1"] > slow["available_year1"]
    assert slow["volatilization_loss"] > quick["volatilization_loss"]


def test_litter_credit_reduces_the_fertilizer_recommendation():
    plain = build_nitrogen_plan("corn_grain", 200)
    with_litter = build_nitrogen_plan(
        "corn_grain", 200,
        manure={"source_key": "broiler_litter", "rate_ton_per_ac": 3.0,
                "incorporation": "within_24h"})
    assert with_litter.recommended_fertilizer_lb < plain.recommended_fertilizer_lb - 50


def test_sandy_soil_gets_a_four_way_split():
    sandy = build_nitrogen_plan("corn_grain", 200, soil_awc_in_per_in=0.07)
    loamy = build_nitrogen_plan("corn_grain", 200, soil_awc_in_per_in=0.19)
    assert len(sandy.split_schedule) > len(loamy.split_schedule)
    assert sandy.leaching_risk == "high"


def test_soybean_gets_no_nitrogen():
    plan = build_nitrogen_plan("soybean_full", 60)
    assert plan.recommended_fertilizer_lb == 0
    assert plan.split_schedule[0]["lb_n_per_ac"] == 0


def test_high_phosphorus_litter_rate_raises_a_warning():
    plan = build_nitrogen_plan(
        "corn_grain", 200,
        manure={"source_key": "broiler_litter", "rate_ton_per_ac": 4.0,
                "incorporation": "within_24h"})
    assert any("P2O5" in w for w in plan.warnings)


def test_every_plan_points_back_to_the_certified_plan():
    plan = build_nitrogen_plan("corn_grain", 180)
    assert any("nutrient management plan" in w for w in plan.warnings)


def test_leaching_scales_with_deep_percolation():
    low = nitrate_leaching_estimate(0.5, 60, 30, 0.20)
    high = nitrate_leaching_estimate(4.0, 60, 30, 0.20)
    assert high["leached_lb_n"] > low["leached_lb_n"]
    assert high["risk_level"] in ("moderate", "high")


def test_irrigation_water_nitrate_is_counted():
    r = nitrate_leaching_estimate(0.0, 40, 30, 0.20,
                                  irrigation_water_no3_n_ppm=10.0,
                                  irrigation_in=12.0)
    assert r["irrigation_n_applied_lb"] > 25


# ---------------------------------------------------------------------------
# Sensor design
# ---------------------------------------------------------------------------

def test_more_budget_buys_more_stations_and_lower_uncertainty():
    cands = grid_candidates(2600, 2600, 400)
    small = design_network(cands, 1200, 30)
    large = design_network(cands, 6000, 30)
    assert large.n_stations >= small.n_stations
    assert large.expected_depletion_sd_in <= small.expected_depletion_sd_in


def test_sensors_always_beat_no_sensors():
    cands = grid_candidates(2600, 2600, 400)
    plan = design_network(cands, 3000, 30)
    assert plan.expected_depletion_sd_in < plan.sd_without_sensors_in


def test_mixed_soils_pull_stations_onto_distinct_map_units():
    cands = grid_candidates(
        2600, 2600, 400,
        soil_map={"evesboro": [(0, 0, 1300, 2600)],
                  "matapeake": [(1300, 0, 2600, 2600)]},
        default_soil="sassafras")
    plan = design_network(cands, 5000, 30, max_stations=4)
    keys = {s["soil_key"] for s in plan.stations}
    assert len(keys) >= 2, "Design must sample both map units"


def test_restrictive_layer_caps_sensor_depth():
    cands = grid_candidates(1600, 1600, 400)
    plan = design_network(cands, 3000, 36, restrictive_layer_in=20)
    assert all(max(s["depths_in"]) <= 22 for s in plan.stations)


def test_budget_below_one_unit_returns_advice_not_an_empty_plan():
    cands = grid_candidates(1600, 1600, 400)
    plan = design_network(cands, 150, 30, sensor_key="tdr_research")
    assert plan.n_stations == 0
    assert plan.rationale


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------

def test_pumping_cost_is_in_a_plausible_delaware_range():
    e = energy_per_acre_inch(800, PumpSpec(lift_ft=60, discharge_pressure_psi=35))
    assert 2.0 <= e["cost_per_ac_in"] <= 12.0, e


def test_higher_pressure_costs_more():
    low = energy_per_acre_inch(800, PumpSpec(discharge_pressure_psi=20))
    high = energy_per_acre_inch(800, PumpSpec(discharge_pressure_psi=60))
    assert high["cost_per_ac_in"] > low["cost_per_ac_in"]


def test_application_cost_scales_with_area_and_depth():
    a = application_cost(0.5, 130, 800, PumpSpec())
    b = application_cost(1.0, 130, 800, PumpSpec())
    assert b["total_cost"] > a["total_cost"] * 1.8


# ---------------------------------------------------------------------------
# Yield response
# ---------------------------------------------------------------------------

def test_stress_during_pollination_costs_more_yield_than_late_stress():
    crop = get_crop("corn_grain")
    soil = get_soil("evesboro")

    def season(stress_window):
        state = BalanceState(root_depth_in=30)
        out = []
        for dap in range(30, 120):
            w = july_day()
            w.day = date(2026, 5, 1) + timedelta(days=dap)
            dry = stress_window[0] <= dap <= stress_window[1]
            state.dr_in = soil.taw_in(30) * (0.90 if dry else 0.20)
            out.append(step_water_balance(state, w, crop, soil, dap, DE_LAT))
        return yield_loss_estimate(out, crop)

    pollination = season((65, 90))
    late = season((100, 118))
    assert pollination["yield_loss_pct"] > late["yield_loss_pct"]
