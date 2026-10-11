"""
Tests for the modules the project narrative specifies by name.

Each test names the Activity or equation it is defending, because the point of
this file is that the implementation follows the proposal rather than something
easier that resembles it. When one of these fails, the question is not only
"is the code wrong" but "has the method drifted away from what was proposed."
"""

from __future__ import annotations

import math
import pathlib
from datetime import date, timedelta

import pytest

from app.engine import ingest, likelihood as lk, optimization as opt, swatplus as sw, vri
from app.engine.fao56 import (
    WeatherDay, dans_from_thermal_image, dans_index, ks_from_cwsi, ks_from_dans,
)
from app.engine.geo import parse_geojson
from app.engine.sensors import optimal_sensor_depths
from app.engine.soils import get_soil

PLANTING = date(2026, 4, 28)
LAT = 38.64


def season(days: int = 130, rain_every: int = 9, rain_in: float = 1.3):
    out = []
    for i in range(days):
        d = PLANTING + timedelta(days=i)
        t = math.sin((d.timetuple().tm_yday - 80) / 200 * math.pi)
        out.append(WeatherDay(day=d, tmax_f=68 + 24 * t, tmin_f=48 + 22 * t,
                              rh_max=93, rh_min=52, wind_mph=5.5,
                              solar_rad_mj_m2=12 + 16 * t,
                              rain_in=rain_in if i % rain_every == 0 else 0.0))
    return out


def pivot(acres: float = 132.0, lat0: float = 38.6431, lon0: float = -75.4664):
    r = math.sqrt(acres * 4046.8564 / math.pi)
    ring = [[lon0 + r * math.sin(2 * math.pi * i / 120) / (111320 * math.cos(math.radians(lat0))),
             lat0 + r * math.cos(2 * math.pi * i / 120) / 110540] for i in range(120)]
    return parse_geojson({"type": "Polygon", "coordinates": [ring]})


# ===========================================================================
# Activity 5 — Eq 4 and Eq 5
# ===========================================================================

def test_eq5_matches_the_analytic_gaussian_log_likelihood():
    # One observation, zero residual, unit sigma: log L = -0.5 log(2 pi).
    assert lk.log_likelihood([0.0], [0.0], [1.0]) == pytest.approx(
        -0.5 * math.log(2 * math.pi), abs=1e-9)


def test_eq5_penalizes_larger_residuals():
    close = lk.log_likelihood([1.0], [1.02], [0.05])
    far = lk.log_likelihood([1.0], [1.20], [0.05])
    assert close > far


def test_eq5_weights_a_precise_sensor_more_than_a_crude_one():
    """The hybrid sensing strategy in the Rationale depends on this.

    A research TDR probe and a cheap capacitance probe reporting the same
    disagreement with the model must not move the posterior equally.
    """
    precise = lk.log_likelihood([1.0], [1.1], [0.010])
    crude = lk.log_likelihood([1.0], [1.1], [0.055])
    assert precise < crude          # the precise sensor is far more surprised


def test_eq5_rejects_a_nonpositive_sigma():
    with pytest.raises(ValueError):
        lk.log_likelihood([1.0], [1.0], [0.0])


def test_correlated_likelihood_reduces_to_eq5_when_uncorrelated():
    cov = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]
    indep = lk.log_likelihood([1, 2, 3], [1.1, 1.9, 3.2], [0.1] * 3)
    full = lk.log_likelihood_correlated([1, 2, 3], [1.1, 1.9, 3.2], cov)
    assert indep == pytest.approx(full, abs=1e-9)


def test_correlated_residuals_are_not_counted_as_independent_evidence():
    """Activity 5: "modified by appropriate transformations".

    Fourteen autocorrelated daily probe readings are not fourteen independent
    observations, and treating them as such makes the filter overconfident.
    """
    resid = [0.10] * 14
    obs = [1.0 + r for r in resid]
    mod = [1.0] * 14
    naive = lk.log_likelihood(obs, mod, 0.05)
    o, m, s = lk.ar1_whiten(obs, mod, 0.05, rho=0.85)
    whitened = lk.log_likelihood(o, m, s)
    assert whitened > naive


def test_ar1_rho_recovers_a_known_autocorrelation():
    import random
    random.seed(11)
    series, prev = [], 0.0
    for _ in range(400):
        prev = 0.75 * prev + random.gauss(0, 0.3)
        series.append(prev)
    assert lk.estimate_ar1_rho(series) > 0.5


def test_bayes_update_is_the_exact_conjugate_posterior():
    # Prior N(2, 0.25), observation 3 with sd 0.5 -> N(2.5, 0.125).
    post = lk.bayes_update(lk.Gaussian(2.0, 0.25), 3.0, 0.5)
    assert post.mean == pytest.approx(2.5)
    assert post.var == pytest.approx(0.125)


def test_posterior_is_never_wider_than_the_prior():
    prior = lk.Gaussian(1.5, 0.30)
    for sd in (0.02, 0.1, 0.5, 2.0):
        assert lk.bayes_update(prior, 2.0, sd).var <= prior.var


def test_prediction_interval_uses_the_5th_and_95th_percentiles():
    """Activity 5 names these two quantiles explicitly."""
    lo, hi = lk.Gaussian(2.0, 0.25).interval(0.05, 0.95)
    assert lo == pytest.approx(2.0 - 1.6449 * 0.5, abs=0.01)
    assert hi == pytest.approx(2.0 + 1.6449 * 0.5, abs=0.01)


def test_evidence_falls_when_observations_keep_surprising_the_model():
    prior = lk.Gaussian(1.0, 0.04)
    assert lk.evidence(prior, 1.05, 0.05) > lk.evidence(prior, 2.20, 0.05)


# --- generative filtering ---------------------------------------------------

def test_generative_filter_narrows_on_an_observation():
    f = lk.GenerativeFilter(n_particles=400)
    f.initialize(dr_in=1.0, dr_sd=0.6, taw_in=3.0)
    before = f.sd("dr")
    f.update([("dr", 1.1, 0.05)])
    assert f.sd("dr") < before
    assert f.mean("dr") == pytest.approx(1.1, abs=0.12)


def test_generative_filter_respects_the_zero_bound():
    """A Gaussian would put mass on negative depletion after a big rain."""
    f = lk.GenerativeFilter(n_particles=300)
    f.initialize(dr_in=0.10, dr_sd=0.40, taw_in=3.0)
    assert f.quantile(0.01, "dr") >= 0.0


def test_generative_filter_estimates_parameters_alongside_states():
    """Activity 5 asks for parameter updating, not only state updating."""
    f = lk.GenerativeFilter(n_particles=500)
    f.initialize(dr_in=1.0, dr_sd=0.3, kcb_bias=1.0, kcb_bias_sd=0.15, taw_in=3.0)
    for _ in range(6):
        f.update([("kcb_bias", 1.25, 0.05)])
    assert f.mean("kcb_bias") > 1.12


def test_generative_filter_is_deterministic():
    def run():
        f = lk.GenerativeFilter(n_particles=200)
        f.initialize(1.0, 0.5, taw_in=3.0)
        f.propagate(lambda p: p["dr"] + 0.2, 0.05, taw_in=3.0)
        f.update([("dr", 1.4, 0.08)])
        return f.summary()
    assert run() == run()


def test_filter_resamples_when_the_sample_degenerates():
    f = lk.GenerativeFilter(n_particles=300)
    f.initialize(1.0, 0.6, taw_in=3.0)
    out = f.update([("dr", 2.4, 0.02)])
    assert out["resampled"] is True


# --- Activity 4 -------------------------------------------------------------

def test_error_characterization_separates_bias_from_precision():
    ref = [0.12, 0.15, 0.18, 0.14, 0.11, 0.16, 0.19, 0.13]
    biased = [v + 0.02 for v in ref]
    out = lk.characterize_error(ref, biased)
    assert out["bias"] == pytest.approx(0.02, abs=1e-6)
    assert out["sd"] == pytest.approx(0.0, abs=1e-6)
    assert "Calibrate" in out["note"]


def test_error_characterization_reports_sigma_for_eq5():
    ref = [0.10, 0.14, 0.18, 0.12, 0.16, 0.11, 0.19, 0.13]
    noisy = [0.11, 0.13, 0.20, 0.11, 0.18, 0.10, 0.18, 0.15]
    out = lk.characterize_error(ref, noisy)
    assert out["sigma_for_eq5"] > 0
    assert out["n"] == 8


# ===========================================================================
# Activity 6 — SWAT+ and the Eq 1 flux terms
# ===========================================================================

def test_curve_number_rises_as_the_profile_fills():
    """SWAT+ continuous soil moisture accounting, not a fixed CN."""
    wet = sw.curve_number_moisture_adjusted(74, depletion_in=0.0, taw_in=3.0)
    dry = sw.curve_number_moisture_adjusted(74, depletion_in=3.0, taw_in=3.0)
    assert wet > 74 > dry


def test_the_same_storm_runs_off_a_wet_profile_and_soaks_into_a_dry_one():
    wet_cn = sw.curve_number_moisture_adjusted(74, 0.0, 3.0)
    dry_cn = sw.curve_number_moisture_adjusted(74, 2.8, 3.0)
    assert sw.runoff_curve_number(2.0, wet_cn) > 3 * sw.runoff_curve_number(2.0, dry_cn)


def test_upflux_falls_off_with_depth_to_the_water_table():
    """Eq 1's U term, which FAO-56 omits entirely."""
    shallow = sw.capillary_upflux(18, 24, 1.5, 2.5)
    deep = sw.capillary_upflux(44, 24, 1.5, 2.5)
    assert shallow > deep > 0


def test_upflux_is_zero_without_a_water_table():
    assert sw.capillary_upflux(None, 24, 1.5, 2.5) == 0.0


def test_upflux_scales_with_how_dry_the_root_zone_is():
    dry = sw.capillary_upflux(24, 20, 2.2, 2.5)
    moist = sw.capillary_upflux(24, 20, 0.2, 2.5)
    assert dry > moist


def test_sand_drains_faster_than_silt_loam():
    sand = sw.percolation_storage_routing(1.0, ksat_in_hr=4.5)
    loam = sw.percolation_storage_routing(1.0, ksat_in_hr=0.2)
    assert sand > loam


def test_root_depth_follows_canopy_not_the_calendar():
    """Activity 6: SWAT+'s linear-from-planting root growth is replaced."""
    early = sw.root_depth_from_canopy(0.15, 6, 36)
    mid = sw.root_depth_from_canopy(0.50, 6, 36)
    full = sw.root_depth_from_canopy(0.95, 6, 36)
    assert early < mid < full == 36.0


def test_a_stressed_crop_with_less_canopy_roots_shallower():
    """This is why the linear model misleads under deficit irrigation."""
    stressed = sw.root_depth_from_canopy(0.35, 6, 36)
    healthy = sw.root_depth_from_canopy(0.70, 6, 36)
    assert healthy - stressed > 5


def test_irrigation_systems_cover_furrow_sprinkler_and_micro():
    """Activity 6 names all three explicitly."""
    codes = {s.swat_code for s in sw.SYSTEMS.values()}
    assert {"furrow", "sprinkler", "drip"} <= codes


def test_furrow_loses_more_to_runoff_than_drip():
    assert sw.get_system("furrow").runoff_ratio > sw.get_system("micro_drip").runoff_ratio


def test_regimes_cover_full_deficit_and_variable_rate():
    assert sw.get_regime("full").et_fraction == 1.0
    assert sw.get_regime("deficit_70").et_fraction == 0.70
    assert sw.get_regime("full_vri").variable_rate
    assert sw.get_regime("deficit_50").et_fraction == 0.50   # matches NMSU trial


def test_field_process_model_returns_every_eq1_flux_term():
    m = sw.FieldProcessModel(74, 4.0, sw.AquiferState(water_table_depth_in=30.0))
    d0 = date(2026, 6, 1)
    for i in range(40):
        m.step(d0 + timedelta(days=i), rain_in=1.2 if i % 8 == 0 else 0.0,
               irrigation_in=0.8 if i % 7 == 3 else 0.0,
               depletion_in=1.2, taw_in=2.5, root_depth_in=24,
               system=sw.get_system("sprinkler_pivot"))
    totals = m.season_totals()
    for key in ("runoff_qs_in", "groundwater_loss_qw_in", "upflux_U_in"):
        assert key in totals


def test_nitrate_leaches_with_percolation_and_phosphorus_with_runoff():
    """The Delmarva split: nitrogen is a leaching problem, phosphorus a runoff one."""
    m = sw.FieldProcessModel(86, 0.2)     # heavy soil, high runoff
    m.step(date(2026, 6, 1), rain_in=3.0, irrigation_in=0.0, depletion_in=0.1,
           taw_in=4.0, root_depth_in=24, system=sw.get_system("sprinkler_pivot"))
    assert m.history[0].soluble_p_lost_lb > 0

    s = sw.FieldProcessModel(68, 4.5)     # sand, high percolation
    s.step(date(2026, 6, 1), rain_in=3.0, irrigation_in=0.0, depletion_in=0.1,
           taw_in=2.0, root_depth_in=24, system=sw.get_system("sprinkler_pivot"))
    assert s.history[0].nitrate_leached_lb > 0


def test_irrigation_water_nitrate_is_credited_into_the_soil_pool():
    m = sw.FieldProcessModel(74, 2.0, irrigation_nitrate_ppm=12.0)
    before = m.soil_nitrate_lb
    m.step(date(2026, 7, 1), rain_in=0.0, irrigation_in=1.0, depletion_in=1.0,
           taw_in=2.5, root_depth_in=24, system=sw.get_system("sprinkler_pivot"))
    assert m.soil_nitrate_lb > before - 1


def test_salinity_penalty_is_zero_below_the_crop_threshold():
    thresh, slope = sw.SALT_TOLERANCE["soybean_full"]
    assert sw.salinity_yield_penalty(0.3, thresh, slope)["yield_loss_pct"] == 0.0


def test_salinity_penalty_grows_above_the_threshold():
    thresh, slope = sw.SALT_TOLERANCE["corn_grain"]
    mild = sw.salinity_yield_penalty(1.0, thresh, slope)["yield_loss_pct"]
    harsh = sw.salinity_yield_penalty(3.0, thresh, slope)["yield_loss_pct"]
    assert harsh > mild


def test_deficit_irrigation_costs_soil_carbon():
    """Activity 6 links limited irrigation to soil health decline."""
    full = sw.simulate_soil_health(6, "full", 4.5, cover_crop=False)
    deficit = sw.simulate_soil_health(6, "deficit_70", 3.0, cover_crop=False)
    assert deficit["change_om_pct"] < full["change_om_pct"]
    assert deficit["change_om_pct"] < 0


def test_a_cover_crop_offsets_the_deficit_penalty():
    bare = sw.simulate_soil_health(6, "deficit_70", 3.0, cover_crop=False)
    covered = sw.simulate_soil_health(6, "deficit_70", 3.0, cover_crop=True)
    assert covered["change_om_pct"] > bare["change_om_pct"]


def test_soil_health_indicators_move_with_organic_matter():
    poor = sw.simulate_soil_health(8, "deficit_50", 2.0, False, "conventional")
    good = sw.simulate_soil_health(8, "full", 5.0, True, "no_till")
    assert good["indicators"]["bulk_density_g_cm3"] < poor["indicators"]["bulk_density_g_cm3"]
    assert good["indicators"]["infiltration_rate_in_hr"] > poor["indicators"]["infiltration_rate_in_hr"]


def test_minimum_data_set_drops_redundant_indicators():
    """Activity 6: identify the MDS by multivariate analysis."""
    n = 30
    base = [math.sin(i / 3.0) for i in range(n)]
    measurements = {
        "pox_carbon": [b * 100 + 350 for b in base],
        "microbial_biomass": [b * 98 + 348 for b in base],     # nearly identical
        "aggregate_stability": [b * 90 + 340 for b in base],   # nearly identical
        "bulk_density": [-b * 0.1 + 1.5 for b in base],
        "infiltration": [math.cos(i / 4.0) * 0.5 + 2.0 for i in range(n)],
    }
    target = [b * 2 + 5 for b in base]
    out = ingest_mds(measurements, target)
    assert len(out["minimum_data_set"]) < len(measurements)
    assert out["dropped_as_redundant"]


def ingest_mds(measurements, target):
    return sw.minimum_data_set(measurements, target)


def test_swatplus_writers_produce_the_files_activity_6_names(tmp_path):
    proj = sw.SwatPlusProject(tmp_path, field_name="home_pivot")
    schedule = {date(2026, 7, 1): 0.8, date(2026, 7, 9): 0.9}
    written = proj.write_all(schedule, sw.get_system("sprinkler_pivot"),
                             sw.get_regime("full_vri"), acres=132,
                             capacity_gpm=800)
    assert set(written) == {"irr.ops", "lum.dtl", "hru.con"}
    ops = (tmp_path / "irr.ops").read_text()
    # 0.8 net inches at 85 percent efficiency is 23.9 mm gross.
    assert "23.9" in ops
    dtl = (tmp_path / "lum.dtl").read_text()
    assert "jday" in dtl and "irrigate" in dtl


def test_swat_run_degrades_clearly_when_the_binary_is_absent(tmp_path):
    proj = sw.SwatPlusProject(tmp_path, executable="/nonexistent/swatplus")
    out = proj.run()
    assert out["ok"] is False
    assert "executable" in out["reason"]


# ===========================================================================
# Activity 9 — NSGA-II
# ===========================================================================

def test_pareto_dominance():
    assert opt.dominates((1, 1), (2, 2))
    assert not opt.dominates((1, 3), (3, 1))
    assert not opt.dominates((1, 1), (1, 1))


def test_fast_non_dominated_sort_finds_the_true_front():
    objs = [(1, 5), (2, 3), (3, 1), (2, 6), (4, 4)]
    fronts = opt.fast_non_dominated_sort(objs)
    assert set(fronts[0]) == {0, 1, 2}


def test_crowding_distance_puts_the_extremes_at_infinity():
    objs = [(1, 5), (2, 3), (3, 1)]
    d = opt.crowding_distance(objs, [0, 1, 2])
    assert d[0] == math.inf and d[2] == math.inf
    assert math.isfinite(d[1])


def test_optimizer_returns_a_genuine_tradeoff_curve():
    ctx = opt.FieldContext(weather=season(), soil=get_soil("evesboro"),
                           crop_key="corn_grain", planting=PLANTING,
                           lat_deg=LAT, acres=132, yield_goal_bu=210)
    res = opt.optimize(ctx, opt.DecisionSpace(), population=20, generations=6)
    assert len(res.front) >= 3
    water = [o.irrigation_in for o in res.front]
    returns = [o.net_return for o in res.front]
    assert max(water) - min(water) > 1.0
    assert max(returns) - min(returns) > 20


def test_no_solution_on_the_front_dominates_another():
    ctx = opt.FieldContext(weather=season(), soil=get_soil("sassafras"),
                           crop_key="corn_grain", planting=PLANTING,
                           lat_deg=LAT, yield_goal_bu=200)
    res = opt.optimize(ctx, opt.DecisionSpace(), population=16, generations=4)
    objs = [o.objectives() for o in res.front]
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i != j:
                assert not opt.dominates(a, b)


def test_allocation_limit_is_enforced_as_a_hard_constraint():
    ctx = opt.FieldContext(weather=season(), soil=get_soil("evesboro"),
                           crop_key="corn_grain", planting=PLANTING,
                           lat_deg=LAT, yield_goal_bu=210)
    res = opt.optimize(ctx, opt.DecisionSpace(allocation_ac_in=6.0),
                       population=20, generations=6)
    assert all(o.irrigation_in <= 6.0 + 1e-6 for o in res.front)


def test_sand_leaches_more_than_silt_loam_at_the_same_management():
    """Emerges from the hydrology, not from a hard-coded rule."""
    strategy = opt.Strategy(trigger_fraction=0.55, depth_in=0.8,
                            n_rate_lb=180, splits=3)
    sand = opt.evaluate(strategy, opt.FieldContext(
        weather=season(), soil=get_soil("evesboro"), crop_key="corn_grain",
        planting=PLANTING, lat_deg=LAT, yield_goal_bu=210))
    loam = opt.evaluate(strategy, opt.FieldContext(
        weather=season(), soil=get_soil("matapeake"), crop_key="corn_grain",
        planting=PLANTING, lat_deg=LAT, yield_goal_bu=210))
    assert sand.nitrate_leached_lb > loam.nitrate_leached_lb


def test_splitting_nitrogen_reduces_leaching_at_the_same_rate():
    """The core Delaware nitrogen recommendation must be visible to the optimizer."""
    ctx = opt.FieldContext(weather=season(), soil=get_soil("evesboro"),
                           crop_key="corn_grain", planting=PLANTING,
                           lat_deg=LAT, yield_goal_bu=210)
    two = opt.evaluate(opt.Strategy(0.55, 0.8, 200, 2), ctx)
    five = opt.evaluate(opt.Strategy(0.55, 0.8, 200, 5), ctx)
    assert five.nitrate_leached_lb < two.nitrate_leached_lb


def test_optimization_is_reproducible():
    ctx = opt.FieldContext(weather=season(90), soil=get_soil("evesboro"),
                           crop_key="corn_grain", planting=PLANTING, lat_deg=LAT)
    a = opt.optimize(ctx, population=12, generations=3, seed=42)
    b = opt.optimize(ctx, population=12, generations=3, seed=42)
    assert [o.net_return for o in a.front] == [o.net_return for o in b.front]


def test_current_practice_is_located_against_the_front():
    ctx = opt.FieldContext(weather=season(), soil=get_soil("evesboro"),
                           crop_key="corn_grain", planting=PLANTING,
                           lat_deg=LAT, yield_goal_bu=210)
    res = opt.optimize(ctx, population=20, generations=6)
    poor = opt.evaluate(opt.Strategy(0.95, 1.2, 250, 1), ctx)
    out = opt.compare_to_front(poor, res.front)
    assert "on_front" in out and "opportunities" in out


def test_front_summary_reports_the_corners_and_the_knee():
    ctx = opt.FieldContext(weather=season(), soil=get_soil("evesboro"),
                           crop_key="corn_grain", planting=PLANTING,
                           lat_deg=LAT, yield_goal_bu=210)
    s = opt.front_summary(opt.optimize(ctx, population=20, generations=6))
    for key in ("most_profitable", "least_water", "least_nitrate", "balanced"):
        assert key in s


# ===========================================================================
# Activity 2 — computational grids and VRI
# ===========================================================================

def test_grid_matches_the_5ft_by_15ft_specification():
    """The narrative gives this exact figure for VRI systems."""
    spec = vri.GRID_SPECS["vri_individual"]
    assert spec.along_span_ft == 5.0
    assert spec.along_travel_ft == 15.0


def test_a_132_acre_pivot_discretizes_to_tens_of_thousands_of_cells():
    g = vri.discretize(pivot(132), "vri_individual")
    assert 40_000 < g.n_cells < 120_000


def test_cell_areas_sum_to_the_field_acreage():
    g = vri.discretize(pivot(132), "vri_bank")
    assert sum(c.area_ac for c in g.cells) == pytest.approx(g.acres, rel=0.001)


def test_speed_control_machines_get_sectors_and_a_warning():
    g = vri.discretize(pivot(132), "vri_speed_only")
    assert g.n_cells < 1000
    assert any("along the span" in w for w in g.warnings)


def test_response_units_collapse_identical_cells():
    """Tens of thousands of cells, a handful of distinct behaviours."""
    g = vri.discretize(pivot(132), "vri_individual")
    vri.attach_soils(g, [{"series_key": "evesboro", "fraction": 0.6},
                         {"series_key": "fallsington", "fraction": 0.4}], get_soil)
    vri.build_response_units(g)
    assert g.response_units <= 6
    assert g.n_cells > 10_000


def test_soil_areas_follow_the_ssurgo_fractions():
    g = vri.discretize(pivot(132), "vri_bank")
    vri.attach_soils(g, [{"series_key": "evesboro", "fraction": 0.6},
                         {"series_key": "matapeake", "fraction": 0.4}], get_soil)
    acres = {}
    for c in g.cells:
        acres[c.soil_key] = acres.get(c.soil_key, 0) + c.area_ac
    assert acres["evesboro"] / g.acres == pytest.approx(0.6, abs=0.05)


def test_prescription_never_exceeds_what_a_cell_can_hold():
    g = vri.discretize(pivot(132), "vri_bank")
    vri.attach_soils(g, [{"series_key": "evesboro", "fraction": 1.0}], get_soil)
    vri.build_response_units(g)
    taw = {u: 2.1 for u in {c.response_unit for c in g.cells}}
    dep = {u: 5.0 for u in taw}          # absurdly dry
    rx = vri.build_prescription(g, date(2026, 7, 20), dep, taw, 0.45, 0.25, 1.2)
    assert rx.max_rate_in <= 0.90 * 2.1 + 1e-6


def test_prescription_varies_across_a_split_field():
    g = vri.discretize(pivot(132), "vri_bank")
    vri.attach_soils(g, [{"series_key": "evesboro", "fraction": 0.5},
                         {"series_key": "matapeake", "fraction": 0.5}], get_soil)
    vri.build_response_units(g)
    taw, dep = {}, {}
    for c in g.cells:
        t = c.awc_in_per_in * 30
        taw[c.response_unit] = t
        dep[c.response_unit] = t * (0.70 if c.soil_key == "evesboro" else 0.35)
    rx = vri.build_prescription(g, date(2026, 7, 20), dep, taw, 0.45, 0.25, 1.2)
    assert rx.n_rates > 1
    assert rx.savings_ac_in > 0


def test_a_uniform_field_is_told_to_run_uniform():
    g = vri.discretize(pivot(132), "vri_bank")
    vri.attach_soils(g, [{"series_key": "sassafras", "fraction": 1.0}], get_soil)
    vri.build_response_units(g)
    taw = {u: 3.6 for u in {c.response_unit for c in g.cells}}
    dep = {u: 2.4 for u in taw}
    rx = vri.build_prescription(g, date(2026, 7, 20), dep, taw, 0.45, 0.25, 1.2)
    assert not rx.worth_it
    assert "uniform" in rx.rationale


def test_vri_payback_declines_to_recommend_hardware_on_a_uniform_field():
    g = vri.discretize(pivot(132), "vri_bank")
    vri.attach_soils(g, [{"series_key": "sassafras", "fraction": 1.0}], get_soil)
    out = vri.vri_payback(g, 2.0, 132)
    assert "uniform" in out["verdict"]


def test_prescription_exports_as_point_features_with_rates():
    g = vri.discretize(pivot(132), "vri_speed_only")
    vri.attach_soils(g, [{"series_key": "evesboro", "fraction": 1.0}], get_soil)
    vri.build_response_units(g)
    taw = {u: 2.1 for u in {c.response_unit for c in g.cells}}
    rx = vri.build_prescription(g, date(2026, 7, 20), {u: 1.6 for u in taw},
                                taw, 0.45, 0.25, 1.0)
    features = vri.prescription_to_shapefile_wkt(rx)
    assert features and features[0]["wkt"].startswith("POINT(")
    assert "RATE_IN" in features[0]


# ===========================================================================
# Activity 3 — canopy temperature indices and data formats
# ===========================================================================

def test_dans_is_zero_for_a_canopy_matching_the_well_watered_reference():
    assert dans_index(88.0, 88.0) == 0.0
    assert ks_from_dans(0.0) == 1.0


def test_ks_falls_as_the_canopy_runs_hotter():
    assert ks_from_dans(2.0) > ks_from_dans(5.0) > ks_from_dans(8.0)
    assert ks_from_dans(12.0) == 0.0


def test_dans_clips_a_bare_soil_pixel():
    """A pixel 30 F above air is soil, not a transpiring canopy."""
    assert dans_index(120.0, 88.0, air_temp_f=90.0) < 120.0 - 88.0


def test_dans_from_an_image_calibrates_against_its_own_cool_tail():
    cool = [86.0 + i * 0.01 for i in range(400)]
    hot = [95.0 + i * 0.01 for i in range(120)]
    out = dans_from_thermal_image(cool + hot)
    assert out["reference_temp_f"] == pytest.approx(86.0, abs=0.5)
    assert 0 < out["fraction_stressed"] < 1


def test_both_stress_indices_the_narrative_names_are_available():
    """Activity 3: "such as Crop Water Stress Index (CWSI) and DANS"."""
    assert ks_from_cwsi(0.0) == 1.0
    assert ks_from_dans(0.0) == 1.0
    assert ks_from_cwsi(1.0) == 0.0


def test_ascii_sensor_export_parses_vendor_columns():
    text = ("Timestamp,VWC_4,VWC_12,VWC_30,SoilTemp_C\n"
            "2026-07-01 06:00:00,12.4,15.1,19.0,24.5\n"
            "2026-07-02 06:00:00,11.1,14.6,19.0,25.1\n")
    readings = ingest.read_sensor_ascii(text)
    assert len(readings) == 6
    assert {r.depth_in for r in readings} == {4.0, 12.0, 30.0}


def test_percent_moisture_is_converted_to_a_fraction():
    text = "date,vwc_12\n2026-07-01,18.5\n"
    assert ingest.read_sensor_ascii(text)[0].vwc == pytest.approx(0.185)


def test_centimetre_depths_are_recognised():
    text = "date,swc_10cm,swc_60cm\n2026-07-01,0.18,0.22\n"
    depths = {r.depth_in for r in ingest.read_sensor_ascii(text)}
    assert max(depths) < 30       # 60 cm is under 24 inches


def test_a_file_without_moisture_columns_says_so():
    with pytest.raises(ValueError, match="depth"):
        ingest.read_sensor_ascii("date,temperature\n2026-07-01,22.5\n")


def test_netcdf_reader_identifies_a_non_netcdf_file():
    assert ingest.read_sensor_netcdf(b"just some text")["ok"] is False


def test_netcdf4_file_asks_for_the_library_rather_than_guessing():
    out = ingest.read_sensor_netcdf(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert out["ok"] is False
    assert "netCDF4" in out.get("hint", "") or "netcdf4" in out["reason"]


def test_weather_upload_accepts_a_growers_own_station():
    text = ("date,Tmax,Tmin,RHmax,RHmin,WindSpeed,Solar,Precip\n"
            "2026-07-01,92.8,71.1,95,48,5.4,24.1,0\n"
            "2026-07-02,88.0,68.4,96,52,4.0,21.5,0.5\n")
    out = ingest.read_weather_csv(text)
    assert out["ok"] and out["n_days"] == 2
    assert out["has_solar_radiation"]


def test_celsius_weather_is_converted():
    text = "date,Tmax,Tmin\n2026-07-01,32.8,21.1\n"
    out = ingest.read_weather_csv(text)
    assert out["days"][0]["tmax_f"] == pytest.approx(91.04, abs=0.1)


def test_weather_upload_without_temperature_is_refused():
    out = ingest.read_weather_csv("date,rain\n2026-07-01,0.5\n")
    assert out["ok"] is False
    assert "hint" in out


def test_utm_converts_correctly_at_the_central_meridian():
    """Zone 18 covers Delaware; its central meridian is exactly -75."""
    lon, lat = ingest.utm_to_wgs84(500_000.0, 4_283_000.0, zone=18)
    assert lon == pytest.approx(-75.0, abs=1e-6)
    assert 38.0 < lat < 39.5


def test_yield_features_normalise_to_an_index():
    feats = [{"geometry": {"coordinates": [-75.47 + i * 1e-4, 38.64]},
              "properties": {"Yld_Vol_Dr": 150 + i * 3}} for i in range(24)]
    out = ingest.yield_from_features(feats)
    assert out["n"] == 24
    assert 0.5 < min(v for _, _, v in out["points"]) < 1.0
    assert 1.0 < max(v for _, _, v in out["points"]) < 1.5


def test_a_uniform_yield_map_says_variable_rate_has_nothing_to_do():
    feats = [{"geometry": {"coordinates": [-75.47, 38.64]},
              "properties": {"yield": 200.0}} for _ in range(20)]
    assert "evenly" in ingest.yield_from_features(feats)["note"]


# ===========================================================================
# Activity 7 — sensor placement depth
# ===========================================================================

def _profiles(n: int = 60):
    """Synthetic profile records with realistic partial independence.

    Real profiles are not perfectly correlated with depth: the surface dries and
    rewets far faster than 30 inches down, which is the entire reason more than
    one sensor is worth installing. Each depth gets its own noise and its own
    responsiveness to the wetting signal.
    """
    import random
    random.seed(3)
    out = []
    for _ in range(n):
        wet = random.uniform(0.05, 0.20)
        surface_swing = random.gauss(0, 0.035)
        out.append({
            6: max(0.02, wet - 0.03 + surface_swing + random.gauss(0, 0.010)),
            12: max(0.02, wet + 0.5 * surface_swing + random.gauss(0, 0.008)),
            18: max(0.02, wet + 0.01 + 0.2 * surface_swing + random.gauss(0, 0.007)),
            24: max(0.02, wet + 0.015 + random.gauss(0, 0.006)),
            36: max(0.02, wet + 0.02 + random.gauss(0, 0.009)),
            48: 0.19 + random.gauss(0, 0.004),
        })
    return out


def test_mlr_selects_depths_spanning_the_profile():
    """Activity 7's universal multiple linear regression for placement depth."""
    out = optimal_sensor_depths(_profiles(), n_sensors=3, root_depth_in=36)
    depths = out["recommended_depths_in"]
    assert len(depths) == 3
    assert max(depths) - min(depths) >= 18      # spread, not clustered


def test_more_sensors_explain_more_variance():
    one = optimal_sensor_depths(_profiles(), n_sensors=1, root_depth_in=36)
    three = optimal_sensor_depths(_profiles(), n_sensors=3, root_depth_in=36)
    assert three["r2"] >= one["r2"]


def test_depths_below_a_restrictive_layer_are_dropped():
    """A probe under a fragipan reads water the crop cannot reach."""
    out = optimal_sensor_depths(_profiles(), n_sensors=2, root_depth_in=36,
                                restrictive_layer_in=20)
    assert all(d <= 20 for d in out["recommended_depths_in"])
    assert 36 in out["depths_dropped_below_restrictive_layer"]


def test_depth_recommendation_explains_itself():
    out = optimal_sensor_depths(_profiles(), n_sensors=3, root_depth_in=36)
    assert "Put the sensors at" in out["note"]
    assert out["single_depth_r2"]


# ---------------------------------------------------------------------------
# Eq 1 closure — D = D0 + ETc - P - Irr - U + qs + qw (+ root growth)
# ---------------------------------------------------------------------------

def test_eq1_closes_exactly_including_root_growth():
    """The narrative's Eq 1 must reconcile term by term from DayResult alone.

    A user auditing an advisory has to be able to add the reported fluxes back
    up and land on the reported deficit. The root-growth term is not in Eq 1 as
    printed in the narrative because Eq 1 assumes a fixed root zone; Activity 6
    requires dynamic rooting, so the term is reported explicitly rather than
    folded silently into D.
    """
    from datetime import date
    from app.engine import fao56
    from app.engine.crops import CROPS
    from app.engine.soils import SOILS

    crop, soil = CROPS["corn_grain"], SOILS["evesboro"]
    cases = [
        # (dap, root0, dr0, rain, irr) — spans growth, wet, dry, and saturated
        (83, 30.0, 1.20, 0.85, 0.30),
        (55, 18.0, 0.60, 0.00, 0.00),
        (40, 14.0, 0.10, 2.50, 0.00),   # forces deep percolation
        (95, 36.0, 2.00, 0.00, 1.00),
    ]
    for dap, root0, dr0, rain, irr in cases:
        state = fao56.BalanceState(dr_in=dr0, de_in=0.20,
                                   root_depth_in=root0, last_root_depth_in=root0)
        wx = fao56.WeatherDay(day=date(2026, 7, 20), tmax_f=91, tmin_f=70,
                              rh_max=92, rh_min=48, wind_mph=6.5,
                              solar_rad_mj_m2=24.0, rain_in=rain)
        r = fao56.step_water_balance(state=state, weather=wx, crop=crop, soil=soil,
                                     dap=dap, lat_deg=38.7, irrigation_in=irr,
                                     curve_number=78)
        rhs = (dr0 + r.root_growth_in + r.etc_adj_in
               - r.rain_in - r.irrigation_in - r.upflux_in
               + r.runoff_in + r.deep_perc_in)
        assert abs(r.depletion_in - rhs) < 5e-3, (
            f"Eq 1 failed to close at dap={dap}: "
            f"D={r.depletion_in} vs {rhs}")


def test_eq1_deep_percolation_is_the_overflow_not_an_input():
    """qw must appear only when the profile cannot hold the water."""
    from datetime import date
    from app.engine import fao56
    from app.engine.crops import CROPS
    from app.engine.soils import SOILS

    crop, soil = CROPS["corn_grain"], SOILS["evesboro"]
    dry = fao56.BalanceState(dr_in=2.00, de_in=0.5, root_depth_in=30.0,
                             last_root_depth_in=30.0)
    wet = fao56.BalanceState(dr_in=0.05, de_in=0.0, root_depth_in=30.0,
                             last_root_depth_in=30.0)
    wx = fao56.WeatherDay(day=date(2026, 7, 20), tmax_f=88, tmin_f=68,
                          rh_max=90, rh_min=50, wind_mph=5.0,
                          solar_rad_mj_m2=22.0, rain_in=1.50)
    r_dry = fao56.step_water_balance(state=dry, weather=wx, crop=crop, soil=soil,
                                     dap=83, lat_deg=38.7, curve_number=78)
    r_wet = fao56.step_water_balance(state=wet, weather=wx, crop=crop, soil=soil,
                                     dap=83, lat_deg=38.7, curve_number=78)
    assert r_dry.deep_perc_in == 0.0
    assert r_wet.deep_perc_in > 0.0


# ---------------------------------------------------------------------------
# Activity 6 — the SWAT+ input files have to be machine readable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name", [
    "sussex_pivot",
    "sussex_center_pivot_north_40",   # long enough to overrun the name column
])
def test_swatplus_input_files_tokenize_to_stable_columns(tmp_path, field_name):
    """SWAT+ reads these whitespace-delimited, so columns must never merge.

    A right-aligned number written flush against a left-aligned string yields
    '53.419hru', which the Fortran read parses as a single token and then fails
    on. Every data row must produce the same field count as its header.
    """
    proj = sw.SwatPlusProject(tmp_path, field_name)
    schedule = {date(2026, 7, 10): 0.9, date(2026, 7, 18): 1.1,
                date(2026, 7, 27): 0.8}
    written = proj.write_all(schedule=schedule,
                             system=sw.SYSTEMS["sprinkler_pivot"],
                             regime=sw.REGIMES["full"],
                             acres=132.0, capacity_gpm=900.0)

    expected_cols = {"irr.ops": 6, "hru.con": 7}
    for name, count in expected_cols.items():
        lines = pathlib.Path(written[name]).read_text().splitlines()
        header, *rows = lines[1:]          # line 0 is the free-text title
        assert len(header.split()) == count, f"{name} header: {header!r}"
        for row in rows:
            assert len(row.split()) == count, f"{name} row merged: {row!r}"

    # lum.dtl action rows carry a trailing 'act=' clause, so check the fixed part
    dtl = pathlib.Path(written["lum.dtl"]).read_text().splitlines()
    for row in dtl[5:]:
        fixed = row.split("act=")[0]
        assert len(fixed.split()) == 6, f"lum.dtl row merged: {row!r}"
        assert "null" in fixed.split(), "the lim_var token merged with obj_num"


def test_swatplus_numbers_survive_a_round_trip_through_the_writer(tmp_path):
    """The amounts SWAT+ reads must be the gross depths pyFAO56 implied."""
    proj = sw.SwatPlusProject(tmp_path, "roundtrip")
    system = sw.SYSTEMS["sprinkler_pivot"]
    schedule = {date(2026, 7, 10): 1.00}
    written = proj.write_all(schedule=schedule, system=system,
                             regime=sw.REGIMES["full"],
                             acres=100.0, capacity_gpm=800.0)
    row = pathlib.Path(written["irr.ops"]).read_text().splitlines()[2].split()
    gross_mm = float(row[1])
    expected = 1.00 * 25.4 / system.application_efficiency
    assert abs(gross_mm - expected) < 0.01


# ---------------------------------------------------------------------------
# Eq 6 — Water-Nutrient Intensification Performance Index (Activity 8, 11)
# ---------------------------------------------------------------------------

def test_wnipi_matches_the_narrative_formula_by_hand():
    """Eq 6, computed independently and compared.

        WNIPI = (Y/Y0 - 1) / [(1 + Irr/ET0)(1 + N/ANU0)(1 + P/APU0)]
    """
    from app.engine.nitrogen import water_nutrient_index

    y, y0 = 232.0, 200.0
    irr, et0 = 6.5, 24.0
    n, anu0 = 180.0, 160.0
    p, apu0 = 60.0, 55.0

    expected = ((y / y0) - 1.0) / (
        (1.0 + irr / et0) * (1.0 + n / anu0) * (1.0 + p / apu0))
    got = water_nutrient_index(yield_farm=y, yield_baseline=y0,
                               irrigation_in=irr, eto_in=et0,
                               n_applied=n, n_uptake_baseline=anu0,
                               p_applied=p, p_uptake_baseline=apu0)
    assert got == pytest.approx(expected, abs=1e-4)


def test_wnipi_is_zero_when_yield_only_matches_the_baseline():
    """No intensification gain means no score, however much was applied."""
    from app.engine.nitrogen import water_nutrient_index
    assert water_nutrient_index(200.0, 200.0, 6.0, 24.0,
                                180.0, 160.0, 60.0, 55.0) == 0.0


def test_wnipi_penalises_buying_the_same_yield_with_more_input():
    """The index has to prefer the cheaper route to an identical yield.

    This is the whole point of the metric for the TAPS competition: two teams
    hitting 230 bu should not score the same if one used half the water and
    nitrogen to get there.
    """
    from app.engine.nitrogen import water_nutrient_index

    lean = water_nutrient_index(230.0, 200.0, 4.0, 24.0, 150.0, 160.0, 50.0, 55.0)
    heavy = water_nutrient_index(230.0, 200.0, 12.0, 24.0, 240.0, 160.0, 90.0, 55.0)
    assert lean > heavy

    # and each input dimension penalises on its own
    base = dict(yield_farm=230.0, yield_baseline=200.0, irrigation_in=6.0,
                eto_in=24.0, n_applied=180.0, n_uptake_baseline=160.0,
                p_applied=60.0, p_uptake_baseline=55.0)
    ref = water_nutrient_index(**base)
    assert water_nutrient_index(**{**base, "irrigation_in": 12.0}) < ref
    assert water_nutrient_index(**{**base, "n_applied": 260.0}) < ref
    assert water_nutrient_index(**{**base, "p_applied": 120.0}) < ref


def test_wnipi_rises_with_yield_when_inputs_are_held_still():
    from app.engine.nitrogen import water_nutrient_index
    scores = [water_nutrient_index(y, 200.0, 6.0, 24.0, 180.0, 160.0, 60.0, 55.0)
              for y in (205.0, 215.0, 225.0, 240.0)]
    assert scores == sorted(scores)


def test_wnipi_goes_negative_when_intensification_loses_yield():
    """Deficit irrigation that costs more yield than it saves input scores below zero."""
    from app.engine.nitrogen import water_nutrient_index
    assert water_nutrient_index(180.0, 200.0, 6.0, 24.0,
                                180.0, 160.0, 60.0, 55.0) < 0


def test_wnipi_survives_a_zero_baseline_instead_of_dividing_by_it():
    from app.engine.nitrogen import water_nutrient_index
    assert water_nutrient_index(200.0, 0.0, 6.0, 24.0,
                                180.0, 160.0, 60.0, 55.0) == 0.0
    # a rain-fed season with no irrigation is a legitimate call, not a crash
    assert water_nutrient_index(230.0, 200.0, 0.0, 24.0,
                                180.0, 160.0, 60.0, 55.0) > 0


def test_wnipi_endpoint_ranks_two_teams_reaching_the_same_yield():
    """Activity 11: TAPS teams calculate their own index directly.

    Two teams both hit 230 bu against a 200 bu baseline. The one that bought it
    with less water and nitrogen has to rank higher, and the bands have to
    actually separate them rather than calling both 'marginal'.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    lean = dict(yield_farm=230, yield_baseline=200, irrigation_in=4, eto_in=24,
                n_applied=150, n_uptake_baseline=160,
                p_applied=50, p_uptake_baseline=55)
    heavy = {**lean, "irrigation_in": 12, "n_applied": 240, "p_applied": 90}

    a = client.post("/api/metrics/wnipi", json=lean).json()
    b = client.post("/api/metrics/wnipi", json=heavy).json()

    assert a["wnipi"] > b["wnipi"]
    assert a["yield_gain_pct"] == b["yield_gain_pct"] == 15.0
    assert a["denominator"] < b["denominator"]
    assert a["reading"] != b["reading"], "the bands failed to separate them"


def test_wnipi_endpoint_rejects_a_zero_baseline_rather_than_returning_zero():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    body = dict(yield_farm=230, yield_baseline=0, irrigation_in=4, eto_in=24,
                n_applied=150, n_uptake_baseline=160,
                p_applied=50, p_uptake_baseline=55)
    assert client.post("/api/metrics/wnipi", json=body).status_code == 422


def test_wnipi_endpoint_names_the_input_carrying_the_most_penalty():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    body = dict(yield_farm=230, yield_baseline=200, irrigation_in=2, eto_in=24,
                n_applied=320, n_uptake_baseline=160,
                p_applied=20, p_uptake_baseline=55)
    assert client.post("/api/metrics/wnipi", json=body).json()[
        "dominant_penalty"] == "nitrogen"
