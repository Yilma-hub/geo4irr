"""
Tests for the pyfao56 bridge.

pyfao56 is the reference implementation named in the project narrative — Thorp
(2022), USDA-ARS, validated against Maricopa lysimeter records and the Ref-ET
software. These tests check two separate things:

  that the translation into pyfao56's classes preserves the Delaware inputs, and
  that the two engines agree where they should.

They skip cleanly when pyfao56 is not installed, because the rest of the system
runs without it and a missing optional dependency should not look like a broken
build.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from app.engine import pyfao56_bridge as bridge
from app.engine.crops import get_crop
from app.engine.fao56 import WeatherDay, f_to_c, mph_to_ms, reference_et
from app.engine.soils import get_soil

pytestmark = pytest.mark.skipif(
    not bridge.PYFAO56_AVAILABLE,
    reason="pyfao56 not installed; pip install pyfao56",
)

PLANTING = date(2026, 4, 28)
LAT = 38.64


def season(days: int = 120, rain_every: int = 6) -> list[WeatherDay]:
    out = []
    for i in range(days):
        d = PLANTING + timedelta(days=i)
        t = math.sin((d.timetuple().tm_yday - 80) / 200 * math.pi)
        out.append(WeatherDay(day=d, tmax_f=68 + 24 * t, tmin_f=48 + 22 * t,
                              rh_max=93, rh_min=52, wind_mph=5.5,
                              solar_rad_mj_m2=12 + 16 * t,
                              rain_in=0.45 if i % rain_every == 0 else 0.0))
    return out


# ---------------------------------------------------------------------------
# Reference ET: the foundation
# ---------------------------------------------------------------------------

def test_reference_et_matches_pyfao56_within_a_tenth_of_a_percent():
    """The most important test in the repository.

    Both implementations claim to be ASCE standardized. If they disagree here,
    one of them is wrong and everything downstream inherits the error.
    """
    import pyfao56 as fao

    cases = [(91, 70, 92, 48, 24, 6.5), (75, 55, 95, 55, 18, 4.0),
             (95, 74, 88, 40, 27, 9.0), (60, 42, 90, 60, 12, 5.0),
             (100, 78, 85, 35, 29, 11.0)]
    for tmax, tmin, rhmax, rhmin, srad, wind in cases:
        w = WeatherDay(day=date(2026, 7, 15), tmax_f=tmax, tmin_f=tmin,
                       rh_max=rhmax, rh_min=rhmin, wind_mph=wind,
                       solar_rad_mj_m2=srad)
        ours = reference_et(w, LAT)
        theirs = fao.refet.ascedaily(
            rfcrp="S", z=12.0, lat=LAT, doy=196, israd=srad,
            tmax=f_to_c(tmax), tmin=f_to_c(tmin), rhmax=rhmax, rhmin=rhmin,
            wndsp=mph_to_ms(wind), wndht=2.0) / 25.4
        assert ours == pytest.approx(theirs, rel=0.001)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def test_parameters_carry_the_crop_curve_across():
    crop = get_crop("corn_grain")
    par = bridge.build_parameters(crop, get_soil("evesboro"))
    assert par.Kcbmid == crop.kcb_mid
    assert par.Lini == crop.l_ini
    assert par.Ldev == crop.l_dev


def test_parameters_use_the_measured_water_limits():
    soil = get_soil("evesboro")
    par = bridge.build_parameters(crop=get_crop("corn_grain"), soil=soil)
    assert par.thetaFC == pytest.approx(soil.fc_vol, abs=1e-4)
    assert par.thetaWP == pytest.approx(soil.wp_vol, abs=1e-4)


def test_available_water_survives_the_translation():
    # If AWC does not come through, TAW is wrong and so is every recommendation.
    for key in ("evesboro", "sassafras", "matapeake"):
        soil = get_soil(key)
        par = bridge.build_parameters(get_crop("corn_grain"), soil)
        assert par.thetaFC - par.thetaWP == pytest.approx(
            soil.awc_in_per_in, abs=0.005), key


def test_restrictive_layer_caps_the_rooting_depth():
    crop = get_crop("corn_grain")
    deep = bridge.build_parameters(crop, get_soil("sassafras"))
    shallow_soil = get_soil("sassafras")
    shallow_soil.restrictive_layer_in = 18.0
    shallow = bridge.build_parameters(crop, shallow_soil)
    assert shallow.Zrmax < deep.Zrmax


def test_weather_frame_is_indexed_the_way_pyfao56_expects():
    wth = bridge.build_weather(season(10), LAT)
    assert list(wth.wdata.columns) == wth.cnames
    assert all("-" in str(i) and len(str(i)) == 8 for i in wth.wdata.index)
    assert wth.rfcrp == "S"          # short crop, grass reference


def test_rain_is_converted_to_millimetres():
    day = WeatherDay(day=date(2026, 6, 1), tmax_f=80, tmin_f=60, rain_in=1.0)
    wth = bridge.build_weather([day], LAT)
    assert wth.wdata.iloc[0]["Rain"] == pytest.approx(25.4, abs=0.01)


def test_provider_eto_passes_through_untouched():
    # A station's own published ETo is the number a grower sees quoted
    # elsewhere. Recomputing it and disagreeing is a support burden.
    day = WeatherDay(day=date(2026, 6, 1), tmax_f=85, tmin_f=64, eto_in=0.25)
    wth = bridge.build_weather([day], LAT)
    assert wth.wdata.iloc[0]["ETref"] == pytest.approx(6.35, abs=0.01)


def test_irrigation_log_converts_to_millimetres():
    irr = bridge.build_irrigation({date(2026, 7, 1): 1.0, date(2026, 7, 9): 0.5})
    depths = sorted(irr.idata["Depth"].tolist())
    assert depths == pytest.approx([12.7, 25.4], abs=0.05)


def test_zero_depth_events_are_not_logged():
    irr = bridge.build_irrigation({date(2026, 7, 1): 0.0})
    assert len(irr.idata) == 0


def test_updates_carry_assimilated_kcb():
    # The Update class is the door pyfao56 provides for remote sensing. Thorp
    # et al. used it for sUAS fractional cover; here it carries the Kcb the
    # Bayesian filter settled on after seeing Sentinel-2.
    upd = bridge.build_updates(kcb_by_day={date(2026, 7, 4): 1.24})
    assert len(upd.udata) == 1
    assert upd.udata.iloc[0]["Kcb"] == pytest.approx(1.24)
    assert list(upd.udata.columns) == ["Kcb", "h", "fc"]


def test_empty_updates_produce_an_empty_frame():
    assert len(bridge.build_updates().udata) == 0


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def test_season_runs_and_returns_our_own_record_type():
    run = bridge.run_season_pyfao56(season(), get_crop("corn_grain"),
                                    get_soil("evesboro"), PLANTING, lat_deg=LAT)
    assert run.engine == "pyfao56"
    assert len(run.results) == 120
    first = run.results[0]
    assert first.dap == 0
    assert 0 < first.eto_in < 0.5              # inches, not millimetres
    assert first.taw_in < 6                    # inches, not millimetres


def test_irrigation_reduces_end_of_season_depletion():
    args = (season(), get_crop("corn_grain"), get_soil("evesboro"), PLANTING)
    dry = bridge.run_season_pyfao56(*args, lat_deg=LAT).results[-1]
    wet = bridge.run_season_pyfao56(
        *args, lat_deg=LAT,
        irrigation={PLANTING + timedelta(days=k): 0.9 for k in range(45, 100, 7)}
    ).results[-1]
    assert wet.depletion_in < dry.depletion_in


def test_sand_holds_less_than_silt_loam_through_the_bridge():
    args = (season(), get_crop("corn_grain"))
    sand = bridge.run_season_pyfao56(*args, get_soil("evesboro"), PLANTING,
                                     lat_deg=LAT).results[-1]
    loam = bridge.run_season_pyfao56(*args, get_soil("matapeake"), PLANTING,
                                     lat_deg=LAT).results[-1]
    assert sand.taw_in < loam.taw_in


def test_water_table_soils_are_flagged_as_a_known_gap():
    # FAO-56 has no capillary rise term, so pyfao56 correctly does not model
    # upflux. On Fallsington and Pocomoke that matters, and the run must say so
    # rather than quietly reading too dry.
    run = bridge.run_season_pyfao56(season(), get_crop("corn_grain"),
                                    get_soil("pocomoke"), PLANTING, lat_deg=LAT)
    assert any("water table" in w for w in run.warnings)


def test_no_warning_for_a_soil_without_a_water_table():
    run = bridge.run_season_pyfao56(season(), get_crop("corn_grain"),
                                    get_soil("evesboro"), PLANTING, lat_deg=LAT)
    assert run.warnings == []


# ---------------------------------------------------------------------------
# Agreement between the engines
# ---------------------------------------------------------------------------

def test_engines_agree_on_reference_et_across_a_whole_season():
    out = bridge.compare_engines(season(), get_crop("corn_grain"),
                                 get_soil("evesboro"), PLANTING, lat_deg=LAT)
    assert out["stats"]["eto"]["max_abs"] < 0.005


def test_engines_agree_on_the_crop_coefficient_curve():
    out = bridge.compare_engines(season(), get_crop("corn_grain"),
                                 get_soil("evesboro"), PLANTING, lat_deg=LAT)
    assert out["stats"]["kcb"]["max_abs"] < 0.02


def test_season_crop_water_use_agrees_within_ten_percent():
    """A loose bound, deliberately.

    The engines differ on root growth — FAO-56 expands the root zone with crop
    development, the built-in engine ties it to canopy cover — so exact
    agreement is not expected and would be suspicious. What matters is that the
    difference stays inside a band where it is a modelling choice rather than a
    bug. Roughly one irrigation over a season is the tolerance.
    """
    out = bridge.compare_engines(season(), get_crop("corn_grain"),
                                 get_soil("evesboro"), PLANTING, lat_deg=LAT)
    assert abs(out["season"]["etc_difference_pct"]) < 10


def test_comparison_reports_every_variable_it_promises():
    out = bridge.compare_engines(season(30), get_crop("corn_grain"),
                                 get_soil("sassafras"), PLANTING, lat_deg=LAT)
    assert set(out["stats"]) == {"eto", "kcb", "ks", "etc_adj", "dr", "taw"}
    assert out["stats"]["eto"]["n"] == 30


def test_engine_choice_respects_the_environment(monkeypatch):
    monkeypatch.setenv("GEO4IRR_ET_ENGINE", "builtin")
    assert bridge.engine_choice() == "builtin"
    monkeypatch.setenv("GEO4IRR_ET_ENGINE", "pyfao56")
    assert bridge.engine_choice() == "pyfao56"
    monkeypatch.setenv("GEO4IRR_ET_ENGINE", "auto")
    assert bridge.engine_choice() == "pyfao56"
