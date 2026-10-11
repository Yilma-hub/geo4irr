"""
Delaware reference values from DIMS, and what they are allowed to touch.

DIMS is the published state reference, which makes it a useful independent
check on the Delaware soil table and a source for several things this build
was missing. It is also a different method, and the most important thing these
tests do is stop the two from being quietly merged: a single crop coefficient
dropped into a dual coefficient slot would count soil evaporation twice, and
nothing downstream would complain.
"""

from __future__ import annotations

import datetime
import math

import pytest

from app.engine import dims
from app.engine.crops import CROPS
from app.engine.fao56 import WeatherDay
from app.engine.soils import SOILS


# --------------------------------------------------------------------------
# the soil table, checked against DIMS
# --------------------------------------------------------------------------

def test_every_delaware_series_lands_on_a_sensible_texture_class():
    """An independent check on the soil table.

    The twelve series in this engine were built from SSURGO. DIMS publishes
    field capacity and wilting point per USDA texture class. If a series whose
    name says sandy loam matches to clay, one of the two is wrong, and it is
    worth knowing which before a schedule is built on it.
    """
    # Texture word in the series name, against the DIMS classes that are a
    # reasonable neighbour for it. Soil naming is a continuum, so an adjacent
    # class is fine and a jump across the triangle is not.
    acceptable = {
        "loamy sand": {"sand", "loamy_sand", "sandy_loam"},
        "sandy loam": {"loamy_sand", "sandy_loam", "loam", "sandy_clay_loam"},
        "silt loam": {"silt", "silt_loam", "silty_clay_loam"},
        "loam": {"sandy_loam", "loam", "silt_loam", "silt"},
    }
    for key, soil in SOILS.items():
        name = soil.series.lower()
        texture = next((t for t in ("loamy sand", "sandy loam", "silt loam", "loam")
                        if t in name), None)
        if texture is None:
            continue
        matched = dims.texture_from_fc_wp(soil.fc_vol, soil.wp_vol)
        assert matched.key in acceptable[texture], (
            f"{soil.series} (FC {soil.fc_vol}, WP {soil.wp_vol}) matched DIMS "
            f"class {matched.label!r}, which is not a neighbour of {texture!r}")


def test_matching_on_available_water_alone_is_not_a_texture_match():
    """The reason texture_from_fc_wp takes both terms.

    Clay holds 0.43 at field capacity with 0.29 still locked away at wilting
    point: 0.14 available, the same figure as a far lighter soil. Matched on
    available water alone this engine called Woodstown sandy loam a clay. The
    AWC-only helper is kept for the case where AWC is genuinely all that is
    known, so this records what it is and is not good for.
    """
    by_awc = dims.texture_from_awc(0.14)
    by_both = dims.texture_from_fc_wp(0.22, 0.09)   # Woodstown sandy loam
    assert by_awc.key == "clay"
    assert by_both.key == "sandy_loam"
    assert by_awc.key != by_both.key, (
        "if these ever agree, the example in the docstring is stale")


def test_the_texture_table_is_internally_consistent():
    for t in dims.TEXTURE_CLASSES.values():
        assert t.wp_vol < t.fc_vol, f"{t.label} has no available water"
        assert 0.0 < t.awc_in_per_in < 0.25, f"{t.label} AWC of {t.awc_in_per_in}"
    assert dims.TEXTURE_CLASSES["sand"].awc_in_per_in == pytest.approx(0.07)
    assert dims.TEXTURE_CLASSES["silt"].awc_in_per_in == pytest.approx(0.20)


def test_a_measured_soil_is_validated_rather_than_trusted():
    good = dims.custom_texture(0.26, 0.12)
    assert good.awc_in_per_in == pytest.approx(0.14)

    # The likeliest data-entry error, and the most damaging: swapped round, it
    # gives a negative available water capacity and a field that never irrigates.
    with pytest.raises(ValueError, match="swapped"):
        dims.custom_texture(0.12, 0.26)
    with pytest.raises(ValueError, match="[Ff]ield capacity"):
        dims.custom_texture(0.95, 0.10)


# --------------------------------------------------------------------------
# the two coefficient methods must stay apart
# --------------------------------------------------------------------------

def test_the_dims_single_kc_never_reaches_the_crop_table():
    """The one change that would break the engine silently.

    DIMS corn is Kc 0.25 / 1.20 / 0.60, a single coefficient covering
    transpiration and soil evaporation together. This engine is Kcb 0.15 /
    1.15 / 0.50, transpiration alone, with Ke added separately. Copy the first
    into the second and every day gains an evaporation term it already had.
    """
    for crop_key, (kc_ini, kc_mid, kc_end) in dims.DIMS_SINGLE_KC.items():
        crop = CROPS.get(crop_key)
        if crop is None:
            continue
        assert crop.kcb_ini != pytest.approx(kc_ini) or kc_ini == 0.15, (
            f"{crop_key} initial Kcb equals the DIMS single Kc of {kc_ini}, "
            "which suggests the two tables have been merged")
        assert crop.kcb_mid < kc_mid + 1e-9, (
            f"{crop_key} basal Kcb {crop.kcb_mid} is not below the DIMS single "
            f"Kc {kc_mid}; a basal coefficient excludes evaporation and must be "
            "the smaller of the two")


def test_the_two_methods_are_compared_potential_against_potential(simple_season):
    """Comparing stress-adjusted ET against unstressed ET is not a comparison.

    The first version of this reported a 15.8 percent gap between the methods
    that was almost entirely the stress coefficient. Both sides are potential
    now, and the stress reduction is reported separately so its size is visible
    rather than hidden inside a difference.
    """
    results, crop_key = simple_season
    out = dims.compare_with_dims_single_kc(results, crop_key)
    assert out["comparable"] is True
    assert out["dual_etc_in"] > out["dual_etc_adjusted_in"], (
        "potential ET should exceed stress-adjusted ET in a season with any "
        "dry spell in it")
    assert out["stress_reduction_in"] == pytest.approx(
        out["dual_etc_in"] - out["dual_etc_adjusted_in"], abs=0.02)
    assert "why_they_differ" in out


def test_the_comparison_explains_the_gap_instead_of_hiding_it(simple_season):
    """Measured at 10 to 26 percent on Delaware corn, and that is expected.

    The dual method resolves each wetting event, so a rain on bare ground in
    May pushes Ke near 1.1 where the single Kc_ini of 0.25 is already a season
    average and cannot rise. Reporting the number with the reason is more
    useful than tuning one method until it agrees with the other.
    """
    results, crop_key = simple_season
    out = dims.compare_with_dims_single_kc(results, crop_key)
    assert 0 < out["difference_pct"] < 60, (
        f"the methods now differ by {out['difference_pct']}%, which is outside "
        "the range this was measured at; something has changed in the crop table")
    assert "wetting event" in out["why_they_differ"]


# --------------------------------------------------------------------------
# emergence
# --------------------------------------------------------------------------

def test_growing_degree_days_cap_before_averaging():
    """The corn method, not a plain daily mean.

    A 95 degree day counts as 86, because corn does not develop faster above
    that. Capping after the average instead of before would credit a hot day
    with development it did not produce.
    """
    assert dims.daily_gdd(95, 68) == pytest.approx((86 + 68) / 2 - 50)
    assert dims.daily_gdd(60, 45) == pytest.approx((60 + 50) / 2 - 50)
    assert dims.daily_gdd(48, 38) == 0.0, "a day below the base earns nothing"


def test_emergence_is_predicted_at_the_dims_threshold():
    planting = datetime.date(2026, 5, 3)
    days = []
    for i in range(40):
        day = planting + datetime.timedelta(days=i)
        days.append(WeatherDay(day=day, tmax_f=78, tmin_f=58, rh_max=90,
                               rh_min=50, wind_mph=5, solar_rad_mj_m2=20,
                               rain_in=0.0))
    # 78/58 gives (78+58)/2-50 = 18 GDD a day, so 125 GDD lands on day 7.
    emerged, gdd = dims.predict_emergence(planting, days, "corn_grain")
    assert emerged == planting + datetime.timedelta(days=6)
    assert gdd >= dims.EMERGENCE_GDD["corn_grain"]


def test_a_cold_season_returns_no_emergence_rather_than_a_guess():
    """A field planted into a cold spring has not emerged, and saying so is
    more use than inventing a date."""
    planting = datetime.date(2026, 4, 1)
    days = [WeatherDay(day=planting + datetime.timedelta(days=i),
                       tmax_f=52, tmin_f=38, rh_max=90, rh_min=60,
                       wind_mph=5, solar_rad_mj_m2=10, rain_in=0.0)
            for i in range(10)]
    emerged, gdd = dims.predict_emergence(planting, days, "corn_grain")
    assert emerged is None
    assert gdd < dims.EMERGENCE_GDD["corn_grain"]


def test_crops_without_a_gdd_model_say_so_rather_than_defaulting():
    planting = datetime.date(2026, 5, 3)
    days = [WeatherDay(day=planting, tmax_f=85, tmin_f=65, rh_max=90,
                       rh_min=50, wind_mph=5, solar_rad_mj_m2=20, rain_in=0.0)]
    assert dims.predict_emergence(planting, days, "watermelon") == (None, 0.0)


# --------------------------------------------------------------------------
# managed root zone
# --------------------------------------------------------------------------

def test_the_managed_root_zone_caps_depth_without_extending_it():
    """It limits what is managed. It cannot invent roots the crop has not grown."""
    assert dims.effective_root_depth_in(36, 18) == 18.0
    assert dims.effective_root_depth_in(12, 18) == 12.0, (
        "a managed zone deeper than the roots must not deepen the root zone")
    assert dims.effective_root_depth_in(36, None) == 36.0


def test_the_managed_root_zone_is_clamped_to_the_dims_range():
    assert dims.clamp_managed_root_zone(40) == 24.0
    assert dims.clamp_managed_root_zone(2) == 8.0
    assert dims.clamp_managed_root_zone(18) == 18.0


# --------------------------------------------------------------------------
# field status
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dr,taw,expected", [
    (0.20, 2.5, "good"),        # 92 percent
    (0.90, 2.5, "fair"),        # 64 percent
    (1.50, 2.5, "poor"),        # 40 percent
    (2.20, 2.5, "very_poor"),   # 12 percent
])
def test_the_status_bands_match_the_dims_thresholds(dr, taw, expected):
    assert dims.field_status(dr, taw).key == expected


def test_a_field_that_has_not_emerged_says_so():
    status = dims.field_status(0.5, 2.5, emerged=False)
    assert status.key == "not_emerged"
    assert "not emerged" in status.note.lower()


def test_status_survives_a_field_with_no_known_capacity():
    """Better an honest unknown than a divide by zero or a confident zero."""
    assert dims.field_status(1.0, 0.0).key == "unknown"


@pytest.fixture(scope="module")
def simple_season():
    """One deterministic Delaware corn season through the engine of record."""
    from app.engine import pyfao56_bridge as pf
    from app.engine.crops import get_crop
    from app.engine.soils import get_soil

    if pf.engine_choice() != "pyfao56":
        pytest.skip("pyfao56 is not installed")

    planting = datetime.date(2026, 5, 3)
    days = []
    for i in range(130):
        day = planting + datetime.timedelta(days=i)
        t = math.sin((day.timetuple().tm_yday - 80) / 200 * math.pi)
        days.append(WeatherDay(day=day, tmax_f=68 + 24 * t, tmin_f=48 + 22 * t,
                               rh_max=93, rh_min=52, wind_mph=5.5,
                               solar_rad_mj_m2=12 + 16 * t,
                               rain_in=1.3 if i % 9 == 0 else 0.0))
    run = pf.run_season_pyfao56(weather=days, crop=get_crop("corn_grain"),
                                soil=get_soil("sassafras"), planting=planting,
                                lat_deg=38.64)
    return run.results, "corn_grain"
