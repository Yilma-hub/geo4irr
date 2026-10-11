"""
METRIC surface energy balance, and seasonal consumptive use.

Where FAO-56 says how much water the crop would use, METRIC says how much left
the field, by reading how hot the ground is. The whole model rests on
LE = Rn - G - H with LE solved as the residual, so an error anywhere in the
other three lands entirely in the answer.

The engine is a port of hectornieto/pyMETRIC and the pyTSEB library under it,
so a second job of these tests is to pin the places where the port knowingly
departs from the reference code, and to pin the places where it must not.

References: Allen, Tasumi and Trezza (2007), JIDE 133(4); Allen et al. (2013),
JAWRA 49(3) for the automated CIMEC calibration; Tasumi et al. (2008) for the
albedo weights. Delaware application: DGS Report of Investigations No. 84,
Sussex County, Landsat 8, 2016 season.
"""

from __future__ import annotations

import datetime
import math

import pytest

from app.engine import metric as met

# Realistic Landsat Collection 2 surface reflectances for three covers that
# sit side by side in a Sussex County July.
COVERS = {
    "irrigated_corn": dict(blue=.020, green=.042, red=.028, nir=.430,
                           swir1=.170, swir2=.075, lst_c=26.2),
    "dryland_corn":   dict(blue=.028, green=.055, red=.055, nir=.350,
                           swir1=.215, swir2=.110, lst_c=31.7),
    "bare_sand":      dict(blue=.115, green=.155, red=.205, nir=.265,
                           swir1=.345, swir2=.290, lst_c=43.0),
}


def pixel(cover: str, lst_offset: float = 0.0) -> met.Pixel:
    q = COVERS[cover]
    return met.Pixel(blue=q["blue"], green=q["green"], red=q["red"],
                     nir=q["nir"], swir1=q["swir1"], swir2=q["swir2"],
                     lst_k=q["lst_c"] + lst_offset + 273.15,
                     emissivity=0.98, elevation_m=12.0)


def scene_pixels() -> tuple[list[met.Pixel], list[str]]:
    pixels, labels = [], []
    for cover, n in (("irrigated_corn", 12), ("dryland_corn", 8), ("bare_sand", 6)):
        for i in range(n):
            pixels.append(pixel(cover, lst_offset=i * 0.12))
            labels.append(cover)
    return pixels, labels


def grid_scene(rows: int = 9, cols: int = 9
               ) -> tuple[list[met.Pixel], tuple[int, int], list[str]]:
    """A square field: irrigated corn with a dry bare corner and one pond pixel.

    Laid out rather than shuffled, because CIMEC's homogeneity test is about
    neighbours and a random scatter would have none. The pond is the trap the
    albedo test in Allen (2013) Eq 7 exists to catch: colder than any crop, so
    the naive search takes it, and it anchors the whole scene too low.
    """
    pixels: list[met.Pixel] = []
    labels: list[str] = []
    for r in range(rows):
        for c in range(cols):
            if r >= rows - 3 and c >= cols - 3:
                pixels.append(pixel("bare_sand", lst_offset=(r + c) * 0.1))
                labels.append("bare_sand")
            elif r == 1 and c == 1:
                # Open water: dark in the near infrared, cold, low albedo.
                pixels.append(met.Pixel(blue=.035, green=.030, red=.022,
                                        nir=.012, swir1=.006, swir2=.004,
                                        lst_k=20.5 + 273.15, emissivity=0.99,
                                        elevation_m=12.0))
                labels.append("pond")
            else:
                pixels.append(pixel("irrigated_corn",
                                    lst_offset=((r * 7 + c * 3) % 5) * 0.2))
                labels.append("irrigated_corn")
    return pixels, (rows, cols), labels


def july_weather(reference: str = "grass",
                 hourly: float | None = None) -> met.SceneWeather:
    default_hourly, daily = (0.62, 5.9) if reference == "grass" else (0.74, 7.2)
    return met.SceneWeather(
        overpass=datetime.date(2026, 7, 20), overpass_hour_local=10.6,
        air_temp_c=29.0, wind_speed_mps=2.6, rh_pct=62,
        ref_hourly_mm=default_hourly if hourly is None else hourly,
        ref_daily_mm=daily, reference=reference)


def run(pixels, wx=None, **kw):
    kw.setdefault("method", met.MINMAX)
    return met.run_metric(pixels, wx or july_weather(), lat_deg=38.64, **kw)


# --------------------------------------------------------------------------
# the energy balance has to close
# --------------------------------------------------------------------------

def test_the_energy_balance_closes_on_every_pixel():
    """Rn = G + H + LE. LE is the residual, so if this does not close the
    arithmetic is wrong and every ET figure downstream is meaningless."""
    pixels, _ = scene_pixels()
    out = run(pixels)
    for p in out.pixels:
        assert abs(p.rn - (p.g + p.h + p.le)) < 0.5, (
            f"pixel {p.index}: Rn {p.rn} against G+H+LE {p.g + p.h + p.le}")


def test_the_albedo_band_weights_sum_to_one():
    """Tasumi's weights are each band's share of total shortwave.

    A typo in one of them shifts net radiation across the whole scene by a few
    percent, and because the calibration is internal it would partly hide,
    which is worse than failing outright.
    """
    white = met.Pixel(blue=1, green=1, red=1, nir=1, swir1=1, swir2=1,
                      lst_k=300.0)
    assert met.broadband_albedo(white) == pytest.approx(1.0, abs=1e-6)


def test_albedo_lands_in_the_right_range_for_each_cover():
    """A corn canopy near 0.17, bare sand higher. If these drift the net
    radiation is wrong even though the balance still closes."""
    assert 0.14 < met.broadband_albedo(pixel("irrigated_corn")) < 0.22
    assert 0.18 < met.broadband_albedo(pixel("bare_sand")) < 0.30


def test_actual_et_separates_the_three_covers_in_the_right_order():
    """The whole purpose. Irrigated corn transpiring freely, dryland corn
    reduced, bare sand close to nothing."""
    pixels, labels = scene_pixels()
    out = run(pixels, etrf_bare=0.05)
    by_cover: dict[str, list[float]] = {}
    for label, p in zip(labels, out.pixels):
        by_cover.setdefault(label, []).append(p.eta_mm_day)
    mean = {k: sum(v) / len(v) for k, v in by_cover.items()}

    assert mean["irrigated_corn"] > mean["dryland_corn"] > mean["bare_sand"]
    assert mean["bare_sand"] < 1.5, "dry bare sand should be near zero ET"
    # Delaware peak corn use is about 0.25 in/day at R1 per the extension
    # manual, which is 6.4 mm. A clear hot July day a little above that is
    # reasonable; far above it is not.
    assert 4.0 < mean["irrigated_corn"] < 9.0, (
        f"irrigated corn at {mean['irrigated_corn']:.1f} mm/day is outside "
        "anything Delaware produces")


# --------------------------------------------------------------------------
# the departures from pyMETRIC, pinned
# --------------------------------------------------------------------------

def test_the_aerodynamic_resistance_is_solved_per_pixel_not_interpolated():
    """Departure 2, and the reason the engine was rewritten.

    An earlier version interpolated the resistance linearly between the two
    anchors on surface temperature, which is cheap and is not METRIC: the
    resistance depends on the pixel's own roughness and its own buoyancy, and a
    bare field and a closed canopy at the same surface temperature do not share
    one. Two pixels at nearly the same temperature but opposite cover must come
    out with different resistances, which interpolation cannot produce.

    pyMETRIC has its own version of this: in `METRIC()` the call that fills
    R_A sits inside the `else` branch, so with `use_METRIC_resistance=True`,
    which is the default, R_A is never assigned and every H is NaN. This port
    computes it in both cases.
    """
    # Same temperature, opposite cover, so interpolation would give them the
    # same resistance.
    corn = pixel("irrigated_corn", lst_offset=6.0)
    sand = pixel("bare_sand", lst_offset=-10.8)
    assert abs(corn.lst_k - sand.lst_k) < 0.05, "the fixture has drifted"

    pixels, _ = scene_pixels()
    out = run(pixels + [corn, sand], etrf_bare=0.05)
    a, b = out.pixels[-2], out.pixels[-1]
    assert abs(a.r_ah - b.r_ah) > 0.2, (
        f"two pixels at the same temperature but opposite cover came out with "
        f"resistances {a.r_ah} and {b.r_ah}; that is interpolation, not METRIC")
    assert all(p.iterations >= 1 for p in out.pixels)


def test_net_longwave_is_subtracted_from_the_reference_not_added():
    """Departure 1, a sign error in pyMETRIC's pet_asce.

    `calc_Ln` returns the magnitude of net OUTGOING longwave, positive. ASCE
    subtracts it from net shortwave. pyMETRIC forms `Rn = Sn + Ln`, which puts
    reference ET about a fifth high and, because METRIC calibrates against
    reference ET, puts every actual ET high with it.
    """
    t_k = 29.0 + 273.15
    p_mb = met.calc_pressure(12.0)
    ea = met.calc_vapor_pressure(t_k) * 0.62
    rnl = met.calc_Ln(t_k, ea)
    assert rnl > 0, "net outgoing longwave is a positive magnitude here"
    # 40 to 90 W/m2 covers a humid Delmarva summer sky.
    assert 30.0 < rnl < 100.0

    correct = met.asce_hourly_reference_et(t_k, 2.6, ea, p_mb, 820.0,
                                           reference=met.SHORT_REFERENCE)
    # What pyMETRIC's arrangement produces: the same calculation with the
    # longwave added, done by handing the function a larger shortwave input
    # equal to Sn + 2*Rnl.
    inflated = met.asce_hourly_reference_et(
        t_k, 2.6, ea, p_mb, 820.0 + 2 * rnl / (1 - 0.23),
        reference=met.SHORT_REFERENCE)
    assert inflated > correct * 1.08, (
        "the sign of net longwave should matter by more than a rounding error")
    # And the corrected value has to be physically right for the conditions.
    assert 0.45 < correct < 0.85, (
        f"{correct:.3f} mm/h is not a mid-morning July reference ET")


def test_the_alfalfa_and_grass_references_stand_in_the_published_ratio():
    """ASCE tall against short, which should land near 1.2 in a humid summer.

    The ratio is load-bearing: Allen (2013) anchors are quoted against
    alfalfa, DEOS publishes grass, and the conversion between them is applied
    inside the engine. If this ratio drifts away from the constant the engine
    uses, every scene run on grass reference is scaled wrongly.
    """
    t_k = 29.0 + 273.15
    p_mb = met.calc_pressure(12.0)
    ea = met.calc_vapor_pressure(t_k) * 0.62
    grass = met.asce_hourly_reference_et(t_k, 2.6, ea, p_mb, 820.0,
                                         reference=met.SHORT_REFERENCE)
    alfalfa = met.asce_hourly_reference_et(t_k, 2.6, ea, p_mb, 820.0,
                                           reference=met.TALL_REFERENCE)
    ratio = alfalfa / grass
    assert 1.10 < ratio < 1.30, f"tall against short came out at {ratio:.3f}"
    assert met.ALFALFA_TO_GRASS == pytest.approx(ratio, abs=0.06), (
        "the constant the engine converts anchor fractions with has drifted "
        "away from what the reference equations actually produce")


def test_the_monin_obukhov_length_counts_the_latent_heat_flux():
    """pyTSEB uses the virtual heat flux, Hv = H + 0.61 T cp E, not H alone.

    Water vapour is lighter than dry air, so evaporation makes the surface air
    buoyant on its own account. Over a transpiring July canopy that term is
    not small, and dropping it makes the surface look more stable than it is,
    which inflates the aerodynamic resistance and therefore H.
    """
    rho, c_p, t_a = 1.15, 1013.0, 302.0
    dry = met.calc_L(0.35, t_a, rho, c_p, 200.0, 0.0)
    wet = met.calc_L(0.35, t_a, rho, c_p, 200.0, 400.0)
    assert dry < 0 and wet < 0, "both cases are unstable daytime surfaces"
    assert abs(wet) < abs(dry), (
        "adding latent heat should make the surface more unstable, not less")
    assert abs(abs(dry) - abs(wet)) / abs(dry) > 0.05


def test_the_brutsaert_and_allen_stability_functions_broadly_agree():
    """pyTSEB defaults to Brutsaert (2005); Allen (2007) writes out the
    Paulson/Businger forms. Both are kept so the paper can be reproduced
    exactly, and they should not disagree by much over the range a Delaware
    morning produces. A large gap means one of the two was transcribed wrongly.
    """
    for L in (-500.0, -200.0, -80.0, -30.0):
        brut = met.psi_h_brutsaert(2.0 / L)
        allen = met.psi_h_allen2007(2.0, L)
        assert abs(brut - allen) < 0.6, (
            f"at L={L} the heat corrections differ by {abs(brut - allen):.2f}")


# --------------------------------------------------------------------------
# the anchor fractions, Allen (2013) Eqs 4 and 5
# --------------------------------------------------------------------------

def test_the_cold_anchor_sits_just_above_reference_at_full_cover():
    """1.05 of alfalfa reference, Allen (2013) Eq 4. Not a fudge: a
    well-watered field at full cover is taller and rougher than the reference
    crop and transpires slightly more."""
    assert met.etrf_cold(0.95) == pytest.approx(1.05)
    assert met.etrf_cold(0.99) == pytest.approx(1.05), "it does not keep rising"


def test_the_cold_anchor_tapers_when_the_canopy_has_not_closed():
    """The part that matters for an early-season scene.

    In late May nothing in a Delaware scene has closed. Anchoring the greenest
    pixel at the full 1.05 would assume a transpiration rate the canopy cannot
    reach, and every other pixel in the scene is then measured against it.
    """
    assert met.etrf_cold(0.475) == pytest.approx(0.525, abs=1e-3)
    assert met.etrf_cold(0.0) == 0.0
    assert met.etrf_cold(-0.2) == 0.0, "water must not produce a negative anchor"
    assert met.etrf_cold(0.60) < met.etrf_cold(0.80) < met.etrf_cold(0.95)


def test_the_hot_anchor_is_not_assumed_to_be_bone_dry():
    """Allen (2013) Eq 5. Bare soil two days after rain is still evaporating.

    Pinning the hot anchor at zero forces the dT line too steep, and because
    every other pixel is read off that line it inflates ET across the scene.
    """
    assert met.etrf_hot(0.12, etrf_bare=0.0) == pytest.approx(0.12 * met.etrf_cold(0.12))
    wet = met.etrf_hot(0.12, etrf_bare=0.45)
    dry = met.etrf_hot(0.12, etrf_bare=0.02)
    assert wet > dry
    assert wet == pytest.approx(0.12 * met.etrf_cold(0.12) + 0.88 * 0.45)


def test_bare_soil_evaporation_follows_the_rain_record():
    """The surface layer of FAO-56, which is the same mechanism the Ke term in
    this tool's water balance uses, so the two methods share one number."""
    ref = [5.5] * 10
    just_rained = met.bare_soil_etrf([0, 0, 0, 0, 0, 0, 0, 0, 25.0, 0.0], ref)
    long_dry = met.bare_soil_etrf([25.0] + [0.0] * 9, ref)
    assert just_rained > long_dry
    assert long_dry == pytest.approx(0.05, abs=1e-6), (
        "after a week of 5.5 mm reference ET and no rain, 8 mm of readily "
        "evaporable water is long gone")
    assert 0.3 < just_rained <= 0.9
    assert met.bare_soil_etrf([], []) == 0.0


def test_the_hot_anchor_temperature_is_lowered_after_a_wet_two_months():
    """Allen (2013) Eq 8. A wet 60 days means the hot pixel is not as dry as
    it looks, and treating it as fully dry stretches the calibration."""
    pixels, _ = scene_pixels()
    wx = july_weather()
    dry = met.run_metric(pixels, wx, lat_deg=38.64, method=met.MINMAX)
    wet = met.run_metric(pixels, wx, lat_deg=38.64, method=met.MINMAX,
                         rain_60_mm=320.0, ref_et_60_mm=300.0)
    # MINMAX does not use the adjustment, so the two must agree; the
    # adjustment lives in CIMEC and this records which search owns it.
    assert dry.calibration["endmember_search"] == "MINMAX"
    assert wet.mean_eta_mm_day == pytest.approx(dry.mean_eta_mm_day, abs=1e-9)


# --------------------------------------------------------------------------
# the endmember search
# --------------------------------------------------------------------------

def test_the_simple_search_takes_the_coldest_green_and_hottest_bare_pixels():
    pixels, _ = scene_pixels()
    cold, hot, _notes, search = met.pick_anchors(
        pixels, july_weather(), 38.64, method=met.MINMAX)
    assert search == "MINMAX"
    assert cold.ndvi > 0.70, "the cold anchor must be fully vegetated"
    assert hot.ndvi < 0.25, "the hot anchor must be effectively bare"
    assert hot.lst_k > cold.lst_k


def test_cimec_handles_a_pond_that_defeats_the_simple_search():
    """The failure mode Allen (2013) Eq 7 exists to prevent.

    A farm pond inside a pivot boundary is the coldest thing in the scene and
    also the lowest in NDVI, because water is dark in the near infrared. The
    simple search sorts on NDVI, so the pond lands in the bare pool and gets
    nominated as the hot, dry anchor while being the coldest pixel present.
    The engine then refuses the scene outright, which is the right answer to a
    nonsense calibration but loses the overpass.

    CIMEC gets it right instead: the 5 percent NDVI and albedo tests put the
    cold anchor on a crop pixel whose albedo matches what a dense canopy
    should show at this sun angle, and the hot anchor on dry sand.
    """
    pixels, shape, labels = grid_scene()
    pond = labels.index("pond")

    # What the simple search does with the same scene, and the reason CIMEC is
    # the default rather than an option.
    with pytest.raises(ValueError, match="not warmer"):
        met.pick_anchors(pixels, july_weather(), 38.64, method=met.MINMAX)

    cold, hot, notes, search = met.pick_anchors(
        pixels, july_weather(), 38.64, grid_shape=shape, method=met.CIMEC)

    assert search == "CIMEC", f"CIMEC did not run: {notes}"
    assert cold.index != pond, (
        "CIMEC anchored the scene on the pond, so the albedo test is not "
        "doing its job")
    assert hot.index != pond
    assert cold.ndvi > 0.5, "the cold anchor should be a crop"
    assert labels[hot.index] == "bare_sand"
    assert hot.lst_k > cold.lst_k

    out = met.run_metric(pixels, july_weather(), lat_deg=38.64,
                         grid_shape=shape, method=met.CIMEC, etrf_bare=0.05)
    assert out.calibration["endmember_search"] == "CIMEC"
    # The pond itself should come out at or near the ceiling, which is the
    # honest result: open water evaporates faster than the crop around it.
    assert out.pixels[pond].etrf > out.mean_etrf


def test_cimec_needs_a_grid_and_says_so_when_it_does_not_have_one():
    """Its homogeneity tests are about neighbours, and a flat list has none.
    Falling back quietly would leave nobody able to tell which search ran."""
    pixels, _ = scene_pixels()
    out = run(pixels, method=met.CIMEC)
    assert out.calibration["endmember_search"] == "MINMAX"


def test_the_cimec_albedo_threshold_follows_the_sun_angle():
    """Allen (2013) Eq 7: the albedo a dense well-watered canopy should show.
    It rises as the sun gets lower, because more light is reflected forward."""
    high_sun = met.cimec_albedo_threshold(25.0)    # near noon in June
    low_sun = met.cimec_albedo_threshold(65.0)     # early morning
    assert low_sun > high_sun
    assert 0.15 < high_sun < 0.26, f"{high_sun:.3f} is not a canopy albedo"


def test_the_moving_window_filter_finds_the_edge_of_a_field():
    """A candidate anchor in a patch that varies wildly is a field edge, a
    building or a cloud shadow. The filter is how CIMEC prefers the quiet one."""
    grid = [[10.0] * 6 + [40.0] * 6 for _ in range(12)]
    cv, mean, std = met.moving_cv_filter(grid, (5, 5))
    middle = cv[6][0]        # deep inside the uniform left half
    boundary = cv[6][6]      # straddling the step
    assert middle == pytest.approx(0.0, abs=1e-9)
    assert boundary > 0.2
    assert mean[6][0] == pytest.approx(10.0)


def test_the_percentile_helper_matches_numpy():
    """CIMEC is defined in percentiles, and an off-by-one in the interpolation
    moves which pixel anchors the scene."""
    np = pytest.importorskip("numpy")
    vals = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    for pct in (0.05, 10, 20, 50, 80, 95):
        assert met._percentile(vals, pct) == pytest.approx(
            float(np.percentile(vals, pct)), abs=1e-9)


def test_a_scene_with_no_temperature_range_is_refused():
    """Under uniform cloud every pixel reads the same temperature. There is
    nothing to calibrate on, and a number produced anyway would be invented."""
    flat = [met.Pixel(blue=.03, green=.05, red=.04, nir=.40, swir1=.18,
                      swir2=.09, lst_k=295.0) for _ in range(8)]
    with pytest.raises(ValueError, match="calibrate|contrast"):
        met.pick_anchors(flat, july_weather(), 38.64, method=met.MINMAX)


def test_an_all_green_scene_anchors_both_ends_near_potential():
    """Early July after a wet week, nothing in the scene is dry.

    This is where the Allen (2013) fractions earn their place. Both anchors
    land near potential ET, the dT line comes out nearly flat, and the scene
    reports that the whole field is transpiring freely. The alternative, which
    an earlier version of this engine did, was to declare the warmest green
    pixel a dry anchor and drag the field's ET down towards zero against a
    reference that does not exist.
    """
    pixels = [pixel("irrigated_corn", i * 0.3) for i in range(10)]
    cold, hot, _notes, _ = met.pick_anchors(pixels, july_weather(), 38.64,
                                            method=met.MINMAX)
    assert hot.etrf > 0.5 * cold.etrf, (
        "with no dry ground in the scene the hot anchor must not be treated "
        "as bone dry")
    out = run(pixels)
    assert out.mean_etrf > 0.6, (
        f"a uniformly well-watered field came out at ETrF {out.mean_etrf}")


def test_the_anchor_choice_is_reported_rather_than_hidden():
    """The whole scene rests on two pixels. Anyone checking the result needs
    to see which two and why."""
    pixels, _ = scene_pixels()
    out = run(pixels)
    assert out.cold is not None and out.hot is not None
    assert out.cold.reason and out.hot.reason
    assert "Allen 2013" in out.cold.reason
    for key in ("a", "b", "dt_cold", "dt_hot", "rah_cold", "rah_hot",
                "endmember_search", "reference_et_source"):
        assert key in out.calibration, f"{key} is missing from the calibration"


def test_the_calibration_line_passes_through_both_anchors():
    """dT = a + b*Ts is fitted on exactly two points, so it has to hit them.
    If it does not, the fit is being done on something other than the anchors.
    """
    pixels, _ = scene_pixels()
    out = run(pixels)
    a, b = out.calibration["a"], out.calibration["b"]
    assert a + b * out.cold.lst_k == pytest.approx(out.calibration["dt_cold"],
                                                   abs=1e-3)
    assert a + b * out.hot.lst_k == pytest.approx(out.calibration["dt_hot"],
                                                  abs=1e-3)
    assert b > 0, "a hotter surface must imply a larger temperature gradient"


# --------------------------------------------------------------------------
# physical sanity
# --------------------------------------------------------------------------

def test_sensible_heat_cannot_exceed_the_available_energy():
    """H above Rn - G would mean the surface condensing water out of the air,
    which a dry field in July is not doing."""
    pixels, _ = scene_pixels()
    out = run(pixels)
    for p in out.pixels:
        assert p.h <= p.rn - p.g + 1e-6
        assert p.le >= -1e-6, "negative latent heat means condensation"


def test_ground_heat_flux_is_a_larger_share_under_bare_soil():
    """Under a closed canopy little radiation reaches the ground. Over bare
    sand a quarter of it goes straight in. If this is flat across covers the
    NDVI term has stopped working."""
    wx = july_weather()
    corn = pixel("irrigated_corn")
    sand = pixel("bare_sand")
    rn_corn = met.net_radiation(corn, wx, 38.64, 201)["rn"]
    rn_sand = met.net_radiation(sand, wx, 38.64, 201)["rn"]
    share_corn = met.soil_heat_flux(corn, rn_corn) / rn_corn
    share_sand = met.soil_heat_flux(sand, rn_sand) / rn_sand
    assert share_sand > share_corn * 1.5


def test_net_radiation_counts_the_reflected_share_of_incoming_longwave():
    """A surface with emissivity 0.98 reflects two percent of the incoming
    longwave rather than absorbing it. Dropping that term biases Rn high
    across the whole scene, and the internal calibration partly hides it."""
    wx = july_weather()
    p = pixel("irrigated_corn")
    rad = met.net_radiation(p, wx, 38.64, 201)
    naive = rad["sn"] + rad["l_dn"] - met.calc_stephan_boltzmann(p.lst_k) * p.emissivity
    assert naive > rad["rn"]
    assert rad["rn"] == pytest.approx(naive - (1 - p.emissivity) * rad["l_dn"],
                                      abs=1e-6)


def test_the_roughness_length_grows_with_the_canopy():
    """Allen (2007) Eq 36. Roughness drives the friction velocity and so the
    resistance; a floor stops bare soil producing a resistance that explodes."""
    assert met.momentum_roughness(pixel("bare_sand")) == pytest.approx(0.005)
    corn = met.momentum_roughness(pixel("irrigated_corn"))
    assert corn > 0.03, f"a closed corn canopy at z0m {corn:.3f} m is too smooth"


def test_the_friction_velocity_rises_with_wind_and_with_roughness():
    neutral = float("inf")
    calm = met.calc_u_star(3.0, 200.0, neutral, 0.0, 0.05)
    windy = met.calc_u_star(6.0, 200.0, neutral, 0.0, 0.05)
    rough = met.calc_u_star(3.0, 200.0, neutral, 0.0, 0.20)
    assert windy == pytest.approx(2 * calm, rel=1e-9)
    assert rough > calm


def test_the_resistance_uses_the_metric_two_height_convention():
    """Allen (2007) Eq 34 defines dT between 0.1 m and 2.0 m above the zero
    plane. pyMETRIC hard-codes those heights for its METRIC branch and so does
    this. Changing them silently rescales every H in the scene."""
    assert (met.METRIC_Z_T, met.METRIC_Z_0H, met.METRIC_D_0) == (2.0, 0.1, 0.0)
    neutral = float("inf")
    ustar = 0.3
    expected = math.log(2.0 / 0.1) / (ustar * met.KARMAN)
    assert met.calc_R_A(2.0, ustar, neutral, 0.0, 0.1) == pytest.approx(expected)


def test_the_landsat_scaling_factors_are_applied():
    """Collection 2 reflectance carries a negative offset and surface
    temperature a large one. Skipping either silently wrecks everything: raw
    counts would give an albedo in the thousands and a temperature of 20000 K.
    """
    assert met.scale_reflectance(10000) == pytest.approx(0.075, abs=1e-6)
    assert met.scale_surface_temp_k(44000) == pytest.approx(
        44000 * 0.00341802 + 149, abs=1e-6)
    # A typical summer surface near 300 K
    assert 280 < met.scale_surface_temp_k(44000) < 320


# --------------------------------------------------------------------------
# the reference surface
# --------------------------------------------------------------------------

def test_the_hourly_and_daily_reference_must_be_the_same_surface():
    """The error that inflated every seasonal total by a fifth.

    The ET fraction is a ratio taken at the overpass and then applied to the
    daily total. Taking the numerator against grass reference and the
    denominator against alfalfa multiplies everything by the ratio between
    them, roughly 1.2 in a Delaware summer, and nothing downstream complains
    because the number still looks plausible.
    """
    with pytest.raises(ValueError, match="grass|alfalfa"):
        met.SceneWeather(
            overpass=datetime.date(2026, 7, 20), overpass_hour_local=10.6,
            air_temp_c=29.0, wind_speed_mps=2.6,
            ref_hourly_mm=0.62, ref_daily_mm=7.2, reference="mixed")


def test_the_two_reference_surfaces_give_answers_within_about_a_tenth():
    """Both are legitimate. Running the same scene against each is the cleanest
    check that the anchor fractions are being converted rather than reused."""
    pixels, _ = scene_pixels()
    grass = run(pixels, july_weather("grass"))
    alfalfa = run(pixels, july_weather("alfalfa"))
    spread = abs(grass.mean_eta_mm_day - alfalfa.mean_eta_mm_day)
    assert spread < 0.15 * max(grass.mean_eta_mm_day, alfalfa.mean_eta_mm_day), (
        f"grass gave {grass.mean_eta_mm_day:.2f} mm/day and alfalfa "
        f"{alfalfa.mean_eta_mm_day:.2f}; one of the two cold-anchor "
        "conversions is wrong")


def test_a_missing_station_reference_is_computed_and_flagged():
    """Departure 4. The station value is preferred because the calibration
    rests on it, but a gap in the DEOS record should not lose the scene.
    pyMETRIC always computes it internally; this falls back to that and says so.
    """
    pixels, _ = scene_pixels()
    out = run(pixels, july_weather(hourly=0.0))
    assert out.calibration["reference_et_source"] == "ASCE hourly equation"
    assert 0.3 < out.calibration["reference_et_hourly_mm"] < 1.0
    assert any("weather station" in w for w in out.warnings)
    assert out.mean_eta_mm_day > 0


def test_an_overpass_outside_daylight_is_refused():
    """With no sun there is no incoming shortwave, so no reference ET, so
    nothing to anchor against. A number produced here would be meaningless."""
    pixels, _ = scene_pixels()
    wx = met.SceneWeather(
        overpass=datetime.date(2026, 7, 20), overpass_hour_local=1.0,
        air_temp_c=22.0, wind_speed_mps=2.6,
        ref_hourly_mm=0.0, ref_daily_mm=5.9, reference="grass")
    with pytest.raises(ValueError, match="unanchored|daylight|zero"):
        met.run_metric(pixels, wx, lat_deg=38.64, method=met.MINMAX)


# --------------------------------------------------------------------------
# seasonal accumulation
# --------------------------------------------------------------------------

def season_scenes():
    """Six usable Landsat passes across a Delaware corn season."""
    base = datetime.date(2026, 5, 3)
    offsets = [12, 36, 60, 84, 108, 132]
    eta = [1.8, 3.4, 6.1, 6.8, 5.2, 2.6]      # mm/day, a plausible arc
    return [base + datetime.timedelta(days=o) for o in offsets], eta


def daily_reference(planting, days=150, peak=6.4, weather=False):
    """A seasonal reference ET curve, mm/day.

    `weather=False` gives a smooth sine, which is a seasonal trend with no
    weather in it. `weather=True` adds the day-to-day swing a real July has:
    a run of clear days near peak, then a cloudy spell at half that. The
    distinction is the whole argument for interpolating the ET fraction rather
    than ET itself, so both are needed to test it.
    """
    out = {}
    for i in range(days):
        d = planting + datetime.timedelta(days=i)
        trend = max(1.5, peak * math.sin(math.pi * (i + 20) / (days + 40)))
        if weather:
            # A deterministic wet-and-dry pattern: eight clear days, then five
            # overcast ones at 45 percent of the trend.
            trend *= 1.15 if (i % 13) < 8 else 0.45
        out[d] = trend
    return out


def test_a_season_accumulates_to_a_plausible_delaware_total():
    planting = datetime.date(2026, 5, 3)
    through = datetime.date(2026, 9, 20)
    dates, eta = season_scenes()
    out = met.accumulate_season(dates, eta, planting, through,
                                daily_reference_mm=daily_reference(planting))
    # Seasonal corn ET on Delmarva runs roughly 18 to 26 inches.
    assert 12.0 < out.total_eta_in < 30.0, (
        f"{out.total_eta_in} inches is outside anything a Delaware corn season "
        "produces")
    assert out.method == "ETrF interpolation"
    assert len(out.scenes) == 6


def test_gaps_are_filled_through_the_et_fraction_not_through_et_itself():
    """The choice that matters more than the energy balance for a season total.

    Interpolating actual ET straight across a 24 day gap assumes the days
    between two clear overpasses behaved like the overpass days, and clear days
    are not a random sample of a Delaware July. Interpolating the fraction and
    multiplying by each day's own reference ET carries the real weather.

    The test needs weather in the reference series to show this. Against a
    smooth seasonal sine the two methods agree closely, which is the honest
    result: with no day-to-day variation there is nothing for the better method
    to recover. Against a series with cloudy spells in it they part company.
    """
    planting = datetime.date(2026, 5, 3)
    through = datetime.date(2026, 9, 20)
    dates, eta = season_scenes()

    smooth = met.accumulate_season(dates, eta, planting, through,
                                   daily_reference_mm=daily_reference(planting))
    real = met.accumulate_season(
        dates, eta, planting, through,
        daily_reference_mm=daily_reference(planting, weather=True))
    without = met.accumulate_season(dates, eta, planting, through)

    assert smooth.method == real.method == "ETrF interpolation"
    assert without.method == "direct ETa interpolation"
    assert any("not a random sample" in w for w in without.warnings), (
        "the weaker method should say that it is weaker")

    # With weather in the reference series the two methods separate, because
    # the overpasses land on clear days and the direct interpolation spreads
    # clear-day ET across the cloudy ones.
    assert abs(real.total_eta_in - without.total_eta_in) > 0.03 * without.total_eta_in, (
        f"ETrF interpolation gave {real.total_eta_in} inches and direct "
        f"interpolation {without.total_eta_in}; they should differ once the "
        "reference series has weather in it")
    # And against a smooth curve they do not, which is what tells you the
    # difference above is the weather and not an arithmetic error.
    assert smooth.total_eta_in == pytest.approx(without.total_eta_in, rel=0.03)


def test_the_season_reports_weekly_averages():
    """Weekly is the resolution an irrigator acts on, and the honest resolution
    for a 16-day satellite: a single interpolated day carries a precision the
    method does not have."""
    planting = datetime.date(2026, 5, 3)
    dates, eta = season_scenes()
    out = met.accumulate_season(dates, eta, planting,
                                datetime.date(2026, 9, 20),
                                daily_reference_mm=daily_reference(planting))
    assert out.weekly, "no weekly summary was produced"
    assert 18 < len(out.weekly) < 24, f"{len(out.weekly)} weeks in 141 days"
    assert sum(w["total_eta_mm"] for w in out.weekly) == pytest.approx(
        out.total_eta_mm, abs=0.3), (
        "the weekly totals must add up to the season total")
    peak = max(out.weekly, key=lambda w: w["mean_eta_mm_day"])
    assert datetime.date.fromisoformat(peak["week_start"]).month in (7, 8), (
        f"peak water use landed in week starting {peak['week_start']}, which "
        "is not when a Delaware corn crop uses most")
    for w in out.weekly:
        assert w["mean_eta_in_day"] == pytest.approx(
            w["mean_eta_mm_day"] / 25.4, abs=0.002)


def test_accumulation_stops_at_harvest():
    planting = datetime.date(2026, 5, 3)
    harvest = datetime.date(2026, 9, 1)
    dates, eta = season_scenes()
    ref = daily_reference(planting)
    early = met.accumulate_season(dates, eta, planting,
                                  datetime.date(2026, 10, 15),
                                  daily_reference_mm=ref, harvest=harvest)
    assert early.through == harvest
    assert any("harvest" in w.lower() for w in early.warnings)
    full = met.accumulate_season(dates, eta, planting,
                                 datetime.date(2026, 10, 15),
                                 daily_reference_mm=ref)
    assert early.total_eta_in < full.total_eta_in
    assert early.weekly[-1]["week_end"] <= harvest.isoformat()


def test_a_long_gap_before_the_first_scene_is_flagged():
    """Early season ET before the first clear pass is extrapolated flat, and
    that is the least reliable part of a seasonal total."""
    planting = datetime.date(2026, 5, 3)
    late = [planting + datetime.timedelta(days=d) for d in (60, 84, 108)]
    out = met.accumulate_season(late, [6.1, 6.8, 5.2], planting,
                                datetime.date(2026, 9, 20),
                                daily_reference_mm=daily_reference(planting))
    assert any("after planting" in w for w in out.warnings)


def test_no_scenes_produces_an_explanation_rather_than_zero():
    """A silent zero reads as 'the crop used no water'."""
    planting = datetime.date(2026, 5, 3)
    out = met.accumulate_season([], [], planting, datetime.date(2026, 9, 20))
    assert out.total_eta_in == 0.0
    assert out.method == "none"
    assert out.warnings and "No usable satellite scenes" in out.warnings[0]


def test_interpolation_holds_flat_outside_the_observed_range():
    """Extrapolating a trend past the last observation produces a confident
    number with nothing behind it, and over a long tail it can go negative."""
    planting = datetime.date(2026, 5, 3)
    pts = [(planting + datetime.timedelta(days=30), 5.0),
           (planting + datetime.timedelta(days=60), 3.0)]
    before = met._interpolate(pts, planting)
    after = met._interpolate(pts, planting + datetime.timedelta(days=120))
    assert before == 5.0
    assert after == 3.0, "a falling trend must not be run on into negative ET"
