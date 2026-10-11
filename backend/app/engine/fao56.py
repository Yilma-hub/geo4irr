"""
FAO-56 dual crop coefficient evapotranspiration and root-zone soil water balance.

This is the computational core of Geo4Irr. It implements:

  * ASCE/FAO-56 Penman-Monteith reference ET (ETo, short grass reference)
  * Dual crop coefficient partitioning:  ETc_adj = ETo * (Kcb * Ks + Ke)
  * Two-layer soil water balance: evaporation layer (Eq. 77-78) and
    root zone depletion (Eq. 85), with an added shallow-water-table upflux
    term that matters on Delaware Coastal Plain soils.

Units are US customary at the interface (inches, degF, mph) because that is what
Delaware growers use, and SI internally because that is what FAO-56 is written in.

Reference:
  Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998) FAO Irrigation and
  Drainage Paper No. 56.  Crop Evapotranspiration.
  ASCE-EWRI (2005) The ASCE Standardized Reference Evapotranspiration Equation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Sequence

MM_PER_IN = 25.4
IN_PER_MM = 1.0 / 25.4


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def f_to_c(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0


def mph_to_ms(u_mph: float) -> float:
    return u_mph * 0.44704


def in_to_mm(x_in: float) -> float:
    return x_in * MM_PER_IN


def mm_to_in(x_mm: float) -> float:
    return x_mm * IN_PER_MM


# ---------------------------------------------------------------------------
# Reference evapotranspiration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WeatherDay:
    """One day of weather at a station or grid cell.

    Native units are the ones DEOS and the National Weather Service publish for
    Delaware users, so no conversion is needed at the ingest boundary.
    """

    day: date
    tmax_f: float
    tmin_f: float
    rh_max: float | None = None        # percent
    rh_min: float | None = None        # percent
    wind_mph: float | None = None      # at 2 m; converted if measured at 10 m
    wind_height_m: float = 2.0
    solar_rad_mj_m2: float | None = None   # MJ m-2 d-1
    rain_in: float = 0.0
    eto_in: float | None = None        # if the provider already computed it
    source: str = "observed"           # observed | forecast | gridded

    @property
    def tmean_f(self) -> float:
        return (self.tmax_f + self.tmin_f) / 2.0


def _sat_vapor_pressure(t_c: float) -> float:
    """Saturation vapour pressure, kPa (FAO-56 Eq. 11)."""
    return 0.6108 * math.exp((17.27 * t_c) / (t_c + 237.3))


def _slope_svp(t_c: float) -> float:
    """Slope of the saturation vapour pressure curve, kPa/degC (Eq. 13)."""
    return (4098.0 * _sat_vapor_pressure(t_c)) / ((t_c + 237.3) ** 2)


def _extraterrestrial_radiation(doy: int, lat_deg: float) -> float:
    """Ra in MJ m-2 d-1 (Eq. 21)."""
    lat = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    decl = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    x = -math.tan(lat) * math.tan(decl)
    x = max(-1.0, min(1.0, x))
    ws = math.acos(x)
    return (24.0 * 60.0 / math.pi) * 0.0820 * dr * (
        ws * math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.sin(ws)
    )


def reference_et(w: WeatherDay, lat_deg: float, elev_m: float = 12.0) -> float:
    """ASCE standardized short-crop reference ET, inches per day.

    Delaware elevations are low (statewide high point is about 135 m); the
    default of 12 m is representative of the Sussex/Kent farmed plain.

    Falls back to Hargreaves-Samani when radiation or humidity are missing,
    which happens for forecast days and for the sparser DEOS sites.
    """
    if w.eto_in is not None:
        return max(0.0, w.eto_in)

    tmax = f_to_c(w.tmax_f)
    tmin = f_to_c(w.tmin_f)
    tmean = (tmax + tmin) / 2.0
    doy = w.day.timetuple().tm_yday
    ra = _extraterrestrial_radiation(doy, lat_deg)

    if w.solar_rad_mj_m2 is None or w.rh_min is None:
        # Hargreaves-Samani (Eq. 52). Adequate for 1-7 day forecast horizons.
        eto_mm = 0.0023 * (tmean + 17.8) * math.sqrt(max(0.0, tmax - tmin)) * ra * 0.408
        return max(0.0, mm_to_in(eto_mm))

    p = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26
    gamma = 0.000665 * p
    delta = _slope_svp(tmean)

    es = (_sat_vapor_pressure(tmax) + _sat_vapor_pressure(tmin)) / 2.0
    rh_max = w.rh_max if w.rh_max is not None else w.rh_min
    ea = 0.5 * (
        _sat_vapor_pressure(tmin) * (rh_max / 100.0)
        + _sat_vapor_pressure(tmax) * (w.rh_min / 100.0)
    )
    vpd = max(0.0, es - ea)

    rs = w.solar_rad_mj_m2
    rso = (0.75 + 2e-5 * elev_m) * ra
    rns = (1.0 - 0.23) * rs
    ratio = min(1.0, rs / rso) if rso > 0 else 1.0
    rnl = (
        4.903e-9
        * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0)
        * (0.34 - 0.14 * math.sqrt(max(0.0, ea)))
        * (1.35 * ratio - 0.35)
    )
    rn = rns - rnl
    g = 0.0  # daily time step over a grass reference

    u2 = mph_to_ms(w.wind_mph if w.wind_mph is not None else 4.5)
    if w.wind_height_m != 2.0:
        # Log-law adjustment to 2 m (Eq. 47)
        u2 *= 4.87 / math.log(67.8 * w.wind_height_m - 5.42)
    u2 = max(0.5, u2)

    num = 0.408 * delta * (rn - g) + gamma * (900.0 / (tmean + 273.0)) * u2 * vpd
    den = delta + gamma * (1.0 + 0.34 * u2)
    return max(0.0, mm_to_in(num / den))


# ---------------------------------------------------------------------------
# Crop coefficient curve
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CropStage:
    """FAO-56 four-stage Kcb curve, expressed in days after planting."""

    name: str
    kcb_ini: float
    kcb_mid: float
    kcb_end: float
    l_ini: int
    l_dev: int
    l_mid: int
    l_late: int
    root_min_in: float
    root_max_in: float
    height_max_ft: float
    mad_default: float
    mad_critical: float          # tighter allowable depletion in the sensitive window
    critical_start_dap: int      # e.g. corn V12 through R2
    critical_end_dap: int
    yield_response_ky: float     # FAO-33 Ky, seasonal

    @property
    def season_days(self) -> int:
        return self.l_ini + self.l_dev + self.l_mid + self.l_late

    def kcb(self, dap: int) -> float:
        """Basal crop coefficient at a given number of days after planting."""
        if dap <= 0:
            return self.kcb_ini
        if dap <= self.l_ini:
            return self.kcb_ini
        if dap <= self.l_ini + self.l_dev:
            frac = (dap - self.l_ini) / max(1, self.l_dev)
            return self.kcb_ini + frac * (self.kcb_mid - self.kcb_ini)
        if dap <= self.l_ini + self.l_dev + self.l_mid:
            return self.kcb_mid
        if dap <= self.season_days:
            frac = (dap - self.l_ini - self.l_dev - self.l_mid) / max(1, self.l_late)
            return self.kcb_mid + frac * (self.kcb_end - self.kcb_mid)
        return self.kcb_end

    def root_depth_in(self, dap: int) -> float:
        """Root depth grows with the Kcb curve rather than linearly to maturity.

        SWAT+ and most schedulers grow roots linearly to physiological maturity,
        which over-predicts late-season extraction. Tying expansion to canopy
        development (Activity 6 in the project narrative) tracks the LIRF and
        Delaware lysimeter data more closely.
        """
        if dap <= 0:
            return self.root_min_in
        full = self.l_ini + self.l_dev
        frac = min(1.0, dap / max(1, full))
        return self.root_min_in + frac * (self.root_max_in - self.root_min_in)

    def height_ft(self, dap: int) -> float:
        full = self.l_ini + self.l_dev
        return self.height_max_ft * min(1.0, dap / max(1, full))

    def mad(self, dap: int) -> float:
        if self.critical_start_dap <= dap <= self.critical_end_dap:
            return self.mad_critical
        return self.mad_default


def kcb_from_canopy_cover(fc: float, kcb_max: float = 1.15) -> float:
    """Basal Kcb inferred from measured fractional canopy cover.

    Follows the density coefficient approach used with UAV and satellite canopy
    products, which is how Activity 3 of the narrative replaces tabular Kcb.
    fc is the fraction of soil shaded at solar noon, 0-1.
    """
    fc = max(0.0, min(1.0, fc))
    fc_eff = fc / 0.94 if fc < 0.94 else 1.0
    return round(min(kcb_max, 0.15 + (kcb_max - 0.15) * fc_eff ** 0.75), 3)


def kcb_from_ndvi(ndvi: float, ndvi_bare: float = 0.17, ndvi_full: float = 0.86,
                  kcb_max: float = 1.15) -> float:
    """Linear NDVI-to-Kcb transfer, clipped at bare soil and full cover."""
    if ndvi <= ndvi_bare:
        return 0.15
    frac = min(1.0, (ndvi - ndvi_bare) / (ndvi_full - ndvi_bare))
    return round(0.15 + frac * (kcb_max - 0.15), 3)


def ks_from_cwsi(cwsi: float) -> float:
    """Water stress coefficient from the Crop Water Stress Index.

    CWSI of 0 is a fully watered canopy, 1 is a non-transpiring canopy.
    """
    return round(max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, cwsi)))), 3)


# ---------------------------------------------------------------------------
# Soil water balance
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SoilProfile:
    """Root-zone soil hydraulic description for one management zone."""

    series: str
    awc_in_per_in: float          # available water capacity
    fc_vol: float                 # field capacity, volumetric
    wp_vol: float                 # wilting point, volumetric
    ksat_in_hr: float
    rew_in: float = 0.28          # readily evaporable water, FAO-56 Table 19
    tew_in: float = 0.71          # total evaporable water
    ze_in: float = 4.0            # evaporation layer depth
    water_table_depth_in: float | None = None   # seasonal high water table
    restrictive_layer_in: float | None = None   # hardpan / fragipan depth

    def taw_in(self, root_depth_in: float) -> float:
        """Total available water in the current root zone, inches."""
        depth = root_depth_in
        if self.restrictive_layer_in:
            depth = min(depth, self.restrictive_layer_in)
        return round(self.awc_in_per_in * depth, 3)

    def raw_in(self, root_depth_in: float, mad: float) -> float:
        """Readily available water: the depletion at which stress begins."""
        return round(self.taw_in(root_depth_in) * mad, 3)

    def upflux_in(self, root_depth_in: float, depletion_in: float) -> float:
        """Capillary rise from a shallow water table, inches per day.

        Fallsington, Pocomoke and other poorly drained Delaware soils sit over a
        seasonal high water table and can supply real water to a stressed root
        zone. Ignoring it is why generic schedulers over-irrigate these fields.
        Modelled as an exponential decay with the gap between the root front and
        the water table, scaled by how dry the root zone already is.
        """
        if not self.water_table_depth_in:
            return 0.0
        gap = self.water_table_depth_in - root_depth_in
        if gap > 48:
            return 0.0
        # Upflux is demand-driven: a root zone sitting at field capacity has no
        # gradient to pull against, however shallow the water table is.
        demand = min(1.0, depletion_in / max(0.1, self.taw_in(root_depth_in)))
        if gap <= 0:
            return round(0.10 * demand, 4)      # roots in the capillary fringe
        return round(0.12 * math.exp(-gap / 14.0) * demand, 4)


@dataclass(slots=True)
class BalanceState:
    """Mutable state carried from one day to the next."""

    dr_in: float = 0.0            # root zone depletion below field capacity
    de_in: float = 0.0            # evaporation layer depletion
    root_depth_in: float = 6.0
    last_root_depth_in: float = 6.0


@dataclass(slots=True)
class DayResult:
    day: date
    dap: int
    eto_in: float
    kcb: float
    ke: float
    ks: float
    kc_act: float
    etc_in: float
    etc_adj_in: float
    rain_in: float
    runoff_in: float
    irrigation_in: float
    upflux_in: float
    deep_perc_in: float
    root_growth_in: float
    depletion_in: float
    taw_in: float
    raw_in: float
    mad: float
    root_depth_in: float
    stress: bool
    fraction_depleted: float


def _runoff_scs(rain_in: float, cn: float, depletion_in: float, taw_in: float) -> float:
    """Curve number runoff with a soil-moisture-adjusted retention parameter.

    Delaware's sandy Coastal Plain soils generate little Hortonian runoff, but
    the adjustment matters after a wet week on the finer Matapeake and
    Fallsington series.
    """
    if rain_in <= 0:
        return 0.0
    s_max = 1000.0 / cn - 10.0
    wetness = 1.0 - (depletion_in / taw_in if taw_in > 0 else 0.0)
    s = s_max * (1.0 - 0.6 * max(0.0, min(1.0, wetness)))
    ia = 0.05 * s          # 0.05 initial abstraction ratio, not the legacy 0.2
    if rain_in <= ia:
        return 0.0
    return round(((rain_in - ia) ** 2) / (rain_in - ia + s), 3)


def step_water_balance(
    state: BalanceState,
    weather: WeatherDay,
    crop: CropStage,
    soil: SoilProfile,
    dap: int,
    lat_deg: float,
    irrigation_in: float = 0.0,
    fw: float = 1.0,
    kcb_override: float | None = None,
    ks_override: float | None = None,
    curve_number: float = 74.0,
) -> DayResult:
    """Advance the two-layer water balance one day.

    kcb_override and ks_override are the hooks the Bayesian assimilation module
    uses to push sensor-derived coefficients into the balance.
    """
    eto = reference_et(weather, lat_deg)
    kcb = kcb_override if kcb_override is not None else crop.kcb(dap)
    root = crop.root_depth_in(dap)
    if soil.restrictive_layer_in:
        root = min(root, soil.restrictive_layer_in)

    # Root growth dilutes existing depletion over a larger volume of soil that
    # is assumed to be at the average water content of the profile below.
    root_growth_in = 0.0
    if root > state.root_depth_in and state.root_depth_in > 0:
        new_soil = root - state.root_depth_in
        root_growth_in = soil.awc_in_per_in * new_soil * 0.35
        state.dr_in += root_growth_in
    state.last_root_depth_in = state.root_depth_in
    state.root_depth_in = root

    taw = soil.taw_in(root)
    mad = crop.mad(dap)
    raw = soil.raw_in(root, mad)

    runoff = _runoff_scs(weather.rain_in, curve_number, state.dr_in, taw)
    infiltration = max(0.0, weather.rain_in - runoff) + irrigation_in

    # --- evaporation layer -------------------------------------------------
    few = max(0.01, min(1.0 - min(1.0, kcb / max(0.01, crop.kcb_mid)) * 0.94, fw))
    kc_max = max(1.20, kcb + 0.05)
    de_prev = max(0.0, state.de_in - infiltration)
    if de_prev < soil.rew_in:
        kr = 1.0
    elif de_prev >= soil.tew_in:
        kr = 0.0
    else:
        kr = (soil.tew_in - de_prev) / max(0.01, soil.tew_in - soil.rew_in)
    ke = min(kr * (kc_max - kcb), few * kc_max)
    ke = max(0.0, ke)
    e_in = ke * eto
    state.de_in = min(soil.tew_in, de_prev + e_in / max(0.01, few))

    # --- root zone ---------------------------------------------------------
    if ks_override is not None:
        ks = max(0.0, min(1.0, ks_override))
    else:
        dr_after_water = max(0.0, state.dr_in - infiltration)
        if dr_after_water <= raw:
            ks = 1.0
        else:
            ks = max(0.0, (taw - dr_after_water) / max(0.01, taw - raw))

    etc = kcb * eto
    etc_adj = (kcb * ks + ke) * eto

    upflux = soil.upflux_in(root, state.dr_in)

    # Eq 1 of the narrative:  D = D0 + ETc - P - Irr - U + qs + qw
    # state.dr_in already carries D0 plus the root-growth term, and `infiltration`
    # is (P - qs) + Irr, so this line is Eq 1 with qw split out below: deep
    # percolation is not an independent input, it is whatever cannot be held once
    # the profile reaches field capacity. Closure is asserted in the test suite.
    dr_new = state.dr_in - infiltration + etc_adj - upflux
    deep_perc = 0.0
    if dr_new < 0:
        deep_perc = -dr_new
        dr_new = 0.0
    dr_new = min(dr_new, taw)
    state.dr_in = dr_new

    return DayResult(
        day=weather.day,
        dap=dap,
        eto_in=round(eto, 3),
        kcb=round(kcb, 3),
        ke=round(ke, 3),
        ks=round(ks, 3),
        kc_act=round(kcb * ks + ke, 3),
        etc_in=round(etc, 3),
        etc_adj_in=round(etc_adj, 3),
        rain_in=round(weather.rain_in, 3),
        runoff_in=round(runoff, 3),
        irrigation_in=round(irrigation_in, 3),
        upflux_in=round(upflux, 4),
        deep_perc_in=round(deep_perc, 3),
        root_growth_in=round(root_growth_in, 4),
        depletion_in=round(dr_new, 3),
        taw_in=round(taw, 3),
        raw_in=round(raw, 3),
        mad=mad,
        root_depth_in=round(root, 1),
        stress=ks < 0.999,
        fraction_depleted=round(dr_new / taw, 3) if taw > 0 else 0.0,
    )


def run_season(
    weather: Sequence[WeatherDay],
    crop: CropStage,
    soil: SoilProfile,
    planting: date,
    lat_deg: float,
    irrigations: dict[date, float] | None = None,
    initial_depletion_in: float = 0.0,
    curve_number: float = 74.0,
) -> list[DayResult]:
    """Replay a season of weather to reconstruct where the field is today."""
    irrigations = irrigations or {}
    state = BalanceState(dr_in=initial_depletion_in,
                         de_in=0.0,
                         root_depth_in=crop.root_min_in)
    out: list[DayResult] = []
    for w in weather:
        dap = (w.day - planting).days
        if dap < 0:
            continue
        out.append(
            step_water_balance(
                state, w, crop, soil, dap, lat_deg,
                irrigation_in=irrigations.get(w.day, 0.0),
                curve_number=curve_number,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Canopy temperature stress indices (Activity 3)
# ---------------------------------------------------------------------------
# Activity 3 names two: "canopy-temperature-based water stress indices such as
# Crop Water Stress Index (CWSI) and Degree Above Non-Stressed (DANS)". CWSI is
# above; DANS is here, and the two answer different questions.
#
# CWSI needs a non-water-stressed baseline and an upper limit, both derived from
# vapour pressure deficit, and it is the more rigorous index — but the baselines
# are crop and location specific and getting them wrong on a humid Delmarva
# afternoon produces confident nonsense. DANS sidesteps that: it compares the
# canopy temperature of this field to the coolest well-watered canopy in the
# same image at the same moment. No baseline equation, no VPD regression, and it
# is robust to the sensor calibration drift that plagues thermal imagery.
#
# That makes DANS the better index for a UAV flight over a field with a
# well-watered reference strip, and CWSI the better one for a fixed infrared
# thermometer with a long record behind it.


def dans_index(canopy_temp_f: float, reference_temp_f: float,
               air_temp_f: float | None = None) -> float:
    """Degree Above Non-Stressed, in degrees Fahrenheit.

    `reference_temp_f` is the canopy temperature of well-watered vegetation in
    the same scene — a reference strip, the wettest zone of the field, or the
    lower percentile of the thermal image. Positive values mean this canopy is
    running hotter than one that is not short of water, which is transpiration
    having shut down.
    """
    dans = canopy_temp_f - reference_temp_f
    if air_temp_f is not None and canopy_temp_f > air_temp_f + 12.0:
        # A canopy more than 12 F above air is almost certainly bare soil in the
        # pixel rather than a transpiring crop.
        return round(min(dans, air_temp_f + 12.0 - reference_temp_f), 2)
    return round(dans, 2)


def ks_from_dans(dans_f: float, threshold_f: float = 1.5,
                 full_stress_f: float = 9.0) -> float:
    """Stress coefficient from DANS.

    Below the threshold the difference is measurement noise and canopy
    architecture, not stress. Above it, Ks falls roughly linearly to zero as the
    canopy approaches the fully non-transpiring case. The default endpoints are
    for corn; Activity 3 calls for these to be fitted against the LIRF record,
    and on Delmarva against a reference strip at Carvel.
    """
    if dans_f <= threshold_f:
        return 1.0
    span = max(0.1, full_stress_f - threshold_f)
    return round(max(0.0, min(1.0, 1.0 - (dans_f - threshold_f) / span)), 3)


def dans_from_thermal_image(canopy_temps_f: list[float],
                            well_watered_percentile: float = 0.05) -> dict:
    """DANS for every pixel of a thermal pass, referenced to its own cool tail.

    This is how DANS is meant to be used with UAV imagery: the reference comes
    out of the same image, so sensor drift, atmospheric correction and time of
    day all cancel. The cool tail is taken at the 5th percentile rather than the
    minimum, because the single coldest pixel in a thermal image is usually a
    puddle, a shadow or a dead sensor element.
    """
    if not canopy_temps_f:
        return {"n": 0}
    ordered = sorted(canopy_temps_f)
    idx = max(0, min(len(ordered) - 1,
                     int(well_watered_percentile * len(ordered))))
    reference = ordered[idx]
    values = [dans_index(t, reference) for t in canopy_temps_f]
    ks_values = [ks_from_dans(v) for v in values]
    n = len(values)
    mean_dans = sum(values) / n
    return {
        "n": n,
        "reference_temp_f": round(reference, 2),
        "mean_dans_f": round(mean_dans, 2),
        "max_dans_f": round(max(values), 2),
        "mean_ks": round(sum(ks_values) / n, 3),
        "fraction_stressed": round(sum(1 for k in ks_values if k < 0.95) / n, 3),
        "note": (
            "Reference taken from the coolest 5 percent of this image, so the "
            "index is internally calibrated and immune to sensor drift."
            if n > 40 else
            "Few pixels — the reference percentile is unreliable below about "
            "40 samples. Use a measured well-watered strip instead."),
    }
