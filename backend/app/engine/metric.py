# This file is a port of pyMETRIC and pyTSEB by Hector Nieto and contributors,
# both published under the GNU General Public License version 3. It is
# therefore a derivative work and carries the same licence, which is why
# Geo4Irr as a whole is GPL-3.0. See ATTRIBUTION.md at the repository root for
# what was ported from where, and for the one ambiguity in the upstream
# licence.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version. It is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See <http://www.gnu.org/licenses/>.

"""
METRIC: Mapping EvapoTranspiration at high Resolution with Internalized
Calibration.

A surface energy balance model. Where FAO-56 asks how much water the crop
*would* use given the weather and a coefficient, METRIC measures how much it
*did* use, by looking at how hot the ground is. Water leaving a surface takes
heat with it. A field that is transpiring freely runs cool; one that has run
out runs hot. METRIC reads that difference pixel by pixel.

    Rn = G + H + LE                 and therefore      LE = Rn - G - H

Rn is net radiation, G is heat going into the ground, H is sensible heat (the
part that warms the air), and LE is latent heat, which is the energy that went
into evaporating water. LE is solved as the residual, which is why the other
three have to be right.

PROVENANCE

Nothing in this module is invented. Every equation is ported, term by term,
from Hector Nieto's pyMETRIC and the pyTSEB library it depends on:

    https://github.com/hectornieto/pyMETRIC
    https://github.com/hectornieto/pyTSEB

which are in turn the reference implementation of:

    Allen, R.G., Tasumi, M., Trezza, R. (2007). Satellite-based energy balance
    for mapping evapotranspiration with internalized calibration (METRIC):
    Model. Journal of Irrigation and Drainage Engineering 133(4), 380-394.

    Allen, R.G., Burnett, B., Kramber, W., Huntington, J., Kjaersgaard, J.,
    Kilic, A., Kelly, C., Trezza, R. (2013). Automated calibration of the
    METRIC Landsat evapotranspiration process. JAWRA 49(3), 563-576.

    Tasumi, M., Allen, R.G., Trezza, R. (2008). At-surface reflectance and
    albedo from satellite for operational calculation of land surface energy
    balance. Journal of Hydrologic Engineering 13(2), 51-63.

    Brutsaert, W. (2005). Hydrology: an Introduction. Cambridge, for the
    Monin-Obukhov stability functions pyTSEB uses by default.

Each function carries the pyMETRIC or pyTSEB name it came from and the Allen
equation number where there is one, so that the two can be read side by side.
The port is to plain Python rather than numpy, because this module has to
import on a laptop where the optional scientific stack is not installed; the
one consequence is that the moving-window filter in the endmember search is a
loop rather than scipy's convolve2d.

FOUR DELIBERATE DEPARTURES FROM pyMETRIC

1. The sign of net longwave in the reference ET. pyMETRIC's `pet_asce`
   computes `Ln` with the FAO net-outgoing-longwave expression, which returns
   a positive number, and then forms `Rn = Sn + Ln`. ASCE subtracts it. At
   298 K with 20 mb vapour pressure and 800 W/m2 incoming shortwave the
   difference is 616 W/m2 against 680, so Rn comes out 23 percent high and
   reference ET with it. This module subtracts.

2. The aerodynamic resistance inside the per-pixel loop. In pyMETRIC's
   `METRIC()` the call that fills `R_A` sits inside the `else` branch, so with
   `use_METRIC_resistance=True`, which is the default, `R_A` is never assigned
   after initialisation and every H is NaN. This module computes R_A in both
   cases, which is plainly what was meant.

3. The wind profile. pyMETRIC takes wind at whatever height the configuration
   names and corrects from there. Allen (2007) Eqs 30 to 33 go through a 200 m
   blending height first: above it the wind has forgotten which field it
   crossed, which is the argument for applying one weather station across a
   whole scene. That step is written out here.

4. Where the cold anchor's ET comes from. pyMETRIC has no weather station, so
   it computes reference ET internally. This tool has one: the nearest DEOS
   station, the same station the FAO-56 side uses. Allen's own METRIC manual
   calibrates against station ETr, so the station value is used when it is
   available and the internal ASCE hourly equation is the fallback. Using one
   station for both methods is also what makes them comparable.

WHY LANDSAT AND NOT SENTINEL-2

METRIC needs land surface temperature. Sentinel-2 has no thermal band, so it
cannot do this at all, at any resolution, ever. That is not a limitation of
this implementation; it is what the satellite carries. Landsat 8 and 9 carry
TIRS, and Collection 2 Level-2 ships a ready-made surface temperature product
at 30 m. That is what the Delaware Geological Survey used for its Sussex
County ET study (Report of Investigations No. 84) and it is what this uses.

Sentinel-2 still earns its place for the vegetation indices, where its 5-day
revisit beats Landsat's 16-day badly. The two are complementary: NDVI tells
you where the crop is struggling, METRIC tells you how much water actually
left the field.

WHAT "INTERNALIZED CALIBRATION" MEANS

The hard term is H, which depends on the near-surface air temperature
difference dT, which nobody measures per pixel. METRIC gets around this by
picking two anchor pixels: a cold one, a well-watered field transpiring at
close to its potential, and a hot one, dry bare soil. At those two pixels ET
is known by assumption, so LE is known, so H is the residual, so dT follows.
A line dT = a + b*Ts through those two points is applied to every pixel.

The strength is that it absorbs most of the bias in the inputs: errors in
atmospheric correction, emissivity and the radiation terms hit the anchors and
the field alike and cancel in the ratio. The weakness is that the whole scene
rests on two pixel choices, so this module reports which pixels it picked and
why, rather than hiding them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Sequence

# ---------------------------------------------------------------------------
# constants, matching pyMETRIC/METRIC.py and pyTSEB
# ---------------------------------------------------------------------------

SOLAR_CONSTANT = 1367.0          # W/m2
SB = 5.670373e-8                 # Stefan-Boltzmann, W/m2/K4 (pyMETRIC value)
KARMAN = 0.41
GRAVITY = 9.8                    # pyTSEB MO_similarity uses 9.8, not 9.81
C_PD = 1003.5                    # specific heat of dry air, J/kg/K
C_PV = 1865.0                    # specific heat of water vapour, J/kg/K
EPSILON = 0.622                  # ratio of molecular weights, water to air
R_D = 287.04                     # gas constant for dry air, J/kg/K
RHO_WATER = 1000.0               # kg/m3

# Iteration control, from pyMETRIC/METRIC.py
L_THRES = 0.001                  # relative change in Monin-Obukhov length
U_FRICTION_MIN = 0.01            # m/s
ITERATIONS = 15

# METRIC's own resistance convention, Allen (2007) Eq 34: the temperature
# gradient is defined between 0.1 m and 2.0 m above the zero plane, with the
# displacement height taken as zero. pyMETRIC's `use_METRIC_resistance=True`
# branch hard-codes exactly these three numbers.
METRIC_Z_T = 2.0
METRIC_Z_0H = 0.1
METRIC_D_0 = 0.0
BLENDING_HEIGHT = 200.0          # Allen (2007) Eq 31
Z0M_STATION = 0.03               # clipped grass at a reference weather station

TALL_REFERENCE = 1               # alfalfa, ETr
SHORT_REFERENCE = 0              # clipped grass, ETo

# Allen (2013) Eq 4: the vegetation index at which a field is dense enough for
# the cold anchor to sit at the full 1.05 of reference ET.
VI_MAX = 0.95
ETRF_COLD_MAX = 1.05

# Landsat Collection 2 Level-2 scaling, from the USGS product guide. Applied
# here rather than assumed, because the reflectance offset is negative and
# silently skipping it shifts every albedo and every index.
L2_REFLECT_SCALE = 0.0000275
L2_REFLECT_OFFSET = -0.2
L2_TEMP_SCALE = 0.00341802
L2_TEMP_OFFSET = 149.0


def scale_reflectance(dn: float) -> float:
    return dn * L2_REFLECT_SCALE + L2_REFLECT_OFFSET


def scale_surface_temp_k(dn: float) -> float:
    return dn * L2_TEMP_SCALE + L2_TEMP_OFFSET


# ---------------------------------------------------------------------------
# meteorology, ported from pyTSEB/meteo_utils.py
#
# Pressure and vapour pressure are in millibars throughout this section,
# because that is the unit pyTSEB works in and mixing the two conventions is
# the easiest way to introduce a factor of ten that nothing catches.
# ---------------------------------------------------------------------------

def calc_pressure(z_m: float) -> float:
    """Atmospheric pressure from elevation, mb. pyTSEB calc_pressure."""
    return 1013.25 * (1.0 - 2.225577e-5 * z_m) ** 5.25588


def calc_vapor_pressure(t_k: float) -> float:
    """Saturation vapour pressure, mb. pyTSEB calc_vapor_pressure."""
    t_c = t_k - 273.15
    return 6.112 * math.exp((17.67 * t_c) / (t_c + 243.5))


def calc_delta_vapor_pressure(t_k: float) -> float:
    """Slope of the saturation vapour pressure curve, kPa/K.

    pyTSEB calc_delta_vapor_pressure. Multiply by 10 for mb/K, which is what
    the combination equation below does.
    """
    t_c = t_k - 273.15
    return (4098.0 * (0.6108 * math.exp(17.27 * t_c / (t_c + 237.3)))
            / ((t_c + 237.3) ** 2))


def calc_c_p(p_mb: float, ea_mb: float) -> float:
    """Heat capacity of moist air, J/kg/K. pyTSEB calc_c_p."""
    q = EPSILON * ea_mb / (p_mb + (EPSILON - 1.0) * ea_mb)
    return (1.0 - q) * C_PD + q * C_PV


def calc_lambda(t_a_k: float) -> float:
    """Latent heat of vaporisation, J/kg. pyTSEB calc_lambda.

    Temperature dependent rather than the flat 2.45e6 often used: at 30 C it
    is 2.430e6, which is a one percent difference in every ET figure.
    """
    return 1e6 * (2.501 - (2.361e-3 * (t_a_k - 273.15)))


def calc_rho(p_mb: float, ea_mb: float, t_a_k: float) -> float:
    """Air density, kg/m3. pyTSEB calc_rho."""
    return ((p_mb * 100.0) / (R_D * t_a_k)) * (1.0 - (1.0 - EPSILON) * ea_mb / p_mb)


def calc_psicr(c_p: float, p_mb: float, lambda_: float) -> float:
    """Psychrometric constant, mb/K. pyTSEB calc_psicr."""
    return c_p * p_mb / (EPSILON * lambda_)


def calc_mixing_ratio(ea_mb: float, p_mb: float) -> float:
    """pyTSEB calc_mixing_ratio."""
    return EPSILON * ea_mb / (p_mb - ea_mb)


def calc_lapse_rate_moist(t_a_k: float, ea_mb: float, p_mb: float) -> float:
    """Saturated adiabatic lapse rate, K/m. pyTSEB calc_lapse_rate_moist.

    Used to bring surface and air temperature to a common datum before the
    anchors are chosen, so that a pixel is not picked as hot merely for
    sitting lower. Sussex County spans about 20 m of relief, so this moves a
    temperature by a tenth of a degree and is kept for completeness rather
    than for its effect.
    """
    r = calc_mixing_ratio(ea_mb, p_mb)
    c_p = calc_c_p(p_mb, ea_mb)
    lambda_v = calc_lambda(t_a_k)
    return (GRAVITY * (R_D * t_a_k ** 2 + lambda_v * r * t_a_k)
            / (c_p * R_D * t_a_k ** 2 + lambda_v ** 2 * r * EPSILON))


def calc_stephan_boltzmann(t_k: float) -> float:
    """pyTSEB calc_stephan_boltzmann."""
    return SB * t_k ** 4


def flux_to_evaporation(flux_w_m2: float, t_k: float = 293.15,
                        hours: float = 1.0) -> float:
    """W/m2 to mm over a time window. pyTSEB flux_2_evaporation."""
    return flux_w_m2 / calc_lambda(t_k) * 3600.0 * hours


# ---------------------------------------------------------------------------
# Monin-Obukhov similarity, ported from pyTSEB/MO_similarity.py
#
# pyTSEB's default stability functions are Brutsaert (2005), not the
# Paulson/Businger forms written out in Allen (2007) Eqs 41 to 48. The
# Brutsaert forms are better behaved near free convection, which is why
# pyTSEB chose them, and they agree with Allen's to within a few percent over
# the range a Delaware morning produces. The Allen forms are kept below as
# `psi_m_allen2007` and `psi_h_allen2007` so the paper can be reproduced
# exactly when that is what is wanted.
# ---------------------------------------------------------------------------

def psi_m_brutsaert(zol: float) -> float:
    """Stability correction for momentum. pyTSEB psi_m_brutsaert."""
    if not math.isfinite(zol):
        return 0.0
    if zol >= 0.0:                      # stable and neutral
        a, b = 6.1, 2.5
        return -a * math.log(zol + (1.0 + zol ** b) ** (1.0 / b))
    y = -zol                            # unstable
    a, b = 0.33, 0.41
    x = (y / a) ** (1.0 / 3.0)
    psi_0 = -math.log(a) + math.sqrt(3.0) * b * a ** (1.0 / 3.0) * math.pi / 6.0
    y = min(y, b ** -3)
    return (math.log(a + y) - 3.0 * b * y ** (1.0 / 3.0)
            + (b * a ** (1.0 / 3.0)) / 2.0
            * math.log((1.0 + x) ** 2 / (1.0 - x + x ** 2))
            + math.sqrt(3.0) * b * a ** (1.0 / 3.0)
            * math.atan((2.0 * x - 1.0) / math.sqrt(3.0))
            + psi_0)


def psi_h_brutsaert(zol: float) -> float:
    """Stability correction for heat. pyTSEB psi_h_brutsaert."""
    if not math.isfinite(zol):
        return 0.0
    if zol >= 0.0:                      # stable and neutral
        a, b = 6.1, 2.5
        return -a * math.log(zol + (1.0 + zol ** b) ** (1.0 / b))
    y = -zol                            # unstable
    c, d, n = 0.33, 0.057, 0.78
    return ((1.0 - d) / n) * math.log((c + y ** n) / c)


def psi_m_allen2007(zol: float, z_m: float, L: float) -> float:
    """Allen (2007) Eqs 41 to 43, the Paulson/Businger form."""
    if not math.isfinite(L) or L == 0:
        return 0.0
    if L < 0:
        x = (1.0 - 16.0 * z_m / L) ** 0.25
        return (2.0 * math.log((1.0 + x) / 2.0)
                + math.log((1.0 + x ** 2) / 2.0)
                - 2.0 * math.atan(x) + 0.5 * math.pi)
    return -5.0 * z_m / L


def psi_h_allen2007(z_m: float, L: float) -> float:
    """Allen (2007) Eqs 44 to 48."""
    if not math.isfinite(L) or L == 0:
        return 0.0
    if L < 0:
        x = (1.0 - 16.0 * z_m / L) ** 0.25
        return 2.0 * math.log((1.0 + x ** 2) / 2.0)
    return -5.0 * z_m / L


def calc_u_star(u: float, z_u: float, L: float, d_0: float,
                z_0m: float) -> float:
    """Friction velocity. pyTSEB calc_u_star.

    Allen (2007) Eq 41 in the stability-corrected form. Note that the
    correction is evaluated twice, once at the measurement height and once at
    the roughness length, and the two are subtracted; dropping the second term
    is a common simplification and it biases u* low over rough canopies.
    """
    if L == 0.0:
        L = 1e-36
    psi_m = psi_m_brutsaert((z_u - d_0) / L)
    psi_m0 = psi_m_brutsaert(z_0m / L)
    denom = math.log((z_u - d_0) / z_0m) - psi_m + psi_m0
    if denom <= 0:
        # Stability has driven the profile past the point where a logarithmic
        # wind law means anything. Fall back to neutral rather than returning
        # a negative friction velocity.
        denom = math.log((z_u - d_0) / z_0m)
    return u * KARMAN / denom


def calc_L(ustar: float, t_a_k: float, rho: float, c_p: float,
           h: float, le: float) -> float:
    """Monin-Obukhov length. pyTSEB calc_L, via calc_mo_length_hv.

    The virtual sensible heat flux Hv = H + 0.61*T*cp*E is used rather than H
    alone. Water vapour is lighter than dry air, so evaporation makes the
    surface air buoyant on its own account; over a transpiring crop in July
    that term is not small, and leaving it out makes the surface look more
    stable than it is.
    """
    lambda_ = calc_lambda(t_a_k)
    e = le / lambda_
    hv = h + (0.61 * t_a_k * c_p * e)
    if hv == 0:
        return float("inf")
    l_const = KARMAN * GRAVITY / t_a_k
    return -ustar ** 3 / (l_const * (hv / (rho * c_p)))


def calc_R_A(z_t: float, ustar: float, L: float, d_0: float,
             z_0h: float) -> float:
    """Aerodynamic resistance to heat transport, s/m. pyTSEB calc_R_A.

    Allen (2007) Eq 34 with the stability corrections applied.
    """
    if ustar == 0:
        return float("inf")
    if L == 0:
        L = 1e-36
    r_a_log = math.log((z_t - d_0) / z_0h)
    psi_h = psi_h_brutsaert((z_t - d_0) / L)
    psi_h0 = psi_h_brutsaert(z_0h / L)
    return (r_a_log - psi_h + psi_h0) / (ustar * KARMAN)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Pixel:
    """One pixel's surface properties, already scaled to physical units."""
    blue: float
    green: float
    red: float
    nir: float
    swir1: float
    swir2: float
    lst_k: float                 # land surface temperature, Kelvin
    emissivity: float = 0.98
    elevation_m: float = 12.0    # Sussex County sits close to sea level

    @property
    def ndvi(self) -> float:
        d = self.nir + self.red
        return 0.0 if abs(d) < 1e-9 else (self.nir - self.red) / d

    @property
    def vi(self) -> float:
        """The vegetation index the endmember search works on.

        pyMETRIC calls this VI and feeds NDVI into it, so this is NDVI with
        the name the reference implementation uses.
        """
        return self.ndvi

    @property
    def savi(self) -> float:
        """Soil-adjusted vegetation index, L = 0.5.

        METRIC uses SAVI rather than NDVI for leaf area index because NDVI
        saturates once a canopy closes, and LAI drives both the roughness
        length and the soil heat flux.
        """
        d = self.nir + self.red + 0.5
        return 0.0 if abs(d) < 1e-9 else 1.5 * (self.nir - self.red) / d

    @property
    def lai(self) -> float:
        """Leaf area index from SAVI, Allen (2007) Eq 19, capped at 6.

        The cap is in the original: beyond about 6 the relationship has no
        information left in it, and an uncapped value lets one noisy pixel
        produce a physically impossible canopy.
        """
        s = max(0.0, min(0.687, self.savi))
        if s >= 0.687:
            return 6.0
        inner = (0.69 - s) / 0.59
        if inner <= 0:
            return 6.0
        return min(6.0, -math.log(inner) / 0.91)


@dataclass(frozen=True, slots=True)
class SceneWeather:
    """Ground weather at the overpass, and for the day around it.

    METRIC is calibrated against reference ET, so these come from the nearest
    DEOS station, which is the same station the FAO-56 side of this tool uses.
    Using one station for both is what lets the two methods be compared.
    """
    overpass: date
    overpass_hour_local: float   # decimal hours, Landsat crosses near 10:30
    air_temp_c: float
    wind_speed_mps: float        # at the measurement height below
    wind_measure_height_m: float = 2.0
    rh_pct: float = 60.0

    # Reference ET at the overpass hour and across the whole day. BOTH MUST BE
    # THE SAME REFERENCE SURFACE. The ET fraction is a ratio taken at the
    # instant of the overpass and then applied to the daily total, so if the
    # numerator is grass reference and the denominator alfalfa, every result is
    # wrong by the ratio between them, which is about 1.2 in a Delaware summer.
    # An early version of this module did exactly that and inflated seasonal
    # totals by a fifth. `reference` records which surface is in use and the
    # model refuses to run on a mismatch.
    ref_hourly_mm: float = 0.0
    ref_daily_mm: float = 0.0
    reference: str = "grass"     # "grass" (ETo) or "alfalfa" (ETr)

    def __post_init__(self):
        if self.reference not in ("grass", "alfalfa"):
            raise ValueError(
                f"Reference surface {self.reference!r} is neither 'grass' nor "
                "'alfalfa'. METRIC is usually run on alfalfa reference; DEOS "
                "publishes grass. Either works as long as both the hourly and "
                "the daily figure come from the same one.")

    @property
    def air_temp_k(self) -> float:
        return self.air_temp_c + 273.15

    @property
    def ea_mb(self) -> float:
        """Actual vapour pressure from relative humidity, mb."""
        return calc_vapor_pressure(self.air_temp_k) * max(0.0, min(100.0, self.rh_pct)) / 100.0

    @property
    def reference_code(self) -> int:
        return TALL_REFERENCE if self.reference == "alfalfa" else SHORT_REFERENCE


# ---------------------------------------------------------------------------
# radiation
# ---------------------------------------------------------------------------

def broadband_albedo(p: Pixel) -> float:
    """Surface albedo from at-surface reflectance. Tasumi et al. (2008).

    Weights are each band's share of the total shortwave energy for Landsat.
    They sum to 1, which is asserted in the tests because a typo in one of
    them shifts every net radiation in the scene by a few percent and nothing
    else would notice.

    pyMETRIC takes a different route, backing albedo out of a canopy
    radiative-transfer model as 1 - Rns/Sdn, because it is built on TSEB's
    two-source machinery. Tasumi is the published METRIC method and needs
    nothing but the reflectances Collection 2 already provides.
    """
    return (0.254 * p.blue + 0.149 * p.green + 0.147 * p.red
            + 0.311 * p.nir + 0.103 * p.swir1 + 0.036 * p.swir2)


def solar_declination(day_of_year: int) -> float:
    return 0.409 * math.sin(2 * math.pi * day_of_year / 365.0 - 1.39)


def solar_incidence_angle_cos(lat_deg: float, day_of_year: int,
                              hour_local: float,
                              slope_rad: float = 0.0,
                              aspect_rad: float = 0.0) -> float:
    """Cosine of the solar incidence angle on the surface. Allen (2007) Eq 7.

    Delaware's Coastal Plain is close to flat, so slope and aspect default to
    zero and the expression collapses to the horizontal case. The terms are
    kept because the same code should not need rewriting for anywhere else.
    """
    lat = math.radians(lat_deg)
    dec = solar_declination(day_of_year)
    hour_angle = math.radians(15.0 * (hour_local - 12.0))

    if slope_rad == 0.0:
        cos_theta = (math.sin(dec) * math.sin(lat)
                     + math.cos(dec) * math.cos(lat) * math.cos(hour_angle))
    else:
        cos_theta = (math.sin(dec) * math.sin(lat) * math.cos(slope_rad)
                     - math.sin(dec) * math.cos(lat) * math.sin(slope_rad)
                       * math.cos(aspect_rad)
                     + math.cos(dec) * math.cos(lat) * math.cos(slope_rad)
                       * math.cos(hour_angle)
                     + math.cos(dec) * math.sin(lat) * math.sin(slope_rad)
                       * math.cos(aspect_rad) * math.cos(hour_angle)
                     + math.cos(dec) * math.sin(aspect_rad)
                       * math.sin(slope_rad) * math.sin(hour_angle))
    return max(0.0, min(1.0, cos_theta))


def solar_zenith_angle_deg(lat_deg: float, day_of_year: int,
                           hour_local: float) -> float:
    """Solar zenith angle, degrees. Needed by the CIMEC albedo threshold."""
    return math.degrees(math.acos(
        max(-1.0, min(1.0, solar_incidence_angle_cos(lat_deg, day_of_year,
                                                     hour_local)))))


def atmospheric_transmissivity(elevation_m: float, cos_theta: float,
                               precipitable_water_mm: float = 15.0,
                               kt: float = 1.0) -> float:
    """Broadband one-way transmissivity, Allen (2007) Eq 4.

    15 mm of precipitable water is a reasonable default for a humid Delmarva
    summer morning. The term matters less than it looks: it moves incoming
    shortwave by a couple of percent, and the internalized calibration absorbs
    most of a systematic error in it. kt is the turbidity coefficient, 1.0 for
    clean air.
    """
    if cos_theta <= 0:
        return 0.0
    p_mb = calc_pressure(elevation_m)
    w = precipitable_water_mm
    tau = 0.35 + 0.627 * math.exp(
        -0.00146 * (p_mb / 10.0) / (kt * cos_theta)
        - 0.075 * (w / cos_theta) ** 0.4)
    return max(0.0, min(1.0, tau))


def net_radiation(p: Pixel, wx: SceneWeather, lat_deg: float,
                  day_of_year: int) -> dict:
    """Net radiation at the surface, W/m2. Allen (2007) Eq 5.

    Written as pyMETRIC writes it:

        Ln = emis * L_dn - emis * sigma * Tr^4
        Rn = Sn + Ln

    which is algebraically the same as the form in the paper,

        Rn = (1-a)Rs_in + RL_in - RL_out - (1-e0)RL_in

    because emis * L_dn is exactly L_dn minus the share the surface reflects.
    Dropping that reflected term biases Rn high across the whole scene.
    """
    cos_theta = solar_incidence_angle_cos(lat_deg, day_of_year,
                                          wx.overpass_hour_local)
    dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365.0)
    tau = atmospheric_transmissivity(p.elevation_m, cos_theta)
    rs_in = SOLAR_CONSTANT * cos_theta * dr * tau

    albedo = broadband_albedo(p)
    air_k = wx.air_temp_k

    # Effective atmospheric emissivity, Allen (2007) Eq 17.
    eps_a = 0.85 * (-math.log(tau)) ** 0.09 if 0 < tau < 1 else 0.85
    l_dn = eps_a * SB * air_k ** 4

    sn = (1 - albedo) * rs_in
    ln_net = p.emissivity * l_dn - p.emissivity * calc_stephan_boltzmann(p.lst_k)
    rn = sn + ln_net

    return {"rn": rn, "rs_in": rs_in, "sn": sn, "l_dn": l_dn,
            "ln_net": ln_net, "albedo": albedo, "tau": tau,
            "cos_theta": cos_theta}


def calc_G_Allen(rn: float, lst_k: float, albedo: float, ndvi: float) -> float:
    """Ground heat flux, W/m2. pyMETRIC calc_G_Allen, Allen (2007) Eq 26.

        G = Rn * (Ts - 273.15) * (0.0038 + 0.0074a) * (1 - 0.98 NDVI^4)

    Note Ts enters in Celsius here while everything else is in Kelvin. That
    inconsistency is in the published equation, not a mistake: it is an
    empirical fit and the coefficients carry the units.

    pyMETRIC applies no bound to the result. A bound is applied here because
    over water NDVI goes negative, the fit stops meaning anything, and an
    unbounded G can exceed Rn and drive LE negative. A farm pond inside a
    pivot boundary is a real case on Delmarva, not a hypothetical.
    """
    ts_c = lst_k - 273.15
    ratio = (ts_c * (0.0038 + 0.0074 * albedo)
             * (1 - 0.98 * max(0.0, ndvi) ** 4))
    return max(0.05, min(0.5, ratio)) * rn


def soil_heat_flux(p: Pixel, rn: float) -> float:
    """calc_G_Allen with the terms pulled off a Pixel."""
    return calc_G_Allen(rn, p.lst_k, broadband_albedo(p), p.ndvi)


def calc_H(dt: float, rho: float, c_p: float, r_ah: float) -> float:
    """Sensible heat flux from the bulk transfer equation.

    pyMETRIC calc_H, Allen (2007) Eq 28.
    """
    return rho * c_p * dt / r_ah


def calc_H_residual(rn: float, g: float, le: float = 0.0) -> float:
    """Sensible heat as the residual of the energy balance. pyMETRIC."""
    return rn - g - le


def calc_dT(h: float, r_ah: float, rho: float, c_p: float) -> float:
    """Near-surface temperature difference. pyMETRIC calc_dT, Allen Eqs 46, 49."""
    return h * r_ah / (rho * c_p)


def calc_Ln(t_a_k: float, ea_mb: float, f_cd: float = 1.0) -> float:
    """Net OUTGOING longwave for a reference surface, W/m2, positive.

    pyMETRIC calc_Ln. The sign is the point: this returns the magnitude of the
    outgoing net, so ASCE subtracts it from net shortwave. pyMETRIC's
    `pet_asce` adds it, which is departure 1 in the module docstring.
    """
    return SB * f_cd * (0.34 - 0.14 * math.sqrt(max(0.0, ea_mb) * 0.1)) * t_a_k ** 4


def calc_cloudiness(s_dn: float, s_0: float) -> float:
    """Cloudiness factor from measured against clear-sky shortwave. pyMETRIC."""
    if s_0 <= 0:
        return 1.0
    return max(0.05, min(1.0, 1.35 * s_dn / s_0 - 0.35))


def wind_profile(u: float, z_u: float, z_0m: float, d: float,
                 z: float) -> float:
    """Neutral logarithmic wind profile. pyMETRIC wind_profile."""
    return u * math.log((z - d) / z_0m) / math.log((z_u - d) / z_0m)


def momentum_roughness(p: Pixel) -> float:
    """z0m from LAI, Allen (2007) Eq 36.

    Floor of 0.005 m stops bare soil producing a roughness so small the
    aerodynamic resistance explodes and H with it.
    """
    return max(0.005, 0.018 * p.lai)


def wind_at_blending_height(wind_mps: float, measure_height_m: float,
                            blending_m: float = BLENDING_HEIGHT,
                            z0m_station: float = Z0M_STATION) -> float:
    """Wind at the blending height. Allen (2007) Eqs 30 to 32.

    0.03 m is the roughness of the clipped grass a reference weather station
    stands on, which is what makes the extrapolation to 200 m legitimate:
    above that height the wind has forgotten what surface it came from, so one
    station's wind can be applied across the whole scene.
    """
    ustar_station = (KARMAN * wind_mps) / math.log(measure_height_m / z0m_station)
    return ustar_station * math.log(blending_m / z0m_station) / KARMAN


# ---------------------------------------------------------------------------
# reference ET at the overpass
# ---------------------------------------------------------------------------

def asce_hourly_reference_et(t_a_k: float, u_mps: float, ea_mb: float,
                             p_mb: float, s_dn_w_m2: float,
                             z_u: float = 2.0, f_cd: float = 1.0,
                             reference: int = TALL_REFERENCE) -> float:
    """ASCE standardized hourly reference ET, mm/h.

    This is pyMETRIC's `pet_asce` with the net longwave subtracted rather than
    added, and with the psychrometric term written in the canonical ASCE form
    (ASCE-EWRI 2005 Eq 1):

        ET = [0.408 D (Rn - G) + g (Cn/T) u2 (es - ea)]
             / [D + g (1 + Cd u2)]

    with Rn and G in MJ/m2/h, T in Kelvin, u2 in m/s, vapour pressures in kPa
    and the curve slope and psychrometric constant in kPa/K. pyMETRIC's
    version drops the 1 inside the denominator's bracket and the 0.408, having
    rescaled to W/m2; the rescaling is fine, the dropped terms are not.

    Daytime coefficients are used because Landsat crosses mid-morning: Cn 66
    and Cd 0.25 for alfalfa, Cn 37 and Cd 0.24 for clipped grass.
    """
    delta = calc_delta_vapor_pressure(t_a_k)            # kPa/K
    lambda_ = calc_lambda(t_a_k)                        # J/kg
    c_p = calc_c_p(p_mb, ea_mb)
    gamma = calc_psicr(c_p, p_mb, lambda_) / 10.0       # mb/K to kPa/K
    es = calc_vapor_pressure(t_a_k) / 10.0              # mb to kPa
    ea = ea_mb / 10.0

    if reference == TALL_REFERENCE:
        g_ratio, h_c, c_d, c_n = 0.04, 0.50, 0.25, 66.0
    else:
        g_ratio, h_c, c_d, c_n = 0.10, 0.12, 0.24, 37.0

    albedo = 0.23
    sn = s_dn_w_m2 * (1.0 - albedo)
    rnl = calc_Ln(t_a_k, ea_mb, f_cd=f_cd)
    rn_w = sn - rnl
    g_w = g_ratio * rn_w

    # W/m2 to MJ/m2/h
    rn = rn_w * 0.0036
    g = g_w * 0.0036

    z_0m = h_c * 0.123
    d = h_c * 2.0 / 3.0
    u2 = wind_profile(u_mps, z_u, z_0m, d, 2.0) if z_u != 2.0 else u_mps

    num = 0.408 * delta * (rn - g) + gamma * (c_n / t_a_k) * u2 * (es - ea)
    den = delta + gamma * (1.0 + c_d * u2)
    return max(0.0, num / den)


def etrf_cold(vi: float) -> float:
    """Cold anchor ET fraction. Allen (2013) Eq 4, pyMETRIC ET_r_f_cold.

    1.05 of alfalfa reference for a dense canopy, tapering linearly to zero as
    the vegetation index falls below 0.95. The 1.05 is not a fudge: a
    well-watered field at full cover transpires slightly above the reference
    because it is taller and rougher than the reference crop. The taper is
    what stops an early-season scene, where nothing has closed, from being
    anchored as though it had.
    """
    if vi >= VI_MAX:
        return ETRF_COLD_MAX
    return ETRF_COLD_MAX / VI_MAX * max(0.0, vi)


def etrf_hot(vi: float, etrf_bare: float = 0.0) -> float:
    """Hot anchor ET fraction. Allen (2013) Eq 5, pyMETRIC ET_r_f_hot.

        ETrF_hot = VI * ETrF_cold + (1 - VI) * ETrF_bare

    ETrF_bare is not always zero. Bare soil two days after an inch of rain is
    still evaporating, and anchoring it at zero forces the dT line too steep
    and inflates ET everywhere else in the scene. It comes from the surface
    layer of the FAO-56 balance this tool already runs, which is the one place
    the two methods genuinely share a number.
    """
    return vi * etrf_cold(vi) + (1.0 - vi) * max(0.0, etrf_bare)


def bare_soil_etrf(rain_mm_by_day: Sequence[float],
                   ref_mm_by_day: Sequence[float],
                   rew_mm: float = 8.0) -> float:
    """ET fraction of dry bare soil on the overpass day.

    Allen (2013) Eq 5 needs this and pyMETRIC takes it from configuration. It
    is not zero: bare soil two days after an inch of rain is still evaporating
    hard, and anchoring the hot pixel at zero tilts the dT line too steep and
    inflates ET across the whole scene.

    The model is the surface-layer half of FAO-56, which is the same mechanism
    the Ke term in this tool's water balance already uses, so the two methods
    share one number rather than making two different guesses. Readily
    evaporable water, 8 mm for a medium-textured soil and the figure FAO-56
    Table 19 gives for a Delaware sandy loam, is spent at close to the
    reference rate after a wetting; once it is gone, evaporation drops to a
    trickle.

    The two sequences run oldest to newest and end on the overpass day.
    """
    if not rain_mm_by_day or not ref_mm_by_day or rew_mm <= 0:
        return 0.0
    n = min(len(rain_mm_by_day), len(ref_mm_by_day))
    depletion = rew_mm      # assume the surface was dry entering the window
    fraction = 0.05
    for i in range(-n, 0):
        # Rain first, then the fraction that rain allows, then the day's own
        # evaporation at that fraction. Deducting the day's evaporation before
        # reporting its rate is circular, and it was: an earlier version of
        # this returned the dry floor for ground rained on yesterday.
        depletion = max(0.0, depletion - rain_mm_by_day[i])
        remaining = max(0.0, rew_mm - depletion) / rew_mm
        fraction = max(0.05, 0.9 * remaining)
        depletion = min(rew_mm, depletion + fraction * ref_mm_by_day[i])
    return round(fraction, 3)


# ---------------------------------------------------------------------------
# endmember search, ported from pyMETRIC/endmember_search.py
# ---------------------------------------------------------------------------

CIMEC = 0
MINMAX = 2


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * pct / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def moving_cv_filter(grid: Sequence[Sequence[float | None]],
                     window: tuple[int, int] = (11, 11)
                     ) -> tuple[list[list[float]], list[list[float]],
                                list[list[float]]]:
    """Coefficient of variation in a moving window. pyMETRIC moving_cv_filter.

    A direct loop rather than scipy's convolve2d, with masked cells skipped
    instead of symmetric padding. The purpose is the same: a candidate anchor
    pixel sitting in a patch that varies wildly is probably a field edge, a
    building or a cloud shadow, and should lose to one in a uniform patch.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    half_r, half_c = window[0] // 2, window[1] // 2
    cv = [[float("nan")] * cols for _ in range(rows)]
    mean = [[float("nan")] * cols for _ in range(rows)]
    std = [[float("nan")] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] is None:
                continue
            vals = []
            for rr in range(max(0, r - half_r), min(rows, r + half_r + 1)):
                for cc in range(max(0, c - half_c), min(cols, c + half_c + 1)):
                    v = grid[rr][cc]
                    if v is not None and math.isfinite(v):
                        vals.append(v)
            if not vals:
                continue
            m = sum(vals) / len(vals)
            s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
            mean[r][c] = m
            std[r][c] = s
            cv[r][c] = s / m if m != 0 else float("nan")
    return cv, mean, std


def maxmin_temperature(vi: Sequence[float], lst: Sequence[float],
                       vi_lower_limit: float | None = None
                       ) -> tuple[int | None, int | None]:
    """The simple endmember search. pyMETRIC maxmin_temperature (MINMAX).

    Coldest pixel among those above a vegetation-index floor, hottest among
    those below it. Used when there is no spatial grid to measure homogeneity
    on, which is the case for a single small field.
    """
    valid = [i for i in range(len(lst))
             if math.isfinite(lst[i]) and math.isfinite(vi[i])]
    if not valid:
        return None, None
    if vi_lower_limit is None:
        vi_lower_limit = _percentile([vi[i] for i in valid], 0.05)
    cold_pool = [i for i in valid if vi[i] >= vi_lower_limit]
    hot_pool = [i for i in valid if vi[i] <= vi_lower_limit]
    if not cold_pool or not hot_pool:
        return None, None
    cold = min(cold_pool, key=lambda i: lst[i])
    hot = max(hot_pool, key=lambda i: lst[i])
    return cold, hot


def cimec_albedo_threshold(sza_deg: float) -> float:
    """Expected albedo of a well-watered dense canopy. Allen (2013) Eq 7."""
    beta = 90.0 - sza_deg            # solar elevation angle
    return 0.001343 * beta + 0.3281 * math.exp(-0.0188 * beta)


def cimec(vi: Sequence[float], lst: Sequence[float], albedo: Sequence[float],
          sza_deg: float, cv_vi: Sequence[float], cv_lst: Sequence[float],
          adjust_rainfall: tuple[float, float] | None = None
          ) -> tuple[int | None, int | None, list[str]]:
    """Calibration using Inverse Modelling at Extreme Conditions.

    pyMETRIC `cimec`, from Allen et al. (2013), the automated calibration that
    replaced hand-picking anchors. The steps are the paper's:

    Cold anchor. Take the top 5 percent of the vegetation index. Of those take
    the coldest 20 percent and average their temperature. Keep candidates
    within 0.2 K of that average whose albedo is within 0.02 of the value a
    dense well-watered canopy should have at this sun angle, Eq 7. Of those
    keep the one sitting in the most uniform temperature neighbourhood.

    Hot anchor. Take the bottom 10 percent of the vegetation index. Of those
    take the hottest 20 percent and average. Keep candidates within 0.2 K of
    that average, and of those the one in the most uniform vegetation-index
    neighbourhood.

    The albedo test is what makes this better than picking the coldest green
    pixel: open water and deep cloud shadow are both cold and both have the
    wrong albedo, and either one as a cold anchor depresses every ET in the
    scene.

    `adjust_rainfall` is (rainfall_60, ETr_60), the ratio of the last 60 days
    of rain to the last 60 days of reference ET. It lowers the hot anchor's
    temperature when the recent past has been wet, Eq 8, because bare soil
    after rain is not fully dry.
    """
    notes: list[str] = []
    valid = [i for i in range(len(lst))
             if math.isfinite(lst[i]) and math.isfinite(vi[i])]
    if not valid:
        return None, None, ["No pixel had both a temperature and a vegetation index."]

    # --- cold anchor ---
    vi_top = _percentile([vi[i] for i in valid], 95)
    vi_index = [i for i in valid if vi[i] >= vi_top]
    if not vi_index:
        return None, None, ["No pixel cleared the 95th percentile of NDVI."]

    lst_low = _percentile([lst[i] for i in vi_index], 20)
    lst_index = [i for i in valid if lst[i] <= lst_low]
    if not lst_index:
        return None, None, ["No pixel fell in the coldest fifth of the green pixels."]
    lst_cold = sum(lst[i] for i in lst_index) / len(lst_index)

    alb_thres = cimec_albedo_threshold(sza_deg)
    cold_candidates = [i for i in lst_index
                       if abs(lst[i] - lst_cold) <= 0.2
                       and abs(albedo[i] - alb_thres) <= 0.02]
    if not cold_candidates:
        # Relax the albedo test before relaxing the temperature test: a 30 m
        # pixel of Delaware corn beside a treeline is a legitimate anchor with
        # an albedo a little off the open-field expectation.
        cold_candidates = [i for i in lst_index if abs(lst[i] - lst_cold) <= 0.2]
        notes.append(
            f"No candidate matched the expected canopy albedo of "
            f"{alb_thres:.3f} within 0.02 at a solar zenith of {sza_deg:.0f} "
            "degrees, so the albedo test was dropped for the cold anchor. "
            "Check that the pixel chosen is not water or cloud shadow.")
    if not cold_candidates:
        cold_candidates = lst_index
        notes.append("The 0.2 K window around the cold average was empty, so "
                     "the whole cold fifth was used.")
    cold = min(cold_candidates,
               key=lambda i: (cv_lst[i] if math.isfinite(cv_lst[i]) else 1e9))

    # --- hot anchor ---
    vi_low = _percentile([vi[i] for i in valid], 10)
    vi_index = [i for i in valid if vi[i] <= vi_low]
    if not vi_index:
        return None, None, notes + ["No pixel fell below the 10th percentile of NDVI."]

    lst_high = _percentile([lst[i] for i in vi_index], 80)
    lst_index = [i for i in valid if lst[i] >= lst_high]
    if not lst_index:
        return None, None, notes + ["No pixel fell in the hottest fifth of the bare pixels."]
    lst_hot = sum(lst[i] for i in lst_index) / len(lst_index)

    if adjust_rainfall is not None:
        rain_60, etr_60 = adjust_rainfall
        if etr_60 > 0:
            shift = 2.6 - 13.0 * rain_60 / etr_60      # Allen (2013) Eq 8
            lst_hot -= shift
            notes.append(
                f"The hot anchor temperature was moved by {-shift:+.2f} K for "
                f"60 days of rain against reference ET at "
                f"{rain_60 / etr_60:.2f}. Bare soil after rain is not dry.")

    hot_candidates = [i for i in lst_index if abs(lst[i] - lst_hot) <= 0.2]
    if not hot_candidates:
        hot_candidates = lst_index
        notes.append("The 0.2 K window around the hot average was empty, so "
                     "the whole hot fifth was used.")
    hot = max(hot_candidates,
              key=lambda i: (-cv_vi[i] if math.isfinite(cv_vi[i]) else -1e9))

    return cold, hot, notes


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Anchor:
    """One calibration pixel, with the evidence for choosing it."""
    kind: str                    # "cold" or "hot"
    index: int
    pixel: Pixel
    ndvi: float
    lst_k: float
    albedo: float
    etrf: float
    reason: str


def pick_anchors(pixels: Sequence[Pixel], wx: SceneWeather, lat_deg: float,
                 grid_shape: tuple[int, int] | None = None,
                 etrf_bare: float = 0.0,
                 method: int = CIMEC,
                 rain_60_mm: float | None = None,
                 ref_et_60_mm: float | None = None
                 ) -> tuple[Anchor, Anchor, list[str], str]:
    """Choose the cold and hot calibration pixels.

    CIMEC when the pixels arrive as a grid, because its homogeneity tests need
    neighbours. MINMAX otherwise, which is what a single small field in a flat
    list reduces to.

    This is where METRIC is most often got wrong, so the choice is returned
    with its reasoning rather than applied invisibly. Two failure modes worth
    knowing about. A cold pixel taken over open water gives an ET no field can
    reach and depresses every ETrF in the scene; CIMEC's albedo test is aimed
    squarely at that. A hot pixel taken over a roof or a road is hotter than
    any soil and stretches the dT line, inflating ET everywhere; on Delmarva,
    where fields sit beside towns and poultry houses, that is a real risk and
    the reason the hot pixel should be checked against the imagery.
    """
    if len(pixels) < 2:
        raise ValueError(
            "METRIC needs at least two pixels to calibrate against. One pixel "
            "cannot anchor a scene.")

    vi = [p.vi for p in pixels]
    lst = [p.lst_k for p in pixels]
    alb = [broadband_albedo(p) for p in pixels]
    notes: list[str] = []

    doy = wx.overpass.timetuple().tm_yday
    sza = solar_zenith_angle_deg(lat_deg, doy, wx.overpass_hour_local)

    used = "MINMAX"
    cold_i = hot_i = None

    if method == CIMEC and grid_shape is not None:
        rows, cols = grid_shape
        if rows * cols == len(pixels) and rows >= 3 and cols >= 3:
            vi_grid = [[vi[r * cols + c] for c in range(cols)] for r in range(rows)]
            lst_grid = [[lst[r * cols + c] for c in range(cols)] for r in range(rows)]
            window = (min(11, rows | 1), min(11, cols | 1))
            cv_vi_g, _, _ = moving_cv_filter(vi_grid, window)
            cv_lst_g, _, _ = moving_cv_filter(lst_grid, window)
            cv_vi = [cv_vi_g[r][c] for r in range(rows) for c in range(cols)]
            cv_lst = [cv_lst_g[r][c] for r in range(rows) for c in range(cols)]
            adjust = None
            if rain_60_mm is not None and ref_et_60_mm:
                adjust = (rain_60_mm, ref_et_60_mm)
            cold_i, hot_i, cimec_notes = cimec(vi, lst, alb, sza,
                                               cv_vi, cv_lst, adjust)
            notes.extend(cimec_notes)
            if cold_i is not None and hot_i is not None:
                used = "CIMEC"
            else:
                notes.append(
                    "The automated CIMEC search could not separate anchors in "
                    "this scene, so the simpler coldest-green against "
                    "hottest-bare search was used instead.")
                cold_i = hot_i = None

    if cold_i is None or hot_i is None:
        cold_i, hot_i = maxmin_temperature(vi, lst)
        used = "MINMAX"

    if cold_i is None or hot_i is None:
        raise ValueError(
            "No pair of anchor pixels could be found in this scene. Every "
            "pixel is either masked or they all share one vegetation index, "
            "which means there is no contrast to calibrate on.")

    cp, hp = pixels[cold_i], pixels[hot_i]
    if hp.lst_k <= cp.lst_k:
        raise ValueError(
            f"The hot anchor ({hp.lst_k:.1f} K) is not warmer than the cold "
            f"anchor ({cp.lst_k:.1f} K). METRIC calibrates on the temperature "
            "difference between them, so there is nothing to calibrate on. "
            "This usually means the scene is under cloud, or the thermal band "
            "did not scale correctly.")

    f_cold = etrf_cold(cp.vi)
    f_hot = etrf_hot(hp.vi, etrf_bare)

    cold_reason = (
        f"{used} search. NDVI {cp.vi:.2f}, albedo {alb[cold_i]:.3f}, "
        f"{cp.lst_k - 273.15:.1f} C. Anchored at {f_cold:.2f} of alfalfa "
        f"reference ET (Allen 2013 Eq 4).")
    if cp.vi < VI_MAX:
        cold_reason += (
            f" The canopy has not closed, so the anchor was tapered down from "
            f"1.05 rather than assumed to be transpiring at full rate.")

    hot_reason = (
        f"{used} search. NDVI {hp.vi:.2f}, albedo {alb[hot_i]:.3f}, "
        f"{hp.lst_k - 273.15:.1f} C. Anchored at {f_hot:.3f} of alfalfa "
        f"reference ET (Allen 2013 Eq 5) with bare-soil evaporation at "
        f"{etrf_bare:.2f}.")

    return (Anchor("cold", cold_i, cp, round(cp.ndvi, 3), round(cp.lst_k, 2),
                   round(alb[cold_i], 4), round(f_cold, 3), cold_reason),
            Anchor("hot", hot_i, hp, round(hp.ndvi, 3), round(hp.lst_k, 2),
                   round(alb[hot_i], 4), round(f_hot, 3), hot_reason),
            notes, used)


# ---------------------------------------------------------------------------
# the dT calibration
# ---------------------------------------------------------------------------

def solve_dt_coefficients(cold: Anchor, hot: Anchor, wx: SceneWeather,
                          lat_deg: float, day_of_year: int,
                          le_ref_w_m2: float,
                          iterations: int = ITERATIONS) -> dict:
    """Fit dT = a + b * Ts through the two anchors.

    At each anchor, ET is known by assumption, so LE is known, so H is the
    residual of the energy balance, so dT follows from H = rho cp dT / rah.
    Two points give the line.

    The loop is the Monin-Obukhov stability correction at the anchors.
    Buoyancy changes the aerodynamic resistance, the resistance changes H, and
    H changes the buoyancy, so it has to be iterated. pyMETRIC stops when the
    relative change in the Monin-Obukhov length falls below 0.001 or after 15
    passes, and that is what happens here.

    The intercept is taken from the hot anchor,

        b = (dT_hot - dT_cold) / (Ts_hot - Ts_cold)
        a = dT_hot - b * Ts_hot

    which is pyMETRIC's arrangement. Its author notes that Eqs 50 and 51 as
    printed in Allen et al. (2007) appear to be wrong; for a line through two
    points the arrangement makes no difference to the result, which is a
    useful thing to be able to say.
    """
    p_mb = calc_pressure(cold.pixel.elevation_m)
    ea = wx.ea_mb
    t_a_k = wx.air_temp_k
    rho = calc_rho(p_mb, ea, t_a_k)
    c_p = calc_c_p(p_mb, ea)
    u200 = wind_at_blending_height(wx.wind_speed_mps, wx.wind_measure_height_m)

    records = []
    for anchor in (cold, hot):
        rad = net_radiation(anchor.pixel, wx, lat_deg, day_of_year)
        rn = rad["rn"]
        g = soil_heat_flux(anchor.pixel, rn)
        le = anchor.etrf * le_ref_w_m2
        h = calc_H_residual(rn, g, le)
        z_0m = momentum_roughness(anchor.pixel)

        L = float("inf")
        ustar = max(U_FRICTION_MIN,
                    calc_u_star(u200, BLENDING_HEIGHT, L, METRIC_D_0, z_0m))
        n = 0
        for n in range(iterations):
            L_new = calc_L(ustar, t_a_k, rho, c_p, h, le)
            if math.isfinite(L) and L != 0:
                if abs(L_new - L) / abs(L) < L_THRES:
                    L = L_new
                    break
            L = L_new
            ustar = max(U_FRICTION_MIN,
                        calc_u_star(u200, BLENDING_HEIGHT, L, METRIC_D_0, z_0m))

        r_ah = calc_R_A(METRIC_Z_T, ustar, L, METRIC_D_0, METRIC_Z_0H)
        dt = calc_dT(h, r_ah, rho, c_p)
        records.append({"anchor": anchor, "rn": rn, "g": g, "le": le, "h": h,
                        "z_0m": z_0m, "ustar": ustar, "L": L, "rah": r_ah,
                        "dt": dt, "iterations": n + 1})

    dt_cold, dt_hot = records[0]["dt"], records[1]["dt"]
    b = (dt_hot - dt_cold) / (hot.lst_k - cold.lst_k)
    a = dt_hot - b * hot.lst_k

    return {"a": a, "b": b, "rho": rho, "c_p": c_p, "u200": u200,
            "p_mb": p_mb, "ea_mb": ea,
            "cold": records[0], "hot": records[1],
            "dt_cold": round(dt_cold, 3), "dt_hot": round(dt_hot, 3)}


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

@dataclass
class PixelET:
    index: int
    ndvi: float
    lst_k: float
    albedo: float
    rn: float
    g: float
    h: float
    le: float
    r_ah: float
    dt: float
    mo_length: float
    iterations: int
    et_inst_mm_hr: float
    etrf: float
    eta_mm_day: float


@dataclass
class MetricScene:
    overpass: date
    pixels: list[PixelET] = dc_field(default_factory=list)
    cold: Anchor | None = None
    hot: Anchor | None = None
    calibration: dict = dc_field(default_factory=dict)
    warnings: list[str] = dc_field(default_factory=list)

    @property
    def mean_eta_mm_day(self) -> float:
        if not self.pixels:
            return 0.0
        return sum(p.eta_mm_day for p in self.pixels) / len(self.pixels)

    @property
    def mean_etrf(self) -> float:
        if not self.pixels:
            return 0.0
        return sum(p.etrf for p in self.pixels) / len(self.pixels)

    @property
    def mean_eta_in_day(self) -> float:
        return self.mean_eta_mm_day / 25.4


# A well-watered full-cover field against each reference surface. Allen (2013)
# Eq 4 gives 1.05 against alfalfa. Against clipped grass the same physical ET
# is a larger multiple, because the reference itself is shorter and uses less
# water; the ratio in a Delaware summer is close to 1.2. Getting this wrong
# scales the entire scene, so it is a named constant rather than a literal.
REFERENCE_CEILING = {"alfalfa": 1.05, "grass": 1.26}
ALFALFA_TO_GRASS = 1.2


def run_metric(pixels: Sequence[Pixel], wx: SceneWeather, lat_deg: float,
               grid_shape: tuple[int, int] | None = None,
               etrf_bare: float = 0.0,
               method: int = CIMEC,
               rain_60_mm: float | None = None,
               ref_et_60_mm: float | None = None,
               iterations: int = ITERATIONS) -> MetricScene:
    """Run the energy balance over a scene and return per-pixel actual ET.

    The per-pixel loop is pyMETRIC's: given dT from the calibration line,
    iterate the aerodynamic resistance, the sensible heat flux and the
    Monin-Obukhov length against each other until the length stops moving.
    Earlier versions of this module interpolated the resistance between the
    two anchors instead, which is cheaper and is not METRIC: the resistance
    depends on the pixel's own roughness and its own buoyancy, and a bare
    field and a closed canopy at the same surface temperature do not share one.

    The daily figure comes from the reference ET fraction, not from
    extrapolating the instantaneous flux. ETrF is close to constant through a
    clear day while the fluxes themselves are not, so

        ETa_day = ETrF * ETref_day

    is far more stable than multiplying an instantaneous rate by daylight
    hours. This is the standard METRIC approach and it is the reason a single
    mid-morning overpass can produce a daily total at all.
    """
    if not pixels:
        raise ValueError("No pixels were given to the energy balance.")

    doy = wx.overpass.timetuple().tm_yday
    scene = MetricScene(overpass=wx.overpass)
    t_a_k = wx.air_temp_k
    p_mb = calc_pressure(pixels[0].elevation_m)
    ea = wx.ea_mb

    # No sun, no energy balance. Every radiation term is built on the solar
    # incidence angle, so with the sun below the horizon net radiation is
    # strongly negative, the ground heat flux changes sign, and the reference
    # ET the calibration divides by collapses to a rounding error. Landsat
    # does occasionally acquire at night, and a scene like that has to be
    # refused rather than run: the numbers would look plausible and mean
    # nothing.
    scene_cos_theta = solar_incidence_angle_cos(lat_deg, doy,
                                                wx.overpass_hour_local)
    if scene_cos_theta <= 0.01:
        raise ValueError(
            f"The overpass at {wx.overpass_hour_local:.1f} hours local on "
            f"{wx.overpass.isoformat()} puts the sun on or below the horizon, "
            "so there is no incoming shortwave radiation to drive an energy "
            "balance and no daylight reference ET to calibrate against. "
            "Landsat 8 and 9 cross Delaware near 10:30 in the morning; check "
            "that the overpass time was read as local rather than UTC.")

    # Reference ET at the instant of the overpass, as latent heat. The station
    # value is preferred; the ASCE hourly equation is the fallback. Either way
    # it has to be on the same reference surface as the daily figure.
    lam = calc_lambda(t_a_k)
    if wx.ref_hourly_mm > 0:
        ref_hourly_mm = wx.ref_hourly_mm
        ref_source = "station"
    else:
        dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365.0)
        tau = atmospheric_transmissivity(pixels[0].elevation_m, scene_cos_theta)
        s_dn = SOLAR_CONSTANT * scene_cos_theta * dr * tau
        ref_hourly_mm = asce_hourly_reference_et(
            t_a_k, wx.wind_speed_mps, ea, p_mb, s_dn,
            z_u=wx.wind_measure_height_m, reference=wx.reference_code)
        ref_source = "ASCE hourly equation"
        scene.warnings.append(
            "No hourly reference ET came from the weather station, so it was "
            f"computed from the station's temperature, humidity and wind: "
            f"{ref_hourly_mm:.3f} mm/h. A measured value is better, because "
            "the calibration rests on it.")
    if ref_hourly_mm <= 0:
        raise ValueError(
            "Reference ET at the overpass came out at zero. METRIC calibrates "
            "against it, so without it the cold anchor has no defined ET and "
            "the whole scene is unanchored. Check the overpass hour: a value "
            "outside daylight gives no incoming radiation.")

    # Anchor fractions in Allen (2013) are against alfalfa. If the station
    # publishes grass reference ET, as DEOS does, the fractions have to be
    # converted before they are multiplied by it.
    to_reference = 1.0 if wx.reference == "alfalfa" else 1.0 / ALFALFA_TO_GRASS
    le_ref_w_m2 = (ref_hourly_mm / 3600.0) * lam * RHO_WATER / 1000.0 / to_reference

    cold, hot, anchor_notes, search = pick_anchors(
        pixels, wx, lat_deg, grid_shape=grid_shape, etrf_bare=etrf_bare,
        method=method, rain_60_mm=rain_60_mm, ref_et_60_mm=ref_et_60_mm)
    scene.cold, scene.hot = cold, hot
    scene.warnings.extend(anchor_notes)

    cal = solve_dt_coefficients(cold, hot, wx, lat_deg, doy, le_ref_w_m2,
                                iterations=iterations)
    rho, c_p = cal["rho"], cal["c_p"]
    a, b = cal["a"], cal["b"]
    u200 = cal["u200"]

    scene.calibration = {
        "a": a, "b": b, "rho": round(rho, 4), "c_p": round(c_p, 1),
        "u200": round(u200, 3), "dt_cold": cal["dt_cold"],
        "dt_hot": cal["dt_hot"], "reference": wx.reference,
        "reference_et_hourly_mm": round(ref_hourly_mm, 4),
        "reference_et_source": ref_source,
        "le_reference_w_m2": round(le_ref_w_m2, 1),
        "endmember_search": search,
        "etrf_bare": etrf_bare,
        "cold_anchor_fraction": cold.etrf,
        "hot_anchor_fraction": hot.etrf,
        "anchor_iterations": [cal["cold"]["iterations"], cal["hot"]["iterations"]],
        "rah_cold": round(cal["cold"]["rah"], 2),
        "rah_hot": round(cal["hot"]["rah"], 2),
    }

    daily_ref = wx.ref_daily_mm
    if daily_ref <= 0:
        scene.warnings.append(
            "No daily reference ET was supplied, so daily actual ET could not "
            "be computed. The instantaneous rate and the reference ET fraction "
            "are still valid.")

    ref_inst_for_ratio = ref_hourly_mm
    ceiling = REFERENCE_CEILING[wx.reference]
    not_converged = 0

    for i, p in enumerate(pixels):
        rad = net_radiation(p, wx, lat_deg, doy)
        rn = rad["rn"]
        g = soil_heat_flux(p, rn)
        z_0m = momentum_roughness(p)
        dt = a + b * p.lst_k

        L = float("inf")
        ustar = max(U_FRICTION_MIN,
                    calc_u_star(u200, BLENDING_HEIGHT, L, METRIC_D_0, z_0m))
        r_ah = calc_R_A(METRIC_Z_T, ustar, L, METRIC_D_0, METRIC_Z_0H)
        h = le = 0.0
        converged = False
        n = 0

        for n in range(iterations):
            r_ah = calc_R_A(METRIC_Z_T, ustar, L, METRIC_D_0, METRIC_Z_0H)
            h = calc_H(dt, rho, c_p, r_ah)
            # H cannot exceed the available energy: more than that would mean
            # the surface condensing water, which a dry field in July is not
            # doing. Nor can it be negative over a surface warmer than the air.
            h = max(0.0, min(h, rn - g))
            le = rn - g - h
            L_new = calc_L(ustar, t_a_k, rho, c_p, h, le)
            if math.isfinite(L) and L != 0 and abs(L_new - L) / abs(L) < L_THRES:
                L = L_new
                converged = True
                break
            L = L_new
            ustar = max(U_FRICTION_MIN,
                        calc_u_star(u200, BLENDING_HEIGHT, L, METRIC_D_0, z_0m))

        if not converged:
            not_converged += 1

        et_inst = max(0.0, flux_to_evaporation(le, t_a_k, hours=1.0))  # mm/h
        etrf = et_inst / ref_inst_for_ratio if ref_inst_for_ratio else 0.0
        # An ETrF above the reference ceiling is not physical for a field crop.
        # It usually means the cold anchor was too cold, often open water.
        etrf = max(0.0, min(ceiling, etrf))
        eta_day = etrf * daily_ref if daily_ref > 0 else 0.0

        scene.pixels.append(PixelET(
            index=i, ndvi=round(p.ndvi, 3), lst_k=round(p.lst_k, 2),
            albedo=round(rad["albedo"], 4), rn=round(rn, 1), g=round(g, 1),
            h=round(h, 1), le=round(le, 1), r_ah=round(r_ah, 2),
            dt=round(dt, 3),
            mo_length=round(L, 1) if math.isfinite(L) else float("inf"),
            iterations=n + 1,
            et_inst_mm_hr=round(et_inst, 4), etrf=round(etrf, 3),
            eta_mm_day=round(eta_day, 2)))

    saturated = sum(1 for p in scene.pixels if p.etrf >= ceiling - 1e-6)
    if saturated > len(scene.pixels) * 0.1:
        scene.warnings.append(
            f"{saturated} of {len(scene.pixels)} pixels hit the upper limit on "
            "the reference ET fraction. The cold anchor is probably too cold, "
            "which happens when it lands on open water or a wet ditch.")
    if not_converged > len(scene.pixels) * 0.1:
        scene.warnings.append(
            f"The Monin-Obukhov length did not settle within {iterations} "
            f"passes on {not_converged} of {len(scene.pixels)} pixels. Their "
            "fluxes come from the last pass and are less certain than the rest.")

    return scene


# ---------------------------------------------------------------------------
# season totals
# ---------------------------------------------------------------------------

@dataclass
class SeasonET:
    planting: date
    through: date
    harvest: date | None
    scenes: list[dict]
    total_eta_mm: float
    total_eta_in: float
    days: int
    gap_days: int
    method: str
    weekly: list[dict] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)


def accumulate_season(scene_dates: Sequence[date],
                      scene_eta_mm_day: Sequence[float],
                      planting: date,
                      through: date,
                      daily_reference_mm: dict[date, float] | None = None,
                      harvest: date | None = None) -> SeasonET:
    """Total actual ET from planting to a chosen date.

    Landsat passes every 16 days, two satellites make that eight, and clouds
    take a share, so a Delaware season might give six to ten usable scenes.
    Something has to fill the gaps between them, and the choice matters more
    than the energy balance itself for a seasonal total.

    Interpolating actual ET straight across a gap is wrong: it assumes the
    weather in between resembled the two clear days at the ends, and clear
    days are not a random sample of the days in a Delaware July. Interpolating
    the reference ET fraction and multiplying by each day's own reference ET is
    better, because ETrF changes slowly with the canopy while reference ET
    carries the day-to-day weather. That is the standard METRIC approach and it
    is what this does when daily reference ET is available.

    Without it, the function falls back to interpolating ETa directly and says
    so in the warnings rather than quietly producing a worse number.
    """
    if len(scene_dates) != len(scene_eta_mm_day):
        raise ValueError("Each scene needs exactly one ET value.")
    if through < planting:
        raise ValueError("The end date is before planting.")

    end = min(through, harvest) if harvest else through
    pairs = sorted((d, v) for d, v in zip(scene_dates, scene_eta_mm_day)
                   if planting <= d <= end)
    warnings: list[str] = []

    if not pairs:
        return SeasonET(planting, end, harvest, [], 0.0, 0.0,
                        (end - planting).days + 1, 0, "none", [],
                        ["No usable satellite scenes fell between planting and "
                         "this date. Nothing can be accumulated."])

    if daily_reference_mm:
        method = "ETrF interpolation"
        points = []
        for d, eta in pairs:
            ref = daily_reference_mm.get(d, 0.0)
            points.append((d, eta / ref if ref > 0 else 0.0))
    else:
        method = "direct ETa interpolation"
        points = list(pairs)
        warnings.append(
            "No daily reference ET was supplied, so actual ET was interpolated "
            "directly between scenes. That assumes the days between two clear "
            "overpasses behaved like the overpass days, and clear days are not "
            "a random sample of a Delaware July. Supply daily reference ET for "
            "a defensible seasonal total.")

    total = 0.0
    covered = 0
    daily: list[tuple[date, float]] = []
    day = planting
    while day <= end:
        value = _interpolate(points, day)
        if daily_reference_mm:
            eta = value * daily_reference_mm.get(day, 0.0)
            if day in daily_reference_mm:
                covered += 1
        else:
            eta = value
            covered += 1
        total += eta
        daily.append((day, eta))
        day = date.fromordinal(day.toordinal() + 1)

    total_days = (end - planting).days + 1
    gap = total_days - covered
    if gap > 0 and daily_reference_mm:
        warnings.append(
            f"{gap} of {total_days} days had no reference ET and contributed "
            "nothing to the total, so this figure is low by roughly that share.")

    first, last = pairs[0][0], pairs[-1][0]
    if (first - planting).days > 20:
        warnings.append(
            f"The first usable scene is {(first - planting).days} days after "
            "planting. Early-season ET before it is extrapolated from that "
            "scene and is the least reliable part of the total.")
    if (end - last).days > 20:
        warnings.append(
            f"The last usable scene is {(end - last).days} days before the end "
            "date, so the tail is extrapolated.")
    if harvest and through > harvest:
        warnings.append(
            f"Accumulation stopped at harvest on {harvest.isoformat()} rather "
            "than running to the date requested.")

    return SeasonET(
        planting=planting, through=end, harvest=harvest,
        scenes=[{"date": d.isoformat(), "eta_mm_day": round(v, 2),
                 "eta_in_day": round(v / 25.4, 3)} for d, v in pairs],
        total_eta_mm=round(total, 1), total_eta_in=round(total / 25.4, 2),
        days=total_days, gap_days=gap, method=method,
        weekly=weekly_means(daily), warnings=warnings)


def weekly_means(daily: Sequence[tuple[date, float]]) -> list[dict]:
    """Average daily ETa by calendar week, with the week's total.

    Weekly is the resolution a Delmarva irrigator actually acts on, and it is
    also the honest resolution for a 16-day satellite: a single day's figure
    from an interpolated ETrF carries a precision the method does not have,
    while a weekly mean does not pretend to.
    """
    if not daily:
        return []
    out: list[dict] = []
    bucket: list[tuple[date, float]] = []
    for d, v in daily:
        if bucket and d.isocalendar()[:2] != bucket[0][0].isocalendar()[:2]:
            out.append(_week_record(bucket))
            bucket = []
        bucket.append((d, v))
    if bucket:
        out.append(_week_record(bucket))
    return out


def _week_record(bucket: Sequence[tuple[date, float]]) -> dict:
    total = sum(v for _, v in bucket)
    iso = bucket[0][0].isocalendar()
    return {
        "week_start": bucket[0][0].isoformat(),
        "week_end": bucket[-1][0].isoformat(),
        "iso_week": f"{iso[0]}-W{iso[1]:02d}",
        "days": len(bucket),
        "mean_eta_mm_day": round(total / len(bucket), 2),
        "mean_eta_in_day": round(total / len(bucket) / 25.4, 3),
        "total_eta_mm": round(total, 1),
        "total_eta_in": round(total / 25.4, 2),
    }


def _interpolate(points: Sequence[tuple[date, float]], when: date) -> float:
    """Linear between the bracketing scenes, flat outside them.

    Flat rather than extrapolated on purpose: running a trend past the last
    observation produces a confident number with nothing behind it, and over a
    20 day tail it can go negative.
    """
    if not points:
        return 0.0
    if when <= points[0][0]:
        return points[0][1]
    if when >= points[-1][0]:
        return points[-1][1]
    for (d0, v0), (d1, v1) in zip(points, points[1:]):
        if d0 <= when <= d1:
            span = (d1 - d0).days
            if span == 0:
                return v0
            f = (when - d0).days / span
            return v0 + (v1 - v0) * f
    return points[-1][1]
