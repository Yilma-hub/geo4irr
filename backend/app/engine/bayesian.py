"""
Recursive Bayesian assimilation of smart sensing data into the water balance.

This is Activity 5 of the project narrative, specialized to Delaware. The state
vector the filter carries forward is

    x = [ Dr, b_kcb, Ks ]

where Dr is root zone depletion (in), b_kcb is a multiplicative bias on the
tabular basal crop coefficient, and Ks is the stress coefficient. The forecast
step is the FAO-56 water balance itself, so the filter is constrained by the
physics rather than free to fit anything - that is the "science-guided" part.

Observations, each with its own error variance measured against gravimetric and
neutron probe reference data:

    z1 : root zone depletion inferred from capacitance or TDR soil moisture probes
    z2 : basal crop coefficient inferred from UAV or Sentinel-2 canopy cover / NDVI
    z3 : stress coefficient inferred from canopy temperature (CWSI or DANS)

The filter is a scalar-decoupled Kalman update rather than a full covariance
propagation. This is deliberate. With 20,000 potential fields the update has to
run in microseconds inside a nightly batch, and the cross-covariances between
depletion and a canopy-derived Kcb bias are weak enough at a daily time step
that the decoupled form loses very little. Where a field carries a dense sensor
array, `EnsembleAssimilator` runs the full particle version instead.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from .fao56 import (
    BalanceState, CropStage, DayResult, SoilProfile, WeatherDay,
    kcb_from_canopy_cover, kcb_from_ndvi, ks_from_cwsi, step_water_balance,
)


# ---------------------------------------------------------------------------
# Sensor error model (Activity 4: characterize errors in smart sensing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SensorClass:
    """Measurement error characteristics for a class of device.

    sigma_vwc is the standard deviation of volumetric water content error in
    v/v after a field calibration, from side-by-side comparison against
    gravimetric sampling. Uncalibrated capacitance probes in sandy soils with
    variable bulk EC are considerably worse than the factory sheet claims,
    which is why the cheap-sensor tier carries a wide sigma here. Cost is
    installed cost per station.
    """

    key: str
    label: str
    sigma_vwc: float
    cost_usd: float
    depths_in: tuple[int, ...]
    telemetry: str
    salinity_sensitive: bool = False

    def sigma_depletion_in(self, root_depth_in: float, n_depths: int | None = None) -> float:
        """Propagate VWC error to a root zone depletion error in inches.

        Errors at separate depths are treated as partially independent; the
        0.6 exponent is fitted to replicate probe data rather than assuming
        either full independence (0.5) or full correlation (1.0).
        """
        n = n_depths or len(self.depths_in)
        return round(self.sigma_vwc * root_depth_in / max(1, n) ** 0.6, 3)


SENSOR_CLASSES: dict[str, SensorClass] = {
    "tdr_research": SensorClass(
        "tdr_research", "Research-grade TDR profile probe",
        sigma_vwc=0.010, cost_usd=2400, depths_in=(6, 12, 18, 24, 36),
        telemetry="cellular",
    ),
    "capacitance_pro": SensorClass(
        "capacitance_pro", "Multi-depth capacitance probe, field calibrated",
        sigma_vwc=0.022, cost_usd=1100, depths_in=(4, 8, 12, 20, 32),
        telemetry="cellular", salinity_sensitive=True,
    ),
    "capacitance_basic": SensorClass(
        "capacitance_basic", "Single-point capacitance sensor, factory calibration",
        sigma_vwc=0.045, cost_usd=280, depths_in=(8, 18),
        telemetry="lorawan", salinity_sensitive=True,
    ),
    "granular_matrix": SensorClass(
        "granular_matrix", "Granular matrix / watermark tension block",
        sigma_vwc=0.055, cost_usd=90, depths_in=(8, 18, 30),
        telemetry="manual",
    ),
    "neutron_probe": SensorClass(
        "neutron_probe", "Neutron probe, access tube",
        sigma_vwc=0.008, cost_usd=0, depths_in=(6, 12, 18, 24, 36, 48),
        telemetry="manual",
    ),
}

# Canopy products and their Kcb error.
CANOPY_SOURCES: dict[str, float] = {
    "uav_multispectral": 0.045,     # 2 cm ground sample distance, calibrated panel
    "uav_rgb_cover": 0.070,
    "sentinel2_10m": 0.090,
    "planet_3m": 0.075,
    "landsat_30m": 0.120,
    "handheld_ceptometer": 0.060,
}

# Canopy temperature products and their Ks error.
THERMAL_SOURCES: dict[str, float] = {
    "uav_thermal": 0.070,
    "infrared_thermometer_fixed": 0.055,
    "landsat_thermal": 0.150,
}


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Observation:
    """A single assimilable measurement for one field on one day."""

    day: date
    kind: str                    # soil_moisture | canopy_cover | ndvi | canopy_temp
    value: float
    sensor_key: str | None = None
    depths_reported: int | None = None
    quality: float = 1.0         # 0-1 multiplier; drops for stale or flagged data

    def variance(self, root_depth_in: float, kcb_mid: float) -> float:
        q = max(0.05, min(1.0, self.quality))
        if self.kind == "soil_moisture":
            sc = SENSOR_CLASSES.get(self.sensor_key or "capacitance_basic")
            sigma = sc.sigma_depletion_in(root_depth_in, self.depths_reported)
        elif self.kind in ("canopy_cover", "ndvi"):
            sigma = CANOPY_SOURCES.get(self.sensor_key or "sentinel2_10m", 0.10)
        elif self.kind == "canopy_temp":
            sigma = THERMAL_SOURCES.get(self.sensor_key or "uav_thermal", 0.10)
        else:
            sigma = 0.15
        return (sigma / q) ** 2


def depletion_from_vwc(
    vwc_by_depth: dict[int, float],
    soil: SoilProfile,
    root_depth_in: float,
) -> float:
    """Convert a profile of volumetric water contents to root zone depletion.

    Each reading represents the layer midway to its neighbours. Readings below
    the current root front are ignored, which is what keeps an early-season
    corn field from looking wetter than the roots can actually reach.
    """
    depths = sorted(d for d in vwc_by_depth if d <= root_depth_in + 4)
    if not depths:
        return 0.0
    depletion = 0.0
    prev_bound = 0.0
    for i, d in enumerate(depths):
        upper = prev_bound
        lower = (d + depths[i + 1]) / 2.0 if i + 1 < len(depths) else root_depth_in
        lower = min(lower, root_depth_in)
        if lower <= upper:
            continue
        thickness = lower - upper
        theta = max(soil.wp_vol, min(soil.fc_vol, vwc_by_depth[d]))
        depletion += (soil.fc_vol - theta) * thickness
        prev_bound = lower
    return round(max(0.0, depletion), 3)


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FilterState:
    dr_in: float = 0.0
    var_dr: float = 0.25          # inches^2
    kcb_bias: float = 1.0
    var_kcb_bias: float = 0.02
    ks: float = 1.0
    var_ks: float = 0.02
    updates: int = 0
    last_obs_day: date | None = None


@dataclass(slots=True)
class AssimilationResult:
    day: date
    prior_depletion_in: float
    posterior_depletion_in: float
    depletion_sd_in: float
    kcb_used: float
    ks_used: float
    n_observations: int
    innovation_in: float
    notes: list[str] = field(default_factory=list)


class BayesianAssimilator:
    """Recursive Bayesian update of the water balance state.

    Process noise grows the model's own uncertainty each day it runs open loop,
    which is the mechanism that makes a field with no sensors report honestly
    wide prediction intervals instead of false precision.
    """

    def __init__(
        self,
        soil: SoilProfile,
        crop: CropStage,
        lat_deg: float,
        process_noise_frac: float = 0.12,
        kcb_random_walk: float = 0.0015,
    ) -> None:
        self.soil = soil
        self.crop = crop
        self.lat = lat_deg
        self.q_frac = process_noise_frac
        self.q_kcb = kcb_random_walk
        self.fs = FilterState()
        self.bs = BalanceState(root_depth_in=crop.root_min_in)
        self.history: list[DayResult] = []

    # -- forecast step ------------------------------------------------------

    def _forecast(
        self,
        weather: WeatherDay,
        dap: int,
        irrigation_in: float,
        curve_number: float,
    ) -> DayResult:
        kcb_prior = self.crop.kcb(dap) * self.fs.kcb_bias
        self.bs.dr_in = self.fs.dr_in
        res = step_water_balance(
            self.bs, weather, self.crop, self.soil, dap, self.lat,
            irrigation_in=irrigation_in,
            kcb_override=kcb_prior,
            curve_number=curve_number,
        )
        self.fs.dr_in = res.depletion_in
        self.fs.ks = res.ks
        # Process noise: a fraction of the day's ET, plus a rainfall term because
        # gauge-to-field rainfall error is the single largest source of drift on
        # a Delaware pivot in a summer air-mass thunderstorm.
        q = (self.q_frac * res.etc_adj_in) ** 2 + (0.25 * res.rain_in) ** 2
        self.fs.var_dr += q
        self.fs.var_kcb_bias += self.q_kcb
        return res

    # -- update step --------------------------------------------------------

    def _update_scalar(self, prior: float, var_prior: float,
                       obs: float, var_obs: float) -> tuple[float, float, float]:
        """Standard Kalman gain for a scalar Gaussian state and observation."""
        k = var_prior / (var_prior + var_obs)
        innovation = obs - prior
        return prior + k * innovation, (1.0 - k) * var_prior, innovation

    def assimilate_day(
        self,
        weather: WeatherDay,
        dap: int,
        observations: Sequence[Observation] = (),
        irrigation_in: float = 0.0,
        curve_number: float = 74.0,
    ) -> AssimilationResult:
        res = self._forecast(weather, dap, irrigation_in, curve_number)
        prior_dr = self.fs.dr_in
        notes: list[str] = []
        innovation_total = 0.0
        n_used = 0

        root = res.root_depth_in
        taw = res.taw_in

        for ob in observations:
            if ob.day != weather.day:
                continue
            var_obs = ob.variance(root, self.crop.kcb_mid)

            if ob.kind == "soil_moisture":
                obs_dr = max(0.0, min(taw, ob.value))
                post, var_post, innov = self._update_scalar(
                    self.fs.dr_in, self.fs.var_dr, obs_dr, var_obs)
                # Gross error screen: a jump larger than four combined standard
                # deviations is a probe that got bumped by a tillage pass or a
                # pivot tower, not a real change in the soil.
                gate = 4.0 * math.sqrt(self.fs.var_dr + var_obs)
                if abs(innov) > gate:
                    notes.append(
                        f"Soil moisture reading rejected: {innov:+.2f} in from model, "
                        f"outside the {gate:.2f} in gate. Check the probe.")
                    continue
                self.fs.dr_in, self.fs.var_dr = max(0.0, post), var_post
                innovation_total += innov
                n_used += 1

            elif ob.kind in ("canopy_cover", "ndvi"):
                kcb_obs = (kcb_from_canopy_cover(ob.value, self.crop.kcb_mid)
                           if ob.kind == "canopy_cover"
                           else kcb_from_ndvi(ob.value, kcb_max=self.crop.kcb_mid))
                kcb_tab = max(0.01, self.crop.kcb(dap))
                bias_obs = kcb_obs / kcb_tab
                var_bias = var_obs / (kcb_tab ** 2)
                post, var_post, innov = self._update_scalar(
                    self.fs.kcb_bias, self.fs.var_kcb_bias, bias_obs, var_bias)
                self.fs.kcb_bias = max(0.4, min(1.8, post))
                self.fs.var_kcb_bias = var_post
                n_used += 1
                if abs(self.fs.kcb_bias - 1.0) > 0.15:
                    direction = "ahead of" if self.fs.kcb_bias > 1 else "behind"
                    notes.append(
                        f"Canopy is {direction} the table curve; water use adjusted "
                        f"{abs(self.fs.kcb_bias - 1) * 100:.0f}%.")

            elif ob.kind == "canopy_temp":
                ks_obs = ks_from_cwsi(ob.value)
                post, var_post, innov = self._update_scalar(
                    self.fs.ks, self.fs.var_ks, ks_obs, var_obs)
                self.fs.ks = max(0.0, min(1.0, post))
                self.fs.var_ks = var_post
                n_used += 1
                if self.fs.ks < 0.9:
                    notes.append(
                        f"Canopy temperature indicates stress (Ks {self.fs.ks:.2f}) "
                        "before the water balance predicted it.")

        if n_used:
            self.fs.updates += 1
            self.fs.last_obs_day = weather.day
            self.bs.dr_in = self.fs.dr_in

        self.history.append(res)
        return AssimilationResult(
            day=weather.day,
            prior_depletion_in=round(prior_dr, 3),
            posterior_depletion_in=round(self.fs.dr_in, 3),
            depletion_sd_in=round(math.sqrt(max(1e-6, self.fs.var_dr)), 3),
            kcb_used=round(res.kcb, 3),
            ks_used=round(self.fs.ks, 3),
            n_observations=n_used,
            innovation_in=round(innovation_total, 3),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Ensemble variant for instrumented fields
# ---------------------------------------------------------------------------

class EnsembleAssimilator:
    """Particle filter over depletion, Kcb bias and rainfall multiplier.

    Used on demonstration and research fields where the extra fidelity is worth
    the compute: it carries a non-Gaussian depletion distribution, which matters
    right after an irrigation set when the posterior is truncated at zero.
    """

    def __init__(self, soil: SoilProfile, crop: CropStage, lat_deg: float,
                 n_particles: int = 250, seed: int | None = None) -> None:
        self.soil, self.crop, self.lat = soil, crop, lat_deg
        self.n = n_particles
        self.rng = random.Random(seed)
        self.particles = [
            {
                "dr": 0.0,
                "kcb_bias": self.rng.gauss(1.0, 0.08),
                "rain_mult": max(0.4, self.rng.gauss(1.0, 0.18)),
                "state": BalanceState(root_depth_in=crop.root_min_in),
                "w": 1.0 / n_particles,
            }
            for _ in range(n_particles)
        ]

    def step(self, weather: WeatherDay, dap: int, irrigation_in: float,
             observations: Sequence[Observation] = (),
             curve_number: float = 74.0) -> dict:
        for p in self.particles:
            w = WeatherDay(
                day=weather.day, tmax_f=weather.tmax_f, tmin_f=weather.tmin_f,
                rh_max=weather.rh_max, rh_min=weather.rh_min,
                wind_mph=weather.wind_mph, solar_rad_mj_m2=weather.solar_rad_mj_m2,
                rain_in=weather.rain_in * p["rain_mult"], eto_in=weather.eto_in,
            )
            p["state"].dr_in = p["dr"]
            res = step_water_balance(
                p["state"], w, self.crop, self.soil, dap, self.lat,
                irrigation_in=irrigation_in,
                kcb_override=self.crop.kcb(dap) * p["kcb_bias"],
                curve_number=curve_number,
            )
            p["dr"] = res.depletion_in
            p["kcb_bias"] = max(0.4, min(1.8, p["kcb_bias"] + self.rng.gauss(0, 0.01)))

        root = self.particles[0]["state"].root_depth_in
        for ob in observations:
            if ob.kind != "soil_moisture" or ob.day != weather.day:
                continue
            var = ob.variance(root, self.crop.kcb_mid)
            for p in self.particles:
                p["w"] *= math.exp(-0.5 * (ob.value - p["dr"]) ** 2 / var) + 1e-12
            self._resample()

        weights = [p["w"] for p in self.particles]
        tot = sum(weights) or 1.0
        mean = sum(p["dr"] * p["w"] for p in self.particles) / tot
        var = sum(p["w"] * (p["dr"] - mean) ** 2 for p in self.particles) / tot
        ordered = sorted(p["dr"] for p in self.particles)
        return {
            "mean_depletion_in": round(mean, 3),
            "sd_in": round(math.sqrt(max(0.0, var)), 3),
            "p05_in": round(ordered[int(0.05 * self.n)], 3),
            "p50_in": round(ordered[int(0.50 * self.n)], 3),
            "p95_in": round(ordered[int(0.95 * self.n)], 3),
        }

    def _resample(self) -> None:
        tot = sum(p["w"] for p in self.particles) or 1.0
        for p in self.particles:
            p["w"] /= tot
        neff = 1.0 / sum(p["w"] ** 2 for p in self.particles)
        if neff > self.n / 2:
            return
        cum, u0, new, i = 0.0, self.rng.random() / self.n, [], 0
        cdf = []
        for p in self.particles:
            cum += p["w"]
            cdf.append(cum)
        for j in range(self.n):
            u = u0 + j / self.n
            while i < self.n - 1 and u > cdf[i]:
                i += 1
            src = self.particles[i]
            new.append({
                "dr": src["dr"],
                "kcb_bias": src["kcb_bias"],
                "rain_mult": src["rain_mult"],
                "state": BalanceState(dr_in=src["dr"],
                                      de_in=src["state"].de_in,
                                      root_depth_in=src["state"].root_depth_in),
                "w": 1.0 / self.n,
            })
        self.particles = new
