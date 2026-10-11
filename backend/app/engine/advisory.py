"""
Irrigation advisory: turn the assimilated water balance into one decision.

The output of this module is what a grower sees at 5:30 in the morning. It has
to answer three questions in order of how much they care:

    1. Do I need to start the pivot today?
    2. If not today, what day?
    3. How much do I put on, and what does it cost me?

Everything else - prediction intervals, stress probabilities, deep percolation -
is supporting detail behind those three answers.

The forecast trajectory runs the same FAO-56 balance forward over the National
Weather Service point forecast, perturbed to produce an ensemble. Rainfall is
the dominant uncertainty at a 7-day horizon on Delmarva, where summer
convection means the difference between a quarter inch and two inches at the
same pivot on the same afternoon, so the ensemble perturbs the probability of
precipitation and the conditional amount separately.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

from .crops import get_crop
from .fao56 import (
    BalanceState, CropStage, DayResult, SoilProfile, WeatherDay,
    step_water_balance,
)

DECISION_IRRIGATE = "irrigate_now"
DECISION_SOON = "irrigate_soon"
DECISION_HOLD = "hold"
DECISION_RAIN_HOLD = "hold_for_rain"
DECISION_WET = "too_wet"


@dataclass(slots=True)
class SystemSpec:
    """The irrigation system's physical limits.

    A recommendation the system cannot deliver is worse than no recommendation,
    so these constraints are hard-applied to every depth this module returns.
    """

    system_type: str = "center_pivot"     # center_pivot | linear | traveler | drip | furrow
    capacity_gpm: float = 800.0
    acres: float = 130.0
    application_efficiency: float = 0.85   # pivot with drops in DE wind
    min_application_in: float = 0.25
    max_application_in: float = 1.20
    hours_per_revolution_at_min: float = 14.0
    vri_capable: bool = False
    fertigation_capable: bool = False
    well_permit_gpm: float | None = None
    allocation_ac_in_remaining: float | None = None

    @property
    def system_capacity_in_per_day(self) -> float:
        """Gross inches per acre per day if the system runs 24 hours.

        450 gpm applied to one acre for one hour is one acre-inch.
        """
        gpm = min(self.capacity_gpm, self.well_permit_gpm or self.capacity_gpm)
        return round((gpm * 24.0) / (450.0 * max(1.0, self.acres)), 3)

    def hours_to_apply(self, gross_in: float) -> float:
        gpm = min(self.capacity_gpm, self.well_permit_gpm or self.capacity_gpm)
        return round((gross_in * self.acres * 450.0) / max(1.0, gpm), 1)


@dataclass(slots=True)
class ForecastDay:
    day: date
    tmax_f: float
    tmin_f: float
    pop: float                # probability of precipitation, 0-1
    qpf_in: float             # forecast amount when it does rain
    wind_mph: float = 6.0
    rh_min: float | None = None
    rh_max: float | None = None

    def to_weather(self, rain_in: float | None = None) -> WeatherDay:
        return WeatherDay(
            day=self.day, tmax_f=self.tmax_f, tmin_f=self.tmin_f,
            rh_min=self.rh_min, rh_max=self.rh_max, wind_mph=self.wind_mph,
            rain_in=self.qpf_in * self.pop if rain_in is None else rain_in,
            source="forecast",
        )


@dataclass(slots=True)
class Advisory:
    field_id: str
    field_name: str
    as_of: date
    decision: str
    headline: str
    detail: str
    depletion_in: float
    depletion_sd_in: float
    raw_in: float
    taw_in: float
    percent_depleted: int
    percent_of_mad: int
    days_to_mad: int | None
    recommended_net_in: float
    recommended_gross_in: float
    run_hours: float
    stress_risk_7day: float
    deep_perc_risk_in: float
    growth_stage: str
    dap: int
    trajectory: list[dict] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    confidence: str = "modeled"
    cost_estimate_usd: float | None = None


def growth_stage_label(crop: CropStage, dap: int) -> str:
    if dap <= crop.l_ini:
        return "Emergence and early vegetative"
    if dap <= crop.l_ini + crop.l_dev:
        return "Rapid vegetative growth"
    if dap <= crop.l_ini + crop.l_dev + crop.l_mid:
        if crop.critical_start_dap <= dap <= crop.critical_end_dap:
            return "Reproductive, most sensitive to stress"
        return "Full canopy"
    if dap <= crop.season_days:
        return "Grain fill and senescence"
    return "Past maturity"


def project_forecast(
    start_depletion_in: float,
    depletion_sd_in: float,
    forecast: Sequence[ForecastDay],
    crop: CropStage,
    soil: SoilProfile,
    dap_today: int,
    lat_deg: float,
    curve_number: float,
    kcb_bias: float = 1.0,
    planned_irrigation: dict[date, float] | None = None,
    n_ensemble: int = 120,
    seed: int | None = 7,
) -> list[dict]:
    """Ensemble forward run. Returns one dict per forecast day with quantiles."""
    rng = random.Random(seed)
    planned = planned_irrigation or {}
    members: list[list[float]] = []

    for _ in range(n_ensemble):
        dr0 = max(0.0, rng.gauss(start_depletion_in, depletion_sd_in))
        state = BalanceState(dr_in=dr0, root_depth_in=crop.root_depth_in(dap_today))
        bias = rng.gauss(kcb_bias, 0.05)
        eto_mult = rng.gauss(1.0, 0.10)
        track: list[float] = []
        for i, fd in enumerate(forecast):
            rain = 0.0
            if rng.random() < fd.pop:
                # Gamma-ish spread around the QPF: convective rain is right-skewed.
                shape = 1.6
                rain = fd.qpf_in * rng.gammavariate(shape, 1.0 / shape)
            w = fd.to_weather(rain_in=rain)
            w.tmax_f += rng.gauss(0, 2.5)
            w.tmin_f += rng.gauss(0, 2.0)
            base = step_water_balance(
                state, w, crop, soil, dap_today + i + 1, lat_deg,
                irrigation_in=planned.get(fd.day, 0.0),
                kcb_override=crop.kcb(dap_today + i + 1) * bias * eto_mult,
                curve_number=curve_number,
            )
            track.append(base.depletion_in)
        members.append(track)

    out = []
    for i, fd in enumerate(forecast):
        vals = sorted(m[i] for m in members)
        n = len(vals)
        root = crop.root_depth_in(dap_today + i + 1)
        raw = soil.raw_in(root, crop.mad(dap_today + i + 1))
        exceed = sum(1 for v in vals if v > raw) / n
        out.append({
            "date": fd.day.isoformat(),
            "p05_in": round(vals[int(0.05 * n)], 2),
            "p50_in": round(vals[int(0.50 * n)], 2),
            "p95_in": round(vals[min(n - 1, int(0.95 * n))], 2),
            "raw_in": round(raw, 2),
            "taw_in": round(soil.taw_in(root), 2),
            "prob_past_mad": round(exceed, 2),
            "pop": round(fd.pop, 2),
            "qpf_in": round(fd.qpf_in, 2),
            "tmax_f": round(fd.tmax_f),
        })
    return out


def build_advisory(
    *,
    field_id: str,
    field_name: str,
    as_of: date,
    crop_key: str,
    planting: date,
    soil: SoilProfile,
    system: SystemSpec,
    depletion_in: float,
    depletion_sd_in: float,
    forecast: Sequence[ForecastDay],
    lat_deg: float = 38.7,
    curve_number: float = 78.0,
    kcb_bias: float = 1.0,
    has_sensors: bool = False,
    energy_cost_per_ac_in: float | None = None,
) -> Advisory:
    crop = get_crop(crop_key)
    dap = (as_of - planting).days
    root = crop.root_depth_in(dap)
    taw = soil.taw_in(root)
    mad = crop.mad(dap)
    raw = soil.raw_in(root, mad)

    trajectory = project_forecast(
        depletion_in, depletion_sd_in, forecast, crop, soil,
        dap, lat_deg, curve_number, kcb_bias,
    )

    pct_dep = int(round(100 * depletion_in / taw)) if taw else 0
    pct_mad = int(round(100 * depletion_in / raw)) if raw else 0

    days_to_mad: int | None = None
    for i, t in enumerate(trajectory):
        if t["p50_in"] >= t["raw_in"]:
            days_to_mad = i + 1
            break
    if depletion_in >= raw:
        days_to_mad = 0

    stress_risk = max((t["prob_past_mad"] for t in trajectory[:7]), default=0.0)
    near_rain = sum(t["pop"] * t["qpf_in"] for t in trajectory[:3])

    # --- how much water ----------------------------------------------------
    # Refill to field capacity, but never plan a depth the profile cannot hold
    # or the system cannot deliver. On droughty sands this is the constraint
    # that actually binds, and it is the reason a 1.0 inch set on Evesboro
    # sends a third of the water past the root zone.
    net_target = min(depletion_in, taw * 0.90)
    constraints: list[str] = []

    if net_target > system.max_application_in:
        constraints.append(
            f"Profile needs {net_target:.2f} in but the system tops out at "
            f"{system.max_application_in:.2f} in per pass. Split it.")
        net_target = system.max_application_in
    if 0 < net_target < system.min_application_in:
        net_target = system.min_application_in

    gross = net_target / max(0.4, system.application_efficiency)
    if system.allocation_ac_in_remaining is not None:
        remaining = system.allocation_ac_in_remaining
        if gross * system.acres > remaining:
            gross = max(0.0, remaining / max(1.0, system.acres))
            net_target = gross * system.application_efficiency
            constraints.append(
                "Application trimmed to what is left of the allocation.")

    daily_capacity = system.system_capacity_in_per_day
    if daily_capacity < 0.30 and depletion_in > raw:
        constraints.append(
            f"System capacity is {daily_capacity:.2f} in/day gross. Peak corn "
            "water use exceeds that, so the field will keep losing ground until "
            "it rains. Prioritize the most sensitive blocks.")

    run_hours = system.hours_to_apply(gross)

    # --- the decision ------------------------------------------------------
    if depletion_in <= 0.15 * taw:
        decision = DECISION_WET
        headline = "Soil is near field capacity"
        detail = (
            f"The root zone is holding {taw - depletion_in:.1f} of "
            f"{taw:.1f} inches. Water applied now mostly drains past the roots "
            "and carries nitrate with it.")
        net_target = gross = run_hours = 0.0
    elif depletion_in >= raw:
        decision = DECISION_IRRIGATE
        headline = "Irrigate today"
        detail = (
            f"Depletion is {depletion_in:.2f} in against an allowed "
            f"{raw:.2f} in for {growth_stage_label(crop, dap).lower()}. "
            f"Apply {net_target:.2f} in net.")
    elif near_rain >= 0.75 * (raw - depletion_in) and near_rain >= 0.5:
        decision = DECISION_RAIN_HOLD
        headline = f"Hold, {near_rain:.1f} in of rain expected"
        detail = (
            f"You have {raw - depletion_in:.2f} in of margin before stress and "
            f"the forecast carries {near_rain:.1f} in through the next three "
            "days. Waiting keeps room in the profile to catch it.")
        net_target = gross = run_hours = 0.0
    elif days_to_mad is not None and days_to_mad <= 2:
        decision = DECISION_SOON
        headline = f"Start within {days_to_mad} day{'s' if days_to_mad != 1 else ''}"
        detail = (
            f"Depletion reaches the {int(mad * 100)} percent threshold around "
            f"{trajectory[days_to_mad - 1]['date'] if days_to_mad else ''}. "
            f"A {run_hours:.0f} hour pass covers the field.")
    else:
        decision = DECISION_HOLD
        headline = "No water needed today"
        nxt = f" Next likely start is in {days_to_mad} days." if days_to_mad else ""
        detail = (
            f"{raw - depletion_in:.2f} in of margin remains before the crop "
            f"begins to feel it.{nxt}")
        net_target = gross = run_hours = 0.0

    deep_perc_risk = max(0.0, gross * system.application_efficiency - depletion_in)
    if deep_perc_risk > 0.15:
        constraints.append(
            f"About {deep_perc_risk:.2f} in would move below the root zone. "
            "On sand that is the nitrate pathway to groundwater.")

    cost = round(gross * system.acres * energy_cost_per_ac_in, 2) \
        if energy_cost_per_ac_in and gross else None

    return Advisory(
        field_id=field_id,
        field_name=field_name,
        as_of=as_of,
        decision=decision,
        headline=headline,
        detail=detail,
        depletion_in=round(depletion_in, 2),
        depletion_sd_in=round(depletion_sd_in, 2),
        raw_in=round(raw, 2),
        taw_in=round(taw, 2),
        percent_depleted=pct_dep,
        percent_of_mad=pct_mad,
        days_to_mad=days_to_mad,
        recommended_net_in=round(net_target, 2),
        recommended_gross_in=round(gross, 2),
        run_hours=run_hours,
        stress_risk_7day=round(stress_risk, 2),
        deep_perc_risk_in=round(deep_perc_risk, 2),
        growth_stage=growth_stage_label(crop, dap),
        dap=dap,
        trajectory=trajectory,
        constraints=constraints,
        confidence="sensor-assimilated" if has_sensors else "modeled",
        cost_estimate_usd=cost,
    )


def yield_loss_estimate(results: Sequence[DayResult], crop: CropStage) -> dict:
    """FAO-33 style seasonal yield response to accumulated water stress.

    Relative yield loss is driven by the ratio of actual to potential
    transpiration, weighted more heavily inside the crop's critical window.
    """
    et_act = et_pot = 0.0
    for r in results:
        weight = 1.6 if crop.critical_start_dap <= r.dap <= crop.critical_end_dap else 1.0
        et_act += r.kcb * r.ks * r.eto_in * weight
        et_pot += r.kcb * r.eto_in * weight
    if et_pot <= 0:
        return {"relative_et": 1.0, "yield_loss_pct": 0.0, "stress_days": 0}
    ratio = et_act / et_pot
    loss = max(0.0, crop.yield_response_ky * (1.0 - ratio))
    return {
        "relative_et": round(ratio, 3),
        "yield_loss_pct": round(min(100.0, loss * 100), 1),
        "stress_days": sum(1 for r in results if r.ks < 0.95),
    }
