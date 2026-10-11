"""
Pumping energy cost and the economics of an irrigation decision.

Growers do not act on inches. They act on dollars and on hours the pivot is
tied up. This module converts every recommendation into both, so the advisory
screen can say "0.8 inches, 11 hours, $340" instead of only the first number.

Delaware specifics: most irrigation is groundwater from the Columbia
(unconfined) aquifer with modest lifts, typically 30 to 90 feet, which makes
pressure rather than lift the dominant energy term for a center pivot. Diesel
still runs a share of the pumps in Sussex County, so both energy paths are here.
"""

from __future__ import annotations

from dataclasses import dataclass

# 1 acre-inch = 27,154 gallons.
GAL_PER_AC_IN = 27154.0


@dataclass(slots=True)
class PumpSpec:
    lift_ft: float = 60.0                 # pumping water level below ground
    discharge_pressure_psi: float = 35.0  # at the pivot point
    friction_loss_ft: float = 12.0
    pump_efficiency: float = 0.72
    drive_efficiency: float = 0.92        # motor or engine to shaft
    energy_source: str = "electric"       # electric | diesel | propane | natural_gas
    electric_rate_per_kwh: float = 0.135
    diesel_price_per_gal: float = 3.65
    demand_charge_per_kw_month: float = 12.0
    variable_frequency_drive: bool = False

    @property
    def total_dynamic_head_ft(self) -> float:
        return self.lift_ft + self.discharge_pressure_psi * 2.31 + self.friction_loss_ft


def brake_horsepower(gpm: float, pump: PumpSpec) -> float:
    """Shaft horsepower required at the pump."""
    return (gpm * pump.total_dynamic_head_ft) / (3960.0 * max(0.05, pump.pump_efficiency))


def energy_per_acre_inch(gpm: float, pump: PumpSpec) -> dict:
    """Energy and cost to deliver one acre-inch, at the given flow rate."""
    bhp = brake_horsepower(gpm, pump)
    hours = GAL_PER_AC_IN / max(1.0, gpm) / 60.0

    if pump.energy_source == "electric":
        kw = bhp * 0.746 / max(0.05, pump.drive_efficiency)
        if pump.variable_frequency_drive:
            kw *= 0.94
        kwh = kw * hours
        cost = kwh * pump.electric_rate_per_kwh
        return {
            "hours_per_ac_in": round(hours, 3),
            "bhp": round(bhp, 1),
            "kw_demand": round(kw, 1),
            "kwh_per_ac_in": round(kwh, 2),
            "cost_per_ac_in": round(cost, 3),
            "energy_source": "electric",
        }

    # Combustion: Nebraska Pumping Plant Performance Criteria give roughly
    # 12.5 bhp-hours per gallon of diesel at 100 percent of criteria; real
    # plants run near 75 percent.
    bhp_hr_per_unit = {"diesel": 12.5, "propane": 6.89, "natural_gas": 61.7}[pump.energy_source]
    plant_factor = 0.75
    units = (bhp * hours) / (bhp_hr_per_unit * plant_factor)
    price = {"diesel": pump.diesel_price_per_gal, "propane": 2.10,
             "natural_gas": 0.85}[pump.energy_source]
    return {
        "hours_per_ac_in": round(hours, 3),
        "bhp": round(bhp, 1),
        "fuel_units_per_ac_in": round(units, 3),
        "cost_per_ac_in": round(units * price, 3),
        "energy_source": pump.energy_source,
    }


def application_cost(
    gross_in: float, acres: float, gpm: float, pump: PumpSpec,
    labor_per_hour: float = 22.0, maintenance_per_ac_in: float = 1.10,
) -> dict:
    """Full cost of one irrigation pass over a field."""
    e = energy_per_acre_inch(gpm, pump)
    ac_in = gross_in * acres
    run_hours = ac_in * e["hours_per_ac_in"]
    energy_cost = ac_in * e["cost_per_ac_in"]
    labor = run_hours * 0.15 * labor_per_hour       # checking, not attending
    maint = ac_in * maintenance_per_ac_in
    total = energy_cost + labor + maint
    return {
        "acre_inches": round(ac_in, 1),
        "run_hours": round(run_hours, 1),
        "energy_cost": round(energy_cost, 2),
        "labor_cost": round(labor, 2),
        "maintenance_cost": round(maint, 2),
        "total_cost": round(total, 2),
        "cost_per_acre": round(total / max(1.0, acres), 2),
        "cost_per_ac_in": round(e["cost_per_ac_in"], 3),
    }


def marginal_value_of_water(
    yield_response_bu_per_in: float, crop_price_per_bu: float,
    cost_per_ac_in: float, acres: float,
) -> dict:
    """Is this pass worth making?

    Late in grain fill, the last inch often returns less than it costs. The
    scheduler should be willing to say so; a tool that only ever says "irrigate"
    loses credibility with growers by the second season.
    """
    revenue = yield_response_bu_per_in * crop_price_per_bu
    net = revenue - cost_per_ac_in
    return {
        "revenue_per_ac_in": round(revenue, 2),
        "cost_per_ac_in": round(cost_per_ac_in, 2),
        "net_per_ac_in": round(net, 2),
        "net_per_field": round(net * acres, 2),
        "worth_applying": net > 0,
        "breakeven_response_bu": round(cost_per_ac_in / max(0.01, crop_price_per_bu), 2),
    }


def season_summary(
    total_irrigation_in: float, acres: float, gpm: float, pump: PumpSpec,
    yield_actual: float, yield_dryland: float, crop_price: float,
    n_applied_lb: float, n_price_per_lb: float = 0.62,
) -> dict:
    """End-of-season accounting for the field report growers get in November."""
    cost = application_cost(total_irrigation_in, acres, gpm, pump)
    yield_gain = max(0.0, yield_actual - yield_dryland)
    gross_return = yield_gain * crop_price * acres
    n_cost = n_applied_lb * n_price_per_lb * acres
    return {
        "total_ac_in": round(total_irrigation_in * acres, 1),
        "irrigation_cost": cost["total_cost"],
        "nitrogen_cost": round(n_cost, 2),
        "yield_gain_per_ac": round(yield_gain, 1),
        "gross_return": round(gross_return, 2),
        "net_return": round(gross_return - cost["total_cost"], 2),
        "net_per_acre": round((gross_return - cost["total_cost"]) / max(1.0, acres), 2),
        "water_use_efficiency_bu_per_in": (
            round(yield_gain / total_irrigation_in, 2) if total_irrigation_in > 0 else None
        ),
        "cost_per_bushel_gained": (
            round(cost["total_cost"] / (yield_gain * acres), 2)
            if yield_gain > 0 else None
        ),
    }
