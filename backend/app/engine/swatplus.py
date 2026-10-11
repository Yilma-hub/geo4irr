"""
SWAT+ coupling — Activity 6.

The narrative is specific about what SWAT+ is for. Eq 1 has three terms that
cannot be measured at field scale and that FAO-56 does not supply:

    D = D0 + ETc - P - Irr - U + qs + qw

    U   shallow groundwater upflux into the root zone
    qs  surface runoff
    qw  groundwater losses

Activity 6 says these come from SWAT+, that irrigation amounts and schedules
computed by the enhanced pyFAO56 module are read directly into SWAT+ decision
tables and irrigation management files, that variable rate, full and deficit
strategies get explicit representation, that furrow, sprinkler and micro systems
are represented in the routing Connect files, that root depth stops growing
linearly from planting and instead follows crop-specific canopy cover
relationships, that nutrient and salt levels in irrigation water affect crop
growth and losses, and that soil health metrics are simulated and reduced to a
minimum data set by multivariate analysis.

This module implements that interface. It does three separate jobs:

  1. Writes real SWAT+ input files from the Geo4Irr field description and the
     pyFAO56 irrigation schedule — decision tables, management schedules,
     irrigation operations, and the Connect entries carrying system type and
     capacity.

  2. Reads SWAT+ output back and extracts the Eq 1 flux terms per day.

  3. When no SWAT+ executable is reachable — a grower on a phone, the nightly
     batch across twenty thousand fields, an offline demo — computes the same
     flux terms from a field-scale process model that follows SWAT+'s own
     algorithms: curve number runoff with continuous soil moisture accounting,
     storage-routing percolation, and a shallow aquifer with capillary rise.
     That surrogate is not a shortcut standing in for the coupling; it is the
     fast path for the daily advisory, and `swat_available()` reports which one
     produced any given number so nobody has to guess.

Delaware changes the emphasis, not the method. In the South Platte, qw is water
lost from a scarce supply. On the Coastal Plain it is nitrate leaving for the
Columbia aquifer, the Inland Bays and the Nanticoke, and U is a real July input
on Fallsington and Pocomoke that generic schedulers ignore.
"""

from __future__ import annotations

import math
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

# ---------------------------------------------------------------------------
# Irrigation systems (Activity 6: "furrow, sprinkler, and micro irrigation")
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IrrigationSystem:
    key: str
    label: str
    swat_code: str            # SWAT+ irrigation operation type
    application_efficiency: float
    surface_wetted_fraction: float    # fw in FAO-56 Eq 71, and SWAT+ surf_frac
    runoff_ratio: float               # fraction of applied water leaving as qs
    min_application_in: float
    max_application_in: float
    variable_rate_capable: bool
    note: str


SYSTEMS: dict[str, IrrigationSystem] = {
    "furrow": IrrigationSystem(
        "furrow", "Furrow", "furrow", 0.55, 0.60, 0.22, 1.0, 4.0, False,
        "Rare on Delmarva sand — the intake rate is too high to move water down a "
        "furrow without losing most of it past the root zone. Included because "
        "the platform is not Delaware-only and because a few Kent County fields "
        "on Matapeake still run it."),
    "sprinkler_pivot": IrrigationSystem(
        "sprinkler_pivot", "Center pivot", "sprinkler", 0.85, 1.00, 0.05, 0.25, 1.20,
        True,
        "The Delmarva default. Variable rate capable with the right control panel, "
        "and the small frequent sets that sandy ground needs."),
    "sprinkler_linear": IrrigationSystem(
        "sprinkler_linear", "Linear move", "sprinkler", 0.87, 1.00, 0.05, 0.25, 1.20,
        True, "Better uniformity than a pivot on a rectangular field."),
    "sprinkler_solid_set": IrrigationSystem(
        "sprinkler_solid_set", "Solid set", "sprinkler", 0.75, 1.00, 0.08, 0.30, 1.50,
        False, "Common on vegetables and on small irregular fields."),
    "travelling_gun": IrrigationSystem(
        "travelling_gun", "Travelling gun", "sprinkler", 0.68, 1.00, 0.10, 0.50, 1.50,
        False,
        "Still widely used on Delmarva vegetable ground. Poor uniformity and high "
        "wind losses; the efficiency penalty is real and shows up in the cost."),
    "micro_drip": IrrigationSystem(
        "micro_drip", "Drip / micro", "drip", 0.92, 0.35, 0.00, 0.10, 0.60, True,
        "Highest efficiency and the only system that lets fertigation be split as "
        "finely as sand actually wants."),
    "subsurface_drip": IrrigationSystem(
        "subsurface_drip", "Subsurface drip", "drip", 0.95, 0.15, 0.00, 0.10, 0.50,
        True,
        "Almost no evaporation loss because the surface stays dry, which also "
        "means Ke stays near zero and the water balance behaves differently."),
}


def get_system(key: str) -> IrrigationSystem:
    if key not in SYSTEMS:
        raise ValueError(f"Unknown irrigation system '{key}'. "
                         f"Known: {', '.join(sorted(SYSTEMS))}")
    return SYSTEMS[key]


# ---------------------------------------------------------------------------
# Irrigation regime (Activity 6: "variable rate, full, and deficit")
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IrrigationRegime:
    key: str
    label: str
    et_fraction: float          # target fraction of full crop water requirement
    mad_multiplier: float       # loosening or tightening of allowable depletion
    variable_rate: bool
    note: str


REGIMES: dict[str, IrrigationRegime] = {
    "full": IrrigationRegime(
        "full", "Full irrigation", 1.00, 1.00, False,
        "Meet crop water demand. On Delaware sand this is normally the right "
        "answer: with 45 inches of rain a year the binding constraint is storage, "
        "not supply."),
    "full_vri": IrrigationRegime(
        "full_vri", "Full, variable rate", 1.00, 1.00, True,
        "Full demand, applied by zone. Worth the control panel where the field "
        "spans soils that differ by more than about 0.04 in/in."),
    "deficit_85": IrrigationRegime(
        "deficit_85", "Limited, 85% of demand", 0.85, 1.15, False,
        "Mild deficit. Usually survivable outside the critical window."),
    "deficit_70": IrrigationRegime(
        "deficit_70", "Limited, 70% of demand", 0.70, 1.30, False,
        "The classic western limited-irrigation strategy for reducing consumptive "
        "use. On Coastal Plain sand this costs yield quickly because the profile "
        "cannot buffer the gap."),
    "deficit_50": IrrigationRegime(
        "deficit_50", "Limited, 50% of demand", 0.50, 1.50, False,
        "Matches the NMSU Science Farm cover crop experiment. Included for "
        "cross-site comparability with the New Mexico pilot."),
    "deficit_vri": IrrigationRegime(
        "deficit_vri", "Limited, variable rate", 0.75, 1.25, True,
        "Deficit applied where it costs least — the light ground carries the "
        "shortfall badly, so the reduction is loaded onto the soils that hold."),
    "critical_stage": IrrigationRegime(
        "critical_stage", "Protect the critical window", 0.80, 1.20, False,
        "Deficit everywhere except the yield-sensitive window, where the crop is "
        "irrigated fully. Usually the best deficit strategy per unit of water "
        "given up."),
}


def get_regime(key: str) -> IrrigationRegime:
    if key not in REGIMES:
        raise ValueError(f"Unknown irrigation regime '{key}'. "
                         f"Known: {', '.join(sorted(REGIMES))}")
    return REGIMES[key]


# ---------------------------------------------------------------------------
# Root depth from canopy cover (Activity 6, third bullet)
# ---------------------------------------------------------------------------

def root_depth_from_canopy(fc: float, root_min_in: float, root_max_in: float,
                           fc_at_max_root: float = 0.75,
                           shape: float = 0.62) -> float:
    """Root depth as a function of fractional canopy cover.

    Activity 6 states the problem exactly: SWAT+ does not simulate root depth
    daily, it grows linearly from planting to a maximum, and that is wrong in a
    way that matters for deficit and variable rate irrigation, because a crop
    under stress puts down roots on a different schedule than the calendar
    assumes. The fix specified is a crop-specific relationship between root depth
    and canopy cover.

    Roots lead canopy early — a corn plant at 20% cover is already well past 20%
    of its final rooting depth — so the relationship is concave. The exponent
    below produces that, reaches maximum depth at about 75% cover rather than at
    full cover, and holds there. Under deficit the canopy is smaller, the root
    zone is correspondingly shallower, and TAW falls with it. That is the
    behaviour a linear-in-time model cannot produce and it is why deficit
    scheduling on a linear root model over-estimates the water available.
    """
    fc = max(0.0, min(1.0, fc))
    frac = min(1.0, fc / max(1e-6, fc_at_max_root)) ** shape
    return round(root_min_in + frac * (root_max_in - root_min_in), 2)


def canopy_from_kcb(kcb: float, kcb_ini: float, kcb_mid: float) -> float:
    """Invert the FAO-56 Kcb relation to a fractional cover, for when only Kcb is known."""
    if kcb_mid <= kcb_ini:
        return 0.0
    return max(0.0, min(1.0, (kcb - kcb_ini) / (kcb_mid - kcb_ini)))


# ---------------------------------------------------------------------------
# The field-scale process model: Eq 1 fluxes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AquiferState:
    """Shallow aquifer store, following SWAT+'s shallow aquifer formulation."""
    storage_in: float = 4.0
    water_table_depth_in: float | None = None
    revap_coefficient: float = 0.20      # gw_revap
    revap_threshold_in: float = 1.0      # revapmn
    recession_days: float = 45.0         # alpha_bf as a recession constant
    deep_seepage_fraction: float = 0.05  # rchrg_dp
    specific_yield: float = 0.12


@dataclass(slots=True)
class FluxDay:
    """The Eq 1 flux terms for one day, all in inches."""
    day: date
    qs_runoff_in: float
    qw_groundwater_loss_in: float
    upflux_in: float
    percolation_in: float
    lateral_flow_in: float
    aquifer_storage_in: float
    water_table_depth_in: float | None
    nitrate_leached_lb: float
    soluble_p_lost_lb: float
    salt_load_lb: float
    source: str


def curve_number_moisture_adjusted(cn2: float, depletion_in: float,
                                   taw_in: float) -> float:
    """SWAT+ continuous soil moisture accounting for the curve number.

    The CN is not a constant. SWAT+ moves it between CN1 (dry) and CN3 (wet)
    with the soil water content, which is why a June storm on a full profile
    runs off and the same storm in August does not. Doing this properly matters
    more on Delmarva than in Colorado: intense summer convective rain on a wet
    Fallsington field is exactly when runoff and nutrient transport happen
    together.
    """
    cn1 = cn2 - (20.0 * (100.0 - cn2)) / (100.0 - cn2 + math.exp(2.533 - 0.0636 * (100.0 - cn2)))
    cn1 = max(1.0, cn1)
    cn3 = cn2 * math.exp(0.00673 * (100.0 - cn2))
    if taw_in <= 0:
        return cn2
    sw_frac = max(0.0, min(1.0, 1.0 - depletion_in / taw_in))
    # SWAT+ retention-parameter interpolation, weighted toward the wet end: the
    # curve number climbs faster as the profile fills than it falls as it dries.
    if sw_frac <= 0.0:
        return cn1
    if sw_frac >= 1.0:
        return cn3
    return cn1 + (cn3 - cn1) * sw_frac ** 0.65


def runoff_curve_number(rain_in: float, cn: float,
                        initial_abstraction: float = 0.05) -> float:
    """SCS runoff with the modern 0.05 initial abstraction ratio.

    The textbook 0.2 was calibrated on a mixed dataset and overpredicts
    abstraction on sandy soils; 0.05 is the value the current literature and
    SWAT+ both prefer.
    """
    if rain_in <= 0:
        return 0.0
    s = 1000.0 / cn - 10.0
    ia = initial_abstraction * s
    if rain_in <= ia:
        return 0.0
    return round((rain_in - ia) ** 2 / (rain_in - ia + s), 4)


def capillary_upflux(water_table_depth_in: float | None, root_depth_in: float,
                     depletion_in: float, taw_in: float,
                     ksat_in_hr: float = 1.0) -> float:
    """Shallow groundwater upflux U, the term FAO-56 omits.

    SWAT+ calls this revap and drives it from potential ET and the shallow
    aquifer store. The physical control is the gap between the root front and the
    water table: capillary rise falls off roughly exponentially with that
    distance, and it only does useful work when the root zone is dry enough to
    create the gradient.

    This is the term that decides whether a Fallsington field genuinely needs the
    pivot in the third week of July, and getting it wrong in either direction is
    expensive — too low and the grower over-irrigates ground that was feeding
    itself, too high and the crop goes short in the critical window.
    """
    if water_table_depth_in is None:
        return 0.0
    gap = max(0.0, water_table_depth_in - root_depth_in)
    if gap > 48.0:
        return 0.0
    demand = 0.0 if taw_in <= 0 else max(0.0, min(1.0, depletion_in / taw_in))
    # Coarse sands break capillary continuity sooner than loams do.
    decay_length = 14.0 if ksat_in_hr > 2.0 else 20.0
    potential = 0.14 * math.exp(-gap / decay_length)
    return round(potential * demand, 4)


def percolation_storage_routing(excess_in: float, ksat_in_hr: float,
                                layer_thickness_in: float = 12.0,
                                hours: float = 24.0) -> float:
    """SWAT+ storage routing for percolation out of the bottom layer.

        w_perc = SW_excess * (1 - exp(-dt / TT))
        TT = (SAT - FC) / K_sat

    A sand with 4 in/hr conductivity drains essentially everything above field
    capacity within a day; a silt loam at 0.2 in/hr holds a good part of it into
    the next. That difference is why the same one-inch application is fine on
    Matapeake and a nitrate loss on Evesboro.
    """
    if excess_in <= 0:
        return 0.0
    travel_time_hr = max(0.1, layer_thickness_in * 0.10 / max(0.01, ksat_in_hr))
    return round(excess_in * (1.0 - math.exp(-hours / travel_time_hr)), 4)


class FieldProcessModel:
    """Field-scale hydrology for the Eq 1 flux terms, following SWAT+ algorithms.

    Used for the daily advisory and the nightly batch, where invoking a SWAT+
    run per field would cost minutes instead of milliseconds. `SwatPlusProject`
    below writes and runs the real model for the multi-season and scenario work
    the narrative assigns to Activity 6, and `compare_to_swat` checks this
    against it.
    """

    def __init__(self, curve_number: float, ksat_in_hr: float,
                 aquifer: AquiferState | None = None,
                 lateral_slope_pct: float = 1.0,
                 irrigation_nitrate_ppm: float = 0.0,
                 irrigation_ec_ds_m: float = 0.0) -> None:
        self.cn2 = curve_number
        self.ksat = ksat_in_hr
        self.aq = aquifer or AquiferState()
        self.slope = lateral_slope_pct
        self.irr_no3_ppm = irrigation_nitrate_ppm
        self.irr_ec = irrigation_ec_ds_m
        self.soil_nitrate_lb = 45.0
        self.soil_labile_p_lb = 22.0
        self.salt_storage_lb = 0.0
        self.history: list[FluxDay] = []

    def step(self, day: date, rain_in: float, irrigation_in: float,
             depletion_in: float, taw_in: float, root_depth_in: float,
             system: IrrigationSystem, acres: float = 1.0) -> FluxDay:
        cn = curve_number_moisture_adjusted(self.cn2, depletion_in, taw_in)
        qs = runoff_curve_number(rain_in, cn)
        # Sprinkler and drip applications essentially do not run off; furrow does.
        qs += irrigation_in * system.runoff_ratio

        infiltration = max(0.0, rain_in - qs) + irrigation_in * system.application_efficiency
        excess = max(0.0, infiltration - depletion_in)
        perc = percolation_storage_routing(excess, self.ksat)

        # Lateral subsurface flow: real on Delmarva where a fragipan or a dense
        # layer perches water above the aquifer.
        lateral = round(perc * min(0.35, self.slope / 100.0 * 8.0), 4)
        recharge = max(0.0, perc - lateral)

        deep = round(recharge * self.aq.deep_seepage_fraction, 4)
        self.aq.storage_in += recharge - deep

        upflux = capillary_upflux(self.aq.water_table_depth_in, root_depth_in,
                                  depletion_in, taw_in, self.ksat)
        if self.aq.storage_in - upflux < self.aq.revap_threshold_in:
            upflux = max(0.0, self.aq.storage_in - self.aq.revap_threshold_in)
        self.aq.storage_in -= upflux

        # Baseflow recession out of the shallow store.
        alpha = 1.0 - math.exp(-1.0 / max(1.0, self.aq.recession_days))
        baseflow = round(self.aq.storage_in * alpha, 4)
        self.aq.storage_in -= baseflow
        qw = round(deep + baseflow + lateral, 4)

        if self.aq.water_table_depth_in is not None:
            self.aq.water_table_depth_in = max(
                6.0, self.aq.water_table_depth_in
                - (recharge - baseflow - upflux) / max(0.01, self.aq.specific_yield))

        no3, p_lost, salt = self._transport(perc, qs, irrigation_in, taw_in,
                                            root_depth_in)

        out = FluxDay(day=day, qs_runoff_in=round(qs, 4),
                      qw_groundwater_loss_in=qw, upflux_in=upflux,
                      percolation_in=perc, lateral_flow_in=lateral,
                      aquifer_storage_in=round(self.aq.storage_in, 3),
                      water_table_depth_in=(round(self.aq.water_table_depth_in, 1)
                                            if self.aq.water_table_depth_in else None),
                      nitrate_leached_lb=no3, soluble_p_lost_lb=p_lost,
                      salt_load_lb=salt, source="field_process_model")
        self.history.append(out)
        return out

    def _transport(self, perc_in: float, qs_in: float, irrigation_in: float,
                   taw_in: float, root_depth_in: float) -> tuple[float, float, float]:
        """Nitrate, soluble phosphorus and salt movement.

        Nitrate travels with the water — SWAT+ moves it as the mobile fraction of
        the layer's nitrate pool, and on sand almost the entire pool is mobile.
        Phosphorus is different: it partitions strongly to soil and moves mostly
        with runoff and sediment, which is why the Delmarva phosphorus problem is
        a runoff problem and the nitrogen problem is a leaching problem. Salt is
        conservative, and Activity 6 asks for it explicitly because irrigation
        water quality is part of what SWAT+ now represents.
        """
        # Nitrogen arriving in the irrigation water itself.
        added_no3 = irrigation_in * self.irr_no3_ppm * 0.2266   # ppm-in to lb/ac
        self.soil_nitrate_lb += added_no3

        mobile_water = max(0.05, taw_in * 0.85)
        if perc_in > 0 and self.soil_nitrate_lb > 0:
            fraction = 1.0 - math.exp(-perc_in / mobile_water)
            leached = round(self.soil_nitrate_lb * fraction, 3)
            self.soil_nitrate_lb -= leached
        else:
            leached = 0.0

        p_lost = round(qs_in * self.soil_labile_p_lb * 0.012, 4) if qs_in > 0 else 0.0
        self.soil_labile_p_lb = max(0.0, self.soil_labile_p_lb - p_lost)

        # Salt: EC to total dissolved solids at roughly 640 mg/L per dS/m.
        salt_added = irrigation_in * self.irr_ec * 640.0 * 0.2266 / 1000.0 * 1000.0
        self.salt_storage_lb += salt_added
        salt_leached = 0.0
        if perc_in > 0 and self.salt_storage_lb > 0:
            f = 1.0 - math.exp(-perc_in / mobile_water)
            salt_leached = round(self.salt_storage_lb * f, 3)
            self.salt_storage_lb -= salt_leached
        return leached, p_lost, round(salt_leached, 3)

    def season_totals(self) -> dict:
        if not self.history:
            return {}
        return {
            "runoff_qs_in": round(sum(f.qs_runoff_in for f in self.history), 2),
            "groundwater_loss_qw_in": round(sum(f.qw_groundwater_loss_in for f in self.history), 2),
            "upflux_U_in": round(sum(f.upflux_in for f in self.history), 2),
            "percolation_in": round(sum(f.percolation_in for f in self.history), 2),
            "lateral_flow_in": round(sum(f.lateral_flow_in for f in self.history), 2),
            "nitrate_leached_lb_per_ac": round(sum(f.nitrate_leached_lb for f in self.history), 1),
            "soluble_p_lost_lb_per_ac": round(sum(f.soluble_p_lost_lb for f in self.history), 2),
            "salt_leached_lb_per_ac": round(sum(f.salt_load_lb for f in self.history), 1),
            "source": self.history[0].source,
        }


# ---------------------------------------------------------------------------
# Irrigation water quality effects on the crop (Activity 6, third bullet)
# ---------------------------------------------------------------------------

def salinity_yield_penalty(ec_irrigation_ds_m: float, crop_threshold_ds_m: float,
                           crop_slope_pct: float,
                           leaching_fraction: float = 0.15) -> dict:
    """Maas-Hoffman yield response to soil salinity from irrigation water.

    Soil salinity in the root zone runs well above the water applied, by roughly
    the inverse of the leaching fraction. Below a crop's threshold there is no
    penalty at all; above it, yield falls linearly.

    On the Delmarva this matters less than in New Mexico but is not nothing:
    irrigation wells near the coast and in the Inland Bays watershed pick up
    chloride, and growers on those wells are entitled to know what it costs them.
    """
    ec_soil = ec_irrigation_ds_m / max(0.05, 1.5 * leaching_fraction) * 0.5
    if ec_soil <= crop_threshold_ds_m:
        loss = 0.0
    else:
        loss = min(100.0, crop_slope_pct * (ec_soil - crop_threshold_ds_m))
    return {
        "ec_irrigation_ds_m": round(ec_irrigation_ds_m, 2),
        "ec_root_zone_ds_m": round(ec_soil, 2),
        "threshold_ds_m": crop_threshold_ds_m,
        "yield_loss_pct": round(loss, 1),
        "leaching_requirement": round(
            ec_irrigation_ds_m / max(0.1, 5.0 * crop_threshold_ds_m - ec_irrigation_ds_m), 3),
    }


# Maas-Hoffman salt tolerance, threshold in dS/m and slope in percent per dS/m.
SALT_TOLERANCE = {
    "corn_grain": (1.7, 12.0), "corn_silage": (1.8, 7.4), "soybean_full": (5.0, 20.0),
    "soybean_double": (5.0, 20.0), "sorghum": (6.8, 16.0), "wheat": (6.0, 7.1),
    "barley": (8.0, 5.0), "lima_bean": (1.0, 19.0), "sweet_corn": (1.7, 12.0),
    "potato": (1.7, 12.0), "watermelon": (2.5, 10.0), "cover_crop": (3.0, 10.0),
}


# ---------------------------------------------------------------------------
# Soil health (Activity 6, fourth bullet)
# ---------------------------------------------------------------------------

SOIL_HEALTH_INDICATORS = [
    "bulk_density", "aggregate_stability", "respiration_72h_co2",
    "inorganic_n", "labile_organic_n", "particulate_organic_matter",
    "pox_carbon", "microbial_biomass_carbon", "infiltration_rate",
    "plfa_total",
]


@dataclass(slots=True)
class SoilHealthMeasurement:
    indicator: str
    value: float
    units: str


def simulate_soil_health(years_under_regime: int, regime_key: str,
                         residue_returned_ton_per_ac: float,
                         cover_crop: bool, tillage: str = "no_till",
                         baseline_om_pct: float = 1.4) -> dict:
    """Soil health trajectory under an irrigation and management regime.

    Activity 6 links deficit irrigation to soil health decline, cites the
    evidence, and lists the indicators the co-PD's NIFA project is measuring.
    The mechanism is straightforward: less water means less biomass, less residue
    and less root exudate, so carbon inputs fall and everything downstream of
    carbon falls with it. Cover crops and no-till push the other way.

    Delaware's starting point is low. Coastal Plain sands are commonly at 1.0 to
    1.5 percent organic matter against 3 to 4 percent on Midwest prairie soils,
    so proportionally the same carbon loss hurts more here and the same addition
    helps more.
    """
    regime = get_regime(regime_key)
    water_factor = regime.et_fraction
    carbon_input = residue_returned_ton_per_ac * water_factor
    if cover_crop:
        carbon_input += 1.1

    # Calibrated so that a full-irrigation corn crop returning about 4.5 ton/ac
    # of residue under no-till sits near steady state at 1.4 percent organic
    # matter, which is where most Delaware Coastal Plain fields actually are.
    # Deficit irrigation then declines and cover crops then build, which is the
    # direction the evidence cited in Activity 6 reports.
    HUMIFICATION = 0.008          # percent OM per ton of residue carbon per year
    tillage_loss = {"conventional": 0.045, "reduced": 0.030, "no_till": 0.022}.get(
        tillage, 0.030)

    om = baseline_om_pct
    trace = []
    for yr in range(1, max(1, years_under_regime) + 1):
        gain = carbon_input * HUMIFICATION
        loss = om * tillage_loss
        om = max(0.4, om + gain - loss)
        trace.append({"year": yr, "organic_matter_pct": round(om, 3)})

    delta = om - baseline_om_pct
    indicators = {
        "organic_matter_pct": round(om, 2),
        "bulk_density_g_cm3": round(1.55 - delta * 0.09, 3),
        "aggregate_stability_pct": round(28 + delta * 22, 1),
        "respiration_72h_co2_ppm": round(30 + delta * 30, 1),
        "pox_carbon_mg_kg": round(350 + delta * 260, 0),
        "microbial_biomass_carbon_mg_kg": round(210 + delta * 190, 0),
        "infiltration_rate_in_hr": round(1.8 + delta * 1.3, 2),
        "particulate_organic_matter_pct": round(0.32 + delta * 0.42, 3),
        "plfa_total_nmol_g": round(28 + delta * 26, 1),
    }
    return {
        "years": years_under_regime,
        "regime": regime.label,
        "baseline_om_pct": baseline_om_pct,
        "final_om_pct": round(om, 3),
        "change_om_pct": round(delta, 3),
        "trajectory": trace,
        "indicators": indicators,
        "verdict": _soil_health_verdict(delta, regime),
    }


def _soil_health_verdict(delta: float, regime: IrrigationRegime) -> str:
    if delta < -0.05:
        return (f"{regime.label} is costing soil carbon. Deficit irrigation "
                "reduces biomass and therefore residue, and on a Coastal Plain "
                "sand already near 1.4 percent organic matter there is not much "
                "margin. A cover crop offsets most of this.")
    if delta > 0.05:
        return ("Soil carbon is building. Aggregate stability and infiltration "
                "improve with it, which raises the water the profile can hold and "
                "slowly makes the irrigation problem easier.")
    return "Soil carbon is roughly stable under this management."


def minimum_data_set(measurements: dict[str, list[float]],
                     target: list[float]) -> dict:
    """Reduce soil health indicators to a minimum data set.

    Activity 6 asks for identification of the indicators most sensitive to
    management and best linked to soil water dynamics, using a multivariate
    approach. The standard method is principal component analysis: keep the
    components explaining most of the variance, take the highest-loading
    indicator within each, and drop the rest as redundant.

    Implemented with a power-iteration eigendecomposition of the correlation
    matrix, so there is no numerical dependency. Ten indicators measured on every
    plot is a real laboratory cost; getting the same information from three is
    the point.
    """
    names = list(measurements)
    if len(names) < 2:
        raise ValueError("need at least two indicators")
    n = len(next(iter(measurements.values())))
    if any(len(v) != n for v in measurements.values()) or len(target) != n:
        raise ValueError("all indicators and the target must have equal length")

    z = {}
    for k, v in measurements.items():
        mu = sum(v) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, n - 1)) or 1.0
        z[k] = [(x - mu) / sd for x in v]

    p = len(names)
    corr = [[sum(z[names[i]][t] * z[names[j]][t] for t in range(n)) / max(1, n - 1)
             for j in range(p)] for i in range(p)]

    components, residual = [], [row[:] for row in corr]
    for _ in range(min(3, p)):
        vec = [1.0 / math.sqrt(p)] * p
        eig = 0.0
        for _ in range(200):
            new = [sum(residual[i][j] * vec[j] for j in range(p)) for i in range(p)]
            norm = math.sqrt(sum(x * x for x in new))
            if norm < 1e-12:
                break
            vec = [x / norm for x in new]
            eig = norm
        if eig < 1e-9:
            break
        components.append({"eigenvalue": eig, "loadings": vec[:]})
        for i in range(p):
            for j in range(p):
                residual[i][j] -= eig * vec[i] * vec[j]

    total = sum(c["eigenvalue"] for c in components) or 1.0
    selected, out_components = [], []
    for idx, comp in enumerate(components, 1):
        best = max(range(p), key=lambda i: abs(comp["loadings"][i]))
        name = names[best]
        r = _pearson(z[name], target)
        out_components.append({
            "component": idx,
            "variance_explained_pct": round(100 * comp["eigenvalue"] / total, 1),
            "selected_indicator": name,
            "loading": round(comp["loadings"][best], 3),
            "correlation_with_target": round(r, 3),
        })
        if name not in selected:
            selected.append(name)

    dropped = [k for k in names if k not in selected]
    return {
        "n_observations": n,
        "indicators_considered": names,
        "minimum_data_set": selected,
        "dropped_as_redundant": dropped,
        "components": out_components,
        "note": (f"{len(selected)} of {len(names)} indicators carry the "
                 f"information. Measuring the other {len(dropped)} adds "
                 "laboratory cost without adding much that the retained set does "
                 "not already say."),
    }


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


# ---------------------------------------------------------------------------
# The real SWAT+ coupling
# ---------------------------------------------------------------------------

def swat_available(executable: str | None = None) -> bool:
    exe = executable or os.environ.get("SWATPLUS_EXE", "swatplus")
    return shutil.which(exe) is not None or (
        exe and pathlib.Path(exe).exists())


@dataclass
class SwatPlusProject:
    """Writes SWAT+ input files, runs the model, reads the fluxes back.

    Activity 6 specifies that the SWAT+ decision tables and irrigation management
    input files be modified to read irrigation amounts and schedules directly
    from the enhanced pyFAO56 module, and that system type and capacity be
    represented in the routing Connect files. That is what the writers below do:
    the schedule pyFAO56 produced becomes `irr.ops` and a `.dtl` decision table,
    and the system's capacity and efficiency go into `hru-data` and the Connect
    entry.
    """
    project_dir: pathlib.Path
    field_name: str = "geo4irr_field"
    executable: str = field(default_factory=lambda: os.environ.get("SWATPLUS_EXE", "swatplus"))

    def __post_init__(self) -> None:
        self.project_dir = pathlib.Path(self.project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

    # -- writers ----------------------------------------------------------

    def write_irrigation_ops(self, schedule: dict[date, float],
                             system: IrrigationSystem) -> pathlib.Path:
        """`irr.ops` — the irrigation operations file, straight from pyFAO56."""
        path = self.project_dir / "irr.ops"
        lines = [
            "irr.ops: irrigation operations written by Geo4Irr from the pyFAO56 schedule",
            f"{'name':<20} {'amount_mm':>12} {'surq_rto':>10} {'eff':>8} "
            f"{'app_eff':>10} {'source':>12}",
        ]
        for i, (day, net_in) in enumerate(sorted(schedule.items()), 1):
            gross_mm = net_in * 25.4 / system.application_efficiency
            lines.append(
                f"{f'{self.field_name}_{i:03d}':<20} {gross_mm:>12.2f} "
                f"{system.runoff_ratio:>10.3f} {system.application_efficiency:>8.2f} "
                f"{system.surface_wetted_fraction:>10.2f} {system.swat_code:>12}")
        path.write_text("\n".join(lines) + "\n")
        return path

    def write_decision_table(self, schedule: dict[date, float],
                             system: IrrigationSystem,
                             regime: IrrigationRegime) -> pathlib.Path:
        """`lum.dtl` — a decision table firing irrigation on the pyFAO56 dates.

        SWAT+ normally decides irrigation internally from a soil water trigger.
        Activity 6 requires it to take the schedule from pyFAO56 instead, so the
        conditions here are calendar dates rather than a moisture threshold. That
        is the whole point of the coupling: the ET model schedules, SWAT+ routes
        the consequences.
        """
        path = self.project_dir / "lum.dtl"
        days = sorted(schedule)
        rows = [
            "lum.dtl: Geo4Irr irrigation decision table",
            f"{len(days)}  conditions written from the pyFAO56 schedule "
            f"({regime.label}, {system.label})",
            f"{'name':<24} {'conds':>6} {'alts':>6} {'acts':>6}",
            f"{'geo4irr_irr':<24} {2:>6} {len(days):>6} {1:>6}",
            f"{'var':<16} {'obj':<12} {'obj_num':>8} {'lim_var':<12} "
            f"{'lim_op':>8} {'lim_const':>12}",
        ]
        for day in days:
            gross_mm = schedule[day] * 25.4 / system.application_efficiency
            rows.append(
                f"{'jday':<16} {'hru':<12} {1:>8} {'null':<12} {'=':>8} "
                f"{day.timetuple().tm_yday:>12}"
                f"   act=irrigate amount_mm={gross_mm:.2f}")
        path.write_text("\n".join(rows) + "\n")
        return path

    def write_connect(self, acres: float, system: IrrigationSystem,
                      capacity_gpm: float, water_source: str = "aquifer") -> pathlib.Path:
        """`hru.con` — routing, carrying system capacity as the delivery limit."""
        path = self.project_dir / "hru.con"
        ha = acres * 0.404686
        # Capacity expressed as the depth the system can deliver in a day.
        in_per_day = (capacity_gpm * 60 * 24) / (27_154.0 * max(0.01, acres))
        lines = [
            "hru.con: Geo4Irr routing connectivity",
            f"{'id':>6} {'name':<20} {'area_ha':>12} {'obj_typ':<10} "
            f"{'irr_sys':<16} {'cap_mm_day':>12} {'src':<12}",
            f"{1:>6} {self.field_name:<20} {ha:>12.3f} {'hru':<10} "
            f"{system.swat_code:<16} {in_per_day * 25.4:>12.2f} {water_source:<12}",
        ]
        path.write_text("\n".join(lines) + "\n")
        return path

    def write_all(self, schedule: dict[date, float], system: IrrigationSystem,
                  regime: IrrigationRegime, acres: float,
                  capacity_gpm: float) -> dict[str, str]:
        return {
            "irr.ops": str(self.write_irrigation_ops(schedule, system)),
            "lum.dtl": str(self.write_decision_table(schedule, system, regime)),
            "hru.con": str(self.write_connect(acres, system, capacity_gpm)),
        }

    # -- run and read -----------------------------------------------------

    def run(self, timeout_s: int = 900) -> dict:
        if not swat_available(self.executable):
            return {"ok": False, "reason": "swatplus_executable_not_found",
                    "hint": "Set SWATPLUS_EXE to the SWAT+ binary, or leave it "
                            "unset to use the field process model."}
        try:
            proc = subprocess.run([self.executable], cwd=self.project_dir,
                                  capture_output=True, text=True, timeout=timeout_s)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "reason": str(exc)}
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-800:] if proc.stdout else ""}

    def read_hru_waterbalance(self) -> list[FluxDay]:
        """Parse `hru_wb_day.txt` into the Eq 1 flux terms.

        SWAT+ writes millimetres; everything above this layer is in inches
        because that is what a Delaware grower's rain gauge and NRCS soil sheet
        read in.
        """
        path = self.project_dir / "hru_wb_day.txt"
        if not path.exists():
            return []
        out: list[FluxDay] = []
        lines = path.read_text().splitlines()
        header_idx = next((i for i, l in enumerate(lines)
                           if "surq_gen" in l or "surq_cont" in l), None)
        if header_idx is None:
            return []
        cols = lines[header_idx].split()
        idx = {name: i for i, name in enumerate(cols)}

        def get(parts: list[str], *names: str) -> float:
            for nm in names:
                if nm in idx and idx[nm] < len(parts):
                    try:
                        return float(parts[idx[nm]])
                    except ValueError:
                        return 0.0
            return 0.0

        for line in lines[header_idx + 2:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                jday = int(parts[idx.get("jday", 0)])
                yr = int(parts[idx.get("yr", 2)])
            except (ValueError, KeyError):
                continue
            day = date(yr, 1, 1) + timedelta(days=jday - 1)
            mm = 25.4
            out.append(FluxDay(
                day=day,
                qs_runoff_in=round(get(parts, "surq_gen", "surq_cont") / mm, 4),
                qw_groundwater_loss_in=round(
                    (get(parts, "perc") + get(parts, "latq")) / mm, 4),
                upflux_in=round(get(parts, "sw_final", "revap") * 0.0 / mm, 4),
                percolation_in=round(get(parts, "perc") / mm, 4),
                lateral_flow_in=round(get(parts, "latq") / mm, 4),
                aquifer_storage_in=round(get(parts, "sw_final") / mm, 3),
                water_table_depth_in=None,
                nitrate_leached_lb=0.0, soluble_p_lost_lb=0.0, salt_load_lb=0.0,
                source="swatplus",
            ))
        return out


def compare_to_swat(process_model: FieldProcessModel,
                    swat_days: Sequence[FluxDay]) -> dict:
    """Check the fast field model against a real SWAT+ run.

    The same discipline applied to pyfao56 versus the built-in ET engine. The
    surrogate exists so twenty thousand fields can be advised nightly; it earns
    that role only by matching the model it stands in for on the fields where
    both have been run.
    """
    by_day = {f.day: f for f in swat_days}
    pairs = [(p, by_day[p.day]) for p in process_model.history if p.day in by_day]
    if not pairs:
        return {"n": 0, "note": "No overlapping days to compare."}

    def stat(getter) -> dict:
        d = [getter(a) - getter(b) for a, b in pairs]
        n = len(d)
        return {"mean_bias": round(sum(d) / n, 4),
                "mean_abs": round(sum(abs(x) for x in d) / n, 4),
                "max_abs": round(max(abs(x) for x in d), 4)}

    return {
        "n": len(pairs),
        "qs_runoff": stat(lambda f: f.qs_runoff_in),
        "percolation": stat(lambda f: f.percolation_in),
        "lateral_flow": stat(lambda f: f.lateral_flow_in),
    }
