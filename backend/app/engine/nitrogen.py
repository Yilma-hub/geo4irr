"""
Nitrogen planning and leaching risk for Delaware irrigated cropland.

Delaware is the sharpest version of the combined water-and-nutrient problem the
project narrative describes. Sussex County grows a large share of the state's
irrigated corn on sandy Coastal Plain soils, and a large share of that acreage
receives poultry litter. Nitrate that moves below the root zone on those soils
reaches shallow groundwater quickly and surfaces in the Inland Bays and the
Nanticoke. That makes irrigation scheduling and nitrogen management the same
decision: every inch applied beyond what the profile can hold is a nitrate
transport event.

Regulatory note carried through to the interface: under the Delaware Nutrient
Management Law, anyone applying nutrients to 10 or more acres must be certified,
and operations above that threshold require a nutrient management plan. This
module produces a decision aid and an in-season adjustment, not a plan. Every
output is labelled as something to reconcile with the grower's certified plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

# Nitrogen removed in harvested grain, lb N per unit of yield.
N_REMOVAL: dict[str, float] = {
    "corn_grain": 0.67,          # lb N / bu
    "corn_silage": 8.0,          # lb N / ton
    "soybean_full": 3.45,        # lb N / bu, mostly from fixation
    "soybean_double": 3.45,
    "sorghum_grain": 0.72,
    "wheat": 1.20,
    "barley": 1.10,
    "sweet_corn": 5.0,
    "potato": 0.35,              # lb N / cwt
    "lima_bean": 0.045,          # lb N / lb
    "watermelon": 0.0025,
}

# Total seasonal N requirement per unit of yield goal, before credits.
N_REQUIREMENT: dict[str, float] = {
    "corn_grain": 1.05,
    "corn_silage": 11.0,
    "soybean_full": 0.0,
    "soybean_double": 0.0,
    "sorghum_grain": 1.00,
    "wheat": 1.30,
    "barley": 1.20,
    "sweet_corn": 18.0,
    "potato": 0.55,
    "lima_bean": 0.020,
    "watermelon": 0.0035,
}

# Previous crop nitrogen credits, lb N/ac.
ROTATION_CREDITS: dict[str, float] = {
    "soybean_full": 35.0,
    "soybean_double": 25.0,
    "lima_bean": 30.0,
    "corn_grain": 0.0,
    "sorghum_grain": 0.0,
    "wheat": 0.0,
    "barley": 0.0,
    "fallow": 0.0,
}

# Cover crop credits depend on species and kill timing.
COVER_CREDITS: dict[str, float] = {
    "none": 0.0,
    "cereal_rye": -10.0,          # can tie up N when killed late with high C:N
    "cereal_rye_early_kill": 0.0,
    "crimson_clover": 45.0,
    "hairy_vetch": 60.0,
    "radish": 15.0,
    "rye_clover_mix": 30.0,
}


@dataclass(slots=True)
class ManureSource:
    """Organic nutrient source with Delaware-typical analysis.

    Broiler litter analysis varies widely with bedding and house management, so
    the values below are book values only. The interface always asks for the
    grower's own manure analysis first and falls back to these.
    """

    key: str
    label: str
    total_n_lb_per_ton: float
    ammonium_frac: float
    organic_n_frac: float
    p2o5_lb_per_ton: float
    k2o_lb_per_ton: float
    moisture_pct: float


MANURE_SOURCES: dict[str, ManureSource] = {
    "broiler_litter": ManureSource(
        "broiler_litter", "Broiler litter, house cleanout",
        total_n_lb_per_ton=60.0, ammonium_frac=0.22, organic_n_frac=0.78,
        p2o5_lb_per_ton=58.0, k2o_lb_per_ton=48.0, moisture_pct=25.0,
    ),
    "broiler_cake": ManureSource(
        "broiler_cake", "Broiler cake, partial cleanout",
        total_n_lb_per_ton=68.0, ammonium_frac=0.28, organic_n_frac=0.72,
        p2o5_lb_per_ton=62.0, k2o_lb_per_ton=52.0, moisture_pct=32.0,
    ),
    "layer_manure": ManureSource(
        "layer_manure", "Layer manure",
        total_n_lb_per_ton=38.0, ammonium_frac=0.45, organic_n_frac=0.55,
        p2o5_lb_per_ton=56.0, k2o_lb_per_ton=30.0, moisture_pct=50.0,
    ),
    "dairy_solid": ManureSource(
        "dairy_solid", "Dairy manure, solid",
        total_n_lb_per_ton=10.0, ammonium_frac=0.25, organic_n_frac=0.75,
        p2o5_lb_per_ton=5.0, k2o_lb_per_ton=10.0, moisture_pct=75.0,
    ),
    "compost": ManureSource(
        "compost", "Composted litter",
        total_n_lb_per_ton=30.0, ammonium_frac=0.08, organic_n_frac=0.92,
        p2o5_lb_per_ton=35.0, k2o_lb_per_ton=28.0, moisture_pct=35.0,
    ),
}


def manure_available_n(
    source_key: str,
    rate_ton_per_ac: float,
    incorporation: str = "within_24h",
    years_since_first_application: int = 1,
    custom_total_n: float | None = None,
) -> dict:
    """Plant-available N from a manure application, lb N/ac.

    Ammonium volatilizes fast on a warm Delmarva spring day; surface applied and
    left on top, half of it is gone before the crop can use it. Organic N
    mineralizes over several seasons, so fields with a long litter history carry
    a carryover credit that regularly gets double-counted.
    """
    src = MANURE_SOURCES.get(source_key)
    if src is None:
        raise ValueError(f"Unknown manure source '{source_key}'")
    total_n = (custom_total_n or src.total_n_lb_per_ton) * rate_ton_per_ac

    volatilization = {
        "immediate": 0.05,
        "within_24h": 0.20,
        "within_72h": 0.40,
        "surface_no_incorporation": 0.55,
        "irrigated_in": 0.15,      # water moves ammonium into the soil quickly
    }.get(incorporation, 0.40)

    nh4 = total_n * src.ammonium_frac * (1.0 - volatilization)
    org = total_n * src.organic_n_frac
    year1 = org * 0.55                     # first-season mineralization
    carryover = 0.0
    if years_since_first_application > 1:
        # Geometric decay of prior years' organic pool.
        carryover = sum(org * 0.55 * (0.20 ** k) for k in range(1, min(4, years_since_first_application)))

    return {
        "total_n_applied": round(total_n, 1),
        "available_year1": round(nh4 + year1, 1),
        "ammonium_after_loss": round(nh4, 1),
        "mineralized_year1": round(year1, 1),
        "carryover_credit": round(carryover, 1),
        "p2o5_applied": round(src.p2o5_lb_per_ton * rate_ton_per_ac, 1),
        "k2o_applied": round(src.k2o_lb_per_ton * rate_ton_per_ac, 1),
        "volatilization_loss": round(total_n * src.ammonium_frac * volatilization, 1),
    }


@dataclass(slots=True)
class NitrogenPlan:
    crop_key: str
    yield_goal: float
    total_requirement_lb: float
    credits: dict[str, float]
    total_credits_lb: float
    recommended_fertilizer_lb: float
    split_schedule: list[dict]
    psnt_note: str
    leaching_risk: str
    warnings: list[str] = field(default_factory=list)


def build_nitrogen_plan(
    crop_key: str,
    yield_goal: float,
    previous_crop: str = "corn_grain",
    cover_crop: str = "none",
    manure: dict | None = None,
    soil_om_pct: float = 1.4,
    psnt_ppm: float | None = None,
    irrigated: bool = True,
    soil_awc_in_per_in: float = 0.10,
    planting: date | None = None,
) -> NitrogenPlan:
    """Build a split nitrogen schedule tuned to sandy irrigated ground.

    The core recommendation on Delaware sand is not the total rate, it is the
    splitting. A single large preplant application on Evesboro sand ahead of a
    wet May loses a meaningful fraction before the crop has a root system to
    catch it. Splitting into a small starter plus two or three fertigated or
    sidedressed increments matches supply to uptake and is the highest-leverage
    change most operations can make.
    """
    warnings: list[str] = []
    req_rate = N_REQUIREMENT.get(crop_key, 1.0)
    total_req = yield_goal * req_rate

    credits: dict[str, float] = {}

    # Organic matter mineralization: roughly 20 lb N per percent OM over a
    # season on these warm, well-aerated soils.
    credits["soil_organic_matter"] = round(soil_om_pct * 20.0, 1)
    credits["previous_crop"] = ROTATION_CREDITS.get(previous_crop, 0.0)
    credits["cover_crop"] = COVER_CREDITS.get(cover_crop, 0.0)

    if manure:
        m = manure_available_n(**manure)
        credits["manure_year1"] = m["available_year1"]
        credits["manure_carryover"] = m["carryover_credit"]
        if m["p2o5_applied"] > 120:
            warnings.append(
                f"This litter rate supplies {m['p2o5_applied']:.0f} lb P2O5/ac. "
                "Phosphorus, not nitrogen, is usually the limiting factor in a "
                "Delaware nutrient management plan. Check your P Site Index.")

    if irrigated and crop_key.startswith("corn"):
        credits["irrigation_water_n"] = 0.0   # populated from well nitrate test below

    if psnt_ppm is not None:
        # Pre-sidedress nitrate test, sampled at 12 inches when corn is 12 inches.
        if psnt_ppm >= 25:
            credits["psnt_adjustment"] = round(min(60.0, (psnt_ppm - 10) * 4.0), 1)
        elif psnt_ppm >= 15:
            credits["psnt_adjustment"] = round((psnt_ppm - 10) * 3.0, 1)
        else:
            credits["psnt_adjustment"] = 0.0
            warnings.append(
                f"PSNT of {psnt_ppm:.0f} ppm is below the response threshold. "
                "Plan on the full sidedress rate.")
        psnt_note = (
            f"PSNT {psnt_ppm:.0f} ppm applied as a {credits['psnt_adjustment']:.0f} "
            "lb/ac credit.")
    else:
        psnt_note = (
            "No PSNT entered. Pull a 12 inch sample when corn is knee high; on "
            "litter ground it frequently cuts the sidedress rate.")

    total_credits = sum(credits.values())
    recommended = max(0.0, total_req - total_credits)

    # --- splitting ---------------------------------------------------------
    leach_prone = soil_awc_in_per_in <= 0.11
    schedule: list[dict] = []
    if crop_key.startswith("corn"):
        if leach_prone:
            splits = [("Starter, at planting", 0.15, "2x2 band"),
                      ("V4 to V6 sidedress", 0.35, "Coulter or Y-drop"),
                      ("V10 to V12", 0.30, "Y-drop or fertigation"),
                      ("VT to R1", 0.20, "Fertigation")]
            note = (
                "Four-way split. On soils holding under 0.11 in/in, N applied "
                "before the crop can take it up is the fraction most likely to "
                "leave with the next two-inch rain.")
        else:
            splits = [("Starter, at planting", 0.25, "2x2 band"),
                      ("V6 sidedress", 0.45, "Coulter or Y-drop"),
                      ("V12 to VT", 0.30, "Y-drop or fertigation")]
            note = "Three-way split matched to uptake."
        for label, frac, method in splits:
            schedule.append({
                "timing": label,
                "lb_n_per_ac": round(recommended * frac),
                "method": method,
            })
        warnings.append(note)
    elif crop_key.startswith("soybean"):
        schedule.append({
            "timing": "No nitrogen recommended",
            "lb_n_per_ac": 0,
            "method": "Inoculate instead",
        })
        if yield_goal >= 70:
            warnings.append(
                "Supplemental N on soybean only tends to pay above about 70 bu/ac, "
                "applied between R2 and R4. Below that, put the money into "
                "inoculant.")
    else:
        schedule.append({"timing": "Preplant", "lb_n_per_ac": round(recommended * 0.4),
                         "method": "Broadcast incorporated"})
        schedule.append({"timing": "In-season", "lb_n_per_ac": round(recommended * 0.6),
                         "method": "Fertigation or sidedress"})

    # --- leaching risk -----------------------------------------------------
    if soil_awc_in_per_in <= 0.09:
        leaching = "high"
    elif soil_awc_in_per_in <= 0.13:
        leaching = "moderate"
    else:
        leaching = "low"

    warnings.append(
        "This is a decision aid. Reconcile rates and timings with your certified "
        "Delaware nutrient management plan before applying.")

    return NitrogenPlan(
        crop_key=crop_key,
        yield_goal=yield_goal,
        total_requirement_lb=round(total_req, 1),
        credits={k: round(v, 1) for k, v in credits.items()},
        total_credits_lb=round(total_credits, 1),
        recommended_fertilizer_lb=round(recommended, 1),
        split_schedule=schedule,
        psnt_note=psnt_note,
        leaching_risk=leaching,
        warnings=warnings,
    )


def nitrate_leaching_estimate(
    deep_percolation_in: float,
    soil_nitrate_lb_per_ac: float,
    root_depth_in: float,
    soil_fc_vol: float,
    irrigation_water_no3_n_ppm: float = 0.0,
    irrigation_in: float = 0.0,
) -> dict:
    """Mass of nitrate-N moving below the root zone, lb N/ac.

    Piston displacement through the mobile fraction of soil water. Crude next to
    a full Richards-equation solute transport model, but it is the right level of
    detail for a decision aid: it is directionally correct, it responds to the
    variables the grower controls, and it does not pretend to a precision the
    inputs cannot support.

    Irrigation water nitrate matters in Delaware. Shallow Columbia aquifer wells
    in Sussex County can carry meaningful nitrate-N, and water applied at 10 ppm
    across 12 inches of irrigation supplies over 25 lb N/ac that most growers do
    not credit.
    """
    water_in_zone = soil_fc_vol * root_depth_in
    if water_in_zone <= 0:
        return {"leached_lb_n": 0.0, "irrigation_n_applied_lb": 0.0}

    # Fraction of the resident soil solution displaced by the drainage event.
    displaced = min(1.0, deep_percolation_in / water_in_zone)
    # Mobile fraction: not all pore water participates; sands are closer to 1.
    mobility = 0.75
    leached = soil_nitrate_lb_per_ac * displaced * mobility

    # 1 ppm N in 1 acre-inch is about 0.2266 lb N/ac.
    irrigation_n = irrigation_water_no3_n_ppm * irrigation_in * 0.2266

    return {
        "leached_lb_n": round(leached, 1),
        "displaced_fraction": round(displaced, 3),
        "irrigation_n_applied_lb": round(irrigation_n, 1),
        "net_change_lb": round(irrigation_n - leached, 1),
        "risk_level": ("high" if leached > 25 else
                       "moderate" if leached > 10 else "low"),
    }


def water_nutrient_index(
    yield_farm: float, yield_baseline: float,
    irrigation_in: float, eto_in: float,
    n_applied: float, n_uptake_baseline: float,
    p_applied: float, p_uptake_baseline: float,
) -> float:
    """Water-Nutrient Intensification Performance Index (Eq. 6 of the narrative).

    Rewards yield gained per unit of water and nutrient intensification. Used as
    the scoring metric for the Delaware on-farm irrigation trials so growers can
    compare a season across fields with different soils.
    """
    if yield_baseline <= 0:
        return 0.0
    gain = (yield_farm / yield_baseline) - 1.0
    denom = (
        (1.0 + irrigation_in / max(0.01, eto_in))
        * (1.0 + n_applied / max(0.01, n_uptake_baseline))
        * (1.0 + p_applied / max(0.01, p_uptake_baseline))
    )
    return round(gain / denom, 4)
