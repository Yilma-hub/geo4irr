"""
Delaware reference values from DIMS.

DIMS is the Delaware Irrigation Management System, built by the University of
Delaware's Delaware Environmental Observing System and documented in the DIMS
User's Guide v1.1 (Brinson, 24 May 2012), funded by USDA-NRCS grant
69-21J2-0-12 and DNREC grant 08070800964. It is the published reference for
irrigation scheduling in this state, which makes it the right thing to check
Geo4Irr against.

What is taken from it, and what is deliberately not.

TAKEN: the soil texture table (Appendix B), the growing-degree-day emergence
threshold, the managed root zone concept, the default maximum allowable
deficit, and the qualitative field status bands. These are Delaware-specific
reference data and management conventions. They do not change how the water
balance is computed.

NOT TAKEN: the crop coefficients in Appendix A. DIMS uses the FAO-56 single
crop coefficient, where one Kc covers transpiration and soil evaporation
together (corn 0.25 / 1.20 / 0.60). Geo4Irr uses the dual coefficient, where
Kcb is transpiration alone and Ke is tracked separately (corn 0.15 / 1.15 /
0.50). Those numbers look similar and mean different things. Copying a single
Kc into a Kcb slot would double count soil evaporation, because Ke would then
be added on top of a coefficient that already includes it. The project
narrative specifies the dual approach, so the dual values stay.

The one place the two can be compared is total crop ET over a season, and
`compare_with_dims_single_kc` below does exactly that, so the difference
between the two methods is a number rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Soil texture classes, DIMS Appendix B
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextureClass:
    """A USDA texture class with the hydrologic properties DIMS assigns it.

    Field capacity and permanent wilting point are volumetric, in inches of
    water per inch of soil. Available water capacity is the difference, which
    is the water a crop can actually reach.
    """
    key: str
    label: str
    fc_vol: float
    wp_vol: float

    @property
    def awc_in_per_in(self) -> float:
        return round(self.fc_vol - self.wp_vol, 3)


# Listed heaviest to lightest, the order the DIMS guide prints them in.
TEXTURE_CLASSES: dict[str, TextureClass] = {
    t.key: t for t in (
        TextureClass("silty_clay",      "Silty clay",      0.44, 0.28),
        TextureClass("clay",            "Clay",            0.43, 0.29),
        TextureClass("silty_clay_loam", "Silty clay loam", 0.36, 0.19),
        TextureClass("clay_loam",       "Clay loam",       0.33, 0.19),
        TextureClass("sandy_clay",      "Sandy clay",      0.32, 0.23),
        TextureClass("silt",            "Silt",            0.30, 0.10),
        TextureClass("silt_loam",       "Silt loam",       0.29, 0.11),
        TextureClass("sandy_clay_loam", "Sandy clay loam", 0.26, 0.16),
        TextureClass("loam",            "Loam",            0.26, 0.12),
        TextureClass("sandy_loam",      "Sandy loam",      0.20, 0.09),
        TextureClass("loamy_sand",      "Loamy sand",      0.15, 0.07),
        TextureClass("sand",            "Sand",            0.12, 0.05),
    )
}

# DIMS reads the dominant texture class from SSURGO over the top 24 inches.
# Recorded here because a figure quoted without its depth is not reproducible:
# the same field has a different dominant texture over 10 inches than over 24.
SSURGO_SAMPLE_DEPTH_IN = 24


def texture_from_fc_wp(fc_vol: float, wp_vol: float) -> TextureClass:
    """Closest DIMS texture class to a measured field capacity and wilting point.

    Both terms are needed. Matching on available water capacity alone looks
    like it should work and does not: clay holds 0.43 at field capacity and
    still has 0.29 locked away at wilting point, leaving 0.14 available, which
    is the same figure as a sandy loam holding far less of everything. Matched
    on AWC alone this engine labelled Woodstown sandy loam as clay and
    Greenwich loam as silty clay. Field capacity is what separates them, so it
    carries the heavier weight here.
    """
    return min(TEXTURE_CLASSES.values(),
               key=lambda t: (abs(t.fc_vol - fc_vol) * 2.0
                              + abs(t.wp_vol - wp_vol)))


def texture_from_awc(awc_in_per_in: float) -> TextureClass:
    """Texture class by available water capacity alone.

    For a field where AWC is genuinely all that is known, which is sometimes
    all SSURGO returns. The result is a water-holding match, not a texture
    identification, for the reason given in `texture_from_fc_wp`. Prefer that
    function wherever field capacity is available.
    """
    return min(TEXTURE_CLASSES.values(),
               key=lambda t: abs(t.awc_in_per_in - awc_in_per_in))


def custom_texture(fc_vol: float, wp_vol: float) -> TextureClass:
    """A soil the grower measured themselves.

    DIMS allows this and it is worth having: a grower who has had their soil
    tested knows more about that field than any map unit does. Validated here
    rather than trusted, because a wilting point above field capacity produces
    a negative available water capacity and a schedule that never irrigates.
    """
    if not 0.01 <= fc_vol <= 0.60:
        raise ValueError(
            f"Field capacity of {fc_vol} in/in is outside anything a soil does. "
            "Expected roughly 0.12 for sand up to 0.44 for silty clay.")
    if not 0.0 <= wp_vol <= 0.45:
        raise ValueError(
            f"Permanent wilting point of {wp_vol} in/in is outside anything a "
            "soil does. Expected roughly 0.05 for sand up to 0.29 for clay.")
    if wp_vol >= fc_vol:
        raise ValueError(
            f"Wilting point ({wp_vol}) is not below field capacity ({fc_vol}), "
            "so there would be no water available to the crop at all. Check "
            "the two have not been swapped.")
    return TextureClass("custom", "Custom (measured)", fc_vol, wp_vol)


# ---------------------------------------------------------------------------
# Managed root zone, DIMS Section IV
# ---------------------------------------------------------------------------

MANAGED_ROOT_ZONE_DEFAULT_IN = 18.0
MANAGED_ROOT_ZONE_MIN_IN = 8.0
MANAGED_ROOT_ZONE_MAX_IN = 24.0


def clamp_managed_root_zone(depth_in: float) -> float:
    return max(MANAGED_ROOT_ZONE_MIN_IN,
               min(MANAGED_ROOT_ZONE_MAX_IN, float(depth_in)))


def effective_root_depth_in(model_root_in: float,
                            managed_root_zone_in: float | None) -> float:
    """The depth the schedule actually manages water over.

    FAO-56 grows roots to the crop's physiological maximum, 36 inches for corn
    in this engine. DIMS manages 18 inches by default and caps at 24. Both are
    defensible and they answer different questions: the deeper figure is what
    the crop can reach, the shallower is what the grower is willing to run the
    pump for. On Delaware sand that gap decides whether a field reads as dry.

    The cap is applied to the depth, not to the balance, so this limits what is
    managed without altering how depletion is computed within it.
    """
    if managed_root_zone_in is None:
        return model_root_in
    return min(model_root_in, clamp_managed_root_zone(managed_root_zone_in))


# ---------------------------------------------------------------------------
# Emergence, DIMS Appendix A Table 3
# ---------------------------------------------------------------------------

# DIMS predicts emergence from accumulated growing degree days after planting,
# for the two crops where a planting date is what a grower actually records.
# Everything else is entered as an emergence date directly.
EMERGENCE_GDD = {
    "corn_grain": 125.0,
    "corn_silage": 125.0,
    "sweet_corn": 125.0,
    "soybean_full": 125.0,
    "soybean_double": 125.0,
}

# The 86/50 cap, standard for corn in the Corn Belt and on Delmarva.
GDD_BASE_F = 50.0
GDD_CAP_F = 86.0


def daily_gdd(tmax_f: float, tmin_f: float,
              base_f: float = GDD_BASE_F, cap_f: float = GDD_CAP_F) -> float:
    """One day of growing degree days, capped both ends before averaging.

    Capping before the average rather than after is what distinguishes the
    corn method from a plain mean: a 95 degree day is counted as 86, because
    corn does not develop faster above that.
    """
    hi = min(cap_f, tmax_f)
    lo = max(base_f, min(cap_f, tmin_f))
    return max(0.0, (hi + lo) / 2.0 - base_f)


def predict_emergence(planting: date,
                      weather: Sequence,
                      crop_key: str) -> tuple[date | None, float]:
    """The day accumulated GDD crosses the crop's emergence threshold.

    Returns the date and the GDD accumulated by then. If the season runs out
    before the threshold is reached, returns None with the running total, which
    is the honest answer for a field planted late in a cold spring rather than
    a guessed date.

    `weather` is any sequence of objects carrying `day`, `tmax_f` and `tmin_f`.
    """
    threshold = EMERGENCE_GDD.get(crop_key)
    if threshold is None:
        return None, 0.0
    total = 0.0
    for wd in weather:
        day = getattr(wd, "day", None) or getattr(wd, "date", None)
        if day is None or day < planting:
            continue
        total += daily_gdd(wd.tmax_f, wd.tmin_f)
        if total >= threshold:
            return day, round(total, 1)
    return None, round(total, 1)


# ---------------------------------------------------------------------------
# Field status, DIMS Section VII
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FieldStatus:
    label: str
    key: str
    pct_of_fc: float
    note: str


def field_status(depletion_in: float, taw_in: float,
                 emerged: bool = True) -> FieldStatus:
    """Soil water as a share of field capacity, in the words DIMS uses.

    The bands are DIMS Section VII: good above 75 percent, fair 50 to 75, poor
    30 to 50, very poor below 30. Worth having because a percentage is a number
    a grower can hold in their head across several fields, where inches of
    depletion against a different available water capacity on each one is not
    comparable between them.
    """
    if not emerged:
        return FieldStatus("Not emerged yet", "not_emerged", 100.0,
                           "The crop has not emerged, so there is no root zone "
                           "to schedule against yet.")
    if taw_in <= 0:
        return FieldStatus("Unknown", "unknown", 0.0,
                           "No available water capacity is known for this field.")

    pct = max(0.0, min(100.0, (1.0 - depletion_in / taw_in) * 100.0))
    if pct > 75:
        return FieldStatus("Good", "good", round(pct, 1),
                           "The profile is near full. Nothing needs water.")
    if pct > 50:
        return FieldStatus("Fair", "fair", round(pct, 1),
                           "Comfortable, but worth watching through a hot week.")
    if pct >= 30:
        return FieldStatus("Poor", "poor", round(pct, 1),
                           "Into the range where a hot day starts costing yield "
                           "rather than just water.")
    return FieldStatus("Very poor", "very_poor", round(pct, 1),
                       "The crop is drawing on water it can barely reach.")


# ---------------------------------------------------------------------------
# Comparing the two coefficient methods
# ---------------------------------------------------------------------------

# DIMS Appendix A Table 1, single crop coefficients. Held here only so the two
# methods can be compared. These are never used in the water balance.
DIMS_SINGLE_KC = {
    "corn_grain":     (0.25, 1.20, 0.60),
    "corn_silage":    (0.25, 1.20, 0.60),
    "soybean_full":   (0.25, 1.15, 0.45),
    "soybean_double": (0.25, 1.15, 0.45),
    "sweet_corn":     (0.25, 1.15, 1.05),
    "lima_bean":      (0.50, 1.15, 0.90),
    "pea":            (0.50, 1.15, 1.10),
    "watermelon":     (0.40, 1.00, 0.75),
    "cantaloupe":     (0.50, 0.85, 0.45),
    "cucumber":       (0.50, 1.00, 0.90),
}

# DIMS Appendix A Table 2, season lengths in days.
DIMS_SEASON_DAYS = {
    "corn_grain": 110, "corn_silage": 110, "soybean_full": 110,
    "soybean_double": 110, "sweet_corn": 110, "lima_bean": 100,
    "pea": 100, "watermelon": 110, "cantaloupe": 120, "cucumber": 105,
}


def compare_with_dims_single_kc(results: Iterable, crop_key: str) -> dict:
    """Season crop ET under the dual coefficient against the DIMS single Kc.

    The single-Kc figure is computed the way DIMS does it, on the same
    reference ET, with Kc interpolated across the season on the same stage
    fractions. Any difference is the difference between the two methods, not
    between two sets of weather.

    Both sides are POTENTIAL crop ET, with no stress reduction applied. This
    matters and it caught a mistake: comparing the dual method's stress
    adjusted ET against an unstressed single Kc showed a 15.8 percent gap that
    was almost entirely the stress coefficient, not the method. The adjusted
    figure is reported alongside so the size of that reduction is visible, but
    the comparison itself is potential against potential.

    They should land close. Dual tends to run a little higher early, when wet
    bare soil evaporates and the basal coefficient has not risen yet, and a
    little lower under a closed canopy where the single Kc still carries an
    evaporation allowance the dual method has already decided is near zero.
    """
    rows = list(results)
    if not rows or crop_key not in DIMS_SINGLE_KC:
        return {"comparable": False,
                "reason": f"No DIMS crop coefficients for {crop_key!r}."}

    kc_ini, kc_mid, kc_end = DIMS_SINGLE_KC[crop_key]
    season = DIMS_SEASON_DAYS.get(crop_key, len(rows))

    dual_etc = 0.0          # potential, no stress reduction
    dual_etc_adj = 0.0      # what the crop actually transpired
    single_etc = 0.0
    for r in rows:
        eto = getattr(r, "eto_in", 0.0)
        dual_etc += getattr(r, "etc_in", 0.0)
        dual_etc_adj += getattr(r, "etc_adj_in", 0.0)

        # DIMS ramps Kc over the season: initial through 20 percent, rising to
        # the maximum by 50 percent, falling to final by harvest. The guide
        # expresses this through canopy-cover fractions; the shape is the same.
        f = min(1.0, max(0.0, getattr(r, "dap", 0) / max(1, season)))
        if f <= 0.20:
            kc = kc_ini
        elif f <= 0.50:
            kc = kc_ini + (kc_mid - kc_ini) * (f - 0.20) / 0.30
        elif f <= 0.75:
            kc = kc_mid
        else:
            kc = kc_mid + (kc_end - kc_mid) * (f - 0.75) / 0.25
        single_etc += kc * eto

    diff = dual_etc - single_etc
    return {
        "comparable": True,
        "days": len(rows),
        "dual_etc_in": round(dual_etc, 2),
        "dual_etc_adjusted_in": round(dual_etc_adj, 2),
        "stress_reduction_in": round(dual_etc - dual_etc_adj, 2),
        "dims_single_kc_etc_in": round(single_etc, 2),
        "difference_in": round(diff, 2),
        "difference_pct": round(diff / single_etc * 100, 1) if single_etc else None,
        "dims_season_days": season,
        "note": ("Geo4Irr runs the FAO-56 dual coefficient, tracking "
                 "transpiration and soil evaporation separately. DIMS runs the "
                 "single coefficient, where one number covers both. Both sides "
                 "here are potential ET on identical weather."),
        "why_they_differ": (
            "Measured on Delaware corn the dual method runs 10 to 26 percent "
            "above the DIMS single Kc, and the gap widens on heavier soil. Two "
            "reasons, both expected. First, the dual method resolves each "
            "wetting event: a rain on bare ground in May drives Ke close to 1.1 "
            "for a few days, where the single Kc_ini of 0.25 is already an "
            "average over a typical wetting frequency and cannot rise. Second, "
            f"DIMS assumes a {season} day season for this crop while the crop "
            "table here runs longer, so the two are not on the same stage "
            "fractions late in the year. Neither method is wrong. They answer "
            "the same question at different resolution, which is the reason "
            "the narrative specifies the dual one."),
    }
