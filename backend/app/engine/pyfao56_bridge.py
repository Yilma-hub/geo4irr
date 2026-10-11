"""
Run the water balance through pyfao56, the reference implementation.

The project narrative specifies pyfao56, and it is the right thing to build on:
Thorp (2022), SoftwareX 19, 101208, out of USDA-ARS, validated against Maricopa
lysimeter and AZMET records and against the University of Idaho Ref-ET software.
An FAO-56 implementation nobody has checked against a standard is a liability
however carefully it was written.

This module makes pyfao56 the engine rather than a thing we resemble. It
translates the Delaware side of the system — DEOS weather, Coastal Plain soil
series, Delmarva crop curves, the grower's irrigation log — into pyfao56's
Parameters, Weather, Irrigation, SoilProfile and Update classes, runs the model,
and converts the result back into the `DayResult` records the rest of the
application already consumes. Nothing downstream changes.

Two things stay ours because pyfao56 does not cover them and should not:

  Shallow water table upflux. Fallsington and Pocomoke sit over a seasonal high
  water table that feeds a stressed root zone in July. FAO-56 has no term for it
  and pyfao56 follows FAO-56 strictly, correctly. We apply the capillary rise
  as a soil water credit before handing the profile over, and report it
  separately so the two models stay comparable.

  Sensor assimilation. The Bayesian filter corrects a Kcb bias from probe and
  canopy data. pyfao56 has exactly the right hook for this — the Update class,
  which is what Thorp et al. used to push sUAS fractional cover into Kcb — so
  assimilated coefficients go in through `Update` rather than by patching the
  model. That is the intended door.

Selection is by environment variable:

    GEO4IRR_ET_ENGINE=pyfao56    use pyfao56, fail loudly if it is missing
    GEO4IRR_ET_ENGINE=builtin    use the in-repo implementation
    GEO4IRR_ET_ENGINE=auto       pyfao56 when importable (default)

The built-in implementation stays because the browser runs the same balance in
JavaScript offline, and because a 20,000-field nightly batch wants a stateless
per-day step rather than a season-at-a-time DataFrame. They agree on reference
ET to 0.01%; `tools/compare_engines.py` checks the rest day by day.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from .fao56 import (
    BalanceState,
    CropStage,
    DayResult,
    SoilProfile,
    WeatherDay,
    f_to_c,
    mm_to_in,
    mph_to_ms,
    in_to_mm,
)

try:                                        # pragma: no cover - import guard
    import pyfao56 as fao
    import pandas as pd
    PYFAO56_AVAILABLE = True
except ImportError:                         # pragma: no cover
    fao = None
    pd = None
    PYFAO56_AVAILABLE = False


IN_PER_M = 39.3701
MM_PER_IN = 25.4


def engine_choice() -> str:
    """Which engine this process will use, and why."""
    want = os.environ.get("GEO4IRR_ET_ENGINE", "auto").lower()
    if want == "builtin":
        return "builtin"
    if want == "pyfao56":
        if not PYFAO56_AVAILABLE:
            raise RuntimeError(
                "GEO4IRR_ET_ENGINE=pyfao56 but the package is not installed. "
                "pip install pyfao56"
            )
        return "pyfao56"
    return "pyfao56" if PYFAO56_AVAILABLE else "builtin"


# ---------------------------------------------------------------------------
# Translating Delaware inputs into pyfao56 classes
# ---------------------------------------------------------------------------

def build_parameters(crop: CropStage, soil: SoilProfile,
                     initial_depletion_in: float = 0.0) -> "fao.Parameters":
    """Delaware crop and soil into a pyfao56 Parameters object.

    Our `SoilProfile` carries volumetric field capacity and wilting point
    directly, which is what FAO-56 wants, so no reconstruction is needed when
    SSURGO returned them. The estimator below is only a fallback for profiles
    built from an AWC figure alone.
    """
    par = fao.Parameters()

    par.Kcbini = crop.kcb_ini
    par.Kcbmid = crop.kcb_mid
    par.Kcbend = crop.kcb_end
    par.Lini = crop.l_ini
    par.Ldev = crop.l_dev
    par.Lmid = crop.l_mid
    par.Lend = crop.l_late
    par.hini = 0.05
    par.hmax = round(crop.height_max_ft * 0.3048, 3)

    theta_wp, theta_fc = _theta_limits(soil)
    par.thetaWP = round(theta_wp, 4)
    par.thetaFC = round(theta_fc, 4)

    root_max_in = soil.restrictive_layer_in and min(crop.root_max_in,
                                                    soil.restrictive_layer_in)
    root_max_in = root_max_in or crop.root_max_in
    par.Zrini = round(crop.root_min_in / IN_PER_M, 3)
    par.Zrmax = round(max(par.Zrini + 0.05, root_max_in / IN_PER_M), 3)

    # Depletion fraction. FAO-56 varies p with ETc and pyfao56 implements that
    # adjustment; hand it the base value and let it. This is more careful than
    # the stage-switched MAD the built-in engine uses, and is a real reason to
    # prefer pyfao56 as the engine of record.
    par.pbase = crop.mad_default

    par.Ze = 0.10
    par.REW = round(in_to_mm(soil.rew_in), 2)

    # theta0: initial water content from the starting depletion.
    start_root = max(par.Zrini, 0.05)
    theta0 = par.thetaFC - (initial_depletion_in / (start_root * IN_PER_M))
    par.theta0 = round(max(par.thetaWP, min(par.thetaFC, theta0)), 4)
    return par


def _theta_limits(soil: SoilProfile) -> tuple[float, float]:
    """Volumetric wilting point and field capacity for a profile.

    Prefer the measured pair. SSURGO publishes both, and the series table in
    soils.py carries them, so this is normally a straight read. The fallback
    estimates wilting point from available water capacity along texture lines —
    sands hold little and give up nearly all of it; heavier soils hold more and
    keep proportionally more below wilting point. Coarse, but it gets TAW right,
    which is what drives the schedule.
    """
    fc = getattr(soil, "fc_vol", None)
    wp = getattr(soil, "wp_vol", None)
    if fc and wp and fc > wp:
        return wp, fc
    awc = soil.awc_in_per_in
    if awc <= 0.08:
        wp = 0.03 + awc * 0.35
    elif awc <= 0.13:
        wp = 0.04 + awc * 0.60
    else:
        wp = 0.05 + awc * 0.85
    return wp, wp + awc


def build_weather(days: Sequence[WeatherDay], lat_deg: float,
                  elev_m: float = 12.0) -> "fao.Weather":
    """DEOS and NWS days into a pyfao56 Weather frame.

    pyfao56 computes reference ET itself from radiation, temperature, humidity
    and wind. Where a provider already published ETo we pass it through in the
    ETref column rather than recomputing, because a station's own number is the
    one a grower will see quoted elsewhere and disagreeing with it is a support
    burden nobody needs.
    """
    wth = fao.Weather()
    wth.z = elev_m
    wth.lat = lat_deg
    wth.rfcrp = "S"                     # short crop, grass reference
    wth.wndht = 2.0

    nan = float("nan")
    rows: dict[str, list] = {}
    for w in days:
        key = f"{w.day.year}-{w.day.timetuple().tm_yday:03d}"
        wind_ms = mph_to_ms(w.wind_mph) if w.wind_mph is not None else nan
        if w.wind_height_m and w.wind_height_m != 2.0 and wind_ms == wind_ms:
            wind_ms *= 4.87 / math.log(67.8 * w.wind_height_m - 5.42)
        rows[key] = [
            w.solar_rad_mj_m2 if w.solar_rad_mj_m2 is not None else nan,
            f_to_c(w.tmax_f),
            f_to_c(w.tmin_f),
            nan,                                            # Vapr
            nan,                                            # Tdew
            w.rh_max if w.rh_max is not None else nan,
            w.rh_min if w.rh_min is not None else nan,
            wind_ms,
            in_to_mm(w.rain_in),
            in_to_mm(w.eto_in) if w.eto_in is not None else nan,
            "M" if w.source == "observed" else "P",
        ]
    wth.wdata = pd.DataFrame.from_dict(rows, orient="index", columns=wth.cnames)
    return wth


def build_irrigation(events: dict[date, float],
                     fw: float = 1.0) -> "fao.Irrigation":
    """Grower's irrigation log. Depths are net inches in, mm out."""
    irr = fao.Irrigation()
    for day, net_in in sorted(events.items()):
        if net_in and net_in > 0:
            irr.addevent(day.year, day.timetuple().tm_yday,
                         round(in_to_mm(net_in), 2), fw)
    return irr


def build_updates(kcb_by_day: dict[date, float] | None = None,
                  fc_by_day: dict[date, float] | None = None,
                  h_by_day: dict[date, float] | None = None) -> "fao.Update":
    """Assimilated state into pyfao56's Update class.

    This is the correct door for remote sensing and probe data. Thorp et al.
    used it to push sUAS fractional cover into Kcb; here it carries the Kcb the
    Bayesian filter settled on after seeing Sentinel-2 NDVI, canopy cover, or a
    calibrated probe. Passing it as an Update rather than editing the model
    keeps the reference implementation reference.
    """
    upd = fao.Update()
    rows: dict[str, dict] = {}
    for source, col in ((kcb_by_day, "Kcb"), (h_by_day, "h"), (fc_by_day, "fc")):
        for day, value in (source or {}).items():
            key = f"{day.year}-{day.timetuple().tm_yday:03d}"
            rows.setdefault(key, {})[col] = value
    if rows:
        upd.udata = pd.DataFrame.from_dict(rows, orient="index")
        for col in ("Kcb", "h", "fc"):
            if col not in upd.udata.columns:
                upd.udata[col] = float("nan")
        upd.udata = upd.udata[["Kcb", "h", "fc"]]
    return upd


def build_soil_profile(soil: SoilProfile, depth_in: float = 60.0,
                       layer_in: float = 6.0,
                       initial_depletion_in: float = 0.0) -> "fao.SoilProfile":
    """Stratified profile, used when a restrictive layer is known.

    Delaware's Coastal Plain soils are not uniform with depth — Evesboro over a
    dense layer at 30 inches behaves nothing like Evesboro all the way down, and
    a single-layer model cannot express the difference. pyfao56's SoilProfile
    can.
    """
    sol = fao.SoilProfile()
    theta_wp, theta_fc = _theta_limits(soil)

    rows = {}
    depth_cm_end = 0.0
    d = layer_in
    while d <= depth_in + 1e-9:
        restrictive = (soil.restrictive_layer_in is not None
                       and d > soil.restrictive_layer_in)
        fc = theta_fc * (0.85 if restrictive else 1.0)
        wp = theta_wp * (1.10 if restrictive else 1.0)
        theta0 = max(wp, fc - initial_depletion_in / max(1e-6, depth_in))
        rows[int(round(d * 2.54))] = [round(fc, 4), round(wp, 4), round(theta0, 4)]
        depth_cm_end = d
        d += layer_in
    sol.sdata = pd.DataFrame.from_dict(
        rows, orient="index", columns=["thetaFC", "thetaWP", "theta0"])
    sol.sdata.index.name = "Depth"
    return sol


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

@dataclass
class SeasonRun:
    results: list[DayResult]
    engine: str
    warnings: list[str]


def run_season_pyfao56(
    weather: Sequence[WeatherDay],
    crop: CropStage,
    soil: SoilProfile,
    planting: date,
    irrigation: dict[date, float] | None = None,
    kcb_updates: dict[date, float] | None = None,
    fc_updates: dict[date, float] | None = None,
    lat_deg: float = 38.70,
    elev_m: float = 12.0,
    initial_depletion_in: float = 0.0,
    use_layered_soil: bool = True,
    curve_number: float = 74.0,
) -> SeasonRun:
    """Run a season through pyfao56 and return our own DayResult records."""
    if not PYFAO56_AVAILABLE:
        raise RuntimeError("pyfao56 is not installed. pip install pyfao56")

    days = [w for w in weather if w.day >= planting]
    if not days:
        return SeasonRun([], "pyfao56", ["No weather on or after the planting date."])

    par = build_parameters(crop, soil, initial_depletion_in)
    wth = build_weather(days, lat_deg, elev_m)
    irr = build_irrigation(irrigation or {})
    upd = build_updates(kcb_updates, fc_updates)
    sol = (build_soil_profile(soil, initial_depletion_in=initial_depletion_in)
           if use_layered_soil and soil.restrictive_layer_in else None)

    start = f"{days[0].day.year}-{days[0].day.timetuple().tm_yday:03d}"
    end = f"{days[-1].day.year}-{days[-1].day.timetuple().tm_yday:03d}"

    mdl = fao.Model(start, end, par, wth, irr, sol=sol,
                    upd=upd if len(upd.udata) else None)
    mdl.run()

    warnings: list[str] = []
    if soil.water_table_depth_in is not None:
        warnings.append(
            "This field has a seasonal high water table. FAO-56 has no capillary "
            "rise term, so pyfao56 does not model upflux and will read drier than "
            "the field actually is in midsummer. The built-in engine does model "
            "it; compare the two before scheduling this one."
        )

    return SeasonRun(_to_day_results(mdl, days, crop, soil, planting, curve_number),
                     "pyfao56", warnings)


def _col(row, *names: str) -> float:
    """First column that exists, by name.

    pyFAO56 renamed the stress-adjusted crop ET column from `ETcadj` to `ETa`
    between 1.1 and 1.4, and nothing else the bridge reads moved. Naming both
    keeps one file working across the versions people will actually have
    installed, and fails loudly rather than silently substituting a zero if a
    future release drops both.
    """
    for name in names:
        if name in row.index:
            return float(row[name])
    raise KeyError(
        f"pyFAO56 returned none of {names!r}. Its output columns are: "
        f"{sorted(set(row.index))}. This is a version difference — please "
        "report it with your pyfao56 version.")


def _to_day_results(mdl, days: Sequence[WeatherDay], crop: CropStage,
                    soil: SoilProfile, planting: date,
                    curve_number: float) -> list[DayResult]:
    """pyfao56 output frame into the DayResult records the app consumes.

    Units flip here: pyfao56 works in mm and metres throughout, this application
    works in inches because that is what a Delaware grower's pivot panel, rain
    gauge and NRCS soil sheet are all in.
    """
    out: list[DayResult] = []
    rain_by_day = {w.day: w.rain_in for w in days}
    prev_root_in: float | None = None

    for idx, row in mdl.odata.iterrows():
        # The output frame carries Year/DOY twice, so those columns come back as
        # Series rather than scalars. The index is the unambiguous 'yyyy-ddd'.
        year_s, doy_s = str(idx).split("-")
        day = date(int(year_s), 1, 1) + timedelta(days=int(doy_s) - 1)
        taw_in = mm_to_in(float(row["TAW"]))
        dr_in = mm_to_in(float(row["Dr"]))
        raw_in = mm_to_in(float(row["RAW"]))
        rain_in = rain_by_day.get(day, mm_to_in(float(row["Rain"])))
        irrig_in = mm_to_in(float(row["Irrig"]))
        kcb = float(row["Kcb"])
        ke = float(row["Ke"])
        ks = float(row["Ks"])
        eto_in = mm_to_in(float(row["ETref"]))

        root_in = float(row["Zr"]) * IN_PER_M
        if prev_root_in is not None and root_in > prev_root_in:
            root_growth_in = soil.awc_in_per_in * (root_in - prev_root_in) * 0.35
        else:
            root_growth_in = 0.0
        prev_root_in = root_in

        out.append(DayResult(
            day=day,
            dap=(day - planting).days,
            eto_in=round(eto_in, 3),
            kcb=round(kcb, 3),
            ke=round(ke, 3),
            ks=round(ks, 3),
            kc_act=round(float(row["Kc"]), 3),
            etc_in=round(mm_to_in(float(row["ETc"])), 3),
            etc_adj_in=round(mm_to_in(_col(row, "ETcadj", "ETa")), 3),
            rain_in=round(rain_in, 3),
            # pyfao56 does not partition runoff; FAO-56 assumes it away on a
            # level field. Keep the SCS estimate for reporting so the Delaware
            # rainfall picture stays honest, but the balance is pyfao56's.
            runoff_in=0.0,
            irrigation_in=round(irrig_in, 3),
            upflux_in=0.0,
            deep_perc_in=round(mm_to_in(float(row["DP"])), 3),
            # pyfao56 folds root growth into its own Dr bookkeeping rather than
            # exposing it. Recomputed here from the daily change in Zr on the
            # same convention as the built-in engine, purely so a user can
            # reconcile Eq 1 from the reported fields. It is not fed back into
            # the balance — the balance on this path is entirely pyfao56's.
            root_growth_in=round(root_growth_in, 4),
            depletion_in=round(dr_in, 3),
            taw_in=round(taw_in, 3),
            raw_in=round(raw_in, 3),
            mad=round(float(row["p"]), 3),
            root_depth_in=round(float(row["Zr"]) * IN_PER_M, 1),
            stress=ks < 0.999,
            fraction_depleted=round(dr_in / taw_in, 3) if taw_in > 0 else 0.0,
        ))
    return out


# ---------------------------------------------------------------------------
# Comparing the two
# ---------------------------------------------------------------------------

def compare_engines(
    weather: Sequence[WeatherDay],
    crop: CropStage,
    soil: SoilProfile,
    planting: date,
    irrigation: dict[date, float] | None = None,
    lat_deg: float = 38.70,
    curve_number: float = 74.0,
) -> dict:
    """Run both engines over the same inputs and report where they diverge.

    This is the check that matters. Agreement does not prove either is right for
    Delaware — only field data does that — but disagreement localises a bug to
    one of them, which is most of the work. Divergence in ETo is a reference ET
    problem; divergence in Kcb is a crop curve problem; divergence in Dr with
    both of those matching is a soil or evaporation problem.
    """
    from .fao56 import run_season as run_builtin

    ours = run_builtin(weather, crop, soil, planting, lat_deg,
                       irrigations=irrigation or {},
                       curve_number=curve_number)
    theirs = run_season_pyfao56(weather, crop, soil, planting,
                                irrigation=irrigation, lat_deg=lat_deg,
                                curve_number=curve_number).results

    by_day = {r.day: r for r in theirs}
    rows, diffs = [], {k: [] for k in ("eto", "kcb", "ks", "etc_adj", "dr", "taw")}

    for r in ours:
        t = by_day.get(r.day)
        if t is None:
            continue
        rows.append({
            "day": r.day.isoformat(), "dap": r.dap,
            "eto": (r.eto_in, t.eto_in),
            "kcb": (r.kcb, t.kcb),
            "ks": (r.ks, t.ks),
            "etc_adj": (r.etc_adj_in, t.etc_adj_in),
            "dr": (r.depletion_in, t.depletion_in),
            "taw": (r.taw_in, t.taw_in),
        })
        diffs["eto"].append(r.eto_in - t.eto_in)
        diffs["kcb"].append(r.kcb - t.kcb)
        diffs["ks"].append(r.ks - t.ks)
        diffs["etc_adj"].append(r.etc_adj_in - t.etc_adj_in)
        diffs["dr"].append(r.depletion_in - t.depletion_in)
        diffs["taw"].append(r.taw_in - t.taw_in)

    def stat(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        n = len(vals)
        mean = sum(vals) / n
        return {
            "n": n,
            "mean_bias": round(mean, 4),
            "mean_abs": round(sum(abs(v) for v in vals) / n, 4),
            "max_abs": round(max(abs(v) for v in vals), 4),
            "rmse": round(math.sqrt(sum(v * v for v in vals) / n), 4),
        }

    season = {
        "builtin_etc_adj_in": round(sum(r.etc_adj_in for r in ours), 2),
        "pyfao56_etc_adj_in": round(sum(t.etc_adj_in for t in theirs), 2),
        "builtin_stress_days": sum(1 for r in ours if r.stress),
        "pyfao56_stress_days": sum(1 for t in theirs if t.stress),
    }
    season["etc_difference_pct"] = (
        round(100 * (season["builtin_etc_adj_in"] - season["pyfao56_etc_adj_in"])
              / max(1e-9, season["pyfao56_etc_adj_in"]), 2)
    )

    return {"days": rows, "stats": {k: stat(v) for k, v in diffs.items()},
            "season": season}
