#!/usr/bin/env python3
"""
Compare the built-in FAO-56 engine against pyfao56, day by day.

pyfao56 (Thorp, K. R., 2022, SoftwareX 19, 101208) is the reference
implementation: USDA-ARS, peer reviewed, validated against Maricopa lysimeter
data, AZMET, and the University of Idaho Ref-ET software. The engine in
`app/engine/fao56.py` is validated against nothing but its own tests. Where the
two disagree, pyfao56 is presumed correct until shown otherwise.

Agreement here does not prove either is right for Delaware — only field data
does that. What it does is localise a problem. Divergence in ETo is a reference
ET bug. Divergence in Kcb is a crop curve or stage length problem. Divergence in
depletion with both of those matching is soil, root growth, or evaporation.

    pip install pyfao56
    python tools/compare_engines.py
    python tools/compare_engines.py --crop soybean_full --soil matapeake
    python tools/compare_engines.py --daily | head -40
    python tools/compare_engines.py --all-soils
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))

from app.engine import pyfao56_bridge as bridge          # noqa: E402
from app.engine.crops import CROPS, get_crop             # noqa: E402
from app.engine.fao56 import WeatherDay                  # noqa: E402
from app.engine.soils import SOILS, get_soil             # noqa: E402


def synthetic_season(planting: date, days: int, lat: float = 38.64,
                     rain_every: int = 6, rain_in: float = 0.45
                     ) -> list[WeatherDay]:
    """A deterministic Delaware-shaped season.

    Not real weather — the point is that both engines see exactly the same
    input, so any difference in output is theirs and not the weather's. Use
    --deos with a real feed once a data agreement is in place.
    """
    out = []
    for i in range(days):
        d = planting + timedelta(days=i)
        doy = d.timetuple().tm_yday
        t = math.sin((doy - 80) / 200 * math.pi)
        out.append(WeatherDay(
            day=d,
            tmax_f=68 + 24 * t,
            tmin_f=48 + 22 * t,
            rh_max=93, rh_min=52,
            wind_mph=5.5,
            solar_rad_mj_m2=12 + 16 * t,
            rain_in=rain_in if i % rain_every == 0 else 0.0,
        ))
    return out


def report(crop_key: str, soil_key: str, planting: date, season_days: int,
           irrigate_every: int, depth_in: float, daily: bool) -> dict:
    crop = get_crop(crop_key)
    soil = get_soil(soil_key)
    weather = synthetic_season(planting, season_days)
    irrigation = {}
    if irrigate_every:
        for k in range(45, season_days - 10, irrigate_every):
            irrigation[planting + timedelta(days=k)] = depth_in

    out = bridge.compare_engines(weather, crop, soil, planting,
                                 irrigation=irrigation, lat_deg=38.64)

    print(f"\n{'=' * 74}")
    print(f"{crop.name}  on  {soil.series}")
    print(f"planted {planting}, {season_days} days, "
          f"{len(irrigation)} irrigations of {depth_in} in")
    print("=" * 74)

    print(f"\n{'variable':10s} {'bias':>10s} {'mean |d|':>10s} "
          f"{'max |d|':>10s} {'rmse':>10s}")
    print("-" * 74)
    labels = {"eto": "ETo (in)", "kcb": "Kcb", "ks": "Ks",
              "etc_adj": "ETc adj", "dr": "Dr (in)", "taw": "TAW (in)"}
    for key, s in out["stats"].items():
        print(f"{labels[key]:10s} {s['mean_bias']:>+10.4f} {s['mean_abs']:>10.4f} "
              f"{s['max_abs']:>10.4f} {s['rmse']:>10.4f}")

    se = out["season"]
    print(f"\nseason crop water use   built-in {se['builtin_etc_adj_in']:6.2f} in"
          f"    pyfao56 {se['pyfao56_etc_adj_in']:6.2f} in"
          f"    ({se['etc_difference_pct']:+.1f}%)")
    print(f"stress days             built-in {se['builtin_stress_days']:6d}   "
          f"    pyfao56 {se['pyfao56_stress_days']:6d}")

    verdict(out)

    if daily:
        print(f"\n{'date':12s} {'dap':>4s} "
              f"{'ETo b/p':>15s} {'Kcb b/p':>13s} {'Dr b/p':>15s} {'Ks b/p':>13s}")
        print("-" * 74)
        for r in out["days"]:
            print(f"{r['day']:12s} {r['dap']:>4d} "
                  f"{r['eto'][0]:>7.3f}/{r['eto'][1]:<7.3f} "
                  f"{r['kcb'][0]:>6.2f}/{r['kcb'][1]:<6.2f} "
                  f"{r['dr'][0]:>7.2f}/{r['dr'][1]:<7.2f} "
                  f"{r['ks'][0]:>6.2f}/{r['ks'][1]:<6.2f}")
    return out


def verdict(out: dict) -> None:
    s = out["stats"]
    print()
    if s["eto"]["max_abs"] > 0.005:
        print("! Reference ET disagrees. Both claim ASCE standardized, so this is")
        print("  a bug in one of them. Fix it before reading anything below.")
    else:
        print("+ Reference ET matches. Both are computing ASCE standardized ETo")
        print("  the same way, which is the foundation everything else sits on.")

    if s["kcb"]["max_abs"] > 0.02:
        print("! Crop coefficients diverge. Check stage lengths and the Kcb triple")
        print("  against FAO-56 Tables 11 and 17.")
    else:
        print("+ Crop coefficient curves match.")

    if s["taw"]["max_abs"] > 0.10:
        print("! Total available water diverges, so the two grow roots differently.")
        print("  FAO-56 expands the root zone in proportion to crop development;")
        print("  the built-in engine ties it to canopy cover. That is a deliberate")
        print("  choice and it propagates into depletion and stress. Decide which")
        print("  behaviour you want before a grower sees either number.")

    if abs(out["season"]["etc_difference_pct"]) > 5:
        print("! Season crop water use differs by more than 5 percent, which on")
        print("  Delaware sand is about one irrigation over a season.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="corn_grain", choices=sorted(CROPS))
    ap.add_argument("--soil", default="evesboro", choices=sorted(SOILS))
    ap.add_argument("--planting", default="2026-04-28")
    ap.add_argument("--days", type=int, default=130)
    ap.add_argument("--irrigate-every", type=int, default=8,
                    help="days between irrigations; 0 for dryland")
    ap.add_argument("--depth", type=float, default=0.8, help="net inches per pass")
    ap.add_argument("--daily", action="store_true", help="print every day")
    ap.add_argument("--all-soils", action="store_true",
                    help="sweep every Delaware series")
    args = ap.parse_args()

    if not bridge.PYFAO56_AVAILABLE:
        print("pyfao56 is not installed.\n\n  pip install pyfao56\n"
              "  pip install 'pyfao56 @ git+https://github.com/isukendall/pyfao56.git'")
        raise SystemExit(1)

    planting = date.fromisoformat(args.planting)
    soils = sorted(SOILS) if args.all_soils else [args.soil]

    worst = 0.0
    for key in soils:
        out = report(args.crop, key, planting, args.days,
                     args.irrigate_every, args.depth, args.daily and not args.all_soils)
        worst = max(worst, abs(out["season"]["etc_difference_pct"]))

    if args.all_soils:
        print(f"\n{'=' * 74}")
        print(f"Largest season crop water use difference across all soils: "
              f"{worst:.1f}%")
        print("=" * 74)


if __name__ == "__main__":
    main()
