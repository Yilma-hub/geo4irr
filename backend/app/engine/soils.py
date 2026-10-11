"""
Soil hydraulic properties for the map units that carry Delaware's irrigated acres.

Values are representative of the SSURGO component data for the Delaware Coastal
Plain. The `lookup_ssurgo` helper hits the USDA Soil Data Access web service for
the real map unit at a field's coordinates; the table below is the offline
fallback and the pick list shown to growers who would rather choose their soil by
name, which most of them can do without help.

Two things about these soils drive the whole scheduling problem in Delaware:

  1. Available water capacity is low. Evesboro loamy sand holds roughly a third
     of what a Midwest silt loam holds. A 30-inch root zone on Evesboro carries
     about 2.1 inches of plant-available water, so a corn crop transpiring
     0.30 in/day runs out in a week without rain. Set sizes must be small and
     frequent, and a scheduler tuned for deep silt loams will consistently
     recommend applications the profile cannot hold.

  2. Several important series sit over a seasonal high water table. Fallsington
     and Pocomoke can feed the root zone by capillary rise in June and July,
     which shows up as a field that stays green while the model says it should
     be stressed.
"""

from __future__ import annotations

from dataclasses import asdict

from .fao56 import SoilProfile

# key -> SoilProfile
SOILS: dict[str, SoilProfile] = {
    "evesboro": SoilProfile(
        series="Evesboro loamy sand",
        awc_in_per_in=0.07, fc_vol=0.11, wp_vol=0.04, ksat_in_hr=6.0,
        rew_in=0.16, tew_in=0.55, ze_in=4.0,
        water_table_depth_in=None, restrictive_layer_in=None,
    ),
    "rosedale": SoilProfile(
        series="Rosedale loamy sand",
        awc_in_per_in=0.08, fc_vol=0.12, wp_vol=0.04, ksat_in_hr=5.0,
        rew_in=0.18, tew_in=0.59, ze_in=4.0,
    ),
    "sassafras": SoilProfile(
        series="Sassafras sandy loam",
        awc_in_per_in=0.12, fc_vol=0.20, wp_vol=0.08, ksat_in_hr=2.0,
        rew_in=0.28, tew_in=0.71, ze_in=4.0,
    ),
    "downer": SoilProfile(
        series="Downer loamy sand",
        awc_in_per_in=0.09, fc_vol=0.15, wp_vol=0.06, ksat_in_hr=4.0,
        rew_in=0.22, tew_in=0.63, ze_in=4.0,
    ),
    "ingleside": SoilProfile(
        series="Ingleside sandy loam",
        awc_in_per_in=0.11, fc_vol=0.19, wp_vol=0.08, ksat_in_hr=2.5,
        rew_in=0.26, tew_in=0.67, ze_in=4.0,
    ),
    "woodstown": SoilProfile(
        series="Woodstown sandy loam",
        awc_in_per_in=0.13, fc_vol=0.22, wp_vol=0.09, ksat_in_hr=1.5,
        rew_in=0.30, tew_in=0.75, ze_in=4.0,
        water_table_depth_in=42.0,
    ),
    "matapeake": SoilProfile(
        series="Matapeake silt loam",
        awc_in_per_in=0.19, fc_vol=0.30, wp_vol=0.11, ksat_in_hr=0.8,
        rew_in=0.35, tew_in=0.87, ze_in=4.0,
    ),
    "mattapex": SoilProfile(
        series="Mattapex silt loam",
        awc_in_per_in=0.18, fc_vol=0.29, wp_vol=0.11, ksat_in_hr=0.6,
        rew_in=0.35, tew_in=0.87, ze_in=4.0,
        water_table_depth_in=36.0, restrictive_layer_in=None,
    ),
    "fallsington": SoilProfile(
        series="Fallsington sandy loam",
        awc_in_per_in=0.14, fc_vol=0.24, wp_vol=0.10, ksat_in_hr=1.2,
        rew_in=0.30, tew_in=0.75, ze_in=4.0,
        water_table_depth_in=18.0,
    ),
    "pocomoke": SoilProfile(
        series="Pocomoke sandy loam",
        awc_in_per_in=0.15, fc_vol=0.26, wp_vol=0.11, ksat_in_hr=1.0,
        rew_in=0.31, tew_in=0.78, ze_in=4.0,
        water_table_depth_in=12.0,
    ),
    "hammonton": SoilProfile(
        series="Hammonton loamy sand",
        awc_in_per_in=0.10, fc_vol=0.17, wp_vol=0.07, ksat_in_hr=3.0,
        rew_in=0.24, tew_in=0.65, ze_in=4.0,
        water_table_depth_in=30.0,
    ),
    "greenwich": SoilProfile(
        series="Greenwich loam",
        awc_in_per_in=0.16, fc_vol=0.27, wp_vol=0.11, ksat_in_hr=1.0,
        rew_in=0.32, tew_in=0.80, ze_in=4.0,
    ),
}

# Hydrologic soil group drives the runoff curve number.
HSG: dict[str, str] = {
    "evesboro": "A", "rosedale": "A", "downer": "A", "hammonton": "B",
    "sassafras": "B", "ingleside": "B", "woodstown": "C", "matapeake": "B",
    "mattapex": "C", "fallsington": "C", "pocomoke": "D", "greenwich": "B",
}

# Row crop, good condition, straight row. Cover and residue shift these.
CURVE_NUMBERS: dict[str, float] = {"A": 67.0, "B": 78.0, "C": 85.0, "D": 89.0}


def get_soil(key: str) -> SoilProfile:
    try:
        return SOILS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown soil '{key}'. Available: {', '.join(sorted(SOILS))}"
        ) from exc


def curve_number_for(key: str, residue_cover_pct: float = 0.0) -> float:
    """Curve number adjusted downward for surface residue and cover crops."""
    base = CURVE_NUMBERS[HSG.get(key, "B")]
    return round(max(55.0, base - 0.08 * max(0.0, min(100.0, residue_cover_pct))), 1)


def soil_catalog() -> list[dict]:
    out = []
    for key, s in SOILS.items():
        d = asdict(s)
        d["key"] = key
        d["hsg"] = HSG.get(key, "B")
        d["awc_30in_total"] = round(s.awc_in_per_in * 30, 2)
        d["droughty"] = s.awc_in_per_in <= 0.10
        d["shallow_water_table"] = s.water_table_depth_in is not None
        out.append(d)
    return sorted(out, key=lambda d: d["series"])


def blend_profiles(components: list[tuple[str, float]]) -> SoilProfile:
    """Area-weighted blend of map unit components for a mixed management zone.

    components is a list of (soil_key, area_fraction). Fractions are normalized.
    The blended profile keeps the shallowest water table present in the mix,
    because that is the component that will surprise the model.
    """
    if not components:
        raise ValueError("blend_profiles requires at least one component")
    total = sum(f for _, f in components) or 1.0
    awc = fc = wp = ksat = rew = tew = 0.0
    wt: float | None = None
    names = []
    for key, frac in components:
        s = get_soil(key)
        w = frac / total
        awc += s.awc_in_per_in * w
        fc += s.fc_vol * w
        wp += s.wp_vol * w
        ksat += s.ksat_in_hr * w
        rew += s.rew_in * w
        tew += s.tew_in * w
        if s.water_table_depth_in is not None:
            wt = s.water_table_depth_in if wt is None else min(wt, s.water_table_depth_in)
        names.append(f"{s.series} {round(w * 100)}%")
    return SoilProfile(
        series=" + ".join(names),
        awc_in_per_in=round(awc, 3), fc_vol=round(fc, 3), wp_vol=round(wp, 3),
        ksat_in_hr=round(ksat, 2), rew_in=round(rew, 2), tew_in=round(tew, 2),
        water_table_depth_in=wt,
    )


SDA_QUERY = """
SELECT co.compname, co.comppct_r, ch.hzdept_r, ch.hzdepb_r,
       ch.awc_r, ch.ksat_r, ch.wthirdbar_r, ch.wfifteenbar_r
FROM legend AS l
INNER JOIN mapunit AS mu ON mu.lkey = l.lkey
INNER JOIN component AS co ON co.mukey = mu.mukey
INNER JOIN chorizon AS ch ON ch.cokey = co.cokey
WHERE mu.mukey = '{mukey}' AND co.majcompflag = 'Yes'
ORDER BY co.comppct_r DESC, ch.hzdept_r ASC
"""


async def lookup_ssurgo(lat: float, lon: float, client) -> dict | None:
    """Fetch the dominant SSURGO component at a point from USDA Soil Data Access.

    `client` is an httpx.AsyncClient supplied by the caller so connection pooling
    and timeouts stay under the API layer's control. Returns None on any failure;
    callers fall back to the grower-selected series, which is the behaviour we
    want in a barn with one bar of signal.
    """
    point_sql = (
        "SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84"
        f"('point({lon} {lat})')"
    )
    try:
        r = await client.post(
            "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
            json={"query": point_sql, "format": "JSON"},
            timeout=8.0,
        )
        r.raise_for_status()
        mukey = r.json()["Table"][0][0]
        r2 = await client.post(
            "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest",
            json={"query": SDA_QUERY.format(mukey=mukey), "format": "JSON"},
            timeout=12.0,
        )
        r2.raise_for_status()
        rows = r2.json()["Table"]
    except Exception:
        return None

    # Depth-weight the top 40 inches (about 100 cm) of the dominant component.
    dominant = rows[0][0]
    awc_sum = depth_sum = 0.0
    for name, _pct, top_cm, bot_cm, awc, ksat, fc, wp in rows:
        if name != dominant or awc is None:
            continue
        top_in = float(top_cm) / 2.54
        bot_in = min(40.0, float(bot_cm) / 2.54)
        if bot_in <= top_in:
            continue
        thickness = bot_in - top_in
        awc_sum += float(awc) * thickness
        depth_sum += thickness
    if depth_sum == 0:
        return None
    return {
        "mukey": mukey,
        "component": dominant,
        "awc_in_per_in": round(awc_sum / depth_sum, 3),
        "profile_depth_in": round(depth_sum, 1),
    }
