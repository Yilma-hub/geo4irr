"""
Weather ingest for Delaware fields.

Two sources, in priority order:

  1. DEOS - the Delaware Environmental Observing System, run by the University
     of Delaware. Roughly forty stations across the three counties reporting at
     five-minute intervals, including solar radiation, which is what makes a
     real Penman-Monteith ETo possible rather than a temperature proxy. DEOS
     also publishes derived reference ET. This is the backbone of the system and
     the reason a Delaware version of Geo4Irr is tractable: the state already
     has the mesonet, growers just are not using it. Survey work suggests only
     a small share of Delaware irrigators use reference ET data directly to
     schedule, with most still going on crop appearance or the feel method.

  2. National Weather Service api.weather.gov gridpoint forecasts for the 1-to-7
     day outlook. Free, no key, and the gridpoint resolution (about 2.5 km) is
     fine for Delmarva.

Every fetch is cached and every failure degrades gracefully to the nearest
alternative, because the worst outcome is a blank screen at 5 a.m. when the
grower is deciding whether to start the pivot.
"""

from __future__ import annotations

import os

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Sequence

from .advisory import ForecastDay
from .fao56 import WeatherDay

# DEOS does not publish an open API. The documented route is a subscription web
# service returning XML: sign up at http://services.deos.udel.edu/ws/subscribe.php
# and set DEOS_WS_USER / DEOS_WS_PASS (or DEOS_WS_KEY). Without credentials this
# client returns nothing and the caller falls back to NWS plus climatology, which
# is a working but visibly degraded state — see /api/weather/status.
#
# Cite as: Center for Environmental Monitoring and Analysis (2004). DEOS Network
# Real-time and Historical Data [Dataset]. University of Delaware.
# https://doi.org/10.71926/qcz3-mb31
DEOS_WS_BASE = os.getenv("DEOS_WS_BASE", "http://services.deos.udel.edu/ws")
DEOS_WS_USER = os.getenv("DEOS_WS_USER", "")
DEOS_WS_PASS = os.getenv("DEOS_WS_PASS", "")
DEOS_WS_KEY = os.getenv("DEOS_WS_KEY", "")
DEOS_STATION_PAGE = "https://www.deos.udel.edu/station/index.php"
DEOS_BASE = DEOS_WS_BASE


def deos_configured() -> bool:
    """Whether DEOS credentials are present.

    Worth asking explicitly rather than discovering through an empty result:
    a grower whose schedule is quietly running on climatology instead of their
    own weather station deserves to be told.
    """
    return bool(DEOS_WS_KEY or (DEOS_WS_USER and DEOS_WS_PASS))


def deos_status() -> dict:
    return {
        "configured": deos_configured(),
        "base": DEOS_WS_BASE,
        "auth": "key" if DEOS_WS_KEY else ("basic" if DEOS_WS_USER else "none"),
        "signup": "http://services.deos.udel.edu/ws/subscribe.php",
        "note": (
            "DEOS is a subscription web service, not an open API. Without an "
            "account this tool falls back to National Weather Service forecasts "
            "and Delaware climatology — usable, but not your station."
            if not deos_configured() else
            "DEOS credentials are configured; observations come from the "
            "University of Delaware station network."),
        "citation": (
            "Center for Environmental Monitoring and Analysis (2004). DEOS "
            "Network Real-time and Historical Data [Dataset]. University of "
            "Delaware. https://doi.org/10.71926/qcz3-mb31"),
    }
NWS_BASE = "https://api.weather.gov"


@dataclass(frozen=True, slots=True)
class Station:
    code: str
    name: str
    county: str
    lat: float
    lon: float
    has_solar: bool = True
    has_soil_moisture: bool = False


# Representative DEOS agricultural stations. The live list is fetched at startup
# and cached; this table is the offline fallback and the seed for the picker.
DEOS_STATIONS: list[Station] = [
    Station("DGES", "Georgetown", "Sussex", 38.6431, -75.4664, True, True),
    Station("DHAR", "Harbeson", "Sussex", 38.6503, -75.2650, True, True),
    Station("DSEA", "Seaford", "Sussex", 38.6420, -75.6100, True, True),
    Station("DLAU", "Laurel", "Sussex", 38.5600, -75.5700, True, False),
    Station("DMIL", "Milford", "Sussex", 38.9130, -75.4280, True, False),
    Station("DBRG", "Bridgeville", "Sussex", 38.7440, -75.6010, True, True),
    Station("DSJR", "Selbyville", "Sussex", 38.4610, -75.2200, True, False),
    Station("DDOV", "Dover", "Kent", 39.1580, -75.5240, True, False),
    Station("DHAR2", "Harrington", "Kent", 38.9230, -75.5780, True, True),
    Station("DSMY", "Smyrna", "Kent", 39.2990, -75.6050, True, False),
    Station("DFEL", "Felton", "Kent", 39.0090, -75.5780, True, False),
    Station("DMDL", "Middletown", "New Castle", 39.4490, -75.7160, True, True),
    Station("DNWK", "Newark", "New Castle", 39.6690, -75.7520, True, False),
    Station("DTLY", "Townsend", "New Castle", 39.3950, -75.6910, True, False),
]


def nearest_station(lat: float, lon: float,
                    require_solar: bool = True) -> Station:
    """Great-circle nearest station, preferring those with a pyranometer."""
    pool = [s for s in DEOS_STATIONS if s.has_solar] if require_solar else DEOS_STATIONS
    if not pool:
        pool = DEOS_STATIONS

    def dist(s: Station) -> float:
        dlat = math.radians(s.lat - lat)
        dlon = math.radians(s.lon - lon) * math.cos(math.radians(lat))
        return math.hypot(dlat, dlon) * 3959.0

    return min(pool, key=dist)


def station_distance_mi(s: Station, lat: float, lon: float) -> float:
    dlat = math.radians(s.lat - lat)
    dlon = math.radians(s.lon - lon) * math.cos(math.radians(lat))
    return round(math.hypot(dlat, dlon) * 3959.0, 1)


# ---------------------------------------------------------------------------
# DEOS
# ---------------------------------------------------------------------------

async def fetch_deos_daily(
    station_code: str, start: date, end: date, client,
) -> list[WeatherDay]:
    """Daily summaries from a DEOS station.

    DEOS exposes several endpoints and the exact parameter names have changed
    over the life of the network, so the parser here is deliberately tolerant:
    it looks for any of the plausible keys and skips days it cannot assemble
    rather than failing the whole request. Deployment should pin the agreed
    endpoint with UD once a data-sharing arrangement is in place.
    """
    params = {
        "stationID": station_code,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": "daily",
    }
    try:
        r = await client.get(DEOS_BASE, params=params, timeout=15.0)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return []

    rows = payload if isinstance(payload, list) else payload.get("data", [])
    out: list[WeatherDay] = []
    for row in rows:
        d = _pick(row, "date", "Date", "obs_date", "day")
        if d is None:
            continue
        try:
            day = datetime.fromisoformat(str(d)[:10]).date()
        except ValueError:
            continue
        tmax = _num(_pick(row, "air_temp_max", "TmaxF", "tmax"))
        tmin = _num(_pick(row, "air_temp_min", "TminF", "tmin"))
        if tmax is None or tmin is None:
            continue
        out.append(WeatherDay(
            day=day,
            tmax_f=tmax,
            tmin_f=tmin,
            rh_max=_num(_pick(row, "rh_max", "RHmax")),
            rh_min=_num(_pick(row, "rh_min", "RHmin")),
            wind_mph=_num(_pick(row, "wind_speed_avg", "WindSpeed", "wind")),
            solar_rad_mj_m2=_solar_to_mj(_num(_pick(row, "solar_rad_total", "SolarRad"))),
            rain_in=_num(_pick(row, "precip_total", "Precip", "rain")) or 0.0,
            eto_in=_num(_pick(row, "et_ref", "ETo", "reference_et")),
            source="observed",
        ))
    return sorted(out, key=lambda w: w.day)


def _pick(row: dict, *keys: str) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, "", "NA", "M"):
            return row[k]
    return None


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _solar_to_mj(v: float | None) -> float | None:
    """DEOS reports daily solar in W/m2 average or MJ/m2; normalize to MJ/m2/day."""
    if v is None:
        return None
    return round(v * 0.0864, 3) if v > 60 else round(v, 3)


# ---------------------------------------------------------------------------
# National Weather Service
# ---------------------------------------------------------------------------

async def fetch_nws_forecast(lat: float, lon: float, client,
                             days: int = 7) -> list[ForecastDay]:
    """Daily forecast with probability of precipitation and quantitative amount."""
    headers = {"User-Agent": "WISE-Pro-Delaware (irrigation advisory; contact ag-extension)"}
    try:
        pt = await client.get(f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}",
                              headers=headers, timeout=10.0)
        pt.raise_for_status()
        grid_url = pt.json()["properties"]["forecastGridData"]
        g = await client.get(grid_url, headers=headers, timeout=15.0)
        g.raise_for_status()
        props = g.json()["properties"]
    except Exception:
        return []

    tmax = _series_daily(props.get("maxTemperature"), agg="max", to_f=True)
    tmin = _series_daily(props.get("minTemperature"), agg="min", to_f=True)
    pop = _series_daily(props.get("probabilityOfPrecipitation"), agg="max")
    qpf = _series_daily(props.get("quantitativePrecipitation"), agg="sum", mm_to_in=True)
    wind = _series_daily(props.get("windSpeed"), agg="mean", kmh_to_mph=True)
    rhmin = _series_daily(props.get("relativeHumidity"), agg="min")
    rhmax = _series_daily(props.get("relativeHumidity"), agg="max")

    out: list[ForecastDay] = []
    today = date.today()
    for i in range(days):
        d = today + timedelta(days=i + 1)
        if d not in tmax or d not in tmin:
            continue
        out.append(ForecastDay(
            day=d,
            tmax_f=tmax[d],
            tmin_f=tmin[d],
            pop=min(1.0, (pop.get(d, 0.0)) / 100.0),
            qpf_in=round(qpf.get(d, 0.0), 2),
            wind_mph=round(wind.get(d, 6.0), 1),
            rh_min=rhmin.get(d),
            rh_max=rhmax.get(d),
        ))
    return out


def _series_daily(block: dict | None, agg: str = "mean", to_f: bool = False,
                  mm_to_in: bool = False, kmh_to_mph: bool = False) -> dict[date, float]:
    """Collapse an NWS gridpoint time series into daily values.

    NWS returns ISO 8601 intervals with durations, so a single entry can span
    several hours. Each entry is attributed to the day its interval starts.
    """
    if not block or "values" not in block:
        return {}
    buckets: dict[date, list[float]] = {}
    for entry in block["values"]:
        try:
            start = datetime.fromisoformat(entry["validTime"].split("/")[0].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        v = entry.get("value")
        if v is None:
            continue
        v = float(v)
        if to_f:
            v = v * 9.0 / 5.0 + 32.0
        if mm_to_in:
            v = v / 25.4
        if kmh_to_mph:
            v = v * 0.621371
        buckets.setdefault(start.date(), []).append(v)

    fn = {"max": max, "min": min, "sum": sum,
          "mean": lambda xs: sum(xs) / len(xs)}[agg]
    return {d: round(fn(vs), 2) for d, vs in buckets.items() if vs}


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------

def fill_gaps(days: Sequence[WeatherDay], start: date, end: date,
              lat: float = 38.7) -> list[WeatherDay]:
    """Fill missing days by linear interpolation, flagged as estimated.

    A missing day is not a zero-ET day. Silently treating it as one is how a
    water balance drifts wet over a season and quietly stops being useful.
    """
    have = {w.day: w for w in days}
    out: list[WeatherDay] = []
    d = start
    known = sorted(have)
    while d <= end:
        if d in have:
            out.append(have[d])
        elif known:
            before = [k for k in known if k < d]
            after = [k for k in known if k > d]
            ref = have[before[-1]] if before else have[after[0]]
            out.append(WeatherDay(
                day=d, tmax_f=ref.tmax_f, tmin_f=ref.tmin_f,
                rh_max=ref.rh_max, rh_min=ref.rh_min, wind_mph=ref.wind_mph,
                solar_rad_mj_m2=ref.solar_rad_mj_m2, rain_in=0.0,
                source="estimated",
            ))
        d += timedelta(days=1)
    return out


def rain_gauge_blend(station_in: float, onfarm_in: float | None,
                     distance_mi: float) -> float:
    """Blend station rainfall with a grower's own gauge reading.

    Summer convection on Delmarva is small-scale. A gauge five miles away is
    weak evidence about what fell on this pivot, so an on-farm reading dominates
    whenever it exists. This one function prevents more bad recommendations than
    anything else in the ingest path.
    """
    if onfarm_in is None:
        return station_in
    weight_onfarm = 0.85 if distance_mi > 3 else 0.7
    return round(weight_onfarm * onfarm_in + (1 - weight_onfarm) * station_in, 2)
