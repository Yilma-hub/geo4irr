"""
Sentinel imagery for the field the grower just drew.

A note on which satellite does what, because the two get conflated constantly:

  Sentinel-2 is optical — thirteen bands from blue to shortwave infrared at 10
  to 20 m, revisit about five days with two satellites. Everything with a
  vegetation index in its name comes from here: NDVI, NDRE, SAVI, GNDVI, NDMI.
  It cannot see through cloud, which on Delmarva in July is a real constraint,
  not a footnote.

  Sentinel-1 is radar. It has no red, no near-infrared and no colour, so it
  cannot produce NDVI or a true-colour image. What it does is see through cloud
  and at night, and its backscatter responds to surface roughness and to water
  in the top few centimetres of soil and canopy. That makes it the thing that
  keeps working through a week of overcast when the optical record goes blank.

Both are free. Copernicus Data Space Ecosystem gives an account, an OAuth
client, and a monthly allowance of processing units that is generous for
field-scale work — a hundred-odd growers pulling a weekly index over a few
pivots each will not come close to it.

The reason this module exists is not to draw pretty maps. The water balance in
fao56.py carries a multiplicative bias on the tabular crop coefficient, and the
filter in bayesian.py will correct that bias from a canopy observation. Until
now nothing produced one. A field-mean NDVI from Sentinel-2 is exactly that
observation: a field that closed canopy two weeks early gets credited for the
water it is actually using, instead of being scheduled off a textbook curve.
That is the loop this closes.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime, timedelta
from typing import Any

# One transfer function, defined once. The NDVI-to-Kcb relation lives in the
# water balance module because that is what consumes it; importing it here
# rather than restating it means the map and the schedule can never disagree.
from .fao56 import kcb_from_ndvi

# Copernicus Data Space Ecosystem. These paths were current as of mid-2026;
# CDSE has been migrating from /api/v1/<service> to /<service>/v1 and both
# forms have worked through the transition. If a call starts returning 404,
# check the endpoint before assuming the credentials are wrong.
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
             "/protocol/openid-connect/token")
BASE_URL = "https://sh.dataspace.copernicus.eu"
PROCESS_URL = f"{BASE_URL}/process/v1"
STATS_URL = f"{BASE_URL}/statistics/v1"
CATALOG_URL = f"{BASE_URL}/catalog/v1/search"

S2_COLLECTION = "sentinel-2-l2a"
S1_COLLECTION = "sentinel-1-grd"


# ---------------------------------------------------------------------------
# Evalscripts
# ---------------------------------------------------------------------------
# Two variants per index. The *visual* one returns an RGBA PNG with a colour
# ramp for the map. The *stats* one returns FLOAT32 plus a dataMask so the
# Statistical API can give a field mean that is honest about which pixels were
# usable.
#
# Cloud masking is not optional here. Scene Classification classes 3, 8, 9 and
# 10 are cloud shadow, medium- and high-probability cloud, and thin cirrus. A
# field mean computed over a cloud edge is worse than no number at all, because
# it looks like a number.

_CLOUD_MASK = """
function usable(scl) {
  return !(scl == 3 || scl == 8 || scl == 9 || scl == 10 || scl == 11 || scl == 0);
}
"""

_RAMP_VEG = """
const RAMP = [
  [0.00, 0x9E8560], [0.15, 0xC7AC86], [0.30, 0xD9CE7A],
  [0.45, 0xA8C060], [0.60, 0x5E9440], [0.75, 0x3E6B3A], [0.90, 0x1E4423]
];
"""

_RAMP_WATER = """
const RAMP = [
  [-0.4, 0xA32C1A], [-0.2, 0xC98A3A], [0.0, 0xE8D9A8],
  [0.2, 0x8FB8C4], [0.4, 0x4E90AE], [0.6, 0x1F5C7A]
];
"""

_COLOR_FN = """
function hexToRgb(h){ return [((h>>16)&255)/255, ((h>>8)&255)/255, (h&255)/255]; }
function ramp(v){
  if (v <= RAMP[0][0]) return hexToRgb(RAMP[0][1]);
  for (let i = 1; i < RAMP.length; i++) {
    if (v <= RAMP[i][0]) {
      const t = (v - RAMP[i-1][0]) / (RAMP[i][0] - RAMP[i-1][0]);
      const a = hexToRgb(RAMP[i-1][1]), b = hexToRgb(RAMP[i][1]);
      return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
    }
  }
  return hexToRgb(RAMP[RAMP.length-1][1]);
}
"""


def _visual_index(expr: str, bands: list[str], ramp: str) -> str:
    band_list = ", ".join(f'"{b}"' for b in bands + ["SCL", "dataMask"])
    return f"""//VERSION=3
{ramp}
{_COLOR_FN}
{_CLOUD_MASK}
function setup() {{
  return {{
    input: [{{ bands: [{band_list}] }}],
    output: {{ bands: 4, sampleType: "AUTO" }}
  }};
}}
function evaluatePixel(s) {{
  if (s.dataMask == 0) return [0, 0, 0, 0];
  if (!usable(s.SCL)) return [0.62, 0.65, 0.63, 0.55];   // flat grey: cloud, not crop
  const v = {expr};
  const c = ramp(v);
  return [c[0], c[1], c[2], 1];
}}"""


def _stats_index(expr: str, bands: list[str]) -> str:
    band_list = ", ".join(f'"{b}"' for b in bands + ["SCL", "dataMask"])
    return f"""//VERSION=3
{_CLOUD_MASK}
function setup() {{
  return {{
    input: [{{ bands: [{band_list}] }}],
    output: [
      {{ id: "index", bands: 1, sampleType: "FLOAT32" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}
function evaluatePixel(s) {{
  const ok = (s.dataMask == 1 && usable(s.SCL)) ? 1 : 0;
  return {{ index: [{expr}], dataMask: [ok] }};
}}"""


TRUE_COLOR = """//VERSION=3
function setup() {
  return { input: [{ bands: ["B04","B03","B02","dataMask"] }],
           output: { bands: 4, sampleType: "AUTO" } };
}
function evaluatePixel(s) {
  const g = 2.5;
  return [g*s.B04, g*s.B03, g*s.B02, s.dataMask];
}"""

FALSE_COLOR = """//VERSION=3
function setup() {
  return { input: [{ bands: ["B08","B04","B03","dataMask"] }],
           output: { bands: 4, sampleType: "AUTO" } };
}
function evaluatePixel(s) {
  const g = 2.5;
  return [g*s.B08, g*s.B04, g*s.B03, s.dataMask];
}"""

# Sentinel-1 VV backscatter, dB stretched for display. Wetter and rougher
# surfaces scatter more energy back to the sensor.
S1_VV = """//VERSION=3
function setup() {
  return { input: [{ bands: ["VV","dataMask"] }],
           output: { bands: 4, sampleType: "AUTO" } };
}
function evaluatePixel(s) {
  if (s.dataMask == 0) return [0,0,0,0];
  const db = 10 * Math.log(s.VV) / Math.LN10;
  const t = Math.max(0, Math.min(1, (db + 22) / 17));
  return [0.12 + 0.55*t, 0.36 + 0.35*t, 0.48 + 0.30*t, 1];
}"""

S1_VV_STATS = """//VERSION=3
function setup() {
  return { input: [{ bands: ["VV","dataMask"] }],
           output: [{ id: "index", bands: 1, sampleType: "FLOAT32" },
                    { id: "dataMask", bands: 1 }] };
}
function evaluatePixel(s) {
  const db = 10 * Math.log(Math.max(s.VV, 1e-8)) / Math.LN10;
  return { index: [db], dataMask: [s.dataMask] };
}"""


@dataclass(frozen=True)
class Index:
    key: str
    label: str
    collection: str
    visual: str
    stats: str | None
    vmin: float
    vmax: float
    unit: str = ""
    blurb: str = ""
    caution: str = ""


INDICES: dict[str, Index] = {
    "true_color": Index(
        "true_color", "True colour", S2_COLLECTION, TRUE_COLOR, None, 0, 1,
        blurb="What the field looks like from above. Best for spotting the "
              "obvious: a blown nozzle, a drowned corner, a replant strip.",
    ),
    "false_color": Index(
        "false_color", "Infrared", S2_COLLECTION, FALSE_COLOR, None, 0, 1,
        blurb="Near-infrared as red. Healthy vegetation glows; bare and stressed "
              "ground goes grey-green. Useful before a crop closes canopy.",
    ),
    "ndvi": Index(
        "ndvi", "NDVI — vigour", S2_COLLECTION,
        _visual_index("(s.B08 - s.B04) / (s.B08 + s.B04)", ["B08", "B04"], _RAMP_VEG),
        _stats_index("(s.B08 - s.B04) / (s.B08 + s.B04)", ["B08", "B04"]),
        0.0, 0.95,
        blurb="The standard vigour index, and the one that feeds this app's water "
              "balance. Roughly tracks how much green canopy is intercepting light.",
        caution="Saturates once corn closes canopy — from about V10 through grain "
                "fill, NDVI reads 0.85 whether the crop is comfortable or hurting. "
                "After canopy closure, use NDRE.",
    ),
    "ndre": Index(
        "ndre", "NDRE — late-season vigour", S2_COLLECTION,
        _visual_index("(s.B08 - s.B05) / (s.B08 + s.B05)", ["B08", "B05"], _RAMP_VEG),
        _stats_index("(s.B08 - s.B05) / (s.B08 + s.B05)", ["B08", "B05"]),
        0.0, 0.6,
        blurb="Red-edge instead of red. Keeps separating good from poor after NDVI "
              "has flattened out, which makes it the better index for corn from "
              "tassel on and the one to use for nitrogen questions.",
        caution="20 m native resolution, so it is coarser than NDVI. Small "
                "features inside a field will be smeared.",
    ),
    "savi": Index(
        "savi", "SAVI — vigour on open ground", S2_COLLECTION,
        _visual_index("1.5 * (s.B08 - s.B04) / (s.B08 + s.B04 + 0.5)",
                      ["B08", "B04"], _RAMP_VEG),
        _stats_index("1.5 * (s.B08 - s.B04) / (s.B08 + s.B04 + 0.5)", ["B08", "B04"]),
        0.0, 0.9,
        blurb="NDVI with a soil adjustment. On Delaware's light-coloured sands a "
              "bright background inflates NDVI early in the season; SAVI backs "
              "that out, so it is the better read from planting to about V6.",
    ),
    "gndvi": Index(
        "gndvi", "GNDVI — chlorophyll", S2_COLLECTION,
        _visual_index("(s.B08 - s.B03) / (s.B08 + s.B03)", ["B08", "B03"], _RAMP_VEG),
        _stats_index("(s.B08 - s.B03) / (s.B08 + s.B03)", ["B08", "B03"]),
        0.0, 0.8,
        blurb="Green instead of red, which makes it more sensitive to chlorophyll "
              "and therefore to nitrogen status than plain NDVI.",
    ),
    "ndmi": Index(
        "ndmi", "NDMI — canopy water", S2_COLLECTION,
        _visual_index("(s.B08 - s.B11) / (s.B08 + s.B11)", ["B08", "B11"], _RAMP_WATER),
        _stats_index("(s.B08 - s.B11) / (s.B08 + s.B11)", ["B08", "B11"]),
        -0.3, 0.6,
        blurb="Shortwave infrared against near-infrared. The closest thing "
              "Sentinel-2 offers to a water stress signal: it responds to water "
              "held in the leaves themselves.",
        caution="Canopy water content, not soil water. It falls as a crop dries "
                "down naturally at the end of the season, and it lags real stress "
                "by a few days. Treat a drop mid-season as a reason to check a "
                "probe, not as a measurement of depletion.",
    ),
    "msi": Index(
        "msi", "MSI — moisture stress", S2_COLLECTION,
        # MSI runs the other way from every other index here: low is wet, high
        # is dry, and it is not bounded to [-1, 1]. Rescale onto the shared
        # water ramp so that red always means dry on every map in the app.
        _visual_index(
            "0.6 - Math.max(0, Math.min(1, ((s.B11 / s.B08) - 0.2) / 1.4))",
            ["B08", "B11"], _RAMP_WATER),
        _stats_index("s.B11 / s.B08", ["B08", "B11"]),
        0.2, 2.0,
        blurb="The inverse view of the same physics as NDMI. Higher means drier "
              "canopy. Shown negated on the map so that dry always reads red.",
        caution="Same limits as NDMI: this is leaf water, not root zone water.",
    ),
    "s1_vv": Index(
        "s1_vv", "Radar (Sentinel-1)", S1_COLLECTION, S1_VV, S1_VV_STATS,
        -22.0, -5.0, unit=" dB",
        blurb="Radar backscatter, which sees through cloud and at night. Rises "
              "with wetter soil and with rougher surfaces. When a week of "
              "overcast blanks the optical record — normal here in July — this "
              "is what keeps reporting.",
        caution="Not a vegetation index and not calibrated soil moisture. Read "
                "the change between passes, not the absolute number, and expect "
                "canopy structure to confound it once corn is tall.",
    ),
}


def index_catalog() -> list[dict]:
    return [
        {"key": i.key, "label": i.label, "collection": i.collection,
         "vmin": i.vmin, "vmax": i.vmax, "unit": i.unit,
         "blurb": i.blurb, "caution": i.caution,
         "has_stats": i.stats is not None}
        for i in INDICES.values()
    ]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class Credentials:
    client_id: str = ""
    client_secret: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


def credentials_from_env() -> Credentials:
    return Credentials(
        client_id=os.environ.get("CDSE_CLIENT_ID", ""),
        client_secret=os.environ.get("CDSE_CLIENT_SECRET", ""),
    )


class SentinelHubError(RuntimeError):
    pass


@dataclass
class SentinelHub:
    """Thin client over the Copernicus Data Space Sentinel Hub APIs.

    Credentials live on the server, never in the browser. A client secret handed
    to a web app is a client secret published.
    """
    creds: Credentials = dc_field(default_factory=credentials_from_env)
    _token: str = ""
    _expires: float = 0.0

    @property
    def configured(self) -> bool:
        return self.creds.configured

    async def token(self, client) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        if not self.configured:
            raise SentinelHubError(
                "No Copernicus credentials configured. Register free at "
                "dataspace.copernicus.eu, create an OAuth client under User "
                "Settings, and set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET."
            )
        r = await client.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials",
                  "client_id": self.creds.client_id,
                  "client_secret": self.creds.client_secret},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20.0,
        )
        if r.status_code != 200:
            raise SentinelHubError(
                f"Copernicus rejected the credentials ({r.status_code}). Check the "
                "OAuth client id and secret in the dashboard."
            )
        j = r.json()
        self._token = j["access_token"]
        self._expires = time.time() + float(j.get("expires_in", 600))
        return self._token

    async def _headers(self, client) -> dict:
        return {"Authorization": f"Bearer {await self.token(client)}",
                "Content-Type": "application/json"}

    # -- catalog ----------------------------------------------------------

    async def scenes(self, client, bbox: tuple[float, float, float, float],
                     start: date, end: date, collection: str = S2_COLLECTION,
                     max_cloud: float = 80.0, limit: int = 60) -> list[dict]:
        """Which passes exist over this field, and how cloudy each one was."""
        body: dict[str, Any] = {
            "collections": [collection],
            "bbox": list(bbox),
            "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
            "limit": limit,
            "fields": {"include": ["id", "properties.datetime",
                                   "properties.eo:cloud_cover"], "exclude": []},
        }
        if collection == S2_COLLECTION:
            body["filter"] = {
                "op": "<=",
                "args": [{"property": "eo:cloud_cover"}, max_cloud],
            }
            body["filter-lang"] = "cql2-json"

        r = await client.post(CATALOG_URL, json=body,
                              headers=await self._headers(client), timeout=30.0)
        if r.status_code != 200:
            raise SentinelHubError(f"Scene search failed ({r.status_code}).")

        out = []
        for f in r.json().get("features", []):
            props = f.get("properties", {})
            dt = props.get("datetime", "")
            cloud = props.get("eo:cloud_cover")
            out.append({
                "id": f.get("id"),
                "datetime": dt,
                "date": dt[:10],
                "cloud_pct": round(cloud, 1) if cloud is not None else None,
                "usable": cloud is None or cloud <= 35,
            })
        # Newest first, and collapse same-day duplicates from adjacent orbits.
        out.sort(key=lambda s: s["datetime"], reverse=True)
        seen, uniq = set(), []
        for s in out:
            if s["date"] in seen:
                continue
            seen.add(s["date"])
            uniq.append(s)
        return uniq

    # -- imagery ----------------------------------------------------------

    async def image(self, client, geometry: dict,
                    bbox: tuple[float, float, float, float],
                    day: date, index_key: str, width: int = 640,
                    window_days: int = 2) -> dict:
        """A single index image, clipped to the field outline.

        The polygon goes in as `bounds.geometry` so pixels outside the field come
        back transparent — a grower looking at their own ground should not have
        to work out where the neighbour's field starts.
        """
        idx = INDICES.get(index_key)
        if idx is None:
            raise SentinelHubError(f"Unknown index '{index_key}'.")

        w, s, e, n = bbox
        aspect = max(0.2, min(5.0, (n - s) / max(1e-9, (e - w) * 0.78)))
        height = max(64, min(1400, int(width * aspect)))

        frm = datetime.combine(day - timedelta(days=window_days),
                               datetime.min.time())
        to = datetime.combine(day + timedelta(days=window_days),
                              datetime.max.time())
        body = {
            "input": {
                "bounds": {"bbox": list(bbox), "geometry": geometry,
                           "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                "data": [{
                    "type": idx.collection,
                    "dataFilter": {
                        "timeRange": {"from": frm.isoformat() + "Z",
                                      "to": to.isoformat() + "Z"},
                        "mosaickingOrder": "leastCC",
                    },
                }],
            },
            "output": {"width": width, "height": height,
                       "responses": [{"identifier": "default",
                                      "format": {"type": "image/png"}}]},
            "evalscript": idx.visual,
        }
        r = await client.post(PROCESS_URL, json=body,
                              headers=await self._headers(client), timeout=60.0)
        if r.status_code != 200:
            raise SentinelHubError(
                f"Imagery request failed ({r.status_code}). "
                + (r.text[:200] if r.text else "")
            )
        return {
            "index": idx.key,
            "label": idx.label,
            "date": day.isoformat(),
            "bbox": list(bbox),
            "width": width,
            "height": height,
            "png_base64": base64.b64encode(r.content).decode("ascii"),
        }

    # -- statistics -------------------------------------------------------

    async def series(self, client, geometry: dict, start: date, end: date,
                     index_key: str = "ndvi", interval: str = "P5D",
                     max_cloud: float = 40.0) -> list[dict]:
        """Field-mean index through the season, cloud-masked.

        Returns only intervals where enough of the field was actually visible.
        A mean over 8% of the pixels is not a field mean.
        """
        idx = INDICES.get(index_key)
        if idx is None or idx.stats is None:
            raise SentinelHubError(f"No statistics available for '{index_key}'.")

        data: dict[str, Any] = {"type": idx.collection}
        if idx.collection == S2_COLLECTION:
            data["dataFilter"] = {"maxCloudCoverage": max_cloud}

        body = {
            "input": {"bounds": {"geometry": geometry,
                                 "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                      "data": [data]},
            "aggregation": {
                "timeRange": {"from": f"{start.isoformat()}T00:00:00Z",
                              "to": f"{end.isoformat()}T23:59:59Z"},
                "aggregationInterval": {"of": interval},
                "evalscript": idx.stats,
                "resx": 10 if idx.collection == S2_COLLECTION else 20,
                "resy": 10 if idx.collection == S2_COLLECTION else 20,
            },
        }
        r = await client.post(STATS_URL, json=body,
                              headers=await self._headers(client), timeout=90.0)
        if r.status_code != 200:
            raise SentinelHubError(f"Statistics request failed ({r.status_code}).")

        out = []
        for item in r.json().get("data", []):
            interval_from = item.get("interval", {}).get("from", "")[:10]
            band = (item.get("outputs", {}).get("index", {})
                        .get("bands", {}).get("B0", {}))
            stats = band.get("stats", {})
            if not stats or stats.get("sampleCount") in (None, 0):
                continue
            valid = stats.get("sampleCount", 0) - stats.get("noDataCount", 0)
            coverage = valid / max(1, stats.get("sampleCount", 1))
            if coverage < 0.55:          # mostly cloud; not a field mean
                continue
            out.append({
                "date": interval_from,
                "mean": round(stats["mean"], 4),
                "min": round(stats.get("min", stats["mean"]), 4),
                "max": round(stats.get("max", stats["mean"]), 4),
                "stdev": round(stats.get("stDev", 0.0), 4),
                "coverage": round(coverage, 3),
            })
        out.sort(key=lambda d: d["date"])
        return out


# ---------------------------------------------------------------------------
# Turning pixels into something the water balance can use
# ---------------------------------------------------------------------------

def canopy_observation(ndvi_mean: float, ndvi_stdev: float, day: date,
                       model_kcb: float) -> dict:
    """Package a scene into an observation the Bayesian filter can assimilate.

    The uncertainty carried with it is not a constant. Spatial variability inside
    the field, atmospheric correction residue, and the saturation problem at high
    NDVI all widen it, and the filter needs to know that so a saturated
    mid-season scene does not overrule a well-calibrated model.
    """
    observed_kcb = kcb_from_ndvi(ndvi_mean)
    sigma = 0.06 + 0.55 * ndvi_stdev
    if ndvi_mean > 0.78:                      # in the saturated part of the curve
        sigma += 0.10
    ratio = observed_kcb / max(0.05, model_kcb)
    return {
        "date": day.isoformat(),
        "ndvi": round(ndvi_mean, 3),
        "observed_kcb": observed_kcb,
        "model_kcb": round(model_kcb, 3),
        "kcb_ratio": round(ratio, 3),
        "sigma_kcb": round(sigma, 3),
        "saturated": ndvi_mean > 0.78,
        "verdict": _canopy_verdict(ratio, ndvi_mean),
    }


def _canopy_verdict(ratio: float, ndvi: float) -> str:
    if ndvi > 0.78:
        return ("Canopy is closed and NDVI has saturated, so this scene mostly "
                "confirms the crop is where the calendar says. NDRE will tell "
                "you more from here on.")
    if ratio >= 1.12:
        return ("The canopy is ahead of the calendar. The crop is using more "
                "water than the textbook curve assumes, and the schedule has "
                "been adjusted up to match.")
    if ratio <= 0.88:
        return ("The canopy is behind the calendar. Either the crop is slow or "
                "something is holding it back — worth a look before the next "
                "irrigation, because the schedule has been eased back to match "
                "what is actually out there.")
    return "The canopy matches what the water balance assumed. No adjustment made."


def uniformity(series_point: dict) -> dict:
    """How even the field is, from the spread within one scene.

    A high standard deviation inside one pivot is the argument for variable rate
    that a soil map alone cannot make: it is the crop reporting the difference,
    not a survey line predicting it.
    """
    sd, mean = series_point.get("stdev", 0.0), max(1e-6, series_point.get("mean", 0))
    cv = sd / mean
    if cv < 0.08:
        verdict = "Even. One rate is serving this field."
    elif cv < 0.16:
        verdict = ("Some variation. Worth walking the weak areas before deciding "
                   "whether it is soil, water, or nitrogen.")
    else:
        verdict = ("Strongly variable. If the pattern holds across scenes and "
                   "matches the soil map, this field is a variable-rate candidate.")
    return {"cv": round(cv, 3), "verdict": verdict}


def season_window(planting: date, today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    start = planting - timedelta(days=10)
    return start, min(today, planting + timedelta(days=200))
