"""
Landsat Collection 2 Level-2, for the METRIC energy balance.

Where the pixels come from. USGS publishes Landsat 8 and 9 Collection 2
Level-2 as cloud-optimized GeoTIFFs on AWS; Element 84 index them at
earth-search; and a TiTiler instance will crop a scene to a polygon and hand
back a numpy array. None of those three needs an account, which matters for a
tool meant to be cloned off GitHub and run.

Collection 2 Level-2 is the right product rather than Level-1 because it
already carries atmospherically corrected surface reflectance and, in the
`lwir11` asset, a surface temperature product with the emissivity and
atmospheric terms already applied. Doing that correction here would mean
reimplementing a USGS pipeline badly.

Revisit is the real constraint. Landsat 8 and 9 together cross Delaware every
eight days or so, and on the Delmarva summer sky perhaps half of those are
usable. A season might yield six to ten scenes. That is enough for a seasonal
total if the gaps are filled sensibly (see `metric.accumulate_season`) and not
enough to schedule irrigation from, which is why this sits alongside the daily
FAO-56 balance rather than replacing it.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime, timedelta

from .metric import Pixel, scale_reflectance, scale_surface_temp_k

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
LANDSAT_COLLECTION = "landsat-c2-l2"
TITILER = os.getenv("TITILER_URL", "https://titiler.xyz")

# Earth Search asset names for Landsat C2 L2. Written out rather than inferred
# because the names differ from the USGS band numbers people quote: there is no
# `ST_B10` here, the surface temperature asset is `lwir11`.
BANDS = {
    "blue": "blue", "green": "green", "red": "red",
    "nir": "nir08", "swir1": "swir16", "swir2": "swir22",
    "lst": "lwir11", "emis": "emis", "qa": "qa_pixel",
}

# Bit flags in qa_pixel, Landsat C2 product guide Table 6-3.
QA_FILL = 1 << 0
QA_DILATED_CLOUD = 1 << 1
QA_CIRRUS = 1 << 2
QA_CLOUD = 1 << 3
QA_CLOUD_SHADOW = 1 << 4
QA_SNOW = 1 << 5
QA_WATER = 1 << 7
QA_BAD = QA_FILL | QA_DILATED_CLOUD | QA_CIRRUS | QA_CLOUD | QA_CLOUD_SHADOW | QA_SNOW


class LandsatError(RuntimeError):
    """Raised with text meant to be read by the person who asked for the map."""


@dataclass(frozen=True, slots=True)
class Scene:
    id: str
    day: date
    cloud_pct: float
    platform: str
    href: str
    datetime_utc: str

    @property
    def overpass_hour_local(self) -> float:
        """Local solar-ish hour of the overpass, Eastern time.

        Landsat crosses Delaware mid-morning. The exact value matters because
        it sets the solar incidence angle, and that drives incoming shortwave,
        which drives everything downstream.
        """
        try:
            stamp = datetime.fromisoformat(self.datetime_utc.replace("Z", "+00:00"))
        except ValueError:
            return 10.6
        # Delaware is UTC-4 through the growing season.
        local = stamp - timedelta(hours=4)
        return local.hour + local.minute / 60.0


def _post_json(url: str, body: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/geo+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise LandsatError(
            f"The scene index answered {exc.code}. It is a free service and "
            "does go down; try again in a few minutes.") from exc
    except Exception as exc:
        raise LandsatError(
            f"Could not reach the Landsat scene index: {exc}") from exc


def search_scenes(bbox: tuple[float, float, float, float],
                  start: date, end: date,
                  max_cloud: float = 60.0,
                  limit: int = 60) -> list[Scene]:
    """Landsat scenes over a field between two dates.

    Cloud cover is the scene-wide figure, so a 60 percent scene may still be
    clear over one field and a 20 percent scene may have its only cloud
    directly overhead. The threshold is deliberately loose and the per-pixel QA
    mask does the real work later.
    """
    body = {
        "collections": [LANDSAT_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lt": max_cloud},
            # Level-2 surface reflectance only, and only the two satellites
            # with a working thermal sensor. Landsat 7 is in the collection and
            # its scan line corrector failed in 2003, leaving wedge-shaped gaps.
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    out = _post_json(STAC_SEARCH, body)
    scenes: list[Scene] = []
    for ft in out.get("features", []):
        props = ft.get("properties", {})
        stamp = props.get("datetime", "")
        if not stamp:
            continue
        scenes.append(Scene(
            id=ft["id"],
            day=date.fromisoformat(stamp[:10]),
            cloud_pct=float(props.get("eo:cloud_cover") or 0.0),
            platform=props.get("platform", "landsat"),
            datetime_utc=stamp,
            href=(f"https://earth-search.aws.element84.com/v1/collections/"
                  f"{LANDSAT_COLLECTION}/items/{ft['id']}")))
    scenes.sort(key=lambda s: s.day)
    return scenes


def _feature_array(scene_href: str, asset: str, geojson: dict,
                   size: int = 64, timeout: float = 90.0):
    """Crop one asset to a polygon and return it as a numpy array.

    TiTiler does the reprojection and the windowed read, which is what avoids
    pulling rasterio and a PROJ database into this project for one job. `size`
    caps the longest edge: 64 is roughly 30 m pixels over a 130 acre pivot,
    which is the native Landsat resolution, and asking for more only
    interpolates.
    """
    import numpy as np

    params = [
        f"url={urllib.parse.quote(scene_href, safe='')}",
        f"assets={asset}",
        f"max_size={size}",
    ]
    url = f"{TITILER}/stac/feature.npy?" + "&".join(params)
    req = urllib.request.Request(
        url, data=json.dumps(geojson).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()[:200]
        except Exception:
            pass
        raise LandsatError(
            f"The tile service answered {exc.code} for the {asset} band"
            + (f": {detail}" if detail else "")) from exc
    except Exception as exc:
        raise LandsatError(f"Could not read the {asset} band: {exc}") from exc

    arr = np.load(io.BytesIO(raw))
    # TiTiler returns data and mask stacked. The last plane is the mask.
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return arr[0], arr[-1]
    return arr, None


def field_pixels(scene: Scene, geojson: dict, size: int = 64,
                 elevation_m: float = 12.0) -> tuple[list[Pixel], dict]:
    """Every clear pixel inside a field boundary, as METRIC inputs.

    Cloud, shadow, cirrus and snow are dropped using the per-pixel QA band
    rather than the scene cloud percentage, because a scene can be 40 percent
    cloudy and perfectly clear over one field. Water pixels are kept but
    flagged: a farm pond inside a boundary is a legitimate part of the field
    and a terrible cold anchor.

    Two shapes come back. The flat `pixels` list is what the energy balance
    iterates over. The `grid` in the metadata holds the same pixels in their
    rows and columns, with None where one was masked, because the CIMEC anchor
    search measures how uniform a candidate's neighbourhood is and cannot do
    that on a flat list. `grid_shape` is set only when the clear pixels form a
    complete rectangle, which is the condition `metric.run_metric` checks
    before reaching for CIMEC.
    """
    import numpy as np

    bands = {}
    for name in ("blue", "green", "red", "nir", "swir1", "swir2", "lst", "emis", "qa"):
        data, _ = _feature_array(scene.href, BANDS[name], geojson, size=size)
        bands[name] = data.astype("float64")

    qa = bands["qa"].astype("int64")
    bad = (qa & QA_BAD) != 0
    water = (qa & QA_WATER) != 0
    # A surface temperature of exactly zero is the fill value, not a reading.
    bad |= bands["lst"] <= 0

    rows, cols = (qa.shape if qa.ndim == 2 else (1, qa.size))
    total = qa.size
    usable = int((~bad).sum())
    pixels: list[Pixel] = []
    grid: list[list[Pixel | None]] = [[None] * cols for _ in range(rows)]
    # Where each clear pixel sits in the raster, so an ET value computed off
    # the flat list can be put back on a map. Without this the browser gets a
    # bag of numbers and can only draw a histogram.
    index_grid: list[list[int | None]] = [[None] * cols for _ in range(rows)]
    flat = {k: v.ravel() for k, v in bands.items()}
    bad_flat, water_flat = bad.ravel(), water.ravel()

    for i in range(total):
        if bad_flat[i]:
            continue
        emis = flat["emis"][i] * 0.0001
        if not (0.8 < emis < 1.0):
            emis = 0.98
        pixel = Pixel(
            blue=scale_reflectance(flat["blue"][i]),
            green=scale_reflectance(flat["green"][i]),
            red=scale_reflectance(flat["red"][i]),
            nir=scale_reflectance(flat["nir"][i]),
            swir1=scale_reflectance(flat["swir1"][i]),
            swir2=scale_reflectance(flat["swir2"][i]),
            lst_k=scale_surface_temp_k(flat["lst"][i]),
            emissivity=emis,
            elevation_m=elevation_m)
        index_grid[i // cols][i % cols] = len(pixels)
        pixels.append(pixel)
        grid[i // cols][i % cols] = pixel

    coverage = usable / total if total else 0.0
    notes = []
    if coverage < 0.6:
        notes.append(
            f"Only {coverage*100:.0f} percent of this field was clear on "
            f"{scene.day.isoformat()}. Cloud or its shadow covered the rest, so "
            "this scene describes part of the field rather than all of it.")
    if int(water_flat.sum()) > total * 0.02:
        notes.append(
            "Some pixels inside this boundary are flagged as water. If that is "
            "a pond or a ditch rather than a flooded crop, it must not be used "
            "as the cold calibration pixel.")

    # CIMEC needs neighbours, so it only gets the grid when nothing was masked
    # out of it. A field with cloud over one corner falls back to the simpler
    # coldest-green against hottest-bare rule, which is honest: the
    # homogeneity test cannot be computed on a grid with holes in it.
    complete = usable == total and rows >= 3 and cols >= 3
    if not complete and usable < total:
        notes.append(
            f"{total - usable} of {total} pixels were masked, so the anchor "
            "search used the simpler coldest-green against hottest-bare rule "
            "rather than the automated CIMEC calibration, which needs a "
            "complete grid to measure neighbourhood uniformity on.")

    return pixels, {"total": total, "usable": usable,
                    "coverage": round(coverage, 3),
                    "water_pixels": int(water_flat.sum()),
                    "rows": rows, "cols": cols,
                    "grid_shape": (rows, cols) if complete else None,
                    "grid": grid,
                    "index_grid": index_grid,
                    "notes": notes}


@dataclass
class SceneResult:
    scene: Scene
    mean_eta_mm_day: float
    mean_etrf: float
    pixel_count: int
    coverage: float
    cold_lst_c: float
    hot_lst_c: float
    notes: list[str] = dc_field(default_factory=list)
    # The calibration behind the scene, and the pixels it produced. Carried
    # through to the browser so that a map can be drawn and so the two anchor
    # choices can be inspected: a METRIC result nobody can audit is a number
    # to be taken on faith, which is not what this is for.
    calibration: dict = dc_field(default_factory=dict)
    cold_reason: str = ""
    hot_reason: str = ""
    pixels: list[dict] = dc_field(default_factory=list)
    raster: dict = dc_field(default_factory=dict)
    bbox: list[float] = dc_field(default_factory=list)


def boundary_to_geojson(boundary) -> dict:
    """A Geo4Irr boundary as a GeoJSON Feature for the crop request."""
    polys = []
    for poly in boundary:
        outer = [[float(x), float(y)] for x, y in poly["outer"]]
        if outer[0] != outer[-1]:
            outer.append(outer[0])
        rings = [outer]
        for hole in poly.get("holes", []):
            h = [[float(x), float(y)] for x, y in hole]
            if h and h[0] != h[-1]:
                h.append(h[0])
            if h:
                rings.append(h)
        polys.append(rings)
    geom = ({"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1
            else {"type": "MultiPolygon", "coordinates": polys})
    return {"type": "Feature", "properties": {}, "geometry": geom}
