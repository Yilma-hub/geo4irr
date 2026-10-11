"""
Geo4Irr - HTTP API.

Serves the grower-facing web app, the nightly batch, and any third-party
integration (equipment telemetry, crop consultant dashboards, Extension
reporting).

Scale note for the 20,000-grower target: the expensive work is not the HTTP
request, it is the nightly recomputation of every field's water balance against
new DEOS observations and a fresh NWS forecast. That runs in `nightly_batch`
below as a fan-out over fields, roughly 15 ms of engine time each, so the whole
state finishes in a few minutes on a handful of workers. Requests during the day
read the cached result. Nothing here needs a GPU or a queue broker at this size.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field as PField, field_validator

from . import auth, notify
from .auth_routes import current_admin as auth_current_admin
from .auth_routes import current_user as auth_current_user
from .auth_routes import router as auth_router
from .engine import advisory as adv
from .engine import landsat as ls
from .engine import metric as met
from .engine import economics as econ
from .engine import geo
from .engine import ingest
from .engine import likelihood as lk
from .engine import optimization as opt
from .engine import pyfao56_bridge as pf
from .engine import swatplus as swat
from .engine import vri
from .engine import satellite as sat
from .engine import nitrogen as nitro
from .engine import sensors as sensor_design
from .engine import weather as wx
from .engine.bayesian import (
    BayesianAssimilator, Observation, SENSOR_CLASSES, depletion_from_vwc,
)
from .engine.crops import CROPS, crop_catalog, get_crop
from .engine.fao56 import SoilProfile, WeatherDay, reference_et, run_season
from .engine.soils import curve_number_for, get_soil, soil_catalog

FRONTEND_DIR = os.environ.get("GEO4IRR_FRONTEND", "../frontend")

http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        headers={"User-Agent": "WISE-Pro-Delaware/1.0"},
        timeout=httpx.Timeout(20.0),
        limits=httpx.Limits(max_connections=60, max_keepalive_connections=20),
    )
    yield
    await http_client.aclose()


app = FastAPI(
    title="Geo4Irr",
    version="1.0.0",
    description="Irrigation and nutrient decision support for Delaware cropland",
    lifespan=lifespan,
)

# Accounts. Mounted unconditionally: /api/auth/status tells the frontend whether
# anyone has signed up yet, and /api/auth/setup closes itself the moment the
# first administrator exists.
auth.init_db()
# Creates the configured administrator the first time this runs, and never
# again. The seeded password is flagged must_change, so the first sign-in
# replaces it before anything else.
_seeded = auth.seed_admin()
app.include_router(auth_router)


def _evening_lines(user: auth.User) -> list:
    """Build one person's evening note.

    Fields live in the browser today, not on the server, so there is nothing
    here to read yet — the round runs, finds nobody with server-side fields, and
    sends nothing. That is the honest state: the schedule, the rendering and the
    delivery are all real and testable, and the piece still missing is field
    storage on the API. Wire this to the fields table and the notes start going
    out with no other change.
    """
    return []


@app.on_event("startup")
async def _start_scheduler() -> None:
    if os.getenv("GEO4IRR_NO_SCHEDULER"):
        return
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        return
    sched = AsyncIOScheduler()
    notify.schedule(sched, _evening_lines)
    sched.start()
    app.state.scheduler = sched


@app.get("/api/notifications/status")
async def notifications_status() -> dict:
    """Whether the evening note is configured, and when it goes out."""
    return notify.status()


@app.post("/api/notifications/test")
async def notifications_test(
        admin: auth.User = Depends(auth_current_admin)) -> dict:
    """Run tonight's round now, so it can be checked without waiting for six."""
    return notify.run_evening_round(_evening_lines)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("GEO4IRR_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request and response models
# ---------------------------------------------------------------------------

class FieldSetup(BaseModel):
    name: str = "Home pivot"
    acres: float = PField(130.0, gt=0, le=5000)
    lat: float = PField(38.70, ge=38.4, le=39.85)
    lon: float = PField(-75.55, ge=-75.80, le=-74.98)
    soil_key: str = "sassafras"
    awc_override: float | None = None
    restrictive_layer_in: float | None = None
    water_table_depth_in: float | None = None
    crop_key: str = "corn_grain"
    planting_date: date
    yield_goal: float | None = None
    residue_cover_pct: float = 0.0

    system_type: str = "center_pivot"
    capacity_gpm: float = 800.0
    application_efficiency: float = 0.85
    min_application_in: float = 0.25
    max_application_in: float = 1.20
    well_permit_gpm: float | None = None
    allocation_ac_in_remaining: float | None = None

    @field_validator("crop_key")
    @classmethod
    def _crop_exists(cls, v: str) -> str:
        if v not in CROPS:
            raise ValueError(f"Unknown crop '{v}'")
        return v

    def soil(self) -> SoilProfile:
        s = get_soil(self.soil_key)
        if self.awc_override:
            s.awc_in_per_in = self.awc_override
        if self.restrictive_layer_in:
            s.restrictive_layer_in = self.restrictive_layer_in
        if self.water_table_depth_in:
            s.water_table_depth_in = self.water_table_depth_in
        return s

    def system(self) -> adv.SystemSpec:
        return adv.SystemSpec(
            system_type=self.system_type,
            capacity_gpm=self.capacity_gpm,
            acres=self.acres,
            application_efficiency=self.application_efficiency,
            min_application_in=self.min_application_in,
            max_application_in=self.max_application_in,
            well_permit_gpm=self.well_permit_gpm,
            allocation_ac_in_remaining=self.allocation_ac_in_remaining,
        )


class SensorReadingIn(BaseModel):
    day: date
    vwc_by_depth: dict[int, float]
    sensor_class: str = "capacitance_pro"
    quality: float = 1.0


class CanopyIn(BaseModel):
    day: date
    kind: str = "canopy_cover"      # canopy_cover | ndvi | canopy_temp
    value: float
    product: str = "uav_multispectral"


class AdvisoryRequest(BaseModel):
    field: FieldSetup
    as_of: date | None = None
    irrigation_log: dict[date, float] = PField(default_factory=dict)
    rain_log: dict[date, float] = PField(default_factory=dict)
    sensor_readings: list[SensorReadingIn] = PField(default_factory=list)
    canopy: list[CanopyIn] = PField(default_factory=list)
    pump: dict[str, Any] | None = None
    use_live_weather: bool = True


# ---------------------------------------------------------------------------
# Reference endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "geo4irr-delaware",
        "date": date.today().isoformat(),
        "et_engine": pf.engine_choice(),
        "pyfao56_available": pf.PYFAO56_AVAILABLE,
    }


@app.get("/api/reference/crops")
async def crops() -> list[dict]:
    return crop_catalog()


@app.get("/api/reference/soils")
async def soils() -> list[dict]:
    return soil_catalog()


@app.get("/api/reference/sensors")
async def sensor_classes() -> list[dict]:
    return [
        {
            "key": s.key, "label": s.label, "sigma_vwc": s.sigma_vwc,
            "cost_usd": s.cost_usd, "depths_in": list(s.depths_in),
            "telemetry": s.telemetry, "salinity_sensitive": s.salinity_sensitive,
        }
        for s in SENSOR_CLASSES.values()
    ]


@app.get("/api/reference/stations")
async def stations(lat: float | None = None, lon: float | None = None) -> list[dict]:
    rows = []
    for s in wx.DEOS_STATIONS:
        row = {
            "code": s.code, "name": s.name, "county": s.county,
            "lat": s.lat, "lon": s.lon, "has_solar": s.has_solar,
            "has_soil_moisture": s.has_soil_moisture,
        }
        if lat is not None and lon is not None:
            row["distance_mi"] = wx.station_distance_mi(s, lat, lon)
        rows.append(row)
    if lat is not None:
        rows.sort(key=lambda r: r.get("distance_mi", 0))
    return rows


# ---------------------------------------------------------------------------
# Field boundaries
# ---------------------------------------------------------------------------

class BoundaryRequest(BaseModel):
    """A drawn or uploaded field outline.

    `geometry` takes GeoJSON in any of the shapes farm software emits — a bare
    geometry, a Feature, or a FeatureCollection. KML and shapefiles are parsed in
    the browser and arrive here as GeoJSON, which keeps file format handling out
    of the server and lets boundary setup work before a grower has an account.
    """
    geometry: dict | list
    samples: int = PField(24, ge=4, le=80)
    resolve_soil: bool = True


@app.post("/api/geo/field")
async def field_from_boundary(req: BoundaryRequest) -> dict:
    try:
        raw = geo.parse_geojson(req.geometry)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    # Acreage and centroid come from the boundary as drawn. Simplification is
    # only for the soil service, which does not need a thousand vertices to say
    # what is under a pivot, and it must never move the number the grower sees
    # against their own field.
    acres = geo.boundary_acres(raw)
    lon, lat = geo.boundary_centroid(raw)
    b = geo.simplify_boundary(raw)
    warnings = list(raw.warnings) + geo.validate(raw)

    station = wx.nearest_station(lat, lon)
    out: dict[str, Any] = {
        "acres": round(acres, 1),
        "centroid": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "bbox": geo.boundary_bbox(raw),
        "vertices": raw.vertex_count,
        "polygons": len(raw.polygons),
        "warnings": warnings,
        "weather": {
            "station_code": station.code,
            "station_name": station.name,
            "county": station.county,
            "distance_mi": wx.station_distance_mi(station, lat, lon),
            "has_solar": station.has_solar,
            "note": (
                "This station measures solar radiation, so reference ET is a full "
                "Penman-Monteith calculation."
                if station.has_solar else
                "No pyranometer at this station, so ET falls back to a temperature "
                "method and carries more uncertainty."
            ),
        },
    }

    if not req.resolve_soil or http_client is None:
        out["soil"] = {"ok": False, "reason": "not_requested", "components": []}
        return out

    soil = await geo.ssurgo_for_boundary(b, http_client, samples=req.samples)
    out["soil"] = soil

    if soil.get("ok") and soil["components"]:
        blended = geo.blended_from_components(soil["components"])
        out["soil"]["blended"] = {
            "series": blended.series,
            "awc_in_per_in": blended.awc_in_per_in,
            "holds_in_30in": round(blended.awc_in_per_in * 30, 2),
            "water_table_depth_in": blended.water_table_depth_in,
            "curve_number": curve_number_for(soil["dominant_key"]),
        }
        out["zones"] = geo.suggest_zones(soil["components"], acres)
        weak = [c for c in soil["components"] if c.get("match") in ("awc", "unresolved")]
        if weak:
            out["warnings"].append(
                f"{len(weak)} of the map units under this field are not in the "
                "Delaware series table and were matched on water-holding capacity "
                "rather than by name. Check those before relying on them."
            )
        out["warnings"].append(
            "SSURGO is mapped at 1:12,000 or coarser and the lines are often off by "
            "a hundred feet or more. You know where the sand blows better than the "
            "survey does — change this if it looks wrong."
        )
    else:
        out["warnings"].append(
            "The USDA soil service did not answer. Pick your series from the list "
            "and carry on; the lookup can be retried later."
        )
    return out


# ---------------------------------------------------------------------------
# Satellite
# ---------------------------------------------------------------------------

sat_hub = sat.SentinelHub()


class SceneRequest(BaseModel):
    geometry: dict | list
    start: date | None = None
    end: date | None = None
    collection: str = sat.S2_COLLECTION
    max_cloud: float = PField(80.0, ge=0, le=100)


class ImageRequest(BaseModel):
    geometry: dict | list
    day: date
    index: str = "ndvi"
    width: int = PField(640, ge=128, le=1400)


class SeriesRequest(BaseModel):
    geometry: dict | list
    start: date
    end: date | None = None
    index: str = "ndvi"
    interval: str = "P5D"
    max_cloud: float = PField(40.0, ge=0, le=100)
    model_kcb: float | None = None


def _geom_and_bbox(raw_geometry) -> tuple[dict, tuple[float, float, float, float]]:
    try:
        b = geo.parse_geojson(raw_geometry)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    # Simplified outline: Sentinel Hub charges by area and pixels, not vertices,
    # but a thousand-vertex polygon in a request body is wasted bandwidth on a
    # phone in a field.
    s = geo.simplify_boundary(b, max_vertices=200)
    coords = [
        [[list(pt) for pt in p.outer + [p.outer[0]]]]
        + [[list(pt) for pt in h + [h[0]]] for h in p.holes]
        for p in s.polygons
    ]
    geometry = ({"type": "Polygon", "coordinates": coords[0]}
                if len(coords) == 1 else
                {"type": "MultiPolygon", "coordinates": coords})
    return geometry, geo.boundary_bbox(s)


@app.get("/api/satellite/status")
async def satellite_status() -> dict:
    return {
        "configured": sat_hub.configured,
        "indices": sat.index_catalog(),
        "setup": (
            None if sat_hub.configured else
            "Register free at dataspace.copernicus.eu, create an OAuth client "
            "under User Settings, then set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET "
            "on the server. Imagery is free; the monthly processing allowance is "
            "far more than field-scale work uses."
        ),
    }


@app.post("/api/satellite/scenes")
async def satellite_scenes(req: SceneRequest) -> dict:
    _, bbox = _geom_and_bbox(req.geometry)
    end = req.end or date.today()
    start = req.start or (end - timedelta(days=60))
    try:
        scenes = await sat_hub.scenes(http_client, bbox, start, end,
                                      collection=req.collection,
                                      max_cloud=req.max_cloud)
    except sat.SentinelHubError as exc:
        raise HTTPException(502, str(exc)) from exc
    clear = [s for s in scenes if s["usable"]]
    return {
        "scenes": scenes,
        "clear_count": len(clear),
        "latest_clear": clear[0]["date"] if clear else None,
        "note": (
            None if clear else
            "No clear pass in this window. That is normal on Delmarva in a humid "
            "spell — the radar layer keeps working through cloud."
        ),
    }


@app.post("/api/satellite/image")
async def satellite_image(req: ImageRequest) -> dict:
    geometry, bbox = _geom_and_bbox(req.geometry)
    try:
        out = await sat_hub.image(http_client, geometry, bbox, req.day,
                                  req.index, width=req.width)
    except sat.SentinelHubError as exc:
        raise HTTPException(502, str(exc)) from exc
    idx = sat.INDICES[req.index]
    out["caution"] = idx.caution
    out["blurb"] = idx.blurb
    return out


@app.post("/api/satellite/series")
async def satellite_series(req: SeriesRequest) -> dict:
    geometry, _ = _geom_and_bbox(req.geometry)
    end = req.end or date.today()
    try:
        series = await sat_hub.series(http_client, geometry, req.start, end,
                                      index_key=req.index, interval=req.interval,
                                      max_cloud=req.max_cloud)
    except sat.SentinelHubError as exc:
        raise HTTPException(502, str(exc)) from exc

    out: dict[str, Any] = {"index": req.index, "series": series}
    if series:
        out["uniformity"] = sat.uniformity(series[-1])
    # The point of the whole module: hand the latest clear scene to the water
    # balance as a canopy observation, so a field that closed early is scheduled
    # on what it is doing rather than on a textbook curve.
    if series and req.index == "ndvi" and req.model_kcb is not None:
        last = series[-1]
        out["canopy_observation"] = sat.canopy_observation(
            last["mean"], last["stdev"],
            date.fromisoformat(last["date"]), req.model_kcb,
        )
    return out


# ---------------------------------------------------------------------------
# ET engine comparison
# ---------------------------------------------------------------------------

class EngineCompareRequest(BaseModel):
    """Replay a season through both water balance engines and diff them.

    This is a validation endpoint, not a scheduling one. pyfao56 (Thorp 2022,
    USDA-ARS) is the reference implementation named in the project narrative and
    is validated against Maricopa lysimeter records; the built-in engine is not
    validated against anything but its own tests. Where they disagree, pyfao56
    is presumed right until shown otherwise.
    """
    field: FieldSetup
    start: date
    end: date
    irrigation_log: dict[date, float] = PField(default_factory=dict)
    detail: bool = False


@app.post("/api/et/compare")
async def compare_et_engines(req: EngineCompareRequest) -> dict:
    if not pf.PYFAO56_AVAILABLE:
        raise HTTPException(
            501,
            "pyfao56 is not installed on this server. pip install pyfao56, or "
            "pip install 'pyfao56 @ git+https://github.com/isukendall/pyfao56.git'",
        )
    f = req.field
    crop = get_crop(f.crop_key)
    soil = f.soil()
    days, _fc, wsource = await _gather_weather(
        f.lat, f.lon, req.start, req.end, {}, True)
    if not days:
        raise HTTPException(502, "No weather available for that window.")

    out = pf.compare_engines(days, crop, soil, f.planting_date,
                             irrigation=req.irrigation_log, lat_deg=f.lat,
                             curve_number=curve_number_for(f.soil_key))
    if not req.detail:
        out.pop("days", None)

    stats = out["stats"]
    notes = []
    if stats["eto"]["max_abs"] > 0.005:
        notes.append(
            "Reference ET disagrees by more than 0.005 in/day. Both claim to be "
            "ASCE standardized, so this is a bug in one of them, not a modelling "
            "choice — chase it before looking at anything else."
        )
    if stats["kcb"]["max_abs"] > 0.02:
        notes.append(
            "Basal crop coefficients differ. Check the stage lengths and the "
            "Kcb triple against FAO-56 Tables 11 and 17."
        )
    if stats["taw"]["max_abs"] > 0.10:
        notes.append(
            "Total available water differs, which means the two engines grow "
            "roots differently. FAO-56 expands the root zone in proportion to "
            "crop development; the built-in engine ties it to canopy cover "
            "instead. That is a deliberate difference and it propagates into "
            "depletion and stress — decide which you want before a grower sees "
            "either number."
        )
    if abs(out["season"]["etc_difference_pct"]) > 5:
        notes.append(
            f"Season crop water use differs by "
            f"{out['season']['etc_difference_pct']:+.1f}%. On a Delaware sand "
            "that is roughly one irrigation over a season."
        )
    out["notes"] = notes or [
        "The two engines agree within tolerance across this season."
    ]
    out["engines"] = {"reference": "pyfao56", "comparison": "builtin"}
    out["weather_source"] = wsource
    if wsource != "observed":
        out["notes"].insert(0, (
            "This comparison ran on climatology rather than observed weather, so "
            "it tests the two engines against each other, not against Delaware."))
    return out


# ---------------------------------------------------------------------------
# Activity 9 — tradeoffs and NSGA-II
# ---------------------------------------------------------------------------

class TradeoffRequest(BaseModel):
    field: FieldSetup
    start: date
    end: date | None = None
    regime_key: str = "full"
    system_key: str = "sprinkler_pivot"
    crop_price: float = 4.60
    n_price: float = 0.62
    water_cost_per_ac_in: float = 9.50
    allocation_ac_in: float | None = None
    population: int = PField(40, ge=12, le=120)
    generations: int = PField(20, ge=4, le=60)
    current_practice: dict | None = None


@app.post("/api/optimize/tradeoffs")
async def tradeoffs(req: TradeoffRequest) -> dict:
    """Pareto-optimal irrigation and nitrogen strategies.

    Activity 9. The response leads with the corners and the knee rather than the
    whole front, because a forty-point Pareto set is a research output and a
    grower needs the handful of points where the exchange rate actually changes.
    """
    f = req.field
    end = req.end or date.today()
    observed, _fc, wsource = await _gather_weather(
        f.lat, f.lon, req.start, end, {}, True)
    if not observed:
        raise HTTPException(502, "No weather available for that window.")

    ctx = opt.FieldContext(
        weather=observed, soil=f.soil(), crop_key=f.crop_key,
        planting=f.planting_date, lat_deg=f.lat, acres=f.acres,
        capacity_gpm=f.capacity_gpm,
        yield_goal_bu=f.yield_goal or 200.0,
        crop_price=req.crop_price, n_price=req.n_price,
        water_cost_per_ac_in=req.water_cost_per_ac_in,
        water_table_depth_in=f.water_table_depth_in,
    )
    space = opt.DecisionSpace(allocation_ac_in=req.allocation_ac_in)
    result = opt.optimize(ctx, space, regime_key=req.regime_key,
                          system_key=req.system_key,
                          population=req.population, generations=req.generations)

    out: dict[str, Any] = opt.front_summary(result)
    out["weather_source"] = wsource
    out["regime"] = swat.get_regime(req.regime_key).label
    out["system"] = swat.get_system(req.system_key).label

    if req.current_practice:
        try:
            current = opt.evaluate(
                opt.Strategy(
                    trigger_fraction=float(req.current_practice.get("trigger_fraction", 0.5)),
                    depth_in=float(req.current_practice.get("depth_in", 0.8)),
                    n_rate_lb=float(req.current_practice.get("n_rate_lb", 180)),
                    splits=int(req.current_practice.get("splits", 2)),
                    regime_key=req.regime_key, system_key=req.system_key),
                ctx, space)
            out["your_practice"] = opt.compare_to_front(current, result.front)
        except (TypeError, ValueError) as exc:
            out["your_practice"] = {"error": str(exc)}

    if wsource != "observed":
        out["caution"] = (
            "This ran on climatology, not observed weather. The shape of the "
            "tradeoff is informative; the absolute numbers are not.")
    return out


# ---------------------------------------------------------------------------
# Activity 2 — computational grids and variable rate prescriptions
# ---------------------------------------------------------------------------

class GridRequest(BaseModel):
    geometry: dict | list
    grid_spec: str = "vri_individual"
    soil_components: list[dict] | None = None
    pivot_lon: float | None = None
    pivot_lat: float | None = None


@app.post("/api/vri/grid")
async def vri_grid(req: GridRequest) -> dict:
    try:
        boundary = geo.parse_geojson(req.geometry)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    pivot = ((req.pivot_lon, req.pivot_lat)
             if req.pivot_lon is not None and req.pivot_lat is not None else None)
    try:
        grid = vri.discretize(boundary, req.grid_spec, pivot_point=pivot)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    if req.soil_components:
        vri.attach_soils(grid, req.soil_components, get_soil)
    vri.build_response_units(grid)

    by_soil: dict[str, float] = {}
    for c in grid.cells:
        by_soil[c.soil_key] = by_soil.get(c.soil_key, 0.0) + c.area_ac

    return {
        "grid_spec": grid.spec.key,
        "grid_label": grid.spec.label,
        "cell_size_ft": [grid.spec.along_span_ft, grid.spec.along_travel_ft],
        "n_cells": grid.n_cells,
        "response_units": grid.response_units,
        "acres": round(grid.acres, 1),
        "acres_by_soil": {k: round(v, 1) for k, v in by_soil.items()},
        "warnings": grid.warnings,
        "note": grid.spec.note,
    }


class PrescriptionRequest(BaseModel):
    geometry: dict | list
    field: FieldSetup
    as_of: date | None = None
    grid_spec: str = "vri_individual"
    soil_components: list[dict] | None = None
    include_cells: bool = False


@app.post("/api/vri/prescription")
async def vri_prescription(req: PrescriptionRequest) -> dict:
    """A per-cell rate map, computed from the same balance as the daily advisory."""
    try:
        boundary = geo.parse_geojson(req.geometry)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    f = req.field
    as_of = req.as_of or date.today()
    crop = get_crop(f.crop_key)
    grid = vri.discretize(boundary, req.grid_spec)
    if req.soil_components:
        vri.attach_soils(grid, req.soil_components, get_soil)
    vri.build_response_units(grid)

    observed, _fc, wsource = await _gather_weather(
        f.lat, f.lon, f.planting_date, as_of, {}, True)

    # One water balance per response unit, not per cell: cells in the same unit
    # are given identical inputs and would return identical answers.
    depletion, taw = {}, {}
    seen: dict[int, Any] = {}
    for c in grid.cells:
        seen.setdefault(c.response_unit, c)
    for unit, cell in seen.items():
        profile = get_soil(cell.soil_key)
        results = run_season(observed, crop, profile, f.planting_date, f.lat,
                             curve_number=curve_number_for(cell.soil_key))
        last = results[-1] if results else None
        depletion[unit] = last.depletion_in if last else 0.0
        taw[unit] = last.taw_in if last else profile.awc_in_per_in * 30

    dap = (as_of - f.planting_date).days
    rx = vri.build_prescription(
        grid, as_of, depletion, taw, mad=crop.mad(dap),
        min_rate_in=f.min_application_in, max_rate_in=f.max_application_in,
        application_efficiency=f.application_efficiency)

    payback = vri.vri_payback(grid, rx.savings_ac_in * 6, f.acres)
    out = {
        "day": rx.day.isoformat(),
        "grid_spec": rx.grid_spec,
        "n_cells": grid.n_cells,
        "response_units": grid.response_units,
        "distinct_rates": rx.n_rates,
        "rate_range_in": [rx.min_rate_in, rx.max_rate_in],
        "applied_ac_in": rx.total_ac_in,
        "uniform_ac_in": rx.uniform_ac_in,
        "savings_ac_in": rx.savings_ac_in,
        "worth_uploading": rx.worth_it,
        "rationale": rx.rationale,
        "payback": payback,
        "weather_source": wsource,
    }
    if req.include_cells:
        out["cells"] = rx.cells
        out["features"] = vri.prescription_to_shapefile_wkt(rx)
    return out


# ---------------------------------------------------------------------------
# Activity 6 — SWAT+ fluxes and soil health
# ---------------------------------------------------------------------------

class FluxRequest(BaseModel):
    field: FieldSetup
    start: date
    end: date | None = None
    system_key: str = "sprinkler_pivot"
    irrigation_log: dict[date, float] = PField(default_factory=dict)
    irrigation_nitrate_ppm: float = 0.0
    irrigation_ec_ds_m: float = 0.0


@app.post("/api/swat/fluxes")
async def swat_fluxes(req: FluxRequest) -> dict:
    """The Eq 1 terms FAO-56 cannot supply: U, qs and qw."""
    f = req.field
    end = req.end or date.today()
    crop = get_crop(f.crop_key)
    soil = f.soil()
    observed, _fc, wsource = await _gather_weather(
        f.lat, f.lon, req.start, end, {}, True)
    if not observed:
        raise HTTPException(502, "No weather available for that window.")

    results = run_season(observed, crop, soil, f.planting_date, f.lat,
                         irrigations=req.irrigation_log,
                         curve_number=curve_number_for(f.soil_key))
    model = swat.FieldProcessModel(
        curve_number=curve_number_for(f.soil_key, f.residue_cover_pct),
        ksat_in_hr=soil.ksat_in_hr,
        aquifer=swat.AquiferState(
            water_table_depth_in=f.water_table_depth_in or soil.water_table_depth_in),
        irrigation_nitrate_ppm=req.irrigation_nitrate_ppm,
        irrigation_ec_ds_m=req.irrigation_ec_ds_m)
    system = swat.get_system(req.system_key)
    for r in results:
        model.step(r.day, rain_in=r.rain_in,
                   irrigation_in=req.irrigation_log.get(r.day, 0.0),
                   depletion_in=r.depletion_in, taw_in=r.taw_in,
                   root_depth_in=r.root_depth_in, system=system, acres=f.acres)

    out: dict[str, Any] = {
        "season_totals": model.season_totals(),
        "weather_source": wsource,
        "swatplus_executable_available": swat.swat_available(),
        "engine": ("swatplus" if swat.swat_available()
                   else "field_process_model"),
    }
    if req.irrigation_ec_ds_m > 0:
        thresh, slope = swat.SALT_TOLERANCE.get(f.crop_key, (2.0, 10.0))
        out["salinity"] = swat.salinity_yield_penalty(
            req.irrigation_ec_ds_m, thresh, slope)
    if not swat.swat_available():
        out["note"] = (
            "SWAT+ is not installed here, so these fluxes come from the "
            "field-scale process model that follows its algorithms. Set "
            "SWATPLUS_EXE to run the full model for multi-season work.")
    return out


class SoilHealthRequest(BaseModel):
    years: int = PField(5, ge=1, le=30)
    regime_key: str = "full"
    residue_ton_per_ac: float = 4.5
    cover_crop: bool = False
    tillage: str = "no_till"
    baseline_om_pct: float = 1.4


@app.post("/api/swat/soil-health")
async def soil_health(req: SoilHealthRequest) -> dict:
    try:
        return swat.simulate_soil_health(
            req.years, req.regime_key, req.residue_ton_per_ac,
            req.cover_crop, req.tillage, req.baseline_om_pct)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class MdsRequest(BaseModel):
    measurements: dict[str, list[float]]
    target: list[float]


@app.post("/api/swat/minimum-data-set")
async def mds(req: MdsRequest) -> dict:
    try:
        return swat.minimum_data_set(req.measurements, req.target)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------------------
# Activity 3 — data ingest
# ---------------------------------------------------------------------------

class SensorUpload(BaseModel):
    text: str
    station: str = ""
    depth_units: str = "auto"


@app.post("/api/ingest/sensors")
async def ingest_sensors(req: SensorUpload) -> dict:
    try:
        readings = ingest.read_sensor_ascii(req.text, req.station, req.depth_units)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    depths = sorted({r.depth_in for r in readings})
    return {
        "n_readings": len(readings),
        "depths_in": depths,
        "first": readings[0].observed_at.isoformat() if readings else None,
        "last": readings[-1].observed_at.isoformat() if readings else None,
        "readings": [
            {"observed_at": r.observed_at.isoformat(), "depth_in": r.depth_in,
             "vwc": r.vwc, "soil_temp_f": r.soil_temp_f, "bulk_ec": r.bulk_ec}
            for r in readings[:5000]
        ],
    }


class WeatherUpload(BaseModel):
    text: str


@app.post("/api/ingest/weather")
async def ingest_weather(req: WeatherUpload) -> dict:
    return ingest.read_weather_csv(req.text)


class YieldUpload(BaseModel):
    features: list[dict]
    value_key: str = "yield"


@app.post("/api/ingest/yield")
async def ingest_yield(req: YieldUpload) -> dict:
    out = ingest.yield_from_features(req.features, req.value_key)
    out.pop("points", None)
    return out


MAX_SHAPEFILE_BYTES = 50 * 1024 * 1024


@app.post("/api/ingest/shapefile")
async def ingest_shapefile(file: UploadFile = File(...),
                           value_key: str = Query("yield")) -> dict:
    """Activity 3: a zipped ESRI shapefile, as the monitor or GIS wrote it.

    The browser parses .shp geometry client-side so boundary drawing works
    offline, but the attribute table is where yield lives, and any non-browser
    client of this API — a consultant's script, another research group's tool —
    needs a server-side path. Both the raw features and the yield statistics
    come back, so the caller can use it as a boundary or as a yield map.
    """
    data = await file.read()
    if len(data) > MAX_SHAPEFILE_BYTES:
        raise HTTPException(413, "That archive is larger than 50 MB. Yield "
                                 "exports this big are usually a whole season "
                                 "of fields; upload one field at a time.")
    try:
        parsed = ingest.read_shapefile_zip(data)
    except ingest.ShapefileError as exc:
        raise HTTPException(400, str(exc)) from exc

    stats = ingest.yield_from_features(parsed["features"], value_key)
    stats.pop("points", None)
    return {
        "n_features": parsed["n_features"],
        "fields": parsed["fields"],
        "epsg": parsed["epsg"],
        "warnings": parsed["warnings"],
        "geometry_types": sorted({f["geometry"]["type"]
                                  for f in parsed["features"]}),
        "features": parsed["features"][:2000],
        "yield": stats,
    }


@app.get("/api/weather/status")
async def weather_status() -> dict:
    """Whether this deployment can actually reach DEOS.

    DEOS is a subscription service, so an unconfigured install silently falls
    back to forecasts and climatology. That is a legitimate mode but the user
    has to know they are in it.
    """
    return wx.deos_status()


class MetricSeasonRequest(BaseModel):
    """Satellite consumptive water use for one field, planting to a date."""
    geometry: dict = PField(..., description="GeoJSON Polygon or MultiPolygon")
    planting: date
    through: date | None = None
    harvest: date | None = None
    lat: float = 38.64
    lon: float = -75.45
    elevation_m: float = 12.0
    max_cloud: float = 60.0
    max_scenes: int = PField(default=12, ge=1, le=30)
    pixel_size: int = PField(default=48, ge=16, le=128)
    # Readily evaporable water for the surface layer, used to work out how wet
    # bare soil was at each overpass. 8 mm is FAO-56 Table 19 for a sandy loam,
    # which covers most of Sussex County; a Sassafras loam runs nearer 9.
    rew_mm: float = PField(default=8.0, gt=0, le=14)
    etrf_bare: float | None = PField(
        default=None,
        description="Override the bare-soil ET fraction at the hot anchor "
                    "(Allen 2013 Eq 5). Left unset it is worked out from the "
                    "rain record.")


def _eta_raster(cover: dict, scene_out) -> dict:
    """The scene's actual ET put back where it came from, as a raster.

    The energy balance works on a flat list of clear pixels, which is all it
    needs. A map needs them back in rows and columns, with holes where cloud
    was masked, and it needs the bounding box the crop was taken from so the
    browser can lay the cells over the field. Without this the only honest
    picture is a histogram, and a histogram does not show which end of the
    pivot ran short.
    """
    index_grid = cover.get("index_grid")
    if not index_grid:
        return {}
    values = [p.eta_mm_day for p in scene_out.pixels]
    etrf = [p.etrf for p in scene_out.pixels]
    rows = []
    rows_etrf = []
    for row in index_grid:
        rows.append([None if i is None or i >= len(values) else values[i]
                     for i in row])
        rows_etrf.append([None if i is None or i >= len(etrf) else etrf[i]
                          for i in row])
    present = [v for row in rows for v in row if v is not None]
    return {
        "rows": len(rows), "cols": len(rows[0]) if rows else 0,
        "eta_mm_day": rows, "etrf": rows_etrf,
        "min_mm_day": round(min(present), 2) if present else None,
        "max_mm_day": round(max(present), 2) if present else None,
        "cold_cell": _cell_of(index_grid, scene_out.cold.index),
        "hot_cell": _cell_of(index_grid, scene_out.hot.index),
    }


def _cell_of(index_grid, flat_index: int):
    """Row and column of one pixel, so the anchors can be marked on the map.

    The two anchor choices carry the whole scene, and a reader who can see
    where they landed can tell a pond or a farm road from a crop. One that
    cannot is taking the number on faith.
    """
    for r, row in enumerate(index_grid):
        for c, i in enumerate(row):
            if i == flat_index:
                return [r, c]
    return None


@app.post("/api/metric/season")
async def metric_season(req: MetricSeasonRequest,
                        user: auth.User = Depends(auth_current_user)) -> dict:
    """Actual ET from Landsat, scene by scene, accumulated across the season.

    This is a different measurement from the irrigation schedule and it is
    worth being clear which is which. The FAO-56 balance says how much water
    the crop would use given the weather and a coefficient. METRIC says how
    much actually left the field, by reading the surface temperature: a crop
    that has run out of water runs hot, and that heat is energy that did not
    go into evaporating anything.

    Reference: Allen, Tasumi and Trezza (2007). The Delaware application is
    Delaware Geological Survey Report of Investigations No. 84, which used
    Landsat 8 over Sussex County for the 2016 season.
    """
    through = req.through or date.today()
    if through < req.planting:
        raise HTTPException(400, "The end date is before the planting date.")

    feature = {"type": "Feature", "properties": {}, "geometry": req.geometry}
    coords = []

    def walk(c):
        if not c:
            return
        if isinstance(c[0], (int, float)):
            coords.append((c[0], c[1]))
            return
        for sub in c:
            walk(sub)
    walk(req.geometry.get("coordinates"))
    if not coords:
        raise HTTPException(400, "That geometry has no coordinates in it.")
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    try:
        scenes = ls.search_scenes(bbox, req.planting, through,
                                  max_cloud=req.max_cloud)
    except ls.LandsatError as exc:
        raise HTTPException(502, str(exc)) from exc

    if not scenes:
        return {"ok": False,
                "reason": (f"No Landsat 8 or 9 scene under {req.max_cloud:.0f} "
                           "percent cloud crossed this field between planting "
                           "and that date."),
                "scenes": []}

    # Newest first, capped: each scene is nine band reads, so an unbounded
    # season is a very long request.
    chosen = sorted(scenes, key=lambda s: s.day)[-req.max_scenes:]

    # Weather for the whole window, from the same station the schedule uses, so
    # the two methods are comparable rather than merely adjacent.
    observed, _fc, wsource = await _gather_weather(
        req.lat, req.lon, req.planting, through, {}, True)
    by_day = {w.day: w for w in observed}
    daily_ref_mm = {w.day: reference_et(w, req.lat, req.elevation_m) * 25.4
                    for w in observed}

    results = []
    scene_dates: list[date] = []
    scene_eta: list[float] = []
    warnings: list[str] = []

    for sc in chosen:
        wd = by_day.get(sc.day)
        if wd is None:
            warnings.append(f"No weather for {sc.day.isoformat()}, so that "
                            "scene was skipped.")
            continue
        daily_mm = daily_ref_mm.get(sc.day, 0.0)
        if daily_mm <= 0:
            warnings.append(f"Reference ET for {sc.day.isoformat()} was zero, "
                            "so that scene was skipped.")
            continue

        try:
            pixels, cover = ls.field_pixels(sc, feature, size=req.pixel_size,
                                            elevation_m=req.elevation_m)
        except ls.LandsatError as exc:
            warnings.append(f"{sc.day.isoformat()}: {exc}")
            continue

        if len(pixels) < 4:
            warnings.append(
                f"{sc.day.isoformat()}: only {len(pixels)} clear pixels inside "
                "the boundary, which is too few to calibrate on. Cloud.")
            continue

        # The hourly reference at the overpass. A clear-sky mid-morning hour
        # carries roughly a tenth of the day's total; this is an approximation
        # and it is the weakest input in the chain, which is why it is named.
        # When it is left at zero the engine computes it from the ASCE hourly
        # equation instead, which is better but needs incoming shortwave.
        hourly_mm = daily_mm * 0.11
        wx = met.SceneWeather(
            overpass=sc.day, overpass_hour_local=sc.overpass_hour_local,
            air_temp_c=(wd.tmax_f + wd.tmin_f) / 2.0 * 5 / 9 - 17.78,
            wind_speed_mps=max(0.5, wd.wind_mph * 0.44704),
            rh_pct=(wd.rh_max + wd.rh_min) / 2.0,
            ref_hourly_mm=hourly_mm, ref_daily_mm=daily_mm,
            reference="grass")

        # How wet the bare ground was, for the hot anchor (Allen 2013 Eq 5),
        # and how wet the preceding two months were, for the temperature
        # adjustment at that anchor (Eq 8). Both come from the same weather
        # record the schedule runs on.
        window = [w for w in observed if sc.day - timedelta(days=14) <= w.day <= sc.day]
        window.sort(key=lambda w: w.day)
        if req.etrf_bare is not None:
            bare = req.etrf_bare
        else:
            bare = met.bare_soil_etrf(
                [w.rain_in * 25.4 for w in window],
                [daily_ref_mm.get(w.day, 0.0) for w in window],
                rew_mm=req.rew_mm)
        sixty = [w for w in observed
                 if sc.day - timedelta(days=60) <= w.day <= sc.day]
        rain_60 = sum(w.rain_in * 25.4 for w in sixty) or None
        ref_60 = sum(daily_ref_mm.get(w.day, 0.0) for w in sixty) or None

        try:
            scene_out = met.run_metric(
                pixels, wx, lat_deg=req.lat,
                grid_shape=cover.get("grid_shape"),
                etrf_bare=bare,
                rain_60_mm=rain_60, ref_et_60_mm=ref_60)
        except ValueError as exc:
            warnings.append(f"{sc.day.isoformat()}: {exc}")
            continue

        results.append(ls.SceneResult(
            scene=sc, mean_eta_mm_day=round(scene_out.mean_eta_mm_day, 2),
            mean_etrf=round(scene_out.mean_etrf, 3),
            pixel_count=len(pixels), coverage=cover["coverage"],
            cold_lst_c=round(scene_out.cold.lst_k - 273.15, 1),
            hot_lst_c=round(scene_out.hot.lst_k - 273.15, 1),
            notes=cover["notes"] + scene_out.warnings,
            calibration=scene_out.calibration,
            cold_reason=scene_out.cold.reason,
            hot_reason=scene_out.hot.reason,
            pixels=[{"ndvi": p.ndvi, "lst_c": round(p.lst_k - 273.15, 1),
                     "albedo": p.albedo, "rn": p.rn, "g": p.g, "h": p.h,
                     "le": p.le, "etrf": p.etrf,
                     "eta_mm_day": p.eta_mm_day,
                     "eta_in_day": round(p.eta_mm_day / 25.4, 3)}
                    for p in scene_out.pixels],
            raster=_eta_raster(cover, scene_out),
            bbox=list(bbox)))
        scene_dates.append(sc.day)
        scene_eta.append(scene_out.mean_eta_mm_day)

    if not scene_dates:
        return {"ok": False,
                "reason": "No scene produced a usable energy balance.",
                "warnings": warnings,
                "scenes": [{"date": s.day.isoformat(),
                            "cloud_pct": s.cloud_pct} for s in chosen]}

    season = met.accumulate_season(
        scene_dates, scene_eta, req.planting, through,
        daily_reference_mm=daily_ref_mm, harvest=req.harvest)

    return {
        "ok": True,
        "planting": req.planting.isoformat(),
        "through": season.through.isoformat(),
        "harvest": req.harvest.isoformat() if req.harvest else None,
        "weather_source": wsource,
        "total_eta_in": season.total_eta_in,
        "total_eta_mm": season.total_eta_mm,
        "days": season.days,
        "gap_filling": season.method,
        "scene_count": len(results),
        "weekly": season.weekly,
        "scenes": [{
            "date": r.scene.day.isoformat(),
            "platform": r.scene.platform,
            "scene_cloud_pct": r.scene.cloud_pct,
            "field_clear_fraction": r.coverage,
            "clear_pixels": r.pixel_count,
            "mean_eta_mm_day": r.mean_eta_mm_day,
            "mean_eta_in_day": round(r.mean_eta_mm_day / 25.4, 3),
            "mean_etrf": r.mean_etrf,
            "cold_anchor_c": r.cold_lst_c,
            "hot_anchor_c": r.hot_lst_c,
            "cold_anchor_reason": r.cold_reason,
            "hot_anchor_reason": r.hot_reason,
            "calibration": r.calibration,
            "pixels": r.pixels,
            "raster": r.raster,
            "bbox": r.bbox,
            "notes": r.notes,
        } for r in results],
        "warnings": warnings + season.warnings,
        "method": ("METRIC surface energy balance on Landsat 8 and 9 "
                   "Collection 2 Level-2, calibrated internally on a cold and "
                   "a hot anchor pixel within the field. Ported from "
                   "hectornieto/pyMETRIC, which implements Allen, Tasumi and "
                   "Trezza (2007) and the automated CIMEC calibration of "
                   "Allen et al. (2013). Delaware application in DGS Report "
                   "of Investigations No. 84."),
    }


class WnipiRequest(BaseModel):
    """A season's inputs and outputs, farm and baseline."""
    yield_farm: float = PField(..., gt=0, description="bu/ac this season")
    yield_baseline: float = PField(..., gt=0, description="bu/ac, baseline mgmt")
    irrigation_in: float = PField(..., ge=0, description="seasonal applied, in")
    eto_in: float = PField(..., gt=0, description="seasonal reference ET, in")
    n_applied: float = PField(..., ge=0, description="lb N/ac, all sources")
    n_uptake_baseline: float = PField(..., gt=0, description="lb N/ac above ground")
    p_applied: float = PField(..., ge=0, description="lb P2O5/ac")
    p_uptake_baseline: float = PField(..., gt=0, description="lb P/ac above ground")


@app.post("/api/metrics/wnipi")
async def wnipi(req: WnipiRequest) -> dict:
    """Eq 6: Water-Nutrient Intensification Performance Index.

    Activity 8 names this as the example metric for communicating tradeoffs, and
    Activity 11 has the CSU TAPS teams calculating it directly for their own
    plots. That is a different shape of request from the optimizer, which only
    yields a WNIPI as a by-product of searching a Pareto front — a team that
    already farmed the season wants the index for what they actually did.

    The index rewards yield gained per unit of water and nutrient
    intensification, so two teams reaching the same yield do not score the same
    if one bought it with twice the water and nitrogen.
    """
    score = nitro.water_nutrient_index(
        yield_farm=req.yield_farm, yield_baseline=req.yield_baseline,
        irrigation_in=req.irrigation_in, eto_in=req.eto_in,
        n_applied=req.n_applied, n_uptake_baseline=req.n_uptake_baseline,
        p_applied=req.p_applied, p_uptake_baseline=req.p_uptake_baseline)

    yield_gain_pct = round((req.yield_farm / req.yield_baseline - 1) * 100, 1)
    water_factor = round(1 + req.irrigation_in / req.eto_in, 3)
    n_factor = round(1 + req.n_applied / req.n_uptake_baseline, 3)
    p_factor = round(1 + req.p_applied / req.p_uptake_baseline, 3)

    # WNIPI has no natural 0-to-1 scale. The denominator is a product of three
    # terms each at least 1, and in realistic corn management it lands between
    # about 3.5 and 10, so even a strong season scores in the hundredths. Bands
    # below are set from that observed range rather than from round numbers,
    # which would put every real result in the same bucket.
    if score > 0.03:
        reading = ("Strong. Yield gained clearly outran the extra water and "
                   "nutrient applied to get it.")
    elif score > 0.015:
        reading = ("Solid. Yield gained more than paid for the input "
                   "intensity.")
    elif score > 0:
        reading = ("Marginal. The yield gain and the input intensity nearly "
                   "cancel, which is within the noise of a single season.")
    elif score == 0:
        reading = "Yield matched the baseline, so there is no gain to divide."
    else:
        reading = ("Negative. Yield came in under baseline, so the "
                   "intensification cost more than it returned this season.")

    return {
        "wnipi": score,
        "yield_gain_pct": yield_gain_pct,
        "denominator": round(water_factor * n_factor * p_factor, 3),
        "factors": {"water": water_factor, "nitrogen": n_factor,
                    "phosphorus": p_factor},
        "dominant_penalty": max(
            (("water", water_factor), ("nitrogen", n_factor),
             ("phosphorus", p_factor)), key=lambda kv: kv[1])[0],
        "reading": reading,
        "note": ("WNIPI is a ranking metric, not an absolute score — it has no "
                 "natural units and typically lands between 0.005 and 0.07 for "
                 "corn. It compares management, not fields, so scores are only "
                 "comparable when they share a baseline. Set the baseline once "
                 "for the trial and keep it fixed across teams."),
    }


class ErrorCharacterization(BaseModel):
    reference: list[float]
    device: list[float]


@app.post("/api/sensors/characterize-error")
async def characterize(req: ErrorCharacterization) -> dict:
    """Activity 4: turn paired field data into the sigma that Eq 5 needs."""
    try:
        return lk.characterize_error(req.reference, req.device)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class DepthRequest(BaseModel):
    profile_observations: list[dict[str, float]]
    n_sensors: int = PField(3, ge=1, le=6)
    root_depth_in: float = 36.0
    restrictive_layer_in: float | None = None


@app.post("/api/sensors/depths")
async def sensor_depths(req: DepthRequest) -> dict:
    """Activity 7: placement depth by universal multiple linear regression."""
    profiles = [{int(float(k)): v for k, v in obs.items()}
                for obs in req.profile_observations]
    try:
        return sensor_design.optimal_sensor_depths(
            profiles, n_sensors=req.n_sensors, root_depth_in=req.root_depth_in,
            restrictive_layer_in=req.restrictive_layer_in)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/reference/systems")
async def reference_systems() -> list[dict]:
    return [{"key": s.key, "label": s.label, "swat_code": s.swat_code,
             "application_efficiency": s.application_efficiency,
             "min_application_in": s.min_application_in,
             "max_application_in": s.max_application_in,
             "variable_rate_capable": s.variable_rate_capable,
             "note": s.note}
            for s in swat.SYSTEMS.values()]


@app.get("/api/reference/regimes")
async def reference_regimes() -> list[dict]:
    return [{"key": r.key, "label": r.label, "et_fraction": r.et_fraction,
             "variable_rate": r.variable_rate, "note": r.note}
            for r in swat.REGIMES.values()]


@app.get("/api/reference/grids")
async def reference_grids() -> list[dict]:
    return [{"key": g.key, "label": g.label,
             "along_span_ft": g.along_span_ft,
             "along_travel_ft": g.along_travel_ft, "note": g.note}
            for g in vri.GRID_SPECS.values()]


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

@app.get("/api/weather/forecast")
async def forecast(lat: float, lon: float, days: int = Query(7, ge=1, le=10)) -> list[dict]:
    fc = await wx.fetch_nws_forecast(lat, lon, http_client, days=days)
    if not fc:
        raise HTTPException(502, "Forecast unavailable from the National Weather "
                                 "Service right now. The advisory will fall back "
                                 "to climatology until it returns.")
    return [
        {"date": f.day.isoformat(), "tmax_f": f.tmax_f, "tmin_f": f.tmin_f,
         "pop": f.pop, "qpf_in": f.qpf_in, "wind_mph": f.wind_mph}
        for f in fc
    ]


async def _gather_weather(
    lat: float, lon: float, start: date, end: date,
    rain_log: dict[date, float], live: bool,
) -> tuple[list[WeatherDay], list[adv.ForecastDay], str]:
    station = wx.nearest_station(lat, lon)
    observed: list[WeatherDay] = []
    fc: list[adv.ForecastDay] = []
    if live and http_client is not None:
        observed, fc = await asyncio.gather(
            wx.fetch_deos_daily(station.code, start, end, http_client),
            wx.fetch_nws_forecast(lat, lon, http_client),
        )
    if not observed:
        observed = _climatology(start, end, lat)
        source = "climatology"
    else:
        source = f"DEOS {station.name}"
    observed = wx.fill_gaps(observed, start, end, lat)

    dist = wx.station_distance_mi(station, lat, lon)
    for w in observed:
        if w.day in rain_log:
            w.rain_in = wx.rain_gauge_blend(w.rain_in, rain_log[w.day], dist)

    if not fc:
        fc = _forecast_climatology(end, lat)
    return observed, fc, source


def _climatology(start: date, end: date, lat: float) -> list[WeatherDay]:
    """Delmarva daily normals, used only when every live source has failed.

    A degraded answer with an honest label beats no answer, but the interface
    must say so: an advisory built on normals is a starting point, not a
    recommendation to act on.
    """
    import math
    out = []
    d = start
    while d <= end:
        doy = d.timetuple().tm_yday
        seasonal = math.sin((doy - 105) / 365.0 * 2 * math.pi)
        tmax = 63.0 + 22.0 * seasonal
        tmin = 43.0 + 21.0 * seasonal
        out.append(WeatherDay(
            day=d, tmax_f=round(tmax, 1), tmin_f=round(tmin, 1),
            rh_max=88.0, rh_min=52.0, wind_mph=6.5,
            solar_rad_mj_m2=round(11.0 + 11.0 * seasonal, 1),
            rain_in=0.13, source="climatology",
        ))
        d += timedelta(days=1)
    return out


def _forecast_climatology(after: date, lat: float) -> list[adv.ForecastDay]:
    base = _climatology(after + timedelta(days=1), after + timedelta(days=7), lat)
    return [
        adv.ForecastDay(day=w.day, tmax_f=w.tmax_f, tmin_f=w.tmin_f,
                        pop=0.30, qpf_in=0.35, wind_mph=6.5,
                        rh_min=w.rh_min, rh_max=w.rh_max)
        for w in base
    ]


# ---------------------------------------------------------------------------
# The advisory
# ---------------------------------------------------------------------------

@app.post("/api/advisory")
async def compute_advisory(req: AdvisoryRequest) -> dict:
    f = req.field
    as_of = req.as_of or date.today()
    if as_of < f.planting_date:
        raise HTTPException(400, "The as-of date is before planting.")

    crop = get_crop(f.crop_key)
    soil = f.soil()
    cn = curve_number_for(f.soil_key, f.residue_cover_pct)

    observed, fc, wsource = await _gather_weather(
        f.lat, f.lon, f.planting_date, as_of, req.rain_log, req.use_live_weather)

    # Assemble observations for assimilation.
    obs_by_day: dict[date, list[Observation]] = {}
    for r in req.sensor_readings:
        root = crop.root_depth_in((r.day - f.planting_date).days)
        dr = depletion_from_vwc(r.vwc_by_depth, soil, root)
        obs_by_day.setdefault(r.day, []).append(Observation(
            day=r.day, kind="soil_moisture", value=dr,
            sensor_key=r.sensor_class,
            depths_reported=len(r.vwc_by_depth), quality=r.quality))
    for c in req.canopy:
        obs_by_day.setdefault(c.day, []).append(Observation(
            day=c.day, kind=c.kind, value=c.value, sensor_key=c.product))

    assimilator = BayesianAssimilator(soil, crop, f.lat)
    assim_log = []
    for w in observed:
        dap = (w.day - f.planting_date).days
        if dap < 0:
            continue
        result = assimilator.assimilate_day(
            w, dap,
            observations=obs_by_day.get(w.day, []),
            irrigation_in=req.irrigation_log.get(w.day, 0.0),
            curve_number=cn,
        )
        if result.n_observations:
            assim_log.append({
                "date": result.day.isoformat(),
                "prior_in": result.prior_depletion_in,
                "posterior_in": result.posterior_depletion_in,
                "sd_in": result.depletion_sd_in,
                "n_obs": result.n_observations,
                "notes": result.notes,
            })

    fs = assimilator.fs
    pump = econ.PumpSpec(**req.pump) if req.pump else econ.PumpSpec()
    energy = econ.energy_per_acre_inch(f.capacity_gpm, pump)

    result = adv.build_advisory(
        field_id="local",
        field_name=f.name,
        as_of=as_of,
        crop_key=f.crop_key,
        planting=f.planting_date,
        soil=soil,
        system=f.system(),
        depletion_in=fs.dr_in,
        depletion_sd_in=max(0.08, (fs.var_dr) ** 0.5),
        forecast=fc,
        lat_deg=f.lat,
        curve_number=cn,
        kcb_bias=fs.kcb_bias,
        has_sensors=bool(req.sensor_readings),
        energy_cost_per_ac_in=energy["cost_per_ac_in"],
    )

    season = assimilator.history
    yl = adv.yield_loss_estimate(season, crop)
    total_irr = sum(req.irrigation_log.values())
    total_rain = sum(r.rain_in for r in season)
    total_perc = sum(r.deep_perc_in for r in season)

    leach = nitro.nitrate_leaching_estimate(
        deep_percolation_in=total_perc,
        soil_nitrate_lb_per_ac=45.0,
        root_depth_in=crop.root_depth_in((as_of - f.planting_date).days),
        soil_fc_vol=soil.fc_vol,
        irrigation_in=total_irr,
    )

    return {
        "advisory": {
            "decision": result.decision,
            "headline": result.headline,
            "detail": result.detail,
            "as_of": result.as_of.isoformat(),
            "field_name": result.field_name,
            "depletion_in": result.depletion_in,
            "depletion_sd_in": result.depletion_sd_in,
            "raw_in": result.raw_in,
            "taw_in": result.taw_in,
            "percent_depleted": result.percent_depleted,
            "percent_of_mad": result.percent_of_mad,
            "days_to_mad": result.days_to_mad,
            "recommended_net_in": result.recommended_net_in,
            "recommended_gross_in": result.recommended_gross_in,
            "run_hours": result.run_hours,
            "stress_risk_7day": result.stress_risk_7day,
            "deep_perc_risk_in": result.deep_perc_risk_in,
            "growth_stage": result.growth_stage,
            "dap": result.dap,
            "constraints": result.constraints,
            "confidence": result.confidence,
            "cost_estimate_usd": result.cost_estimate_usd,
        },
        "trajectory": result.trajectory,
        "season_to_date": {
            "irrigation_in": round(total_irr, 2),
            "rain_in": round(total_rain, 2),
            "etc_in": round(sum(r.etc_adj_in for r in season), 2),
            "deep_percolation_in": round(total_perc, 2),
            "stress_days": yl["stress_days"],
            "estimated_yield_loss_pct": yl["yield_loss_pct"],
            "relative_et": yl["relative_et"],
        },
        "nitrogen": leach,
        "assimilation": assim_log[-10:],
        "weather_source": wsource,
        "energy": energy,
    }


@app.post("/api/nitrogen/plan")
async def nitrogen_plan(payload: dict) -> dict:
    try:
        plan = nitro.build_nitrogen_plan(**payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "crop": plan.crop_key,
        "yield_goal": plan.yield_goal,
        "total_requirement_lb": plan.total_requirement_lb,
        "credits": plan.credits,
        "total_credits_lb": plan.total_credits_lb,
        "recommended_fertilizer_lb": plan.recommended_fertilizer_lb,
        "split_schedule": plan.split_schedule,
        "psnt_note": plan.psnt_note,
        "leaching_risk": plan.leaching_risk,
        "warnings": plan.warnings,
    }


@app.post("/api/sensors/design")
async def sensor_plan(payload: dict) -> dict:
    width = payload.get("width_ft", 2600)
    height = payload.get("height_ft", 2600)
    spacing = payload.get("spacing_ft", 400)
    cands = sensor_design.grid_candidates(
        width, height, spacing,
        soil_map=payload.get("soil_map"),
        default_soil=payload.get("soil_key", "sassafras"),
    )
    plan = sensor_design.design_network(
        cands,
        budget_usd=payload.get("budget_usd", 3000),
        root_depth_in=payload.get("root_depth_in", 30),
        sensor_key=payload.get("sensor_key", "capacitance_pro"),
        restrictive_layer_in=payload.get("restrictive_layer_in"),
    )
    return {
        "stations": plan.stations,
        "n_stations": plan.n_stations,
        "total_cost_usd": plan.total_cost_usd,
        "expected_depletion_sd_in": plan.expected_depletion_sd_in,
        "sd_without_sensors_in": plan.sd_without_sensors_in,
        "coverage_score": plan.coverage_score,
        "rationale": plan.rationale,
        "tier_comparison": sensor_design.compare_sensor_tiers(
            cands, payload.get("budget_usd", 3000), payload.get("root_depth_in", 30)),
    }


@app.post("/api/economics/cost")
async def cost(payload: dict) -> dict:
    pump = econ.PumpSpec(**payload.get("pump", {}))
    return econ.application_cost(
        gross_in=payload.get("gross_in", 0.8),
        acres=payload.get("acres", 130),
        gpm=payload.get("gpm", 800),
        pump=pump,
    )


# ---------------------------------------------------------------------------
# Nightly batch
# ---------------------------------------------------------------------------

async def nightly_batch(field_rows: list[dict], concurrency: int = 24) -> list[dict]:
    """Recompute every field's balance and queue the morning alerts.

    Called by a scheduler at 03:30 local so results are waiting before growers
    are. Bounded concurrency keeps DEOS and the NWS API from seeing a thundering
    herd; both are public services and this system should be a good neighbour on
    them.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        async with sem:
            try:
                req = AdvisoryRequest(**row)
                out = await compute_advisory(req)
                return {"field_id": row.get("field_id"), "ok": True,
                        "decision": out["advisory"]["decision"],
                        "headline": out["advisory"]["headline"]}
            except Exception as exc:
                return {"field_id": row.get("field_id"), "ok": False,
                        "error": str(exc)}

    return await asyncio.gather(*(one(r) for r in field_rows))


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="app")

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
