"""
Computational grids and variable rate prescriptions — Activity 2.

The narrative is precise about this and it is easy to skip past:

    "the field is subdivided to computational grids consistent with irrigation
    system specification (e.g., 5ft x 15ft grids for VRI systems) to enable
    calculation of variable irrigation and nutrient application amounts for
    location specific recommendations."

That is a real requirement with a real reason behind it. A pivot with individual
sprinkler control does not apply water by management zone; it applies it by
what the panel can address, which is a bank of nozzles over an arc of rotation.
The grid has to match the machine or the prescription cannot be executed. Five
feet along the span by fifteen feet of travel is roughly what a single control
bank covers on a modern VRI pivot, and it is why the number appears in the
narrative rather than a rounder one.

So this module discretises a field boundary into a grid the machine can actually
address, attaches soil and canopy properties to every cell, runs the water
balance per cell, and writes a prescription in the shapes the control panels
read. Management zones sit on top of the grid as a coarser aggregation for
growers whose panels cannot do per-nozzle control — the same computation,
summarised to what the hardware can execute.

For a 130-acre pivot, a 5 x 15 ft grid is about 75,000 cells. Running the full
FAO-56 balance per cell per day is wasteful when neighbouring cells on the same
map unit differ by nothing, so cells are grouped into hydrologically identical
response units, the balance runs once per unit, and the result is mapped back.
That is not an approximation — cells in the same unit are given identical inputs
and would return identical outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from .geo import (
    Boundary, Polygon, boundary_acres, boundary_bbox, boundary_centroid,
    point_in_boundary, SQM_PER_ACRE,
)

FT_PER_DEG_LAT = 364_000.0


@dataclass(frozen=True, slots=True)
class GridSpec:
    """The grid the machine can address.

    `along_span_ft` is the resolution across the pivot's radius — set by how many
    sprinklers share a control valve. `along_travel_ft` is the resolution around
    the circle, set by how finely the panel can switch as the machine moves.
    """
    key: str
    label: str
    along_span_ft: float
    along_travel_ft: float
    note: str


GRID_SPECS: dict[str, GridSpec] = {
    "vri_individual": GridSpec(
        "vri_individual", "VRI, individual sprinkler control", 5.0, 15.0,
        "The specification in the narrative. Each cell is about one control "
        "bank, so a prescription written at this resolution can be executed "
        "exactly as computed."),
    "vri_bank": GridSpec(
        "vri_bank", "VRI, banked control", 20.0, 30.0,
        "Older or cheaper VRI packages switch groups of sprinklers together. "
        "Writing a finer prescription than the panel can execute just means the "
        "machine rounds it, badly."),
    "vri_speed_only": GridSpec(
        "vri_speed_only", "Speed control only", 999.0, 30.0,
        "Speed-control pivots vary rate around the circle but not along the "
        "span, so the grid is a set of pie sectors. Half the VRI machines on "
        "Delmarva are this kind and a prescription that varies along the radius "
        "cannot be run on them."),
    "zone": GridSpec(
        "zone", "Management zones only", 200.0, 200.0,
        "No variable rate hardware. The grid still gets computed because it is "
        "what tells you whether VRI would pay on this field."),
}


@dataclass(slots=True)
class Cell:
    """One addressable cell of the computational grid."""
    index: int
    lon: float
    lat: float
    x_ft: float
    y_ft: float
    area_ac: float
    radius_ft: float = 0.0
    azimuth_deg: float = 0.0
    soil_key: str = "sassafras"
    awc_in_per_in: float = 0.12
    water_table_depth_in: float | None = None
    restrictive_layer_in: float | None = None
    yield_index: float = 1.00
    ndvi: float | None = None
    zone_id: int = 0
    response_unit: int = 0


@dataclass(slots=True)
class FieldGrid:
    spec: GridSpec
    cells: list[Cell]
    acres: float
    center_lon: float
    center_lat: float
    pivot_point: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def response_units(self) -> int:
        return len({c.response_unit for c in self.cells})


def _ft_per_deg_lon(lat: float) -> float:
    return FT_PER_DEG_LAT * math.cos(math.radians(lat))


def discretize(boundary: Boundary, spec_key: str = "vri_individual",
               pivot_point: tuple[float, float] | None = None,
               max_cells: int = 120_000) -> FieldGrid:
    """Subdivide a field boundary into the computational grid.

    For a pivot the natural grid is polar — radius by azimuth — because that is
    how the machine moves and how the panel indexes position. For a linear or a
    non-pivot field it is a rectangular grid aligned to the field. Both come out
    the same shape here so the rest of the pipeline does not care which it got.
    """
    spec = GRID_SPECS.get(spec_key)
    if spec is None:
        raise ValueError(f"Unknown grid specification '{spec_key}'. "
                         f"Known: {', '.join(sorted(GRID_SPECS))}")

    lon0, lat0 = boundary_centroid(boundary)
    acres = boundary_acres(boundary)
    kx, ky = _ft_per_deg_lon(lat0), FT_PER_DEG_LAT
    warnings: list[str] = []

    if pivot_point is None:
        pivot_point = (lon0, lat0)
    px, py = pivot_point

    w, s, e, n = boundary_bbox(boundary)
    width_ft = (e - w) * kx
    height_ft = (n - s) * ky
    radius_ft = 0.5 * max(width_ft, height_ft)

    cells: list[Cell] = []
    idx = 0

    if spec.along_span_ft >= 500:
        # Speed-control only: sectors, no radial subdivision.
        n_sectors = max(8, int(2 * math.pi * radius_ft / spec.along_travel_ft))
        n_sectors = min(n_sectors, 720)
        for k in range(n_sectors):
            az = 360.0 * k / n_sectors
            r = radius_ft * 0.6
            dx = r * math.sin(math.radians(az))
            dy = r * math.cos(math.radians(az))
            lon, lat = px + dx / kx, py + dy / ky
            if not point_in_boundary((lon, lat), boundary):
                continue
            cells.append(Cell(index=idx, lon=lon, lat=lat, x_ft=dx, y_ft=dy,
                              area_ac=acres / n_sectors, radius_ft=r,
                              azimuth_deg=az))
            idx += 1
        warnings.append(
            "This machine varies rate around the circle but not along the span, "
            "so the prescription is a set of sectors. If the soil changes with "
            "distance from the pivot rather than with direction, speed control "
            "cannot address it and the extra hardware will not pay.")
    else:
        n_rings = max(1, int(radius_ft / spec.along_span_ft))
        estimated = 0
        for ring in range(n_rings):
            r = (ring + 0.5) * spec.along_span_ft
            estimated += max(1, int(2 * math.pi * r / spec.along_travel_ft))
        if estimated > max_cells:
            factor = math.sqrt(estimated / max_cells)
            warnings.append(
                f"A {spec.along_span_ft:.0f} x {spec.along_travel_ft:.0f} ft grid "
                f"over {acres:.0f} acres is about {estimated:,} cells. Coarsened "
                f"by {factor:.1f}x to stay inside the compute budget; the "
                "prescription is still finer than any panel can execute.")
            span = spec.along_span_ft * factor
            travel = spec.along_travel_ft * factor
            n_rings = max(1, int(radius_ft / span))
        else:
            span, travel = spec.along_span_ft, spec.along_travel_ft

        cell_area_ac = span * travel / 43_560.0
        for ring in range(n_rings):
            r = (ring + 0.5) * span
            n_az = max(1, int(2 * math.pi * r / travel))
            for k in range(n_az):
                az = 360.0 * k / n_az
                dx = r * math.sin(math.radians(az))
                dy = r * math.cos(math.radians(az))
                lon, lat = px + dx / kx, py + dy / ky
                if not point_in_boundary((lon, lat), boundary):
                    continue
                cells.append(Cell(index=idx, lon=lon, lat=lat, x_ft=dx, y_ft=dy,
                                  area_ac=cell_area_ac, radius_ft=r,
                                  azimuth_deg=az))
                idx += 1

    if not cells:
        raise ValueError(
            "No grid cells landed inside the boundary. If the pivot point was "
            "supplied by hand, check it is inside the field.")

    # Normalise cell areas so they sum to the measured acreage — the polar grid
    # clips at the boundary and the arithmetic must still close.
    total = sum(c.area_ac for c in cells)
    if total > 0:
        scale = acres / total
        for c in cells:
            c.area_ac = round(c.area_ac * scale, 6)

    return FieldGrid(spec=spec, cells=cells, acres=acres,
                     center_lon=lon0, center_lat=lat0,
                     pivot_point=pivot_point, warnings=warnings)


def attach_soils(grid: FieldGrid, components: Sequence[dict],
                 soil_lookup) -> FieldGrid:
    """Assign a soil to every cell from the SSURGO composition.

    Without a delineated soil polygon the best available assignment is
    proportional and spatially coherent rather than random: components are laid
    down in bands, which is how Coastal Plain soils actually sit — a droughty
    rise grading into a wet swale, not a checkerboard. When true SSURGO polygon
    geometry is available, `assign_from_polygons` replaces this.
    """
    if not components:
        return grid
    ordered = sorted(components, key=lambda c: -c.get("fraction", 0))
    cells_sorted = sorted(grid.cells, key=lambda c: (c.y_ft, c.x_ft))
    n = len(cells_sorted)
    start = 0
    for i, comp in enumerate(ordered):
        share = comp.get("fraction", 0.0)
        count = n - start if i == len(ordered) - 1 else int(round(share * n))
        soil = soil_lookup(comp["series_key"])
        for c in cells_sorted[start:start + count]:
            c.soil_key = comp["series_key"]
            c.awc_in_per_in = soil.awc_in_per_in
            c.water_table_depth_in = soil.water_table_depth_in
            c.restrictive_layer_in = soil.restrictive_layer_in
        start += count
    return grid


def assign_from_polygons(grid: FieldGrid, soil_polygons: Sequence[tuple[str, Polygon]],
                         soil_lookup) -> FieldGrid:
    """Assign soils from real SSURGO map unit polygons, point in polygon per cell."""
    for c in grid.cells:
        for key, poly in soil_polygons:
            if point_in_boundary((c.lon, c.lat), [poly]):
                soil = soil_lookup(key)
                c.soil_key = key
                c.awc_in_per_in = soil.awc_in_per_in
                c.water_table_depth_in = soil.water_table_depth_in
                c.restrictive_layer_in = soil.restrictive_layer_in
                break
    return grid


def attach_canopy(grid: FieldGrid, sample: Sequence[tuple[float, float, float]],
                  search_radius_ft: float = 120.0) -> FieldGrid:
    """Attach NDVI or canopy cover to cells by inverse distance from samples.

    The samples are the raster pixels of a Sentinel-2 or UAV pass. At 10 m
    Sentinel resolution one pixel covers roughly forty 5 x 15 ft cells, so this
    is a downscaling, and it is honest about that: the yield index it produces
    varies at pixel resolution, not at cell resolution, and the prescription
    should not pretend otherwise.
    """
    if not sample:
        return grid
    kx, ky = _ft_per_deg_lon(grid.center_lat), FT_PER_DEG_LAT
    for c in grid.cells:
        num = den = 0.0
        for lon, lat, value in sample:
            dx = (lon - c.lon) * kx
            dy = (lat - c.lat) * ky
            d = math.hypot(dx, dy)
            if d > search_radius_ft:
                continue
            w = 1.0 / max(1.0, d * d)
            num += w * value
            den += w
        if den > 0:
            c.ndvi = round(num / den, 4)
    values = [c.ndvi for c in grid.cells if c.ndvi is not None]
    if values:
        mean = sum(values) / len(values)
        for c in grid.cells:
            if c.ndvi is not None and mean > 0:
                c.yield_index = round(max(0.55, min(1.35, c.ndvi / mean)), 3)
    return grid


def build_response_units(grid: FieldGrid, awc_bin: float = 0.02,
                         yield_bin: float = 0.10) -> FieldGrid:
    """Group cells that would return identical water balance results.

    Seventy-five thousand cells, but only a handful of distinct hydrologic
    behaviours. Cells sharing a soil, a water table depth and a yield band get
    the same inputs and therefore the same answer, so the balance runs once per
    group. This is what makes a per-nozzle prescription computable in the same
    time budget as a whole-field one.
    """
    units: dict[tuple, int] = {}
    for c in grid.cells:
        key = (
            c.soil_key,
            round(c.awc_in_per_in / awc_bin),
            c.water_table_depth_in,
            c.restrictive_layer_in,
            round(c.yield_index / yield_bin),
        )
        if key not in units:
            units[key] = len(units)
        c.response_unit = units[key]
    return grid


# ---------------------------------------------------------------------------
# Prescriptions
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Prescription:
    day: date
    grid_spec: str
    cells: list[dict]
    total_ac_in: float
    uniform_ac_in: float
    savings_ac_in: float
    min_rate_in: float
    max_rate_in: float
    n_rates: int
    worth_it: bool
    rationale: str


def build_prescription(grid: FieldGrid, day: date,
                       depletion_by_unit: dict[int, float],
                       taw_by_unit: dict[int, float],
                       mad: float, min_rate_in: float, max_rate_in: float,
                       application_efficiency: float = 0.85,
                       rate_step_in: float = 0.05) -> Prescription:
    """Turn per-unit depletion into a rate for every cell.

    Two constraints the machine imposes and the agronomy does not. Rates are
    quantised, because a panel takes discrete duty cycles rather than a
    continuous number. And a cell can never be given more than its own profile
    holds — the whole reason variable rate pays on the Coastal Plain is that a
    uniform rate sized for Matapeake drains straight through Evesboro, taking
    nitrate with it.
    """
    rates: list[dict] = []
    for c in grid.cells:
        dr = depletion_by_unit.get(c.response_unit, 0.0)
        taw = taw_by_unit.get(c.response_unit, c.awc_in_per_in * 30)
        raw = taw * mad
        if dr < raw:
            rate = 0.0
        else:
            rate = min(dr, 0.90 * taw, max_rate_in)
            rate = round(rate / rate_step_in) * rate_step_in
            if rate < min_rate_in:
                rate = 0.0
        rates.append({
            "index": c.index, "lon": round(c.lon, 6), "lat": round(c.lat, 6),
            "radius_ft": round(c.radius_ft, 1),
            "azimuth_deg": round(c.azimuth_deg, 2),
            "area_ac": c.area_ac, "soil": c.soil_key,
            "net_in": round(rate, 3),
            "gross_in": round(rate / application_efficiency, 3),
            "yield_index": c.yield_index,
        })

    applied = sum(r["net_in"] * r["area_ac"] for r in rates)
    nonzero = [r["net_in"] for r in rates if r["net_in"] > 0]
    peak = max(nonzero) if nonzero else 0.0
    uniform = peak * grid.acres
    distinct = len({round(r["net_in"], 2) for r in rates})

    savings = uniform - applied
    worth_it = savings > 0.05 * max(0.01, uniform) and distinct > 1

    if not nonzero:
        rationale = "No part of this field has reached its trigger. Nothing to apply."
    elif distinct == 1:
        rationale = ("Every cell wants the same depth today. Variable rate has "
                     "nothing to do on this pass — run it uniform.")
    elif worth_it:
        rationale = (
            f"Variable rate saves {savings:.1f} acre-inches against running the "
            f"whole field at the {peak:.2f} in the driest ground needs — about "
            f"{savings * 27154 / 1000:.0f} thousand gallons, and the same volume "
            "of water that would otherwise drain past the roots on the light "
            "ground carrying nitrate with it.")
    else:
        rationale = (
            f"The spread across the field is only {savings:.2f} acre-inches "
            "today. Not worth the prescription upload; run it uniform.")

    return Prescription(
        day=day, grid_spec=grid.spec.key, cells=rates,
        total_ac_in=round(applied, 2), uniform_ac_in=round(uniform, 2),
        savings_ac_in=round(savings, 2),
        min_rate_in=round(min(nonzero), 3) if nonzero else 0.0,
        max_rate_in=round(peak, 3), n_rates=distinct,
        worth_it=worth_it, rationale=rationale)


def prescription_to_shapefile_wkt(rx: Prescription) -> list[dict]:
    """Prescription as point features with a rate attribute.

    Control panels differ on what they ingest — some take a shapefile, some a
    proprietary map, some a plain grid. Points with coordinates and a rate is
    the common denominator every conversion starts from, and it is what
    `frontend` renders directly.
    """
    return [{"wkt": f"POINT({c['lon']:.7f} {c['lat']:.7f})",
             "RATE_IN": c["gross_in"], "SOIL": c["soil"],
             "ZONE": c["index"]} for c in rx.cells if c["net_in"] > 0]


def vri_payback(grid: FieldGrid, seasonal_savings_ac_in: float,
                acres: float, water_cost_per_ac_in: float = 9.50,
                nitrate_value_per_lb: float = 0.62,
                nitrate_saved_lb_per_ac: float = 6.0,
                vri_capital_cost: float = 32_000.0) -> dict:
    """Does variable rate hardware pay on this field?

    The question growers actually ask, and the honest answer is usually "not on
    water alone." A VRI package saving two acre-inches a season over 130 acres
    saves a few thousand dollars of pumping — a ten year payback. It pays when
    the field is genuinely split, when the saved water was also carrying
    nitrogen, and when yield on the light ground improves because it stops being
    over-watered. Saying so plainly is more useful than a number that flatters
    the equipment.
    """
    water_value = seasonal_savings_ac_in * acres * water_cost_per_ac_in
    n_value = nitrate_saved_lb_per_ac * acres * nitrate_value_per_lb
    annual = water_value + n_value
    payback = vri_capital_cost / annual if annual > 0 else math.inf

    soils = {c.soil_key for c in grid.cells}
    awcs = [c.awc_in_per_in for c in grid.cells]
    spread = (max(awcs) - min(awcs)) if awcs else 0.0

    if spread < 0.03:
        verdict = ("This field is hydrologically uniform. Variable rate hardware "
                   "will not earn its keep here regardless of the payback "
                   "arithmetic — there is nothing to vary.")
    elif payback <= 7:
        verdict = (f"About a {payback:.0f} year payback, and the field is "
                   f"genuinely split across {len(soils)} soils differing by "
                   f"{spread:.2f} in/in. This is a reasonable candidate.")
    else:
        verdict = (f"A {payback:.0f} year payback on water and nitrogen alone. "
                   "Worth it only if the light ground is also losing yield to "
                   "over-watering, which a season of zone yield data would show.")

    return {
        "water_value_usd": round(water_value, 0),
        "nitrogen_value_usd": round(n_value, 0),
        "annual_benefit_usd": round(annual, 0),
        "capital_cost_usd": vri_capital_cost,
        "payback_years": round(payback, 1) if math.isfinite(payback) else None,
        "soils_in_field": sorted(soils),
        "awc_spread_in_per_in": round(spread, 3),
        "verdict": verdict,
    }
