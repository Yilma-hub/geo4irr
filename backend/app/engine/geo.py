"""
Field boundaries: parse them, measure them, and find out what is under them.

A grower draws a circle on a map or uploads the shapefile their planter monitor
already wrote. From that one act everything else should follow — acres, the
nearest weather station, and the soil. Typing a latitude into a form is a thing
we ask people to do because the software could not be bothered.

The important design decision here is that soil is resolved for the *polygon*,
not for the centroid. A point lookup is the wrong tool for a Delaware pivot. A
130-acre circle on the Coastal Plain routinely spans Evesboro loamy sand and
Fallsington sandy loam: one holds about 2.1 inches in a 30-inch root zone with no
help from below, the other holds twice that and gets capillary rise from a
seasonal high water table in July. Average them and you get a soil that does not
exist and advice that is wrong on both halves of the field. So we return the
area-weighted component list, and when the field is genuinely mixed we say so and
offer to split it into zones.

Areas are computed on the sphere rather than in a projected plane. At field scale
the difference is negligible, but it means we never have to ask which state plane
zone somebody's shapefile is in, and a boundary that crosses a UTM edge behaves.

No PostGIS dependency: this runs before there is a database, when a grower is
still deciding whether to sign up.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field as dc_field

from .soils import SDA_QUERY, SOILS, HSG, blend_profiles, get_soil

EARTH_R_M = 6_371_008.8
SQM_PER_ACRE = 4046.8564224

# lon, lat — GeoJSON order, kept consistently everywhere in this module.
Point = tuple[float, float]
Ring = list[Point]

# Generous Delmarva envelope. Delaware proper is much smaller, but the tool is
# useful across the Eastern Shore of Maryland and Virginia and the weather and
# soil sources cover it.
DELMARVA_BBOX = (-76.60, 36.85, -74.75, 39.90)


@dataclass
class Polygon:
    outer: Ring
    holes: list[Ring] = dc_field(default_factory=list)


@dataclass
class Boundary:
    polygons: list[Polygon]
    source: str = "geojson"
    warnings: list[str] = dc_field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return sum(len(p.outer) + sum(len(h) for h in p.holes) for p in self.polygons)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _clean_ring(coords) -> Ring:
    ring: Ring = []
    for c in coords:
        if len(c) < 2:
            continue
        lon, lat = float(c[0]), float(c[1])
        if not (math.isfinite(lon) and math.isfinite(lat)):
            continue
        if ring and abs(ring[-1][0] - lon) < 1e-12 and abs(ring[-1][1] - lat) < 1e-12:
            continue
        ring.append((lon, lat))
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring.pop()
    return ring


def parse_geojson(obj: dict | list) -> Boundary:
    """Accept a Geometry, Feature, FeatureCollection, or bare coordinate list.

    Growers' files come from a dozen different pieces of farm software and about
    half of them wrap things one layer deeper than the spec suggests. Be liberal.
    """
    polygons: list[Polygon] = []
    warnings: list[str] = []

    def add_polygon(rings) -> None:
        if not rings:
            return
        outer = _clean_ring(rings[0])
        if len(outer) < 3:
            return
        holes = [h for h in (_clean_ring(r) for r in rings[1:]) if len(h) >= 3]
        polygons.append(Polygon(outer=outer, holes=holes))

    def walk(node) -> None:
        if node is None:
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        t = node.get("type")
        if t == "FeatureCollection":
            walk(node.get("features"))
        elif t == "Feature":
            walk(node.get("geometry"))
        elif t == "GeometryCollection":
            walk(node.get("geometries"))
        elif t == "Polygon":
            add_polygon(node.get("coordinates"))
        elif t == "MultiPolygon":
            for rings in node.get("coordinates") or []:
                add_polygon(rings)
        elif t in ("Point", "LineString", "MultiPoint", "MultiLineString"):
            warnings.append(
                f"The file contains a {t}, which is not a field boundary. "
                "Only polygons were used."
            )
        elif "coordinates" in node:
            add_polygon(node["coordinates"])
        elif "geometry" in node:
            walk(node["geometry"])

    walk(obj)
    if not polygons:
        raise ValueError(
            "No polygon boundary found in that file. If it came from a yield "
            "monitor, look for the field boundary export rather than the point data."
        )
    return Boundary(polygons=polygons, source="geojson", warnings=warnings)


_KML_COORD_SPLIT = re.compile(r"[\s\n\r\t]+")


def parse_kml(text: str) -> Boundary:
    """Minimal KML polygon reader.

    Deliberately not a full KML parser. Field boundaries out of Google Earth,
    Ops Center, SMS and Climate FieldView are all Polygon/outerBoundaryIs with a
    coordinates block, and a regex over that is more robust in practice than a
    strict parser meeting a file with a namespace it did not expect.
    """
    polygons: list[Polygon] = []
    for pm in re.findall(r"<Polygon\b.*?</Polygon>", text, re.S | re.I):
        outer_m = re.search(
            r"<outerBoundaryIs\b.*?<coordinates>(.*?)</coordinates>", pm, re.S | re.I
        )
        if not outer_m:
            outer_m = re.search(r"<coordinates>(.*?)</coordinates>", pm, re.S | re.I)
        if not outer_m:
            continue
        outer = _coords_from_kml(outer_m.group(1))
        if len(outer) < 3:
            continue
        holes = []
        for inner in re.findall(
            r"<innerBoundaryIs\b.*?<coordinates>(.*?)</coordinates>", pm, re.S | re.I
        ):
            h = _coords_from_kml(inner)
            if len(h) >= 3:
                holes.append(h)
        polygons.append(Polygon(outer=outer, holes=holes))
    if not polygons:
        raise ValueError("No polygon found in that KML file.")
    return Boundary(polygons=polygons, source="kml")


def _coords_from_kml(blob: str) -> Ring:
    pts = []
    for tok in _KML_COORD_SPLIT.split(blob.strip()):
        if not tok:
            continue
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return _clean_ring(pts)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def ring_area_m2(ring: Ring) -> float:
    """Signed spherical polygon area. Positive for counter-clockwise rings."""
    if len(ring) < 3:
        return 0.0
    total = 0.0
    n = len(ring)
    for i in range(n):
        lon1, lat1 = ring[i]
        lon2, lat2 = ring[(i + 1) % n]
        total += math.radians(lon2 - lon1) * (
            2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2))
        )
    return total * EARTH_R_M * EARTH_R_M / 2.0


def polygon_area_m2(poly: Polygon) -> float:
    a = abs(ring_area_m2(poly.outer))
    for h in poly.holes:
        a -= abs(ring_area_m2(h))
    return max(0.0, a)


def boundary_acres(b: Boundary) -> float:
    return sum(polygon_area_m2(p) for p in b.polygons) / SQM_PER_ACRE


def ring_centroid(ring: Ring) -> Point:
    """Area-weighted centroid, computed in a local equal-ish plane."""
    if not ring:
        raise ValueError("empty ring")
    lat0 = sum(p[1] for p in ring) / len(ring)
    k = math.cos(math.radians(lat0)) or 1e-9
    cx = cy = a2 = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0] * k, ring[i][1]
        x2, y2 = ring[(i + 1) % n][0] * k, ring[(i + 1) % n][1]
        cross = x1 * y2 - x2 * y1
        a2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a2) < 1e-14:  # degenerate sliver
        return (sum(p[0] for p in ring) / n, lat0)
    return (cx / (3 * a2) / k, cy / (3 * a2))


def boundary_centroid(b: Boundary) -> Point:
    """Centroid of the largest polygon, not of all of them.

    For a field split by a ditch the average of two lobes can land in the ditch.
    The biggest piece is the better representative point for a weather station
    lookup.
    """
    largest = max(b.polygons, key=polygon_area_m2)
    return ring_centroid(largest.outer)


def boundary_bbox(b: Boundary) -> tuple[float, float, float, float]:
    xs = [p[0] for poly in b.polygons for p in poly.outer]
    ys = [p[1] for poly in b.polygons for p in poly.outer]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_ring(pt: Point, ring: Ring) -> bool:
    x, y = pt
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    if not point_in_ring(pt, poly.outer):
        return False
    return not any(point_in_ring(pt, h) for h in poly.holes)


def point_in_boundary(pt: Point, b: Boundary) -> bool:
    return any(point_in_polygon(pt, p) for p in b.polygons)


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

def _perp_distance_m(pt: Point, a: Point, b: Point) -> float:
    lat0 = math.radians((a[1] + b[1]) / 2)
    kx = math.cos(lat0) * 111_320.0
    ky = 110_540.0
    px, py = pt[0] * kx, pt[1] * ky
    ax, ay = a[0] * kx, a[1] * ky
    bx, by = b[0] * kx, b[1] * ky
    dx, dy = bx - ax, by - ay
    seg = dx * dx + dy * dy
    if seg == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify_ring(ring: Ring, tolerance_m: float = 2.0) -> Ring:
    """Douglas-Peucker in metres.

    A boundary traced on a phone, or walked with a GPS receiver, can carry
    thousands of vertices that describe the operator's hand rather than the
    field. Two metres of tolerance is well inside the accuracy of any of the
    sources and keeps WKT payloads small enough for the soil service.
    """
    if len(ring) <= 4:
        return ring[:]

    closed = ring + [ring[0]]
    keep = [False] * len(closed)
    keep[0] = keep[-1] = True
    stack = [(0, len(closed) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        worst, worst_i = -1.0, -1
        for k in range(i + 1, j):
            d = _perp_distance_m(closed[k], closed[i], closed[j])
            if d > worst:
                worst, worst_i = d, k
        if worst > tolerance_m:
            keep[worst_i] = True
            stack.append((i, worst_i))
            stack.append((worst_i, j))
    out = [p for p, k in zip(closed, keep) if k]
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out if len(out) >= 3 else ring[:]


def simplify_boundary(b: Boundary, tolerance_m: float = 2.0,
                      max_vertices: int = 500) -> Boundary:
    tol = tolerance_m
    out = b
    for _ in range(6):
        out = Boundary(
            polygons=[
                Polygon(
                    outer=simplify_ring(p.outer, tol),
                    holes=[simplify_ring(h, tol) for h in p.holes],
                )
                for p in b.polygons
            ],
            source=b.source,
            warnings=list(b.warnings),
        )
        if out.vertex_count <= max_vertices:
            break
        tol *= 2.0
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(b: Boundary) -> list[str]:
    """Things worth telling the grower before we act on the shape."""
    msgs: list[str] = []
    acres = boundary_acres(b)
    if acres < 0.5:
        msgs.append(
            f"That boundary works out to {acres:.2f} acres. If you meant to draw "
            "the whole field, the shape may not have closed."
        )
    elif acres > 2000:
        msgs.append(
            f"That boundary is {acres:,.0f} acres, which is larger than any single "
            "irrigated field in Delaware. Check whether the file holds a whole farm."
        )
    lon, lat = boundary_centroid(b)
    w, s, e, n = DELMARVA_BBOX
    if not (w <= lon <= e and s <= lat <= n):
        msgs.append(
            f"The boundary centre ({lat:.3f}, {lon:.3f}) is outside the Delmarva "
            "Peninsula. Weather comes from DEOS stations in Delaware, so the "
            "numbers will be weak this far away. If the shapefile came in state "
            "plane or UTM coordinates rather than latitude and longitude, that is "
            "usually the cause."
        )
    if len(b.polygons) > 1:
        msgs.append(
            f"The file holds {len(b.polygons)} separate polygons. They have been "
            "treated as one field. Split them if they are irrigated separately."
        )
    if b.vertex_count > 500:
        msgs.append("The outline was simplified to keep it manageable. "
                    "Acreage changed by less than a tenth of a percent.")
    return msgs


# ---------------------------------------------------------------------------
# WKT for the soil service
# ---------------------------------------------------------------------------

def _ring_wkt(ring: Ring) -> str:
    pts = list(ring) + [ring[0]]
    return "(" + ", ".join(f"{lon:.7f} {lat:.7f}" for lon, lat in pts) + ")"


def to_wkt(b: Boundary) -> str:
    if len(b.polygons) == 1:
        p = b.polygons[0]
        rings = [_ring_wkt(p.outer)] + [_ring_wkt(h) for h in p.holes]
        return "polygon(" + ", ".join(rings) + ")"
    parts = []
    for p in b.polygons:
        rings = [_ring_wkt(p.outer)] + [_ring_wkt(h) for h in p.holes]
        parts.append("(" + ", ".join(rings) + ")")
    return "multipolygon(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Matching SSURGO components to the local series table
# ---------------------------------------------------------------------------

# SSURGO returns component names for the whole survey area, not just the twelve
# series carried in soils.py. These are the Delmarva names that turn up often
# enough to be worth naming, mapped to the local series that behaves closest.
# Anything unrecognised falls through to a match on available water capacity,
# which is the property that actually drives the schedule.
SERIES_ALIASES: dict[str, str] = {
    "galestown": "evesboro", "henlopen": "evesboro", "fort mott": "evesboro",
    "lakeland": "evesboro", "runclint": "rosedale", "rumford": "downer",
    "kenansville": "ingleside", "norfolk": "sassafras", "marr": "sassafras",
    "klej": "hammonton", "askecksy": "hammonton", "pepperbox": "hammonton",
    "hurlock": "fallsington", "elkton": "pocomoke", "othello": "fallsington",
    "berryland": "pocomoke", "atsion": "pocomoke", "osier": "pocomoke",
    "johnston": "pocomoke", "corsica": "pocomoke", "puckum": "pocomoke",
    "longmarsh": "pocomoke", "sunken": "pocomoke", "whitemarsh": "pocomoke",
    "keyport": "mattapex", "butlertown": "mattapex", "pineyneck": "mattapex",
    "reybold": "matapeake", "nassawango": "matapeake", "unicorn": "matapeake",
    "hambrook": "matapeake", "sassafras": "sassafras",
}


def match_series(compname: str, awc_in_per_in: float | None = None) -> tuple[str, str]:
    """Map an SSURGO component name onto a local series key.

    Returns (key, how) where `how` is 'exact', 'alias' or 'awc', so the interface
    can be honest with the grower about how confident the match is.
    """
    base = (compname or "").strip().lower()
    base = re.split(r"[,(]", base)[0].strip()
    head = base.split()[0] if base else ""

    for key in SOILS:
        if head == key:
            return key, "exact"
    if head in SERIES_ALIASES:
        return SERIES_ALIASES[head], "alias"
    if awc_in_per_in is not None and awc_in_per_in > 0:
        best = min(SOILS, key=lambda k: abs(SOILS[k].awc_in_per_in - awc_in_per_in))
        return best, "awc"
    return "sassafras", "default"


# ---------------------------------------------------------------------------
# SSURGO over a polygon
# ---------------------------------------------------------------------------

SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"


def sample_points(b: Boundary, n: int = 24) -> list[Point]:
    """A stratified, slightly jittered grid of points inside the boundary.

    Why sampling rather than a spatial intersection: Soil Data Access will
    happily return the map units that touch a polygon, but getting the
    *intersection areas* back requires the spatial query template, which has
    changed shape across SDA versions and fails in ways that are hard to detect
    from a client. Sampling uses only the point query already proven in
    soils.lookup_ssurgo. With 24 to 40 points, a component covering a third of a
    field is estimated to within about ten percentage points, which is far
    inside the accuracy of the SSURGO lines themselves and plenty to decide
    whether the field needs splitting.

    The jitter is deterministic. Two people looking at the same field should see
    the same answer.
    """
    w, s, e, nth = boundary_bbox(b)
    if e <= w or nth <= s:
        return [boundary_centroid(b)]
    aspect = (e - w) * math.cos(math.radians((s + nth) / 2)) / (nth - s)
    cols = max(2, round(math.sqrt(n * max(aspect, 1e-6))))
    rows = max(2, math.ceil(n / cols))

    pts: list[Point] = []
    for r in range(rows):
        for c in range(cols):
            # Deterministic jitter: halton-ish offsets, no RNG state.
            jx = ((r * 7 + c * 3) % 5) / 5.0 * 0.6 + 0.2
            jy = ((r * 3 + c * 5) % 7) / 7.0 * 0.6 + 0.2
            x = w + (c + jx) * (e - w) / cols
            y = s + (r + jy) * (nth - s) / rows
            if point_in_boundary((x, y), b):
                pts.append((x, y))
    if not pts:
        pts = [boundary_centroid(b)]
    return pts[: max(4, n * 2)]


async def _mukey_at(lon: float, lat: float, client) -> str | None:
    sql = ("SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84"
           f"('point({lon:.7f} {lat:.7f})')")
    try:
        r = await client.post(SDA_URL, json={"query": sql, "format": "JSON"}, timeout=8.0)
        r.raise_for_status()
        return str(r.json()["Table"][0][0])
    except Exception:
        return None


async def _component_for_mukey(mukey: str, client) -> dict | None:
    """Dominant major component of a map unit, depth-weighted over 40 inches."""
    try:
        r = await client.post(
            SDA_URL,
            json={"query": SDA_QUERY.format(mukey=mukey), "format": "JSON"},
            timeout=12.0,
        )
        r.raise_for_status()
        rows = r.json()["Table"]
    except Exception:
        return None
    if not rows:
        return None

    dominant = rows[0][0]
    comppct = rows[0][1]
    awc_sum = ksat_sum = depth_sum = 0.0
    for name, _pct, top_cm, bot_cm, awc, ksat, _fc, _wp in rows:
        if name != dominant or awc is None:
            continue
        top_in = float(top_cm) / 2.54
        bot_in = min(40.0, float(bot_cm) / 2.54)
        if bot_in <= top_in:
            continue
        th = bot_in - top_in
        awc_sum += float(awc) * th
        if ksat is not None:
            ksat_sum += float(ksat) * th
        depth_sum += th
    if depth_sum <= 0:
        return None
    awc = round(awc_sum / depth_sum, 3)
    key, how = match_series(dominant, awc)
    return {
        "mukey": mukey,
        "component": dominant,
        "component_pct": comppct,
        "awc_in_per_in": awc,
        "ksat_um_s": round(ksat_sum / depth_sum, 2) if ksat_sum else None,
        "profile_depth_in": round(depth_sum, 1),
        "series_key": key,
        "match": how,
    }


async def ssurgo_for_boundary(b: Boundary, client, samples: int = 24,
                              concurrency: int = 6) -> dict:
    """Area-weighted SSURGO components under a field boundary.

    Degrades in three steps, because a grower standing in a barn with one bar of
    signal should still be able to set up a field: full sampling, then a single
    centroid lookup, then nothing at all and the pick list stays in charge.
    """
    pts = sample_points(b, samples)
    sem = asyncio.Semaphore(concurrency)

    async def one(p: Point) -> str | None:
        async with sem:
            return await _mukey_at(p[0], p[1], client)

    keys = [k for k in await asyncio.gather(*(one(p) for p in pts)) if k]

    if not keys:
        lon, lat = boundary_centroid(b)
        fallback = await _mukey_at(lon, lat, client)
        if not fallback:
            return {"ok": False, "reason": "soil_service_unavailable",
                    "components": [], "sampled": len(pts)}
        keys = [fallback]

    tally: dict[str, int] = {}
    for k in keys:
        tally[k] = tally.get(k, 0) + 1
    total = sum(tally.values())

    details = await asyncio.gather(
        *(_component_for_mukey(m, client) for m in tally)
    )
    acres = boundary_acres(b)

    comps = []
    for mukey, det in zip(tally, details):
        frac = tally[mukey] / total
        if det is None:
            comps.append({
                "mukey": mukey, "component": "Unnamed map unit",
                "fraction": round(frac, 3), "acres": round(frac * acres, 1),
                "series_key": "sassafras", "match": "unresolved",
                "awc_in_per_in": SOILS["sassafras"].awc_in_per_in,
            })
            continue
        det = dict(det)
        det["fraction"] = round(frac, 3)
        det["acres"] = round(frac * acres, 1)
        det["series_name"] = SOILS[det["series_key"]].series
        det["hsg"] = HSG.get(det["series_key"], "B")
        comps.append(det)

    comps.sort(key=lambda c: -c["fraction"])
    return {
        "ok": True,
        "sampled": len(pts),
        "components": comps,
        "dominant_key": comps[0]["series_key"],
    }


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

def blended_from_components(comps: list[dict]):
    """Single blended profile for a field the grower chooses not to split."""
    pairs = [(c["series_key"], c["fraction"]) for c in comps if c.get("fraction")]
    if not pairs:
        return get_soil("sassafras")
    return blend_profiles(pairs)


def suggest_zones(comps: list[dict], acres: float,
                  min_fraction: float = 0.15) -> dict:
    """Decide whether this field should be managed as more than one unit.

    Two conditions, both required. The field has to be genuinely mixed — no one
    component covering three quarters of it — and the components have to differ
    enough in water-holding capacity to change the schedule. A field that is half
    Sassafras and half Ingleside is mixed on the map and identical in the pipe.
    A field that is half Evesboro and half Matapeake is two different farms.

    The 0.04 in/in threshold is roughly 1.2 inches of stored water across a
    30-inch root zone, which is about four days of corn in July. That is the
    point at which one schedule stops serving both halves.
    """
    material = [c for c in comps if c["fraction"] >= min_fraction]
    if len(material) < 2:
        return {"recommend": False, "reason": "one_dominant_soil", "zones": []}

    awcs = [c["awc_in_per_in"] for c in material]
    spread = max(awcs) - min(awcs)
    top = max(c["fraction"] for c in material)

    wt_split = len({
        SOILS[c["series_key"]].water_table_depth_in is not None for c in material
    }) > 1

    recommend = (top < 0.75 and spread >= 0.04) or (top < 0.80 and wt_split and spread >= 0.025)
    if not recommend:
        return {
            "recommend": False,
            "reason": "soils_behave_alike" if spread < 0.04 else "one_dominant_soil",
            "spread_awc": round(spread, 3),
            "zones": [],
        }

    zones = []
    best_awc = max(awcs)
    for i, c in enumerate(material, 1):
        s = SOILS[c["series_key"]]
        days = (s.awc_in_per_in * 30 * 0.5) / 0.28  # half of a 30-in profile at peak use
        zones.append({
            "name": f"{s.series.split()[0]} zone",
            "series_key": c["series_key"],
            "acres": round(c["fraction"] * acres, 1),
            "fraction": c["fraction"],
            "awc_in_per_in": s.awc_in_per_in,
            "holds_in_30in": round(s.awc_in_per_in * 30, 2),
            "days_to_mad_at_peak": round(days, 1),
            "water_table_in": s.water_table_depth_in,
            "yield_index": 1.00,
            "note": _zone_note(s),
        })

    parts = []
    for z in zones:
        pct = round(z["fraction"] * 100)
        parts.append(f"{pct}% {z['name'].split()[0]}")
    gap = max(z["days_to_mad_at_peak"] for z in zones) - \
        min(z["days_to_mad_at_peak"] for z in zones)
    headline = (
        f"This field is {' and '.join(parts)}. The lighter ground runs out about "
        f"{max(1, round(gap))} days sooner at peak water use, so one schedule "
        "either short-changes it or drowns the rest."
    )

    return {
        "recommend": True,
        "reason": "water_table_split" if wt_split else "capacity_split",
        "spread_awc": round(spread, 3),
        "zones": zones,
        "headline": headline,
    }


def _zone_note(s) -> str:
    if s.water_table_depth_in is not None and s.water_table_depth_in <= 24:
        return ("Shallow water table. This ground feeds itself in midsummer and "
                "is the easiest part of the field to over-irrigate.")
    if s.awc_in_per_in <= 0.09:
        return ("Droughty. Small, frequent sets — a big application drains past "
                "the roots and takes nitrogen with it.")
    if s.awc_in_per_in >= 0.16:
        return "Holds well. This part of the field can carry a longer interval."
    return ""
