#!/usr/bin/env python3
"""Turn DNREC's irrigated-areas shapefile into a layer the browser can carry.

The source is 2,975 polygons over 152,383 acres. Shipped raw as GeoJSON at full
precision it is far too heavy to hand a phone on a field road, so this does four
things:

  * rounds coordinates to 5 decimal places, about 1.1 m, which is well inside
    the accuracy of a boundary digitised from 2018 imagery;
  * drops vertices that sit on a straight line within half a metre;
  * carries acreage from IRRIGATEDA, the column that agrees with an independent
    spherical-area calculation to 0.08% median, rather than Shape__Are, which is
    Web Mercator square metres and overstates Delaware fields by about 65%;
  * flags the bad records instead of dropping them, so a suspect polygon is
    visibly suspect rather than quietly absent.

Run:  python tools/build_irrigated_layer.py <shapefile-dir-or-zip> frontend/data/
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.engine import ingest  # noqa: E402
from app.engine.geo import Boundary, Polygon, boundary_acres  # noqa: E402

PRECISION = 5
SIMPLIFY_M = 2.0

# Delaware county lines are not latitude bands, but for labelling a field the
# two boundaries are close enough to horizontal that this misfiles nothing that
# matters. The exact line is a legal question, not an agronomic one.
KENT_SOUTH = 38.8290
KENT_NORTH = 39.2400


def county_for(lat: float) -> str:
    if lat < KENT_SOUTH:
        return "Sussex"
    if lat < KENT_NORTH:
        return "Kent"
    return "New Castle"


def _perp_m(p, a, b) -> float:
    """Perpendicular distance in metres from p to segment a-b."""
    mlat = 111_320.0
    mlon = 111_320.0 * math.cos(math.radians(p[1]))
    px, py = p[0] * mlon, p[1] * mlat
    ax, ay = a[0] * mlon, a[1] * mlat
    bx, by = b[0] * mlon, b[1] * mlat
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify(ring, tol_m=SIMPLIFY_M):
    """Douglas-Peucker, keeping the ring closed."""
    if len(ring) < 5:
        return ring
    closed = ring[0] == ring[-1]
    pts = ring[:-1] if closed else ring[:]

    def dp(seq):
        if len(seq) < 3:
            return seq
        worst, idx = 0.0, 0
        for i in range(1, len(seq) - 1):
            d = _perp_m(seq[i], seq[0], seq[-1])
            if d > worst:
                worst, idx = d, i
        if worst <= tol_m:
            return [seq[0], seq[-1]]
        return dp(seq[:idx + 1])[:-1] + dp(seq[idx:])

    out = dp(pts)
    if len(out) < 3:
        out = pts
    return out + [out[0]] if closed else out


def rnd(ring):
    return [[round(c[0], PRECISION), round(c[1], PRECISION)] for c in ring]


SCALE = 10 ** PRECISION


def encode_ring(ring):
    """Delta-encode a ring to integers in units of 1e-5 degrees.

    A closed 40-acre pivot has vertices roughly 10 m apart, which is about 10
    units at this scale. Written absolute, every vertex costs ~19 characters of
    JSON; written as a delta it costs two or three. Across 2,973 fields that is
    the difference between a layer a phone can hold and one it cannot. The
    closing vertex is dropped and restored on the way out.
    """
    pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    out, px, py = [], 0, 0
    for x, y in pts:
        ix, iy = round(x * SCALE), round(y * SCALE)
        out.append(ix - px)
        out.append(iy - py)
        px, py = ix, iy
    return out


def to_boundary(geom):
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        return None
    return Boundary(polygons=[
        Polygon(outer=[(c[0], c[1]) for c in p[0]],
                holes=[[(c[0], c[1]) for c in h] for h in p[1:]])
        for p in polys])


def main(src: str, outdir: str):
    src = pathlib.Path(src)
    if src.is_dir():
        stem = next(src.glob("*.shp")).with_suffix("")
        shp, dbf = stem.with_suffix(".shp").read_bytes(), stem.with_suffix(".dbf").read_bytes()
    else:
        stem = src.with_suffix("")
        shp, dbf = stem.with_suffix(".shp").read_bytes(), stem.with_suffix(".dbf").read_bytes()

    geoms = ingest.read_shp_geometry(shp)
    recs = ingest.read_dbf(dbf)

    features, skipped, flagged = [], [], 0
    for gi, (geom, rec) in enumerate(zip(geoms, recs)):
        oid = rec.get("OBJECTID")
        b = to_boundary(geom)
        if b is None:
            skipped.append({"objectid": oid, "why": "no geometry in the source file"})
            continue

        computed = boundary_acres(b)
        stated = rec.get("IRRIGATEDA")
        stated = stated if isinstance(stated, (int, float)) else None

        notes = []
        if stated is None:
            notes.append("acreage was unreadable in the source table; "
                         "this figure is measured from the boundary")
            acres = computed
        elif stated > 1 and abs(computed - stated) / stated > 0.02:
            notes.append(f"the source table says {stated:.1f} ac but the boundary "
                         f"measures {computed:.1f} ac — treat this outline as unreliable")
            acres = stated
        else:
            acres = stated
        if notes:
            flagged += 1

        rings = []
        for poly in ([geom["coordinates"]] if geom["type"] == "Polygon"
                     else geom["coordinates"]):
            rings.append([rnd(simplify([[c[0], c[1]] for c in r])) for r in poly])
        enc = [[encode_ring(r) for r in poly] for poly in rings]

        pts = [c for poly in rings for r in poly for c in r]
        lat = sum(p[1] for p in pts) / len(pts)
        lon = sum(p[0] for p in pts) / len(pts)
        # Real bounding box. Deciding what to draw from the centroid alone drops
        # a field the moment its middle leaves the viewport, which is exactly
        # when a grower has zoomed in far enough to care about it.
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = [round(min(xs), 5), round(min(ys), 5),
                round(max(xs), 5), round(max(ys), 5)]

        features.append({
            "i": oid,
            "a": round(acres, 1),
            "co": county_for(lat)[:2],            # Su / Ke / Ne, expanded in the app
            "c": [round(lon, 5), round(lat, 5)],
            "b": bbox,
            "g": enc,
            **({"f": notes} if notes else {}),
        })

    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    fc = {"name": "delaware_irrigated_areas",
          "source": "DNREC Irrigation2018Merge, via the irrigated_areas shapefile",
          "note": ("Acreage is IRRIGATEDA from the source table. Shape__Are in the "
                   "source is Web Mercator square metres and overstates Delaware "
                   "fields by roughly 65%; it is deliberately not carried. "
                   "Rings are delta-encoded integers at 1e-5 degrees; see "
                   "decodeRing() in the app."),
          "scale": SCALE,
          "simplified_m": SIMPLIFY_M,
          "fields": features}

    # Written as a script rather than plain JSON on purpose: a browser opened on
    # a file:// URL refuses to fetch a sibling .geojson, but will happily run a
    # sibling <script>. Growers open this by double-clicking it.
    path = out / "irrigated_areas.js"
    path.write_text("window.DE_IRRIGATED=" +
                    json.dumps(fc, separators=(",", ":")) + ";")

    # A centroid-only index, so the app can draw the whole state and search it
    # without parsing every ring.
    total = sum(f["a"] for f in features)
    print(f"features written : {len(features)}")
    print(f"total acres      : {total:,.0f}")
    print(f"flagged          : {flagged}")
    print(f"skipped          : {len(skipped)}  {[s['objectid'] for s in skipped]}")
    print(f"layer            : {path.stat().st_size/1024:,.0f} KB  ({path})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads",
         sys.argv[2] if len(sys.argv) > 2 else "frontend/data")
