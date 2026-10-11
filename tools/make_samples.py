#!/usr/bin/env python3
"""
Generate sample field boundaries for testing the upload paths.

Writes GeoJSON, KML and a zipped shapefile for three real-ish Delaware fields,
so the boundary reader can be exercised without waiting for a grower to send a
file. Also writes one deliberately broken case and one in the wrong projection,
because those are the two things that will actually happen in the field and both
should produce a message a person can act on rather than a stack trace.

    python tools/make_samples.py            # writes into samples/
    python tools/make_samples.py --out /tmp
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import zipfile

SQM_ACRE = 4046.8564224

FIELDS = [
    # name,           lat,      lon,       acres, shape
    ("home_pivot", 38.6431, -75.4664, 132.0, "circle"),
    ("airport_field", 38.7440, -75.6010, 96.0, "square"),
    ("back_forty", 38.9230, -75.5780, 58.0, "square"),
]


def circle(lat0: float, lon0: float, acres: float, n: int = 96):
    r = math.sqrt(acres * SQM_ACRE / math.pi)
    ring = []
    for i in range(n):
        a = 2 * math.pi * i / n
        dy = r * math.cos(a) / 110_540.0
        dx = r * math.sin(a) / (111_320.0 * math.cos(math.radians(lat0)))
        ring.append([round(lon0 + dx, 7), round(lat0 + dy, 7)])
    ring.append(ring[0])
    return ring


def square(lat0: float, lon0: float, acres: float):
    side = math.sqrt(acres * SQM_ACRE)
    dlat = side / 110_540.0
    dlon = side / (111_320.0 * math.cos(math.radians(lat0)))
    return [
        [round(lon0, 7), round(lat0, 7)],
        [round(lon0 + dlon, 7), round(lat0, 7)],
        [round(lon0 + dlon, 7), round(lat0 + dlat, 7)],
        [round(lon0, 7), round(lat0 + dlat, 7)],
        [round(lon0, 7), round(lat0, 7)],
    ]


def ring_for(kind, lat, lon, acres):
    return circle(lat, lon, acres) if kind == "circle" else square(lat, lon, acres)


def write_geojson(path, rings_by_name):
    features = [
        {"type": "Feature",
         "properties": {"name": name, "grower": "Sample Farms"},
         "geometry": {"type": "Polygon", "coordinates": [ring]}}
        for name, ring in rings_by_name
    ]
    path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=1))


def write_kml(path, name, ring):
    coords = " ".join(f"{lon},{lat},0" for lon, lat in ring)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        f'  <Placemark><name>{name}</name><Polygon><outerBoundaryIs><LinearRing>\n'
        f'    <coordinates>{coords}</coordinates>\n'
        '  </LinearRing></outerBoundaryIs></Polygon></Placemark>\n'
        '</Document></kml>\n'
    )


def shp_bytes(ring):
    """Minimal ESRI shapefile with one polygon record.

    Shapefile convention has outer rings clockwise, so the ring is reversed if
    it came in counter-clockwise.
    """
    pts = ring[:-1] if ring[0] == ring[-1] else ring[:]
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    if area > 0:                      # counter-clockwise -> flip to CW
        pts = list(reversed(pts))
    pts = pts + [pts[0]]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    box = (min(xs), min(ys), max(xs), max(ys))

    content = (struct.pack("<i", 5) + struct.pack("<4d", *box)
               + struct.pack("<ii", 1, len(pts)) + struct.pack("<i", 0)
               + b"".join(struct.pack("<2d", x, y) for x, y in pts))
    rec = struct.pack(">ii", 1, len(content) // 2) + content
    hdr = (struct.pack(">i", 9994) + b"\x00" * 20
           + struct.pack(">i", (100 + len(rec)) // 2)
           + struct.pack("<ii", 1000, 5)
           + struct.pack("<4d", *box) + struct.pack("<4d", 0, 0, 0, 0))
    return hdr + rec


WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
             '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
             'UNIT["Degree",0.0174532925199433]]')
UTM_PRJ = ('PROJCS["NAD_1983_UTM_Zone_18N",GEOGCS["GCS_North_American_1983",'
           'DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,'
           '298.257222101]]],PROJECTION["Transverse_Mercator"]]')


def write_shapefile_zip(path, ring, prj=WGS84_PRJ, compress=True):
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", mode) as z:
        z.writestr("field.shp", shp_bytes(ring))
        z.writestr("field.prj", prj)
        z.writestr("field.dbf", b"\x00" * 32)   # placeholder; not read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent.parent / "samples"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rings = {name: ring_for(kind, lat, lon, ac)
             for name, lat, lon, ac, kind in FIELDS}

    write_geojson(out / "fields.geojson", list(rings.items()))
    write_geojson(out / "home_pivot.geojson", [("home_pivot", rings["home_pivot"])])
    write_kml(out / "home_pivot.kml", "Home pivot", rings["home_pivot"])
    write_shapefile_zip(out / "airport_field_shapefile.zip", rings["airport_field"])
    write_shapefile_zip(out / "airport_field_stored.zip", rings["airport_field"],
                        compress=False)
    write_shapefile_zip(out / "wrong_projection.zip", rings["back_forty"],
                        prj=UTM_PRJ)

    # A field in Iowa: right file format, wrong part of the country. Should be
    # accepted and flagged, not rejected.
    write_geojson(out / "off_peninsula.geojson",
                  [("iowa_field", square(41.6, -93.6, 120.0))])

    # Points, not a boundary. The commonest wrong export off a yield monitor.
    (out / "points_not_boundary.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {},
                      "geometry": {"type": "Point", "coordinates": [-75.47, 38.64]}}],
    }, indent=1))

    for p in sorted(out.iterdir()):
        print(f"  {p.name:34s} {p.stat().st_size:>7,} bytes")
    print(f"\nWrote {len(list(out.iterdir()))} sample files to {out}")


if __name__ == "__main__":
    main()
