"""
The bundled statewide irrigated-areas layer.

2,973 fields over 152,435 acres, built from DNREC's Irrigation2018Merge by way
of tools/build_irrigated_layer.py. The layer ships inside the app, so a mistake
in the build is a mistake every user sees with no way to tell. These tests hold
the encoding, the acreage and the known-bad records in place.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from app.engine.geo import Boundary, Polygon, boundary_acres

LAYER = (pathlib.Path(__file__).resolve().parents[2]
         / "frontend" / "data" / "irrigated_areas.js")

pytestmark = pytest.mark.skipif(not LAYER.exists(),
                                reason="layer not built; run tools/build_irrigated_layer.py")


def load():
    raw = LAYER.read_text()
    return json.loads(raw[raw.index("=") + 1:-1])


def decode(enc, scale):
    out, x, y = [], 0, 0
    for i in range(0, len(enc), 2):
        x += enc[i]
        y += enc[i + 1]
        out.append((x / scale, y / scale))
    out.append(out[0])
    return out


def as_boundary(f, scale):
    return Boundary(polygons=[
        Polygon(outer=decode(poly[0], scale),
                holes=[decode(r, scale) for r in poly[1:]])
        for poly in f["g"]])


def test_the_layer_holds_every_field_that_survived_the_build():
    d = load()
    assert d["scale"] == 100_000
    assert len(d["fields"]) == 2973
    total = sum(f["a"] for f in d["fields"])
    assert 152_000 < total < 153_000, f"total acreage drifted to {total}"


def test_every_field_sits_inside_delaware():
    """A decoding error would scatter fields across the globe silently."""
    d = load()
    for f in d["fields"]:
        lon, lat = f["c"]
        assert -76.0 < lon < -74.9, f"field {f['i']} at longitude {lon}"
        assert 38.0 < lat < 40.0, f"field {f['i']} at latitude {lat}"


def test_delta_encoding_round_trips_to_the_stated_acreage():
    """Decoded geometry must reproduce the acreage the popup shows.

    Simplification is 2 m, so a small difference is expected; a large one means
    the encoding or the ring order is wrong.
    """
    d = load()
    scale = d["scale"]
    errs = []
    for f in d["fields"]:
        if f["a"] <= 1 or "f" in f:
            continue
        computed = boundary_acres(as_boundary(f, scale))
        errs.append(abs(computed - f["a"]) / f["a"] * 100)
    errs.sort()
    median = errs[len(errs) // 2]
    p99 = errs[int(len(errs) * 0.99)]
    assert median < 1.0, f"median acreage error {median:.2f}%"
    assert p99 < 5.0, f"99th percentile acreage error {p99:.2f}%"


def test_rings_are_closed_and_have_enough_vertices_to_be_a_field():
    d = load()
    scale = d["scale"]
    for f in d["fields"][:400]:
        for poly in f["g"]:
            ring = decode(poly[0], scale)
            assert ring[0] == ring[-1], f"field {f['i']} ring is not closed"
            assert len(ring) >= 4, f"field {f['i']} ring has {len(ring)} vertices"


def test_the_known_bad_records_are_flagged_rather_than_quietly_shipped():
    """OBJECTID 7084-7108 is one bad batch in the source.

    Two have no geometry at all and are dropped; several disagree with their own
    stated acreage by 38-79% where the rest of the file agrees to 0.1%. A field
    that cannot be trusted has to say so on screen, because an unmarked bad
    outline is worse than a missing one.
    """
    d = load()
    ids = {f["i"] for f in d["fields"]}
    assert 7106 not in ids and 7108 not in ids, "null-geometry records were shipped"

    flagged = {f["i"] for f in d["fields"] if "f" in f}
    assert flagged, "no records flagged at all — the check is not running"
    for oid in (7085, 7100, 7105, 7107):
        assert oid in flagged, f"OBJECTID {oid} disagrees with its geometry but is unflagged"
    for f in d["fields"]:
        if "f" in f:
            assert all(isinstance(t, str) and t for t in f["f"])


def test_counties_are_labelled_and_the_split_matches_where_irrigation_is():
    """Sussex carries most of Delaware's irrigation; New Castle very little."""
    d = load()
    counts = {}
    for f in d["fields"]:
        counts[f["co"]] = counts.get(f["co"], 0) + 1
    assert set(counts) <= {"Su", "Ke", "Ne"}
    assert counts["Su"] > counts["Ke"] > counts["Ne"]


def test_acreage_does_not_come_from_the_web_mercator_column():
    """Shape__Are in the source is Web Mercator square metres.

    Divided by 4046.86 it overstates a Delaware field by about 1/cos^2(39 deg),
    roughly 65%. If anyone ever wires the popup to that column instead of
    IRRIGATEDA, the totals jump and this catches it.
    """
    d = load()
    total = sum(f["a"] for f in d["fields"])
    inflated = total / math.cos(math.radians(39.0)) ** 2
    assert abs(total - inflated) > 50_000     # the two are far apart
    assert total < 200_000, "acreage looks like Web Mercator, not real acres"


def test_every_field_carries_a_bounding_box_that_contains_it():
    """Drawing decisions are made from the bbox, not the centroid.

    A 130-acre pivot is roughly half a mile across. Judging visibility by the
    centroid drops the field the moment its middle leaves the viewport — which
    is exactly when the grower has zoomed in far enough to care about it. If a
    bbox were wrong, the field would flicker in and out with no obvious cause.
    """
    d = load()
    scale = d["scale"]
    for f in d["fields"]:
        assert "b" in f, f"field {f['i']} has no bounding box"
        minx, miny, maxx, maxy = f["b"]
        assert minx < maxx and miny < maxy, f"field {f['i']} has a degenerate bbox"
        lon, lat = f["c"]
        assert minx <= lon <= maxx, f"field {f['i']} centroid outside its own bbox"
        assert miny <= lat <= maxy, f"field {f['i']} centroid outside its own bbox"

    # and the box really does bound the geometry
    for f in d["fields"][:300]:
        minx, miny, maxx, maxy = f["b"]
        for poly in f["g"]:
            for x, y in decode(poly[0], scale):
                assert minx - 1e-4 <= x <= maxx + 1e-4, f"field {f['i']} vertex outside bbox"
                assert miny - 1e-4 <= y <= maxy + 1e-4, f"field {f['i']} vertex outside bbox"


def test_the_encoded_geometry_still_matches_the_source_shapefile():
    """Guards the whole encode path against a silent shift.

    Compares decoded bounding boxes back to the original DNREC shapefile when it
    is available. Simplification moves a vertex by a couple of metres at most; a
    coordinate-order bug would move it by degrees.
    """
    src = pathlib.Path("/mnt/user-data/uploads/irrigated_areas.shp")
    if not src.exists():
        pytest.skip("source shapefile not present")
    from app.engine import ingest

    geoms = ingest.read_shp_geometry(src.read_bytes())
    recs = ingest.read_dbf(src.with_suffix(".dbf").read_bytes())
    d = load()
    scale = d["scale"]
    by = {f["i"]: f for f in d["fields"]}

    worst = 0.0
    for g, r in zip(geoms, recs):
        oid = r.get("OBJECTID")
        if oid not in by or g["type"] != "Polygon":
            continue
        pts = [(c[0], c[1]) for c in g["coordinates"][0]]
        sb = (min(p[0] for p in pts), min(p[1] for p in pts),
              max(p[0] for p in pts), max(p[1] for p in pts))
        worst = max(worst, max(abs(a - b) for a, b in zip(sb, by[oid]["b"])))
    # 0.0002 degrees is about 70 ft — far above simplification, far below a bug
    assert worst < 0.0002, f"encoded geometry drifted {worst:.6f} deg from the source"
