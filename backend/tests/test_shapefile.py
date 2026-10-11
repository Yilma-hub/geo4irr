"""
Server-side ESRI shapefile ingest — Activity 3.

The narrative requires Geo4Irr to accept "crop yield in raster and feature
class (e.g., ESRI shapefile or Google KML) formats." The browser parses .shp
geometry so a grower can draw or upload a boundary, but a yield export is a
different problem: the numbers are in the .dbf attribute table, and any
non-browser client of the API needs to be able to post the file the combine
wrote. These tests drive the reader with real bytes rather than a mock, because
the whole point is that the byte layout is handled correctly.
"""

from __future__ import annotations

import pytest

from app.engine import ingest
from tests.shapefile_fixtures import dbf, point_shp, polygon_shp, zip_it

PRJ_WGS84 = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
    '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)
PRJ_UTM18 = (
    'PROJCS["WGS_1984_UTM_Zone_18N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]]],'
    'PROJECTION["Transverse_Mercator"],UNIT["Meter",1.0]]'
)
PRJ_STATEPLANE = (
    'PROJCS["NAD_1983_StatePlane_Delaware_FIPS_0700",'
    'PROJECTION["Transverse_Mercator"],UNIT["Foot_US",0.3048006096012192]]'
)

YIELD_FIELDS = [("Yld_Vol_Dr", b"N", 12, 3), ("Moisture", b"N", 6, 1),
                ("Variety", b"C", 16, 0)]


def yield_zip(n: int = 60, compress: bool = True, with_dbf: bool = True,
              prj: str | None = PRJ_WGS84) -> bytes:
    pts = [(-75.52 + (i % 10) * 0.0004, 38.64 + (i // 10) * 0.0003)
           for i in range(n)]
    members = {"harvest/Field7.shp": point_shp(pts)}
    if with_dbf:
        rows = [{"Yld_Vol_Dr": 180 + (i % 17) * 2.5, "Moisture": 15.5,
                 "Variety": "P1197AM"} for i in range(n)]
        members["harvest/Field7.dbf"] = dbf(rows, YIELD_FIELDS)
    if prj:
        members["harvest/Field7.prj"] = prj
    members["__MACOSX/._Field7.shp"] = b"resource fork junk"
    return zip_it(members, compress=compress)


# --- geometry --------------------------------------------------------------

def test_reads_every_point_and_its_attributes():
    parsed = ingest.read_shapefile_zip(yield_zip(60))
    assert parsed["n_features"] == 60
    assert parsed["epsg"] == 4326
    assert parsed["warnings"] == []
    first = parsed["features"][0]
    assert first["geometry"]["type"] == "Point"
    assert first["properties"]["Variety"] == "P1197AM"
    assert first["properties"]["Moisture"] == pytest.approx(15.5)


@pytest.mark.parametrize("compress", [True, False])
def test_reads_both_deflated_and_stored_archives(compress):
    """Some monitors write stored entries; the reader must not care."""
    parsed = ingest.read_shapefile_zip(yield_zip(20, compress=compress))
    assert parsed["n_features"] == 20


def test_a_hole_is_a_hole_not_a_second_field():
    """An irrigation pond cut out of a field must not inflate the acreage.

    Outer rings are clockwise by ESRI convention and holes are anticlockwise, so
    a reader that treats every ring as a separate polygon would bill the grower
    for water on the pond.
    """
    outer = [[-75.52, 38.64], [-75.52, 38.65], [-75.51, 38.65],
             [-75.51, 38.64], [-75.52, 38.64]]                 # clockwise
    hole = [[-75.517, 38.643], [-75.515, 38.643], [-75.515, 38.645],
            [-75.517, 38.645], [-75.517, 38.643]]              # anticlockwise
    shp = polygon_shp([[outer, hole]])
    parsed = ingest.read_shapefile(shp, prj=PRJ_WGS84)
    geom = parsed["features"][0]["geometry"]
    assert geom["type"] == "Polygon"
    assert len(geom["coordinates"]) == 2, "the hole became its own polygon"


def test_two_separate_fields_come_back_as_two_polygons():
    a = [[-75.52, 38.64], [-75.52, 38.65], [-75.51, 38.65], [-75.52, 38.64]]
    b = [[-75.40, 38.64], [-75.40, 38.65], [-75.39, 38.65], [-75.40, 38.64]]
    parsed = ingest.read_shapefile(polygon_shp([[a], [b]]))
    assert parsed["n_features"] == 2


# --- projection ------------------------------------------------------------

def test_utm_coordinates_are_converted_and_the_conversion_is_reported():
    """A UTM export must land in Delaware, not in the Gulf of Guinea."""
    eastings = [(454_000.0 + i * 30, 4_277_000.0 + i * 30) for i in range(10)]
    members = {"f.shp": point_shp(eastings), "f.prj": PRJ_UTM18}
    parsed = ingest.read_shapefile_zip(zip_it(members))
    lon, lat = parsed["features"][0]["geometry"]["coordinates"]
    assert -76.0 < lon < -74.9, f"longitude {lon} is not on Delmarva"
    assert 38.0 < lat < 40.0, f"latitude {lat} is not on Delmarva"
    assert any("UTM zone 18" in w for w in parsed["warnings"])


def test_an_unconvertible_projection_warns_instead_of_placing_the_field_wrongly():
    members = {"f.shp": point_shp([(700_000.0, 200_000.0)]),
               "f.prj": PRJ_STATEPLANE}
    parsed = ingest.read_shapefile_zip(zip_it(members))
    assert parsed["epsg"] is None
    assert any("re-export" in w for w in parsed["warnings"])


# --- attribute table -------------------------------------------------------

def test_deleted_records_are_skipped():
    rows = [{"Yld_Vol_Dr": 200.0, "Moisture": 15.0, "Variety": "A"},
            {"Yld_Vol_Dr": 999.0, "Moisture": 0.0, "Variety": "GONE",
             "_deleted": True},
            {"Yld_Vol_Dr": 210.0, "Moisture": 15.0, "Variety": "B"}]
    recs = ingest.read_dbf(dbf(rows, YIELD_FIELDS))
    assert len(recs) == 2
    assert [r["Variety"] for r in recs] == ["A", "B"]


def test_typed_columns_come_back_typed():
    fields = [("YIELD", b"N", 10, 2), ("IRRIGATED", b"L", 1, 0),
              ("HARVESTED", b"D", 8, 0), ("HYBRID", b"C", 10, 0)]
    rows = [{"YIELD": 212.5, "IRRIGATED": True, "HARVESTED": "20251014",
             "HYBRID": "P1197AM"}]
    r = ingest.read_dbf(dbf(rows, fields))[0]
    assert r["YIELD"] == pytest.approx(212.5)
    assert r["IRRIGATED"] is True
    assert r["HARVESTED"] == "2025-10-14"
    assert r["HYBRID"] == "P1197AM"


def test_a_short_dbf_is_reported_rather_than_silently_truncating():
    """A partial copy off a USB stick must not become a quiet zone map."""
    pts = [(-75.52 + i * 0.0004, 38.64) for i in range(30)]
    rows = [{"Yld_Vol_Dr": 190.0, "Moisture": 15.0, "Variety": "A"}
            for _ in range(12)]
    parsed = ingest.read_shapefile(point_shp(pts), dbf(rows, YIELD_FIELDS))
    assert any("30 shapes" in w and "12" in w for w in parsed["warnings"])


# --- failure modes the grower has to be told about -------------------------

def test_missing_dbf_still_returns_geometry_but_says_the_yield_is_gone():
    parsed = ingest.read_shapefile_zip(yield_zip(10, with_dbf=False))
    assert parsed["n_features"] == 10
    assert any(".dbf" in w for w in parsed["warnings"])


def test_zip_without_a_shp_is_refused_with_actionable_text():
    with pytest.raises(ingest.ShapefileError) as e:
        ingest.read_shapefile_zip(zip_it({"readme.txt": b"notes"}))
    assert "export the layer again" in str(e.value).lower()


def test_a_file_that_is_not_a_zip_is_refused():
    with pytest.raises(ingest.ShapefileError):
        ingest.read_shapefile_zip(b"this is not a zip archive at all")


def test_a_dbf_uploaded_as_the_shp_is_caught_by_the_magic_number():
    with pytest.raises(ingest.ShapefileError) as e:
        ingest.read_shp_geometry(dbf([], YIELD_FIELDS) + b"\x00" * 200)
    assert ".shx" in str(e.value) or ".dbf" in str(e.value)


def test_a_truncated_shp_keeps_the_records_that_did_parse():
    pts = [(-75.52 + i * 0.001, 38.64) for i in range(20)]
    full = point_shp(pts)
    geoms = ingest.read_shp_geometry(full[:100 + 10 * 28])
    assert 0 < len(geoms) < 20


# --- the whole point: a yield map --------------------------------------------

def test_yield_zip_becomes_the_same_statistics_as_a_raster():
    stats = ingest.yield_from_shapefile_zip(yield_zip(120))
    assert stats["n"] == 120
    assert 170 < stats["mean"] < 220
    assert stats["cv"] > 0
    assert "Yld_Vol_Dr" in stats["fields_available"]


def test_an_unrecognised_yield_column_lists_what_is_available():
    """Monitors name the column a dozen different ways; say what we found."""
    pts = [(-75.52 + i * 0.0004, 38.64) for i in range(15)]
    fields = [("WEIRD_NAME", b"N", 10, 2)]
    rows = [{"WEIRD_NAME": 200.0} for _ in range(15)]
    data = zip_it({"f.shp": point_shp(pts), "f.dbf": dbf(rows, fields)})
    stats = ingest.yield_from_shapefile_zip(data)
    assert stats["n"] == 0
    assert "WEIRD_NAME" in stats["note"]


def test_naming_the_column_explicitly_makes_it_work():
    pts = [(-75.52 + i * 0.0004, 38.64) for i in range(15)]
    fields = [("WEIRD_NAME", b"N", 10, 2)]
    rows = [{"WEIRD_NAME": 200.0 + i} for i in range(15)]
    data = zip_it({"f.shp": point_shp(pts), "f.dbf": dbf(rows, fields)})
    stats = ingest.yield_from_shapefile_zip(data, value_key="WEIRD_NAME")
    assert stats["n"] == 15


def test_a_prj_that_contradicts_the_coordinates_is_not_believed():
    """A projected .prj beside geographic coordinates must not be trusted.

    Export tools write this combination routinely. Reprojecting degrees as
    though they were metres drops the field in the Atlantic, so the coordinate
    magnitudes settle it and the mismatch is reported.
    """
    degrees = [(-75.52 + i * 0.001, 38.64 + i * 0.001) for i in range(8)]
    members = {"f.shp": point_shp(degrees), "f.prj": PRJ_UTM18}
    parsed = ingest.read_shapefile_zip(zip_it(members))

    lon, lat = parsed["features"][0]["geometry"]["coordinates"]
    assert lon == pytest.approx(-75.52), "the coordinates were reprojected anyway"
    assert lat == pytest.approx(38.64)
    assert parsed["epsg"] == 4326
    assert any("already latitude and longitude" in w for w in parsed["warnings"])


def test_the_bundled_wrong_projection_sample_still_lands_on_delmarva():
    """samples/wrong_projection.zip is deliberately mislabelled."""
    import pathlib
    sample = (pathlib.Path(__file__).resolve().parents[2]
              / "samples" / "wrong_projection.zip")
    if not sample.exists():
        pytest.skip("sample archive not generated")
    parsed = ingest.read_shapefile_zip(sample.read_bytes())
    for feat in parsed["features"]:
        for lon, lat in ingest._iter_coords(feat["geometry"]):
            assert -76.5 < lon < -74.5, f"longitude {lon} is off Delmarva"
            assert 38.0 < lat < 40.0, f"latitude {lat} is off Delmarva"
