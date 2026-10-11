"""
Tests for field boundary handling.

Same principle as test_engine.py: these are assertions about behaviour a grower
would notice, not about internal representation. A boundary that measures 132
acres when the pivot is 132 acres, a soil split that gets recommended when the
field really is split and not when it only looks that way on the map.
"""

from __future__ import annotations

import math

import pytest

from app.engine import geo
from app.engine.soils import SOILS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GEORGETOWN = (-75.4664, 38.6431)


def circle_ring(acres: float, center=GEORGETOWN, n: int = 240) -> list[list[float]]:
    """A pivot circle of a known size, in lon/lat."""
    lon0, lat0 = center
    r_m = math.sqrt(acres * geo.SQM_PER_ACRE / math.pi)
    ring = []
    for i in range(n):
        a = 2 * math.pi * i / n
        dy = r_m * math.cos(a) / 110_540.0
        dx = r_m * math.sin(a) / (111_320.0 * math.cos(math.radians(lat0)))
        ring.append([lon0 + dx, lat0 + dy])
    return ring


def feature(ring) -> dict:
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def test_pivot_circle_measures_its_own_acres():
    b = geo.parse_geojson(feature(circle_ring(132.0)))
    assert geo.boundary_acres(b) == pytest.approx(132.0, rel=0.01)


def test_square_mile_section_is_640_acres():
    # A section is the unit every farm map in the country is built on. If this
    # is wrong, everything downstream is wrong by the same factor.
    lat0 = 38.9
    side_deg_lat = 1609.344 / 110_540.0
    side_deg_lon = 1609.344 / (111_320.0 * math.cos(math.radians(lat0)))
    ring = [[-75.6, lat0], [-75.6 + side_deg_lon, lat0],
            [-75.6 + side_deg_lon, lat0 + side_deg_lat], [-75.6, lat0 + side_deg_lat]]
    b = geo.parse_geojson(feature(ring))
    assert geo.boundary_acres(b) == pytest.approx(640.0, rel=0.005)


def test_area_is_orientation_independent():
    ring = circle_ring(80.0, n=64)
    cw = geo.parse_geojson(feature(list(reversed(ring))))
    ccw = geo.parse_geojson(feature(ring))
    assert geo.boundary_acres(cw) == pytest.approx(geo.boundary_acres(ccw), rel=1e-9)


def test_hole_is_subtracted():
    outer = circle_ring(100.0, n=120)
    inner = circle_ring(10.0, n=120)
    gj = {"type": "Polygon", "coordinates": [outer, inner]}
    b = geo.parse_geojson(gj)
    assert geo.boundary_acres(b) == pytest.approx(90.0, rel=0.02)


def test_centroid_lands_in_the_middle_of_a_pivot():
    b = geo.parse_geojson(feature(circle_ring(132.0)))
    lon, lat = geo.boundary_centroid(b)
    assert lon == pytest.approx(GEORGETOWN[0], abs=1e-4)
    assert lat == pytest.approx(GEORGETOWN[1], abs=1e-4)


def test_centroid_of_split_field_uses_the_larger_lobe():
    # Two lobes either side of a ditch. Averaging both puts the point in the
    # ditch, which is not where the crop is.
    big = circle_ring(120.0, center=(-75.50, 38.60), n=64)
    small = circle_ring(20.0, center=(-75.44, 38.60), n=64)
    gj = {"type": "MultiPolygon", "coordinates": [[big], [small]]}
    b = geo.parse_geojson(gj)
    lon, _ = geo.boundary_centroid(b)
    assert lon == pytest.approx(-75.50, abs=1e-3)


# ---------------------------------------------------------------------------
# parsing the files growers actually have
# ---------------------------------------------------------------------------

def test_accepts_featurecollection_geometry_and_bare_feature():
    ring = circle_ring(50.0, n=48)
    shapes = [
        {"type": "Polygon", "coordinates": [ring]},
        feature(ring),
        {"type": "FeatureCollection", "features": [feature(ring)]},
    ]
    acres = [geo.boundary_acres(geo.parse_geojson(s)) for s in shapes]
    assert all(a == pytest.approx(acres[0], rel=1e-9) for a in acres)


def test_point_file_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="boundary export"):
        geo.parse_geojson({"type": "Point", "coordinates": [-75.5, 38.6]})


def test_kml_polygon_parses_with_altitude_triples():
    kml = (
        "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
        "-75.500,38.600,0 -75.490,38.600,0 -75.490,38.610,0 -75.500,38.610,0 "
        "-75.500,38.600,0"
        "</coordinates></LinearRing></outerBoundaryIs></Polygon>"
    )
    b = geo.parse_kml(kml)
    assert len(b.polygons) == 1
    assert geo.boundary_acres(b) > 200


def test_duplicate_closing_vertex_does_not_double_count():
    ring = circle_ring(64.0, n=40)
    closed = ring + [ring[0]]
    a1 = geo.boundary_acres(geo.parse_geojson(feature(ring)))
    a2 = geo.boundary_acres(geo.parse_geojson(feature(closed)))
    assert a1 == pytest.approx(a2, rel=1e-9)


# ---------------------------------------------------------------------------
# simplification
# ---------------------------------------------------------------------------

def test_simplify_cuts_vertices_without_moving_the_acres():
    b = geo.parse_geojson(feature(circle_ring(132.0, n=2000)))
    s = geo.simplify_boundary(b)
    assert s.vertex_count < 200
    assert geo.boundary_acres(s) == pytest.approx(geo.boundary_acres(b), rel=0.01)


def test_simplify_leaves_a_simple_rectangle_alone():
    ring = [[-75.6, 38.9], [-75.59, 38.9], [-75.59, 38.91], [-75.6, 38.91]]
    b = geo.parse_geojson(feature(ring))
    assert geo.simplify_boundary(b).vertex_count == 4


def test_simplify_respects_the_vertex_ceiling():
    b = geo.parse_geojson(feature(circle_ring(400.0, n=5000)))
    assert geo.simplify_boundary(b, max_vertices=120).vertex_count <= 120


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_boundary_outside_delmarva_is_flagged():
    # The classic failure: a shapefile in state plane feet read as degrees, or a
    # field in Iowa. Either way DEOS is the wrong weather source.
    b = geo.parse_geojson(feature(circle_ring(100.0, center=(-93.6, 41.6), n=40)))
    assert any("Delmarva" in m for m in geo.validate(b))


def test_absurdly_large_boundary_is_flagged():
    b = geo.parse_geojson(feature(circle_ring(5000.0, n=60)))
    assert any("larger than" in m for m in geo.validate(b))


def test_a_normal_delaware_pivot_produces_no_warnings():
    b = geo.parse_geojson(feature(circle_ring(132.0, n=64)))
    assert geo.validate(b) == []


def test_multipolygon_tells_the_grower_it_was_merged():
    a = circle_ring(60.0, center=(-75.50, 38.60), n=32)
    c = circle_ring(60.0, center=(-75.44, 38.60), n=32)
    b = geo.parse_geojson({"type": "MultiPolygon", "coordinates": [[a], [c]]})
    assert any("separate polygons" in m for m in geo.validate(b))


# ---------------------------------------------------------------------------
# sampling and WKT
# ---------------------------------------------------------------------------

def test_every_sample_point_falls_inside_the_field():
    b = geo.parse_geojson(feature(circle_ring(132.0, n=120)))
    pts = geo.sample_points(b, 32)
    assert len(pts) >= 12
    assert all(geo.point_in_boundary(p, b) for p in pts)


def test_sampling_is_deterministic():
    b = geo.parse_geojson(feature(circle_ring(90.0, n=80)))
    assert geo.sample_points(b, 24) == geo.sample_points(b, 24)


def test_sample_points_avoid_a_hole():
    outer = circle_ring(200.0, n=160)
    inner = circle_ring(60.0, n=160)
    b = geo.parse_geojson({"type": "Polygon", "coordinates": [outer, inner]})
    hole = geo.Polygon(outer=[(x, y) for x, y in inner])
    assert not any(geo.point_in_polygon(p, hole) for p in geo.sample_points(b, 40))


def test_wkt_closes_its_rings():
    b = geo.parse_geojson(feature(circle_ring(40.0, n=8)))
    wkt = geo.to_wkt(b)
    assert wkt.startswith("polygon((")
    body = wkt[wkt.index("((") + 2: wkt.rindex("))")]
    pts = [p.strip() for p in body.split(",")]
    assert pts[0] == pts[-1]


# ---------------------------------------------------------------------------
# matching SSURGO names to the local table
# ---------------------------------------------------------------------------

def test_exact_series_name_matches_itself():
    key, how = geo.match_series("Evesboro loamy sand, 0 to 5 percent slopes")
    assert (key, how) == ("evesboro", "exact")


def test_known_delmarva_series_maps_to_a_local_analogue():
    key, how = geo.match_series("Galestown sand")
    assert how == "alias"
    assert SOILS[key].awc_in_per_in <= 0.09  # a droughty sand stays droughty


def test_unknown_series_falls_back_to_water_holding_capacity():
    key, how = geo.match_series("Nonesuch silt loam", awc_in_per_in=0.19)
    assert how == "awc"
    assert SOILS[key].awc_in_per_in == pytest.approx(0.19, abs=0.02)


def test_wet_series_maps_to_something_with_a_water_table():
    key, _ = geo.match_series("Berryland mucky sand")
    assert SOILS[key].water_table_depth_in is not None


# ---------------------------------------------------------------------------
# zone recommendations
# ---------------------------------------------------------------------------

def _c(key, frac):
    return {"series_key": key, "fraction": frac,
            "awc_in_per_in": SOILS[key].awc_in_per_in}


def test_split_recommended_when_sand_meets_a_wet_soil():
    z = geo.suggest_zones([_c("evesboro", 0.55), _c("fallsington", 0.45)], 132.0)
    assert z["recommend"]
    assert sum(x["acres"] for x in z["zones"]) == pytest.approx(132.0, rel=0.02)


def test_no_split_when_the_soils_behave_alike():
    # Sassafras and Ingleside look different on the map and irrigate the same.
    z = geo.suggest_zones([_c("sassafras", 0.6), _c("ingleside", 0.4)], 100.0)
    assert not z["recommend"]
    assert z["reason"] == "soils_behave_alike"


def test_no_split_when_one_soil_dominates():
    z = geo.suggest_zones([_c("evesboro", 0.88), _c("matapeake", 0.12)], 110.0)
    assert not z["recommend"]


def test_minor_inclusions_are_ignored():
    comps = [_c("sassafras", 0.70), _c("evesboro", 0.08),
             _c("pocomoke", 0.07), _c("matapeake", 0.15)]
    z = geo.suggest_zones(comps, 100.0)
    assert all(x["series_key"] != "evesboro" for x in z.get("zones", []))


def test_droughty_zone_runs_out_sooner_than_the_heavy_one():
    z = geo.suggest_zones([_c("evesboro", 0.5), _c("matapeake", 0.5)], 120.0)
    by_key = {x["series_key"]: x for x in z["zones"]}
    assert by_key["evesboro"]["days_to_mad_at_peak"] < by_key["matapeake"]["days_to_mad_at_peak"]


def test_shallow_water_table_zone_gets_the_over_irrigation_warning():
    z = geo.suggest_zones([_c("pocomoke", 0.45), _c("evesboro", 0.55)], 90.0)
    notes = " ".join(x["note"] for x in z["zones"])
    assert "over-irrigate" in notes


# ---------------------------------------------------------------------------
# blending
# ---------------------------------------------------------------------------

def test_blend_sits_between_its_components():
    blended = geo.blended_from_components([_c("evesboro", 0.5), _c("matapeake", 0.5)])
    assert SOILS["evesboro"].awc_in_per_in < blended.awc_in_per_in < SOILS["matapeake"].awc_in_per_in


def test_blend_keeps_the_shallowest_water_table():
    # The wet corner is what surprises the model, so it must survive the average.
    blended = geo.blended_from_components([_c("evesboro", 0.8), _c("pocomoke", 0.2)])
    assert blended.water_table_depth_in == SOILS["pocomoke"].water_table_depth_in


def test_blend_of_one_component_is_that_component():
    blended = geo.blended_from_components([_c("sassafras", 1.0)])
    assert blended.awc_in_per_in == pytest.approx(SOILS["sassafras"].awc_in_per_in)
