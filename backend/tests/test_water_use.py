"""
The satellite water use panel: what it draws from a METRIC season.

The energy balance is tested in test_metric.py. This is about the other half
of the problem, which is presenting a measurement with real uncertainty in it
without either hiding the uncertainty or burying the number. Three things in
particular are checked, because each of them has been got wrong in a published
ET product at some point: that the geometry sent to the server is a closed
ring, that a satellite date is printed as the day it happened rather than the
evening before, and that the two calibration pixels the whole scene rests on
are visible to whoever is reading the map.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

# One pass, as /api/metric/season returns it. Small enough to read, with a
# masked corner and the two anchors in known cells.
SCENE = {
    "date": "2026-07-18",
    "platform": "landsat-9",
    "scene_cloud_pct": 11.0,
    "field_clear_fraction": 0.89,
    "clear_pixels": 14,
    "mean_eta_mm_day": 6.4,
    "mean_eta_in_day": 0.252,
    "mean_etrf": 1.08,
    "cold_anchor_c": 26.2,
    "hot_anchor_c": 43.1,
    "cold_anchor_reason": ("CIMEC search. NDVI 0.88, albedo 0.185, 26.2 C. "
                           "Anchored at 0.97 of alfalfa reference ET "
                           "(Allen 2013 Eq 4)."),
    "hot_anchor_reason": ("CIMEC search. NDVI 0.14, albedo 0.236, 43.1 C. "
                          "Anchored at 0.064 of alfalfa reference ET "
                          "(Allen 2013 Eq 5) with bare-soil evaporation at 0.05."),
    "calibration": {
        "a": -27.43, "b": 0.1008, "rho": 1.1514, "c_p": 1017.6, "u200": 6.29,
        "dt_cold": 2.835, "dt_hot": 4.953, "reference": "grass",
        "reference_et_hourly_mm": 0.52, "reference_et_source": "station",
        "le_reference_w_m2": 421.2, "endmember_search": "CIMEC",
        "etrf_bare": 0.05, "cold_anchor_fraction": 0.97,
        "hot_anchor_fraction": 0.064, "anchor_iterations": [5, 4],
        "rah_cold": 15.91, "rah_hot": 18.3,
    },
    "raster": {
        "rows": 4, "cols": 4,
        "eta_mm_day": [
            [7.1, 7.0, 6.8, 6.6],
            [6.9, 6.7, 6.4, 6.0],
            [6.2, 5.8, 5.1, None],
            [4.4, 3.1, None, None],
        ],
        "etrf": [[1.20, 1.19, 1.15, 1.12],
                 [1.17, 1.14, 1.08, 1.02],
                 [1.05, 0.98, 0.86, None],
                 [0.75, 0.53, None, None]],
        "min_mm_day": 3.1, "max_mm_day": 7.1,
        "cold_cell": [0, 0], "hot_cell": [3, 1],
    },
    "bbox": [-75.47, 38.63, -75.45, 38.65],
    "notes": [],
}

RESULT = {
    "ok": True,
    "planting": "2026-05-03",
    "through": "2026-09-20",
    "harvest": None,
    "weather_source": "DEOS Georgetown",
    "total_eta_in": 22.41,
    "total_eta_mm": 569.2,
    "days": 141,
    "gap_filling": "ETrF interpolation",
    "scene_count": 2,
    "weekly": [
        {"week_start": "2026-05-03", "week_end": "2026-05-09", "iso_week": "2026-W18",
         "days": 7, "mean_eta_mm_day": 1.9, "mean_eta_in_day": 0.075,
         "total_eta_mm": 13.3, "total_eta_in": 0.52},
        {"week_start": "2026-07-13", "week_end": "2026-07-19", "iso_week": "2026-W29",
         "days": 7, "mean_eta_mm_day": 6.4, "mean_eta_in_day": 0.252,
         "total_eta_mm": 44.8, "total_eta_in": 1.76},
        {"week_start": "2026-09-14", "week_end": "2026-09-20", "iso_week": "2026-W38",
         "days": 7, "mean_eta_mm_day": 2.6, "mean_eta_in_day": 0.102,
         "total_eta_mm": 18.2, "total_eta_in": 0.72},
    ],
    "scenes": [dict(SCENE, date="2026-06-16", mean_eta_in_day=0.188,
                    mean_eta_mm_day=4.8), SCENE],
    "warnings": ["The first usable scene is 44 days after planting. Early-season "
                 "ET before it is extrapolated from that scene and is the least "
                 "reliable part of the total."],
    "method": "METRIC surface energy balance on Landsat 8 and 9 Collection 2 Level-2.",
}


def run(tmp_path, body):
    script = tmp_path / "t.mjs"
    script.write_text(
        'import { JSDOM } from "jsdom"; import fs from "fs";\n'
        f'const dom=new JSDOM(fs.readFileSync({str(INDEX)!r},"utf8"),\n'
        ' {runScripts:"dangerously",pretendToBeVisual:true,url:"http://localhost/",\n'
        '  beforeParse(w){ w.fetch=async()=>({ok:false,status:503,json:async()=>({})}); }});\n'
        'const w=dom.window, d=w.document;\n'
        'await new Promise(r=>w.addEventListener("load",r));\n'
        'await new Promise(r=>setTimeout(r,250));\n'
        f'const RESULT={json.dumps(RESULT)};\n'
        f'const SCENE=RESULT.scenes[1];\n'
        + body + "\n")
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=180, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# the request
# --------------------------------------------------------------------------

def test_the_geometry_sent_to_the_server_has_closed_rings(tmp_path):
    """An open ring is read as an open polygon and the crop returns nothing.

    No error, no warning, just an empty raster and a reader wondering why the
    satellite saw nothing. Closing the ring is one line and this is the test
    that keeps it there.
    """
    r = run(tmp_path, '''
const b=[{outer:[[-75.47,38.63],[-75.45,38.63],[-75.45,38.65],[-75.47,38.65]],
          holes:[[[-75.465,38.635],[-75.460,38.635],[-75.460,38.640]]]}];
const g=w.eval("boundaryGeoJSON")(b);
console.log(JSON.stringify({type:g.type, outer:g.coordinates[0],
  hole:g.coordinates[1], rings:g.coordinates.length}));''')
    assert r["type"] == "Polygon"
    assert r["outer"][0] == r["outer"][-1], "the outer ring is not closed"
    assert len(r["outer"]) == 5, "closing should add one coordinate, not more"
    assert r["hole"][0] == r["hole"][-1], "the hole is not closed either"
    assert r["rings"] == 2


def test_a_hole_goes_inside_its_polygon_not_beside_it(tmp_path):
    """A MultiPolygon is a list of polygons, each a list of rings. Put a hole
    one level too high and it becomes a second field, which in a pivot with a
    wet spot excluded means the excluded part gets counted."""
    r = run(tmp_path, '''
const b=[{outer:[[0,0],[1,0],[1,1],[0,1]], holes:[[[0.2,0.2],[0.4,0.2],[0.4,0.4]]]},
         {outer:[[2,2],[3,2],[3,3],[2,3]], holes:[]}];
const g=w.eval("boundaryGeoJSON")(b);
console.log(JSON.stringify({type:g.type, polys:g.coordinates.length,
  firstRings:g.coordinates[0].length, secondRings:g.coordinates[1].length}));''')
    assert r["type"] == "MultiPolygon"
    assert r["polys"] == 2
    assert r["firstRings"] == 2, "the hole belongs to the first polygon"
    assert r["secondRings"] == 1


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------

def test_a_satellite_date_prints_as_the_day_it_happened(tmp_path):
    """`new Date("2026-07-18")` is UTC midnight, which in Delaware is the
    evening of the 17th. A pass labelled a day early is not cosmetic: it is
    the difference between a reading before and after an irrigation set."""
    r = run(tmp_path, '''
console.log(JSON.stringify({
  jul: w.eval("fmtDateISO")("2026-07-18"),
  jan: w.eval("fmtDateISO")("2026-01-01"),
  empty: w.eval("fmtDateISO")(""),
  stamped: w.eval("fmtDateISO")("2026-07-18T15:42:00Z")}));''')
    assert r["jul"] == "Jul 18"
    assert r["jan"] == "Jan 1", "a new year's day pass must not print as Dec 31"
    assert r["empty"] == ""
    assert r["stamped"] == "Jul 18", "a full timestamp should work the same way"


# --------------------------------------------------------------------------
# the map
# --------------------------------------------------------------------------

def test_the_map_marks_both_calibration_pixels(tmp_path):
    """The whole scene is read off a line through two pixels. A reader who can
    see where they landed can tell a pond or a farm road from a crop; one who
    cannot is taking the number on faith."""
    r = run(tmp_path, '''
const html=w.eval("wuMapHTML")(RESULT.scenes, 1, {});
console.log(JSON.stringify({
  cold: html.includes(">cold<"), hot: html.includes(">hot<"),
  circles: (html.match(/<circle/g)||[]).length,
  cells: (html.match(/<rect /g)||[]).length,
  explains: /pond|farm road|poultry/.test(html)}));''')
    assert r["cold"] and r["hot"]
    assert r["circles"] == 2
    assert r["cells"] >= 16, "every cell in a 4 by 4 raster should be drawn"
    assert r["explains"], (
        "marking the anchors is only useful alongside what a bad one looks like")


def test_masked_pixels_are_drawn_as_absent_rather_than_as_zero(tmp_path):
    """Cloud is missing data. Filling it with zero says the crop used no water
    there, which is the one reading that must not be possible."""
    r = run(tmp_path, '''
const html=w.eval("wuMapHTML")(RESULT.scenes, 1, {});
console.log(JSON.stringify({
  pattern: html.includes("url(#wumask)"),
  defined: html.includes('id="wumask"'),
  masked: (html.match(/url\\(#wumask\\)/g)||[]).length,
  says: /masked/.test(html)}));''')
    assert r["pattern"] and r["defined"]
    assert r["masked"] == 3, "three cells are null in the fixture"
    assert r["says"], "hovering a blank cell should say why it is blank"


def test_the_colour_ramp_spans_the_scene_rather_than_a_fixed_scale(tmp_path):
    """A fixed scale makes every mid-season Delaware scene look the same
    shade. The question a reader asks of an ET map is which part of this field
    is below the rest, which needs the ramp stretched over this scene."""
    r = run(tmp_path, '''
const html=w.eval("wuMapHTML")(RESULT.scenes, 1, {});
console.log(JSON.stringify({
  lo: html.includes((3.1/25.4).toFixed(3)),
  hi: html.includes((7.1/25.4).toFixed(3)),
  shades: new Set(html.match(/rgb\\(\\d+,\\d+,\\d+\\)/g)||[]).size}));''')
    assert r["lo"] and r["hi"], "the legend should carry the scene's own range"
    assert r["shades"] >= 8, (
        "a 4 by 4 raster spanning 3.1 to 7.1 mm/day should produce a spread of "
        "colours, not two or three steps")


def test_the_ramp_interpolates_and_tolerates_nonsense(tmp_path):
    r = run(tmp_path, '''
const f=w.eval("rampAt"), R=w.eval("RAMP_ET");
console.log(JSON.stringify({
  lo:f(R,0), hi:f(R,1), mid:f(R,0.5),
  under:f(R,-3), over:f(R,9), nan:f(R,NaN)}));''')
    assert r["lo"] != r["hi"] != r["mid"]
    assert r["under"] == r["lo"] and r["over"] == r["hi"], (
        "a value off the end of the ramp should clamp, not wrap or throw")
    assert r["nan"].startswith("#"), "a missing value needs a defined colour"


# --------------------------------------------------------------------------
# the weekly figure
# --------------------------------------------------------------------------

def test_the_weekly_figure_shows_the_rate_and_the_running_total(tmp_path):
    """Two different questions, both asked of the same chart: how fast is it
    using water now, and how much has the season cost so far."""
    r = run(tmp_path, '''
const html=w.eval("wuWeeklyHTML")(RESULT);
console.log(JSON.stringify({
  bars:(html.match(/<rect /g)||[]).length,
  curve:(html.match(/<path /g)||[]).length,
  leftAxis:/Weekly average, in\\/day/.test(html),
  rightAxis:/Cumulative, in/.test(html),
  peakNamed:/Jul 13/.test(html),
  rows:(html.match(/<tr>/g)||[]).length}));''')
    assert r["bars"] == 3, "one bar per week in the fixture"
    assert r["curve"] == 1
    assert r["leftAxis"] and r["rightAxis"], (
        "two axes carrying different units must both be labelled")
    assert r["peakNamed"], "name the peak week rather than leaving it to be read off"
    assert r["rows"] == 4, "three weeks plus a header"


def test_the_running_total_is_cumulative_and_not_restated_per_week(tmp_path):
    """0.52, then 2.28, then 3.00 inches. A column that repeats the weekly
    figure under a 'running total' heading is worse than no column."""
    r = run(tmp_path, '''
const html=w.eval("wuWeeklyHTML")(RESULT);
const nums=[...html.matchAll(/>([\\d.]+) in<\\/td>/g)].map(m=>parseFloat(m[1]));
console.log(JSON.stringify({nums}));''')
    nums = r["nums"]
    # Each row prints "that week" then "running total", so the running values
    # are every second number.
    running = nums[1::2]
    assert running == sorted(running), f"the running total goes backwards: {running}"
    assert running[-1] == pytest.approx(3.00, abs=0.02)


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------

def test_the_calibration_is_shown_term_by_term(tmp_path):
    """METRIC calibrates itself inside each scene, which is both why it is
    robust and why it can be wrong in a way nothing downstream catches. The
    terms are laid out so the result can be argued with."""
    r = run(tmp_path, '''
const html=w.eval("wuAuditHTML")(RESULT.scenes, 1);
console.log(JSON.stringify({
  slope:/0\\.1008/.test(html), search:/CIMEC/.test(html),
  coldReason:/Allen 2013 Eq 4/.test(html), hotReason:/Allen 2013 Eq 5/.test(html),
  resistance:/15\\.91/.test(html), blending:/6\\.29/.test(html),
  refSource:/station/.test(html),
  weakness:/weakness/.test(html)}));''')
    for key in ("slope", "search", "coldReason", "hotReason", "resistance",
                "blending", "refSource"):
        assert r[key], f"{key} is missing from the calibration audit"
    assert r["weakness"], (
        "a self-calibrating method should say what its calibration rests on")


def test_every_pass_can_be_inspected_not_only_the_last(tmp_path):
    r = run(tmp_path, '''
const html=w.eval("wuAuditHTML")(RESULT.scenes, 0);
const map=w.eval("wuMapHTML")(RESULT.scenes, 0, {});
console.log(JSON.stringify({
  chips:(html.match(/data-scene=/g)||[]).length,
  pressed:/data-scene="0" aria-pressed="true"/.test(html),
  mapChips:(map.match(/data-scene=/g)||[]).length,
  mapDate:/Jun 16/.test(map)}));''')
    assert r["chips"] == 2 and r["mapChips"] == 2
    assert r["pressed"], "the pass being shown should read as selected"
    assert r["mapDate"]


def test_a_pass_with_no_raster_says_so_instead_of_drawing_nothing(tmp_path):
    r = run(tmp_path, '''
const scenes=[Object.assign({}, RESULT.scenes[1], {raster:{}})];
const html=w.eval("wuMapHTML")(scenes, 0, {});
console.log(JSON.stringify({warn:/note warn/.test(html),
  points:/Passes/.test(html)}));''')
    assert r["warn"]
    assert r["points"], "say where the numbers are instead, rather than only failing"
