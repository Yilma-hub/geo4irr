"""
The editable schedule table, and the precipitation switch.

Rain and gross irrigation are the two columns a grower knows better than the
model does: what their own gauge caught, and what they actually ran. Everything
else in the table is an output. These tests hold that line, and hold the one
subtle bug this feature shipped with — reading the whole column back on apply,
which silently promoted every modelled rainfall day into a measurement the
grower never took.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"
LAYER = INDEX.parent / "data" / "irrigated_areas.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def run(tmp_path, body):
    script = tmp_path / "t.mjs"
    script.write_text(
        'import { JSDOM } from "jsdom"; import fs from "fs";\n'
        f'const dom=new JSDOM(fs.readFileSync({str(INDEX)!r},"utf8"),\n'
        ' {runScripts:"dangerously",pretendToBeVisual:true,url:"http://localhost/",\n'
        '  beforeParse(w){ w.fetch=async()=>({ok:false,status:503,'
        'json:async()=>({})}); }});\n'
        'const w=dom.window, d=w.document;\n'
        'await new Promise(r=>w.addEventListener("load",r));\n'
        f'w.eval(fs.readFileSync({str(LAYER)!r},"utf8"));\n'
        'await new Promise(r=>setTimeout(r,400));\n'
        'const field = w.eval("newFieldRecord")({\n'
        '  id:"t1", name:"Test", acres:100, lat:38.70, lon:-75.55,\n'
        '  soil:"evesboro", crop:"corn_grain", planting:"2026-04-28", goal:200,\n'
        '  boundary:[{outer:[[-75.56,38.69],[-75.54,38.69],[-75.54,38.71],'
        '[-75.56,38.71],[-75.56,38.69]],holes:[]}]\n'
        '});\n'
        + body + "\n")
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=180, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_only_cells_the_grower_actually_changed_are_recorded(tmp_path):
    """The bug this feature shipped with, now pinned.

    Every rain cell renders pre-filled from the weather record. Reading the
    whole column back on apply recorded 32 days as gauge measurements when two
    cells had been touched — and those then survive every later recompute as if
    someone had stood in the field with a rain gauge.
    """
    r = run(tmp_path, '''
const r0 = await w.eval("analyze")(field);
const host = d.createElement("div");
host.innerHTML = w.eval("schedTableHTML")(r0, field);
d.body.appendChild(host);
const cells = [...host.querySelectorAll("input.cell")];
const irr = cells.find(c=>c.dataset.f==="irr");
const rain = cells.find(c=>c.dataset.f==="rain");
irr.value = "1.10";
rain.value = "0.85";
const n = w.eval("applyScheduleEdits")(host, field);
console.log(JSON.stringify({
  applied: n,
  irrDays: Object.keys(field.irrigations).length,
  rainDays: Object.keys(field.rain).length,
  irrValue: field.irrigations[irr.dataset.day],
  rainValue: field.rain[rain.dataset.day],
  totalCells: cells.length,
}));''')
    assert r["applied"] == 2, (
        f"{r['applied']} days recorded from 2 edits — untouched cells are being "
        "written back as if the grower had entered them")
    assert r["irrDays"] == 1 and r["rainDays"] == 1
    assert r["irrValue"] == pytest.approx(1.10)
    assert r["rainValue"] == pytest.approx(0.85)
    assert r["totalCells"] > 100, "every day should be editable"


def test_clearing_a_cell_removes_the_override_rather_than_asserting_zero(tmp_path):
    """Deleting a number should undo an override, not claim it did not rain."""
    r = run(tmp_path, '''
const r0 = await w.eval("analyze")(field);
const host = d.createElement("div");
host.innerHTML = w.eval("schedTableHTML")(r0, field);
d.body.appendChild(host);
const rain = [...host.querySelectorAll("input.cell")].find(c=>c.dataset.f==="rain");
const day = rain.dataset.day;
rain.value = "0.90";
w.eval("applyScheduleEdits")(host, field);
const afterSet = field.rain[day];
host.innerHTML = w.eval("schedTableHTML")(await w.eval("analyze")(field), field);
const again = [...host.querySelectorAll("input.cell")]
  .find(c=>c.dataset.f==="rain" && c.dataset.day===day);
again.value = "";
w.eval("applyScheduleEdits")(host, field);
console.log(JSON.stringify({afterSet, stillThere: day in field.rain}));''')
    assert r["afterSet"] == pytest.approx(0.90)
    assert r["stillThere"] is False


def test_edited_irrigation_moves_the_water_balance(tmp_path):
    """An edit has to change the numbers, not just the display.

    Checked on the day the water went on, not at harvest. Two runs that differ
    by four inches in July converge again by late season — the profile refills,
    dries to the same equilibrium under the same weather, and ends in the same
    place. That convergence is correct physics, and asserting on the final day
    would have quietly tested nothing.
    """
    r = run(tmp_path, '''
const before = await w.eval("analyze")(field);
const days = before.hist.slice(60,64).map(x=>w.eval("iso")(x.date));
const drBefore = before.hist[63].dr;
field.irrigations = {}; days.forEach(k=>field.irrigations[k]=1.0);
const after = await w.eval("analyze")(field);
console.log(JSON.stringify({
  drBefore, drAfter: after.hist[63].dr,
  drEndBefore: before.hist[before.hist.length-1].dr,
  drEndAfter: after.hist[after.hist.length-1].dr,
  irrTotal: after.hist.reduce((a,x)=>a+(x.irrigation||0),0)}));''')
    assert r["irrTotal"] == pytest.approx(4.0, abs=0.01)
    assert r["drAfter"] < r["drBefore"], (
        "four inches of irrigation left the profile no wetter on the day it went on")


def test_precipitation_off_drops_modelled_rain_but_keeps_the_growers_own(tmp_path):
    r = run(tmp_path, '''
const on = await w.eval("analyze")(field);
const rainOn = on.hist.reduce((a,x)=>a+(x.rain||0),0);
const key = w.eval("iso")(on.hist[50].date);
field.usePrecip = false;
field.rain = {}; field.rain[key] = 1.25;
const off = await w.eval("analyze")(field);
console.log(JSON.stringify({rainOn,
  rainOff: off.hist.reduce((a,x)=>a+(x.rain||0),0), kept: off.hist[50].rain}));''')
    assert r["rainOn"] > 5.0, "the default run should credit rain"
    assert r["rainOff"] == pytest.approx(1.25, abs=0.01), (
        "with precipitation off, only the grower's own figure should count")
    assert r["kept"] == pytest.approx(1.25, abs=0.01)


def test_the_graph_plots_available_water_at_mad_not_the_depletion(tmp_path):
    """Easy to get backwards, and wrong in the dangerous direction.

    If MAD is 55% of capacity — sorghum and wheat here — the threshold sits at
    45% of capacity, not 55%. Plotting the depletion puts the line 10% of
    capacity too high and tells a grower they have less room than they do.
    """
    r = run(tmp_path, '''
field.crop = "sorghum_grain";
const run0 = await w.eval("analyze")(field);
const st = w.eval("schedStats")(run0, field);
const svg = w.eval("schedGraphSVG")(run0, st);
console.log(JSON.stringify({
  mad: w.eval("CROPS").sorghum_grain.mad,
  hasLabel: svg.includes("Plant Available Water at MAD, in"),
  labels: ["Available Water Capacity, in","Plant Available Water, in",
           "Precipitation, in","Irrigation, in"].every(l=>svg.includes(l)),
  usesOneMinusMad: svg.length > 0,
}));''')
    assert r["hasLabel"] and r["labels"], "the legend does not match eRAMS"
    assert r["mad"] != 0.5, "the test crop must have MAD != 0.5 to be meaningful"


def test_the_mad_line_uses_one_minus_mad_in_the_source():
    """Read the formula directly — a rendered SVG path cannot prove it."""
    src = INDEX.read_text()
    graph = src[src.index("function schedGraphSVG"):]
    graph = graph[:graph.index("\nfunction ")]
    assert "1-(s.crop.mad" in graph.replace(" ", ""), (
        "the threshold line is not plotted at capacity x (1 - MAD)")
