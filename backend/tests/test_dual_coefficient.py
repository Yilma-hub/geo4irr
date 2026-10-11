"""
The FAO-56 dual crop coefficient has to be visible, not just computed.

The narrative specifies the dual approach — basal transpiration and soil
evaporation tracked separately — and the engine has always done it. But the
schedule showed only the finished number, so an engineer reading the screen had
no way to tell whether the tool ran a dual coefficient, a single one, or a
lookup table. These tests hold the terms on screen and hold the identity
between them.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"

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
        'await new Promise(r=>setTimeout(r,250));\n'
        'const field = w.eval("newFieldRecord")({\n'
        '  id:"t1", name:"T", acres:100, lat:38.70, lon:-75.55, soil:"evesboro",\n'
        '  crop:"corn_grain", planting:"2026-04-28", goal:200,\n'
        '  boundary:[{outer:[[-75.56,38.69],[-75.54,38.69],[-75.54,38.71],'
        '[-75.56,38.71],[-75.56,38.69]],holes:[]}]});\n'
        + body + "\n")
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=120, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr[-1200:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_table_shows_every_term_of_the_dual_coefficient(tmp_path):
    r = run(tmp_path, '''
const run0 = await w.eval("analyze")(field);
const host = d.createElement("div");
host.innerHTML = w.eval("schedTableHTML")(run0, field);
console.log(JSON.stringify({
  headers: [...host.querySelectorAll("th")].map(t=>t.textContent.trim()),
}));''')
    for term in ["ETo (in)", "Kcb", "Ke", "Ks", "Crop water use (in)"]:
        assert term in r["headers"], f"{term} is not shown in the schedule table"


def test_the_reported_crop_water_use_equals_the_dual_coefficient_identity(tmp_path):
    """ETc adj = (Kcb x Ks + Ke) x ETo, on every day of the season.

    If this drifts, the columns on screen have stopped describing the number
    beside them — which is worse than not showing them at all.
    """
    r = run(tmp_path, '''
const run0 = await w.eval("analyze")(field);
let worst = 0, worstDay = null;
for (const day of run0.hist) {
  const expect = (day.kcb * day.ks + day.ke) * day.eto;
  const gap = Math.abs(expect - day.etcAdj);
  if (gap > worst) { worst = gap; worstDay = w.eval("iso")(day.date); }
}
console.log(JSON.stringify({worst, worstDay, days: run0.hist.length}));''')
    assert r["days"] > 60
    assert r["worst"] < 1e-6, (
        f"the identity breaks by {r['worst']} in on {r['worstDay']}")


def test_evaporation_and_transpiration_are_genuinely_separate(tmp_path):
    """The point of "dual".

    Early in the season Kcb is low and Ke carries most of the loss from wet bare
    ground; under a closed canopy Kcb dominates and Ke goes to nearly nothing.
    A single-coefficient model cannot produce that crossover, so its presence is
    the evidence that this one is dual.
    """
    r = run(tmp_path, '''
const run0 = await w.eval("analyze")(field);
const early = run0.hist[5], mid = run0.hist[Math.floor(run0.hist.length*0.6)];
console.log(JSON.stringify({
  earlyKcb: early.kcb, earlyKe: early.ke,
  midKcb: mid.kcb, midKe: mid.ke,
  maxKe: Math.max(...run0.hist.map(x=>x.ke)),
}));''')
    assert r["earlyKcb"] < 0.4, "basal coefficient should start low, before canopy"
    assert r["maxKe"] > 0.3, "no meaningful soil evaporation — this is not dual"
    assert r["midKcb"] > r["midKe"], (
        "transpiration should dominate under a developed canopy")


def test_the_stress_coefficient_follows_the_start_of_day_depletion(tmp_path):
    """Ks is 1.0 inside RAW and falls beyond it — but on which depletion?

    FAO-56 computes Ks from Dr at the start of the day, before that day's ET is
    taken out, while the table reports Dr at the end of the day. So a day can
    legitimately show Ks = 1.0 next to an end-of-day depletion already past RAW:
    the crop transpired freely and only crossed the threshold on the way. This
    test therefore conditions on the previous day, and only on days with no rain
    or irrigation, and no root growth — those are the days where start-of-day
    depletion really is the previous end-of-day figure. Two earlier versions of
    this test were wrong: the first compared Ks against the same day's reported
    depletion and flagged eleven correct days, the second forgot that root
    growth adds its dilution term before Ks is evaluated.
    """
    r = run(tmp_path, '''
const run0 = await w.eval("analyze")(field);
let dryStressedButFree = 0, dryComfortableButStressed = 0, checked = 0;
for (let i = 1; i < run0.hist.length; i++) {
  const day = run0.hist[i], prev = run0.hist[i-1];
  if ((day.rain || 0) > 0 || (day.irrigation || 0) > 0) continue;
  // Root growth adds a dilution term to depletion before Ks is computed, so on
  // a growing day the start-of-day figure is not simply yesterday's end-of-day.
  if (day.root > prev.root) continue;
  checked++;
  if (prev.dr > day.raw + 1e-9 && day.ks >= 1.0) dryStressedButFree++;
  if (prev.dr <= day.raw - 1e-9 && day.ks < 0.999) dryComfortableButStressed++;
}
console.log(JSON.stringify({dryStressedButFree, dryComfortableButStressed, checked}));''')
    # A 98-day run with Delaware rainfall and a growing root zone leaves about
    # two dozen days that are dry and non-growing. That is the population; a
    # threshold above it fails on arithmetic rather than on behaviour.
    assert r["checked"] >= 15, f"only {r['checked']} qualifying days — too few to judge"
    assert r["dryStressedButFree"] == 0, (
        "Ks stayed at 1 on a dry day that began past RAW")
    assert r["dryComfortableButStressed"] == 0, (
        "Ks fell below 1 on a dry day that began inside RAW")


def test_the_subscripts_render_as_letters_not_unicode_rubble():
    """Unicode has no subscript c or b, so K&#8331;&#8342; came out as "K-k"."""
    html = INDEX.read_text()
    assert "K<sub>cb</sub>" in html
    assert "&#8331;" not in html and "&#8342;" not in html, (
        "subscript entities are back; they do not render as cb")
