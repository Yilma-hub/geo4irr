"""
The browser's FAO-56 mirror against pyFAO56.

The project narrative names pyFAO56 as the reference implementation. It is the
engine of record on the API, but it is Python and cannot run in a browser, so a
grower who opens index.html directly is running the JavaScript mirror. Nobody
had ever compared those two — the earlier engine comparison was Python built-in
against pyFAO56, which is a different pair.

This pins the agreement where it is genuinely close and pins the divergence
where it is not, so neither can drift unnoticed.
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib
import shutil
import statistics
import subprocess

import pytest

from app.engine import pyfao56_bridge as pf
from app.engine.crops import get_crop
from app.engine.fao56 import WeatherDay
from app.engine.soils import get_soil

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or pf.engine_choice() != "pyfao56",
    reason="needs both node and pyfao56")


def weather():
    out, d0 = [], datetime.date(2026, 4, 28)
    for i in range(130):
        d = d0 + datetime.timedelta(days=i)
        t = math.sin((d.timetuple().tm_yday - 80) / 200 * math.pi)
        out.append({"date": d.isoformat(),
                    "tmax": round(68 + 24 * t, 2), "tmin": round(48 + 22 * t, 2),
                    "rhMax": 93.0, "rhMin": 52.0, "wind": 5.5,
                    "solar": round(12 + 16 * t, 3),
                    "rain": 1.3 if i % 9 == 0 else 0.0})
    return out


def run_js(wx, tmp_path):
    (tmp_path / "wx.json").write_text(json.dumps(wx))
    script = tmp_path / "run.mjs"
    script.write_text(f'''
import {{ JSDOM }} from "jsdom"; import fs from "fs";
const dom=new JSDOM(fs.readFileSync({str(INDEX)!r},"utf8"),
 {{runScripts:"dangerously",pretendToBeVisual:true,url:"http://localhost/",
  beforeParse(w){{ w.fetch=async()=>({{ok:false,status:503,json:async()=>({{}})}}); }}}});
const w=dom.window;
await new Promise(r=>w.addEventListener("load",r));
const wx=JSON.parse(fs.readFileSync({str(tmp_path / "wx.json")!r},"utf8"));
const out=w.eval(`(function(wx){{
  const crop=CROPS.corn_grain, soil=SOILS.evesboro;
  const st={{dr:0,de:0,root:crop.rootMin}};
  return wx.map((d,i)=>{{
    const wd={{date:new Date(d.date+"T00:00:00Z"),tmax:d.tmax,tmin:d.tmin,
      rhMax:d.rhMax,rhMin:d.rhMin,wind:d.wind,solar:d.solar,rain:d.rain}};
    const r=stepBalance(st,wd,crop,soil,i,38.64,0,soil.cn);
    return {{d:d.date,eto:r.eto,kcb:r.kcb,etc:r.etcAdj,dr:r.dr,taw:r.taw}};
  }});
}})`)(wx);
console.log(JSON.stringify(out));
''')
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=180, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_py(wx):
    days = [WeatherDay(day=datetime.date.fromisoformat(d["date"]),
                       tmax_f=d["tmax"], tmin_f=d["tmin"], rh_max=d["rhMax"],
                       rh_min=d["rhMin"], wind_mph=d["wind"],
                       solar_rad_mj_m2=d["solar"], rain_in=d["rain"]) for d in wx]
    run = pf.run_season_pyfao56(weather=days, crop=get_crop("corn_grain"),
                                soil=get_soil("evesboro"),
                                planting=datetime.date(2026, 4, 28), lat_deg=38.64)
    assert run.engine == "pyfao56"
    return [{"d": r.day.isoformat(), "eto": r.eto_in, "kcb": r.kcb,
             "etc": r.etc_adj_in, "dr": r.depletion_in, "taw": r.taw_in}
            for r in run.results]


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    wx = weather()
    tmp = tmp_path_factory.mktemp("engines")
    js = run_js(wx, tmp)
    py = {r["d"]: r for r in run_py(wx)}
    return [(a, py[a["d"]]) for a in js if a["d"] in py]


def test_reference_et_agrees_closely(paired):
    """ASCE Penman-Monteith is the same equation in both languages.

    This is the piece that must not drift: if ETo disagrees, the JavaScript has
    a transcription error rather than a modelling difference.
    """
    rel = [abs(a["eto"] - b["eto"]) / b["eto"] * 100 for a, b in paired if b["eto"] > 1e-4]
    assert statistics.median(rel) < 1.0
    js_season = sum(a["eto"] for a, _ in paired)
    py_season = sum(b["eto"] for _, b in paired)
    assert abs(js_season - py_season) / py_season < 0.01


def test_the_crop_curve_is_identical(paired):
    """Kcb is a table lookup and interpolation — no room for a real difference."""
    assert max(abs(a["kcb"] - b["kcb"]) for a, b in paired) < 0.01


def test_seasonal_crop_et_stays_within_the_gap_we_have_measured(paired):
    """The engines differ on evaporation and root growth, not on ET reference.

    Measured at -4.5%. Held loosely here so a real regression trips it while the
    known, explained difference does not.
    """
    js = sum(a["etc"] for a, _ in paired)
    py = sum(b["etc"] for _, b in paired)
    assert abs(js - py) / py < 0.10, f"seasonal ETc diverged to {(js-py)/py*100:.1f}%"


def test_the_depletion_gap_is_documented_and_bounded(paired):
    """This is the number the irrigation decision actually turns on.

    Measured max gap is about 0.75 in. On Evesboro sand holding roughly 2.5 in
    that is most of a day of July corn water use, which is exactly why the
    schedule view carries a badge naming the engine. If this grows past 1.2 in
    the mirror has drifted far enough that the badge is no longer sufficient.
    """
    gaps = [abs(a["dr"] - b["dr"]) for a, b in paired]
    assert max(gaps) < 1.2, f"depletion gap grew to {max(gaps):.2f} in"
    assert statistics.mean(gaps) < 0.5


def test_the_schedule_view_names_which_engine_produced_it():
    """A grower must never be unable to tell which engine they are reading."""
    html = INDEX.read_text()
    assert "function engineBadge()" in html
    assert "not pyFAO56" in html
    assert "engineBadge()" in html.split("function engineBadge()")[0] or \
           "${engineBadge()}" in html
