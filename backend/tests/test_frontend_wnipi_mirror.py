"""
The browser's WNIPI must agree with the server's.

The frontend carries its own copy of Eq 6 so the season score works with no
connection — the same reason it carries an offline mirror of the water balance.
Two implementations of one equation is exactly the situation where they quietly
drift apart, and a grower comparing a score on their phone against one from the
API would have no way to tell which was wrong. This test extracts the real
JavaScript out of index.html and runs it against the Python.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

from app.engine.nitrogen import water_nutrient_index

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"

CASES = [
    # yield, baseline, irrigation, ET0, N, P2O5, N uptake/unit, P uptake/unit
    (230.0, 200.0,  4.0, 24.0, 150.0,  50.0, 0.90, 0.37),   # lean
    (230.0, 200.0, 12.0, 24.0, 240.0,  90.0, 0.90, 0.37),   # heavy
    (180.0, 200.0,  6.0, 24.0, 180.0,  60.0, 0.90, 0.37),   # under baseline
    (200.0, 200.0,  6.0, 24.0, 180.0,  60.0, 0.90, 0.37),   # exactly level
    (265.0, 195.0,  9.5, 27.3, 205.0,  75.0, 0.90, 0.37),   # awkward numbers
    ( 68.0,  60.0,  2.0, 22.0,   0.0,  40.0, 3.80, 0.84),   # soybean, no N
    (240.0, 210.0,  0.0, 25.0, 190.0,  65.0, 0.90, 0.37),   # rain-fed season
]

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is needed to run the browser copy")


def _extract_js() -> str:
    html = INDEX.read_text()
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    app_js = max(scripts, key=len)
    start = app_js.index("function wnipiScore")
    end = app_js.index("function wnipiReading")
    return app_js[start:end]


def test_browser_wnipi_matches_the_server_to_the_last_digit(tmp_path):
    script = tmp_path / "check.mjs"
    script.write_text(
        _extract_js()
        + "\nconst cases = " + json.dumps(CASES) + ";\n"
        + "const out = cases.map(([y,y0,irr,et0,n,p,un,up]) =>\n"
        + "  wnipiScore({y,y0,irr,et0,n,p,anu0:y0*un,apu0:y0*up}).score);\n"
        + "console.log(JSON.stringify(out));\n"
    )
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=30)
    assert proc.returncode == 0, proc.stderr
    js_scores = json.loads(proc.stdout)

    for (y, y0, irr, et0, n, p, un, up), js in zip(CASES, js_scores):
        py = water_nutrient_index(
            yield_farm=y, yield_baseline=y0, irrigation_in=irr, eto_in=et0,
            n_applied=n, n_uptake_baseline=y0 * un,
            p_applied=p, p_uptake_baseline=y0 * up)
        # the server rounds to 4 dp for display; compare at that resolution
        assert round(js, 4) == pytest.approx(py, abs=1e-4), (
            f"browser {js} vs server {py} for yield {y} against {y0}")


def test_the_browser_guards_a_zero_baseline_instead_of_dividing_by_it(tmp_path):
    script = tmp_path / "guard.mjs"
    script.write_text(
        _extract_js()
        + "console.log(JSON.stringify(wnipiScore({y:230,y0:0,irr:4,et0:24,"
          "n:150,p:50,anu0:180,apu0:74})));\n")
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) is None, "a zero baseline must not divide"
