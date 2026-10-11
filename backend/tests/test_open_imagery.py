"""
The credential-free imagery path: what gets asked of the open tile service.

Real Sentinel-2 pixels come from AWS via Element 84's STAC index, and the band
maths is done by a public TiTiler. None of that needs an account, but all of it
is a request that has to be built exactly right, and when it is not the result
is an empty map rather than an error. These tests build the requests and check
them without the network, and check that a refusal is reported with its reason
rather than as a shrug.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

SCENE = {
    "id": "S2A_18SVF_20260723_0_L2A",
    "date": "2026-07-23",
    "cloud": 3,
    "collection": "sentinel-2-l2a",
    "href": ("https://earth-search.aws.element84.com/v1/collections/"
             "sentinel-2-l2a/items/S2A_18SVF_20260723_0_L2A"),
}


def run(tmp_path, body, fetch="w.fetch=async()=>({ok:false,status:503,json:async()=>({})});"):
    script = tmp_path / "t.mjs"
    script.write_text(
        'import { JSDOM } from "jsdom"; import fs from "fs";\n'
        f'const dom=new JSDOM(fs.readFileSync({str(INDEX)!r},"utf8"),\n'
        ' {runScripts:"dangerously",pretendToBeVisual:true,url:"http://localhost/",\n'
        f'  beforeParse(w){{ {fetch} }}}});\n'
        'const w=dom.window, d=w.document;\n'
        'await new Promise(r=>w.addEventListener("load",r));\n'
        'await new Promise(r=>setTimeout(r,250));\n'
        f'const SCENE={json.dumps(SCENE)};\n'
        + body + "\n")
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=120, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr[-1200:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_true_colour_uses_the_prebuilt_visual_asset(tmp_path):
    """ESA already rendered it; compositing three COGs by hand is worse.

    Building true colour from red/green/blue needs a stretch that depends on the
    processing baseline and fetches three files instead of one. The `visual`
    asset is 8-bit RGB and needs neither.
    """
    r = run(tmp_path, '''
const p = w.eval("openTileParams")(SCENE, "truecolor");
console.log(JSON.stringify({
  assets: p.getAll("assets"),
  band: p.get("asset_as_band"),
  rescale: p.get("rescale"),
  expression: p.get("expression"),
}));''')
    assert r["assets"] == ["visual"]
    assert r["band"] is None, "asset_as_band on an RGB composite splits it apart"
    assert r["rescale"] is None and r["expression"] is None


@pytest.mark.parametrize("index,bands", [
    ("ndvi", ["nir", "red"]),
    ("ndre", ["nir", "rededge1"]),
    ("savi", ["nir", "red"]),
    ("gndvi", ["nir", "green"]),
    ("ndmi", ["nir", "swir16"]),
    ("msi", ["swir16", "nir"]),
])
def test_each_index_requests_exactly_the_bands_its_expression_uses(tmp_path, index, bands):
    """A missing asset is a 4xx; an extra one is a wasted COG read."""
    r = run(tmp_path, f'''
const p = w.eval("openTileParams")(SCENE, "{index}");
console.log(JSON.stringify({{
  assets: p.getAll("assets").sort(),
  band: p.get("asset_as_band"),
  expr: p.get("expression"),
  rescale: p.get("rescale"),
  ramp: p.get("colormap_name"),
}}));''')
    assert r["assets"] == sorted(bands)
    assert r["band"] == "true", "expressions need asset_as_band"
    assert r["ramp"] and r["rescale"]
    for b in bands:
        assert b in r["expr"], f"{b} requested but not used in the expression"


def test_radar_searches_its_own_archive(tmp_path):
    """Sentinel-1 has no optical bands, so it cannot share the Sentinel-2 list.

    This was broken on release: the Radar button fell through to the NDVI
    expression against a Sentinel-2 scene, which simply fails. Radar now carries
    its own collection and the pass list reloads when it is selected — the two
    satellites do not fly over on the same days.
    """
    r = run(tmp_path, '''
const ix = w.eval("OPEN_INDICES").radar;
const p = w.eval("openTileParams")({...SCENE, collection:"sentinel-1-grd"}, "radar");
console.log(JSON.stringify({
  collection: ix.collection,
  assets: p.getAll("assets"),
  expr: p.get("expression"),
  band: p.get("asset_as_band"),
}));''')
    assert r["collection"] == "sentinel-1-grd"
    assert r["assets"] == ["vv"]
    assert "log10" in r["expr"], "radar backscatter should be shown in decibels"
    assert r["band"] == "true"


def test_the_tile_template_is_discovered_rather_than_assumed(tmp_path):
    """Ask the server for its own URL shape.

    Hard-coding the route means a renamed parameter or a moved tile matrix set
    shows up as an empty map. The tilejson endpoint hands back the exact
    template, so the only thing that can be wrong is the query itself.
    """
    r = run(tmp_path, '''
const tpl = await w.eval("openTileTemplate")(SCENE, "ndvi");
console.log(JSON.stringify({template: tpl.template, min: tpl.minzoom, max: tpl.maxzoom}));''',
        fetch='''w.fetch=async(u)=>{
          if(String(u).includes("tilejson")) return {ok:true,status:200,json:async()=>({
            tiles:["https://titiler.xyz/stac/tiles/WebMercatorQuad/{z}/{x}/{y}.png?ok=1"],
            minzoom:8,maxzoom:14})};
          return {ok:false,status:503,json:async()=>({})};};''')
    assert "{z}/{x}/{y}" in r["template"]
    assert r["min"] == 8 and r["max"] == 14


def test_a_refusal_is_reported_with_its_reason(tmp_path):
    """"Did not return this layer" tells the reader nothing they can act on."""
    r = run(tmp_path, '''
let msg = "";
try { await w.eval("openTileTemplate")(SCENE, "ndvi"); }
catch (e) { msg = e.message; }
console.log(JSON.stringify({msg}));''',
        fetch='''w.fetch=async()=>({ok:false,status:429,
          json:async()=>({detail:"Too many requests"})});''')
    assert "429" in r["msg"]
    assert "Too many requests" in r["msg"]


def test_a_network_failure_is_distinguished_from_a_refusal(tmp_path):
    """Different problems, different fixes.

    A refusal (HTTP 429, say) means the request itself is wrong, so it is
    raised and the map is not attempted. A failed `fetch` means something
    else: either the browser blocked a cross-origin call, which the tile
    <img> requests will make anyway and succeed at, or the connection is
    down. Neither is a reason to refuse, because the tiles are worth
    attempting either way, so this returns a usable template carrying the
    reason rather than throwing.
    """
    r = run(tmp_path, '''
let msg = "", out = null;
try { out = await w.eval("openTileTemplate")(SCENE, "ndvi"); }
catch (e) { msg = e.message; }
console.log(JSON.stringify({msg, blocked: out && out.blocked,
  reason: out && out.reason, template: out && out.template,
  discovered: out && out.discovered}));''',
        fetch='w.fetch=async()=>{ throw new TypeError("Failed to fetch"); };')
    assert r["msg"] == "", "a blocked fetch is not a refusal and must not throw"
    assert r["blocked"] is True
    assert r["discovered"] is False
    assert "{z}/{x}/{y}" in r["template"], (
        "a usable tile template must still come back, because <img> requests "
        "are not subject to the CORS rule that blocked the discovery call")
    assert "reach" in r["reason"].lower()


def test_being_offline_is_named_as_such_rather_than_called_a_cors_problem(tmp_path):
    """When the browser itself says it is offline there is no ambiguity left,
    and telling the reader it is a cross-origin quirk would send them looking
    in the wrong place."""
    r = run(tmp_path, '''
const out = await w.eval("openTileTemplate")(SCENE, "ndvi");
console.log(JSON.stringify({reason: out.reason, blocked: out.blocked}));''',
        fetch='''w.fetch=async()=>{ throw new TypeError("Failed to fetch"); };
                 Object.defineProperty(w.navigator,"onLine",{value:false,
                   configurable:true});''')
    assert r["blocked"] is True
    assert "offline" in r["reason"].lower()
    assert "schedule" in r["reason"].lower(), (
        "say what still works without a connection, not only what does not")
