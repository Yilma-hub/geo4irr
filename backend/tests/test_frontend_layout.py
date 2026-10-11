"""
Layout invariants for the workbench shell.

These exist because of a bug that was invisible in every check that had been
running. The shell is a two-column grid: rail plus map. A third child, the
output panel, is hidden with the `hidden` attribute — but `.outpanel` and
`.rail` both set `display:flex`, and a class selector outranks the browser's
own `[hidden]{display:none}`. The "hidden" panel therefore kept its box, claimed
a grid column, and pushed the map into a second row that the fixed shell height
clipped away.

The map was simply gone. No console error, valid JavaScript, every unit test
green. Only computing the styles catches it, so that is what these do.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def computed(tmp_path, script_body: str):
    script = tmp_path / "layout.mjs"
    script.write_text(f'''
import {{ JSDOM }} from "jsdom"; import fs from "fs";
const dom=new JSDOM(fs.readFileSync({str(INDEX)!r},"utf8"),
 {{runScripts:"dangerously",pretendToBeVisual:true,url:"http://localhost/",
  beforeParse(w){{ w.fetch=async()=>({{ok:false,status:503,json:async()=>({{}})}}); }}}});
const w=dom.window, d=w.document;
await new Promise(r=>w.addEventListener("load",r));
await new Promise(r=>setTimeout(r,300));
const disp=sel=>w.getComputedStyle(d.querySelector(sel)).display;
{script_body}
''')
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=120, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr[-1200:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_hidden_output_panel_does_not_claim_a_grid_column(tmp_path):
    """Two columns while picking a field: the rail and the map. Not three."""
    r = computed(tmp_path, '''
console.log(JSON.stringify({
  rail: disp("#rail"), out: disp("#outpanel"), stage: disp(".stage"),
}));''')
    assert r["out"] == "none", (
        "the hidden output panel still has a box, so it is taking a grid column "
        "and pushing the map off screen")
    assert r["rail"] != "none"
    assert r["stage"] != "none", "the map is not displayed while selecting a field"


def test_the_map_is_removed_not_shrunk_once_a_field_is_chosen(tmp_path):
    r = computed(tmp_path, '''
d.body.classList.add("mode-work");
console.log(JSON.stringify({stage: disp(".stage"), rail: disp("#rail")}));''')
    assert r["stage"] == "none", "the map should be gone in work mode, not resized"
    assert r["rail"] != "none"


def test_hiding_the_rail_from_javascript_actually_hides_it(tmp_path):
    """`.rail{display:flex}` outranks [hidden] too — same bug, second element."""
    r = computed(tmp_path, '''
d.getElementById("rail").hidden = true;
d.getElementById("outpanel").hidden = false;
console.log(JSON.stringify({rail: disp("#rail"), out: disp("#outpanel")}));''')
    assert r["rail"] == "none", "the rail ignored its hidden attribute"
    assert r["out"] != "none"


def test_a_global_hidden_rule_exists_and_is_declared_first():
    """The general fix, not a per-class patch.

    Declared before the class rules and marked important so no later selector
    can resurrect a hidden element's box.
    """
    css = INDEX.read_text()
    m = re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important", css)
    assert m, "no global [hidden]{display:none !important} rule"
    first_class_display = re.search(r"\n\.[A-Za-z][\w-]*\s*\{[^}]*display:", css)
    assert first_class_display is None or m.start() < first_class_display.start(), \
        "the [hidden] rule is declared after class rules that set display"


def test_every_leaflet_mount_has_a_height(tmp_path):
    """A map container with no height renders a working map zero pixels tall.

    This is how the satellite panel lost its imagery: the rule was written as
    `#map{height:340px}`, but that panel mounts into `#omapel`. Tiles were
    fetched, layers attached, the STAC search returned 23 passes — and nothing
    appeared, with no error anywhere. The rule now targets the mount point by
    position inside `.mapwrap` rather than by name, so a new panel cannot
    reintroduce it by choosing a different id.
    """
    r = computed(tmp_path, '''
const mk = id => {
  const box = d.createElement("div");
  box.className = "mapwrap";
  box.innerHTML = `<div id="${id}"></div>`;
  d.body.appendChild(box);
  return w.getComputedStyle(box.firstElementChild).height;
};
console.log(JSON.stringify({
  map: mk("map"), omapel: mk("omapel"), novel: mk("some_new_panel_map"),
}));''')
    for name, height in r.items():
        assert height and height != "0px" and height != "auto", (
            f"a Leaflet mount called #{name} would render zero pixels tall")


def test_the_index_view_masks_everything_outside_the_field():
    """A Sentinel-2 tile is 110 km across; the field is a few hundred metres.

    Without a mask the reader compares their corn to a neighbour's soybeans on a
    different soil and planting date. Leaflet has no clip operation, so this is
    an inverse mask: a world-sized polygon with the field punched out as a hole.
    """
    html = INDEX.read_text()
    assert "idxmask" in html, "no mask layer on the index view"
    assert "const WORLD=" in html, "the inverse mask has no outer ring"
    mask = html[html.index("const WORLD="):html.index("const WORLD=") + 700]
    assert "[WORLD].concat(holes)" in mask, (
        "the field is not punched out of the mask, so it would cover the "
        "imagery instead of framing it")
