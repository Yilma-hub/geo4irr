"""
Installing Geo4Irr on a phone.

The app installs to an Android home screen as a Progressive Web App: same code,
same internet access, an icon instead of a browser tab. These tests hold the
manifest valid, the icons real, and — the part that matters — the service
worker's caching policy honest.

That last one is the whole risk of offline support in a tool like this. Caching
the app so it opens on a farm road is useful. Caching an *answer* is dangerous:
a grower reading yesterday's NDVI or last week's soil water deficit, believing
it is today's, is worse off than one who can see the request failed. So the
shell is cached and the data never is, and that is asserted here rather than
left to a comment.
"""

from __future__ import annotations

import json
import pathlib
import re
import struct

import pytest

FRONTEND = pathlib.Path(__file__).resolve().parents[2] / "frontend"
MANIFEST = FRONTEND / "manifest.json"
SW = FRONTEND / "sw.js"
INDEX = FRONTEND / "index.html"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(),
                                reason="the installable app files are not present")


def manifest():
    return json.loads(MANIFEST.read_text())


def test_the_manifest_has_what_android_needs_to_offer_an_install():
    m = manifest()
    assert m["name"] and m["short_name"]
    assert len(m["short_name"]) <= 12, "long names are truncated under the icon"
    assert m["display"] == "standalone", (
        "without standalone display it opens in a browser tab, not as an app")
    assert m["start_url"]
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", m["theme_color"])
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", m["background_color"])


def test_there_is_a_maskable_icon():
    """Android crops icons to whatever shape the launcher uses.

    A non-maskable icon gets its corners cut off, which on a mark that is a
    circle inside a square means the ring loses its edges. The maskable variant
    keeps the art inside the safe middle 80%.
    """
    icons = manifest()["icons"]
    assert any(i.get("purpose") == "maskable" for i in icons)
    assert any(i.get("sizes") == "512x512" for i in icons)
    assert any(i.get("sizes") == "192x192" for i in icons)


def test_every_icon_the_manifest_promises_actually_exists_and_is_a_real_image():
    """A 404 icon is an install prompt that never appears, with no error."""
    for icon in manifest()["icons"]:
        path = FRONTEND / icon["src"]
        assert path.exists(), f"{icon['src']} is listed but missing"
        data = path.read_bytes()
        assert len(data) > 200, f"{icon['src']} is too small to be an icon"

        if path.suffix == ".png":
            assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{icon['src']} is not a PNG"
            width, height = struct.unpack(">II", data[16:24])
            declared = icon["sizes"].split("x")[0]
            assert width == height == int(declared), (
                f"{icon['src']} is {width}x{height} but claims {icon['sizes']}")


def test_the_page_links_the_manifest_and_a_theme_colour():
    html = INDEX.read_text()
    assert 'rel="manifest"' in html
    assert 'name="theme-color"' in html
    assert 'rel="apple-touch-icon"' in html


# --------------------------------------------------------------------------
# the service worker's caching policy
# --------------------------------------------------------------------------

def test_the_service_worker_never_caches_live_data():
    """The one rule that matters.

    Satellite tiles, weather, the soil survey and the API must go to the
    network every time. A cached irrigation decision is a wrong irrigation
    decision, and it fails silently — which is the worst way for a tool like
    this to fail.
    """
    sw = SW.read_text()
    for host in ("titiler.xyz", "earth-search.aws.element84.com",
                 "sdmdataaccess.sc.egov.usda.gov", "api.weather.gov",
                 "deos.udel.edu"):
        assert host in sw, f"{host} is not on the never-cache list"
    assert 'url.pathname.startsWith("/api/")' in sw, (
        "API responses are not excluded from the cache")
    assert "isLiveData(url)) return;" in sw, (
        "the live-data check does not actually skip the cache")


def test_the_app_shell_is_cached_so_it_opens_without_a_signal():
    sw = SW.read_text()
    for asset in ("./index.html", "./data/irrigated_areas.js", "./manifest.json"):
        assert asset in sw, f"{asset} is not in the installed shell"


def test_one_missing_file_cannot_stop_the_whole_shell_caching():
    """cache.addAll is all-or-nothing: a single 404 leaves nothing cached and
    the app uninstallable, with only a console warning to say so."""
    # Strip comments first: the file explains *why* addAll is avoided, and
    # searching the prose for the thing it warns against finds the warning.
    sw = SW.read_text()
    code = re.sub(r"/\*.*?\*/", "", sw, flags=re.S)
    code = re.sub(r"//.*", "", code)
    assert "addAll" not in code, (
        "addAll makes shell caching all-or-nothing — add each file separately")
    assert "cache.add(url).catch" in code


def test_old_caches_are_cleared_when_the_version_changes():
    """Otherwise a stale app shell survives every update forever."""
    sw = SW.read_text()
    assert "caches.keys()" in sw and "caches.delete" in sw


def test_the_service_worker_is_valid_javascript():
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("needs node")
    proc = subprocess.run(["node", "--check", str(SW)],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_service_worker_registration_is_skipped_for_a_file_url():
    """Registering from file:// throws. Guarding it keeps the console clean for
    the many people who will still just double-click the HTML."""
    html = INDEX.read_text()
    reg = html[html.index("function registerServiceWorker"):]
    reg = reg[:reg.index("\n}")]
    assert 'location.protocol === "file:"' in reg


def test_matchmedia_is_guarded_everywhere_it_is_used():
    """Some embedded webviews have no matchMedia, and an exception during boot
    would take the map down before it drew. A cosmetic install chip is not
    worth that."""
    html = INDEX.read_text()
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    app = max(scripts, key=len)
    direct = [m for m in re.finditer(r"window\.matchMedia\(", app)]
    assert len(direct) == 1, (
        "matchMedia should be called in exactly one guarded helper, "
        f"found {len(direct)} call sites")
    helper = app[app.index("function mq(query)"):]
    helper = helper[:helper.index("\n}")]
    assert "try" in helper and "catch" in helper
