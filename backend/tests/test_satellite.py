"""
Tests for the Sentinel imagery module.

Nothing here touches the network. What is worth testing without credentials is
the part that carries agronomic meaning: that every index is wired to the right
bands, that cloud never reaches a field mean, that a saturated NDVI scene is
allowed to move the water balance less than a clean one, and that the colour
ramps agree with each other about which end is dry.
"""

from __future__ import annotations

from datetime import date

import json
import shutil
import subprocess
import pytest

from app.engine import satellite as sat
from app.engine.fao56 import kcb_from_ndvi


# ---------------------------------------------------------------------------
# catalogue and evalscripts
# ---------------------------------------------------------------------------

def test_every_index_is_catalogued():
    keys = {i["key"] for i in sat.index_catalog()}
    assert {"true_color", "ndvi", "ndre", "savi", "gndvi", "ndmi", "msi"} <= keys


def test_evalscripts_declare_version_3():
    for idx in sat.INDICES.values():
        assert idx.visual.startswith("//VERSION=3")
        if idx.stats:
            assert idx.stats.startswith("//VERSION=3")


def test_evalscript_braces_balance():
    # A malformed evalscript fails inside Sentinel Hub with a message that says
    # very little. Catching it here is cheaper than catching it in the field.
    for key, idx in sat.INDICES.items():
        for script in filter(None, (idx.visual, idx.stats)):
            assert script.count("{") == script.count("}"), key


def test_indices_request_the_bands_they_need():
    assert '"B08"' in sat.INDICES["ndvi"].visual and '"B04"' in sat.INDICES["ndvi"].visual
    assert '"B05"' in sat.INDICES["ndre"].visual        # red edge, not red
    assert '"B03"' in sat.INDICES["gndvi"].visual       # green
    assert '"B11"' in sat.INDICES["ndmi"].visual        # shortwave infrared


def test_optical_indices_mask_cloud_in_statistics():
    # Scene classification 8, 9 and 10 are cloud. A field mean that includes
    # them is not a field mean.
    for key in ("ndvi", "ndre", "savi", "gndvi", "ndmi", "msi"):
        script = sat.INDICES[key].stats
        assert "SCL" in script
        assert "usable" in script


def test_radar_does_not_pretend_to_be_optical():
    s1 = sat.INDICES["s1_vv"]
    assert s1.collection == sat.S1_COLLECTION
    # No red or near-infrared bands exist on a radar satellite.
    assert "B08" not in s1.visual and "B04" not in s1.visual
    assert "VV" in s1.visual


def test_sentinel1_carries_a_caution_about_what_it_is_not():
    assert "soil moisture" in sat.INDICES["s1_vv"].caution


def test_ndvi_warns_about_saturation():
    assert "saturat" in sat.INDICES["ndvi"].caution.lower()


def test_moisture_indices_say_they_are_not_soil_water():
    for key in ("ndmi", "msi"):
        assert "not" in sat.INDICES[key].caution.lower()


# ---------------------------------------------------------------------------
# canopy observations feeding the water balance
# ---------------------------------------------------------------------------

def test_canopy_ahead_of_the_calendar_raises_the_crop_coefficient():
    obs = sat.canopy_observation(0.74, 0.04, date(2026, 6, 20), model_kcb=0.70)
    assert obs["kcb_ratio"] > 1.1
    assert "ahead" in obs["verdict"]


def test_canopy_behind_the_calendar_lowers_it():
    obs = sat.canopy_observation(0.34, 0.05, date(2026, 6, 20), model_kcb=0.90)
    assert obs["kcb_ratio"] < 0.9
    assert "behind" in obs["verdict"]


def test_matching_canopy_changes_nothing():
    ndvi = 0.60
    obs = sat.canopy_observation(ndvi, 0.03, date(2026, 7, 1),
                                 model_kcb=kcb_from_ndvi(ndvi))
    assert obs["kcb_ratio"] == pytest.approx(1.0, abs=0.02)
    assert "matches" in obs["verdict"]


def test_saturated_scene_is_trusted_less():
    # Closed canopy: NDVI cannot tell a comfortable crop from a hurting one, so
    # the observation must arrive with wider uncertainty or it will overrule a
    # water balance that knows better.
    clean = sat.canopy_observation(0.55, 0.03, date(2026, 6, 10), 0.75)
    saturated = sat.canopy_observation(0.86, 0.03, date(2026, 7, 15), 1.10)
    assert saturated["sigma_kcb"] > clean["sigma_kcb"]
    assert saturated["saturated"] is True
    assert "NDRE" in saturated["verdict"]


def test_a_patchy_field_produces_a_less_certain_observation():
    even = sat.canopy_observation(0.60, 0.02, date(2026, 6, 25), 0.80)
    patchy = sat.canopy_observation(0.60, 0.18, date(2026, 6, 25), 0.80)
    assert patchy["sigma_kcb"] > even["sigma_kcb"]


def test_observation_uses_the_same_transfer_as_the_water_balance():
    # Two definitions of NDVI-to-Kcb would let the map and the schedule disagree.
    obs = sat.canopy_observation(0.58, 0.04, date(2026, 7, 2), 0.80)
    assert obs["observed_kcb"] == kcb_from_ndvi(0.58)


# ---------------------------------------------------------------------------
# uniformity
# ---------------------------------------------------------------------------

def test_even_field_is_reported_as_even():
    assert "Even" in sat.uniformity({"mean": 0.72, "stdev": 0.03})["verdict"]


def test_variable_field_is_flagged_for_variable_rate():
    out = sat.uniformity({"mean": 0.55, "stdev": 0.14})
    assert out["cv"] > 0.16
    assert "variable-rate" in out["verdict"]


def test_uniformity_survives_a_zero_mean():
    assert sat.uniformity({"mean": 0.0, "stdev": 0.0})["cv"] >= 0


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def test_season_window_starts_before_planting():
    start, end = sat.season_window(date(2026, 5, 1), today=date(2026, 8, 1))
    assert start < date(2026, 5, 1)      # pre-plant scenes show residue and cover
    assert end == date(2026, 8, 1)


def test_season_window_does_not_run_into_the_future():
    _, end = sat.season_window(date(2026, 4, 20), today=date(2026, 6, 1))
    assert end == date(2026, 6, 1)


def test_missing_credentials_explain_how_to_get_them():
    hub = sat.SentinelHub(creds=sat.Credentials())
    assert not hub.configured
    import asyncio
    with pytest.raises(sat.SentinelHubError, match="dataspace.copernicus.eu"):
        asyncio.run(hub.token(None))


def test_endpoints_point_at_the_copernicus_data_space():
    # The old services.sentinel-hub.com endpoints are the commercial ones and
    # will not accept a free Copernicus account.
    assert "dataspace.copernicus.eu" in sat.TOKEN_URL
    assert "dataspace.copernicus.eu" in sat.PROCESS_URL
    assert "dataspace.copernicus.eu" in sat.STATS_URL


# ---------------------------------------------------------------------------
# The open, credential-free imagery path in the browser
# ---------------------------------------------------------------------------

def _open_imagery_js() -> str:
    import pathlib, re
    index = (pathlib.Path(__file__).resolve().parents[2]
             / "frontend" / "index.html")
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", index.read_text(), re.S)
    app = max(scripts, key=len)
    start = app.index("const S2_OFFSET_FROM")
    end = app.index("async function openStacScenes")
    return app[start:end]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_open_imagery_indices_are_computed_in_reflectance_not_raw_counts(tmp_path):
    """Sentinel-2 L2A stores scaled integers with a -1000 offset since 2022.

    That offset does not cancel in a normalised difference — the numerator loses
    it, the denominator keeps twice it. Using raw counts on a canopy at 0.40 NIR
    and 0.05 red gives NDVI 0.538 against a true 0.778. A quarter of an NDVI unit
    would carry straight into Kcb and corrupt the water balance, so this asserts
    the browser recovers the true value on both sides of the baseline change.
    """
    script = tmp_path / "ix.mjs"
    script.write_text(
        _open_imagery_js()
        + """
const mk = d => ({id:"x", date:d, cloud:2,
  href:"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/x"});
const Rn = 0.40, Rr = 0.05;
const out = {};
for (const [tag, d, off] of [["post", "2026-07-20", 1000], ["pre", "2019-07-20", 0]]) {
  const nir = Rn*10000 + off, red = Rr*10000 + off;
  const e = OPEN_INDICES.ndvi.expr(s2Offset(mk(d).date));
  out[tag] = Function("nir","red",`return ${e};`)(nir, red);
}
console.log(JSON.stringify(out));
"""
    )
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=30)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)

    true_ndvi = (0.40 - 0.05) / (0.40 + 0.05)
    assert got["post"] == pytest.approx(true_ndvi, abs=1e-4)
    assert got["pre"] == pytest.approx(true_ndvi, abs=1e-4)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_open_imagery_offset_applies_only_after_the_baseline_change(tmp_path):
    script = tmp_path / "off.mjs"
    script.write_text(
        _open_imagery_js()
        + 'console.log(JSON.stringify(["2021-12-31","2022-01-25","2026-07-20"]'
          '.map(s2Offset)));\n')
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [0, 1000, 1000]
