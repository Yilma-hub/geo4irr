# Testing Geo4Irr

Everything here runs without credentials, without a database, and mostly without
a network. Work down it in order; each section assumes the one before it passed.

Expected numbers are given throughout. They are not arbitrary — they come from
the verification runs and are the values to compare against if you change
something and want to know whether you broke it.

---

## 1. Install

```bash
unzip geo4irr.zip
cd geo4irr
cd backend
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 or newer. Check with `python3 --version`.

---

## 2. The automated suite

```bash
cd backend
python -m pytest tests/ -q
```

**Expected: `244 passed`**, in about twelve seconds. No network is used.

| File | Count | What it covers |
|---|---|---|
| `tests/test_engine.py` | 47 | ET, water balance, soils, assimilation, nitrogen, sensors, economics |
| `tests/test_geo.py` | 34 | Boundary parsing, area, validation, SSURGO matching, zoning |
| `tests/test_satellite.py` | 22 | Index definitions, cloud masking, canopy observations, uniformity |
| `tests/test_pyfao56_bridge.py` | 22 | Translation into pyfao56 classes, and agreement between the two engines |
| `tests/test_proposal_methods.py` | 97 | Eq 1 closure, Eq 4/5 likelihood, Eq 6 WNIPI, generative filtering, SWAT+ fluxes and file formats, NSGA-II, VRI grids, DANS, data formats, MLR depths |
| `tests/test_shapefile.py` | 20 | Server-side `.shp`/`.dbf`/`.prj`/zip reading, holes, UTM, failure modes |
| `tests/test_frontend_wnipi_mirror.py` | 2 | The browser's copy of Eq 6 against the Python, run under node |

Two results that look like failures but are not:

- **`222 passed, 22 skipped`** — pyfao56 is not installed (see below).
- **`242 passed, 2 skipped`** — `node` is not on the PATH, so the frontend
  mirror tests cannot run.

Every test in `test_proposal_methods.py` names the Activity or equation it
defends, so a failure asks two questions: is the code wrong, and has the method
drifted from what was proposed. See `PROPOSAL_COMPLIANCE.md`.

The bridge tests skip cleanly if pyfao56 is not installed — `103 passed, 22
skipped` is a correct result, not a broken one. To install it:

```bash
pip install "pyfao56 @ git+https://github.com/isukendall/pyfao56.git"
```

Useful variations:

```bash
python -m pytest tests/ -v                    # every test name
python -m pytest tests/test_geo.py -v         # one file
python -m pytest tests/ -k "nitrogen"         # by keyword
python -m pytest tests/ -k "zone or split"    # the zoning logic
python -m pytest tests/ -x --tb=short         # stop at first failure
```

The tests are written as assertions about agronomic behaviour, not about
internals, so the names read as claims. `python -m pytest tests/ -v` is a decent
specification of what the system believes:

```
test_evesboro_holds_far_less_than_matapeake PASSED
test_stress_engages_past_the_allowable_depletion PASSED
test_filter_moves_toward_an_accurate_probe PASSED
test_bumped_probe_is_gated_out PASSED
test_litter_credit_reduces_the_fertilizer_bill PASSED
test_sandy_soil_gets_a_four_way_split PASSED
test_pollination_stress_costs_more_than_late_stress PASSED
test_split_recommended_when_sand_meets_a_wet_soil PASSED
test_no_split_when_the_soils_behave_alike PASSED
test_saturated_scene_is_trusted_less PASSED
```

---

## 3. Check the engine by hand

Values you can reproduce in a REPL. If these drift, something moved.

```bash
cd backend
python
```

### Reference ET on a hot Delaware day

```python
from datetime import date
from app.engine.fao56 import WeatherDay, reference_et

d = WeatherDay(day=date(2026, 7, 15), tmax_f=91, tmin_f=70, rh_max=92, rh_min=48,
               wind_mph=6.5, solar_rad_mj_m2=24.0)
print(reference_et(d, 38.7))                       # lat_deg
```

**Expected: 0.2401 in/day.** Peak-season Delmarva ETo belongs in the 0.19–0.26
band; above 0.30 or below 0.12 means radiation or humidity handling broke.

Drop the radiation and humidity and it falls back to Hargreaves-Samani:

```python
print(reference_et(WeatherDay(day=date(2026, 7, 15), tmax_f=91, tmin_f=70), 38.7))
```

**Expected: 0.2305 in/day** — close here, but the gap widens on humid overcast
days. This is why a DEOS station with a pyranometer is worth choosing over a
nearer one without.

### Two soils, same field

```python
from app.engine.soils import get_soil
for k in ("evesboro", "matapeake"):
    s = get_soil(k)
    print(f"{s.series:26s} {s.awc_in_per_in * 30:.2f} in in a 30-inch root zone")
```

**Expected: Evesboro 2.10 in, Matapeake 5.70 in.** That 2.7× difference is the
whole reason Delaware scheduling differs from the Midwest.

### Nitrogen with poultry litter

Note `manure` is a dict passed through to `manure_available_n`, not loose
keyword arguments.

```python
from app.engine.nitrogen import build_nitrogen_plan

plan = build_nitrogen_plan(
    crop_key="corn_grain", yield_goal=210,
    previous_crop="soybean_full", cover_crop="cereal_rye",
    manure={"source_key": "broiler_litter", "rate_ton_per_ac": 3.0,
            "incorporation": "within_24h", "years_since_first_application": 3},
    psnt_ppm=18, soil_awc_in_per_in=0.07)

print(plan.total_requirement_lb, plan.total_credits_lb,
      plan.recommended_fertilizer_lb, len(plan.split_schedule),
      plan.leaching_risk)
for w in plan.warnings:
    print(" -", w[:70])
```

**Expected: requirement 220.5, credits 204.4, apply 16.1 lb, 4 splits, leaching
risk `high`.** Three years of litter at 3 ton/ac has covered nearly the whole
crop.

The first warning should be about **174 lb P₂O₅/ac** — phosphorus, not nitrogen,
is the binding constraint on Delmarva litter, and a nitrogen tool that stays
quiet about that is doing harm.

Change `soil_awc_in_per_in` to 0.19 (Matapeake) and it should give **3 splits
and `leaching_risk: low`**. Heavy ground holds nitrogen between passes; sand does
not. That single parameter driving both the split count and the risk rating is
the water-and-nitrogen link the whole system is built around.

### Sensor placement

`design_network` takes a candidate grid, not acres. Build one with
`grid_candidates`:

```python
from app.engine import sensors as sd

cands = sd.grid_candidates(2600, 2600, 400, default_soil="evesboro")
plan = sd.design_network(cands, budget_usd=3000, root_depth_in=30,
                         sensor_key="capacitance_pro")
print(plan.n_stations, plan.total_cost_usd,
      plan.expected_depletion_sd_in, plan.sd_without_sensors_in)
```

**Expected: 2 stations, $2,200, uncertainty 0.385 in against 0.55 in with no
sensors.** Cutting the budget to $1,500 should give 1 station — the design is
budget-constrained, not a rule of thumb.

## 4. Boundary handling

Sample files are generated for you:

```bash
cd ..                       # project root
python tools/make_samples.py
```

Writes eight files into `samples/`:

| File | Purpose | Expected result |
|---|---|---|
| `home_pivot.geojson` | 132-acre pivot circle | **132.5 acres**, no warnings |
| `home_pivot.kml` | Same field as KML | **132.5 acres** — must match the GeoJSON |
| `fields.geojson` | Three fields in one file | **287.3 acres**, warns that it merged 3 polygons |
| `airport_field_shapefile.zip` | Zipped shapefile, deflated | **96.5 acres** |
| `airport_field_stored.zip` | Same, uncompressed | **96.5 acres** — must match |
| `wrong_projection.zip` | UTM `.prj` | **58.3 acres** + a projection warning |
| `off_peninsula.geojson` | A field in Iowa | **120.6 acres** + an off-Delmarva warning |
| `points_not_boundary.geojson` | Points, not a polygon | Refused, with a message about boundary exports |

Verify them against the Python engine:

```bash
cd backend
python -c "
import json, pathlib, sys; sys.path.insert(0,'.')
from app.engine import geo
for f in ['home_pivot.geojson','fields.geojson','off_peninsula.geojson']:
    b = geo.parse_geojson(json.loads(pathlib.Path('../samples/'+f).read_text()))
    print(f'{f:26s} {geo.boundary_acres(b):7.1f} ac  warnings={len(geo.validate(b))}')
"
```

Two independent checks worth running because they catch different mistakes:

A square mile section is 640 acres, and every farm map in the country is built
on that. The snippet below builds one with flat degree-to-metre constants, so it
lands at 643 — the 0.5% is the snippet's approximation, not the engine's. The
test suite builds the same section more carefully and asserts 640 within 0.5%.

```python
import math, sys; sys.path.insert(0,'.')
from app.engine import geo
lat = 38.9
dlat = 1609.344 / 110540.0
dlon = 1609.344 / (111320.0 * math.cos(math.radians(lat)))
ring = [[-75.6, lat], [-75.6+dlon, lat], [-75.6+dlon, lat+dlat], [-75.6, lat+dlat]]
b = geo.parse_geojson({"type": "Polygon", "coordinates": [ring]})
print(round(geo.boundary_acres(b), 1))     # expect 643.0
```

```python
# Zone recommendation: mixed and different -> split; mixed and alike -> no split.
c = lambda k, f: {"series_key": k, "fraction": f,
                  "awc_in_per_in": geo.SOILS[k].awc_in_per_in}
print(geo.suggest_zones([c("evesboro",.55), c("fallsington",.45)], 132)["recommend"])
print(geo.suggest_zones([c("sassafras",.60), c("ingleside",.40)], 100)["recommend"])
```

**Expected: `True` then `False`.** The second pair differ on the soil map and
behave identically in the pipe; warning about them would be noise.

---

## 4b. The two ET engines

pyfao56 (Thorp 2022, USDA-ARS) is the reference implementation named in the
narrative. It is validated against Maricopa lysimeter data and the Ref-ET
software; the built-in engine is validated against nothing but its own tests.
Where they disagree, pyfao56 is presumed right.

The single most important check in the repository — do two independent
implementations of ASCE standardized reference ET produce the same number?

```bash
cd backend
python -c "
import sys; sys.path.insert(0,'.')
from datetime import date
import pyfao56 as fao
from app.engine.fao56 import WeatherDay, reference_et, f_to_c, mph_to_ms
w = WeatherDay(day=date(2026,7,15), tmax_f=91, tmin_f=70, rh_max=92, rh_min=48,
               wind_mph=6.5, solar_rad_mj_m2=24.0)
mine = reference_et(w, 38.7)
theirs = fao.refet.ascedaily('S', 12.0, 38.7, 196, 24.0, f_to_c(91), f_to_c(70),
                             rhmax=92, rhmin=48, wndsp=mph_to_ms(6.5)) / 25.4
print(f'{mine:.4f} vs {theirs:.4f}  ({abs(mine-theirs)/theirs*100:.2f}%)')
"
```

**Expected: `0.2401 vs 0.2401 (0.01%)`.** Anything above about 0.5% is a bug in
one of them and everything downstream inherits it.

Then the full season comparison:

```bash
cd ..
python tools/compare_engines.py
```

**Expected output:**

```
ETo (in)      -0.0000     0.0000     0.0010     0.0002
Kcb           +0.0000     0.0000     0.0000     0.0000
Ks            +0.0178     0.2598     1.0000     0.3242
ETc adj       -0.0074     0.0741     0.2680     0.0946
Dr (in)       +0.3201     0.3223     0.8150     0.3690
TAW (in)      +0.2030     0.2030     0.8090     0.3305

season crop water use   built-in  18.43 in    pyfao56  19.40 in    (-5.0%)
```

Read it this way. **ETo and Kcb matching exactly** is the result that matters —
the foundation is sound. **TAW diverging** is real and explainable: FAO-56 grows
roots in proportion to crop development, the built-in engine ties root depth to
canopy cover. That difference propagates into Dr and then into Ks, which is why
the stress coefficient looks so different. It is a modelling choice, not a bug,
and pyfao56's is the one to prefer.

Other useful runs:

```bash
python tools/compare_engines.py --soil matapeake --crop soybean_full
python tools/compare_engines.py --daily | head -40      # day by day
python tools/compare_engines.py --all-soils             # sweep all 12 series
python tools/compare_engines.py --irrigate-every 0      # dryland
```

Confirm which engine the API is using:

```bash
curl -s localhost:8000/api/health | python3 -m json.tool
```

**Expected: `"et_engine": "pyfao56"`, `"pyfao56_available": true`.** Set
`GEO4IRR_ET_ENGINE=builtin` and restart to force the other one.

And the same comparison over a real field and real weather:

```bash
curl -s -X POST localhost:8000/api/et/compare -H 'Content-Type: application/json' \
  -d '{"field":{"name":"Home pivot","acres":132,"lat":38.6431,"lon":-75.4664,
       "soil_key":"evesboro","crop_key":"corn_grain","planting_date":"2026-04-28"},
       "start":"2026-04-28","end":"2026-08-01"}' | python3 -m json.tool
```

Add `"detail": true` for the day-by-day table. The `notes` field explains any
divergence in plain language.

---

## 5. The API

Two ways in. The quickest needs no server at all:

```python
# cd backend, then:
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)

print(c.get("/api/health").json())
print(len(c.get("/api/reference/crops").json()),
      len(c.get("/api/reference/soils").json()))
print(c.get("/api/reference/stations?lat=38.64&lon=-75.47").json()[0])
```

**Expected: 12 crops, 12 soils, and Georgetown 0.3 mi away** at those
coordinates.

Or run it properly:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs> — every endpoint is callable
from the browser there, which is easier than assembling curl bodies by hand.

### Reference data — no network needed

```bash
curl -s localhost:8000/api/health
curl -s localhost:8000/api/reference/soils | python3 -m json.tool | head -20
curl -s "localhost:8000/api/reference/stations?lat=38.64&lon=-75.47" | python3 -m json.tool | head -12
```

### A full advisory

Note the field is `planting_date`, and `use_live_weather: false` keeps this
offline and deterministic:

```bash
curl -s -X POST localhost:8000/api/advisory \
  -H 'Content-Type: application/json' \
  -d '{
    "field": {"name":"Home pivot","acres":132,"lat":38.6431,"lon":-75.4664,
              "soil_key":"evesboro","crop_key":"corn_grain",
              "planting_date":"2026-04-28","capacity_gpm":800,
              "application_efficiency":0.85,
              "min_application_in":0.25,"max_application_in":1.0},
    "as_of":"2026-07-20","use_live_weather":false}' | python3 -m json.tool
```

**Expected, exactly:**

```
decision              irrigate_now
headline              Irrigate today
depletion_in          1.87        (against raw_in 1.01, taw_in 2.52)
recommended_net_in    1.00        (capped at max_application_in)
recommended_gross_in  1.18        (net ÷ 0.85 efficiency)
run_hours             87.4
dap                   83
growth_stage          Reproductive, most sensitive to stress
cost_estimate_usd     412.46
weather_source        climatology
```

Check the hydraulics by hand: 1.18 gross in × 132 ac × 27,154 gal/ac-in ÷ 800
gpm ÷ 60 ≈ **87.8 hours**. The API says 87.4. If it says 8 or 800, a unit
conversion broke.

Two things in that response are worth reading rather than skipping:

`season_to_date` reports **49 stress days and 49% estimated yield loss**. That is
correct and is the point — this request logs no irrigation at all, so it is
telling you what happens to Evesboro corn on rainfall alone. Add irrigation
events to `irrigation_log` and watch it fall.

`weather_source: climatology` means live DEOS and NWS were switched off. With
`use_live_weather: true` and a network it becomes `observed`. Never trust an
advisory that says `climatology` for a real decision.

### Boundary → acres, station, soil

```bash
python3 -c "
import json, pathlib
gj = json.loads(pathlib.Path('samples/home_pivot.geojson').read_text())
pathlib.Path('/tmp/body.json').write_text(json.dumps({'geometry': gj, 'samples': 16}))
"
curl -s -X POST localhost:8000/api/geo/field \
  -H 'Content-Type: application/json' -d @/tmp/body.json | python3 -m json.tool
```

**Expected: `acres` 132.5, centroid 38.6431 / −75.4664, `vertices` 96, weather
naming Georgetown at 0.0 mi**, and a `soil.components` list whose fractions sum
to 1.0. The soil part reaches out to USDA and takes a few seconds. Add
`"resolve_soil": false` to skip it and test the geometry alone.

If USDA is down, `soil.ok` comes back `false` with a warning telling the grower
to pick a series by hand. That is designed behaviour, not a failure.

### Nitrogen and sensors

```bash
curl -s -X POST localhost:8000/api/nitrogen/plan -H 'Content-Type: application/json' \
  -d '{"crop_key":"corn_grain","yield_goal":210,"previous_crop":"soybean_full",
       "cover_crop":"cereal_rye",
       "manure":{"source_key":"broiler_litter","rate_ton_per_ac":3.0,
                 "incorporation":"within_24h","years_since_first_application":3},
       "psnt_ppm":18,"soil_awc_in_per_in":0.07}' | python3 -m json.tool
```

**Expected: 220.5 required, 204.4 credited, 16.1 to apply, 4 splits, risk
`high`.** Same numbers as section 3.

```bash
curl -s -X POST localhost:8000/api/sensors/design -H 'Content-Type: application/json' \
  -d '{"width_ft":2600,"height_ft":2600,"spacing_ft":400,"budget_usd":3000,
       "root_depth_in":30,"soil_key":"evesboro"}' | python3 -m json.tool
```

**Expected: 2 stations, $2,200, expected uncertainty 0.385 in against 0.55 in
unsensed**, plus a `tier_comparison` showing what the same budget buys in
cheaper and dearer probes. On variable Delaware ground more mid-tier probes
usually beat one excellent one, and that table is where you can see it.

### Satellite

```bash
curl -s localhost:8000/api/satellite/status | python3 -m json.tool
```

**Expected without credentials: `"configured": false`** plus setup instructions.
That is the right answer until section 7.

## 6. The web app by hand

```bash
open frontend/index.html        # or xdg-open / start
```

No server needed. Work through these; each one has a specific thing to look for.

### 6.1 It loads

Three example fields — Home pivot, Airport field, Back forty. Each shows a
decision chip and a small tank graphic. **Look for:** no blank cards, no `NaN`,
no `undefined`.

### 6.2 A field's advisory

Tap **Home pivot**. **Look for:** a headline that reads like a sentence a person
would say; the tank graphic joined to a seven-day forecast band with a dashed red
stress line running across both; run hours that make sense against the acreage.

Without internet the forecast falls back to climatology and the app says so in a
note. That note appearing offline is correct.

### 6.3 Log something and watch it move

In **Log what happened**, enter 1.20 in of rain today, save. **Expected:** the
decision moves toward `hold` or `too wet`, the tank refills, and season rain
increases by 1.2. Then log 0.80 in of irrigation and confirm season irrigation
moves by 0.80 gross.

### 6.4 Draw a boundary

**Add a field** → the map appears on satellite imagery. Tap **Draw boundary**,
tap four corners around a field, tap **Finish**.

**Look for:**
- the acre badge updating as you place points
- acres, latitude and longitude filling themselves into the form
- the nearest DEOS station line appearing
- orange handles you can drag, with acres following

### 6.5 Upload each sample file

Same screen, **Upload boundary file**. Try each file from `samples/` and check
against the table in section 4. The two that matter most:

- `wrong_projection.zip` → **must warn** about the projected coordinate system
- `points_not_boundary.geojson` → **must refuse** with a sentence about boundary
  exports, not a stack trace

### 6.6 Soil comes back for the polygon

After a boundary lands in Delaware, the app queries USDA. **Look for:** a
composition bar, one row per map unit with a percentage, and the soil dropdown
pre-set to the dominant series.

If the field straddles two soils that behave differently, a warning appears with
two buttons — *schedule to the lighter ground* or *use the area-weighted
average*. **Look for:** it explaining the tradeoff rather than choosing for you.

Browser console note: USDA Soil Data Access may refuse a direct browser request
on CORS. That is why the server path exists. With `API_BASE` set this goes
through your API and works regardless.

### 6.7 Nitrogen and sensors tabs

**Nitrogen** with 210 bu goal, 3 ton broiler litter, 3 years, PSNT 18, and 6 ppm
well nitrate → four splits and a phosphorus warning. The browser also credits
nitrate in the irrigation water, which the bare API example does not, so it will
show a slightly larger credit and a smaller amount to apply than the 16.1 lb in
section 3. That difference is the well-water nitrate credit and should be
roughly 6 ppm × 8 in ≈ 12 lb N/ac.

**Sensors** → a placement diagram and a station count that rises with acreage,
budget and soil variability rather than sitting at a fixed number.

### 6.8 Satellite tab

Without `API_BASE` it explains how to switch imagery on. That is the expected
state until section 7.

### 6.9 Score tab — the WNIPI season score

This is Eq 6 of the narrative. The proposal frames it around the CSU TAPS
competition, where a committee fixes one baseline for every team; Delaware has
no TAPS, so the baseline is what Eq 6 actually specifies — *baseline
management*, meaning what the grower was doing before.

Open **Score**. Before touching anything:

- **Look for:** the *Season reference ET* box already filled in, with a note
  saying it was added up from DEOS weather for this field and giving the date
  window. That is the term a grower cannot produce for themselves, and it is
  the reason the screen can exist. On the seeded corn field it lands near
  **17.6 in** for a season run to today's date — it grows as the season does,
  so the exact number depends on when you run it.
- **Look for:** *Irrigation applied* pre-filled from what the field has logged.
- Open the *Crop uptake* section. **Look for:** values that match the crop —
  0.90 lb N and 0.37 lb P2O5 per bushel for corn, 3.80 and 0.84 for soybean.
  These derive ANU0 and APU0 so nobody is asked for a tissue test they have
  never run.

Now drive it. Set *This season* 230, *Your usual* 200, ET 24, uptake 0.90/0.37,
and score each row:

| Irrigation | N | P2O5 | Expected score | Expected verdict |
|---|---|---|---|---|
| 4 in | 150 | 50 | **0.0419** | Strong |
| 12 in | 240 | 90 | **0.0193** | Solid |

Both rows are the same 15 percent yield gain. **The point of the metric is that
they do not score the same** — the second bought the identical yield with three
times the water and nearly twice the nitrogen. If both rows give the same
number, Eq 6 is not being applied.

Then set *This season* to 180, irrigation 6, N 180, P2O5 60 → **−0.0221**,
"Negative". A season that lost yield against baseline must score below zero.

Cross-check any of these against the server, which must agree exactly:

```bash
curl -X POST http://localhost:8000/api/metrics/wnipi \
  -H 'Content-Type: application/json' \
  -d '{"yield_farm":230,"yield_baseline":200,"irrigation_in":4,"eto_in":24,
       "n_applied":150,"n_uptake_baseline":180,"p_applied":50,
       "p_uptake_baseline":74}'
```

`n_uptake_baseline` is 200 × 0.90 and `p_uptake_baseline` is 200 × 0.37 — the
same derivation the browser does. The agreement is pinned by
`tests/test_frontend_wnipi_mirror.py`, which extracts the real JavaScript out of
`index.html` and runs it under node.

Finally, blank *Your usual* and score. **Look for:** a refusal explaining that
without a baseline there is nothing to compare against, not a division by zero
and not a score of 0.

### 6.10 Shapefile upload through the API

The browser reads `.shp` geometry itself, but yield values live in the `.dbf`,
which it does not read. With the API running:

```bash
curl -X POST http://localhost:8000/api/ingest/shapefile \
  -F 'file=@samples/airport_field_shapefile.zip'
```

**Look for:** `n_features: 1`, `geometry_types: ["Polygon"]`, `epsg: 4326`.

```bash
curl -X POST http://localhost:8000/api/ingest/shapefile \
  -F 'file=@samples/wrong_projection.zip'
```

This sample is deliberately mislabelled: it holds latitude and longitude but
carries a UTM `.prj`. **Look for:** `epsg: 4326` and a warning that the `.prj`
claims UTM zone 18N but the coordinates are already geographic, so the `.prj`
was ignored. The vertices must still land on Delmarva — roughly −76 to −75
longitude, 38 to 40 latitude.

A `.prj` is a claim, not a guarantee, and export tools write a projected `.prj`
beside unprojected coordinates all the time. Two versions of this went wrong
during development: first the reprojection swapped latitude and longitude, and
then, once fixed, it believed the lying `.prj` and reprojected degrees as though
they were metres. Both put the field in the ocean. If this sample comes back
with a longitude near −79 and a latitude near 0, the magnitude guard has
regressed.

```bash
echo "not a zip" > /tmp/bad.zip
curl -X POST http://localhost:8000/api/ingest/shapefile -F 'file=@/tmp/bad.zip'
```

**Look for:** HTTP 400 and text telling the grower what to do, not a stack
trace.

---

## 7. Satellite imagery (optional, free)

1. Register at <https://dataspace.copernicus.eu>.
2. Sentinel Hub dashboard → **User Settings** → create an OAuth client.
3. Export the credentials and restart the API:

```bash
export CDSE_CLIENT_ID=your-client-id
export CDSE_CLIENT_SECRET=your-client-secret
uvicorn app.main:app --reload --port 8000
```

```bash
curl -s localhost:8000/api/satellite/status | python -m json.tool
```

**Expected: `"configured": true`** and a list of eight layers.

Then, with a boundary in hand:

```bash
BODY=$(python -c "
import json,pathlib
gj=json.loads(pathlib.Path('samples/home_pivot.geojson').read_text())
print(json.dumps({'geometry':gj,'start':'2026-06-01','end':'2026-08-01'}))
")
curl -s -X POST localhost:8000/api/satellite/scenes \
  -H 'Content-Type: application/json' -d "$BODY" | python -m json.tool
```

**Expected:** roughly 10–14 passes over two months (Sentinel-2 revisits every
five days), each with a cloud percentage, and `clear_count` well below the total.
On Delmarva in summer, half or more of the passes being cloudy is normal and is
the reason the radar layer exists.

In the web app, set `API_BASE` at the top of `index.html`, reload, and open
**Satellite** on a field with a boundary. **Look for:**

- a date picker defaulting to the clearest recent pass
- the index image clipped to the field outline, transparent outside it
- grey inside the field where cloud was masked, not green
- a legend that runs the right way — on NDMI and MSI, **dry must read red**
- a season trace with a field-mean value
- an NDVI panel saying whether the canopy is ahead of or behind the calendar,
  with the crop coefficient it handed the water balance

Test the evalscripts without touching the network:

```bash
cd backend
python -c "
import sys, json; sys.path.insert(0,'.')
from app.engine import satellite as sat
for k, i in sat.INDICES.items():
    for name, s in (('visual', i.visual), ('stats', i.stats)):
        if s: assert s.count('{') == s.count('}'), (k, name)
print('all evalscripts balanced')
print([i['key'] for i in sat.index_catalog()])
"
```

Check the canopy-to-schedule bridge:

```python
from datetime import date
from app.engine import satellite as sat
print(sat.canopy_observation(0.74, 0.04, date(2026,6,20), model_kcb=0.70)["verdict"])
print(sat.canopy_observation(0.34, 0.05, date(2026,6,20), model_kcb=0.90)["verdict"])
print(sat.canopy_observation(0.86, 0.03, date(2026,7,15), model_kcb=1.10)["sigma_kcb"])
print(sat.canopy_observation(0.55, 0.03, date(2026,6,10), model_kcb=0.75)["sigma_kcb"])
```

**Expected:** "ahead of the calendar", then "behind the calendar", then a larger
sigma for the saturated scene than the clean one. That last comparison is the
important one — a saturated NDVI must not be allowed to overrule a water balance
that knows better.

---

## 8. Full stack

```bash
docker compose up --build
```

Postgres loads `infra/schema.sql` on first boot; the API waits on its health
check. Everything at <http://localhost:8000>.

```bash
docker compose exec db psql -U geo4irr -d geo4irr -c '\dt'
docker compose exec db psql -U geo4irr -d geo4irr -c '\d+ daily_balance'
```

**Expected:** the tables listed, and `daily_balance` shown as partitioned by
season with `daily_balance_2026` through `2028` present.

Note the API is currently stateless — it computes from the request and returns.
The models in `db.py` and the schema exist but the endpoints do not yet write to
them. Wiring that up is the first real task; see the README.

---

## 9. Frontend without a browser

If you have Node 18+:

```bash
npm install jsdom
```

```javascript
// render.js — renders the app headlessly and reports runtime errors
const fs = require('fs');
const { JSDOM } = require('jsdom');
const html = fs.readFileSync('frontend/index.html', 'utf8');
const errors = [];
const dom = new JSDOM(html, { runScripts: 'dangerously', url: 'http://localhost/',
  pretendToBeVisual: true,
  beforeParse(w) {
    w.fetch = () => Promise.reject(new Error('offline'));
    w.addEventListener('error', e => errors.push(e.message));
  }});
setTimeout(() => {
  const w = dom.window, d = w.document;
  for (const h of ['#/', '#/add', '#/nitrogen', '#/sensors', '#/about']) {
    w.location.hash = h;
    w.dispatchEvent(new w.Event('hashchange'));
    const t = d.getElementById('view').textContent;
    console.log(h.padEnd(12), t.length > 40 ? 'rendered' : 'EMPTY');
  }
  console.log(errors.length ? 'ERRORS: ' + errors.join('; ') : 'no runtime errors');
}, 800);
```

```bash
node render.js
```

**Expected:** every route rendered, no runtime errors.

Syntax-check the embedded script on its own:

```bash
python3 -c "
import re, pathlib
js = re.search(r'<script>(.*)</script>', pathlib.Path('frontend/index.html').read_text(), re.S).group(1)
pathlib.Path('/tmp/app.js').write_text(js)"
node --check /tmp/app.js
```

---

## 10. When something fails

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: app` | Running pytest from the wrong directory | `cd backend` first |
| 22 tests skipped | pyfao56 not installed | `pip install "pyfao56 @ git+https://github.com/isukendall/pyfao56.git"` — optional |
| `et_engine: builtin` unexpectedly | pyfao56 missing, or `GEO4IRR_ET_ENGINE` set | Check `pip show pyfao56` and `env \| grep GEO4IRR` |
| `/api/et/compare` returns 501 | pyfao56 not installed on the server | Same install; it is an optional dependency |
| Tests pass but the API 500s | Old `__pycache__` | `find . -name __pycache__ -exec rm -rf {} +` |
| `soil.ok: false` | USDA Soil Data Access down or slow | Retry; the pick list still works. Not a bug |
| Soil lookup fails in the browser, works via curl | CORS on the USDA endpoint | Set `API_BASE` so it routes through your API |
| Map area blank | Leaflet CDN unreachable | Needs internet. Upload and manual entry still work |
| Shapefile upload fails | Browser lacks `DecompressionStream` | Chrome, Edge, Firefox 113+, Safari 16.4+ — or upload the `.shp` alone |
| Boundary lands in the ocean | Shapefile in state plane or UTM | Re-export as WGS84 geographic. The app warns about this |
| `satellite/status` says not configured | `CDSE_*` not exported | Export before starting uvicorn; check with `env \| grep CDSE` |
| Copernicus 401 | Wrong client or old endpoint | Credentials must be from dataspace.copernicus.eu, not the commercial services.sentinel-hub.com |
| Forecast unavailable | NWS API rate limit or outage | Falls back to climatology and says so. Not a bug |
| `docker compose` port conflict | 5432 or 8000 in use | Change the host-side port mapping |

---

## 11. What is *not* tested, and matters

Be clear-eyed about this before anyone acts on the output.

**No validation against Delaware field data.** The balance now runs through
pyfao56, which *is* validated — against Maricopa lysimeter records, in Arizona,
on cotton. That is a real gain over an unchecked implementation and it is not the
same as being right for corn on Evesboro sand in Sussex County. Every test here
checks internal consistency and behaviour, not agreement with Delaware reality. Nothing has been compared to a lysimeter, an atmometer,
or a probe trace in this state. A season of hindcasts against UD Carvel Research
and Education Center records, and the atmometer work out of Georgetown, Harbeson
and Seaford, is the missing piece — and it is the piece that decides whether any
of this is usable.

**No live Sentinel Hub call has been made.** The evalscripts have been executed as
JavaScript against synthetic pixels and produce correct values, and the request
bodies follow the documented API. But nobody has run a real scene through with
real credentials. Expect to debug the first call.

**DEOS parsing is deliberately tolerant** because the endpoint has changed over
the network's life. Get a data-sharing arrangement with the University of
Delaware, pin the contract, then tighten the parser.

**The nitrogen module has not been reviewed by an agronomist.** The credits are
book values. Broiler litter varies enormously with bedding and house management.
UD Cooperative Extension and the Nutrient Management Program should sign off
before a grower sees a number, and an operation applying nutrients to 10 or more
acres still needs a certified plan regardless.

**No load testing.** The nightly batch is designed for it and has never met
20,000 fields.
