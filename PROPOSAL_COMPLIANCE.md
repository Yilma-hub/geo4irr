# Proposal compliance

Every objective, activity, equation and named method in the project narrative,
mapped to where it is implemented and how it was verified. Where something is
partial or absent, it says so rather than claiming coverage.

Delaware substitutions are noted where a Colorado or New Mexico data source has
a regional equivalent — the method is the narrative's, the endpoint is local.

---

## Equations

| | Equation | Implementation | Verified by |
|---|---|---|---|
| **Eq 1** | `D = D0 + ETc − P − Irr − U + qs + qw` | `fao56.step_water_balance` for the balance; `swatplus.FieldProcessModel` supplies `U`, `qs`, `qw` | `test_field_process_model_returns_every_eq1_flux_term`, `test_upflux_falls_off_with_depth_to_the_water_table` |
| **Eq 2** | `ETc = ETo · Kc` | `pyfao56` via `pyfao56_bridge`; `fao56.reference_et` for ETo | ETo agrees with pyfao56 to **0.01%** across hot, cool, humid and windy cases |
| **Eq 3** | `ETc_adj = ETo (Kcb·Ks + Ke)` | `fao56.step_water_balance`, dual coefficient with full TEW/REW evaporation | `test_engines_agree_on_the_crop_coefficient_curve` |
| **Eq 4** | `L(x\|y) = p(y\|x) = p(x\|y)p(y)/p(x)` | `likelihood.bayes_update`, `likelihood.evidence` (the `p(y)` denominator) | `test_bayes_update_is_the_exact_conjugate_posterior` |
| **Eq 5** | `log L = −(n/2)log(2π) − Σ log σᵢ − ½ Σ ((yᵢ−ỹᵢ)/σᵢ)²` | `likelihood.log_likelihood` | Matches the analytic Gaussian to 1e−9 |
| **Eq 6** | WNIPI | `nitrogen.water_nutrient_index` | Verified term by term against the narrative |

Eq 5's correlated-residual case — "modified by appropriate transformations" — is
`likelihood.ar1_whiten` for temporal correlation and
`likelihood.log_likelihood_correlated` (Cholesky) for the general covariance.

---

## Objectives

| Obj | | Status |
|---|---|---|
| 1 | Extension co-design | **Out of scope for software.** The schema supports it: `advisory_records.followed` is the evaluation backbone, and `v_adoption` reports what fraction of recommendations were acted on |
| 2 | Science-guided ML assimilation | **Implemented** — `likelihood.py`, `bayesian.py` |
| 3 | Tradeoff characterization | **Implemented** — `optimization.py` |
| 4 | Pilot studies | **Out of scope.** `TESTING.md` §11 sets out what validation must happen first |
| 5 | Service-oriented architecture | **Implemented** — FastAPI, documented OpenAPI, engine modules independently importable |
| 6 | Adoption | **Partial** — the web and mobile app exists; the SMS delivery path is designed but not built |
| 7 | Education | **Partial** — `README.md`, `TESTING.md`, and the in-app *How it works* page |

---

## Activities

### Activity 1 — Co-design engagement
Not software. The database records what was recommended, whether it was seen and
whether it was followed, which is what the co-design and evaluation work needs.

### Activity 2 — Farm/field setup and data acquisition

| Narrative requirement | Implementation | Notes |
|---|---|---|
| Boundary from shapefile / KML upload | `geo.parse_geojson`, `parse_kml`; browser `parseShapefileZip`, own zip and `.shp` readers | Verified against generated shapefiles: 96.5 ac against a 96 ac target |
| Heads-up digitizing | `frontend/index.html` — draw on NAIP/Esri imagery, draggable vertices, live acreage | |
| **Computational grids, 5 ft × 15 ft for VRI** | `vri.discretize`, `GRID_SPECS["vri_individual"]` | Exactly the narrative's figure. A 132 ac pivot → **76,499 cells**, collapsed to response units so it computes in about a second |
| Management zones | `vri.build_response_units`, `geo.suggest_zones`, `management_zones` table | |
| Cropping system from NASS CDL | `ingest.fetch_cdl_crop`, `CDL_TO_CROP` | 30 m gridded, as specified |
| Soils from SSURGO | `soils.lookup_ssurgo`, `geo.ssurgo_for_boundary` | Area-weighted over the polygon, not the centroid |
| Climate from station networks | `weather.py` — **DEOS** replaces CoAgMET/ZiaMet | |
| Gridded Daymet / PRISM | `ingest.fetch_daymet`, `PRISM_URL` | Daymet has no wind; flagged in the response |
| User spreadsheet upload | `ingest.read_weather_csv` | Loose column matching, Celsius auto-detected |
| Water diversions from state databases | `ingest.fetch_water_allocation` — **DNREC allocation permits** replaces Colorado DSS | Delaware permits any withdrawal above 50,000 gal/day |

### Activity 3 — Water balance and crop ET parameters

| Requirement | Implementation |
|---|---|
| **pyFAO56 incorporated** | `pyfao56_bridge.py` — a real dependency, not a reimplementation. Parameters, Weather, Irrigation, SoilProfile and Update classes all populated |
| Kcb from fractional canopy cover | `fao56.kcb_from_canopy_cover` |
| Kcb from NDVI | `fao56.kcb_from_ndvi`, fed by `satellite.canopy_observation` |
| Ks from CWSI | `fao56.ks_from_cwsi` |
| **Ks from DANS** | `fao56.ks_from_dans`, `dans_index`, `dans_from_thermal_image` |
| Assimilate Kcb into pyFAO56 | `pyfao56_bridge.build_updates` → pyfao56's `Update` class, the intended door |
| Soil moisture and EC from probes | `bayesian.Observation`, `SensorClass`, `sensor_readings` table |
| UAV NDVI, biomass, canopy temperature | `ingest.read_geotiff`, `vri.attach_canopy` |
| Remote sensing NDVI, ETc, yield | `satellite.py` — Sentinel-2 and Sentinel-1 |
| Spatially variable yield monitoring | `ingest.yield_from_features`, `yield_from_raster` |
| **ASCII** sensor data | `ingest.read_sensor_ascii` — vendor-agnostic column sniffing |
| **Multidimensional NetCDF** | `ingest.read_sensor_netcdf` — classic CDF-1/CDF-2 parsed directly; NetCDF-4 says it needs the library rather than guessing |
| **GeoTIFF** raster | `ingest.read_geotiff` — header, georeferencing, strips, packbits; UTM 18N reprojection |
| Yield in raster and feature class | `ingest.yield_from_raster`, `yield_from_features` |

### Activity 4 — Measurement error characterization

`likelihood.characterize_error` — bias, precision, RMSE, R², AR(1) ρ, and the
effective independent sample size. Bias and precision are reported separately
because they are different problems: a systematic offset is calibrated out, a
wide error distribution can only be down-weighted. Output feeds σ in Eq 5.
`bayesian.SensorClass` carries the five sensor tiers with their error models.

### Activity 5 — Bayesian short-term advisory

| Requirement | Implementation |
|---|---|
| Recursive Bayesian assimilation | `bayesian.BayesianAssimilator` |
| Statistically correct likelihood (Eq 5) | `likelihood.log_likelihood` |
| Correlated residual transformation | `likelihood.ar1_whiten`, `log_likelihood_correlated` |
| **Generative filtering for streaming data** | `likelihood.GenerativeFilter` — sequential Monte Carlo, systematic resampling, joint state and parameter estimation |
| Posterior parameter distributions | The filter estimates `Kcb` bias and `Ks` alongside depletion |
| Simulate ETc from FAO-56 tabular Kc | `crops.py` trapezoidal curves through pyfao56 |
| Update Kcb from canopy cover | `kcb_from_canopy_cover`, assimilated via `Update` |
| Estimate Ks from canopy temperature | `ks_from_cwsi`, `ks_from_dans` |
| Water balance with SWAT+ fluxes | `swatplus.FieldProcessModel` |
| Reset D from soil moisture observations | `bayesian.depletion_from_vwc` with 4σ gross-error gating |
| **5th and 95th percentile intervals** | `GenerativeFilter.interval`, `Gaussian.interval` |
| 1–7 day forecasts of stress likelihood, loss and amount | `advisory.project_forecast` — 120-member ensemble |

### Activity 6 — SWAT+ integration

| Requirement | Implementation | Status |
|---|---|---|
| U, qs, qw from SWAT+ | `swatplus.FieldProcessModel` + `SwatPlusProject` | Fast path follows SWAT+ algorithms; real binary used when `SWATPLUS_EXE` is set |
| Decision tables read pyFAO56 schedule | `SwatPlusProject.write_decision_table` → `lum.dtl` with jday conditions | Implemented |
| Irrigation management input files | `write_irrigation_ops` → `irr.ops` | Implemented |
| Furrow / sprinkler / micro in Connect files | `write_connect` → `hru.con`; 7 systems in `SYSTEMS` | Implemented |
| Variable rate, full and deficit strategies | 7 entries in `REGIMES` including `deficit_50` matching the NMSU trial | Implemented |
| **Root depth from canopy cover, not linear** | `swatplus.root_depth_from_canopy` | The narrative's stated fix for SWAT+'s linear root growth |
| Nutrient and salt levels in irrigation water | `FieldProcessModel._transport`, `salinity_yield_penalty` (Maas-Hoffman) | Implemented |
| Soil health metrics under irrigation regimes | `simulate_soil_health` — 9 indicators, calibrated so deficit declines and cover crops build | Implemented |
| **Minimum data set by multivariate analysis** | `minimum_data_set` — PCA by power iteration, highest loading per component | Implemented |
| Calibration and uncertainty analysis tool | `likelihood.py` provides the likelihood; automated SWAT+ parameter calibration is **not implemented** | **Gap** |
| Costs and production value | `economics.py`, `optimization.Outcome` | Implemented |

**Honest limitation.** Running the real SWAT+ executable per field per night is
not viable at scale, so the daily advisory uses the field-scale process model,
which implements SWAT+'s own algorithms — moisture-adjusted curve number,
storage-routing percolation, shallow aquifer with revap and recession.
`swat_available()` and every response report which engine produced the numbers,
and `compare_to_swat` checks one against the other. The real coupling is for the
multi-season and scenario work, which is what the narrative assigns it.

### Activity 7 — Optimal sensor placement

| Requirement | Implementation |
|---|---|
| Number and location at lowest cost | `sensors.design_network` — budget-constrained |
| **Maximum entropy technique** | `sensors._entropy_gain` — Gaussian process conditional variance |
| Eq 5 for device measurement errors | `SensorClass` σ values feed the covariance |
| Reduce redundancy across the network | Conditional variance falls as nearby sensors are chosen |
| **Placement depth by universal MLR** | `sensors.optimal_sensor_depths` — every subset fitted, scored on adjusted R², ridge-regularised for the collinearity that neighbouring depths always have |
| Hybrid accuracy and cost sensors | `compare_sensor_tiers` — more mid-tier probes usually beat one excellent one on variable ground |

### Activity 8 — Communication metrics

`nitrogen.water_nutrient_index` (Eq 6). Activity 8's three categories are the
three NSGA-II objectives: farm profitability, irrigation and fertilization costs,
environmental sustainability.

### Activity 9 — Tradeoffs

| Requirement | Implementation |
|---|---|
| **NSGA-II** | `optimization.py` — fast non-dominated sort, crowding distance, binary tournament on rank then crowding, SBX crossover, polynomial mutation, elitist survival. Not a weighted sum |
| Non-dominated solutions | `fast_non_dominated_sort`; verified no front member dominates another |
| Scenario analysis module | `compare_to_front` — locates the grower's own practice and names what it would cost to move |
| Quantify tradeoffs in yield, costs, conservation | `front_summary` — corners, knee, and the exchange rate per acre-inch |

### Activity 10 — Service-oriented deployment

FastAPI with generated OpenAPI, engine modules importable independently,
Docker/compose, partitioned Postgres. eRAMS itself is CSU infrastructure and is
not reproduced; the architecture is service-oriented so modules can be added the
way the narrative intends.

### Activities 11 and 12 — Pilots and producer deployment

Not software. Delaware equivalents of the pilot sites would be UD Carvel REC and
cooperator farms in Sussex and Kent; `TESTING.md` §11 states what has to happen
before either.

---

## Work added beyond the proposal

These were requested during development and are not in the narrative.

| | Implementation |
|---|---|
| Delaware localization throughout | 12 Coastal Plain soil series, 12 crops with Delmarva planting windows, DEOS weather, DNREC allocation, broiler litter credits, Nutrient Management Law framing |
| Shallow water table upflux | `SoilProfile.upflux_in`, `swatplus.capillary_upflux` — Fallsington and Pocomoke feed themselves in July, and generic schedulers over-irrigate them |
| Browser boundary drawing and file upload | Own zip, `.shp` and KML readers, no libraries |
| Polygon SSURGO with zone recommendation | `geo.ssurgo_for_boundary`, `suggest_zones` |
| **Sentinel-2 and Sentinel-1 imagery** | `satellite.py` — 8 layers, cloud masking, NDVI feeding the crop coefficient |
| Standalone offline web app | `frontend/index.html` — the same FAO-56 balance in JavaScript |
| pyfao56 comparison harness | `tools/compare_engines.py`, `POST /api/et/compare` |

---

## Known gaps

1. **SWAT+ parameter calibration and uncertainty quantification** is not
   automated. `likelihood.py` supplies the machinery; the calibration loop
   itself is not built.
2. **The real SWAT+ binary has never been run against this.** The file writers
   follow the documented formats but have not been executed by SWAT+ itself.
   A column-merge defect found in this audit (a right-aligned number written
   flush against a left-aligned string produced `53.419hru`, one token to a
   whitespace-delimited read) is fixed and now regression-tested, but only
   running SWAT+ will confirm the rest.
3. **No Sentinel Hub call with live credentials.** Evalscripts are verified as
   JavaScript against synthetic pixels; the request bodies follow the API docs.
4. **No validation against Delaware field data.** The balance runs through
   pyfao56, which is validated — in Arizona, on cotton. Not the same thing.
5. **API endpoints are stateless.** The schema and models exist; the endpoints
   do not yet write to them.
6. **SMS delivery is designed, not built.**
7. **Frontend covers the daily advisory, nitrogen, sensors, boundary,
   satellite and the WNIPI season score.** Tradeoffs, VRI prescriptions and
   soil health are still API-only.
8. **Shapefile ingest is now server-side as well as in-browser**, but the
   reader handles points, polylines and polygons only. MultiPatch is refused
   with an explanatory message rather than parsed; no yield monitor writes it.


## Audit trail — full re-read of the narrative

The proposal was re-read end to end and each Activity checked against the code
rather than against memory. Findings:

| Check | Result |
|---|---|
| Eq 1, all seven terms reconcile from `DayResult` | **Was failing by 0.147 in** — fixed |
| Eq 5 against the analytic Gaussian | agrees to 4.4e-16 |
| Eq 6 against the narrative, term by term | matches |
| 5 ft x 15 ft VRI grid | exact |
| Ten soil health indicators | all present |
| Furrow / sprinkler / micro | seven systems across the three families |
| NSGA-II front, all pairs tested for dominance | zero dominated pairs |
| SWAT+ files tokenize to stable columns | **was failing** — fixed |
| Shapefile yield ingest, server-side | **was missing** — built |
| Eq 6 (WNIPI) reachable for the TAPS use case | **was optimizer-only** — endpoint added |
| Eq 6 in the frontend, baseline adapted for Delaware | **was absent** — Season score screen added |
| UTM shapefile reprojection | **was swapping lat/lon** — fixed |

Three defects and one missing capability, all closed:

1. **Eq 1 did not close.** Root growth dilutes depletion into newly occupied
   soil. The physics was right but the term was invisible, so nobody auditing an
   advisory could reconcile the balance. Eq 1 as printed assumes a fixed root
   zone; Activity 6 requires dynamic rooting, so the term is now reported as
   `root_growth_in` on both engines rather than folded in silently.
2. **SWAT+ input files were unparseable.** Fixed in all three writers, with a
   test that also exercises a field name long enough to overrun the name column.
3. **UTM shapefiles landed in the Gulf of Guinea.** `utm_to_wgs84` returns
   `(lon, lat)`; the new reprojection path unpacked it as `(lat, lon)`.
4. **WNIPI was computed but not reachable.** Eq 6 was correct and returned
   inside the tradeoffs response, but the only way to obtain one was to run a
   full NSGA-II search. Activity 11 has the CSU TAPS teams calculating the index
   for the season they actually farmed, which is a different request shape, so
   `POST /api/metrics/wnipi` now does it directly. It also had no test coverage;
   it now has nine. The interpretive bands were recalibrated after checking the
   metric's realistic range — the denominator is a product of three terms each
   at least 1, so a strong corn season scores around 0.03 to 0.07, and the
   original 0.05 cutoff would have called almost every real result marginal.
5. **Server-side shapefile ingest was missing.** Activity 3 requires crop yield
   in shapefile format. Geometry parsing existed only as browser JavaScript, and
   yield values live in the `.dbf`, which that code does not read. A stdlib-only
   `.shp` + `.dbf` + `.prj` + zip reader now closes it, exposed at
   `POST /api/ingest/shapefile`.

### Eq 6 in the Delaware frontend

The narrative frames WNIPI around the CSU TAPS competition, where a scoring
committee fixes one baseline for every team. Delaware has no TAPS, so the
baseline is the thing Eq 6 actually specifies — "baseline management", meaning
what the grower was doing before. Their own previous seasons on the same field
is the defensible comparison and the only one they can check.

The term a grower cannot produce for themselves is seasonal reference ET, and
that is supplied automatically: the Season score screen sums ET0 over the
field's own balance run, which is driven by DEOS, the University of Delaware
station network. ANU0 and APU0 are derived from baseline yield times a
crop-specific above-ground uptake rate, so nobody is asked for a tissue-test
number they have never measured; every value stays editable.

The browser carries its own copy of Eq 6 so the score works offline, which is
the standing risk of two implementations drifting. `test_frontend_wnipi_mirror`
extracts the real JavaScript out of `index.html` and runs it under node against
the Python across seven cases including a soybean season with no applied N, a
rain-fed season with no irrigation, and an exactly-level season.

### A .prj is a claim, not a guarantee

Two defects in the shapefile reprojection path, both caught by running the
bundled samples rather than by the unit tests:

1. `utm_to_wgs84` returns `(lon, lat)`; the new code unpacked it as
   `(lat, lon)`, putting every UTM upload off the coast of West Africa.
2. Once that was fixed, the reader still believed a `.prj` that contradicted its
   own coordinates. `samples/wrong_projection.zip` is deliberately mislabelled —
   latitude and longitude under a UTM `.prj` — and the reader reprojected
   degrees as though they were metres, landing the field at 0 degrees north.

The guard is a magnitude check: nothing inside a real UTM zone has an easting
under 180, so coordinates that fit inside degree bounds are geographic whatever
the `.prj` says. The coordinates win and the mismatch is reported.

## Checked against DIMS

DIMS is the Delaware Irrigation Management System, published by the University
of Delaware's Delaware Environmental Observing System (DIMS User's Guide v1.1,
Brinson, 24 May 2012; USDA-NRCS grant 69-21J2-0-12, DNREC grant 08070800964).
It is the state reference scheduler, so it is the right thing to check this
build against.

### What was taken

Soil texture table (Appendix B), twelve USDA classes with field capacity and
wilting point. Used as a fallback where SSURGO returns no named series, as the
reference list behind manual soil entry, and as an independent check on the
twelve Delaware series already in the engine.

Managed root zone (Section IV). Default 18 inches, range 8 to 24. FAO-56 grows
roots to the crop maximum, 36 inches for corn here. The two answer different
questions: how deep the crop can reach, against how deep the grower will run
the pump for. Exposed as an optional cap, off unless set. On Sassafras sandy
loam it halves total available water from 4.32 to 2.16 inches, which moves a
field several points down the status scale.

Emergence by growing degree days (Appendix A, Table 3). 125 GDD on the 86/50
cap for corn and soybean, with an emergence date entered directly for anything
else. The crop step now takes planting, emergence and estimated harvest, which
also matches the eRAMS configuration panel.

Field status bands (Section VII). Good above 75 percent of field capacity, fair
50 to 75, poor 30 to 50, very poor below 30. A percentage carries across
several fields where inches of depletion against a different capacity on each
one does not.

Custom soil entry (Section IV). A lab result for a specific field beats any map
unit. Validated rather than trusted, because a wilting point above field
capacity yields a negative available water capacity and a field that never
irrigates.

### What was deliberately not taken

The crop coefficients in Appendix A. DIMS uses the FAO-56 single crop
coefficient, where one Kc covers transpiration and soil evaporation together
(corn 0.25 / 1.20 / 0.60). This engine uses the dual coefficient the project
narrative specifies, where Kcb is transpiration alone and Ke is tracked
separately (corn 0.15 / 1.15 / 0.50). The numbers look similar and mean
different things. A single Kc dropped into a Kcb slot would count soil
evaporation twice, because Ke is then added on top of a coefficient that
already contains it, and nothing downstream would complain. A test now fails
if the two tables are ever merged.

### Comparing the two methods

`dims.compare_with_dims_single_kc` runs both on identical weather. On Delaware
corn the dual method gives 10 to 26 percent more potential crop ET than the
DIMS single Kc, and the gap widens on heavier soil. Both reasons are expected.
The dual method resolves each wetting event, so rain on bare ground in May
drives Ke close to 1.1, where the single Kc_ini of 0.25 is already an average
over a typical wetting frequency and cannot rise. And DIMS assumes a 110 day
corn season against 140 in this crop table, so the two are on different stage
fractions late in the year.

### Two errors this exercise caught

Texture matching on available water capacity alone. Clay holds 0.43 at field
capacity with 0.29 locked away at wilting point: 0.14 available, the same
figure as a much lighter soil. Matched that way the engine called Woodstown
sandy loam a clay and Greenwich loam a silty clay. Matching now weighs field
capacity and wilting point together, and all twelve Delaware series land on a
sensible class.

Comparing stress-adjusted ET against unstressed ET. The first version of the
method comparison reported a 15.8 percent gap that was almost entirely the
stress coefficient rather than any difference between the methods. Both sides
are potential now, with the stress reduction reported separately.
