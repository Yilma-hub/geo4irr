# Where the code came from, and what that means for publishing it

Read this before pushing Geo4Irr to a public repository. One part of it carries
a licence obligation that decides the licence of the whole project.

## The short version

`backend/app/engine/metric.py` is a port of Hector Nieto's pyMETRIC and of the
pyTSEB library it depends on. Both are published under the GNU General Public
License, version 3. A port is a derivative work, not an independent
reimplementation, so Geo4Irr inherits that licence when it is distributed. The
`LICENSE` file at the root of this repository is GPL-3.0 for that reason.

That matches the intent for this tool, which is free software. It is worth
knowing what it commits you to:

- Anyone you give the software to, or who downloads it, may use, study, change
  and redistribute it.
- If you distribute a changed version, the changed version is GPL-3.0 too and
  you have to offer its source.
- You cannot relicense it as closed source, and you cannot fold it into a
  closed-source product.
- Running it yourself, on your own fields, with your own data, carries no
  obligation at all. The licence is about distribution, not use. Nothing here
  stops a private repository or a private deployment.

Signing in is a separate matter from the licence. The code being free does not
make your account open: authentication, your field boundaries and your weather
subscription are all configuration, and none of them is in this repository.
See `.env.example`.

## One ambiguity in the upstream licence

pyMETRIC and pyTSEB are inconsistent about which GPL they mean. The `LICENSE`
file in each repository is the full GPL-3.0 text, while the header comment at
the top of each source file says "GNU Lesser General Public License". Those are
different licences with materially different obligations, and the two
repositories have shipped that way for years.

GPL-3.0 is the stricter of the two and is what the `LICENSE` file says, so it
is what this project assumes. If you wanted the weaker obligation you would
have to ask the author which was intended; and for a port rather than a
library call, LGPL section 2 lands in much the same place anyway, because this
is a modified version of the work rather than an application linking against
it.

If you ever need Geo4Irr under a permissive licence, the only clean route is to
rewrite `metric.py` from the published papers alone, without reading the
pyMETRIC source. Equations in a journal article are not copyrightable; a
particular arrangement of them in code is. That is a real piece of work and not
worth doing unless something actually requires it.

## What was ported, specifically

From `pyMETRIC/pyMETRIC/METRIC.py`:

- `calc_G_Allen`, the soil heat flux ratio (Allen 2007 Eq 26)
- `calc_H`, `calc_H_residual`, `calc_dT` (Eqs 28, 46, 49)
- the arrangement of the dT line's slope and intercept, with the author's note
  that Eqs 50 and 51 as printed in Allen et al. (2007) appear to be wrong
- `calc_Ln`, `calc_cloudiness`, `wind_profile`
- the structure of `pet_asce` and `pet_fao56`
- the two-stage iteration: Monin-Obukhov length at the anchors first, then per
  pixel
- the METRIC resistance convention, z_T 2.0 m, z_0H 0.1 m, d_0 0.0

From `pyMETRIC/pyMETRIC/PyMETRIC.py`:

- the Allen (2013) anchor fractions, Eq 4 for the cold pixel and Eq 5 for the
  hot one, and `VI_MAX` of 0.95

From `pyMETRIC/pyMETRIC/endmember_search.py`:

- `cimec`, the Calibration using Inverse Modelling at Extreme Conditions, step
  by step including the albedo threshold of Allen (2013) Eq 7 and the rainfall
  adjustment of Eq 8
- `maxmin_temperature`, the simple coldest-green against hottest-bare search
- `moving_cv_filter`, as a loop rather than scipy's `convolve2d`

From `pyTSEB/pyTSEB/MO_similarity.py`:

- `calc_u_star`, `calc_L` through `calc_mo_length_hv`, `calc_Psi_M`, `calc_Psi_H`
- `psi_m_brutsaert` and `psi_h_brutsaert` with their exact coefficients

From `pyTSEB/pyTSEB/resistances.py`:

- `calc_R_A`

From `pyTSEB/pyTSEB/meteo_utils.py`:

- `calc_pressure`, `calc_vapor_pressure`, `calc_delta_vapor_pressure`,
  `calc_c_p`, `calc_lambda`, `calc_rho`, `calc_psicr`, `calc_mixing_ratio`,
  `calc_lapse_rate_moist`, `calc_stephan_boltzmann`, `flux_2_evaporation`

## Where this port departs from pyMETRIC

Four places, all documented in `metric.py` and pinned by tests in
`backend/tests/test_metric.py`.

1. **Net longwave sign in the reference ET.** pyMETRIC's `pet_asce` computes
   net outgoing longwave as a positive number and then adds it to net
   shortwave. ASCE subtracts it. At 298 K, 20 mb vapour pressure and 800 W/m2
   incoming shortwave the difference is 616 W/m2 against 680, so reference ET
   comes out roughly a fifth high; since METRIC calibrates against reference
   ET, every actual ET goes with it. This port subtracts, and writes the ASCE
   standardized hourly equation in its canonical form rather than pyMETRIC's
   rescaled one, which also drops the 1 inside the denominator's bracket.

2. **The aerodynamic resistance inside the per-pixel loop.** In pyMETRIC's
   `METRIC()` the call that fills `R_A` sits inside the `else` branch, so with
   `use_METRIC_resistance=True`, which is the default, `R_A` is never assigned
   after initialisation and every sensible heat flux is NaN. This port computes
   it in both branches.

3. **The wind profile.** pyMETRIC corrects wind from whatever height its
   configuration names. Allen (2007) Eqs 30 to 33 go through a 200 m blending
   height first, which is the argument for applying one weather station across
   a whole scene. That step is written out here.

4. **Where the cold anchor's ET comes from.** pyMETRIC has no weather station
   so it computes reference ET internally. This tool has one, the nearest DEOS
   station, which is also the station the FAO-56 side uses. Allen's METRIC
   manual calibrates against station ETr, so the station value is preferred and
   the internal ASCE equation is the fallback. Using one station for both is
   also what makes the two methods comparable.

Neither of the first two is reported as a bug upstream as far as I can tell. If
you have the time, reporting them would be the decent thing: they are both
small fixes and the second one stops the default configuration working at all.

## The science

The code is a port; the method is not mine or pyMETRIC's.

- Allen, R.G., Tasumi, M., Trezza, R. (2007). Satellite-based energy balance
  for mapping evapotranspiration with internalized calibration (METRIC):
  Model. *Journal of Irrigation and Drainage Engineering* 133(4), 380-394.
- Allen, R.G., Tasumi, M., Morse, A., Trezza, R., Wright, J.L., Bastiaanssen,
  W., Kramber, W., Lorite, I., Robison, C.W. (2007). Satellite-based energy
  balance for mapping evapotranspiration with internalized calibration
  (METRIC): Applications. *JIDE* 133(4), 395-406.
- Allen, R.G., Burnett, B., Kramber, W., Huntington, J., Kjaersgaard, J.,
  Kilic, A., Kelly, C., Trezza, R. (2013). Automated calibration of the METRIC
  Landsat evapotranspiration process. *JAWRA* 49(3), 563-576.
- Tasumi, M., Allen, R.G., Trezza, R. (2008). At-surface reflectance and albedo
  from satellite for operational calculation of land surface energy balance.
  *Journal of Hydrologic Engineering* 13(2), 51-63.
- Bastiaanssen, W.G.M. (2000). SEBAL-based sensible and latent heat fluxes in
  the irrigated Gediz Basin, Turkey. *Journal of Hydrology* 229, 87-100.
- Brutsaert, W. (2005). *Hydrology: an Introduction*. Cambridge, for the
  stability functions pyTSEB uses by default.
- Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). Crop
  evapotranspiration: guidelines for computing crop water requirements. FAO
  Irrigation and Drainage Paper 56, for the dual crop coefficient balance that
  runs alongside METRIC.
- ASCE-EWRI (2005). The ASCE standardized reference evapotranspiration
  equation.
- Thorp, K.R. (2022). pyfao56: FAO-56 evapotranspiration in Python.
  *SoftwareX* 19, 101208. Used as the engine of record for the water balance.
- Delaware Geological Survey, Report of Investigations No. 84: estimating
  evapotranspiration in Sussex County from Landsat 8, the 2016 growing season.
  The closest published application of METRIC to this ground.
- Delaware Environmental Observing System (DEOS), University of Delaware, for
  weather and for the Delaware Irrigation Management System user guide, which
  supplied the texture classes, the managed root zone, the emergence model and
  the status bands in `backend/app/engine/dims.py`.
- USDA-NRCS SSURGO, through Soil Data Access, for soils.
- USGS Landsat Collection 2 Level-2, through Element 84's Earth Search index
  and a public TiTiler instance, for the imagery. Neither needs an account.

## Other code in this repository

Everything outside `metric.py` was written for this project. `dims.py` encodes
published Delaware reference values but is not a port of any code, since DIMS
does not publish source. The frontend is original.

pyfao56 is a dependency rather than a port: it is installed from PyPI and
called, not copied. Its package metadata gives its licence as Public Domain,
which fits a USDA-ARS work, and it imposes no obligation on this project.
