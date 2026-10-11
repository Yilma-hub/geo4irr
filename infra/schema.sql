-- Geo4Irr — database schema
--
-- Sizing target: 20,000 growers. Delaware has roughly 2,300 farms statewide, so
-- 20,000 users only makes sense if the tool reaches beyond farm operators to
-- crop consultants, conservation district staff, NRCS field offices, irrigation
-- dealers, and neighbouring Delmarva counties in Maryland and Virginia. Plan
-- the account model for that from the start: a consultant managing 60 fields
-- across 12 operations is a different shape of user than a grower with 4 pivots,
-- and bolting on delegated access later is painful.
--
-- Volume: ~20k users x ~5 fields x ~200 season days = 20M daily_balance rows a
-- year. Partitioned by season below so old years drop cheaply and the hot
-- partition stays in cache.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- ---------------------------------------------------------------------------
-- People and access
-- ---------------------------------------------------------------------------

CREATE TYPE de_county AS ENUM ('New Castle', 'Kent', 'Sussex', 'Other');
CREATE TYPE account_role AS ENUM ('grower', 'consultant', 'extension', 'agency', 'admin');

CREATE TABLE accounts (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role                    account_role NOT NULL DEFAULT 'grower',
    name                    TEXT NOT NULL,
    farm_or_org             TEXT,
    email                   CITEXT UNIQUE,
    phone                   TEXT,
    county                  de_county DEFAULT 'Sussex',
    nm_certification_number TEXT,
    nm_certification_expires DATE,
    sms_alerts              BOOLEAN NOT NULL DEFAULT TRUE,
    alert_hour_local        SMALLINT NOT NULL DEFAULT 5,
    preferred_language      TEXT NOT NULL DEFAULT 'en',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at            TIMESTAMPTZ
);

-- A consultant or district office acting on a grower's fields. Explicit and
-- revocable; never implied by organization membership.
CREATE TABLE field_delegations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id      UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    delegate_id   UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    scope         TEXT NOT NULL DEFAULT 'all_fields',
    can_edit      BOOLEAN NOT NULL DEFAULT FALSE,
    granted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at    TIMESTAMPTZ,
    UNIQUE (owner_id, delegate_id)
);

-- ---------------------------------------------------------------------------
-- Fields
-- ---------------------------------------------------------------------------

CREATE TABLE fields (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id             UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    name                   TEXT NOT NULL,
    acres                  NUMERIC(8,2) NOT NULL CHECK (acres > 0),
    centroid               GEOGRAPHY(POINT, 4326) NOT NULL,
    boundary               GEOGRAPHY(POLYGON, 4326),

    soil_key               TEXT NOT NULL DEFAULT 'sassafras',
    soil_components        JSONB,
    ssurgo_mukey           TEXT,
    awc_override           NUMERIC(5,3),
    restrictive_layer_in   NUMERIC(5,1),
    water_table_depth_in   NUMERIC(5,1),

    crop_key               TEXT NOT NULL DEFAULT 'corn_grain',
    planting_date          DATE,
    yield_goal             NUMERIC(10,2),
    previous_crop          TEXT,
    cover_crop             TEXT NOT NULL DEFAULT 'none',

    system_type            TEXT NOT NULL DEFAULT 'center_pivot',
    capacity_gpm           NUMERIC(8,1) NOT NULL DEFAULT 800,
    application_efficiency NUMERIC(4,3) NOT NULL DEFAULT 0.85,
    min_application_in     NUMERIC(4,2) NOT NULL DEFAULT 0.25,
    max_application_in     NUMERIC(4,2) NOT NULL DEFAULT 1.20,
    vri_capable            BOOLEAN NOT NULL DEFAULT FALSE,
    fertigation_capable    BOOLEAN NOT NULL DEFAULT FALSE,
    well_permit_gpm        NUMERIC(8,1),
    well_nitrate_ppm       NUMERIC(6,2),

    deos_station           TEXT,
    active                 BOOLEAN NOT NULL DEFAULT TRUE,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_fields_account       ON fields (account_id) WHERE active;
CREATE INDEX ix_fields_centroid_gist ON fields USING GIST (centroid);
CREATE INDEX ix_fields_station       ON fields (deos_station) WHERE active;

CREATE TABLE management_zones (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    field_id     UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    acres        NUMERIC(8,2) NOT NULL,
    soil_key     TEXT NOT NULL,
    geometry     GEOGRAPHY(POLYGON, 4326),
    yield_index  NUMERIC(4,2) NOT NULL DEFAULT 1.00
);
CREATE INDEX ix_zones_field ON management_zones (field_id);

-- ---------------------------------------------------------------------------
-- The daily ledger
-- ---------------------------------------------------------------------------
-- Append-only. When a grower asks in November why the tool said to irrigate on
-- July 12th, the answer has to come from what was known on July 12th. Sensor
-- data that arrives later revises the forward state, never the record.

CREATE TABLE daily_balance (
    id                BIGSERIAL,
    field_id          UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    zone_id           UUID REFERENCES management_zones(id) ON DELETE CASCADE,
    day               DATE NOT NULL,
    season            SMALLINT NOT NULL,
    dap               SMALLINT NOT NULL,

    eto_in            NUMERIC(5,3) NOT NULL,
    kcb               NUMERIC(4,3) NOT NULL,
    ke                NUMERIC(4,3) NOT NULL,
    ks                NUMERIC(4,3) NOT NULL,
    etc_adj_in        NUMERIC(5,3) NOT NULL,
    rain_in           NUMERIC(5,2) NOT NULL DEFAULT 0,
    runoff_in         NUMERIC(5,2) NOT NULL DEFAULT 0,
    irrigation_in     NUMERIC(5,2) NOT NULL DEFAULT 0,
    upflux_in         NUMERIC(5,3) NOT NULL DEFAULT 0,
    deep_perc_in      NUMERIC(5,2) NOT NULL DEFAULT 0,
    depletion_in      NUMERIC(5,2) NOT NULL,
    depletion_sd_in   NUMERIC(5,3) NOT NULL DEFAULT 0.5,
    taw_in            NUMERIC(5,2) NOT NULL,
    raw_in            NUMERIC(5,2) NOT NULL,
    root_depth_in     NUMERIC(5,1) NOT NULL,
    n_observations    SMALLINT NOT NULL DEFAULT 0,
    weather_source    TEXT NOT NULL DEFAULT 'observed',
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, field_id, day, id)
) PARTITION BY LIST (season);

CREATE TABLE daily_balance_2026 PARTITION OF daily_balance FOR VALUES IN (2026);
CREATE TABLE daily_balance_2027 PARTITION OF daily_balance FOR VALUES IN (2027);
CREATE TABLE daily_balance_2028 PARTITION OF daily_balance FOR VALUES IN (2028);

CREATE INDEX ix_balance_field_day ON daily_balance (field_id, day DESC);

-- ---------------------------------------------------------------------------
-- What the grower did
-- ---------------------------------------------------------------------------

CREATE TABLE irrigation_events (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    field_id          UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    day               DATE NOT NULL,
    gross_in          NUMERIC(5,2) NOT NULL CHECK (gross_in >= 0),
    net_in            NUMERIC(5,2) NOT NULL,
    hours_run         NUMERIC(6,1),
    fertigated_n_lb   NUMERIC(6,1) NOT NULL DEFAULT 0,
    source            TEXT NOT NULL DEFAULT 'grower',
    notes             TEXT,
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_irrig_field_day ON irrigation_events (field_id, day DESC);

CREATE TABLE rain_reports (
    id          BIGSERIAL PRIMARY KEY,
    field_id    UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    day         DATE NOT NULL,
    amount_in   NUMERIC(5,2) NOT NULL CHECK (amount_in >= 0),
    source      TEXT NOT NULL DEFAULT 'grower_gauge',
    UNIQUE (field_id, day, source)
);

CREATE TABLE nutrient_applications (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    field_id          UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    day               DATE NOT NULL,
    product           TEXT NOT NULL,
    n_lb_per_ac       NUMERIC(7,2) NOT NULL DEFAULT 0,
    p2o5_lb_per_ac    NUMERIC(7,2) NOT NULL DEFAULT 0,
    k2o_lb_per_ac     NUMERIC(7,2) NOT NULL DEFAULT 0,
    method            TEXT NOT NULL DEFAULT 'sidedress',
    manure_source     TEXT,
    manure_rate_ton   NUMERIC(6,2),
    incorporation     TEXT,
    analysis_on_file  BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX ix_nutrient_field_day ON nutrient_applications (field_id, day DESC);

-- ---------------------------------------------------------------------------
-- Sensing
-- ---------------------------------------------------------------------------

CREATE TABLE sensor_stations (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    field_id          UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    label             TEXT NOT NULL,
    sensor_class      TEXT NOT NULL DEFAULT 'capacitance_pro',
    location          GEOGRAPHY(POINT, 4326),
    depths_in         SMALLINT[] NOT NULL DEFAULT '{}',
    telemetry         TEXT NOT NULL DEFAULT 'cellular',
    installed_on      DATE,
    field_calibrated  BOOLEAN NOT NULL DEFAULT FALSE,
    calibration_notes TEXT,
    active            BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE sensor_readings (
    id            BIGSERIAL,
    station_id    UUID NOT NULL REFERENCES sensor_stations(id) ON DELETE CASCADE,
    observed_at   TIMESTAMPTZ NOT NULL,
    depth_in      SMALLINT NOT NULL,
    vwc           NUMERIC(5,4),
    soil_temp_f   NUMERIC(5,2),
    bulk_ec       NUMERIC(6,3),
    quality_flag  TEXT NOT NULL DEFAULT 'ok',
    PRIMARY KEY (observed_at, station_id, depth_in, id)
) PARTITION BY RANGE (observed_at);

CREATE TABLE sensor_readings_2026 PARTITION OF sensor_readings
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
CREATE TABLE sensor_readings_2027 PARTITION OF sensor_readings
    FOR VALUES FROM ('2027-01-01') TO ('2028-01-01');

CREATE INDEX ix_reading_station_time ON sensor_readings (station_id, observed_at DESC);

CREATE TABLE canopy_observations (
    id              BIGSERIAL PRIMARY KEY,
    field_id        UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    day             DATE NOT NULL,
    product         TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('canopy_cover','ndvi','canopy_temp')),
    value           NUMERIC(7,4) NOT NULL,
    cloud_fraction  NUMERIC(4,3),
    raster_uri      TEXT
);
CREATE INDEX ix_canopy_field_day ON canopy_observations (field_id, day DESC);

-- ---------------------------------------------------------------------------
-- What we told them, and whether it worked
-- ---------------------------------------------------------------------------
-- This table is the evaluation backbone. Extension reporting on a project like
-- this usually counts workshop attendance and downloads. Being able to report
-- what fraction of recommendations were acted on, and the water and nitrogen
-- consequences, is a far stronger outcome measure and is the thing funders and
-- the Nutrient Management Commission will actually want.

CREATE TABLE advisory_records (
    id                   BIGSERIAL,
    field_id             UUID NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
    day                  DATE NOT NULL,
    season               SMALLINT NOT NULL,
    decision             TEXT NOT NULL,
    headline             TEXT NOT NULL,
    recommended_net_in   NUMERIC(5,2) NOT NULL DEFAULT 0,
    depletion_in         NUMERIC(5,2) NOT NULL,
    raw_in               NUMERIC(5,2) NOT NULL,
    confidence           TEXT NOT NULL DEFAULT 'modeled',
    payload              JSONB NOT NULL DEFAULT '{}'::jsonb,
    delivered_channel    TEXT,
    delivered_at         TIMESTAMPTZ,
    viewed_at            TIMESTAMPTZ,
    followed             BOOLEAN,
    PRIMARY KEY (season, field_id, day, id)
) PARTITION BY LIST (season);

CREATE TABLE advisory_records_2026 PARTITION OF advisory_records FOR VALUES IN (2026);
CREATE TABLE advisory_records_2027 PARTITION OF advisory_records FOR VALUES IN (2027);

-- ---------------------------------------------------------------------------
-- Weather cache
-- ---------------------------------------------------------------------------
-- One row per station-day, shared by every field that uses that station. With
-- roughly 40 DEOS sites this table stays tiny, and it means the nightly batch
-- makes about 40 upstream calls instead of 100,000.

CREATE TABLE station_daily (
    station_code   TEXT NOT NULL,
    day            DATE NOT NULL,
    tmax_f         NUMERIC(5,2),
    tmin_f         NUMERIC(5,2),
    rh_max         NUMERIC(5,2),
    rh_min         NUMERIC(5,2),
    wind_mph       NUMERIC(5,2),
    solar_mj_m2    NUMERIC(6,3),
    rain_in        NUMERIC(5,2),
    eto_in         NUMERIC(5,3),
    quality        TEXT NOT NULL DEFAULT 'ok',
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (station_code, day)
);

-- ---------------------------------------------------------------------------
-- Reporting views
-- ---------------------------------------------------------------------------

CREATE VIEW v_season_summary AS
SELECT
    b.field_id,
    b.season,
    f.account_id,
    f.crop_key,
    f.soil_key,
    f.acres,
    SUM(b.irrigation_in)                             AS irrigation_in,
    SUM(b.rain_in)                                   AS rain_in,
    SUM(b.etc_adj_in)                                AS crop_water_use_in,
    SUM(b.deep_perc_in)                              AS deep_percolation_in,
    COUNT(*) FILTER (WHERE b.ks < 0.95)              AS stress_days,
    SUM(b.irrigation_in) * f.acres                   AS acre_inches_applied
FROM daily_balance b
JOIN fields f ON f.id = b.field_id
GROUP BY b.field_id, b.season, f.account_id, f.crop_key, f.soil_key, f.acres;

CREATE VIEW v_adoption AS
SELECT
    season,
    COUNT(DISTINCT field_id)                                    AS fields_advised,
    COUNT(*)                                                    AS advisories_issued,
    COUNT(*) FILTER (WHERE viewed_at IS NOT NULL)               AS advisories_viewed,
    COUNT(*) FILTER (WHERE followed)                            AS advisories_followed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE followed)
          / NULLIF(COUNT(*) FILTER (WHERE decision = 'irrigate_now'), 0), 1)
                                                                AS pct_followed_when_told_to_irrigate
FROM advisory_records
GROUP BY season;
