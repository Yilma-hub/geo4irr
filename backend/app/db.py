"""
Persistence layer.

Sized for the target: 20,000 Delaware growers, each with a handful of fields,
each field carrying a daily water balance record for a 200-day season. That is
on the order of 30 million balance rows over five years, which a single
well-indexed Postgres instance handles without partitioning. The daily balance
table is the one that grows; everything else is small.

The schema keeps the water balance as an append-only daily ledger rather than a
mutable "current state" row. That matters for trust: when a grower asks why the
tool told them to irrigate on July 12th, the answer has to be reconstructable
from what was known on July 12th, not from the state as it has since been
revised by later sensor readings.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean, Date, DateTime, Enum, Float, ForeignKey, Index, Integer,
    JSON, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class County(str, enum.Enum):
    new_castle = "New Castle"
    kent = "Kent"
    sussex = "Sussex"
    other = "Other"


class Grower(Base):
    __tablename__ = "growers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(160))
    farm_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    email: Mapped[str | None] = mapped_column(String(200), unique=True, nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    county: Mapped[County] = mapped_column(Enum(County), default=County.sussex)
    # Delaware Nutrient Management Law: certification required at 10+ acres.
    nm_certification_number: Mapped[str | None] = mapped_column(String(40), nullable=True)
    nm_certification_expires: Mapped[date | None] = mapped_column(Date, nullable=True)
    sms_alerts: Mapped[bool] = mapped_column(Boolean, default=True)
    alert_hour_local: Mapped[int] = mapped_column(Integer, default=5)
    units: Mapped[str] = mapped_column(String(8), default="us")
    preferred_language: Mapped[str] = mapped_column(String(8), default="en")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    fields: Mapped[list["Field"]] = relationship(back_populates="grower",
                                                 cascade="all, delete-orphan")


class Field(Base):
    __tablename__ = "fields"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    grower_id: Mapped[str] = mapped_column(ForeignKey("growers.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    acres: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    boundary_geojson: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    soil_key: Mapped[str] = mapped_column(String(40), default="sassafras")
    soil_components: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ssurgo_mukey: Mapped[str | None] = mapped_column(String(20), nullable=True)
    awc_override: Mapped[float | None] = mapped_column(Float, nullable=True)
    restrictive_layer_in: Mapped[float | None] = mapped_column(Float, nullable=True)
    water_table_depth_in: Mapped[float | None] = mapped_column(Float, nullable=True)

    crop_key: Mapped[str] = mapped_column(String(40), default="corn_grain")
    planting_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    yield_goal: Mapped[float | None] = mapped_column(Float, nullable=True)
    previous_crop: Mapped[str | None] = mapped_column(String(40), nullable=True)
    cover_crop: Mapped[str] = mapped_column(String(40), default="none")

    system_type: Mapped[str] = mapped_column(String(30), default="center_pivot")
    capacity_gpm: Mapped[float] = mapped_column(Float, default=800.0)
    application_efficiency: Mapped[float] = mapped_column(Float, default=0.85)
    min_application_in: Mapped[float] = mapped_column(Float, default=0.25)
    max_application_in: Mapped[float] = mapped_column(Float, default=1.20)
    vri_capable: Mapped[bool] = mapped_column(Boolean, default=False)
    fertigation_capable: Mapped[bool] = mapped_column(Boolean, default=False)
    well_permit_gpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    well_nitrate_ppm: Mapped[float | None] = mapped_column(Float, nullable=True)

    deos_station: Mapped[str | None] = mapped_column(String(12), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    grower: Mapped[Grower] = relationship(back_populates="fields")
    balances: Mapped[list["DailyBalance"]] = relationship(
        back_populates="field", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_fields_grower_active", "grower_id", "active"),
        Index("ix_fields_location", "lat", "lon"),
    )


class ManagementZone(Base):
    """A VRI prescription zone or a distinct soil block within a field."""

    __tablename__ = "management_zones"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(80))
    acres: Mapped[float] = mapped_column(Float)
    soil_key: Mapped[str] = mapped_column(String(40))
    geometry_geojson: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    yield_index: Mapped[float] = mapped_column(Float, default=1.0)


class DailyBalance(Base):
    """Append-only daily water balance ledger. One row per field per day."""

    __tablename__ = "daily_balance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"))
    zone_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    day: Mapped[date] = mapped_column(Date)
    dap: Mapped[int] = mapped_column(Integer)

    eto_in: Mapped[float] = mapped_column(Float)
    kcb: Mapped[float] = mapped_column(Float)
    ke: Mapped[float] = mapped_column(Float)
    ks: Mapped[float] = mapped_column(Float)
    etc_adj_in: Mapped[float] = mapped_column(Float)
    rain_in: Mapped[float] = mapped_column(Float, default=0.0)
    runoff_in: Mapped[float] = mapped_column(Float, default=0.0)
    irrigation_in: Mapped[float] = mapped_column(Float, default=0.0)
    upflux_in: Mapped[float] = mapped_column(Float, default=0.0)
    deep_perc_in: Mapped[float] = mapped_column(Float, default=0.0)
    depletion_in: Mapped[float] = mapped_column(Float)
    depletion_sd_in: Mapped[float] = mapped_column(Float, default=0.5)
    taw_in: Mapped[float] = mapped_column(Float)
    raw_in: Mapped[float] = mapped_column(Float)
    root_depth_in: Mapped[float] = mapped_column(Float)
    n_observations: Mapped[int] = mapped_column(Integer, default=0)
    weather_source: Mapped[str] = mapped_column(String(20), default="observed")

    __table_args__ = (
        UniqueConstraint("field_id", "zone_id", "day", name="uq_balance_field_day"),
        Index("ix_balance_field_day", "field_id", "day"),
    )

    field: Mapped[Field] = relationship(back_populates="balances")


class IrrigationEvent(Base):
    __tablename__ = "irrigation_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"))
    day: Mapped[date] = mapped_column(Date)
    gross_in: Mapped[float] = mapped_column(Float)
    net_in: Mapped[float] = mapped_column(Float)
    hours_run: Mapped[float | None] = mapped_column(Float, nullable=True)
    fertigated_n_lb: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(20), default="grower")   # grower | telemetry
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (Index("ix_irrig_field_day", "field_id", "day"),)


class NutrientApplication(Base):
    __tablename__ = "nutrient_applications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"))
    day: Mapped[date] = mapped_column(Date)
    product: Mapped[str] = mapped_column(String(80))
    n_lb_per_ac: Mapped[float] = mapped_column(Float, default=0.0)
    p2o5_lb_per_ac: Mapped[float] = mapped_column(Float, default=0.0)
    k2o_lb_per_ac: Mapped[float] = mapped_column(Float, default=0.0)
    method: Mapped[str] = mapped_column(String(40), default="sidedress")
    manure_source: Mapped[str | None] = mapped_column(String(40), nullable=True)
    manure_rate_ton: Mapped[float | None] = mapped_column(Float, nullable=True)
    incorporation: Mapped[str | None] = mapped_column(String(40), nullable=True)


class SensorStation(Base):
    __tablename__ = "sensor_stations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(String(80))
    sensor_class: Mapped[str] = mapped_column(String(40), default="capacitance_pro")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    depths_in: Mapped[list] = mapped_column(JSON, default=list)
    telemetry: Mapped[str] = mapped_column(String(20), default="cellular")
    installed_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    field_calibrated: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(ForeignKey("sensor_stations.id",
                                                       ondelete="CASCADE"))
    observed_at: Mapped[datetime] = mapped_column(DateTime)
    depth_in: Mapped[int] = mapped_column(Integer)
    vwc: Mapped[float | None] = mapped_column(Float, nullable=True)
    soil_temp_f: Mapped[float | None] = mapped_column(Float, nullable=True)
    bulk_ec: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_flag: Mapped[str] = mapped_column(String(16), default="ok")

    __table_args__ = (
        Index("ix_reading_station_time", "station_id", "observed_at"),
    )


class CanopyObservation(Base):
    """UAV, satellite or handheld canopy products."""

    __tablename__ = "canopy_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"))
    day: Mapped[date] = mapped_column(Date)
    product: Mapped[str] = mapped_column(String(40))     # uav_multispectral, sentinel2_10m ...
    kind: Mapped[str] = mapped_column(String(24))        # canopy_cover | ndvi | canopy_temp
    value: Mapped[float] = mapped_column(Float)
    cloud_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    raster_uri: Mapped[str | None] = mapped_column(String(300), nullable=True)

    __table_args__ = (Index("ix_canopy_field_day", "field_id", "day"),)


class AdvisoryRecord(Base):
    """What the tool told the grower, and whether they acted on it.

    The `followed` column is the evaluation backbone. Extension reporting for
    this kind of project usually counts downloads and workshop attendance; being
    able to say what fraction of recommendations were acted on, and what the
    water and nitrogen consequences were, is a much stronger outcome measure.
    """

    __tablename__ = "advisory_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    field_id: Mapped[str] = mapped_column(ForeignKey("fields.id", ondelete="CASCADE"))
    day: Mapped[date] = mapped_column(Date)
    decision: Mapped[str] = mapped_column(String(24))
    headline: Mapped[str] = mapped_column(String(200))
    recommended_net_in: Mapped[float] = mapped_column(Float, default=0.0)
    depletion_in: Mapped[float] = mapped_column(Float)
    raw_in: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(24), default="modeled")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    delivered_channel: Mapped[str | None] = mapped_column(String(20), nullable=True)
    viewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    followed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    __table_args__ = (
        UniqueConstraint("field_id", "day", name="uq_advisory_field_day"),
    )
