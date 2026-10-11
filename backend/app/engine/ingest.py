"""
Data acquisition and ingest — Activities 2 and 3.

Activity 3 lists the formats Geo4Irr has to accept, and the reason is stated
plainly: "This approach ensures the utility and usefulness of Geo4Irr as new
spatial products emerge." The platform is not supposed to be the only source of
its own data.

    In-situ soil moisture sensor data in ASCII and multidimensional NetCDF
    Remote sensing and UAV canopy cover, temperature, NDVI in raster (GeoTIFF)
    Crop yield in raster and feature class (ESRI shapefile or Google KML)

Activity 2 adds the data sources for farm setup: cropping system from the NASS
Cropland Data Layer, soils from SSURGO, climate from meteorological networks
plus gridded Daymet and PRISM with user spreadsheet upload as a fallback, and
water diversions from state databases.

Two decisions run through this module.

First, no heavy dependencies. GeoTIFF and NetCDF are both self-describing binary
formats with well-documented headers, and reading the parts needed here —
georeferencing plus a single band or variable — is a few hundred lines. Adding
GDAL and netCDF4 to a container that has to start fast and run twenty thousand
nightly field evaluations is a poor trade for that. When those libraries are
present the readers use them; when they are not, the built-in parsers handle the
common cases, and both paths report which one ran.

Second, the Delaware sources replace the Colorado ones one for one. CoAgMET and
ZiaMet become DEOS; the Colorado Decision Support System becomes the DNREC water
allocation permit record. The structure is the same because the narrative's
structure is right — it is the endpoints that are regional.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import struct
import zipfile
import struct
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# In-situ sensor data: ASCII
# ---------------------------------------------------------------------------

_TIME_KEYS = ("timestamp", "datetime", "date_time", "time", "date", "obs_time",
              "readingdatetime", "measurement_time")
_DEPTH_RE = re.compile(r"(?:vwc|swc|sm|moist\w*|theta)[_\s-]*(\d+)\s*(in|inch|cm|mm)?",
                       re.I)


@dataclass(slots=True)
class SensorReading:
    observed_at: datetime
    depth_in: float
    vwc: float
    soil_temp_f: float | None = None
    bulk_ec: float | None = None
    station: str = ""
    quality: str = "ok"


def read_sensor_ascii(text: str, station: str = "",
                      depth_units: str = "auto") -> list[SensorReading]:
    """Parse a delimited sensor export.

    Every logger vendor writes a different file. Sentek, AquaSpy and CropX are
    all in use at the CSU TAPS site named in Activity 11, and their exports
    differ in delimiter, header depth, timestamp format and how depth is encoded
    in the column name. Rather than a parser per vendor, this sniffs the
    structure: find the timestamp column, find columns whose names carry a depth,
    and read those.

    Depth in a column name is ambiguous between inches and centimetres. `auto`
    guesses from magnitude — a 100 in that column is centimetres, a 12 is
    inches — and any file with depths above 60 is treated as centimetres
    throughout, which covers every logger seen so far.
    """
    if not text.strip():
        return []
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample else ","

    lines = [l for l in text.splitlines() if l.strip()]
    header_idx = 0
    for i, line in enumerate(lines[:20]):
        low = line.lower()
        if any(k in low for k in _TIME_KEYS) and _DEPTH_RE.search(low):
            header_idx = i
            break

    reader = csv.DictReader(lines[header_idx:], delimiter=delimiter)
    fields = [f for f in (reader.fieldnames or []) if f]
    time_col = next((f for f in fields if f.strip().lower() in _TIME_KEYS), None)
    if time_col is None:
        time_col = next((f for f in fields
                         if any(k in f.lower() for k in _TIME_KEYS)), None)
    if time_col is None:
        raise ValueError(
            "No timestamp column found. The export needs a column named "
            "something like 'timestamp', 'datetime' or 'date'.")

    depth_cols: dict[str, tuple[float, str]] = {}
    for f in fields:
        m = _DEPTH_RE.search(f)
        if m:
            depth_cols[f] = (float(m.group(1)), (m.group(2) or "").lower())
    if not depth_cols:
        raise ValueError(
            "No moisture columns found. Column names need to carry the depth, "
            "for example 'VWC_12' or 'swc_30cm'.")

    raw_depths = [d for d, _ in depth_cols.values()]
    centimetres = (depth_units == "cm" or
                   (depth_units == "auto" and max(raw_depths) > 60))

    temp_cols = [f for f in fields if re.search(r"temp", f, re.I)]
    ec_cols = [f for f in fields if re.search(r"\bec\b|salin|conduct", f, re.I)]

    out: list[SensorReading] = []
    for row in reader:
        ts = _parse_timestamp(row.get(time_col, ""))
        if ts is None:
            continue
        for col, (depth, unit) in depth_cols.items():
            raw = (row.get(col) or "").strip()
            if raw in ("", "NaN", "nan", "-9999", "-99.99", "null", "NA"):
                continue
            try:
                vwc = float(raw)
            except ValueError:
                continue
            # Some loggers report percent rather than a fraction.
            if vwc > 1.5:
                vwc /= 100.0
            if not 0.0 <= vwc <= 0.75:
                continue
            depth_in = depth / 2.54 if (unit.startswith("cm") or
                                        (not unit and centimetres)) else depth
            out.append(SensorReading(
                observed_at=ts, depth_in=round(depth_in, 1), vwc=round(vwc, 4),
                soil_temp_f=_first_float(row, temp_cols),
                bulk_ec=_first_float(row, ec_cols), station=station))
    out.sort(key=lambda r: (r.observed_at, r.depth_in))
    return out


def _first_float(row: dict, cols: Sequence[str]) -> float | None:
    for c in cols:
        try:
            return float((row.get(c) or "").strip())
        except (ValueError, AttributeError):
            continue
    return None


_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
               "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
               "%m/%d/%y %H:%M", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y %H:%M")


def _parse_timestamp(raw: str) -> datetime | None:
    raw = (raw or "").strip().strip('"')
    if not raw:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:                                     # epoch seconds
        v = float(raw)
        if 1e8 < v < 4e9:
            return datetime.utcfromtimestamp(v)
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# In-situ sensor data: NetCDF
# ---------------------------------------------------------------------------

def read_sensor_netcdf(data: bytes, variable: str | None = None) -> dict:
    """Read a NetCDF classic file without netCDF4.

    Activity 3 asks for "multidimensional NetCDF" specifically, which is what
    research soil moisture networks and gridded products publish. If the
    netCDF4 or scipy readers are installed they are used; otherwise the classic
    format (CDF-1 and CDF-2) header is parsed here. That covers what the sensor
    networks actually write. HDF5-backed NetCDF-4 files need the library, and
    this says so rather than returning something wrong.
    """
    if data[:3] == b"CDF":
        version = data[3]
        if version not in (1, 2):
            return {"ok": False, "reason": f"unsupported classic version {version}"}
        return _parse_netcdf_classic(data, variable)
    if data[:8] == b"\x89HDF\r\n\x1a\n":
        try:
            import netCDF4                                  # noqa: F401
        except ImportError:
            return {
                "ok": False,
                "reason": "netcdf4_file_requires_library",
                "hint": ("This is an HDF5-backed NetCDF-4 file. pip install "
                         "netCDF4, or ask the provider for a classic NetCDF-3 "
                         "or an ASCII export."),
            }
        import netCDF4, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as fh:
            fh.write(data)
            path = fh.name
        try:
            ds = netCDF4.Dataset(path)
            names = list(ds.variables)
            var = variable or _pick_moisture_variable(names)
            values = ds.variables[var][:].tolist() if var else []
            return {"ok": True, "format": "netcdf4", "variables": names,
                    "variable": var, "values": values,
                    "dimensions": {d: len(ds.dimensions[d]) for d in ds.dimensions}}
        finally:
            os.unlink(path)
    return {"ok": False, "reason": "not_a_netcdf_file"}


_NC_TYPES = {1: ("b", 1), 2: ("c", 1), 3: (">h", 2), 4: (">i", 4),
             5: (">f", 4), 6: (">d", 8)}


def _parse_netcdf_classic(data: bytes, variable: str | None) -> dict:
    pos = 4
    version = data[3]
    offset_fmt = ">q" if version == 2 else ">i"
    offset_size = 8 if version == 2 else 4

    def read_int() -> int:
        nonlocal pos
        v = struct.unpack_from(">i", data, pos)[0]
        pos += 4
        return v

    def read_name() -> str:
        nonlocal pos
        n = read_int()
        raw = data[pos:pos + n].decode("utf-8", "replace")
        pos += n + ((4 - n % 4) % 4)
        return raw

    def skip_attrs() -> dict:
        nonlocal pos
        tag = read_int()
        n = read_int()
        attrs: dict[str, Any] = {}
        if tag == 0 and n == 0:
            return attrs
        for _ in range(n):
            name = read_name()
            nc_type = read_int()
            count = read_int()
            fmt, size = _NC_TYPES.get(nc_type, ("b", 1))
            raw = data[pos:pos + count * size]
            pos += count * size + ((4 - (count * size) % 4) % 4)
            if nc_type == 2:
                attrs[name] = raw.decode("utf-8", "replace").rstrip("\x00")
            else:
                try:
                    attrs[name] = list(struct.unpack(f">{count}{fmt[-1]}", raw))
                except struct.error:
                    attrs[name] = None
        return attrs

    read_int()                               # numrecs
    dims: list[tuple[str, int]] = []
    tag = read_int()
    n = read_int()
    if tag == 10:                            # NC_DIMENSION
        for _ in range(n):
            dims.append((read_name(), read_int()))
    global_attrs = skip_attrs()

    tag = read_int()
    n = read_int()
    variables: dict[str, dict] = {}
    if tag == 11:                            # NC_VARIABLE
        for _ in range(n):
            name = read_name()
            ndims = read_int()
            dimids = [read_int() for _ in range(ndims)]
            attrs = skip_attrs()
            nc_type = read_int()
            vsize = read_int()
            begin = struct.unpack_from(offset_fmt, data, pos)[0]
            pos += offset_size
            variables[name] = {"type": nc_type, "size": vsize, "begin": begin,
                               "dims": [dims[i][0] for i in dimids if i < len(dims)],
                               "shape": [dims[i][1] for i in dimids if i < len(dims)],
                               "attrs": attrs}

    var = variable or _pick_moisture_variable(list(variables))
    values: list[float] = []
    if var and var in variables:
        meta = variables[var]
        fmt, size = _NC_TYPES.get(meta["type"], (">f", 4))
        count = 1
        for s in meta["shape"] or [0]:
            count *= max(1, s)
        raw = data[meta["begin"]:meta["begin"] + count * size]
        try:
            values = list(struct.unpack(f">{count}{fmt[-1]}", raw))
        except struct.error:
            values = []
        scale = meta["attrs"].get("scale_factor")
        offset = meta["attrs"].get("add_offset")
        if scale:
            values = [v * scale[0] for v in values]
        if offset:
            values = [v + offset[0] for v in values]
        fill = meta["attrs"].get("_FillValue")
        if fill:
            values = [v for v in values if abs(v - fill[0]) > 1e-6]

    return {
        "ok": True, "format": f"netcdf_classic_v{version}",
        "dimensions": dict(dims),
        "variables": list(variables),
        "variable": var,
        "shape": variables.get(var, {}).get("shape", []),
        "units": (variables.get(var, {}).get("attrs", {}).get("units")),
        "values": values[:100_000],
        "n_values": len(values),
        "global_attributes": {k: v for k, v in global_attrs.items()
                              if isinstance(v, str)},
    }


def _pick_moisture_variable(names: Sequence[str]) -> str | None:
    for pattern in (r"^(vwc|swc|soil_moisture|sm)$", r"moist", r"vwc|swc",
                    r"theta", r"water_content"):
        for n in names:
            if re.search(pattern, n, re.I):
                return n
    return names[0] if names else None


# ---------------------------------------------------------------------------
# Raster: GeoTIFF
# ---------------------------------------------------------------------------

@dataclass
class Raster:
    width: int
    height: int
    values: list[float]
    origin_x: float
    origin_y: float
    pixel_x: float
    pixel_y: float
    epsg: int | None = None
    nodata: float | None = None
    source: str = "builtin"

    def sample_points(self, stride: int = 1) -> list[tuple[float, float, float]]:
        """Pixel centres as (x, y, value), skipping nodata."""
        out = []
        for row in range(0, self.height, stride):
            for col in range(0, self.width, stride):
                v = self.values[row * self.width + col]
                if self.nodata is not None and abs(v - self.nodata) < 1e-6:
                    continue
                if not math.isfinite(v):
                    continue
                x = self.origin_x + (col + 0.5) * self.pixel_x
                y = self.origin_y + (row + 0.5) * self.pixel_y
                out.append((x, y, v))
        return out

    def stats(self) -> dict:
        vals = [v for v in self.values
                if math.isfinite(v) and (self.nodata is None
                                         or abs(v - self.nodata) > 1e-6)]
        if not vals:
            return {"n": 0}
        n = len(vals)
        mean = sum(vals) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, n - 1))
        ordered = sorted(vals)
        return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
                "min": round(ordered[0], 4), "max": round(ordered[-1], 4),
                "p05": round(ordered[int(0.05 * n)], 4),
                "p50": round(ordered[n // 2], 4),
                "p95": round(ordered[min(n - 1, int(0.95 * n))], 4),
                "cv": round(sd / mean, 4) if mean else None}


_TIFF_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4,
                   10: 8, 11: 4, 12: 8}


def read_geotiff(data: bytes) -> Raster:
    """Read a single-band GeoTIFF: georeferencing plus pixel values.

    Activity 3 requires raster ingest for UAV and satellite canopy products, and
    a UAV pass over a Delaware pivot is exactly this — one band of NDVI or
    canopy temperature, uncompressed or LZW, in a projected or geographic CRS.

    This handles the strip-organised, uncompressed and packbits cases that
    drone processing software writes by default. Tiled or LZW files fall back to
    the imaging library if it is installed, and say clearly what is needed if it
    is not. Nothing here silently returns wrong pixels.
    """
    if len(data) < 8:
        raise ValueError("Not a TIFF file.")
    endian = data[:2]
    if endian == b"II":
        bo = "<"
    elif endian == b"MM":
        bo = ">"
    else:
        raise ValueError("Not a TIFF file — bad byte order marker.")
    magic = struct.unpack_from(bo + "H", data, 2)[0]
    if magic != 42:
        raise ValueError("BigTIFF is not supported; re-export as a standard TIFF.")

    ifd_offset = struct.unpack_from(bo + "I", data, 4)[0]
    count = struct.unpack_from(bo + "H", data, ifd_offset)[0]
    tags: dict[int, Any] = {}
    for i in range(count):
        off = ifd_offset + 2 + i * 12
        tag, ttype, n = struct.unpack_from(bo + "HHI", data, off)
        size = _TIFF_TYPE_SIZE.get(ttype, 1) * n
        if size <= 4:
            value_off = off + 8
        else:
            value_off = struct.unpack_from(bo + "I", data, off + 8)[0]
        tags[tag] = _read_tag(data, bo, ttype, n, value_off)

    width = int(_scalar(tags.get(256, 0)))
    height = int(_scalar(tags.get(257, 0)))
    bits = int(_scalar(tags.get(258, 8)))
    compression = int(_scalar(tags.get(259, 1)))
    sample_format = int(_scalar(tags.get(339, 1)))
    samples = int(_scalar(tags.get(277, 1)))
    strip_offsets = _as_list(tags.get(273, []))
    strip_counts = _as_list(tags.get(279, []))
    rows_per_strip = int(_scalar(tags.get(278, height))) or height

    pixel_scale = _as_list(tags.get(33550, [1.0, 1.0, 0.0]))
    tie_point = _as_list(tags.get(33922, [0, 0, 0, 0, 0, 0]))
    nodata = None
    if 42113 in tags:
        try:
            nodata = float(str(tags[42113]).strip("\x00"))
        except (TypeError, ValueError):
            nodata = None

    epsg = None
    if 34735 in tags:
        keys = _as_list(tags[34735])
        for i in range(4, len(keys) - 3, 4):
            if keys[i] in (3072, 2048):
                epsg = int(keys[i + 3])
                break

    if compression not in (1, 32773) or 322 in tags:
        raster = _read_raster_with_library(data)
        if raster is not None:
            return raster
        raise ValueError(
            "This GeoTIFF is tiled or uses a compression this reader does not "
            "handle. pip install Pillow, or re-export uncompressed and "
            "strip-organised, which most drone software offers as an option.")

    fmt_char = {(32, 3): "f", (64, 3): "d", (8, 1): "B", (16, 1): "H",
                (32, 1): "I", (8, 2): "b", (16, 2): "h", (32, 2): "i"}.get(
        (bits, sample_format), "B")

    values: list[float] = []
    for idx, (soff, scount) in enumerate(zip(strip_offsets, strip_counts)):
        raw = data[int(soff):int(soff) + int(scount)]
        if compression == 32773:
            raw = _unpackbits(raw)
        per = struct.calcsize(bo + fmt_char)
        n_vals = len(raw) // per
        try:
            strip = struct.unpack_from(bo + f"{n_vals}{fmt_char}", raw, 0)
        except struct.error:
            continue
        values.extend(float(v) for v in strip)

    if samples > 1:
        values = values[::samples]
    expected = width * height
    values = (values + [float("nan")] * expected)[:expected]

    return Raster(width=width, height=height, values=values,
                  origin_x=float(tie_point[3]) if len(tie_point) > 3 else 0.0,
                  origin_y=float(tie_point[4]) if len(tie_point) > 4 else 0.0,
                  pixel_x=float(pixel_scale[0]) if pixel_scale else 1.0,
                  pixel_y=-abs(float(pixel_scale[1])) if len(pixel_scale) > 1 else -1.0,
                  epsg=epsg, nodata=nodata, source="builtin")


def _read_tag(data: bytes, bo: str, ttype: int, n: int, off: int):
    fmt = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 6: "b", 8: "h", 9: "i",
           11: "f", 12: "d"}.get(ttype, "B")
    if ttype == 2:
        return data[off:off + n].decode("ascii", "replace").rstrip("\x00")
    if ttype == 5:
        out = []
        for i in range(n):
            num, den = struct.unpack_from(bo + "II", data, off + i * 8)
            out.append(num / den if den else 0.0)
        return out
    try:
        vals = struct.unpack_from(bo + f"{n}{fmt}", data, off)
    except struct.error:
        return []
    return list(vals) if n > 1 else vals[0]


def _scalar(v):
    return v[0] if isinstance(v, (list, tuple)) and v else v


def _as_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _unpackbits(raw: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(raw):
        n = raw[i]
        i += 1
        if n < 128:
            out.extend(raw[i:i + n + 1])
            i += n + 1
        elif n > 128:
            if i < len(raw):
                out.extend(bytes([raw[i]]) * (257 - n))
                i += 1
    return bytes(out)


def _read_raster_with_library(data: bytes) -> Raster | None:
    try:
        from PIL import Image
        import tempfile, os
    except ImportError:
        return None
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as fh:
        fh.write(data)
        path = fh.name
    try:
        img = Image.open(path)
        values = [float(v) for v in img.getdata()]
        return Raster(width=img.width, height=img.height, values=values,
                      origin_x=0.0, origin_y=0.0, pixel_x=1.0, pixel_y=-1.0,
                      source="pillow")
    except Exception:
        return None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Yield: raster and feature class
# ---------------------------------------------------------------------------

def yield_from_raster(raster: Raster, stride: int = 1) -> dict:
    """Summarise a yield raster and normalise it to a yield index per point.

    A yield map is the one dataset that reports what actually happened, and it
    is the ground truth for every zone the rest of the system proposes. The
    index is relative to the field mean so that fields, crops and years compare.
    """
    pts = raster.sample_points(stride)
    if not pts:
        return {"n": 0}
    values = [v for _, _, v in pts]
    mean = sum(values) / len(values)
    stats = raster.stats()
    return {
        "n": len(pts),
        "stats": stats,
        "points": [(x, y, round(v / mean, 3) if mean else 1.0)
                   for x, y, v in pts],
        "uniformity_cv": stats.get("cv"),
        "note": _yield_note(stats.get("cv")),
    }


def yield_from_features(features: Sequence[dict], value_key: str = "yield"
                        ) -> dict:
    """Yield from point features — a shapefile or KML export off the monitor.

    Combine harvesters write a point every second or so with a wet mass flow.
    This takes the already-parsed features from the geo module's readers and
    treats them the same way as a raster, so a grower with either format gets
    the same zone analysis.
    """
    pts = []
    for f in features:
        props = f.get("properties", f)
        geom = f.get("geometry", {})
        coords = geom.get("coordinates") if geom else None
        if not coords or len(coords) < 2:
            continue
        value = None
        for key in (value_key, "Yld_Vol_Dr", "YIELD", "Yield", "DryYield",
                    "yld_mass_d", "VRYIELDVOL"):
            if key in props:
                try:
                    value = float(props[key])
                    break
                except (TypeError, ValueError):
                    continue
        if value is None or value <= 0:
            continue
        pts.append((float(coords[0]), float(coords[1]), value))
    if not pts:
        return {"n": 0, "note": "No usable yield values in those features."}

    values = [v for _, _, v in pts]
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / max(1, n - 1))
    return {
        "n": n,
        "mean": round(mean, 2),
        "sd": round(sd, 2),
        "cv": round(sd / mean, 4) if mean else None,
        "points": [(x, y, round(v / mean, 3)) for x, y, v in pts],
        "note": _yield_note(sd / mean if mean else None),
    }


def _yield_note(cv: float | None) -> str:
    if cv is None:
        return ""
    if cv < 0.10:
        return ("The field yielded evenly. Variable rate has little to work with "
                "here, whatever the soil map suggests.")
    if cv < 0.20:
        return ("Moderate variation. Worth checking whether the pattern lines up "
                "with the soil map — if it does, it will repeat next year and is "
                "worth managing; if it does not, it was this season's weather.")
    return ("Strong variation. If the same pattern appears in the imagery and the "
            "soil components, this field is a real variable-rate candidate.")


# ---------------------------------------------------------------------------
# Weather: user upload and gridded fallbacks
# ---------------------------------------------------------------------------

def read_weather_csv(text: str) -> dict:
    """Accept a grower's own weather record.

    Activity 2: "Users can upload their own time series data in spreadsheet
    formats." Growers with an on-farm station usually trust it more than the
    nearest DEOS site, and they are often right — a station three miles away did
    not see the thunderstorm that hit their field.

    Column names are matched loosely because every station brand exports
    different headers.
    """
    if not text.strip():
        return {"ok": False, "reason": "empty"}
    lines = [l for l in text.splitlines() if l.strip()]
    delimiter = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(lines, delimiter=delimiter)
    fields = [f for f in (reader.fieldnames or []) if f]

    def find(*patterns) -> str | None:
        for p in patterns:
            for f in fields:
                if re.search(p, f, re.I):
                    return f
        return None

    cols = {
        "date": find(r"^date$", r"date", r"time"),
        "tmax": find(r"tmax", r"max.*temp", r"air_temp_max"),
        "tmin": find(r"tmin", r"min.*temp", r"air_temp_min"),
        "rh_max": find(r"rh.*max", r"max.*humid", r"humidity_max"),
        "rh_min": find(r"rh.*min", r"min.*humid", r"humidity_min"),
        "wind": find(r"wind.*sp", r"^wind", r"ws_"),
        "solar": find(r"solar", r"srad", r"radiation"),
        "rain": find(r"rain", r"precip", r"pcpn"),
        "eto": find(r"^eto$", r"et_?ref", r"reference_et"),
    }
    if not cols["date"] or not cols["tmax"] or not cols["tmin"]:
        return {"ok": False, "reason": "missing_required_columns",
                "found": [f for f in fields],
                "hint": "At minimum the file needs a date, a maximum temperature "
                        "and a minimum temperature column."}

    rows, skipped = [], 0
    for row in reader:
        ts = _parse_timestamp(row.get(cols["date"], ""))
        if ts is None:
            skipped += 1
            continue

        def val(key: str) -> float | None:
            col = cols.get(key)
            if not col:
                return None
            try:
                return float((row.get(col) or "").strip())
            except (ValueError, AttributeError):
                return None

        tmax, tmin = val("tmax"), val("tmin")
        if tmax is None or tmin is None:
            skipped += 1
            continue
        # Celsius stations are common; convert when the range gives it away.
        if tmax < 45 and tmin < 35:
            tmax, tmin = tmax * 9 / 5 + 32, tmin * 9 / 5 + 32
        rows.append({
            "day": ts.date().isoformat(), "tmax_f": round(tmax, 2),
            "tmin_f": round(tmin, 2), "rh_max": val("rh_max"),
            "rh_min": val("rh_min"), "wind_mph": val("wind"),
            "solar_rad_mj_m2": val("solar"), "rain_in": val("rain") or 0.0,
            "eto_in": val("eto"),
        })

    has_solar = any(r["solar_rad_mj_m2"] is not None for r in rows)
    return {
        "ok": bool(rows), "n_days": len(rows), "skipped_rows": skipped,
        "columns_matched": {k: v for k, v in cols.items() if v},
        "has_solar_radiation": has_solar,
        "days": rows,
        "note": (
            "Solar radiation present, so reference ET is a full Penman-Monteith "
            "calculation." if has_solar else
            "No solar radiation column, so ET will fall back to a temperature "
            "method with more uncertainty. If the station measures radiation, "
            "include that column."),
    }


DAYMET_URL = "https://daymet.ornl.gov/single-pixel/api/data"
PRISM_URL = "https://services.nacse.org/prism/data/public/4km"
CDL_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue"
DNREC_ALLOCATION_URL = ("https://data.delaware.gov/resource/water-allocation-permits.json")


async def fetch_daymet(lat: float, lon: float, start: date, end: date,
                       client) -> dict:
    """Gridded Daymet, 1 km daily. The fallback Activity 2 names.

    Used when no DEOS station is close enough, or to backfill a station outage.
    Daymet has no wind, so ET from it falls back to a temperature method — the
    interface says so rather than letting a gridded number look like a measured
    one.
    """
    params = {"lat": lat, "lon": lon,
              "vars": "tmax,tmin,prcp,srad,dayl,vp",
              "start": start.isoformat(), "end": end.isoformat(),
              "format": "csv"}
    try:
        r = await client.get(DAYMET_URL, params=params, timeout=30.0)
        r.raise_for_status()
    except Exception as exc:
        return {"ok": False, "reason": str(exc)[:200]}

    lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
    header_idx = next((i for i, l in enumerate(lines) if l.lower().startswith("year")), 0)
    reader = csv.DictReader(lines[header_idx:])
    days = []
    for row in reader:
        try:
            yr = int(float(row["year"]))
            yday = int(float(row["yday"]))
            srad = float(row.get("srad (W/m^2)", 0))
            dayl = float(row.get("dayl (s)", 0))
            days.append({
                "day": (date(yr, 1, 1) + timedelta(days=yday - 1)).isoformat(),
                "tmax_f": float(row["tmax (deg c)"]) * 9 / 5 + 32,
                "tmin_f": float(row["tmin (deg c)"]) * 9 / 5 + 32,
                "rain_in": float(row["prcp (mm/day)"]) / 25.4,
                # Daymet reports mean daylight radiation; convert to daily total.
                "solar_rad_mj_m2": round(srad * dayl / 1e6, 3),
                "wind_mph": None,
            })
        except (KeyError, ValueError):
            continue
    return {"ok": bool(days), "source": "daymet_1km", "n_days": len(days),
            "days": days,
            "note": ("Daymet carries no wind speed, so reference ET from it is a "
                     "temperature method rather than full Penman-Monteith. Prefer "
                     "a DEOS station where one is within reasonable distance.")}


async def fetch_cdl_crop(lat: float, lon: float, year: int, client) -> dict:
    """Cropping history from the NASS Cropland Data Layer, 30 m.

    Activity 2 uses CDL to identify the primary crops from 2008 to present. The
    rotation matters directly to the nitrogen recommendation — a soybean year
    before corn is worth 30 to 40 pounds of nitrogen credit on Delmarva — so
    reading it rather than asking the grower to remember is worth the call.
    """
    params = {"year": year, "x": lon, "y": lat}
    try:
        r = await client.get(CDL_URL, params=params, timeout=25.0)
        r.raise_for_status()
        m = re.search(r"<Result>(.*?)</Result>", r.text, re.S)
        payload = m.group(1) if m else r.text
        code = re.search(r"value:\s*\"?(\d+)", payload)
        name = re.search(r"category:\s*\"([^\"]+)", payload)
    except Exception as exc:
        return {"ok": False, "reason": str(exc)[:200]}
    if not code:
        return {"ok": False, "reason": "no_cdl_value"}
    value = int(code.group(1))
    return {"ok": True, "year": year, "cdl_code": value,
            "cdl_name": name.group(1) if name else CDL_CROPS.get(value, "Unknown"),
            "crop_key": CDL_TO_CROP.get(value)}


# The CDL classes that actually appear on Delmarva cropland.
CDL_CROPS = {
    1: "Corn", 5: "Soybeans", 4: "Sorghum", 24: "Winter Wheat", 21: "Barley",
    26: "Winter Wheat/Soybeans Double Crop", 42: "Dry Beans", 43: "Potatoes",
    48: "Watermelons", 12: "Sweet Corn", 61: "Fallow/Idle", 37: "Other Hay",
    59: "Sod/Grass Seed", 2: "Cotton", 36: "Alfalfa", 176: "Grassland/Pasture",
}

CDL_TO_CROP = {
    1: "corn_grain", 5: "soybean_full", 4: "sorghum", 24: "wheat", 21: "barley",
    26: "soybean_double", 42: "lima_bean", 43: "potato", 48: "watermelon",
    12: "sweet_corn",
}


async def fetch_water_allocation(parcel_or_permit: str, client) -> dict:
    """Water withdrawal permit lookup.

    Activity 2 pulls diversions and withdrawals from state databases — the
    Colorado Decision Support System there, the DNREC water allocation permit
    record here. Delaware requires an allocation permit to withdraw more than
    50,000 gallons per day, which covers essentially every irrigated field, and
    the permitted rate is a hard constraint on any recommendation the software
    makes.
    """
    try:
        r = await client.get(DNREC_ALLOCATION_URL,
                             params={"permit_number": parcel_or_permit},
                             timeout=20.0)
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:
        return {"ok": False, "reason": str(exc)[:200],
                "hint": ("The DNREC open data endpoint moves. Enter the permitted "
                         "rate by hand from the permit; it is the number in "
                         "gallons per minute on the allocation certificate.")}
    if not rows:
        return {"ok": False, "reason": "not_found"}
    row = rows[0]
    return {"ok": True, "permit": row.get("permit_number"),
            "permitted_gpm": _safe_float(row.get("allocation_gpm")),
            "annual_limit_gallons": _safe_float(row.get("annual_allocation")),
            "source": row.get("source_type"),
            "raw": row}


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Reprojection for rasters that arrive in UTM
# ---------------------------------------------------------------------------

def utm_to_wgs84(easting: float, northing: float, zone: int = 18,
                 northern: bool = True) -> tuple[float, float]:
    """UTM to latitude and longitude, WGS84.

    Delaware is UTM zone 18N, and most UAV processing software writes its
    orthomosaics and index rasters in UTM rather than geographic coordinates.
    Without this, every drone raster lands in the Gulf of Guinea.
    """
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    x = easting - 500_000.0
    y = northing if northern else northing - 10_000_000.0
    m = y / 0.9996
    mu = m / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu))
    n1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    ep2 = e2 / (1 - e2)
    c1 = ep2 * math.cos(phi1) ** 2
    r1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * 0.9996)
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120
           ) / math.cos(phi1)
    return (math.degrees(lon) + (zone * 6 - 183), math.degrees(lat))


def raster_to_wgs84_points(raster: Raster, stride: int = 1,
                           utm_zone: int = 18) -> list[tuple[float, float, float]]:
    """Raster pixels as (lon, lat, value), reprojecting from UTM when needed."""
    pts = raster.sample_points(stride)
    if raster.epsg in (4326, None) and abs(raster.origin_x) <= 180:
        return pts
    zone = utm_zone
    if raster.epsg and 32600 < raster.epsg < 32700:
        zone = raster.epsg - 32600
    return [(*utm_to_wgs84(x, y, zone), v) for x, y, v in pts]


# ---------------------------------------------------------------------------
# ESRI shapefile — Activity 3, "crop yield in ... feature class (e.g., ESRI
# shapefile or Google KML) formats"
#
# The browser parses .shp geometry for drawing a boundary, but a yield export is
# a different problem: the numbers a grower cares about are in the .dbf
# attribute table, not the geometry file, and the pair has to be read together.
# Parsed here rather than pulled in as a dependency because the whole ingest
# layer is stdlib-only by design — a research group installing this on a machine
# without GDAL should still be able to read a yield monitor export.
# ---------------------------------------------------------------------------

# ESRI shape type codes. Multipatch (31) is deliberately unsupported: it never
# comes off a yield monitor and silently returning nothing for it would be worse
# than saying so.
_SHP_NULL = 0
_SHP_POINT, _SHP_POLYLINE, _SHP_POLYGON, _SHP_MULTIPOINT = 1, 3, 5, 8
_SHP_POINT_Z, _SHP_POLYLINE_Z, _SHP_POLYGON_Z, _SHP_MULTIPOINT_Z = 11, 13, 15, 18
_SHP_POINT_M, _SHP_POLYLINE_M, _SHP_POLYGON_M, _SHP_MULTIPOINT_M = 21, 23, 25, 28

_SHP_NAMES = {
    0: "Null", 1: "Point", 3: "Polyline", 5: "Polygon", 8: "MultiPoint",
    11: "PointZ", 13: "PolylineZ", 15: "PolygonZ", 18: "MultiPointZ",
    21: "PointM", 23: "PolylineM", 25: "PolygonM", 28: "MultiPointM",
    31: "MultiPatch",
}

_POINT_TYPES = {_SHP_POINT, _SHP_POINT_Z, _SHP_POINT_M}
_POLYLINE_TYPES = {_SHP_POLYLINE, _SHP_POLYLINE_Z, _SHP_POLYLINE_M}
_POLYGON_TYPES = {_SHP_POLYGON, _SHP_POLYGON_Z, _SHP_POLYGON_M}
_MULTIPOINT_TYPES = {_SHP_MULTIPOINT, _SHP_MULTIPOINT_Z, _SHP_MULTIPOINT_M}


class ShapefileError(ValueError):
    """Raised with text meant to be shown to the person who uploaded the file."""


def _u32be(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 4], "big", signed=False)


def _i32le(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 4], "little", signed=True)


def _f64le(b: bytes, o: int) -> float:
    return struct.unpack_from("<d", b, o)[0]


def read_shp_geometry(shp: bytes) -> list[dict]:
    """Geometry records from a .shp file, as GeoJSON geometry dicts.

    Ring orientation is preserved as written. ESRI's convention is the opposite
    of GeoJSON's — outer rings run clockwise — so holes are identified by signed
    area rather than by position, which is what actually distinguishes them.
    """
    if len(shp) < 100:
        raise ShapefileError("That .shp file is too short to be a shapefile.")
    if _u32be(shp, 0) != 9994:
        raise ShapefileError(
            "That does not look like a .shp file. The first four bytes are "
            "wrong — check you uploaded the geometry file and not the .shx "
            "index or the .dbf table.")

    out: list[dict] = []
    offset = 100                      # past the fixed header
    n = len(shp)
    while offset + 8 <= n:
        content_words = _u32be(shp, offset + 4)
        rec_start = offset + 8
        rec_end = rec_start + content_words * 2
        if rec_end > n:
            break                     # truncated final record; keep what parsed
        if rec_end - rec_start < 4:
            offset = rec_end
            continue
        stype = _i32le(shp, rec_start)
        out.append(_parse_shape(shp, rec_start, rec_end, stype))
        offset = rec_end
    return out


def _parse_shape(b: bytes, start: int, end: int, stype: int) -> dict:
    if stype == _SHP_NULL:
        return {"type": None, "coordinates": None}

    if stype in _POINT_TYPES:
        return {"type": "Point",
                "coordinates": [_f64le(b, start + 4), _f64le(b, start + 12)]}

    if stype in _MULTIPOINT_TYPES:
        npts = _i32le(b, start + 36)
        base = start + 40
        return {"type": "MultiPoint",
                "coordinates": [[_f64le(b, base + 16 * i),
                                 _f64le(b, base + 16 * i + 8)]
                                for i in range(npts)]}

    if stype in _POLYLINE_TYPES or stype in _POLYGON_TYPES:
        nparts = _i32le(b, start + 36)
        npts = _i32le(b, start + 40)
        part_base = start + 44
        pt_base = part_base + 4 * nparts
        parts = [_i32le(b, part_base + 4 * i) for i in range(nparts)]
        parts.append(npts)
        rings: list[list[list[float]]] = []
        for i in range(nparts):
            ring = [[_f64le(b, pt_base + 16 * j),
                     _f64le(b, pt_base + 16 * j + 8)]
                    for j in range(parts[i], parts[i + 1])]
            if ring:
                rings.append(ring)
        if stype in _POLYLINE_TYPES:
            if len(rings) == 1:
                return {"type": "LineString", "coordinates": rings[0]}
            return {"type": "MultiLineString", "coordinates": rings}
        return _rings_to_polygon(rings)

    raise ShapefileError(
        f"Shape type {_SHP_NAMES.get(stype, stype)} is not supported. "
        "Field boundaries and yield exports are points, lines or polygons.")


def _signed_area(ring: Sequence[Sequence[float]]) -> float:
    s = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % len(ring)][0], ring[(i + 1) % len(ring)][1]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _rings_to_polygon(rings: list[list[list[float]]]) -> dict:
    """Group rings into polygons using signed area to find holes.

    A shapefile stores every ring of every polygon in one flat list. In ESRI's
    convention an outer ring is clockwise (negative signed area in standard
    orientation) and a hole is counter-clockwise. A grassed waterway or an
    irrigation pond cut out of a field is a hole, and treating it as a separate
    field would inflate the acreage the grower is billed for water on.
    """
    if not rings:
        return {"type": "Polygon", "coordinates": []}

    polys: list[list[list[list[float]]]] = []
    for ring in rings:
        if _signed_area(ring) < 0:            # outer, by ESRI convention
            polys.append([ring])
        elif polys:                            # hole in the most recent outer
            polys[-1].append(ring)
        else:                                  # hole with no outer: treat as outer
            polys.append([ring])

    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def read_dbf(dbf: bytes) -> list[dict]:
    """Attribute records from a .dbf table.

    dBase III/IV as written by every yield monitor and GIS package. Numeric
    fields come back as floats or ints, logicals as bools, dates as ISO strings,
    and anything unparseable as the raw trimmed text rather than None — a yield
    column with a stray unit suffix is still worth showing the grower.
    """
    if len(dbf) < 32:
        raise ShapefileError("That .dbf attribute table is too short to read.")

    n_records = int.from_bytes(dbf[4:8], "little")
    header_len = int.from_bytes(dbf[8:10], "little")
    record_len = int.from_bytes(dbf[10:12], "little")

    fields: list[tuple[str, str, int, int]] = []
    pos = 32
    while pos < header_len - 1 and dbf[pos] != 0x0D:
        raw = dbf[pos:pos + 32]
        name = raw[0:11].split(b"\x00")[0].decode("latin-1").strip()
        ftype = chr(raw[11])
        flen = raw[16]
        fdec = raw[17]
        if name:
            fields.append((name, ftype, flen, fdec))
        pos += 32

    out: list[dict] = []
    base = header_len
    for r in range(n_records):
        start = base + r * record_len
        if start + record_len > len(dbf):
            break
        if dbf[start:start + 1] == b"*":      # tombstoned record
            continue
        rec: dict[str, Any] = {}
        off = start + 1
        for name, ftype, flen, fdec in fields:
            raw = dbf[off:off + flen]
            off += flen
            rec[name] = _dbf_value(raw, ftype, fdec)
        out.append(rec)
    return out


def _dbf_value(raw: bytes, ftype: str, fdec: int):
    text = raw.decode("latin-1").strip()
    if ftype in ("N", "F", "B", "O"):
        if not text or text in ("-", "."):
            return None
        try:
            return float(text) if (fdec or "." in text) else int(text)
        except ValueError:
            return text
    if ftype == "L":
        return text.upper() in ("Y", "T")
    if ftype == "D":
        if len(text) == 8 and text.isdigit():
            return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
        return text or None
    if ftype == "I":                           # 32-bit int, stored binary
        return int.from_bytes(raw[:4], "little", signed=True)
    return text or None


def _epsg_from_prj(prj_text: str) -> tuple[int | None, str | None]:
    """Best-effort CRS identification from a .prj, for the UTM warning path."""
    t = prj_text.upper()
    if "GEOGCS" in t and "PROJCS" not in t:
        return 4326, None
    m = re.search(r"UTM[_ ]ZONE[_ ](\d+)N", t)
    if m:
        return 32600 + int(m.group(1)), None
    m = re.search(r'PROJCS\s*\[\s*"([^"]+)"', prj_text)
    return None, (m.group(1) if m else None)


def _looks_geographic(geoms: list[dict]) -> bool:
    """True if the coordinate values can only be degrees.

    A .prj is a claim, not a guarantee. Export tools routinely write a projected
    .prj beside coordinates that were never projected, and believing the file
    would reproject degrees as though they were metres and drop the field in the
    Atlantic. Nothing in UTM is under 180 easting inside a real zone, so the
    magnitudes settle it.
    """
    for g in geoms:
        for x, y in _iter_coords(g):
            if abs(x) > 180.0 or abs(y) > 90.0:
                return False
    return True


def _iter_coords(geom: dict):
    def walk(c):
        if not c:
            return
        if isinstance(c[0], (int, float)):
            yield c[0], c[1]
            return
        for sub in c:
            yield from walk(sub)
    yield from walk(geom.get("coordinates") or [])


def read_shapefile(shp: bytes, dbf: bytes | None = None,
                   prj: str | None = None) -> dict:
    """A .shp plus its .dbf into GeoJSON features.

    Geometry and attributes are matched by record order, which is what the
    format guarantees. If the two files disagree on record count the shorter one
    wins and the mismatch is reported rather than silently truncated — a yield
    file whose .dbf is short is usually a partial copy off a USB stick, and the
    grower needs to know that before they zone their field on it.
    """
    geoms = read_shp_geometry(shp)
    attrs = read_dbf(dbf) if dbf else []
    warnings: list[str] = []

    epsg = None
    reproject = False
    if prj:
        epsg, projcs = _epsg_from_prj(prj)
        if epsg and 32600 < epsg < 32700:
            if _looks_geographic(geoms):
                # The .prj is lying. Trusting it would put the field in the
                # Atlantic, so the coordinates win and the mismatch is reported.
                warnings.append(
                    f"The .prj claims UTM zone {epsg - 32600}N, but the "
                    "coordinates are already latitude and longitude. The .prj "
                    "was ignored and the coordinates used as written. Re-export "
                    "with a matching .prj if the field plots in the wrong place.")
                epsg = 4326
            else:
                reproject = True
                warnings.append(
                    f"The .prj says UTM zone {epsg - 32600}N. Coordinates were "
                    "converted to latitude and longitude on the way in.")
        elif epsg is None:
            warnings.append(
                f"The .prj describes a projected system ({projcs or 'unknown'}) "
                "that this reader does not convert. If the field lands in the "
                "wrong place on the map, re-export it in WGS84 geographic "
                "coordinates.")

    if attrs and len(attrs) != len(geoms):
        warnings.append(
            f"The .shp has {len(geoms)} shapes but the .dbf has {len(attrs)} "
            "records. They should match; the extra records were dropped.")

    features = []
    for i, geom in enumerate(geoms):
        if geom.get("type") is None:
            continue
        if reproject:
            geom = _reproject_geometry(geom, epsg - 32600)
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": attrs[i] if i < len(attrs) else {},
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "n_features": len(features),
        "fields": sorted(attrs[0]) if attrs else [],
        "epsg": epsg,
        "warnings": warnings,
    }


def _reproject_geometry(geom: dict, zone: int) -> dict:
    def conv(c):
        if not c:
            return c
        if isinstance(c[0], (int, float)):
            # utm_to_wgs84 returns (lon, lat), which is already GeoJSON order.
            lon, lat = utm_to_wgs84(c[0], c[1], zone=zone)
            return [lon, lat]
        return [conv(x) for x in c]
    return {"type": geom["type"], "coordinates": conv(geom["coordinates"])}


def read_shapefile_zip(data: bytes) -> dict:
    """A zipped shapefile, the way GIS software and yield monitors export it.

    Everything is read from the archive rather than the filesystem so an upload
    never has to be written to disk. Sidecar files are optional — a .shp alone
    still yields geometry — but without the .dbf there are no yield values, and
    that is called out rather than left as an empty result.
    """
    members: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                name = info.filename
                if name.startswith("__MACOSX/") or name.endswith("/"):
                    continue
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext in ("shp", "dbf", "prj", "cpg"):
                    members[ext] = zf.read(info)
    except zipfile.BadZipFile:
        raise ShapefileError(
            "That file could not be opened as a .zip. Export the boundary "
            "again and upload the whole folder the software writes.")

    if "shp" not in members:
        raise ShapefileError(
            "No .shp inside that zip. A shapefile is a set of files — export "
            "the layer again and include every file, not just one.")

    prj_text = None
    if "prj" in members:
        prj_text = members["prj"].decode("latin-1", errors="replace")

    result = read_shapefile(members["shp"], members.get("dbf"), prj_text)
    if "dbf" not in members:
        result["warnings"].append(
            "No .dbf in the archive, so the shapes came through without any "
            "attributes. Yield values live in the .dbf — re-export with it if "
            "you meant to upload a yield map.")
    return result


def yield_from_shapefile_zip(data: bytes, value_key: str = "yield") -> dict:
    """Yield map straight from a zipped monitor export.

    Closes the loop Activity 3 asks for: the grower uploads what their combine
    wrote, and it becomes the same zone statistics as a raster yield map.
    """
    parsed = read_shapefile_zip(data)
    stats = yield_from_features(parsed["features"], value_key=value_key)
    stats["warnings"] = parsed["warnings"]
    stats["fields_available"] = parsed["fields"]
    if stats.get("n", 0) == 0 and parsed["fields"]:
        stats["note"] = (
            "No yield column was recognised. The attribute table has: "
            + ", ".join(parsed["fields"][:12])
            + ". Pass one of these as the value column.")
    return stats
