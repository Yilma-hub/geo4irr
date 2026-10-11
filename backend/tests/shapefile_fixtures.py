import io, struct, zipfile

HEADER_TAIL = b"\x00" * 20

def point_shp(points):
    recs = b""
    for i, (x, y) in enumerate(points, 1):
        content = struct.pack("<i", 1) + struct.pack("<dd", x, y)
        recs += struct.pack(">ii", i, len(content) // 2) + content
    return _hdr(recs, 1, points) + recs

def polygon_shp(rings):
    """rings: list of list-of-rings; first ring outer (CW), rest holes (CCW)."""
    recs = b""
    for i, parts in enumerate(rings, 1):
        flat = [p for r in parts for p in r]
        idx, n = [], 0
        for r in parts:
            idx.append(n); n += len(r)
        xs = [p[0] for p in flat]; ys = [p[1] for p in flat]
        content = struct.pack("<i", 5)
        content += struct.pack("<4d", min(xs), min(ys), max(xs), max(ys))
        content += struct.pack("<ii", len(parts), len(flat))
        content += b"".join(struct.pack("<i", v) for v in idx)
        content += b"".join(struct.pack("<dd", p[0], p[1]) for p in flat)
        recs += struct.pack(">ii", i, len(content) // 2) + content
    allp = [p for parts in rings for r in parts for p in r]
    return _hdr(recs, 5, allp) + recs

def _hdr(recs, stype, pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    h = struct.pack(">i", 9994) + HEADER_TAIL
    h += struct.pack(">i", (100 + len(recs)) // 2)
    h += struct.pack("<i", 1000) + struct.pack("<i", stype)
    h += struct.pack("<4d", min(xs), min(ys), max(xs), max(ys))
    h += struct.pack("<4d", 0, 0, 0, 0)
    return h

def dbf(rows, fields):
    hlen = 32 + 32 * len(fields) + 1
    rlen = 1 + sum(f[2] for f in fields)
    out = bytearray(bytes([3, 125, 1, 1]) + struct.pack("<i", len(rows)))
    out += struct.pack("<hh", hlen, rlen) + b"\x00" * 20
    for name, ftype, flen, fdec in fields:
        out += name.encode().ljust(11, b"\x00") + ftype + b"\x00" * 4
        out += bytes([flen, fdec]) + b"\x00" * 14
    out += b"\x0d"
    for row in rows:
        out += b"*" if row.get("_deleted") else b" "
        for name, ftype, flen, fdec in fields:
            v = row.get(name, "")
            if ftype == b"N":
                s = f"{float(v):.{fdec}f}".rjust(flen)
            elif ftype == b"L":
                s = ("T" if v else "F").ljust(flen)
            elif ftype == b"D":
                s = str(v).ljust(flen)
            else:
                s = str(v).ljust(flen)
            out += s.encode()[:flen].ljust(flen)
    out += b"\x1a"
    return bytes(out)

def zip_it(members, compress=True):
    buf = io.BytesIO()
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buf, "w", mode) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()
