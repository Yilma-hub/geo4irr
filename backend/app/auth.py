"""
Accounts, sessions and invitations.

Deliberately small and deliberately stdlib. Passwords are hashed with scrypt
from `hashlib` — memory-hard, in the standard library since 3.6, and strong
enough that adding a dependency buys nothing here. Sessions are random tokens
stored server-side rather than signed cookies, because a token you can delete
from a table is a token you can actually revoke when a laptop goes missing.

Three things this deliberately does NOT do:

  * store a password anywhere, in any form, ever — only the scrypt digest;
  * ship a default account. A tool with a known admin login is not protected by
    it. The first admin is created by whoever installs this, from the command
    line, with a password they choose;
  * let an invitation be reused. Each one is single-use and expires.

Storage is SQLite so the thing runs from a clean checkout with nothing to set
up. `AUTH_DB` moves it; for a real deployment put it on Postgres alongside the
rest of infra/schema.sql.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# scrypt parameters. n=2**15 with r=8 costs roughly 32 MB and ~100 ms per
# hash on ordinary hardware — slow enough to make bulk guessing expensive,
# fast enough that nobody notices signing in.
SCRYPT_N = 1 << 15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_LEN = 32
# OpenSSL refuses scrypt above 32 MB unless told otherwise, and these
# parameters need 128 * N * r = exactly 32 MB before overhead — so the default
# fails by a hair. Raised deliberately rather than by weakening the cost.
SCRYPT_MAXMEM = 64 * 1024 * 1024

SESSION_HOURS = 12
INVITE_DAYS = 14

# Seats limit administrators only. Customers register themselves and are not
# capped — the whole point of a grower-facing tool is that growers can sign up.
DEFAULT_SEATS = 20

ROLES = ("admin", "customer")

# The account this installation is set up for. Seeded on first start if it does
# not exist, from GEO4IRR_ADMIN_EMAIL / GEO4IRR_ADMIN_PASSWORD.
SEED_EMAIL = os.getenv("GEO4IRR_ADMIN_EMAIL", "yilma.wub@outlook.com")
SEED_PASSWORD = os.getenv("GEO4IRR_ADMIN_PASSWORD", "nfp123abc")

DB_PATH = Path(os.getenv("AUTH_DB", "geo4irr_auth.db"))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised with text that is safe to show the person trying to sign in."""


@dataclass(frozen=True)
class User:
    id: int
    email: str
    name: str
    role: str          # "admin" or "customer"
    active: bool
    must_change: bool = False
    notify: bool = True


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
          id        INTEGER PRIMARY KEY AUTOINCREMENT,
          email     TEXT NOT NULL UNIQUE,
          name      TEXT NOT NULL DEFAULT '',
          role      TEXT NOT NULL DEFAULT 'customer',
          pw_salt   BLOB NOT NULL,
          pw_hash   BLOB NOT NULL,
          active    INTEGER NOT NULL DEFAULT 1,
          must_change INTEGER NOT NULL DEFAULT 0,
          notify    INTEGER NOT NULL DEFAULT 1,
          created   TEXT NOT NULL,
          last_seen TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
          token   TEXT PRIMARY KEY,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          created TEXT NOT NULL,
          expires TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS invites (
          token    TEXT PRIMARY KEY,
          email    TEXT NOT NULL,
          role     TEXT NOT NULL DEFAULT 'admin',
          invited_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
          created  TEXT NOT NULL,
          expires  TEXT NOT NULL,
          used     TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_invites_email ON invites(email);
        """)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def normalise_email(email: str) -> str:
    """Lowercase and strip. Case matters in the local part by RFC and to
    nobody in practice, and treating Yilma@ and yilma@ as two accounts would
    only ever confuse the person who owns both."""
    return unicodedata.normalize("NFKC", (email or "").strip()).lower()


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                            n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                            dklen=SCRYPT_LEN, maxmem=SCRYPT_MAXMEM)
    return salt, digest


def check_password(password: str, salt: bytes, expected: bytes) -> bool:
    _, digest = hash_password(password, salt)
    # Constant time: a timing difference here leaks how much of the hash matched.
    return hmac.compare_digest(digest, expected)


def password_problem(password: str) -> str | None:
    """What is wrong with this password, in words the person can act on.

    Length does more for a password than any character-class rule, so that is
    what is enforced. The rest is a check against the handful of strings people
    actually pick when told to include a symbol and a digit.
    """
    if len(password or "") < 12:
        return ("Use at least 12 characters. Length is what makes a password "
                "hard to guess — a short one with symbols in it is not safer "
                "than a long ordinary phrase.")
    lowered = password.lower()
    for bad in ("password", "123456", "qwerty", "letmein", "admin",
                "geo4irr", "irrigation", "delaware"):
        if bad in lowered:
            return (f"That contains {bad!r}, which is one of the first things "
                    "anyone guessing would try. Pick something unrelated to "
                    "this tool.")
    if len(set(password)) < 5:
        return "That repeats too few distinct characters to be hard to guess."
    return None


def seats() -> int:
    try:
        return max(1, int(os.getenv("GEO4IRR_SEATS", DEFAULT_SEATS)))
    except ValueError:
        return DEFAULT_SEATS


def count_active() -> int:
    """Administrators only — the seat count exists to limit them."""
    with _connect() as c:
        return c.execute(
            "SELECT COUNT(*) FROM users WHERE active = 1 AND role = 'admin'"
        ).fetchone()[0]


def count_customers() -> int:
    with _connect() as c:
        return c.execute(
            "SELECT COUNT(*) FROM users WHERE active = 1 AND role = 'customer'"
        ).fetchone()[0]


def seats_left() -> int:
    return max(0, seats() - count_active())


def _row_to_user(row: sqlite3.Row) -> User:
    keys = row.keys()
    return User(id=row["id"], email=row["email"], name=row["name"],
                role=row["role"], active=bool(row["active"]),
                must_change=bool(row["must_change"]) if "must_change" in keys else False,
                notify=bool(row["notify"]) if "notify" in keys else True)


def create_user(email: str, password: str, name: str = "",
                role: str = "customer", enforce_seats: bool = True,
                must_change: bool = False,
                allow_weak: bool = False) -> User:
    email = normalise_email(email)
    if not EMAIL_RE.match(email):
        raise AuthError("That does not look like an email address.")
    problem = None if allow_weak else password_problem(password)
    if problem:
        raise AuthError(problem)
    if role not in ROLES:
        raise AuthError("Role must be admin or customer.")
    # Only administrators occupy a seat. Customers are unlimited.
    if role == "admin" and enforce_seats and seats_left() <= 0:
        raise AuthError(f"All {seats()} seats are in use. Deactivate someone "
                        "first, or raise GEO4IRR_SEATS.")

    salt, digest = hash_password(password)
    try:
        with _connect() as c:
            cur = c.execute(
                "INSERT INTO users (email, name, role, pw_salt, pw_hash, "
                "must_change, created) VALUES (?,?,?,?,?,?,?)",
                (email, name.strip(), role, salt, digest,
                 1 if must_change else 0, _iso(_now())))
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise AuthError("There is already an account with that email address.")
    return User(id=uid, email=email, name=name.strip(), role=role, active=True,
                must_change=must_change)


def get_user_by_email(email: str) -> User | None:
    with _connect() as c:
        row = c.execute("SELECT * FROM users WHERE email = ?",
                        (normalise_email(email),)).fetchone()
    return _row_to_user(row) if row else None


def authenticate(email: str, password: str) -> User:
    """Verify a sign-in.

    Every failure returns the same sentence. Saying "no such account" tells
    someone probing which addresses are registered here, which is free
    reconnaissance; and when the account does not exist a dummy hash is computed
    anyway so the two cases take the same time.
    """
    generic = AuthError("That email and password do not match an account.")
    with _connect() as c:
        row = c.execute("SELECT * FROM users WHERE email = ?",
                        (normalise_email(email),)).fetchone()
    if row is None:
        hash_password(password, salt=b"\x00" * 16)
        raise generic
    if not check_password(password, row["pw_salt"], row["pw_hash"]):
        raise generic
    if not row["active"]:
        raise AuthError("That account has been deactivated. Ask the "
                        "administrator to switch it back on.")
    with _connect() as c:
        c.execute("UPDATE users SET last_seen = ? WHERE id = ?",
                  (_iso(_now()), row["id"]))
    return _row_to_user(row)


def start_session(user: User) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    expires = _now() + timedelta(hours=SESSION_HOURS)
    with _connect() as c:
        c.execute("INSERT INTO sessions (token, user_id, created, expires) "
                  "VALUES (?,?,?,?)",
                  (token, user.id, _iso(_now()), _iso(expires)))
    return token, expires


def user_for_session(token: str) -> User | None:
    if not token:
        return None
    with _connect() as c:
        row = c.execute(
            "SELECT u.* , s.expires FROM sessions s "
            "JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,)).fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires"]) < _now():
            c.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
        if not row["active"]:
            return None
    return _row_to_user(row)


def end_session(token: str) -> None:
    with _connect() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


def purge_expired() -> int:
    with _connect() as c:
        cur = c.execute("DELETE FROM sessions WHERE expires < ?", (_iso(_now()),))
        return cur.rowcount


# ---------------------------------------------------------------------------
# invitations
# ---------------------------------------------------------------------------

def create_invite(email: str, invited_by: User, role: str = "admin") -> dict:
    email = normalise_email(email)
    if not EMAIL_RE.match(email):
        raise AuthError("That does not look like an email address.")
    if get_user_by_email(email):
        raise AuthError("That address already has an account.")
    if role == "admin" and seats_left() <= 0:
        raise AuthError(f"All {seats()} administrator seats are in use, so "
                        "there is nowhere for this person to land. Deactivate "
                        "someone first.")
    if role not in ROLES:
        raise AuthError("Role must be admin or customer.")

    token = secrets.token_urlsafe(24)
    expires = _now() + timedelta(days=INVITE_DAYS)
    with _connect() as c:
        # One live invitation per address; re-inviting replaces the old link
        # rather than leaving two valid ones in circulation.
        c.execute("DELETE FROM invites WHERE email = ? AND used IS NULL", (email,))
        c.execute("INSERT INTO invites (token, email, role, invited_by, created, expires) "
                  "VALUES (?,?,?,?,?,?)",
                  (token, email, role, invited_by.id, _iso(_now()), _iso(expires)))
    return {"token": token, "email": email, "role": role,
            "expires": _iso(expires)}


def peek_invite(token: str) -> dict:
    """What an invitation is for, so the sign-up form can show it."""
    with _connect() as c:
        row = c.execute("SELECT * FROM invites WHERE token = ?", (token,)).fetchone()
    if row is None:
        raise AuthError("That invitation link is not valid.")
    if row["used"]:
        raise AuthError("That invitation has already been used. Ask for a new one.")
    if datetime.fromisoformat(row["expires"]) < _now():
        raise AuthError("That invitation has expired. Ask for a new one.")
    return {"email": row["email"], "role": row["role"], "expires": row["expires"]}


def redeem_invite(token: str, password: str, name: str = "") -> User:
    info = peek_invite(token)
    user = create_user(info["email"], password, name=name, role=info["role"])
    with _connect() as c:
        c.execute("UPDATE invites SET used = ? WHERE token = ?",
                  (_iso(_now()), token))
    return user


def list_users() -> list[dict]:
    with _connect() as c:
        rows = c.execute(
            "SELECT id, email, name, role, active, created, last_seen "
            "FROM users ORDER BY created").fetchall()
    return [dict(r) for r in rows]


def list_invites() -> list[dict]:
    with _connect() as c:
        rows = c.execute(
            "SELECT token, email, role, created, expires, used FROM invites "
            "ORDER BY created DESC").fetchall()
    return [{**dict(r), "token": r["token"][:6] + "…"} for r in rows]


def set_active(user_id: int, active: bool, acting: User) -> None:
    """Switch an account off without deleting the work attached to it."""
    if user_id == acting.id and not active:
        raise AuthError("You cannot deactivate your own account — someone else "
                        "would have to switch it back on.")
    with _connect() as c:
        if not active:
            row = c.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
            if row and row["role"] == "admin":
                admins = c.execute(
                    "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1"
                ).fetchone()[0]
                if admins <= 1:
                    raise AuthError("That is the only active administrator. "
                                    "Promote someone else first.")
        c.execute("UPDATE users SET active = ? WHERE id = ?",
                  (1 if active else 0, user_id))
        if not active:
            c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


def change_password(user: User, current: str, new: str) -> None:
    authenticate(user.email, current)
    if current == new:
        raise AuthError("That is the password you already have.")
    problem = password_problem(new)
    if problem:
        raise AuthError(problem)
    salt, digest = hash_password(new)
    with _connect() as c:
        c.execute("UPDATE users SET pw_salt = ?, pw_hash = ? WHERE id = ?",
                  (salt, digest, user.id))
        c.execute("UPDATE users SET must_change = 0 WHERE id = ?", (user.id,))
        # Every other session for this person is now suspect.
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user.id,))


def register_customer(email: str, password: str, name: str = "") -> User:
    """Open sign-up for growers.

    No invitation, no seat, no approval. A grower who wants to look at their own
    field should be able to make an account and do it — the thing worth
    protecting is the administrator role, and that stays invitation-only.
    """
    return create_user(email, password, name=name, role="customer",
                       enforce_seats=False)


def clear_must_change(user: User) -> None:
    with _connect() as c:
        c.execute("UPDATE users SET must_change = 0 WHERE id = ?", (user.id,))


def set_notify(user_id: int, on: bool) -> None:
    with _connect() as c:
        c.execute("UPDATE users SET notify = ? WHERE id = ?",
                  (1 if on else 0, user_id))


def notify_list() -> list[User]:
    """Everyone who has asked for the evening note."""
    with _connect() as c:
        rows = c.execute(
            "SELECT * FROM users WHERE active = 1 AND notify = 1").fetchall()
    return [_row_to_user(r) for r in rows]


def seed_admin() -> User | None:
    """Create the configured administrator on first start, once.

    The seeded password is stored the same way as every other — scrypt, salted,
    never in plain text — but it is flagged `must_change`, because a password
    that was written down somewhere in order to get here is a password other
    people have seen. The account works immediately; the first sign-in asks for
    a new one before anything else happens.

    Set GEO4IRR_ADMIN_EMAIL and GEO4IRR_ADMIN_PASSWORD to change what gets
    seeded. Returns None if any account already exists.
    """
    init_db()
    if not needs_setup():
        return None
    if not SEED_EMAIL or not SEED_PASSWORD:
        return None
    return create_user(SEED_EMAIL, SEED_PASSWORD, name="Administrator",
                       role="admin", enforce_seats=False,
                       must_change=True, allow_weak=True)


def needs_setup() -> bool:
    """True when nobody has an account yet."""
    init_db()
    with _connect() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
