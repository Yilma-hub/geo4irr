"""
Credentials and configuration.

Everything secret is read from the environment, never from this file. The
repository is intended to go on GitHub, and a password committed to a public
repository is found by automated scrapers in minutes, not days. Rotating it
afterwards is not enough on its own: the old value stays in the git history
forever unless the history itself is rewritten.

So the rule here is simple and absolute. This module reads values. It never
contains them.

Setting them up:

    cp .env.example .env
    # open .env, put your real values in it
    python -m app.check_setup

`.env` is in `.gitignore` and must stay there. `.env.example` has the variable
names with empty values and is safe to commit, which is what makes the real
file discoverable by anyone cloning the repo without exposing anything.

For a server, set real environment variables instead of shipping a .env file.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = pathlib.Path(os.getenv("GEO4IRR_ENV_FILE", REPO_ROOT / ".env"))


def load_env_file(path: pathlib.Path = ENV_FILE) -> dict[str, str]:
    """Read a .env file into the process environment.

    Deliberately small rather than pulling in a dependency: the format is
    KEY=value, one per line, # for comments. Values already set in the real
    environment win, because a server's own configuration should never be
    overridden by a file that happened to be left in the working directory.
    """
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


load_env_file()


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Credential:
    """One secret, and what breaks without it."""
    name: str
    value: str
    purpose: str
    without_it: str
    signup: str = ""

    @property
    def present(self) -> bool:
        return bool(self.value)

    def masked(self) -> str:
        """Enough to recognise, never enough to use.

        Printed by the setup check so you can tell a wrong value from a missing
        one without the real thing ending up in a terminal scrollback, a
        screenshot, or a support thread.
        """
        if not self.value:
            return "(not set)"
        if len(self.value) <= 8:
            return self.value[0] + "*" * (len(self.value) - 1)
        return f"{self.value[:4]}{'*' * 8}{self.value[-2:]}"


# ---------------------------------------------------------------------------
# Copernicus, for Sentinel-2 through Sentinel Hub
# ---------------------------------------------------------------------------
# Free account at https://dataspace.copernicus.eu. Create an OAuth client under
# Sentinel Hub and use the client id and secret, NOT your sign-in email and
# password: an OAuth client can be revoked on its own and cannot be used to log
# into your account if it leaks.

CDSE_CLIENT_ID = Credential(
    "CDSE_CLIENT_ID", _get("CDSE_CLIENT_ID"),
    purpose="Sentinel-2 imagery through Sentinel Hub",
    without_it=("Falls back to the open AWS mirror, which needs no account and "
                "serves the same pixels. You only need this for the higher rate "
                "limits and the processing API."),
    signup="https://dataspace.copernicus.eu")

CDSE_CLIENT_SECRET = Credential(
    "CDSE_CLIENT_SECRET", _get("CDSE_CLIENT_SECRET"),
    purpose="Sentinel-2 imagery through Sentinel Hub",
    without_it="See CDSE_CLIENT_ID.",
    signup="https://dataspace.copernicus.eu")

# ---------------------------------------------------------------------------
# DEOS, for Delaware weather
# ---------------------------------------------------------------------------

DEOS_WS_USER = Credential(
    "DEOS_WS_USER", _get("DEOS_WS_USER"),
    purpose="Delaware Environmental Observing System weather web service",
    without_it=("Falls back to National Weather Service forecasts and Delaware "
                "climatology. Usable for a demo, wrong for a real schedule."),
    signup="http://services.deos.udel.edu/ws/subscribe.php")

DEOS_WS_PASS = Credential(
    "DEOS_WS_PASS", _get("DEOS_WS_PASS"),
    purpose="DEOS weather web service",
    without_it="See DEOS_WS_USER.",
    signup="http://services.deos.udel.edu/ws/subscribe.php")

# ---------------------------------------------------------------------------
# Mail, for the evening note
# ---------------------------------------------------------------------------

SMTP_HOST = Credential(
    "SMTP_HOST", _get("SMTP_HOST"),
    purpose="Sending the six o'clock evening note",
    without_it="Notes are built and written to the log instead of sent.")

SMTP_PASS = Credential(
    "SMTP_PASS", _get("SMTP_PASS"),
    purpose="Sending the evening note",
    without_it="See SMTP_HOST. Outlook and Gmail need an app password here, "
               "not your normal account password.")

# ---------------------------------------------------------------------------
# The administrator account
# ---------------------------------------------------------------------------

ADMIN_EMAIL = Credential(
    "GEO4IRR_ADMIN_EMAIL", _get("GEO4IRR_ADMIN_EMAIL"),
    purpose="The administrator account created on first start",
    without_it="No account is seeded. Create one with python -m app.create_admin.")

ADMIN_PASSWORD = Credential(
    "GEO4IRR_ADMIN_PASSWORD", _get("GEO4IRR_ADMIN_PASSWORD"),
    purpose="The seeded administrator's first password",
    without_it="See GEO4IRR_ADMIN_EMAIL.")


ALL_CREDENTIALS = [
    CDSE_CLIENT_ID, CDSE_CLIENT_SECRET,
    DEOS_WS_USER, DEOS_WS_PASS,
    SMTP_HOST, SMTP_PASS,
    ADMIN_EMAIL, ADMIN_PASSWORD,
]


def status() -> dict:
    """What is configured, without revealing any of it.

    Safe to expose over HTTP and safe to paste into a bug report, which is the
    point: the commonest way a secret escapes is somebody pasting a config dump
    to ask why something is not working.
    """
    return {
        "env_file": str(ENV_FILE),
        "env_file_present": ENV_FILE.exists(),
        "credentials": [
            {"name": c.name, "configured": c.present, "value": c.masked(),
             "purpose": c.purpose,
             "consequence_if_missing": None if c.present else c.without_it,
             "signup": c.signup or None}
            for c in ALL_CREDENTIALS
        ],
        "note": ("Values are read from the environment, or from .env if it "
                 "exists. Nothing secret is stored in the source. Never commit "
                 ".env: it is in .gitignore and should stay there."),
    }
