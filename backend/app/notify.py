"""
The evening note.

Once a day at six, everyone who wants one gets a short email about their fields:
what the balance says, whether anything needs water tomorrow, and how much.
Six in the evening because that is when the decision actually gets made — after
the day's heat is in, early enough to move a pivot before dark.

Two design choices worth stating.

**Nothing goes out unless there is something to say.** A daily email that says
"nothing to do" four days in a row is an email people stop opening, and then
they miss the fifth one that mattered. If no field on the account needs
attention, the note is skipped for that person that day.

**Sending requires SMTP credentials you supply.** There is no mail server built
in and no third-party sending account baked into this code. Until SMTP_HOST is
set, the scheduler still runs, still builds every message, and writes them to
the log instead of sending — so the timing and the content can be checked
before anything reaches a real inbox.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import date, datetime
from email.message import EmailMessage
from email.utils import formataddr
from zoneinfo import ZoneInfo

from . import auth

log = logging.getLogger("geo4irr.notify")

SEND_HOUR = int(os.getenv("NOTIFY_HOUR", "18"))
SEND_MINUTE = int(os.getenv("NOTIFY_MINUTE", "0"))
TIMEZONE = os.getenv("NOTIFY_TZ", "America/New_York")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "Geo4Irr <no-reply@geo4irr.local>")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") not in ("0", "false", "False")

APP_URL = os.getenv("GEO4IRR_URL", "http://localhost:8080")


def configured() -> bool:
    return bool(SMTP_HOST)


def status() -> dict:
    return {
        "enabled": configured(),
        "sends_at": f"{SEND_HOUR:02d}:{SEND_MINUTE:02d} {TIMEZONE}",
        "smtp_host": SMTP_HOST or None,
        "recipients": len(auth.notify_list()),
        "note": ("Evening notes will be sent." if configured() else
                 "No SMTP server is configured, so the evening note is written "
                 "to the log instead of sent. Set SMTP_HOST, SMTP_USER and "
                 "SMTP_PASS to turn sending on."),
    }


@dataclass
class FieldLine:
    name: str
    decision: str          # irrigate_now, irrigate_soon, hold, ...
    depletion_in: float
    raw_in: float
    net_in: float
    hours: float
    reason: str

    @property
    def urgent(self) -> bool:
        return self.decision in ("irrigate_now", "irrigate_soon")


def _subject(lines: list[FieldLine]) -> str:
    now = [f for f in lines if f.decision == "irrigate_now"]
    soon = [f for f in lines if f.decision == "irrigate_soon"]
    if now and len(now) == 1:
        return f"Geo4Irr — {now[0].name} needs water"
    if now:
        return f"Geo4Irr — {len(now)} fields need water"
    if soon:
        return f"Geo4Irr — {len(soon)} field{'s' if len(soon) > 1 else ''} coming due"
    return "Geo4Irr — nothing needs water today"


def render(user: auth.User, lines: list[FieldLine], day: date) -> tuple[str, str]:
    """Subject and body. Plain text on purpose — it survives every mail client,
    reads fine on a phone in a truck, and cannot hide anything in styling."""
    urgent = [f for f in lines if f.urgent]
    rest = [f for f in lines if not f.urgent]
    who = (user.name or user.email).split()[0] if user.name else "there"

    out = [f"Good evening {who},", ""]
    if urgent:
        out.append(f"As of {day:%A %d %B}, these fields want water:")
        out.append("")
        for f in urgent:
            when = "tonight or tomorrow" if f.decision == "irrigate_now" else "in a day or two"
            out.append(f"  {f.name}")
            out.append(f"    Apply about {f.net_in:.2f} in — roughly "
                       f"{f.hours:.1f} hours of run time — {when}.")
            out.append(f"    Depletion {f.depletion_in:.2f} in against "
                       f"{f.raw_in:.2f} in of readily available water.")
            if f.reason:
                out.append(f"    {f.reason}")
            out.append("")
    if rest:
        out.append("Holding for now:")
        for f in rest:
            out.append(f"  {f.name} — depletion {f.depletion_in:.2f} in "
                       f"of {f.raw_in:.2f} in available")
        out.append("")

    out += [
        "These figures come from the weather record and from what you have told",
        "Geo4Irr about each field. Check the soil before you turn on a pump —",
        "a model five miles from a thunderstorm does not know it missed you.",
        "",
        f"Open Geo4Irr: {APP_URL}",
        "",
        "To stop these, turn off evening notes in your account.",
    ]
    return _subject(lines), "\n".join(out)


def send(to_email: str, subject: str, body: str) -> bool:
    """True if it actually went out; False if it was only logged."""
    if not configured():
        log.info("evening note for %s (not sent, no SMTP configured)\n"
                 "Subject: %s\n%s", to_email, subject, body)
        return False

    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as s:
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            if SMTP_STARTTLS:
                s.starttls(context=context)
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
    return True


def run_evening_round(build_lines, day: date | None = None) -> dict:
    """Build and send tonight's notes.

    `build_lines(user)` returns that person's fields as FieldLine objects. It is
    passed in rather than imported so this module stays testable without a
    database full of fields and a weather service behind it.
    """
    day = day or datetime.now(ZoneInfo(TIMEZONE)).date()
    sent = skipped = failed = 0
    for user in auth.notify_list():
        try:
            lines = build_lines(user)
        except Exception:
            log.exception("could not build the evening note for %s", user.email)
            failed += 1
            continue

        if not lines or not any(f.urgent for f in lines):
            # Nothing to act on. Silence is the useful message here.
            skipped += 1
            continue

        subject, body = render(user, lines, day)
        try:
            send(user.email, subject, body)
            sent += 1
        except Exception:
            log.exception("could not send the evening note to %s", user.email)
            failed += 1

    log.info("evening round for %s: %d sent, %d skipped, %d failed",
             day, sent, skipped, failed)
    return {"date": day.isoformat(), "sent": sent, "skipped": skipped,
            "failed": failed, "delivered": configured()}


def schedule(scheduler, build_lines) -> None:
    """Attach the daily job to an APScheduler instance."""
    scheduler.add_job(
        lambda: run_evening_round(build_lines),
        "cron", hour=SEND_HOUR, minute=SEND_MINUTE,
        timezone=ZoneInfo(TIMEZONE),
        id="evening_note", replace_existing=True,
        misfire_grace_time=3600)
    log.info("evening note scheduled for %02d:%02d %s (%s)",
             SEND_HOUR, SEND_MINUTE, TIMEZONE,
             "sending" if configured() else "logging only")
