"""
Two kinds of account, and the evening note.

Growers register themselves and are uncapped. Administrators are invited and
occupy one of twenty seats. The seeded administrator exists so the tool is
usable the moment it starts, and is forced to change its password on first
sign-in because a password that had to be written down to reach someone is a
password other people have seen.
"""

from __future__ import annotations

import datetime
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_DB", str(tmp_path / "auth.db"))
    monkeypatch.setenv("GEO4IRR_SEATS", "20")
    monkeypatch.setenv("GEO4IRR_ADMIN_EMAIL", "yilma.wub@outlook.com")
    monkeypatch.setenv("GEO4IRR_ADMIN_PASSWORD", "nfp123abc")
    from app import auth as auth_mod
    importlib.reload(auth_mod)
    auth_mod.init_db()
    return auth_mod


@pytest.fixture()
def client(env, monkeypatch):
    monkeypatch.setenv("GEO4IRR_NO_SCHEDULER", "1")
    from app import auth_routes, main, notify
    importlib.reload(notify)
    importlib.reload(auth_routes)
    importlib.reload(main)
    return TestClient(main.app)


# --------------------------------------------------------------------------
# the seeded administrator
# --------------------------------------------------------------------------

def test_the_configured_administrator_exists_on_first_start(env):
    user = env.seed_admin()
    assert user.email == "yilma.wub@outlook.com"
    assert user.role == "admin"
    assert env.authenticate("yilma.wub@outlook.com", "nfp123abc").role == "admin"


def test_the_seeded_password_is_hashed_like_any_other(env):
    """Convenience at setup is not a reason to store it in the clear."""
    env.seed_admin()
    raw = open(env.DB_PATH, "rb").read()
    assert b"nfp123abc" not in raw


def test_the_seeded_account_must_change_its_password_before_anything_else(env):
    """It was written down somewhere to reach the person. Treat it as spent."""
    env.seed_admin()
    assert env.authenticate("yilma.wub@outlook.com", "nfp123abc").must_change is True

    user = env.get_user_by_email("yilma.wub@outlook.com")
    env.change_password(user, "nfp123abc", "a much stronger passphrase")
    assert env.authenticate("yilma.wub@outlook.com",
                            "a much stronger passphrase").must_change is False


def test_seeding_happens_once_and_never_overwrites(env):
    first = env.seed_admin()
    assert first is not None
    assert env.seed_admin() is None, "a second seed would reset a live account"


def test_the_seeded_password_bypasses_the_strength_rule_but_a_chosen_one_does_not(env):
    """nfp123abc is nine characters — accepted only as a seed, never as a choice."""
    env.seed_admin()
    user = env.get_user_by_email("yilma.wub@outlook.com")
    with pytest.raises(env.AuthError) as exc:
        env.change_password(user, "nfp123abc", "short1")
    assert "12 characters" in str(exc.value)


# --------------------------------------------------------------------------
# growers register themselves
# --------------------------------------------------------------------------

def test_anyone_can_make_a_grower_account_without_an_invitation(env):
    user = env.register_customer("grower@farm.com", "a fine long passphrase",
                                 name="A Grower")
    assert user.role == "customer"
    assert env.authenticate("grower@farm.com", "a fine long passphrase")


def test_growers_are_not_capped_by_the_seat_limit(env, monkeypatch):
    """Seats limit administrators. A grower-facing tool that turns growers away
    at twenty has the limit in the wrong place."""
    monkeypatch.setenv("GEO4IRR_SEATS", "2")
    env.seed_admin()
    for i in range(25):
        env.register_customer(f"g{i}@farm.com", "a fine long passphrase")
    assert env.count_customers() == 25
    assert env.seats_left() == 1, "growers should not consume administrator seats"


def test_administrators_still_need_an_invitation(client):
    client.post("/api/auth/signup", json={
        "email": "grower@farm.com", "password": "a fine long passphrase"})
    assert client.get("/api/auth/me").json()["role"] == "customer"
    assert client.post("/api/auth/invite",
                       json={"email": "x@y.org"}).status_code == 403
    assert client.get("/api/auth/users").status_code == 403


def test_the_seat_limit_still_applies_to_invited_administrators(env, monkeypatch):
    monkeypatch.setenv("GEO4IRR_SEATS", "1")
    admin = env.seed_admin()
    with pytest.raises(env.AuthError) as exc:
        env.create_invite("second@admin.org", admin, role="admin")
    assert "administrator seats" in str(exc.value)


def test_login_reports_that_a_password_change_is_due(client):
    r = client.post("/api/auth/login", json={
        "email": "yilma.wub@outlook.com", "password": "nfp123abc"})
    assert r.status_code == 200
    assert r.json()["must_change"] is True


# --------------------------------------------------------------------------
# the evening note
# --------------------------------------------------------------------------

def lines(notify):
    return [
        notify.FieldLine("Home pivot", "irrigate_now", 1.42, 1.01, 1.00, 18.4,
                         "Evesboro sand holds 2.5 in."),
        notify.FieldLine("Airport field", "hold", 0.30, 1.26, 0.0, 0.0, ""),
    ]


def test_the_note_goes_out_at_six_in_the_evening(env):
    from app import notify
    importlib.reload(notify)
    assert notify.SEND_HOUR == 18 and notify.SEND_MINUTE == 0
    assert "18:00" in notify.status()["sends_at"]


def test_nothing_is_sent_when_no_field_needs_water(env):
    """A daily "nothing to do" is a mail people stop opening — and then they
    miss the one that mattered."""
    from app import notify
    importlib.reload(notify)
    env.seed_admin()
    quiet = [notify.FieldLine("Airport field", "hold", 0.3, 1.26, 0, 0, "")]
    result = notify.run_evening_round(lambda u: quiet,
                                      day=datetime.date(2026, 8, 4))
    assert result["sent"] == 0 and result["skipped"] >= 1


def test_a_field_that_needs_water_produces_a_note_naming_it(env):
    from app import notify
    importlib.reload(notify)
    env.seed_admin()
    user = env.get_user_by_email("yilma.wub@outlook.com")
    subject, body = notify.render(user, lines(notify), datetime.date(2026, 8, 4))
    assert "Home pivot" in subject
    assert "1.00 in" in body and "18.4 hours" in body
    assert "Airport field" in body           # listed, but under "holding"
    assert "Check the soil before you turn on a pump" in body


def test_turning_notes_off_takes_someone_out_of_the_round(env):
    from app import notify
    importlib.reload(notify)
    env.seed_admin()
    grower = env.register_customer("g@farm.com", "a fine long passphrase")
    assert len(env.notify_list()) == 2
    env.set_notify(grower.id, False)
    assert [u.email for u in env.notify_list()] == ["yilma.wub@outlook.com"]


def test_without_an_smtp_server_the_note_is_logged_not_silently_dropped(env, caplog):
    """A scheduler that appears to work but sends nothing is the worst outcome."""
    from app import notify
    importlib.reload(notify)
    env.seed_admin()
    assert notify.configured() is False
    with caplog.at_level("INFO", logger="geo4irr.notify"):
        result = notify.run_evening_round(lambda u: lines(notify),
                                          day=datetime.date(2026, 8, 4))
    assert result["delivered"] is False
    assert result["sent"] == 1
    assert any("Home pivot" in r.getMessage() for r in caplog.records), (
        "the message was neither sent nor written down")


def test_the_status_endpoint_says_plainly_whether_mail_will_go_out(client):
    body = client.get("/api/notifications/status").json()
    assert body["enabled"] is False
    assert "18:00" in body["sends_at"]
    assert "SMTP_HOST" in body["note"]


def test_only_an_administrator_can_fire_a_test_round(client):
    client.post("/api/auth/signup", json={
        "email": "grower@farm.com", "password": "a fine long passphrase"})
    assert client.post("/api/notifications/test").status_code == 403
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={
        "email": "yilma.wub@outlook.com", "password": "nfp123abc"})
    assert client.post("/api/notifications/test").status_code == 200
