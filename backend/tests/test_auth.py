"""
Accounts, sessions and invitations.

Authentication is the one part of this tool where a mistake is not a wrong
number on a screen — it is somebody else's field data. So these tests are less
about features working and more about the failures: what a wrong password
reveals, whether an invitation can be spent twice, whether the last
administrator can lock everyone out.
"""

from __future__ import annotations

import importlib
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A fresh database per test — accounts are global state by nature."""
    monkeypatch.setenv("AUTH_DB", str(tmp_path / "auth.db"))
    monkeypatch.setenv("GEO4IRR_SEATS", "20")
    # Seeding is covered in test_roles_and_notify. These tests are about the
    # empty-installation path, so it is switched off here — otherwise the first
    # account already exists and /setup is closed before the test starts.
    monkeypatch.setenv("GEO4IRR_ADMIN_EMAIL", "")
    monkeypatch.setenv("GEO4IRR_ADMIN_PASSWORD", "")
    from app import auth as auth_mod
    importlib.reload(auth_mod)
    auth_mod.init_db()
    return auth_mod


@pytest.fixture()
def client(env, monkeypatch):
    monkeypatch.setenv("GEO4IRR_NO_SCHEDULER", "1")
    from app import auth_routes, main
    importlib.reload(auth_routes)
    importlib.reload(main)
    return TestClient(main.app)


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------

def test_the_password_is_never_stored_in_any_readable_form(env):
    secret = "a memorable garden fence"
    env.create_user("a@example.org", secret, role="admin", enforce_seats=False)
    raw = open(env.DB_PATH, "rb").read()
    assert secret.encode() not in raw, "the password itself is in the database file"
    assert b"a memorable" not in raw


def test_the_same_password_hashes_differently_for_two_people(env):
    """Per-user salt. Without it, identical passwords are visibly identical,
    and one cracked hash unlocks everyone who chose the same phrase."""
    env.create_user("a@example.org", "identical passphrase here",
                    role="admin", enforce_seats=False)
    env.create_user("b@example.org", "identical passphrase here")
    users = env.list_users()
    assert len(users) == 2
    import sqlite3
    con = sqlite3.connect(env.DB_PATH)
    hashes = [r[0] for r in con.execute("SELECT pw_hash FROM users")]
    salts = [r[0] for r in con.execute("SELECT pw_salt FROM users")]
    assert hashes[0] != hashes[1]
    assert salts[0] != salts[1]


@pytest.mark.parametrize("bad,why", [
    ("short", "12 characters"),
    ("geo4irr rocks 2026", "geo4irr"),
    ("aaaaaaaaaaaaaaaa", "distinct"),
])
def test_weak_passwords_are_refused_with_a_reason(env, bad, why):
    problem = env.password_problem(bad)
    assert problem and why in problem


def test_a_long_ordinary_phrase_is_accepted(env):
    assert env.password_problem("the back forty needs water") is None


# --------------------------------------------------------------------------
# signing in
# --------------------------------------------------------------------------

def test_a_wrong_password_and_a_missing_account_are_indistinguishable(env):
    """Different messages would let anyone enumerate who has an account here."""
    env.create_user("real@example.org", "the correct passphrase",
                    role="admin", enforce_seats=False)
    messages = set()
    for email, pw in (("real@example.org", "wrong one"),
                      ("ghost@example.org", "wrong one")):
        with pytest.raises(env.AuthError) as e:
            env.authenticate(email, pw)
        messages.add(str(e.value))
    assert len(messages) == 1, f"the two failures read differently: {messages}"


def test_a_missing_account_still_costs_time_to_reject(env):
    """Returning instantly for an unknown address is a timing oracle."""
    env.create_user("real@example.org", "the correct passphrase",
                    role="admin", enforce_seats=False)

    def timed(email):
        start = time.perf_counter()
        with pytest.raises(env.AuthError):
            env.authenticate(email, "definitely wrong")
        return time.perf_counter() - start

    known = min(timed("real@example.org") for _ in range(3))
    unknown = min(timed("ghost@example.org") for _ in range(3))
    # Within 3x is enough to defeat a practical timing probe over a network.
    assert unknown > known / 3, (
        f"unknown addresses reject {known/max(unknown,1e-9):.1f}x faster")


def test_email_is_matched_regardless_of_case_and_stray_spaces(env):
    env.create_user("Yilma.Wub@Outlook.com", "the correct passphrase",
                    role="admin", enforce_seats=False)
    assert env.authenticate("  yilma.wub@outlook.com ",
                            "the correct passphrase").role == "admin"


def test_a_session_stops_working_when_it_is_ended(env):
    u = env.create_user("a@example.org", "the correct passphrase",
                        role="admin", enforce_seats=False)
    token, _ = env.start_session(u)
    assert env.user_for_session(token).email == "a@example.org"
    env.end_session(token)
    assert env.user_for_session(token) is None


def test_deactivating_someone_signs_them_out_immediately(env):
    """A revoked account with a live session is not revoked."""
    admin = env.create_user("admin@example.org", "the correct passphrase",
                            role="admin", enforce_seats=False)
    member = env.create_user("m@example.org", "another good passphrase")
    token, _ = env.start_session(member)
    assert env.user_for_session(token) is not None
    env.set_active(member.id, False, admin)
    assert env.user_for_session(token) is None


def test_changing_a_password_ends_every_other_session(env):
    u = env.create_user("a@example.org", "the first passphrase",
                        role="admin", enforce_seats=False)
    old, _ = env.start_session(u)
    env.change_password(u, "the first passphrase", "the second passphrase")
    assert env.user_for_session(old) is None
    assert env.authenticate("a@example.org", "the second passphrase")


# --------------------------------------------------------------------------
# invitations and seats
# --------------------------------------------------------------------------

def test_an_invitation_can_only_be_spent_once(env):
    admin = env.create_user("admin@example.org", "the correct passphrase",
                            role="admin", enforce_seats=False)
    inv = env.create_invite("new@example.org", admin)
    env.redeem_invite(inv["token"], "a brand new passphrase")
    with pytest.raises(env.AuthError) as e:
        env.redeem_invite(inv["token"], "trying it again now")
    assert "already been used" in str(e.value)


def test_reinviting_replaces_the_previous_link_rather_than_adding_one(env):
    """Two live links to one seat is one more than anybody intended."""
    admin = env.create_user("admin@example.org", "the correct passphrase",
                            role="admin", enforce_seats=False)
    first = env.create_invite("new@example.org", admin)
    second = env.create_invite("new@example.org", admin)
    with pytest.raises(env.AuthError):
        env.peek_invite(first["token"])
    assert env.peek_invite(second["token"])["email"] == "new@example.org"


def test_the_seat_limit_is_enforced_and_freed_by_deactivating(env, monkeypatch):
    monkeypatch.setenv("GEO4IRR_SEATS", "3")
    admin = env.create_user("admin@example.org", "the correct passphrase",
                            role="admin", enforce_seats=False)
    # Seats count administrators; growers are uncapped, so these are admins.
    env.create_user("b@example.org", "second good passphrase", role="admin")
    env.create_user("c@example.org", "third good passphrase", role="admin")
    assert env.seats_left() == 0
    with pytest.raises(env.AuthError) as e:
        env.create_invite("d@example.org", admin, role="admin")
    assert "seats are in use" in str(e.value)

    victim = env.get_user_by_email("c@example.org")
    env.set_active(victim.id, False, admin)
    assert env.seats_left() == 1
    assert env.create_invite("d@example.org", admin,
                             role="admin")["email"] == "d@example.org"


def test_the_last_administrator_cannot_be_switched_off(env):
    """A grower is not a substitute — deactivating the only admin locks the
    installation with nobody able to unlock it."""
    admin = env.create_user("admin@example.org", "the correct passphrase",
                            role="admin", enforce_seats=False)
    other = env.create_user("m@example.org", "another good passphrase",
                            role="customer")
    with pytest.raises(env.AuthError) as e:
        env.set_active(admin.id, False, other)
    assert "your own account" in str(e.value) or "only active" in str(e.value)
    assert env.get_user_by_email("admin@example.org").active


# --------------------------------------------------------------------------
# over HTTP
# --------------------------------------------------------------------------

def test_setup_closes_itself_after_the_first_account(client):
    assert client.get("/api/auth/status").json()["needs_setup"] is True
    ok = client.post("/api/auth/setup", json={
        "email": "yilma.wub@outlook.com", "password": "a fresh passphrase here",
        "name": "Yilma"})
    assert ok.status_code == 200 and ok.json()["role"] == "admin"

    again = client.post("/api/auth/setup", json={
        "email": "someone@else.org", "password": "sneaking in behind you"})
    assert again.status_code == 409, (
        "a second person could still claim administrator")


def test_protected_routes_refuse_a_visitor_with_no_session(client):
    client.post("/api/auth/setup", json={
        "email": "a@example.org", "password": "a fresh passphrase here"})
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/users").status_code == 401
    assert client.post("/api/auth/invite",
                       json={"email": "x@example.org"}).status_code == 401


def test_a_grower_cannot_invite_or_list_people(client):
    client.post("/api/auth/setup", json={
        "email": "admin@example.org", "password": "a fresh passphrase here"})
    client.post("/api/auth/logout")
    client.post("/api/auth/signup", json={
        "email": "grower@example.org", "password": "grower chosen passphrase"})
    assert client.get("/api/auth/me").json()["role"] == "customer"
    assert client.get("/api/auth/users").status_code == 403
    assert client.post("/api/auth/invite",
                       json={"email": "z@example.org"}).status_code == 403


def test_the_session_cookie_is_not_readable_by_page_scripts(client):
    """HttpOnly is what stops a cross-site script walking off with the session."""
    client.post("/api/auth/setup", json={
        "email": "a@example.org", "password": "a fresh passphrase here"})
    r = client.post("/api/auth/login", json={
        "email": "a@example.org", "password": "a fresh passphrase here"})
    cookie = r.headers.get("set-cookie", "")
    assert "httponly" in cookie.lower(), "the session cookie is readable by JavaScript"
    assert "samesite=lax" in cookie.lower()


def test_an_invitation_can_be_read_without_signing_in_but_reveals_only_its_target(client):
    client.post("/api/auth/setup", json={
        "email": "admin@example.org", "password": "a fresh passphrase here"})
    inv = client.post("/api/auth/invite",
                      json={"email": "new@example.org"}).json()
    client.post("/api/auth/logout")
    peek = client.get(f"/api/auth/invite/{inv['token']}")
    assert peek.status_code == 200
    assert set(peek.json()) == {"email", "role", "expires"}
    assert client.get("/api/auth/invite/not-a-real-token").status_code == 404
