"""
Sign-in, invitations and account management over HTTP.

Mounted by app/main.py. Every route here is thin: the rules live in app/auth.py
and this file only translates them into requests and responses.

On cookies: the session token goes back as an HttpOnly cookie so page
JavaScript cannot read it, which takes a stolen token off the table for the
commonest kind of cross-site scripting. SameSite=Lax keeps it off cross-site
requests. `Secure` is set whenever the request arrived over HTTPS — not
unconditionally, because that would break sign-in on a laptop running the API
on plain localhost, which is how everyone will first try this.
"""

from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import auth, notify

router = APIRouter(prefix="/api/auth", tags=["auth"])

COOKIE = "geo4irr_session"


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------

def current_user(geo4irr_session: str | None = Cookie(default=None)) -> auth.User:
    user = auth.user_for_session(geo4irr_session or "")
    if user is None:
        raise HTTPException(401, "Sign in to continue.")
    return user


def current_admin(user: auth.User = Depends(current_user)) -> auth.User:
    if user.role != "admin":
        raise HTTPException(403, "That is an administrator action.")
    return user


def optional_user(geo4irr_session: str | None = Cookie(default=None)) -> auth.User | None:
    return auth.user_for_session(geo4irr_session or "")


def _set_cookie(resp: Response, request: Request, token: str) -> None:
    resp.set_cookie(
        COOKIE, token, httponly=True, samesite="lax",
        secure=request.url.scheme == "https",
        max_age=auth.SESSION_HOURS * 3600, path="/")


# --------------------------------------------------------------------------
# request bodies
# --------------------------------------------------------------------------

class Credentials(BaseModel):
    email: str
    password: str


class SetupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class InviteRequest(BaseModel):
    email: str
    role: str = Field(default="admin", pattern="^(admin|customer)$")


class SignUpRequest(BaseModel):
    """Open registration, for growers."""
    email: str
    password: str
    name: str = ""


class NotifyFlag(BaseModel):
    notify: bool


class RedeemRequest(BaseModel):
    token: str
    password: str
    name: str = ""


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class ActiveFlag(BaseModel):
    active: bool


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@router.get("/status")
async def status(user: auth.User | None = Depends(optional_user)) -> dict:
    """What the sign-in screen needs to know before anyone types anything."""
    return {
        "needs_setup": auth.needs_setup(),
        "signed_in": user is not None,
        "user": None if user is None else {
            "email": user.email, "name": user.name, "role": user.role,
            "must_change": user.must_change, "notify": user.notify},
        "seats": auth.seats(),               # administrators only
        "seats_left": auth.seats_left(),
        "customers": auth.count_customers(), # uncapped
        "open_signup": True,
        "notifications": notify.status(),
    }


@router.post("/setup")
async def setup(req: SetupRequest, request: Request, response: Response) -> dict:
    """Create the first administrator.

    Open only while there are no accounts at all — the moment one exists this
    returns 409 forever. That is what makes it safe to leave mounted: there is
    no window in which a second person can call it, and no default credentials
    shipped in the code for anyone to find.
    """
    if not auth.needs_setup():
        raise HTTPException(409, "This installation already has accounts. "
                                 "Sign in, or ask an administrator to invite you.")
    try:
        user = auth.create_user(req.email, req.password, name=req.name,
                                role="admin", enforce_seats=False)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    token, _ = auth.start_session(user)
    _set_cookie(response, request, token)
    return {"email": user.email, "role": user.role, "name": user.name}


@router.post("/signup")
async def signup(req: SignUpRequest, request: Request, response: Response) -> dict:
    """Anyone can make a customer account. No invitation, no seat, no approval.

    A grower who wants to look at their own field should be able to. What is
    worth guarding is the administrator role, and that stays invitation-only.
    """
    try:
        user = auth.register_customer(req.email, req.password, name=req.name)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    token, _ = auth.start_session(user)
    _set_cookie(response, request, token)
    return {"email": user.email, "name": user.name, "role": user.role}


@router.post("/login")
async def login(creds: Credentials, request: Request, response: Response) -> dict:
    try:
        user = auth.authenticate(creds.email, creds.password)
    except auth.AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    token, expires = auth.start_session(user)
    _set_cookie(response, request, token)
    return {"email": user.email, "name": user.name, "role": user.role,
            "expires": expires.isoformat(),
            # A seeded or reset password has been written down somewhere to get
            # here, so the first thing asked for is a new one.
            "must_change": user.must_change}


@router.post("/logout")
async def logout(response: Response,
                 geo4irr_session: str | None = Cookie(default=None)) -> dict:
    auth.end_session(geo4irr_session or "")
    response.delete_cookie(COOKIE, path="/")
    return {"signed_out": True}


@router.get("/me")
async def me(user: auth.User = Depends(current_user)) -> dict:
    return {"email": user.email, "name": user.name, "role": user.role}


@router.post("/password")
async def password(req: PasswordChange,
                   user: auth.User = Depends(current_user)) -> dict:
    try:
        auth.change_password(user, req.current_password, req.new_password)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"changed": True,
            "note": "Every other session for this account was signed out."}


# ---- invitations ----------------------------------------------------------

@router.post("/invite")
async def invite(req: InviteRequest, request: Request,
                 admin: auth.User = Depends(current_admin)) -> dict:
    try:
        made = auth.create_invite(req.email, admin, role=req.role)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc

    base = str(request.base_url).rstrip("/")
    return {
        **made,
        "link": f"{base}/?invite={made['token']}",
        "seats_left": auth.seats_left(),
        "note": ("Send this link to them yourself. It works once, expires in "
                 f"{auth.INVITE_DAYS} days, and anyone holding it can claim the "
                 "seat — so send it the way you would send a key."),
    }


@router.get("/invite/{token}")
async def check_invite(token: str) -> dict:
    """Read an invitation without spending it, so the form can show who it is for."""
    try:
        return auth.peek_invite(token)
    except auth.AuthError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/register")
async def register(req: RedeemRequest, request: Request,
                   response: Response) -> dict:
    try:
        user = auth.redeem_invite(req.token, req.password, name=req.name)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    token, _ = auth.start_session(user)
    _set_cookie(response, request, token)
    return {"email": user.email, "name": user.name, "role": user.role}


# ---- administration -------------------------------------------------------

@router.post("/notifications")
async def notifications(flag: NotifyFlag,
                        user: auth.User = Depends(current_user)) -> dict:
    auth.set_notify(user.id, flag.notify)
    return {"notify": flag.notify, **notify.status()}


@router.get("/users")
async def users(admin: auth.User = Depends(current_admin)) -> dict:
    return {"users": auth.list_users(), "invites": auth.list_invites(),
            "seats": auth.seats(), "seats_left": auth.seats_left()}


@router.post("/users/{user_id}/active")
async def set_active(user_id: int, flag: ActiveFlag,
                     admin: auth.User = Depends(current_admin)) -> dict:
    try:
        auth.set_active(user_id, flag.active, admin)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": user_id, "active": flag.active, "seats_left": auth.seats_left()}
