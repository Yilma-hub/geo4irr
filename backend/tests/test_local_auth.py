"""
Sign-in with no server behind it.

Accounts kept in the browser. This is a weaker thing than server-side auth and
the tool says so on screen, but "weaker" is not "sloppy": the password itself is
never stored, one account's digest is useless against another's, and a wrong
password and an unknown address are indistinguishable. These tests hold those
properties, and hold the honesty notice in place — because the notice is what
makes it reasonable to ship a browser-only lock at all.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

INDEX = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def run(tmp_path, body, secure=True):
    """Drive the page with Web Crypto present (as on localhost or https)."""
    script = tmp_path / "t.mjs"
    script.write_text(
        'import { JSDOM } from "jsdom"; import fs from "fs";\n'
        'import { webcrypto } from "crypto";\n'
        f'const dom=new JSDOM(fs.readFileSync({str(INDEX)!r},"utf8"),\n'
        ' {runScripts:"dangerously",pretendToBeVisual:true,\n'
        '  url:"http://localhost:8080/",\n'
        '  beforeParse(w){\n'
        f'    Object.defineProperty(w,"crypto",{{value:webcrypto,configurable:true}});\n'
        f'    Object.defineProperty(w,"isSecureContext",{{value:{str(secure).lower()},configurable:true}});\n'
        '    w.fetch=async()=>({ok:false,status:503,json:async()=>({})});\n'
        '  }});\n'
        'const w=dom.window, d=w.document;\n'
        'await new Promise(r=>w.addEventListener("load",r));\n'
        + body + "\n")
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True,
                          timeout=180, cwd="/tmp")
    assert proc.returncode == 0, proc.stderr[-1500:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_sign_in_screen_appears_with_no_backend_at_all(tmp_path):
    r = run(tmp_path, '''
await w.eval("requireSignIn")();
await new Promise(r=>setTimeout(r,400));
const card=d.querySelector(".authcard");
console.log(JSON.stringify({
  shown: !!card,
  usingServer: w.eval("usingServer()"),
  title: card && card.querySelector("h1").textContent,
  roles: card ? [...card.querySelectorAll(".rolebtn strong")].map(x=>x.textContent) : [],
}));''')
    assert r["usingServer"] is False
    assert r["shown"] is True, "no sign-in screen without a server"
    assert r["roles"] == ["Grower", "Administrator"]


def test_the_password_is_never_written_into_browser_storage(tmp_path):
    """The whole point of hashing. If the seeded password is sitting in
    localStorage in plain text, the lock is decorative."""
    r = run(tmp_path, '''
await w.eval("localSeedAdmin")();
const dump = w.localStorage.getItem("geo4irr.accounts");
console.log(JSON.stringify({
  containsPassword: dump.includes("nfp123abc"),
  algorithm: JSON.parse(dump)[0].hash.split("$")[0],
  hasSalt: !!JSON.parse(dump)[0].salt,
}));''')
    assert r["containsPassword"] is False
    assert r["algorithm"] == "pbkdf2"
    assert r["hasSalt"] is True


def test_two_accounts_with_the_same_password_get_different_digests(tmp_path):
    """Per-account salt. Without it, one cracked digest unlocks everyone who
    picked the same phrase, and identical passwords are visible as identical."""
    r = run(tmp_path, '''
await w.eval("localCreateAccount")("a@x.com","the very same passphrase","A","customer",{});
await w.eval("localCreateAccount")("b@x.com","the very same passphrase","B","customer",{});
const [a,b]=w.eval("localAccounts()");
console.log(JSON.stringify({sameHash:a.hash===b.hash, sameSalt:a.salt===b.salt}));''')
    assert r["sameHash"] is False
    assert r["sameSalt"] is False


def test_a_wrong_password_and_an_unknown_address_read_the_same(tmp_path):
    """Otherwise the sign-in box is a tool for discovering who has an account."""
    r = run(tmp_path, '''
await w.eval("localCreateAccount")("real@x.com","a genuine long passphrase","R","customer",{});
const msgs=[];
for(const [e,p] of [["real@x.com","wrong entirely here"],["ghost@x.com","wrong entirely here"]]){
  try{ await w.eval("localSignIn")(e,p); msgs.push("(no error)"); }
  catch(err){ msgs.push(err.message); }
}
console.log(JSON.stringify({msgs, distinct:new Set(msgs).size}));''')
    assert r["distinct"] == 1, f"the two failures read differently: {r['msgs']}"


def test_the_owner_account_is_seeded_once_and_must_change_its_password(tmp_path):
    r = run(tmp_path, '''
const first = await w.eval("localSeedAdmin")();
const second = await w.eval("localSeedAdmin")();
const me = w.eval("localAccounts()")[0];
console.log(JSON.stringify({
  email: first.email, role: first.role, mustChange: first.mustChange,
  secondWasSkipped: second === null,
  total: w.eval("localAccounts().length"),
}));''')
    assert r["email"] == "yilma.wub@outlook.com"
    assert r["role"] == "admin"
    assert r["mustChange"] is True, (
        "a seeded password was written down to get here; it has to be replaced")
    assert r["secondWasSkipped"] is True
    assert r["total"] == 1


def test_changing_the_password_clears_the_prompt_and_rerolls_the_salt(tmp_path):
    r = run(tmp_path, '''
await w.eval("localSeedAdmin")();
const before = w.eval("localAccounts()")[0];
const user = w.eval("localAccounts()")[0];
await w.eval("localChangePassword")(user, "nfp123abc", "a much better passphrase");
const after = w.eval("localAccounts()")[0];
let oldStillWorks = true;
try { await w.eval("localSignIn")(after.email, "nfp123abc"); }
catch(e){ oldStillWorks = false; }
console.log(JSON.stringify({
  mustChange: after.mustChange,
  saltChanged: before.salt !== after.salt,
  oldStillWorks,
}));''')
    assert r["mustChange"] is False
    assert r["saltChanged"] is True, "reusing the salt leaks that the password changed"
    assert r["oldStillWorks"] is False


def test_a_weak_password_is_refused_when_chosen_but_allowed_when_seeded(tmp_path):
    """nfp123abc is nine characters. Fine as a starting point that must be
    replaced; not fine as something someone picks and keeps."""
    r = run(tmp_path, '''
let refused = null;
try { await w.eval("localCreateAccount")("x@y.com","short1","X","customer",{}); }
catch(e){ refused = e.message; }
const seeded = await w.eval("localSeedAdmin")();
console.log(JSON.stringify({refused, seededAnyway: !!seeded}));''')
    assert r["refused"] and "12 characters" in r["refused"]
    assert r["seededAnyway"] is True


def test_growers_can_register_themselves_with_no_invitation(tmp_path):
    r = run(tmp_path, '''
await w.eval("localSeedAdmin")();
await w.eval("submitLocal")("/api/auth/signup",
  {email:"g@farm.com", password:"grower own passphrase", name:"A Grower"});
console.log(JSON.stringify({
  total: w.eval("localAccounts().length"),
  role: w.eval("localCurrentUser()").role,
  signedIn: w.eval("localCurrentUser()").email,
}));''')
    assert r["total"] == 2
    assert r["role"] == "customer"
    assert r["signedIn"] == "g@farm.com"


def test_the_screen_says_what_a_browser_only_lock_cannot_do(tmp_path):
    """This is the part that makes shipping it defensible.

    Someone deciding whether to keep a grower's records here has to be told
    where the lock actually is — that anyone with the developer tools open can
    walk past it, because the data is already on their machine.
    """
    r = run(tmp_path, '''
const foot = w.eval("localModeFoot()");
console.log(JSON.stringify({
  mentionsDevTools: /developer tools/i.test(foot),
  mentionsAlgorithm: /PBKDF2/i.test(foot),
  pointsAtTheBackend: /API_BASE/.test(foot),
  emptyWhenServerConfigured: w.eval('API_BASE="http://x"; localModeFoot()') === "",
}));''')
    assert r["mentionsDevTools"] is True, "the honest limit is not stated on screen"
    assert r["mentionsAlgorithm"] is True
    assert r["pointsAtTheBackend"] is True
    assert r["emptyWhenServerConfigured"] is True


def test_an_insecure_context_falls_back_and_says_so(tmp_path):
    """Opened as a file, the browser withholds Web Crypto. The weaker hash is
    used — and named — rather than the page pretending nothing changed."""
    r = run(tmp_path, '''
const foot = w.eval("localModeFoot()");
await w.eval("localCreateAccount")("x@y.com","a fine long passphrase","X","customer",{});
console.log(JSON.stringify({
  strong: w.eval("strongCryptoAvailable()"),
  algorithm: w.eval("localAccounts()")[0].hash.split("$")[0],
  warns: /weaker hash|withheld/i.test(foot),
}));''', secure=False)
    assert r["strong"] is False
    assert r["algorithm"] == "weak"
    assert r["warns"] is True, "the fallback happened silently"
