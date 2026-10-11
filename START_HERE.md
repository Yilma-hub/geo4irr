# START HERE

A plain walkthrough. No prior setup assumed.

There are two ways to run Geo4Irr. Do **Part 1** first — it takes two minutes
and everything works except accounts. Do Part 2 later, when you want other
people signing in.

---

## Part 1 — See it working (2 minutes)

### Step 1. Unzip

Unzip `geo4irr.zip`. You get a folder called **`geo4irr`**. Inside it:

```
geo4irr/
├── frontend/     ← the app you look at
├── backend/      ← the server (Part 2)
├── samples/
├── tools/
└── README.md
```

### Step 2. Open a terminal

- **Windows** — press the Start button, type `powershell`, press Enter.
- **Mac** — press Cmd+Space, type `terminal`, press Enter.

A window with a text prompt opens. That's it.

### Step 3. Check you have Python

Type this and press Enter:

- **Windows** — `py --version`
- **Mac** — `python3 --version`

If it prints something like `Python 3.12.1`, go to Step 4.

If Windows says *"Python was not found; run without arguments to install from
the Microsoft Store"*, that is a placeholder, not Python. Install the real
thing:

1. Go to <https://www.python.org/downloads/>
2. Click the yellow **Download Python** button, then run the file
3. On the first screen of the installer, **tick "Add python.exe to PATH"**
   before you click Install. This is easy to miss and everything else depends
   on it.
4. Close PowerShell completely and open a fresh one, then check again.

### Step 4. Go to the frontend folder

Type `cd `, then a space, then **drag the `frontend` folder from your file
browser into the terminal window** and let go. It fills in the path for you.
Press Enter.

**The space after `cd` matters.** `cd C:\Users\...` works;
`cdC:\Users\...` gives you "is not recognized as the name of a cmdlet".

It should look something like:

- **Windows** — `cd C:\Users\you\Downloads\geo4irr\frontend`
- **Mac** — `cd /Users/you/Downloads/geo4irr/frontend`

### Step 5. Start a small web server

Type this and press Enter:

- **Windows** — `py -m http.server 8080`
- **Mac** — `python3 -m http.server 8080`

The computer answers with something like:

```
Serving HTTP at 0.0.0.0 port 8080 ...
```

**That line is the computer talking back to you, not a command to type.** It
means it worked.

**Leave this window open.** It has to keep running. Closing it stops the app.

### Step 6. Open the app

In your browser, go to:

**http://localhost:8080**

You get a **sign-in screen** with two doors:

**Grower** — anyone can make one, no invitation, no limit.

**Administrator** — your account already exists:

```
yilma.wub@outlook.com
nfp123abc
```

The first sign-in asks you to choose a new password. That one was written down
in order to reach you, so it is not one to keep. It never asks again.

Sign in and the map flies in to Delaware with all 2,973 irrigated fields on it.

> **What this sign-in is, honestly.** With no server, accounts live in this
> browser on this computer. Passwords are stretched with PBKDF2-SHA256 and only
> the digest is kept, so the account list is not a list of passwords. But it is
> a lock on your own machine: anyone who opens the browser's developer tools can
> walk past it, because the data is already on their computer. It keeps the tool
> to you on a shared machine and gives each grower a separate set of fields.
> For accounts that hold against someone determined — and for the evening
> emails — do Part 2 and run the backend.
>
> To set it up for a different person, open `frontend/index.html` in a text
> editor and find `SEEDED_ADMIN`.

> **Why not just double-click `index.html`?**
> Because the browser then blocks the satellite requests. It has nothing to do
> with your internet — a file opened directly has no web address, and browsers
> refuse to let such a page call out to Sentinel. The map and the water balance
> would still work, but NDVI and NDRE would not. Step 4 gives the page a real
> address, which fixes it.

### Step 7. Use it

1. **Zoom in** until the field outlines appear, then **click your field.**
   (Or use **Draw a boundary** / **Upload a boundary** on the map.)
2. The map disappears and the left panel becomes your workflow. Work down it:
   - **Weather station** — pick the DEOS station whose weather drives this field
   - **Soils (SSURGO)** — look up what's under the field, pick the series
   - **Crop & irrigation** — crop, planting date, system, capacity. Press
     **Save this field**
3. Under **Outputs**, everything is now unlocked:
   - **Irrigation schedule** — Summary / Table / Graph
   - **Nitrogen plan**, **Satellite & indices**, **Sensor placement**,
     **Season score**

To go back to the map: open step 1 and press **Back to the map**.

### To stop

Click the terminal window and press **Ctrl+C**.

---

## Part 2 — Accounts, and the pyFAO56 engine (15 minutes)

Do this when you want other people to sign in, or when you want the numbers
computed by pyFAO56 rather than the browser's own copy.

### Step 1. Open a *second* terminal

Leave the Part 1 window running. Open a new terminal window.

### Step 2. Go to the backend folder

Same trick — type `cd `, drag the **`backend`** folder in, press Enter.

### Step 3. Install (once only)

- **Windows** — `py -m venv .venv`
- **Mac** — `python3 -m venv .venv`

Then activate it:

- **Mac** — `source .venv/bin/activate`
- **Windows** — `.venv\Scripts\activate`

Your prompt now starts with `(.venv)`. Then:

```
pip install -r requirements.txt
```

This takes a few minutes. It downloads pyFAO56 and everything else.

### Step 4. Check it works

```
python -m pytest tests/ -q
```

(Once the virtual environment is active, `python` is correct on both Windows
and Mac — the `py` / `python3` difference only applies before that.)

You want to see a line ending in **`passed`**. If it says something like
`287 passed, 22 skipped`, that is fine — skipped means an optional piece isn't
installed.

### Step 5. Create your administrator account

```
python -m app.create_admin
```

It asks for your email, your name, and a password. **Nothing appears as you
type the password — that is normal, keep typing and press Enter.**

Use at least 12 characters. A phrase you'll remember is better than a short
string of symbols.

> This password is hashed and thrown away. Nobody can read it back — not you,
> not me, not anyone with the database file. If you lose it, delete
> `geo4irr_auth.db` and run this command again.

### Step 6. Start the server

```
uvicorn app.main:app --port 8000
```

Leave this window open too. You now have two terminals running.

### Step 7. Point the app at the server

Open `frontend/index.html` in a text editor (Notepad, TextEdit, anything).

Find this line near the top — around line 480:

```javascript
let API_BASE = ""; // e.g. "https://geo4irr.udel.edu"
```

Change it to:

```javascript
let API_BASE = "http://localhost:8000";
```

Save the file. Reload **http://localhost:8080** in your browser.

You should now get a **sign-in screen**. Sign in with what you made in Step 5.

### Step 8. Sign in

Go to **http://localhost:8080**. You get a choice of two doors:

**Grower** — anyone can make one of these, no invitation, no limit. This is what
your customers use.

**Administrator** — invitation only, twenty seats. Yours already exists:

```
yilma.wub@outlook.com
nfp123abc
```

The first time you sign in with it, you will be asked to choose a new password
before anything else. That is deliberate: this one was written down in order to
reach you, so it is not one to keep. Everything else works normally afterwards
and the prompt does not come back.

To seed a different account, set `GEO4IRR_ADMIN_EMAIL` and
`GEO4IRR_ADMIN_PASSWORD` before the first start.

### Step 9. Evening notes

Everyone gets a short email at **6:00 pm** when one of their fields needs water
— what to apply, how many hours to run, and how far into the soil water the
crop already is. Nothing is sent on a day when nothing needs doing, because a
daily "nothing to do" is a mail people stop opening.

Each person can switch it off with the **Notes on / Notes off** button in the
top bar.

**To actually send mail you need an email server.** There is none built in.
Before starting the API:

```
set SMTP_HOST=smtp.office365.com
set SMTP_PORT=587
set SMTP_USER=yilma.wub@outlook.com
set SMTP_PASS=your-app-password
set SMTP_FROM=Geo4Irr <yilma.wub@outlook.com>
```

(On Mac use `export` instead of `set`.)

Outlook and Gmail both need an **app password** rather than your normal one —
your ordinary password will be refused. Search your account's security settings
for "app password".

Until that is set, the schedule still runs at six and every message is still
built, but they go to the log instead of an inbox. Check which state you are in:

**http://localhost:8000/api/notifications/status**

To fire a round immediately instead of waiting for six, signed in as the
administrator: **http://localhost:8000/api/notifications/test**

> One honest limitation: fields currently live in your browser, not on the
> server, so the server does not yet know what anyone's fields are. The
> schedule, the wording and the delivery all work and are tested — what is
> missing is field storage on the API. Until that is wired, the six o'clock
> round runs and finds nothing to say.

### Step 10. Invite other administrators

Once signed in, **Invite** appears in the blue bar at the top right.

1. Click it.
2. Type their email, choose Member or Administrator, click **Create invitation
   link**.
3. **Copy the link and send it to them yourself** — by email, text, however you
   like. The tool does not send it.

Each link works **once** and expires after **14 days**. Anyone holding it can
claim the seat, so send it the way you'd send a key.

You have **20 seats** including your own.

---

## Part 3 — Water you actually used, from Landsat (5 minutes)

This is the newest part of the tool and it answers a different question from
everything above.

The irrigation schedule tells you how much water the crop **would** use, from
the weather and a crop coefficient. This tells you how much water **did** leave
the field, by looking at how hot the ground was when a satellite went over. A
field transpiring freely runs cool, because evaporating water carries heat
away. One that has run out runs hot. The method is called METRIC and it reads
that difference pixel by pixel, 30 metres at a time.

You need the backend running and signed in, from Part 2. Then:

1. Pick a field on the map, pick a weather station, and set a planting date
   under **Crop & irrigation**.
2. In the left rail under Outputs, click **Water used (METRIC)**.
3. Set the date to total through. If the field has been harvested, put the
   harvest date in as well: the total then stops there, because a harvested
   field is not transpiring and counting those days would inflate the season.
4. Click **Run the energy balance**. It takes a minute or two. Each satellite
   pass means reading nine Landsat bands and solving the energy balance on
   every pixel.

You get four views:

- **Map** — one cell per Landsat pixel, coloured by how much water left that
  spot that day. This is where you see that the north end of the pivot is
  running short while the south is fine.
- **By week** — the average rate each week in inches per day, with the running
  season total. Weekly rather than daily on purpose: Landsat 8 and 9 together
  cross Delaware about every eight days, and claiming a single day's figure
  from that would be claiming a precision the method does not have.
- **Passes** — one row per usable satellite pass, with how much of the field
  was clear of cloud.
- **Calibration** — the two pixels the whole scene is calibrated on, and why
  each was chosen. Worth a look the first few times. METRIC anchors itself on
  one well-watered pixel and one dry one inside your own field, so if the cold
  one has landed on a farm pond or the hot one on a poultry house roof, the
  scene is wrong. The map circles both so you can check.

No account is needed for the imagery. Landsat Collection 2 Level-2 is free and
open, indexed by Element 84 and cropped by a public TiTiler, and none of those
asks you to register.

Two honest limits. First, this measures the past. With six to ten usable passes
in a Delaware season it is a verification and accounting tool, not a scheduling
one, so do not decide tonight's set from a picture taken last Tuesday. The
daily FAO-56 balance is what schedules, and it runs alongside rather than being
replaced. Second, nothing here has been checked against a Delaware
eddy-covariance or lysimeter record. The energy balance closes and the numbers
land where Delaware corn should land, which is not the same thing as being
validated. Treat the totals as good to maybe 10 or 15 percent until someone
hindcasts them against Carvel.

---

## Optional extras

### Real weather from DEOS

DEOS is not open — you need an account. Sign up at
<http://services.deos.udel.edu/ws/subscribe.php>, then before Step 6 above:

```
export DEOS_WS_USER=your-username
export DEOS_WS_PASS=your-password
```

(On Windows use `set` instead of `export`.)

Without it the tool still runs, on National Weather Service forecasts and
Delaware climatology. Check which you're on at
<http://localhost:8000/api/weather/status>.

### Better satellite imagery

Free Copernicus account at <https://dataspace.copernicus.eu>, then:

```
export CDSE_CLIENT_ID=your-id
export CDSE_CLIENT_SECRET=your-secret
```

Without it, satellite still works — it uses the open AWS mirror instead, which
needs no account at all.

### Keeping your passwords out of GitHub

If you are going to push this to GitHub, do not type a password into any file
in this project. There is a place for them that is already excluded:

```
cp .env.example .env
```

Open `.env`, fill in the values you have, save it. `.env` is listed in
`.gitignore` so it never gets committed; `.env.example` holds only the variable
names and is safe to publish. To check what the server can see, without
printing any of it:

```
python -m app.check_setup
```

Two things worth knowing. For Copernicus, create an OAuth client under Sentinel
Hub and use the client id and secret rather than your sign-in email and
password: a client can be revoked on its own and cannot be used to log into
your account if it leaks. For email, Outlook and Gmail both want an app
password here, not your normal one.

If a password does get committed by accident, changing it afterwards is not
enough on its own. The old value stays in the git history forever unless the
history itself is rewritten, and automated scrapers find committed secrets in
minutes rather than days. Rotate the password and rewrite the history, in that
order.

### The licence, since this is going on GitHub

Geo4Irr is GPL-3.0, and that is not a free choice. The METRIC engine in
`backend/app/engine/metric.py` is a port of pyMETRIC and pyTSEB, both GPL-3.0,
and a port is a derivative work rather than an independent implementation.
`ATTRIBUTION.md` sets out exactly what came from where, what this port changed
and why, and what the licence commits you to.

The practical version: anyone you give the software to can use, study, change
and redistribute it, and a changed version they pass on has to be GPL-3.0 with
its source available. Running it yourself on your own fields carries no
obligation at all, because the licence is about distribution rather than use. A
private repository is fine too. And the code being open does not open your
account: your sign-in, your field boundaries and your weather subscription are
configuration, and none of them is in the repository.

### Pushing it to github.com/Yilma-hub

Do these three checks first. They take a minute and each one has caught a real
mistake in somebody's first push.

```
cd geo4irr
git init
git add -A
git status --short | grep -iE "\.env$|\.db$|\.sqlite" || echo "clean"
```

That last line should print `clean`. If it prints a filename instead, that file
is about to be committed and must not be. `git rm --cached <file>` takes it back
out of the staging area without deleting it from your disk.

Then:

```
git commit -m "Geo4Irr: FAO-56 scheduling and METRIC water use for Delaware"
git branch -M main
git remote add origin https://github.com/Yilma-hub/geo4irr.git
git push -u origin main
```

Create the empty repository on GitHub first, without a README or a licence
file, since this project already has both and GitHub's would conflict on the
first push.

One thing to set deliberately rather than by default: GitHub will offer to pick
a licence for you and the answer is already decided. The repository description
field is a good place for a sentence saying so, something like "FAO-56
irrigation scheduling and METRIC satellite water use for Delaware. GPL-3.0,
ported from pyMETRIC."

---

## If something goes wrong

**"Cannot find command 'git'"** during `pip install` — fixed. Older copies of
this project pulled pyFAO56 from GitHub, which needs Git installed. It now
comes from PyPI instead. If you hit this, you have an older zip: either
re-download, or open `backend/requirements.txt` in a text editor, find the line
starting `pyfao56 @ git+` and replace the whole line with:

```
pyfao56>=1.4.0
```

Then run `pip install -r requirements.txt` again. No Git needed.

**"python3: command not found"**, or Windows offers you the Microsoft Store —
on Windows the command is `py`, not `python3`. If `py --version` also fails,
Python is not installed: get it from <https://www.python.org/downloads/> and
tick **"Add python.exe to PATH"** in the installer.

**"is not recognized as the name of a cmdlet"** — nearly always a missing
space after `cd`, or you pasted in a line that was output rather than a
command. `Serving HTTP at 0.0.0.0 port 8080 ...` is output.

**"Address already in use"** — something is already on that port. Use a
different number: `python3 -m http.server 8081`, then open
`http://localhost:8081`.

**The map is blank** — the map tiles come from the internet. Check you're
connected.

**No NDVI or NDRE** — you opened the file by double-clicking instead of doing
Part 1 Step 4. Go back and serve it properly.

**Sign-in screen won't appear** — `API_BASE` is still empty, or the backend
isn't running. Both terminals need to stay open.

**"That email and password do not match an account"** — this is the same
message whether the password is wrong or the account doesn't exist. That's on
purpose, so nobody can use the login screen to discover who has an account.

---

## Putting it on your Android phone

Geo4Irr installs to your home screen as an icon that opens full screen with no
browser bar. It is the same app with the same internet access — Sentinel
imagery, DEOS weather, drawing a new boundary, all of it works exactly as it
does on the laptop.

### What it needs first

**An https address.** Android will only offer to install a page served over
https, or over `http://localhost` on the same machine. So the phone cannot
install it from your laptop's `http://localhost:8080` — the phone is not that
machine.

Two ways to get one:

**Quickest — a tunnel.** With the server from Part 1 running, in a second
terminal:

```
py -m pip install cloudflared
cloudflared tunnel --url http://localhost:8080
```

It prints an `https://something.trycloudflare.com` address. Open that on your
phone. The address changes each time you restart it, which is fine for trying
it out and no good for daily use.

**Properly — host it.** The `frontend` folder is static files: `index.html`,
`manifest.json`, `sw.js`, the icons, and `data/`. Any static host will serve it
with https for free. That gives you a permanent address, and it is also the
point at which a `.com` domain of your own makes sense.

### Installing it

1. Open the https address in **Chrome on Android**.
2. Either tap **Install** in the top bar of the app, or use Chrome's menu
   (three dots) → **Add to Home screen**.
3. The Geo4Irr icon appears on your home screen. Open it from there.

### What changes on the phone

- The map and the workflow take turns instead of sharing the screen, with a
  **Back to the map** button at the top of the workflow.
- Drawing a boundary: tap each corner, then press **Finish**. Not double-tap —
  on a touch screen that is the zoom gesture.
- The daily water balance table scrolls sideways rather than crushing twelve
  columns into 380 pixels.
- Buttons and rows are bigger, for a fingertip rather than a mouse.

### About working offline

The app itself is cached, so it opens instantly and still opens when the signal
drops on a farm road.

**Live data is never cached.** Satellite tiles, weather, the soil survey and
anything from the API go to the network every time. This is deliberate: a
cached irrigation decision is a wrong irrigation decision, and it would fail
silently. If you are out of signal you will see the request fail rather than a
stale number presented as current.

### iPhone

The same thing works, through Safari: **Share → Add to Home Screen**. Safari
does not show an install button, so it has to be done from that menu.
