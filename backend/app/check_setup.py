"""
What this install can and cannot reach, without printing any of it.

    python -m app.check_setup

The commonest way a password escapes is somebody pasting a configuration dump
into a chat or a bug report to ask why something is not working. So this prints
whether each value is set, enough of it to tell a wrong value from a missing
one, and what breaks without it. It never prints a usable secret, which makes
its output safe to screenshot and safe to send to anyone.

Everything here is optional. The tool runs with none of it configured, on
National Weather Service forecasts and Delaware climatology for weather and the
open AWS mirrors for imagery. That is a legitimate way to run it. What matters
is knowing which mode you are in rather than discovering it from a number that
looked plausible.
"""

from __future__ import annotations

import sys

from . import settings


def main() -> int:
    st = settings.status()

    print("Geo4Irr configuration")
    print("=" * 72)
    print(f"Environment file: {st['env_file']}")
    if st["env_file_present"]:
        print("  found")
    else:
        print("  not there. Copy .env.example to .env and fill it in, or set")
        print("  real environment variables instead, which is what a server")
        print("  should do.")
    print()

    missing = []
    for c in st["credentials"]:
        mark = "set " if c["configured"] else "    "
        print(f"[{mark}] {c['name']}")
        print(f"        {c['purpose']}")
        print(f"        value: {c['value']}")
        if not c["configured"]:
            missing.append(c)
            print(f"        without it: {c['consequence_if_missing']}")
            if c["signup"]:
                print(f"        get one at: {c['signup']}")
        print()

    print("=" * 72)
    configured = len(st["credentials"]) - len(missing)
    print(f"{configured} of {len(st['credentials'])} values configured.")
    if missing:
        print()
        print("Nothing here is required. Each line above says what falls back")
        print("to what, so you can decide which are worth setting up for the")
        print("way you actually use this.")
    print()
    print("Landsat and METRIC need no credentials at all: Collection 2 Level-2")
    print("is open, and the scene index and the tile service are both free.")
    print()
    print(st["note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
