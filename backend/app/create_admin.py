#!/usr/bin/env python3
"""Create the first Geo4Irr administrator.

    python -m app.create_admin

Prompts for the password rather than taking it as an argument, because an
argument lands in your shell history and in the process list where anyone on
the machine can read it. Nothing is echoed while you type.

There is also a browser route: start the API, open the app, and the first-run
screen does the same thing. Use whichever suits — both refuse once an account
exists, and neither ships a default password.
"""

from __future__ import annotations

import getpass
import sys

from . import auth


def main() -> int:
    auth.init_db()

    if not auth.needs_setup():
        print("This installation already has accounts.")
        print(f"  {auth.count_active()} active, {auth.seats_left()} seats free "
              f"of {auth.seats()}.")
        print("\nTo add someone, sign in as an administrator and send them an "
              "invitation — there is no way to mint an account from here once "
              "the first one exists.")
        return 1

    print("Geo4Irr — first administrator\n")
    email = input("Email: ").strip()
    name = input("Name (optional): ").strip()

    while True:
        password = getpass.getpass("Password (12+ characters): ")
        problem = auth.password_problem(password)
        if problem:
            print(f"  {problem}\n")
            continue
        if password != getpass.getpass("Confirm: "):
            print("  Those did not match.\n")
            continue
        break

    try:
        user = auth.create_user(email, password, name=name, role="admin",
                                enforce_seats=False)
    except auth.AuthError as exc:
        print(f"\n{exc}")
        return 1

    print(f"\nAdministrator created: {user.email}")
    print(f"Seats: {auth.seats_left()} free of {auth.seats()}.")
    print("\nThe password was hashed with scrypt and discarded. It is not "
          "stored anywhere, so nobody — including you — can read it back. "
          "If it is lost, delete the database file and run this again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
