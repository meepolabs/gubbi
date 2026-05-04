"""Generate a bcrypt hash for the operator password.

Usage: python -m gubbi.oauth.crypto
"""

import getpass
import sys

import bcrypt


def main() -> None:
    password = getpass.getpass("Enter operator password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords don't match!")  # noqa: T201
        sys.exit(1)

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    print("\nAdd to your secrets manager:")  # noqa: T201
    print(f"JOURNAL_PASSWORD_HASH={hashed}")  # noqa: T201


if __name__ == "__main__":
    main()
