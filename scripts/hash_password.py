"""Generate bcrypt password hashes for the Quant Portfolio-Kaizen auth config.

Usage::

    python scripts/hash_password.py

The script prompts twice (no echo) and prints the bcrypt hash. Paste the
hash into the `password_hash` field of ``.streamlit/secrets.toml``.

Notes
-----
- Uses bcrypt directly (no streamlit-authenticator import) so it works
  outside the running Streamlit context.
- Default work factor is 12 rounds, which is sound for 2026 hardware.
- Plaintext passwords are never written to disk or printed.
"""

from __future__ import annotations

import getpass
import secrets
import sys

try:
    import bcrypt
except ImportError:  # pragma: no cover
    print(
        "bcrypt is not installed. Install via `pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    sys.exit(2)


MIN_LEN = 12
ROUNDS = 12


def _read_password_twice() -> str:
    while True:
        pw1 = getpass.getpass("Password: ")
        if len(pw1) < MIN_LEN:
            print(f"Password must be at least {MIN_LEN} characters.", file=sys.stderr)
            continue
        if pw1.lower() in {"password", "admin", "letmein", "qwerty"}:
            print("Refused: trivial password.", file=sys.stderr)
            continue
        pw2 = getpass.getpass("Confirm:  ")
        if pw1 != pw2:
            print("Passwords did not match. Try again.", file=sys.stderr)
            continue
        return pw1


def main() -> int:
    print("=== Quant Portfolio-Kaizen — password hash generator ===")
    print(f"Bcrypt cost factor: {ROUNDS}. Output is safe to paste into secrets.toml.")
    print()
    plaintext = _read_password_twice().encode("utf-8")
    digest = bcrypt.hashpw(plaintext, bcrypt.gensalt(rounds=ROUNDS)).decode("ascii")
    # Wipe the plaintext from memory ASAP (best-effort; CPython interns strings).
    plaintext = secrets.token_bytes(len(plaintext))
    print()
    print("password_hash =", repr(digest))
    print()
    print("Paste this line (without the leading variable name) under the user entry, e.g.:")
    print()
    print("[auth.users.<username>]")
    print('name          = "Display Name"')
    print('email         = "user@example.com"')
    print(f"password_hash = {digest!r}")
    print('role          = "viewer"  # admin | analyst | viewer')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
