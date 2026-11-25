"""Invite-code-only beta auth & session store (docs/beta-plan.md §2, §5, §10).

Privacy & legal posture:
- Email is the ONLY PII stored, isolated to `beta_user` and mapped to an
  opaque lowercase UUID (`u-<32 hex chars>`). Lowercase hex guarantees that
  using `user_id` as the watermark/owner handle in `Ledger` and `CardLayer`
  never collides with `_TICKER_RE` and keeps email completely out of the
  registry, card faces, and telemetry.
- Invite-only: no open signup. Account creation requires a valid, unrevoked,
  unexpired invite code with remaining uses AND explicit `tos_accepted=True`
  (fail closed).
- No money, no billing, no paid tiers in beta.
- Stdlib-only password hashing (`hashlib.scrypt` + per-user 16-byte salt)
  and DB-backed SHA-256 session tokens.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")
TOS_VERSION = "beta-v1-day1"
SESSION_TTL_S = 14 * 86400.0
MIN_PASSWORD_LEN = 10

AUTH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS beta_invite (
    invite_id  TEXT PRIMARY KEY,
    code_hash  TEXT UNIQUE NOT NULL,
    wave       TEXT NOT NULL,
    max_uses   INTEGER NOT NULL DEFAULT 1,
    uses       INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL,
    revoked_at REAL
);

CREATE TABLE IF NOT EXISTS beta_user (
    user_id          TEXT PRIMARY KEY,
    email            TEXT UNIQUE NOT NULL,
    password_hash    TEXT NOT NULL,
    beta             INTEGER NOT NULL DEFAULT 1,
    tos_accepted_at  REAL NOT NULL,
    tos_version      TEXT NOT NULL,
    invite_code_hash TEXT NOT NULL,
    kill_rate_opt_in INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS beta_session (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES beta_user(user_id),
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_session_user ON beta_session(user_id);
"""


class AuthError(Exception):
    """Fail-closed auth error carrying an HTTP status hint."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = int(status_code)


@dataclass(frozen=True)
class BetaUser:
    user_id: str
    beta: bool
    tos_accepted_at: float
    tos_version: str
    kill_rate_opt_in: bool
    created_at: float


def _sha256_hex(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_invite(code: str) -> str:
    return "".join(str(code or "").strip().split()).upper()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    )
    return f"scrypt${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, dk_hex = stored.split("$", 2)
        if algo != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
    except ValueError:
        return False
    actual = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    )
    return hmac.compare_digest(actual, expected)


def _new_user_id() -> str:
    # Strictly lowercase hex so it never matches _TICKER_RE ([A-Z]{2,5}).
    return f"u-{uuid.uuid4().hex}"


class AuthStore:
    """SQLite store for beta invites, email-isolated accounts, and sessions."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(AUTH_SCHEMA_SQL)
        self.con.commit()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- invites -------------------------------------------------------------

    def create_invite(
        self,
        *,
        wave: str = "wave-1",
        max_uses: int = 1,
        code: str | None = None,
        expires_at: float | None = None,
        now: float | None = None,
    ) -> dict:
        """Create a beta invite code (stores SHA-256 hash only; returns raw
        code once to the founder/operator)."""
        if max_uses < 1:
            raise AuthError("max_uses must be >= 1", 400)
        ts = time.time() if now is None else float(now)
        raw_code = (
            _normalize_invite(code)
            if code is not None
            else f"SIFT-{secrets.token_hex(3).upper()}-{secrets.token_hex(3).upper()}"
        )
        if len(raw_code) < 6:
            raise AuthError("invite code must be at least 6 characters", 400)
        invite_id = f"inv-{uuid.uuid4().hex[:12]}"
        code_hash = _sha256_hex(raw_code)
        try:
            self.con.execute(
                "INSERT INTO beta_invite"
                " (invite_id, code_hash, wave, max_uses, uses, created_at, expires_at)"
                " VALUES (?,?,?,?,0,?,?)",
                (invite_id, code_hash, wave.strip() or "wave-1", int(max_uses), ts, expires_at),
            )
            self.con.commit()
        except sqlite3.IntegrityError as exc:
            raise AuthError("invite code already exists", 409) from exc
        return {
            "invite_id": invite_id,
            "code": raw_code,
            "wave": wave.strip() or "wave-1",
            "max_uses": int(max_uses),
            "created_at": ts,
            "expires_at": expires_at,
        }

    def revoke_invite(self, invite_id: str, *, now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        self.con.execute(
            "UPDATE beta_invite SET revoked_at=? WHERE invite_id=?",
            (ts, invite_id),
        )
        self.con.commit()

    # -- signup / login / sessions -------------------------------------------

    def signup(
        self,
        *,
        email: str,
        password: str,
        invite_code: str,
        tos_accepted: bool,
        now: float | None = None,
    ) -> tuple[BetaUser, str, str]:
        """Redeem an invite code and create a beta account.

        Returns `(user, session_token, wave)`. Fail-closed on missing ToS,
        invalid/exhausted/revoked/expired invite, bad email, or short password.
        """
        if tos_accepted is not True:
            raise AuthError(
                "Terms acceptance is mandatory to create a beta account.", 400
            )
        norm_email = str(email or "").strip().lower()
        if not EMAIL_RE.match(norm_email):
            raise AuthError("Please provide a valid email address.", 400)
        if not isinstance(password, str) or len(password) < MIN_PASSWORD_LEN:
            raise AuthError(
                f"Password must be at least {MIN_PASSWORD_LEN} characters.", 400
            )
        norm_code = _normalize_invite(invite_code)
        if not norm_code:
            raise AuthError("A valid beta invite code is required.", 403)

        ts = time.time() if now is None else float(now)
        code_hash = _sha256_hex(norm_code)
        inv = self.con.execute(
            "SELECT * FROM beta_invite WHERE code_hash=?", (code_hash,)
        ).fetchone()
        if inv is None:
            raise AuthError("Invalid beta invite code.", 403)
        if inv["revoked_at"] is not None:
            raise AuthError("This beta invite code has been revoked.", 403)
        if inv["expires_at"] is not None and ts > float(inv["expires_at"]):
            raise AuthError("This beta invite code has expired.", 403)
        if int(inv["uses"]) >= int(inv["max_uses"]):
            raise AuthError("This beta invite code has already been used.", 403)

        existing = self.con.execute(
            "SELECT user_id FROM beta_user WHERE email=?", (norm_email,)
        ).fetchone()
        if existing is not None:
            raise AuthError("An account with this email already exists.", 409)

        user_id = _new_user_id()
        pw_hash = _hash_password(password)
        cur = self.con.execute(
            "UPDATE beta_invite SET uses = uses + 1"
            " WHERE invite_id=? AND uses < max_uses AND revoked_at IS NULL",
            (inv["invite_id"],),
        )
        if cur.rowcount != 1:
            self.con.rollback()
            raise AuthError("This beta invite code has already been used.", 403)

        self.con.execute(
            "INSERT INTO beta_user"
            " (user_id, email, password_hash, beta, tos_accepted_at,"
            "  tos_version, invite_code_hash, kill_rate_opt_in, created_at)"
            " VALUES (?,?,?,1,?,?,?,0,?)",
            (user_id, norm_email, pw_hash, ts, TOS_VERSION, code_hash, ts),
        )
        token = self._create_session_row(user_id, now=ts)
        self.con.commit()
        return self.get_user(user_id), token, str(inv["wave"])

    def login(
        self,
        *,
        email: str,
        password: str,
        now: float | None = None,
    ) -> tuple[BetaUser, str]:
        norm_email = str(email or "").strip().lower()
        row = self.con.execute(
            "SELECT * FROM beta_user WHERE email=?", (norm_email,)
        ).fetchone()
        if row is None or not _verify_password(str(password or ""), row["password_hash"]):
            raise AuthError("Invalid email or password.", 401)
        ts = time.time() if now is None else float(now)
        token = self._create_session_row(str(row["user_id"]), now=ts)
        self.con.commit()
        return self._row_to_user(row), token

    def _create_session_row(self, user_id: str, *, now: float) -> str:
        raw_token = secrets.token_hex(32)
        self.con.execute(
            "INSERT INTO beta_session (token_hash, user_id, created_at, expires_at)"
            " VALUES (?,?,?,?)",
            (_sha256_hex(raw_token), user_id, now, now + SESSION_TTL_S),
        )
        return raw_token

    def resolve_session(
        self, token: str | None, *, now: float | None = None
    ) -> BetaUser | None:
        if not token:
            return None
        ts = time.time() if now is None else float(now)
        row = self.con.execute(
            "SELECT u.* FROM beta_session s"
            " JOIN beta_user u ON u.user_id = s.user_id"
            " WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at > ?",
            (_sha256_hex(token), ts),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    def logout(self, token: str | None, *, now: float | None = None) -> None:
        if not token:
            return
        ts = time.time() if now is None else float(now)
        self.con.execute(
            "UPDATE beta_session SET revoked_at=? WHERE token_hash=?",
            (ts, _sha256_hex(token)),
        )
        self.con.commit()

    def get_user(self, user_id: str) -> BetaUser:
        row = self.con.execute(
            "SELECT * FROM beta_user WHERE user_id=?", (user_id,)
        ).fetchone()
        if row is None:
            raise AuthError("Unknown user.", 404)
        return self._row_to_user(row)

    def set_kill_rate_opt_in(self, user_id: str, enabled: bool) -> BetaUser:
        self.con.execute(
            "UPDATE beta_user SET kill_rate_opt_in=? WHERE user_id=?",
            (1 if enabled else 0, user_id),
        )
        self.con.commit()
        return self.get_user(user_id)

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> BetaUser:
        return BetaUser(
            user_id=str(row["user_id"]),
            beta=bool(row["beta"]),
            tos_accepted_at=float(row["tos_accepted_at"]),
            tos_version=str(row["tos_version"]),
            kill_rate_opt_in=bool(row["kill_rate_opt_in"]),
            created_at=float(row["created_at"]),
        )


__all__ = [
    "MIN_PASSWORD_LEN",
    "SESSION_TTL_S",
    "TOS_VERSION",
    "AuthError",
    "AuthStore",
    "BetaUser",
]
