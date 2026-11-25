"""Invite-code-only beta auth & session store (docs/beta-plan.md §2, §5, §10).

Privacy & security posture:
- Stored in its OWN dedicated SQLite file (`auth.db`), physically separated
  from `telemetry.db` and `app.db` so analytics exports/copies never contain
  emails or joinable identity tables.
- Email is the ONLY PII stored, mapped to an opaque lowercase UUID
  (`u-<32 hex chars>`). Lowercase hex guarantees watermark handles never
  collide with `_TICKER_RE`.
- Invite-only: no open signup. Requires a valid, unrevoked, unexpired invite
  code (>= 12 chars; generated codes carry 64 bits of entropy) AND explicit
  `tos_accepted=True` (fail closed).
- Password hashing (`hashlib.scrypt`) is exposed as pure functions so callers
  run CPU-intensive KDF off the DB thread, with a dummy scrypt check on
  unknown emails to prevent timing-based email enumeration.
- Account deletion permanently removes `beta_user` (and its `email`) and all
  sessions while leaving anonymized append-only telemetry untouched.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Self

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")
TOS_VERSION = "beta-v1-day1"
SESSION_TTL_S = 14 * 86400.0
MIN_PASSWORD_LEN = 10
MIN_INVITE_LEN = 12

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
    user_id    TEXT NOT NULL REFERENCES beta_user(user_id) ON DELETE CASCADE,
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


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Compute scrypt password hash (pure CPU; safe to run off the DB thread)."""
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    )
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify password against stored scrypt hash in constant time."""
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


# Precomputed dummy scrypt hash used on unknown-email login attempts so
# response timing matches existing-email verification (no timing oracle).
DUMMY_SCRYPT_HASH = hash_password(
    "dummy-timing-equalizer-password", salt=b"0123456789abcdef"
)


def _new_user_id() -> str:
    # Strictly lowercase hex so it never matches _TICKER_RE ([A-Z]{2,5}).
    return f"u-{uuid.uuid4().hex}"


class RateLimiter:
    """Thread-safe in-memory sliding-window rate limiter per (bucket, key)."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def check(
        self,
        bucket: str,
        key: str,
        *,
        limit: int,
        window_s: float,
        now: float | None = None,
    ) -> bool:
        """Record a hit and return True if allowed, False if rate-limited."""
        if not self.enabled:
            return True
        ts = time.monotonic() if now is None else float(now)
        cutoff = ts - float(window_s)
        k = (bucket, key)
        with self._lock:
            q = self._hits.get(k)
            if q is None:
                # Bound total tracked keys to avoid memory bloat under floods.
                if len(self._hits) > 4096:
                    self._hits.clear()
                q = deque()
                self._hits[k] = q
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= int(limit):
                return False
            q.append(ts)
            return True


class AuthStore:
    """Isolated SQLite store (`auth.db`) for invites, email accounts, and sessions."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path), timeout=10.0)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode = WAL")
        self.con.execute("PRAGMA busy_timeout = 10000")
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
            else f"SIFT-{secrets.token_hex(4).upper()}-{secrets.token_hex(4).upper()}"
        )
        if len(raw_code) < MIN_INVITE_LEN:
            raise AuthError(
                f"Invite code must be at least {MIN_INVITE_LEN} characters.", 400
            )
        invite_id = f"inv-{uuid.uuid4().hex[:12]}"
        code_hash = _sha256_hex(raw_code)
        try:
            self.con.execute(
                "INSERT INTO beta_invite"
                " (invite_id, code_hash, wave, max_uses, uses, created_at, expires_at)"
                " VALUES (?,?,?,?,0,?,?)",
                (
                    invite_id,
                    code_hash,
                    wave.strip() or "wave-1",
                    int(max_uses),
                    ts,
                    expires_at,
                ),
            )
            self.con.commit()
        except sqlite3.IntegrityError as exc:
            self.con.rollback()
            raise AuthError("Invite code already exists.", 409) from exc
        except Exception:
            self.con.rollback()
            raise
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
        try:
            self.con.execute(
                "UPDATE beta_invite SET revoked_at=? WHERE invite_id=?",
                (ts, invite_id),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    # -- signup / login / sessions / deletion --------------------------------

    def validate_signup_inputs(
        self,
        *,
        email: str,
        password: str,
        invite_code: str,
        tos_accepted: bool,
    ) -> tuple[str, str]:
        """Fast input validation prior to off-thread scrypt hashing."""
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
        return norm_email, norm_code

    def complete_signup(
        self,
        *,
        norm_email: str,
        password_hash: str,
        norm_code: str,
        now: float | None = None,
    ) -> tuple[BetaUser, str, str]:
        """Atomically consume invite, create user with pre-hashed password, and
        mint a session. Uniform 400 on duplicate email (no 409 email oracle)."""
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
            # Uniform 400 (no 409 email-existence oracle; invite not consumed).
            raise AuthError(
                "Could not create account with the provided details.", 400
            )

        user_id = _new_user_id()
        try:
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
                (
                    user_id,
                    norm_email,
                    password_hash,
                    ts,
                    TOS_VERSION,
                    code_hash,
                    ts,
                ),
            )
            token = self._create_session_row(user_id, now=ts)
            self.con.commit()
        except AuthError:
            self.con.rollback()
            raise
        except Exception:
            self.con.rollback()
            raise
        return self.get_user(user_id), token, str(inv["wave"])

    def signup(
        self,
        *,
        email: str,
        password: str,
        invite_code: str,
        tos_accepted: bool,
        now: float | None = None,
    ) -> tuple[BetaUser, str, str]:
        """Convenience sync signup (validates, hashes password, commits)."""
        norm_email, norm_code = self.validate_signup_inputs(
            email=email,
            password=password,
            invite_code=invite_code,
            tos_accepted=tos_accepted,
        )
        pw_hash = hash_password(password)
        return self.complete_signup(
            norm_email=norm_email,
            password_hash=pw_hash,
            norm_code=norm_code,
            now=now,
        )

    def lookup_for_login(self, email: str) -> tuple[BetaUser, str] | None:
        """Fast DB lookup returning `(user, password_hash)` or None (no scrypt)."""
        norm_email = str(email or "").strip().lower()
        row = self.con.execute(
            "SELECT * FROM beta_user WHERE email=?", (norm_email,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row), str(row["password_hash"])

    def create_session(self, user_id: str, *, now: float | None = None) -> str:
        ts = time.time() if now is None else float(now)
        try:
            token = self._create_session_row(user_id, now=ts)
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        return token

    def login(
        self,
        *,
        email: str,
        password: str,
        now: float | None = None,
    ) -> tuple[BetaUser, str]:
        """Constant-time sync login (always runs scrypt even for unknown email)."""
        found = self.lookup_for_login(email)
        stored_hash = found[1] if found is not None else DUMMY_SCRYPT_HASH
        ok = verify_password(str(password or ""), stored_hash)
        if found is None or not ok:
            raise AuthError("Invalid email or password.", 401)
        token = self.create_session(found[0].user_id, now=now)
        return found[0], token

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
        try:
            self.con.execute(
                "UPDATE beta_session SET revoked_at=? WHERE token_hash=?",
                (ts, _sha256_hex(token)),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def delete_account(self, user_id: str) -> bool:
        """Permanently erase a user's PII (email) and active sessions.

        Leaves anonymized, opaque-ID telemetry in `telemetry.db` untouched.
        Returns True if the user existed and was deleted.
        """
        try:
            self.con.execute(
                "DELETE FROM beta_session WHERE user_id=?", (user_id,)
            )
            cur = self.con.execute(
                "DELETE FROM beta_user WHERE user_id=?", (user_id,)
            )
            self.con.commit()
            return cur.rowcount == 1
        except Exception:
            self.con.rollback()
            raise

    def get_user(self, user_id: str) -> BetaUser:
        row = self.con.execute(
            "SELECT * FROM beta_user WHERE user_id=?", (user_id,)
        ).fetchone()
        if row is None:
            raise AuthError("Unknown user.", 404)
        return self._row_to_user(row)

    def set_kill_rate_opt_in(self, user_id: str, enabled: bool) -> BetaUser:
        try:
            self.con.execute(
                "UPDATE beta_user SET kill_rate_opt_in=? WHERE user_id=?",
                (1 if enabled else 0, user_id),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
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
    "DUMMY_SCRYPT_HASH",
    "EMAIL_RE",
    "MIN_INVITE_LEN",
    "MIN_PASSWORD_LEN",
    "SESSION_TTL_S",
    "TOS_VERSION",
    "AuthError",
    "AuthStore",
    "BetaUser",
    "RateLimiter",
    "hash_password",
    "verify_password",
]
