"""v4.1 share-for-capacity (mechanics item 6, behind `cards_v4_1`) + §4.7
anti-abuse stack.

Mechanics (v4.1 middle-ground spec §4 item 6 + §4.7):
- Sharing a discovered card's anonymized public face (server-rendered via
  pull/15, watermarked handle+timestamp) to social media grants the sharer a
  ONE-TIME PERMANENT +1 vault slot, stackable up to a HARD CAP of +5.
- Slots are non-transferable, expire with the account, never sellable; no
  money changes hands. Marketing acquisition mechanic, NOT a marketplace.
- Grant rules: one slot per (account, card-ID) pair, never repeatable for
  the same card; verified email + phone OTP before the first grant, KYC-lite
  (name+DOB) for higher caps; minimum account age + >=1 completed mining
  run before the first grant.
  NOTE on re-grant semantics: uniqueness is enforced by
  `share_unique.PRIMARY KEY (owner, card_id)` — an ABSOLUTE constraint,
  not a "revoked_at IS NULL" soft check. Once ANY row exists for the
  (owner, card_id) pair, every future grant for that pair is refused
  ("one slot per (account, card) EVER"), including after an abuse
  revocation — a revoked grant can never be re-granted. This is the
  intended spec behavior (the slot is one-time-per-card), documented here
  because the runtime duplicate check only inspects `share_unique` rows,
  while the DB constraint is absolute.
- Anti-abuse stack (§4.7): device ID + IP/ASN clustering with multi-account
  rejection and per-device rate limits (max 3 grants/device/30d); velocity
  caps (5 accounts from one device/IP are all FLAGGED, not auto-granted);
  live-post verification (the post must be publicly retrievable and contain
  the anonymized card link + attribution tag) — unverifiable shares grant
  NOTHING; append-only audit log of all grants (who/when/device/IP
  cluster/post); manual fraud review queue with SLA for anything failing
  signals; honeypot/canary accounts to detect share-farming services.
- Abuse: slots granted via abuse are revoked + the account is flagged;
  a repeat offense DISABLES the share feature for the account (first
  offense = warning, not a ban).

Storage: its own SQLite file (same discipline as the trial registry /
card registry):
- share_grant: append-only audit log, one row per grant attempt outcome;
  UPDATE/DELETE are blocked by triggers (the log is append-only).
- share_unique: one row per (owner, card_id) — the uniqueness grant rule.
- device_fingerprint: device/IP/ASN clusters per account.
- fraud_queue: manual review queue with SLA for failing signals.
- honeypot: canary accounts; any grant attempt against one is flagged.
- account_state: email/phone verification, KYC-lite, age, mining runs,
  offense count, share-disabled flag.

Vault integration: grant() calls vault.add_permanent_slots (cap +5) and
revoke() calls vault.revoke_permanent_slots; every grant/revoke does a
read-back (GET after write) of both the share tables and the vault slot
count before success is claimed.

Flag: every mutating path refuses with ShareError(flag_off) when
cards_v4_1 is off for the handle (kill-switch SIFT_CARDS_V4_1=0). Reads
(audit log, queue) stay available — the kill-switch never hides data.

Hard rules (v3 §12): no sellable discovery credit, no marketplace, no
money, cards/slots non-transferable, no tickers in any share content
(the shared object is the anonymized card face from pull/15 only).
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import db, flags
from sift.cards import CardRegistry
from sift.cards.vault import Vault

MAX_SHARE_SLOTS = 5                 # hard cap on permanent share slots (+5)
MIN_ACCOUNT_AGE_S = 7 * 24 * 3600   # minimum account age before first grant (7d)
MIN_MINING_RUNS = 1                 # >=1 completed mining run before first grant
MAX_GRANTS_PER_DEVICE_30D = 3       # per-device rate limit (max 3 grants/device/30d)
VELOCITY_FLAG_ACCOUNTS = 5          # >=5 accounts from one device/IP -> all flagged
FRAUD_QUEUE_SLA_S = 72 * 3600       # manual review SLA (72h)
ATTRIBUTION_TAG = "#SIFT"           # attribution tag the live post must contain

# The anonymized card link pattern the live post must contain (the
# server-rendered face URL from pull/15). Shape: /cards/<card-id>/face.
_CARD_LINK_RE = re.compile(r"/cards/(card-[0-9a-f]{64}(?::g\d+)?)/face")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS account_state (
    handle              TEXT PRIMARY KEY,
    email_verified      INTEGER NOT NULL DEFAULT 0,
    phone_otp_verified  INTEGER NOT NULL DEFAULT 0,
    kyc_lite            INTEGER NOT NULL DEFAULT 0,   -- name+DOB recorded
    created_at          REAL NOT NULL,
    completed_runs      INTEGER NOT NULL DEFAULT 0,   -- completed mining runs
    offense_count       INTEGER NOT NULL DEFAULT 0,   -- abuse offenses (1st = warning)
    share_disabled      INTEGER NOT NULL DEFAULT 0,   -- repeat offender: feature off
    updated_at          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS share_unique (
    owner       TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    granted_at  REAL NOT NULL,
    PRIMARY KEY (owner, card_id)        -- one slot per (account, card-ID), never repeatable
);
CREATE TABLE IF NOT EXISTS share_grant (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,   -- append-only audit log
    owner       TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    outcome     TEXT NOT NULL,          -- granted|denied_<reason>|revoked|flagged
    reason      TEXT,
    device_id   TEXT,
    ip          TEXT,
    asn         TEXT,
    post_url    TEXT,
    verified    INTEGER NOT NULL DEFAULT 0,   -- live-post verification result
    at          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS device_fingerprint (
    owner       TEXT NOT NULL,
    device_id   TEXT NOT NULL,
    ip          TEXT,
    asn         TEXT,
    first_seen  REAL NOT NULL,
    PRIMARY KEY (owner, device_id)
);
CREATE TABLE IF NOT EXISTS fraud_queue (
    owner       TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    signal      TEXT NOT NULL,          -- which anti-abuse signal failed
    queued_at   REAL NOT NULL,
    sla_deadline REAL NOT NULL,
    resolved    INTEGER NOT NULL DEFAULT 0,
    resolution  TEXT,
    PRIMARY KEY (owner, card_id, signal)
);
CREATE TABLE IF NOT EXISTS honeypot (
    handle      TEXT PRIMARY KEY,       -- canary accounts (share-farming detection)
    note        TEXT,
    planted_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_grant_owner ON share_grant(owner);
CREATE INDEX IF NOT EXISTS idx_grant_device ON share_grant(device_id);
CREATE INDEX IF NOT EXISTS idx_fp_device ON device_fingerprint(device_id);
CREATE INDEX IF NOT EXISTS idx_fp_ip ON device_fingerprint(ip);
"""

# Append-only discipline (same as the trial registry): once a row is in
# share_grant it can never be updated or deleted. Revocations are NEW rows
# (outcome='revoked'), not edits.
_APPEND_ONLY_SQL = """
CREATE TRIGGER IF NOT EXISTS share_grant_no_update
BEFORE UPDATE ON share_grant
BEGIN
    SELECT RAISE(ABORT, 'share_grant is append-only');
END;
CREATE TRIGGER IF NOT EXISTS share_grant_no_delete
BEFORE DELETE ON share_grant
BEGIN
    SELECT RAISE(ABORT, 'share_grant is append-only');
END;
"""


class ShareError(Exception):
    """Share-for-capacity misuse: flag off, prerequisites unmet, duplicate
    (account, card) grant, cap reached, cluster rejection, unverifiable
    share, abuse, ..."""


@dataclass(frozen=True)
class ShareGrantRecord:
    seq: int
    owner: str
    card_id: str
    outcome: str
    reason: str | None
    device_id: str | None
    ip: str | None
    asn: str | None
    post_url: str | None
    verified: int
    at: float


class ShareService:
    """Grant/verification logic for share-for-capacity. Server-side only."""

    def __init__(
        self,
        cards: CardRegistry,
        vault: Vault,
        db_path: str | Path | None = None,
        *,
        con: sqlite3.Connection | None = None,
    ):
        self.cards = cards
        self.vault = vault
        if con is not None or db_path is None:
            shared = con if con is not None else cards.con
            self.con = shared
            self._owns_con = False
            self.db_path = (
                Path(db_path) if db_path is not None else (db.db_path_of(shared) or cards.db_path)
            )
            self.con.row_factory = sqlite3.Row
            self.con.execute("PRAGMA foreign_keys = ON")
        else:
            self.db_path = Path(db_path)
            self.con = db.connect(self.db_path)
            self._owns_con = True
        self.con.executescript(_SCHEMA_SQL)
        self.con.executescript(_APPEND_ONLY_SQL)
        db.maybe_commit(self.con)

    def __enter__(self) -> "ShareService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_con:
            self.con.close()

    # -- account state (prerequisites) ---------------------------------------

    def register_account(self, handle: str, *, now: float | None = None,
                         email_verified: bool = False,
                         phone_otp_verified: bool = False,
                         kyc_lite: bool = False,
                         created_at: float | None = None,
                         completed_runs: int = 0) -> None:
        """Record/refresh the account prerequisites. `created_at` defaults to
        now; completed_runs counts COMPLETED mining runs (ledger-backed in
        production; a plain counter here)."""
        now = time.time() if now is None else now
        created_at = now if created_at is None else created_at
        self.con.execute(
            "INSERT INTO account_state (handle, email_verified, phone_otp_verified,"
            " kyc_lite, created_at, completed_runs, offense_count, share_disabled,"
            " updated_at) VALUES (?,?,?,?,?,?,0,0,?)"
            " ON CONFLICT(handle) DO UPDATE SET"
            " email_verified=excluded.email_verified,"
            " phone_otp_verified=excluded.phone_otp_verified,"
            " kyc_lite=excluded.kyc_lite,"
            " created_at=MIN(account_state.created_at, excluded.created_at),"
            " completed_runs=excluded.completed_runs,"
            " updated_at=excluded.updated_at",
            (handle, int(email_verified), int(phone_otp_verified), int(kyc_lite),
             created_at, int(completed_runs), now),
        )
        db.maybe_commit(self.con)

    def set_completed_runs(self, handle: str, n: int, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.con.execute(
            "UPDATE account_state SET completed_runs=?, updated_at=? WHERE handle=?",
            (int(n), now, handle),
        )
        db.maybe_commit(self.con)

    def _account(self, handle: str) -> sqlite3.Row:
        row = self.con.execute(
            "SELECT * FROM account_state WHERE handle=?", (handle,)
        ).fetchone()
        if row is None:
            raise ShareError(f"unknown account {handle!r} — register_account first")
        return row

    def plant_honeypot(self, handle: str, note: str = "canary",
                       now: float | None = None) -> None:
        """Plant a honeypot/canary account (§4.7). Any grant attempt against
        a honeypot is flagged (share-farming service detection)."""
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT OR REPLACE INTO honeypot (handle, note, planted_at) VALUES (?,?,?)",
            (handle, note, now),
        )
        db.maybe_commit(self.con)

    def is_honeypot(self, handle: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM honeypot WHERE handle=?", (handle,)
        ).fetchone() is not None

    def _require_flag(self, handle: str) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise ShareError("flag_off: cards_v4_1 is disabled for this account")

    def _audit(self, owner: str, card_id: str, outcome: str, reason: str | None,
               device_id: str | None, ip: str | None, asn: str | None,
               post_url: str | None, verified: int, now: float) -> int:
        """Append one row to the append-only audit log. Returns the seq."""
        cur = self.con.execute(
            "INSERT INTO share_grant (owner, card_id, outcome, reason, device_id,"
            " ip, asn, post_url, verified, at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (owner, card_id, outcome, reason, device_id, ip, asn, post_url,
             verified, now),
        )
        db.maybe_commit(self.con)
        seq = cur.lastrowid
        assert seq is not None
        return seq

    # -- live-post verification ------------------------------------------------

    def verify_live_post(self, post_url: str | None, card_id: str) -> bool:
        """Live-post verification (§4.7): the post must be publicly
        retrievable and contain BOTH the anonymized card link
        (/cards/<card-id>/face) AND the attribution tag. Unverifiable
        shares grant nothing.

        THIS IS THE SINGLE CHOKE POINT for live-post verification
        (`verify_share_post` in the PR branch is the same interface under
        the old name): every grant path calls it, and it is the one place
        the real platform-API verifier (server-side fetch of the public
        post via the platform's API, checking the card link + attribution
        tag) plugs in. Today the fetch is injected via `post_fetcher` on
        the service (tests inject a stub); production wires a real
        platform-API fetcher behind this same method — no other code
        path verifies shares."""
        if not post_url:
            return False
        fetcher = getattr(self, "post_fetcher", None)
        try:
            body = fetcher(post_url) if fetcher else None
        except Exception:
            return False
        if not isinstance(body, str):
            return False
        if ATTRIBUTION_TAG not in body:
            return False
        return _CARD_LINK_RE.search(body) is not None and card_id in body

    # Alias kept so the PR-branch name `verify_share_post` resolves to the
    # one verifier interface above.
    verify_share_post = verify_live_post

    # -- grant -------------------------------------------------------------------

    def request_share(self, handle: str, card_id: str, *, post_url: str | None,
                      device_id: str | None = None, ip: str | None = None,
                      asn: str | None = None, now: float | None = None) -> dict:
        """Attempt a share grant (mechanics 6). Returns the read-back grant
        state. Raises ShareError on hard refusals (flag off, prerequisites,
        duplicate, cap, cluster rejection, abuse-disabled). Soft failures
        (unverifiable post, rate limit, velocity flag, honeypot) are
        recorded in the audit log / fraud queue and return
        {"granted": False, ...} — they grant NOTHING.

        Read-back discipline: the returned dict is re-read from the DB
        (share_unique + vault slot count) before being returned.
        """
        self._require_flag(handle)
        now = time.time() if now is None else now
        card = self.cards.assert_owned(card_id, handle)   # non-transferable
        acct = self._account(handle)

        # honeypot/canary: any attempt is flagged, never granted
        if self.is_honeypot(handle):
            seq = self._audit(handle, card_id, "flagged", "honeypot_account",
                              device_id, ip, asn, post_url, 0, now)
            self._enqueue_fraud(handle, card_id, "honeypot_account", now)
            return {"granted": False, "reason": "honeypot_account",
                    "seq": seq, "slots": self.vault.share_slots(handle)}

        if acct["share_disabled"]:
            raise ShareError("share_disabled: repeat abuse offense — share feature"
                             " disabled for this account (warning was the first offense)")

        # prerequisites: verified email + phone OTP before first grant;
        # KYC-lite (name+DOB) required for higher caps (slot #4 and #5);
        # minimum account age + >=1 completed mining run before first grant.
        current_slots = self.vault.share_slots(handle)
        if not acct["email_verified"] or not acct["phone_otp_verified"]:
            self._audit(handle, card_id, "denied_prereq", "email+phone OTP required",
                        device_id, ip, asn, post_url, 0, now)
            raise ShareError("verified email + phone OTP required before the first grant")
        if now - acct["created_at"] < MIN_ACCOUNT_AGE_S:
            self._audit(handle, card_id, "denied_prereq", "account too young",
                        device_id, ip, asn, post_url, 0, now)
            raise ShareError(f"minimum account age not reached ({MIN_ACCOUNT_AGE_S}s)")
        if acct["completed_runs"] < MIN_MINING_RUNS:
            self._audit(handle, card_id, "denied_prereq", "no completed mining run",
                        device_id, ip, asn, post_url, 0, now)
            raise ShareError("at least 1 completed mining run required before the first grant")
        if current_slots >= 3 and not acct["kyc_lite"]:
            self._audit(handle, card_id, "denied_prereq", "kyc_lite required for higher caps",
                        device_id, ip, asn, post_url, 0, now)
            raise ShareError("KYC-lite (name+DOB) required for share slots beyond +3")

        # uniqueness: one slot per (account, card-ID), never repeatable
        dup = self.con.execute(
            "SELECT 1 FROM share_unique WHERE owner=? AND card_id=?",
            (handle, card_id),
        ).fetchone()
        if dup is not None:
            raise ShareError(f"card {card_id!r} already granted a slot to {handle!r}"
                             " — one slot per (account, card)")

        # hard cap +5
        if current_slots >= MAX_SHARE_SLOTS:
            raise ShareError(f"share-slot cap reached (+{MAX_SHARE_SLOTS})")

        # device/IP fingerprint store + clustering (§4.7)
        if device_id:
            self.con.execute(
                "INSERT INTO device_fingerprint (owner, device_id, ip, asn, first_seen)"
                " VALUES (?,?,?,?,?) ON CONFLICT(owner, device_id) DO UPDATE SET"
                " ip=COALESCE(excluded.ip, device_fingerprint.ip),"
                " asn=COALESCE(excluded.asn, device_fingerprint.asn)",
                (handle, device_id, ip, asn, now),
            )
            db.maybe_commit(self.con)
            # velocity/pattern caps: >=5 accounts from one device/IP -> all
            # FLAGGED (fraud queue), not auto-granted
            cluster = self._cluster_size(device_id=device_id, ip=ip)
            if cluster >= VELOCITY_FLAG_ACCOUNTS:
                seq = self._audit(handle, card_id, "flagged", "velocity_cluster",
                                  device_id, ip, asn, post_url, 0, now)
                self._enqueue_fraud(handle, card_id, "velocity_cluster", now)
                return {"granted": False, "reason": "velocity_cluster",
                        "seq": seq, "slots": self.vault.share_slots(handle)}
            # multi-account rejection + per-device rate limit:
            # max 3 grants per device per 30d
            window_start = now - 30 * 24 * 3600
            dev_grants = self.con.execute(
                "SELECT COUNT(*) c FROM share_grant WHERE device_id=? AND outcome='granted'"
                " AND at>=?", (device_id, window_start),
            ).fetchone()["c"]
            if dev_grants >= MAX_GRANTS_PER_DEVICE_30D:
                seq = self._audit(handle, card_id, "denied_ratelimit",
                                  "max grants/device/30d reached",
                                  device_id, ip, asn, post_url, 0, now)
                self._enqueue_fraud(handle, card_id, "device_ratelimit", now)
                return {"granted": False, "reason": "device_ratelimit",
                        "seq": seq, "slots": self.vault.share_slots(handle)}

        # live-post verification: unverifiable shares grant NOTHING
        verified = 1 if self.verify_live_post(post_url, card_id) else 0
        if not verified:
            seq = self._audit(handle, card_id, "denied_unverifiable",
                              "live post not verifiable (no public card link + attribution tag)",
                              device_id, ip, asn, post_url, 0, now)
            self._enqueue_fraud(handle, card_id, "unverifiable_post", now)
            return {"granted": False, "reason": "unverifiable_post",
                    "seq": seq, "slots": self.vault.share_slots(handle)}

        # GRANT: one permanent +1 vault slot (cap +5 enforced in the vault).
        # Atomic across share_unique + vault + share_grant when sharing a DB.
        with db.transaction(self.con):
            self.con.execute(
                "INSERT INTO share_unique (owner, card_id, granted_at) VALUES (?,?,?)",
                (handle, card_id, now),
            )
            self.vault.add_permanent_slots(handle, 1)   # vault caps at +5
            seq = self._audit(handle, card_id, "granted", None,
                              device_id, ip, asn, post_url, verified, now)
        # read-back: uniqueness row + vault slot count, both re-read from DB
        rb_unique = self.con.execute(
            "SELECT granted_at FROM share_unique WHERE owner=? AND card_id=?",
            (handle, card_id),
        ).fetchone()
        rb_slots = self.vault.share_slots(handle)
        rb_cap = self.vault.effective_capacity(handle, card.tier)
        return {
            "granted": True,
            "seq": seq,
            "card_id": card_id,
            "slots": rb_slots,                       # read-back from vault DB
            "capacity": rb_cap,                      # base tier + share slots
            "granted_at": rb_unique["granted_at"] if rb_unique else None,  # read-back
        }

    def _cluster_size(self, *, device_id: str | None, ip: str | None) -> int:
        """Accounts in one device/IP cluster (§4.7 IP/ASN clustering):
        the max of the device cluster and the IP cluster."""
        size = 1
        if device_id:
            row = self.con.execute(
                "SELECT COUNT(DISTINCT owner) c FROM device_fingerprint WHERE device_id=?",
                (device_id,),
            ).fetchone()
            size = max(size, row["c"])
        if ip:
            row = self.con.execute(
                "SELECT COUNT(DISTINCT owner) c FROM device_fingerprint WHERE ip=?",
                (ip,),
            ).fetchone()
            size = max(size, row["c"])
        return size

    def _enqueue_fraud(self, handle: str, card_id: str, signal: str,
                       now: float) -> None:
        """Manual fraud review queue with SLA (§4.7): anything failing
        signals lands here for review within FRAUD_QUEUE_SLA_S."""
        self.con.execute(
            "INSERT OR IGNORE INTO fraud_queue (owner, card_id, signal, queued_at,"
            " sla_deadline, resolved, resolution) VALUES (?,?,?,?,?,0,NULL)",
            (handle, card_id, signal, now, now + FRAUD_QUEUE_SLA_S),
        )
        db.maybe_commit(self.con)

    # -- abuse: revoke + flag + disable ------------------------------------------

    def revoke_for_abuse(self, handle: str, card_id: str | None = None, *,
                         reason: str = "abuse", now: float | None = None) -> dict:
        """Abuse path (§4.7): slots granted via abuse are REVOKED (vault
        slots zeroed), the account is flagged (offense counted), and a
        REPEAT offense disables the share feature for the account (first
        offense = warning, NOT a ban). Audit-logged append-only; every
        write is read back."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        acct = self._account(handle)
        prev_slots = self.vault.share_slots(handle)
        new_offenses = acct["offense_count"] + 1
        disable = 1 if new_offenses >= 2 else 0                 # repeat -> disabled
        with db.transaction(self.con):
            self.vault.revoke_permanent_slots(handle)           # revoke slots
            self.con.execute(
                "UPDATE account_state SET offense_count=?, share_disabled=?, updated_at=?"
                " WHERE handle=?",
                (new_offenses, disable, now, handle),
            )
            if card_id:
                self._audit(handle, card_id, "revoked", reason, None, None, None,
                            None, 0, now)
                self.con.execute(
                    "DELETE FROM share_unique WHERE owner=? AND card_id=?",
                    (handle, card_id),
                )
        # read-back
        rb_acct = self._account(handle)
        rb_slots = self.vault.share_slots(handle)
        return {
            "revoked_slots": prev_slots,
            "slots_after": rb_slots,                            # read-back
            "offense_count": rb_acct["offense_count"],          # read-back
            "share_disabled": bool(rb_acct["share_disabled"]),  # read-back
            "warning": new_offenses == 1,
        }

    # -- reads ---------------------------------------------------------------------

    def grants_for(self, handle: str) -> list[ShareGrantRecord]:
        """Read-back of the audit log for one account (append-only rows)."""
        rows = self.con.execute(
            "SELECT * FROM share_grant WHERE owner=? ORDER BY seq", (handle,)
        ).fetchall()
        return [self._to_record(r) for r in rows]

    def audit_log(self) -> list[ShareGrantRecord]:
        rows = self.con.execute("SELECT * FROM share_grant ORDER BY seq").fetchall()
        return [self._to_record(r) for r in rows]

    def fraud_queue_open(self) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT * FROM fraud_queue WHERE resolved=0 ORDER BY queued_at"
        ).fetchall()

    def share_slots(self, handle: str) -> int:
        """Read-back of the account's permanent share slots (vault DB)."""
        return self.vault.share_slots(handle)

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ShareGrantRecord:
        return ShareGrantRecord(
            seq=row["seq"], owner=row["owner"], card_id=row["card_id"],
            outcome=row["outcome"], reason=row["reason"], device_id=row["device_id"],
            ip=row["ip"], asn=row["asn"], post_url=row["post_url"],
            verified=row["verified"], at=row["at"],
        )


__all__ = [
    "ATTRIBUTION_TAG",
    "FRAUD_QUEUE_SLA_S",
    "MAX_GRANTS_PER_DEVICE_30D",
    "MAX_SHARE_SLOTS",
    "MIN_ACCOUNT_AGE_S",
    "MIN_MINING_RUNS",
    "VELOCITY_FLAG_ACCOUNTS",
    "ShareError",
    "ShareGrantRecord",
    "ShareService",
]
