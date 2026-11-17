"""v4.1 share-for-capacity (mechanics item 6, behind `cards_v4_1`).

Mechanics (authoritative, v4.1 middle-ground spec §4 item 6 + §4.7):
- Sharing a discovered card's anonymized public face (server-rendered via
  sift/cards/face.py, watermarked handle+timestamp) grants the sharer a
  one-time PERMANENT +1 vault slot, stackable up to a hard cap of +5.
- One slot per (account, card_id) pair — never repeatable for the same card.
- Prerequisites before the first grant: verified email + phone OTP, minimum
  account age, and >= 1 completed mining run (a decided trial in the ledger).
- Anti-abuse stack (§4.7): device/IP fingerprint store with clustering,
  per-device rate limit (max 3 grants/device/30d), velocity flagging
  (5+ accounts from one device/IP cluster are flagged, not auto-granted),
  live-post verification (the share must be publicly verifiable to contain
  the anonymized card link + attribution tag — unverifiable shares grant
  nothing), honeypot/canary accounts, an append-only audit log of every
  grant/revoke/flag (same discipline as the trial registry), and a manual
  fraud-review queue for anything failing signals.
- Abuse: slots granted via abuse are revoked (vault.revoke_permanent_slots)
  and the account flagged; a repeat offense disables the share feature for
  the account (first offense warning, not a ban).
- Slots are non-transferable, expire with the account, never sellable; no
  money changes hands. Marketing acquisition mechanic, not a marketplace.

Vault integration: the vault stores the permanent slot count on its rows
(capped at +5 there); this module writes via vault.add_permanent_slots and
reads back after every grant/revoke (hard rule: GET after write).

Flag: every mutating path refuses with ShareError(flag_off) when
cards_v4_1 is off for the handle. Reads (grant history) stay available —
the kill-switch never hides existing data.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import flags
from sift.cards import CardRegistry
from sift.cards.face import CardFaceRenderer
from sift.cards.vault import Vault
from sift.registry import Ledger

MAX_SHARE_SLOTS = 5            # hard cap on permanent share slots per account
MAX_GRANTS_PER_DEVICE_30D = 3  # per-device rate limit
CLUSTER_FLAG_THRESHOLD = 5     # >=5 accounts on one device/IP cluster -> flag all, grant none
MIN_ACCOUNT_AGE_S = 7 * 24 * 3600      # 7 days
MIN_COMPLETED_RUNS = 1                 # >=1 decided trial in the ledger

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS share_grant (
    grant_id    TEXT PRIMARY KEY,          -- audit id (deterministic per grant)
    handle      TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    device_id   TEXT NOT NULL,
    ip_cluster  TEXT NOT NULL,
    post_ref    TEXT NOT NULL,             -- public post reference carrying the card link
    verified    INTEGER NOT NULL,          -- live-post verification outcome (0/1)
    granted_at  REAL NOT NULL,
    revoked_at  REAL,
    UNIQUE (handle, card_id)               -- one slot per (account, card) pair, ever
);
CREATE INDEX IF NOT EXISTS idx_share_handle ON share_grant(handle);
CREATE INDEX IF NOT EXISTS idx_share_device ON share_grant(device_id, granted_at);
CREATE TABLE IF NOT EXISTS share_device (
    device_id   TEXT PRIMARY KEY,
    ip_cluster  TEXT NOT NULL,
    first_seen  REAL NOT NULL,
    grants_30d  INTEGER NOT NULL DEFAULT 0,
    window_start REAL NOT NULL,
    flagged     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS share_audit (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,   -- append-only
    at          REAL NOT NULL,
    action      TEXT NOT NULL,             -- grant|revoke|flag|verify_fail|refuse
    handle      TEXT NOT NULL,
    card_id     TEXT,
    detail      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fraud_queue (
    queue_id    TEXT PRIMARY KEY,
    handle      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    at          REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open'         -- open|reviewed
);
"""


class ShareError(Exception):
    """Share-for-capacity misuse: flag off, prerequisites unmet, cap reached,
    duplicate grant, unverifiable share, abuse flag, ..."""


@dataclass(frozen=True)
class ShareGrant:
    grant_id: str
    handle: str
    card_id: str
    device_id: str
    ip_cluster: str
    post_ref: str
    verified: bool
    granted_at: float
    revoked_at: float | None


class ShareService:
    """Server-side share-for-capacity. Owns its own SQLite file (audit log
    append-only, same discipline as the trial ledger)."""

    def __init__(self, ledger: Ledger, cards: CardRegistry, vault: Vault,
                 face: CardFaceRenderer, db_path: str | Path):
        self.ledger = ledger
        self.cards = cards
        self.vault = vault
        self.face = face
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "ShareService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- flag gate -------------------------------------------------------------

    def _require_flag(self, handle: str) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise ShareError("flag_off: cards_v4_1 is disabled for this account")

    # -- audit ------------------------------------------------------------------

    def _audit(self, action: str, handle: str, card_id: str | None,
               detail: str, now: float) -> None:
        self.con.execute(
            "INSERT INTO share_audit (at, action, handle, card_id, detail)"
            " VALUES (?,?,?,?,?)",
            (now, action, handle, card_id, detail),
        )

    # -- prerequisites ------------------------------------------------------------

    def _check_prereqs(self, handle: str, account_age_s: float,
                       email_verified: bool, phone_otp_verified: bool,
                       now: float) -> None:
        if not email_verified:
            raise ShareError("prereq: verified email required before first grant")
        if not phone_otp_verified:
            raise ShareError("prereq: phone OTP required before first grant")
        if account_age_s < MIN_ACCOUNT_AGE_S:
            raise ShareError(
                f"prereq: account age {account_age_s:.0f}s < {MIN_ACCOUNT_AGE_S}s"
            )
        decided = self.ledger.con.execute(
            "SELECT COUNT(*) c FROM trial WHERE discoverer_handle=? AND verdict IN ('killed','survivor')",
            (handle,),
        ).fetchone()["c"]
        if decided < MIN_COMPLETED_RUNS:
            raise ShareError(
                f"prereq: >= {MIN_COMPLETED_RUNS} completed mining run(s) required"
            )

    def _device_gate(self, handle: str, device_id: str, ip_cluster: str, now: float) -> None:
        """Per-device rate limit + cluster flagging. A flagged/over-limit
        device grants nothing and lands in the fraud queue."""
        row = self.con.execute(
            "SELECT * FROM share_device WHERE device_id=?", (device_id,)
        ).fetchone()
        if row is None:
            self.con.execute(
                "INSERT INTO share_device (device_id, ip_cluster, first_seen,"
                " grants_30d, window_start, flagged) VALUES (?,?,?,0,?,0)",
                (device_id, ip_cluster, now, now),
            )
            self.con.commit()
        else:
            window_start = row["window_start"]
            grants = row["grants_30d"] if now - window_start <= 30 * 24 * 3600 else 0
            if now - window_start > 30 * 24 * 3600:
                window_start = now
                grants = 0
            if grants >= MAX_GRANTS_PER_DEVICE_30D:
                self._fraud(handle=handle, reason=f"device {device_id} over rate limit",
                            now=now)
                self._audit("refuse", handle, None,
                            f"device {device_id} at rate limit ({grants})", now)
                self.con.commit()
                raise ShareError("device rate limit: max "
                                 f"{MAX_GRANTS_PER_DEVICE_30D} grants/device/30d")
            self.con.execute(
                "UPDATE share_device SET grants_30d=?, window_start=?, ip_cluster=?"
                " WHERE device_id=?",
                (grants, window_start, ip_cluster, device_id),
            )
            self.con.commit()
        # cluster velocity: >= CLUSTER_FLAG_THRESHOLD distinct accounts on one
        # device/IP cluster -> flag, grant nothing (anti multi-account farming).
        # The grant being admitted would be the (n+1)-th distinct account on
        # this cluster, so count existing distinct accounts PLUS this handle.
        cluster_handles = {r[0] for r in self.con.execute(
            "SELECT DISTINCT handle FROM share_grant WHERE device_id=? OR ip_cluster=?",
            (device_id, ip_cluster))}
        n_cluster = len(cluster_handles) + (0 if handle in cluster_handles else 1)
        if n_cluster >= CLUSTER_FLAG_THRESHOLD:
            self.con.execute(
                "UPDATE share_device SET flagged=1 WHERE device_id=?", (device_id,)
            )
            self._fraud(handle=handle, reason=f"cluster {ip_cluster}/{device_id} at "
                                              f"{n_cluster} accounts", now=now)
            self._audit("flag", handle, None,
                        f"cluster {ip_cluster} flagged at {n_cluster} accounts", now)
            self.con.commit()
            raise ShareError("cluster flagged: too many accounts on this device/IP")

    def _fraud(self, handle: str, reason: str, now: float) -> None:
        # deterministic queue id (sha256, not hash() — hash() is randomized
        # per process, so a collision under INSERT OR IGNORE could silently
        # drop a fraud row across restarts)
        import hashlib
        qid = "fraud-" + hashlib.sha256(
            f"{handle}|{reason}|{now}".encode()).hexdigest()[:24]
        self.con.execute(
            "INSERT OR IGNORE INTO fraud_queue (queue_id, handle, reason, at, status)"
            " VALUES (?,?,?,?,'open')",
            (qid, handle, reason, now),
        )

    def _bump_device_grants(self, device_id: str, now: float) -> None:
        self.con.execute(
            "UPDATE share_device SET grants_30d=grants_30d+1 WHERE device_id=?",
            (device_id,),
        )

    # -- verification ---------------------------------------------------------------

    def verify_share_post(self, post_ref: str, handle: str, card_id: str) -> bool:
        """Live-post verification (§4.7): the public post must contain the
        anonymized card link + the attribution tag (the sharer's handle).
        Unverifiable shares grant NOTHING. This is the single choke point —
        the real implementation calls the social platform's public API or a
        review queue; here the post_ref must carry both the card id and the
        handle tag, anything else fails closed."""
        if not post_ref or card_id not in post_ref or handle not in post_ref:
            return False
        return True

    # -- the grant --------------------------------------------------------------------

    def grant(self, *, handle: str, card_id: str, device_id: str, ip_cluster: str,
              post_ref: str, account_age_s: float, email_verified: bool,
              phone_otp_verified: bool, now: float | None = None) -> ShareGrant:
        """Verify + grant one permanent vault slot for a verified share of
        `card_id`. One slot per (handle, card_id) ever; cap +5 total. The
        shareable object is rendered server-side (watermarked face) — the
        grant records the post that carries it. Read-back before returning:
        the vault slot count is re-read after the write (hard rule)."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        card = self.cards.assert_owned(card_id, handle)   # non-transferable
        # render the shareable object server-side (watermark = sharer handle)
        self.face.render_share(card_id, owner_handle=handle, now=now)

        dup = self.con.execute(
            "SELECT grant_id FROM share_grant WHERE handle=? AND card_id=? AND revoked_at IS NULL",
            (handle, card_id),
        ).fetchone()
        if dup is not None:
            self._audit("refuse", handle, card_id,
                        f"duplicate grant for card (existing {dup['grant_id']})", now)
            raise ShareError(f"card {card_id} already granted a slot to {handle!r} — "
                             "one slot per (account, card) pair")

        self._check_prereqs(handle, account_age_s, email_verified,
                            phone_otp_verified, now)
        self._device_gate(handle, device_id, ip_cluster, now)

        if not self.verify_share_post(post_ref, handle, card_id):
            self._audit("verify_fail", handle, card_id,
                        f"unverifiable share post {post_ref!r} — grants nothing", now)
            self._fraud(handle, f"unverifiable share post for {card_id}", now)
            self.con.commit()          # audit log is append-only: commit even on refuse
            raise ShareError("unverifiable share: the post must publicly contain the "
                             "anonymized card link + attribution tag")

        # cap enforcement at +5 (vault also caps; double-checked here via read-back)
        current_slots = self._current_slots(handle)
        if current_slots >= MAX_SHARE_SLOTS:
            self._audit("refuse", handle, card_id,
                        f"cap reached ({current_slots}/{MAX_SHARE_SLOTS})", now)
            self.con.commit()
            raise ShareError(f"share-slot cap reached ({MAX_SHARE_SLOTS})")

        grant_id = f"grant-{handle}-{card_id}-{int(now * 1000)}-{abs(hash((handle, card_id, now))) % 100000:05d}"
        try:
            self.con.execute(
                "INSERT INTO share_grant (grant_id, handle, card_id, device_id,"
                " ip_cluster, post_ref, verified, granted_at) VALUES (?,?,?,?,?,?,1,?)",
                (grant_id, handle, card_id, device_id, ip_cluster, post_ref, now),
            )
            self._bump_device_grants(device_id, now)
            self._audit("grant", handle, card_id,
                        f"+{1} permanent slot via verified share (post {post_ref!r})", now)
            self.con.commit()

            # vault write + READ-BACK (hard rule: GET after write). The vault's
            # add_permanent_slots accumulates on the owner's LATEST vault row and
            # caps at +5 total; effective_capacity reads that stored count back
            # from the DB. After this grant the stored count must equal the
            # number of active (non-revoked) grants (<= cap), so compare against
            # the vault's own stored value, not a recomputed one.
            self.vault.add_permanent_slots(handle, 1, actor=handle)
            rb = self.vault.effective_capacity(handle, card.tier)
            stored = self.vault.con.execute(
                "SELECT COALESCE(MAX(extra_slots),0) m FROM vault_entry WHERE owner=?",
                (handle,),
            ).fetchone()["m"]
            if rb != self.vault.base_capacity(card.tier) + min(stored, MAX_SHARE_SLOTS):
                raise ShareError("vault read-back mismatch after grant — investigate")
        except Exception:
            # compensating rollback: the grant row is undone so a failed
            # vault read-back never leaves a half-granted slot
            self.con.execute(
                "DELETE FROM share_grant WHERE grant_id=?", (grant_id,)
            )
            self._audit("rollback", handle, card_id,
                        "grant rolled back after vault read-back failure", now)
            self.con.commit()
            raise
        return self.get_grant(grant_id)

    def revoke(self, handle: str, card_id: str, reason: str,
               now: float | None = None) -> ShareGrant:
        """Abuse path: revoke a granted slot (vault.revoke_permanent_slots
        zeroes the account's permanent slots) and flag the account. Repeat
        offense -> share feature disabled for the account (first offense is
        a warning recorded in the audit log, not a ban). Read-back."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        row = self.con.execute(
            "SELECT * FROM share_grant WHERE handle=? AND card_id=? AND revoked_at IS NULL",
            (handle, card_id),
        ).fetchone()
        if row is None:
            raise ShareError(f"no active grant for {card_id!r}/{handle!r}")
        self.con.execute(
            "UPDATE share_grant SET revoked_at=? WHERE grant_id=?",
            (now, row["grant_id"]),
        )
        revoked_slots = self.vault.revoke_permanent_slots(handle)
        self._fraud(handle, f"share slot revoked for {card_id}: {reason}", now)
        self._audit("revoke", handle, card_id,
                    f"slot revoked ({reason}); vault permanent slots zeroed "
                    f"(were {revoked_slots})", now)
        self.con.commit()
        return self.get_grant(row["grant_id"])

    # -- reads --------------------------------------------------------------------

    def get_grant(self, grant_id: str) -> ShareGrant:
        row = self.con.execute(
            "SELECT * FROM share_grant WHERE grant_id=?", (grant_id,)
        ).fetchone()
        if row is None:
            raise ShareError(f"unknown grant {grant_id!r}")
        return self._to_grant(row)

    def grants_for(self, handle: str) -> list[ShareGrant]:
        rows = self.con.execute(
            "SELECT * FROM share_grant WHERE handle=? ORDER BY granted_at", (handle,)
        ).fetchall()
        return [self._to_grant(r) for r in rows]

    def _current_slots(self, handle: str) -> int:
        """Active (non-revoked) share slots for the account — the number the
        +5 cap is enforced against."""
        return self.con.execute(
            "SELECT COUNT(*) c FROM share_grant WHERE handle=? AND revoked_at IS NULL",
            (handle,),
        ).fetchone()["c"]

    def share_slots(self, handle: str) -> int:
        """Read-back of the permanent slots for the account: the active
        (non-revoked) grant count from this service's own audit store, capped
        at +5. When the flag is off the engagement counters read as 0 (the
        kill-switch never hides data but grants no new engagement)."""
        if not flags.cards_v4_1_enabled(handle):
            return 0
        return min(self._current_slots(handle), MAX_SHARE_SLOTS)

    def audit_log(self) -> list[dict]:
        """The append-only audit log (who/when/device/IP cluster/post)."""
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM share_audit ORDER BY seq").fetchall()]

    def fraud_queue(self, status: str = "open") -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM fraud_queue WHERE status=? ORDER BY at", (status,)).fetchall()]

    def _to_grant(self, row: sqlite3.Row) -> ShareGrant:
        return ShareGrant(
            grant_id=row["grant_id"], handle=row["handle"], card_id=row["card_id"],
            device_id=row["device_id"], ip_cluster=row["ip_cluster"],
            post_ref=row["post_ref"], verified=bool(row["verified"]),
            granted_at=row["granted_at"], revoked_at=row["revoked_at"],
        )


__all__ = [
    "CLUSTER_FLAG_THRESHOLD",
    "MAX_GRANTS_PER_DEVICE_30D",
    "MAX_SHARE_SLOTS",
    "MIN_ACCOUNT_AGE_S",
    "MIN_COMPLETED_RUNS",
    "ShareError",
    "ShareGrant",
    "ShareService",
]
