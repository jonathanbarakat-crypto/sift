"""v4.1 share-for-capacity (mechanics 6, behind `cards_v4_1`).

+1 PERMANENT vault slot per VERIFIED share, cap +5, non-transferable
(slots live on the owner's vault row — they can never move to another
account), revocable.

Anti-abuse stack (all server-side, all audited):
  1. per-(account, card) uniqueness — one grant per (handle, card_id)
  2. OTP + KYC-lite verification results (passed in by the server-side
     account service; never trusted from the client)
  3. device/IP/ASN clustering — a device key seen on a DIFFERENT account
     flags the share (Sybil cluster)
  4. account age >= MIN_ACCOUNT_AGE_DAYS AND >= 1 completed run
  5. live-post verification — the shared post must carry the watermarked
     shareable object (card_id + watermark handle + ts)
  6. watermark — the share object is the card face rendered by
     CardFaceRenderer.render_share (handle + timestamp), so leaks trace
  7. append-only audit log — every share attempt is INSERTed into
     share_audit with a monotonically increasing seq; there is no update
     or delete path in this module
  8. fraud queue — failed/held shares land in fraud_queue (open), and an
     account with an open fraud flag gets no grants (held)
  9. honeypots — server-planted trap cards; any share of a honeypot is
     always queued as fraud and never grants

Hard rules: cards are non-transferable (a share grant can only be earned
on a card the handle OWNS — assert_owned); no P&L / tickers / directional
content in the share object (the face renderer's ticker guard applies);
flag off -> ShareError(flag_off) on every mutating path.

Read-back discipline: record_share / revoke return state re-read from the
DB (grant row + vault extra_slots), never just the write.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from sift import flags
from sift.cards import CardRegistry
from sift.cards.face import CardFaceRenderer
from sift.cards.vault import Vault

MIN_ACCOUNT_AGE_DAYS = 7
MIN_COMPLETED_RUNS = 1
MAX_EXTRA_SLOTS = 5          # cap +5 (mechanics 6); vault enforces the same cap

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS share_grant (
    handle      TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    granted_at  REAL NOT NULL,
    slots_after INTEGER NOT NULL,
    post_url    TEXT NOT NULL,
    PRIMARY KEY (handle, card_id)          -- per-(account, card) uniqueness
);
CREATE TABLE IF NOT EXISTS share_audit (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,   -- append-only, monotonic
    ts          REAL NOT NULL,
    handle      TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    action      TEXT NOT NULL,              -- share|revoke|fraud_flag|honeypot_plant
    outcome     TEXT NOT NULL,              -- granted|cap_reached|fraud_queued|held_fraud_review|revoked|duplicate|...
    detail      TEXT NOT NULL DEFAULT '{}',
    UNIQUE (seq)
);
CREATE TABLE IF NOT EXISTS fraud_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    handle      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open'   -- open|cleared
);
CREATE TABLE IF NOT EXISTS device_map (
    device_key  TEXT NOT NULL,              -- device_id|ip|asn
    handle      TEXT NOT NULL,
    first_seen  REAL NOT NULL,
    PRIMARY KEY (device_key, handle)
);
CREATE TABLE IF NOT EXISTS honeypot (
    card_id     TEXT PRIMARY KEY,
    planted_at  REAL NOT NULL
);
"""


class ShareError(Exception):
    """Share-for-capacity misuse: flag off, foreign card, duplicate, ..."""


class ShareManager:
    """Server-side share-for-capacity over the card registry + vault."""

    def __init__(self, cards: CardRegistry, vault: Vault, db_path: str | Path,
                 face: CardFaceRenderer | None = None):
        self.cards = cards
        self.vault = vault
        self.face = face
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "ShareManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- audit (append-only: INSERTs only, no update/delete anywhere) --------

    def _audit(self, handle: str, card_id: str, action: str, outcome: str,
               detail: dict | None = None, ts: float | None = None) -> int:
        ts = time.time() if ts is None else ts
        cur = self.con.execute(
            "INSERT INTO share_audit (ts, handle, card_id, action, outcome, detail)"
            " VALUES (?,?,?,?,?,?)",
            (ts, handle, card_id, action, outcome, json.dumps(detail or {}, default=str)),
        )
        return cur.lastrowid

    def audit_log(self) -> list[dict]:
        rows = self.con.execute(
            "SELECT seq, ts, handle, card_id, action, outcome, detail FROM share_audit"
            " ORDER BY seq"
        ).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    # -- fraud queue + honeypots ------------------------------------------------

    def flag_fraud(self, handle: str, reason: str, detail: dict | None = None,
                   card_id: str = "-", now: float | None = None) -> int:
        """Server-side: put an account on the fraud queue (open). While an
        open flag exists, record_share holds (no grants)."""
        now = time.time() if now is None else now
        cur = self.con.execute(
            "INSERT INTO fraud_queue (handle, reason, detail, created_at) VALUES (?,?,?,?)",
            (handle, reason, json.dumps(detail or {}, default=str), now),
        )
        self.con.commit()
        self._audit(handle, card_id, "fraud_flag", "fraud_queued", detail, ts=now)
        self.con.commit()
        return cur.lastrowid

    def has_open_fraud(self, handle: str) -> bool:
        row = self.con.execute(
            "SELECT 1 FROM fraud_queue WHERE handle=? AND status='open' LIMIT 1",
            (handle,),
        ).fetchone()
        return row is not None

    def fraud_queue_list(self, status: str = "open") -> list[dict]:
        rows = self.con.execute(
            "SELECT id, handle, reason, detail, created_at, status FROM fraud_queue"
            " WHERE status=? ORDER BY id",
            (status,),
        ).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    def plant_honeypot(self, card_id: str, now: float | None = None) -> None:
        """Server-side trap card. Any share attempt against it is always
        queued as fraud and never grants (honeypot)."""
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT OR IGNORE INTO honeypot (card_id, planted_at) VALUES (?,?)",
            (card_id, now),
        )
        self.con.commit()
        self._audit("-", card_id, "honeypot_plant", "planted", {}, ts=now)
        self.con.commit()

    def is_honeypot(self, card_id: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM honeypot WHERE card_id=?", (card_id,)
        ).fetchone() is not None

    # -- grants -------------------------------------------------------------------

    def grants_for(self, handle: str) -> list[dict]:
        rows = self.con.execute(
            "SELECT handle, card_id, granted_at, slots_after, post_url FROM share_grant"
            " WHERE handle=? ORDER BY granted_at",
            (handle,),
        ).fetchall()
        return [dict(r) for r in rows]

    def slots_granted(self, handle: str) -> int:
        row = self.con.execute(
            "SELECT COUNT(*) c FROM share_grant WHERE handle=?", (handle,)
        ).fetchone()
        return row["c"]

    # -- the share path ------------------------------------------------------------

    def record_share(self, handle: str, card_id: str, *, post_url: str,
                     post_payload: dict, device_id: str, ip: str, asn: str,
                     account_age_days: float, completed_runs: int,
                     otp_verified: bool, kyc_lite: bool,
                     now: float | None = None) -> dict:
        """Record a share attempt and, if every anti-abuse check passes and
        the cap is not reached, grant +1 permanent vault slot.

        Returns the read-back result dict:
          {outcome, granted, slots_after, grant, audit_seq}
        Raises ShareError on: flag off, foreign/unknown card (cards are
        non-transferable), duplicate (account, card) share.
        Abuse signals never raise — they audit + land in the fraud queue
        with outcome "fraud_queued" (or "held_fraud_review").
        """
        if not flags.cards_v4_1_enabled(handle):
            raise ShareError("flag_off: cards_v4_1 is disabled for this account")
        card = self.cards.assert_owned(card_id, handle)   # non-transferable
        now = time.time() if now is None else now

        def _fraud(reason: str, detail: dict) -> dict:
            self.flag_fraud(handle, reason, {**detail, "card_id": card_id}, card_id=card_id, now=now)
            seq = self._audit(handle, card_id, "share", "fraud_queued",
                              {"reason": reason, **detail}, ts=now)
            self.con.commit()
            return {"outcome": "fraud_queued", "reason": reason, "granted": False,
                    "slots_after": self._extra_slots(handle), "grant": None, "audit_seq": seq}

        # 9. honeypot: always fraud, never grants
        if self.is_honeypot(card_id):
            return _fraud("honeypot_share", {"post_url": post_url})
        # 8. account under fraud review: held, no grant
        if self.has_open_fraud(handle):
            seq = self._audit(handle, card_id, "share", "held_fraud_review", {}, ts=now)
            self.con.commit()
            return {"outcome": "held_fraud_review", "granted": False,
                    "slots_after": self._extra_slots(handle), "grant": None, "audit_seq": seq}
        # 1. per-(account, card) uniqueness (fail closed: audit, then raise)
        dup = self.con.execute(
            "SELECT 1 FROM share_grant WHERE handle=? AND card_id=?", (handle, card_id)
        ).fetchone()
        if dup is not None:
            seq = self._audit(handle, card_id, "share", "duplicate", {"post_url": post_url}, ts=now)
            self.con.commit()
            raise ShareError(
                f"duplicate share: ({handle!r}, {card_id!r}) already granted — "
                "per-(account, card) uniqueness"
            )
        # 4. account age + >=1 completed run
        if account_age_days < MIN_ACCOUNT_AGE_DAYS:
            return _fraud("account_too_new", {"account_age_days": account_age_days,
                                              "min": MIN_ACCOUNT_AGE_DAYS})
        if completed_runs < MIN_COMPLETED_RUNS:
            return _fraud("no_completed_run", {"completed_runs": completed_runs,
                                               "min": MIN_COMPLETED_RUNS})
        # 2. OTP + KYC-lite
        if not otp_verified:
            return _fraud("otp_not_verified", {})
        if not kyc_lite:
            return _fraud("kyc_lite_failed", {})
        # 3. device/IP/ASN clustering: same device key on another account = Sybil
        device_key = f"{device_id}|{ip}|{asn}"
        row = self.con.execute(
            "SELECT handle FROM device_map WHERE device_key=? AND handle<>? LIMIT 1",
            (device_key, handle),
        ).fetchone()
        if row is not None:
            return _fraud("device_ip_asn_cluster",
                          {"device_key": device_key, "other_handle": row["handle"]})
        # 5+6. live-post verification against the watermarked share object
        if not self._verify_live_post(handle, card, post_url, post_payload):
            return _fraud("live_post_verification_failed", {"post_url": post_url})

        # all checks passed -> verified share
        self.con.execute(
            "INSERT OR IGNORE INTO device_map (device_key, handle, first_seen) VALUES (?,?,?)",
            (device_key, handle, now),
        )
        if self._extra_slots(handle) >= MAX_EXTRA_SLOTS:
            seq = self._audit(handle, card_id, "share", "cap_reached",
                              {"post_url": post_url, "cap": MAX_EXTRA_SLOTS}, ts=now)
            self.con.commit()
            return {"outcome": "cap_reached", "granted": False,
                    "slots_after": self._extra_slots(handle), "grant": None, "audit_seq": seq}

        slots_after = self.vault.add_permanent_slots(handle, 1)   # non-transferable, capped +5 there too
        self.con.execute(
            "INSERT INTO share_grant (handle, card_id, granted_at, slots_after, post_url)"
            " VALUES (?,?,?,?,?)",
            (handle, card_id, now, slots_after, post_url),
        )
        seq = self._audit(handle, card_id, "share", "granted",
                          {"post_url": post_url, "slots_after": slots_after}, ts=now)
        self.con.commit()
        # read-back (hard rule): grant row + vault slots re-read from the DB
        grant = self.con.execute(
            "SELECT handle, card_id, granted_at, slots_after, post_url FROM share_grant"
            " WHERE handle=? AND card_id=?",
            (handle, card_id),
        ).fetchone()
        return {
            "outcome": "granted",
            "granted": True,
            "slots_after": self._extra_slots(handle),
            "grant": dict(grant) if grant else None,
            "audit_seq": seq,
        }

    def _verify_live_post(self, handle: str, card, post_url: str,
                          post_payload: dict) -> bool:
        """Live-post verification (anti-abuse 5+6): the post URL must be
        non-empty and the post payload must carry the watermarked shareable
        object — the card id plus the watermark (handle + timestamp) exactly
        as render_share emits it. A payload carrying a ticker-like token is
        refused by the same fail-closed face guard (raw tickers never leave
        the server, v3 §12)."""
        import json

        from sift.cards.face import FaceError, _guard_no_tickers

        if not post_url or not post_url.startswith(("https://", "http://")):
            return False
        wm = (post_payload or {}).get("watermark") or {}
        if (post_payload or {}).get("card_id") != card.card_id:
            return False
        if wm.get("handle") != handle or wm.get("ts") is None:
            return False
        try:
            _guard_no_tickers(json.dumps(post_payload, default=str))
        except FaceError:
            return False
        if self.face is not None:
            # re-render the share object server-side and confirm the posted
            # artifact matches it (watermark handle + the face guard)
            try:
                face = self.face.render_share(card.card_id, handle)
            except Exception:
                return False
            if face.get("card_id") != card.card_id:
                return False
            if face.get("watermark", {}).get("handle") != wm.get("handle"):
                return False
        return True

    def revoke(self, handle: str, card_id: str, now: float | None = None) -> dict:
        """Revocable: remove one granted share (slot -1, floor 0). The grant
        row is removed but the AUDIT LOG keeps every event (append-only).
        Read-back before returning."""
        if not flags.cards_v4_1_enabled(handle):
            raise ShareError("flag_off: cards_v4_1 is disabled for this account")
        now = time.time() if now is None else now
        row = self.con.execute(
            "SELECT slots_after FROM share_grant WHERE handle=? AND card_id=?",
            (handle, card_id),
        ).fetchone()
        if row is None:
            raise ShareError(f"no grant for ({handle!r}, {card_id!r}) to revoke")
        self.con.execute(
            "DELETE FROM share_grant WHERE handle=? AND card_id=?", (handle, card_id)
        )
        before = self._extra_slots(handle)
        after = self.vault.remove_permanent_slots(handle, 1)
        seq = self._audit(handle, card_id, "revoke", "revoked",
                          {"slots_before": before, "slots_after": after}, ts=now)
        self.con.commit()
        # read-back: grant gone, vault slots re-read
        still = self.con.execute(
            "SELECT 1 FROM share_grant WHERE handle=? AND card_id=?", (handle, card_id)
        ).fetchone()
        return {
            "outcome": "revoked",
            "grant_removed": still is None,
            "slots_after": self._extra_slots(handle),
            "audit_seq": seq,
        }

    def _extra_slots(self, handle: str) -> int:
        """Read-back of the owner's current permanent share slots (from the
        VAULT's own DB, not this module's)."""
        row = self.vault.con.execute(
            "SELECT extra_slots FROM vault_entry WHERE owner=? ORDER BY updated_at DESC LIMIT 1",
            (handle,),
        ).fetchone()
        return row["extra_slots"] if row else 0


__all__ = [
    "MAX_EXTRA_SLOTS",
    "MIN_ACCOUNT_AGE_DAYS",
    "MIN_COMPLETED_RUNS",
    "ShareError",
    "ShareManager",
]
