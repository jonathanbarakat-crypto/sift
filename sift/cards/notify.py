"""v4.1 per-card opt-in hit notifications + digest mode (behind `cards_v4_1`).

Mechanics (authoritative, spec §2/§3/§4 + v3 §7.2/§9.5/§6.1):
- Live "hit" notifications for a user's OWN saved cards: opt-in PER CARD.
  Free tier defaults to DIGEST (rate-capped); paid gets live. Lifecycle
  emails stay capped at 1/week (v3 §7.2).
- Notifications are signal-delivery — monitoring the user's own saved
  connection. NOT bets: signals-only, no execution (v3 §12.1 unchanged).
  No countdown timers, no loss-aversion triggers, no push spam.
- Ghost cards keep full personal utility incl. notifications (pull/14).
- Queue with an append-only delivery log (same discipline as the trial
  registry): every enqueue/deliver/digest-batch is logged and read back.

Flag: flag off -> NotifyError(flag_off) on every mutating path; the queue
read stays available (kill-switch never hides existing data).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import flags
from sift.cards import CardRegistry

TIERS = ("free", "paid")
MODES = ("off", "digest", "live")
MAX_LIVE_PER_CARD_PER_DAY = 3          # push-spam cap (no loss-aversion triggers)
MAX_DIGEST_PER_DAY = 1                 # digest batching: at most 1 digest/day
MAX_LIFECYCLE_EMAILS_PER_WEEK = 1      # v3 §7.2

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notify_pref (
    handle      TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'off',     -- off|digest|live (per-card opt-in)
    updated_at  REAL NOT NULL,
    PRIMARY KEY (handle, card_id)
);
CREATE TABLE IF NOT EXISTS notify_queue (
    event_id    TEXT PRIMARY KEY,
    handle      TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,                    -- hit|decay|digest
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    delivered_at REAL,
    channel     TEXT                              -- push|email|digest
);
CREATE TABLE IF NOT EXISTS notify_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT, -- append-only delivery log
    at          REAL NOT NULL,
    event_id    TEXT NOT NULL,
    action      TEXT NOT NULL,                     -- enqueue|deliver|digest_batch|rate_cap
    detail      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rate_window (
    handle      TEXT NOT NULL,
    card_id     TEXT NOT NULL,
    channel     TEXT NOT NULL,
    window_day  TEXT NOT NULL,                    -- UTC date key
    count       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (handle, card_id, channel, window_day)
);
"""


class NotifyError(Exception):
    """Notification misuse: flag off, bad mode, unknown card, ..."""


@dataclass(frozen=True)
class NotifyEvent:
    event_id: str
    handle: str
    card_id: str
    kind: str
    payload: dict
    created_at: float
    delivered_at: float | None
    channel: str | None


def _day_key(now: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(now, datetime.timezone.utc).strftime("%Y-%m-%d")


class NotifyService:
    """Per-card opt-in notification prefs + rate-capped delivery queue.
    Server-side only; signals-only content (no execution, no bets)."""

    def __init__(self, cards: CardRegistry, db_path: str | Path):
        self.cards = cards
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "NotifyService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def _require_flag(self, handle: str) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise NotifyError("flag_off: cards_v4_1 is disabled for this account")

    def _log(self, event_id: str, action: str, detail: str, now: float) -> None:
        self.con.execute(
            "INSERT INTO notify_log (at, event_id, action, detail) VALUES (?,?,?,?)",
            (now, event_id, action, detail),
        )

    # -- prefs ---------------------------------------------------------------------

    def set_mode(self, handle: str, card_id: str, mode: str,
                 now: float | None = None) -> str:
        """Per-card opt-in. mode in off|digest|live. The card must be owned
        by the handle (non-transferable; ghosts included — ghosts keep full
        personal utility). Read-back before returning."""
        self._require_flag(handle)
        if mode not in MODES:
            raise NotifyError(f"mode must be one of {MODES}, got {mode!r}")
        self.cards.assert_owned(card_id, handle)
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT INTO notify_pref (handle, card_id, mode, updated_at) VALUES (?,?,?,?)"
            " ON CONFLICT(handle, card_id) DO UPDATE SET mode=excluded.mode,"
            " updated_at=excluded.updated_at",
            (handle, card_id, mode, now),
        )
        self.con.commit()
        return self.get_mode(handle, card_id)          # read-back

    def get_mode(self, handle: str, card_id: str) -> str:
        row = self.con.execute(
            "SELECT mode FROM notify_pref WHERE handle=? AND card_id=?",
            (handle, card_id),
        ).fetchone()
        return row["mode"] if row else "off"           # default OFF (opt-in)

    def effective_mode(self, handle: str, card_id: str, tier: str = "free") -> str:
        """The mode actually in force: per-card pref, but free tier can never
        get live — free defaults to digest (rate-capped); paid may go live."""
        mode = self.get_mode(handle, card_id)
        if mode == "live" and tier == "free":
            return "digest"
        return mode

    # -- rate caps -------------------------------------------------------------------

    def _bump(self, handle: str, card_id: str, channel: str, now: float) -> int:
        day = _day_key(now)
        self.con.execute(
            "INSERT INTO rate_window (handle, card_id, channel, window_day, count)"
            " VALUES (?,?,?,?,1)"
            " ON CONFLICT(handle, card_id, channel, window_day)"
            " DO UPDATE SET count=count+1",
            (handle, card_id, channel, day),
        )
        return self.con.execute(
            "SELECT count c FROM rate_window WHERE handle=? AND card_id=? AND channel=? AND window_day=?",
            (handle, card_id, channel, day),
        ).fetchone()["c"]

    def _cap_for(self, channel: str) -> int:
        if channel == "push":
            return MAX_LIVE_PER_CARD_PER_DAY
        if channel == "digest":
            return MAX_DIGEST_PER_DAY
        return MAX_LIFECYCLE_EMAILS_PER_WEEK  # email handled weekly elsewhere; cap here too

    # -- enqueue ----------------------------------------------------------------------

    def notify_hit(self, *, handle: str, card_id: str, payload: dict,
                   tier: str = "free", now: float | None = None) -> NotifyEvent | None:
        """Enqueue a "hit" notification for the user's own saved card.
        - mode off -> nothing (opt-in respected).
        - live (paid only) -> push channel, capped at MAX_LIVE_PER_CARD_PER_DAY.
        - digest (free default) -> queued for the daily digest batch, capped at
          MAX_DIGEST_PER_DAY.
        Returns the queued event, or None when suppressed (opt-out / rate cap).
        Signals-only payload — no execution, no bets (v3 §12.1)."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        self.cards.assert_owned(card_id, handle)
        mode = self.effective_mode(handle, card_id, tier)
        if mode == "off":
            return None
        channel = "push" if mode == "live" else "digest"
        # rate cap check BEFORE enqueue (no push spam)
        day = _day_key(now)
        cur = self.con.execute(
            "SELECT count c FROM rate_window WHERE handle=? AND card_id=? AND channel=? AND window_day=?",
            (handle, card_id, channel, day),
        ).fetchone()
        if cur is not None and cur["c"] >= self._cap_for(channel):
            self._log("", "rate_cap",
                      f"{handle}/{card_id}/{channel} at cap {self._cap_for(channel)}/day", now)
            self.con.commit()
            return None
        event_id = f"evt-{handle}-{card_id}-{int(now * 1000)}-{abs(hash((handle, card_id, 'hit', now))) % 100000:05d}"
        self.con.execute(
            "INSERT INTO notify_queue (event_id, handle, card_id, kind, payload,"
            " created_at, channel) VALUES (?,?,?,?,?,?,?)",
            (event_id, handle, card_id, "hit",
             json.dumps(payload, sort_keys=True, default=str), now, channel),
        )
        self._bump(handle, card_id, channel, now)
        self._log(event_id, "enqueue", f"hit via {channel} (mode={mode})", now)
        self.con.commit()
        return self.get_event(event_id)                # read-back

    def enqueue_decay_notice(self, *, handle: str, card_id: str, payload: dict,
                             now: float | None = None) -> NotifyEvent | None:
        """Decay/downgrade notice through the per-card channel (the decay job
        calls this; ghosts keep full utility incl. notifications).

        Rate-cap note: this is a single system-generated event per decay
        transition (decay is downgrade-only and monotonic, so it fires at
        most once per tier step per card), so it intentionally does NOT
        consume the per-card push/digest rate budget — but it still respects
        the per-card opt-in mode (off -> nothing) and is logged."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        self.cards.assert_owned(card_id, handle)
        mode = self.get_mode(handle, card_id)
        if mode == "off":
            return None
        channel = "push" if mode == "live" else "digest"
        event_id = f"evt-{handle}-{card_id}-decay-{int(now * 1000)}-{abs(hash((handle, card_id, now))) % 100000:05d}"
        self.con.execute(
            "INSERT INTO notify_queue (event_id, handle, card_id, kind, payload,"
            " created_at, channel) VALUES (?,?,?,?,?,?,?)",
            (event_id, handle, card_id, "decay",
             json.dumps(payload, sort_keys=True, default=str), now, channel),
        )
        self._log(event_id, "enqueue", f"decay notice via {channel}", now)
        self.con.commit()
        return self.get_event(event_id)

    def digest_batch(self, handle: str, now: float | None = None) -> list[NotifyEvent]:
        """Flush the pending digest queue for the user into ONE digest event
        (free tier: rate-capped to MAX_DIGEST_PER_DAY). Returns the delivered
        events (read-back)."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        pending = self.con.execute(
            "SELECT * FROM notify_queue WHERE handle=? AND channel='digest'"
            " AND delivered_at IS NULL ORDER BY created_at",
            (handle,),
        ).fetchall()
        if not pending:
            return []
        day = _day_key(now)
        cur = self.con.execute(
            "SELECT count c FROM rate_window WHERE handle=? AND card_id='*'"
            " AND channel='digest' AND window_day=?",
            (handle, day),
        ).fetchone()
        if cur is not None and cur["c"] >= MAX_DIGEST_PER_DAY:
            self._log("", "rate_cap", f"{handle} digest at cap {MAX_DIGEST_PER_DAY}/day", now)
            self.con.commit()
            return []
        delivered = []
        for r in pending:
            self.con.execute(
                "UPDATE notify_queue SET delivered_at=? WHERE event_id=?",
                (now, r["event_id"]),
            )
            self._log(r["event_id"], "digest_batch",
                      f"batched {r['kind']} for {r['card_id']} into daily digest", now)
            delivered.append(self.get_event(r["event_id"]))
        self.con.execute(
            "INSERT INTO rate_window (handle, card_id, channel, window_day, count)"
            " VALUES (?, '*', 'digest', ?, 1)"
            " ON CONFLICT(handle, card_id, channel, window_day) DO UPDATE SET count=count+1",
            (handle, day),
        )
        self.con.commit()
        return delivered

    # -- reads ------------------------------------------------------------------------

    def get_event(self, event_id: str) -> NotifyEvent:
        row = self.con.execute(
            "SELECT * FROM notify_queue WHERE event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            raise NotifyError(f"unknown event {event_id!r}")
        return NotifyEvent(
            event_id=row["event_id"], handle=row["handle"], card_id=row["card_id"],
            kind=row["kind"], payload=json.loads(row["payload"]),
            created_at=row["created_at"], delivered_at=row["delivered_at"],
            channel=row["channel"],
        )

    def queue_for(self, handle: str) -> list[NotifyEvent]:
        rows = self.con.execute(
            "SELECT * FROM notify_queue WHERE handle=? ORDER BY created_at", (handle,)
        ).fetchall()
        return [NotifyEvent(
            event_id=r["event_id"], handle=r["handle"], card_id=r["card_id"],
            kind=r["kind"], payload=json.loads(r["payload"]),
            created_at=r["created_at"], delivered_at=r["delivered_at"],
            channel=r["channel"],
        ) for r in rows]

    def delivery_log(self) -> list[dict]:
        """The append-only delivery log."""
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM notify_log ORDER BY seq").fetchall()]


__all__ = [
    "MAX_DIGEST_PER_DAY",
    "MAX_LIVE_PER_CARD_PER_DAY",
    "MODES",
    "NotifyError",
    "NotifyEvent",
    "NotifyService",
    "TIERS",
]
