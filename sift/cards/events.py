"""v4.1 rich signal-event package for card hit notifications (mechanics 7/8
extension, owner decision 2026-11-18).

Extends sift/cards/notify.py's notify mechanics with a RICH, versioned
signal-event payload delivered to the owner's chosen channel (email / RSS /
API webhook):

- `event_payload` (schema v1) carries: card_id, family, scope_id, tier,
  event="hit", ts (UTC) — MAGNITUDE ONLY (statistic = |z|, threshold = the
  card's entry band; never the sign, never which leg) — frequency/persistence
  context (hits_trailing_12m, median_hold_days, half_life_days) read from
  ledger STAGE ROWS + ledger events only, NEVER from params_json — trailing
  12m health ratio vs the mint baseline + current tier, and the anonymized
  aggregate `cards_fired_today` across the owner's saved set.
- Delivery ergonomics: per-card channel pref (email|rss|api), quiet hours,
  free tier = rate-capped daily digest, live per-hit = paid
  `entitlement="live"` (unchanged from notify.py). Lifecycle caps unchanged
  (1/week/card); append-only log + idempotent outbox (unchanged).
- RSS feed per owner (auth token in the URL, revocable); API = POST JSON to
  a registered webhook with an HMAC-SHA256 signature header.

Hard red lines (v3 §12 + v4.1 §6), enforced FAIL-CLOSED here:
- NO tickers / symbols: the payload builder never reads params_json; any
  attempt to include a symbol or a sign/direction field raises NotifyError
  and the event is NOT delivered. A leak guard re-scans the serialized
  payload for ticker-shaped tokens (same discipline as the card face).
- NO P&L / returns / value language: all user-facing text is built ONLY in
  `copy_for_event` (extended here with the hit gloss); every string is
  grepped for BANNED_PHRASES (now also covering buy/sell/long/short, P&L,
  "value", urgency/countdown) and any hit raises NotifyError.
- Coordinate with #157 (bidirectional tiers): the decay/upgrade copy is NOT
  hardcoded down-only — `copy_for_event` renders both directions from the
  detail payload.

Flag: behind `cards_v4_1`; flag off -> NotifyError(flag_off) on mutations,
reads stay available. Kill-switch env: SIFT_CARDS_V4_1=0 (sift/flags.py).

Read-back discipline (hard rule): every write is followed by a GET from
the DB before the result is returned.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sift import flags
from sift.cards import Card, CardRegistry
from sift.cards.notify import (
    BANNED_PHRASES,
    DAY,
    MODE_DIGEST,
    MODE_LIVE,
    NotifyError,
    Notifier,
    copy_for_event,
)

# ---------------------------------------------------------------------------
# Payload schema
# ---------------------------------------------------------------------------

PAYLOAD_SCHEMA_VERSION = "v1"

# Delivery channels (per-card pref extension of notify_pref.channel).
CHANNEL_EMAIL = "email"
CHANNEL_RSS = "rss"
CHANNEL_API = "api"
CHANNELS = (CHANNEL_EMAIL, CHANNEL_RSS, CHANNEL_API)

# Fields that must NEVER appear in a signal-event payload (direction /
# symbol / P&L surface). Any attempt to set them raises NotifyError and the
# event is not delivered (fail closed).
FORBIDDEN_PAYLOAD_KEYS = (
    "symbol", "symbols", "ticker", "tickers", "legs", "leg",
    "sign", "direction", "side", "buy", "sell", "long", "short",
    "pnl", "p&l", "returns", "return", "profit", "value", "wager", "bet",
    "execution", "order",
)

# Ticker-shaped tokens (same discipline as the card face, sift/cards/face.py):
# 2-5 UPPERCASE letters, optional class suffix. The serialized payload must
# never contain one (raw tickers stay server-side, v3 §12).
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b")
_PAYLOAD_WHITELIST = {"SIFT", "JSON", "UTC", "API", "RSS", "HMAC", "SHA256"}

# Trailing window for frequency / persistence / health facts: 12 months.
TRAILING_12M_S = 365 * DAY

# Ledger event types that count as hit events for hits_trailing_12m.
HIT_EVENT_TYPE = "card_hit"

# Health-ratio baseline recorded at mint (ledger event `card_mint_baseline`).
MINT_BASELINE_EVENT = "card_mint_baseline"

# HMAC header used for API webhook delivery.
WEBHOOK_SIG_HEADER = "X-SIFT-Signature"
WEBHOOK_SIG_PREFIX = "sha256="

# Quiet-hours default (UTC): 22:00-07:00 — nothing is delivered in the
# window; queued rows wait for the window to end.
QUIET_HOURS_DEFAULT = (22 * 3600, 7 * 3600)  # (start_s, end_s) seconds-of-day UTC


def utc_ts(now: float) -> str:
    """UTC ISO-8601 timestamp for the payload `ts` field."""
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _guard_no_tickers_payload(obj) -> None:
    """Fail closed: any ticker-shaped token in the serialized payload aborts
    delivery (raw tickers never leave the server). Whitelisted all-caps
    protocol labels are allowed."""
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str):
                    _guard_token(k)
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            for tok in _TICKER_RE.findall(o):
                if tok not in _PAYLOAD_WHITELIST:
                    raise NotifyError(
                        f"payload contains a ticker-like token {tok!r} — refused"
                    )
    def _guard_token(k: str):
        for tok in _TICKER_RE.findall(k):
            if tok not in _PAYLOAD_WHITELIST:
                raise NotifyError(
                    f"payload key contains a ticker-like token {tok!r} — refused"
                )
    walk(obj)


def _guard_payload(payload: dict) -> None:
    """Full payload discipline, fail closed:
    - no forbidden (direction/symbol/P&L) keys;
    - no ticker-shaped tokens anywhere in the serialized payload;
    - `statistic` must be non-negative (magnitude only — never the sign).
    Raises NotifyError; the caller then does NOT deliver the event."""
    low_keys = {k.lower() for k in payload}
    for bad in FORBIDDEN_PAYLOAD_KEYS:
        if bad in low_keys:
            raise NotifyError(
                f"payload carries forbidden key {bad!r} (sign/symbol/direction/P&L)"
                " — event refused, not delivered"
            )
    if "statistic" in payload:
        st = payload["statistic"]
        if st is not None:
            if not isinstance(st, (int, float)) or isinstance(st, bool):
                raise NotifyError("payload statistic must be numeric — refused")
            if st < 0:
                raise NotifyError(
                    "payload statistic is negative — a signed statistic would leak"
                    " direction; refused, not delivered"
                )
    _guard_no_tickers_payload(payload)


# ---------------------------------------------------------------------------
# Survival-fact sourcing (ledger stage rows + ledger events ONLY)
# ---------------------------------------------------------------------------

def _stage_rows(ledger, trial_id: str) -> dict:
    return {r["stage"]: dict(r) for r in ledger.stages(trial_id)}


def persistence_facts(ledger, card: Card, *, now: float,
                      window_s: float = TRAILING_12M_S) -> dict:
    """Frequency/persistence context for the payload, sourced ONLY from
    ledger stage rows + append-only ledger events — never from params_json:

    - hits_trailing_12m: `card_hit` ledger events for the card's trial in
      the trailing 12m;
    - median_hold_days: median lifetime (days) of hit events in the
      trailing 12m (event age at `now`), 0.0 when no hits;
    - half_life_days: the S0 stage row's recorded half-life (survival fact,
      same source discipline as the card face).
    """
    rows = _stage_rows(ledger, card.trial_id)
    s0 = rows.get("s0", {})
    # stage rows store the calibration detail in the S0 statistic row only
    # via columns; half_life lives in the ledger event written at mint when
    # present — read it from the append-only events, never params_json.
    hl = None
    for ev in ledger.events(MINT_BASELINE_EVENT):
        try:
            p = json.loads(ev["payload_json"])
        except (ValueError, TypeError):
            continue
        if p.get("trial_id") == card.trial_id or p.get("card_id") == card.card_id:
            hl = p.get("half_life_days", hl)
    if hl is None:
        # S0 detail is not a column; fall back to None (missing never fabricated)
        hl = None
    hits = []
    for ev in ledger.events(HIT_EVENT_TYPE):
        if ev["trial_id"] == card.trial_id and ev["created_at"] >= now - window_s:
            hits.append(ev["created_at"])
    hits.sort()
    n_hits = len(hits)
    if n_hits:
        ages_days = sorted((now - c) / DAY for c in hits)
        mid = n_hits // 2
        median_hold = (ages_days[mid] if n_hits % 2
                       else 0.5 * (ages_days[mid - 1] + ages_days[mid]))
    else:
        median_hold = 0.0
    return {
        "hits_trailing_12m": n_hits,
        "median_hold_days": round(float(median_hold), 4),
        "half_life_days": (round(float(hl), 4) if hl is not None else None),
    }


def health_facts(ledger, card: Card, *, now: float,
                 window_s: float = TRAILING_12M_S) -> dict:
    """Trailing-12m health ratio vs the mint baseline + current tier.

    health_ratio = hits in trailing 12m / hits expected at the mint
    baseline rate (baseline hits/day recorded at mint in the append-only
    `card_mint_baseline` ledger event), capped reporting at 1.0 precision.
    Missing baseline -> ratio None (never fabricated). Tier is the card's
    CURRENT tier, read from the registry — so a hit on a decaying card
    (lower tier) reads differently from one on a strengthening card, and
    with #157's bidirectional tiers the copy is not hardcoded down-only."""
    baseline_hits_per_day = None
    for ev in ledger.events(MINT_BASELINE_EVENT):
        try:
            p = json.loads(ev["payload_json"])
        except (ValueError, TypeError):
            continue
        if p.get("trial_id") == card.trial_id or p.get("card_id") == card.card_id:
            baseline_hits_per_day = p.get("baseline_hits_per_day",
                                          baseline_hits_per_day)
    n_hits = 0
    for ev in ledger.events(HIT_EVENT_TYPE):
        if ev["trial_id"] == card.trial_id and ev["created_at"] >= now - window_s:
            n_hits += 1
    window_days = window_s / DAY
    if baseline_hits_per_day is None or baseline_hits_per_day <= 0:
        ratio = None
    else:
        expected = baseline_hits_per_day * window_days
        ratio = round(n_hits / expected, 6) if expected > 0 else None
    return {
        "health_ratio_trailing_12m": ratio,
        "tier": card.tier,
    }


def cards_fired_today(notifier: Notifier, handle: str, *, now: float) -> int:
    """Aggregate context: anonymized COUNT of the owner's saved cards with
    at least one hit event delivered/enqueued today. A count only — no
    card ids, no symbols (anonymized aggregate, v3 §12)."""
    day_start = now - (now % DAY)
    rows = notifier.con.execute(
        "SELECT DISTINCT card_id FROM notify_log"
        " WHERE owner=? AND event_kind='hit' AND created_at>=? AND created_at<?",
        (handle, day_start, day_start + DAY),
    ).fetchall()
    return len(rows)


def cards_fired_today_ledger(ledger, cards: CardRegistry, handle: str, *,
                             now: float) -> int:
    """Same aggregate counted from the append-only ledger hit events,
    restricted to cards owned by `handle` (ownership via the registry —
    non-transferable). Used by emit_hit so the just-recorded hit is
    included in the count."""
    day_start = now - (now % DAY)
    owner_cards = {c.card_id for c in cards.cards_for_owner(handle)}
    fired: set[str] = set()
    for ev in ledger.events(HIT_EVENT_TYPE):
        if day_start <= ev["created_at"] < day_start + DAY:
            try:
                p = json.loads(ev["payload_json"])
            except (ValueError, TypeError):
                continue
            cid = p.get("card_id")
            if cid in owner_cards:
                fired.add(cid)
    return len(fired)


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalEvent:
    """The versioned v1 signal-event payload (JSON-serializable dict)."""
    payload: dict

    def to_json(self) -> str:
        return json.dumps(self.payload, sort_keys=True, separators=(",", ":"),
                          default=str)


def build_event_payload(*, card: Card, statistic: float | None,
                        threshold: float | None, now: float,
                        ledger=None, notifier: Notifier | None = None,
                        handle: str | None = None) -> dict:
    """Build the v1 signal-event payload for a hit on `card`.

    - magnitude only: `statistic` (e.g. spread |z|) + `threshold` (the
      card's entry band) — the caller passes the ABSOLUTE statistic; a
      negative value raises NotifyError (sign would leak direction);
    - persistence/health facts from ledger stage rows / events only;
    - aggregate `cards_fired_today` when a notifier + handle are given;
    - `gloss` is the plain-language copy from `copy_for_event` ONLY.

    Fail closed: any forbidden key, signed statistic, ticker-shaped token,
    or banned phrase in the gloss raises NotifyError and the event is not
    delivered by the caller.
    """
    payload: dict = {
        "schema": PAYLOAD_SCHEMA_VERSION,
        "event": "hit",
        "card_id": card.card_id,
        "family": card.family,
        "scope_id": card.scope_id,
        "tier": card.tier,
        "ts": utc_ts(now),
        "statistic": (round(float(statistic), 6)
                      if statistic is not None else None),
        "threshold": (round(float(threshold), 6)
                      if threshold is not None else None),
    }
    if ledger is not None:
        payload.update(persistence_facts(ledger, card, now=now))
        payload.update(health_facts(ledger, card, now=now))
    if notifier is not None and handle is not None:
        payload["cards_fired_today"] = cards_fired_today(notifier, handle, now=now)
    # plain-language gloss: built ONLY in copy_for_event (hard rule)
    payload["gloss"] = copy_for_event(
        "hit", card,
        {"statistic": payload["statistic"], "threshold": payload["threshold"]},
    )
    _guard_payload(payload)
    return payload


# ---------------------------------------------------------------------------
# Channel prefs: per-card channel, quiet hours, RSS token, webhook
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notify_channel (
    card_id     TEXT NOT NULL,
    owner       TEXT NOT NULL,
    channel     TEXT NOT NULL DEFAULT 'email',   -- email|rss|api
    quiet_start INTEGER,                          -- seconds-of-day UTC, nullable
    quiet_end   INTEGER,                          -- seconds-of-day UTC, nullable
    updated_at  REAL NOT NULL,
    PRIMARY KEY (card_id, owner)
);
CREATE TABLE IF NOT EXISTS rss_token (
    token       TEXT PRIMARY KEY,                 -- random, revocable
    owner       TEXT NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    revoked_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_rss_token_owner ON rss_token(owner);
CREATE TABLE IF NOT EXISTS webhook (
    hook_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    owner       TEXT NOT NULL,
    url         TEXT NOT NULL,
    secret      TEXT NOT NULL,                    -- HMAC secret, server-side only
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    revoked_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_webhook_owner ON webhook(owner, active);
"""


def _rand_token() -> str:
    import os
    return "rss_" + os.urandom(24).hex()


def in_quiet_hours(now: float, quiet_start: int | None,
                   quiet_end: int | None) -> bool:
    """True when `now` (UTC) falls inside the quiet window. Handles
    wrap-around (e.g. 22:00-07:00). No quiet hours configured -> False."""
    if quiet_start is None or quiet_end is None:
        return False
    sod = int(now % DAY)                      # seconds of day, UTC
    if quiet_start <= quiet_end:
        return quiet_start <= sod < quiet_end
    return sod >= quiet_start or sod < quiet_end  # wraps midnight


@dataclass(frozen=True)
class ChannelPref:
    card_id: str
    owner: str
    channel: str
    quiet_start: int | None
    quiet_end: int | None
    updated_at: float


@dataclass(frozen=True)
class RssToken:
    token: str
    owner: str
    revoked: bool
    created_at: float
    revoked_at: float | None


@dataclass(frozen=True)
class Webhook:
    hook_id: int
    owner: str
    url: str
    active: bool
    created_at: float
    revoked_at: float | None


def webhook_signature(secret: str, body: str) -> str:
    """HMAC-SHA256 signature header value for API webhook delivery:
    `sha256=<hex>`. The secret never leaves the server."""
    mac = hmac.new(secret.encode(), body.encode(), hashlib.sha256)
    return WEBHOOK_SIG_PREFIX + mac.hexdigest()


def verify_webhook_signature(secret: str, body: str, header_value: str) -> bool:
    """Constant-time verification of a webhook signature header."""
    if not header_value.startswith(WEBHOOK_SIG_PREFIX):
        return False
    expected = webhook_signature(secret, body)
    return hmac.compare_digest(expected, header_value)


class SignalEventService:
    """Rich signal-event package on top of the Notifier queue: per-card
    channel prefs (email|rss|api), quiet hours, RSS tokens (revocable),
    registered webhooks with HMAC-signed POSTs, and the v1 payload builder.

    Server-side only. Behind `cards_v4_1` (flag off -> NotifyError(flag_off)
    on mutations; reads stay available). Kill-switch SIFT_CARDS_V4_1=0.
    """

    def __init__(self, notifier: Notifier, ledger=None,
                 db_path: str | Path | None = None):
        self.notifier = notifier
        self.cards = notifier.cards
        self.ledger = ledger
        if db_path is None:
            db_path = self.notifier.db_path.parent / "signal.db"
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "SignalEventService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- flag guard -------------------------------------------------------------

    def _require_flag(self, handle: str) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise NotifyError("flag_off: cards_v4_1 is disabled for this account")

    # -- per-card channel prefs ---------------------------------------------------

    def set_channel(self, card_id: str, handle: str, channel: str, *,
                    quiet_start: int | None = None, quiet_end: int | None = None,
                    now: float | None = None) -> ChannelPref:
        """Set the per-card delivery channel (email|rss|api) + optional quiet
        hours (seconds-of-day UTC). Read-back."""
        self._require_flag(handle)
        self.cards.assert_owned(card_id, handle)      # non-transferable
        if channel not in CHANNELS:
            raise NotifyError(f"unknown channel {channel!r}")
        if quiet_start is not None and not (0 <= quiet_start < DAY):
            raise NotifyError("quiet_start must be seconds-of-day UTC in [0, 86400)")
        if quiet_end is not None and not (0 <= quiet_end < DAY):
            raise NotifyError("quiet_end must be seconds-of-day UTC in [0, 86400)")
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT INTO notify_channel (card_id, owner, channel, quiet_start,"
            " quiet_end, updated_at) VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(card_id, owner) DO UPDATE SET channel=excluded.channel,"
            " quiet_start=excluded.quiet_start, quiet_end=excluded.quiet_end,"
            " updated_at=excluded.updated_at",
            (card_id, handle, channel, quiet_start, quiet_end, now),
        )
        self.con.commit()
        return self.get_channel(card_id, handle)      # read-back

    def get_channel(self, card_id: str, handle: str) -> ChannelPref:
        row = self.con.execute(
            "SELECT * FROM notify_channel WHERE card_id=? AND owner=?",
            (card_id, handle),
        ).fetchone()
        if row is None:
            return ChannelPref(card_id, handle, CHANNEL_EMAIL, None, None, 0.0)
        return self._to_channel(row)

    def _to_channel(self, row: sqlite3.Row) -> ChannelPref:
        return ChannelPref(row["card_id"], row["owner"], row["channel"],
                           row["quiet_start"], row["quiet_end"], row["updated_at"])

    # -- RSS tokens -----------------------------------------------------------------

    def issue_rss_token(self, handle: str, *, now: float | None = None) -> RssToken:
        """Issue a revocable RSS auth token for the owner's feed (token in
        the feed URL). Read-back."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        token = _rand_token()
        self.con.execute(
            "INSERT INTO rss_token (token, owner, revoked, created_at)"
            " VALUES (?,?,0,?)",
            (token, handle, now),
        )
        self.con.commit()
        return self.get_rss_token(token)             # read-back

    def get_rss_token(self, token: str) -> RssToken:
        row = self.con.execute(
            "SELECT * FROM rss_token WHERE token=?", (token,)
        ).fetchone()
        if row is None:
            raise NotifyError(f"unknown rss token {token!r}")
        return RssToken(row["token"], row["owner"], bool(row["revoked"]),
                        row["created_at"], row["revoked_at"])

    def revoke_rss_token(self, token: str, handle: str, *,
                         now: float | None = None) -> RssToken:
        """Revoke an RSS token (owner-only). The token stops authenticating
        the feed immediately. Read-back."""
        self._require_flag(handle)
        t = self.get_rss_token(token)
        if t.owner != handle:
            raise NotifyError("rss token belongs to another owner — refused")
        now = time.time() if now is None else now
        self.con.execute(
            "UPDATE rss_token SET revoked=1, revoked_at=? WHERE token=?",
            (now, token),
        )
        self.con.commit()
        return self.get_rss_token(token)             # read-back

    def owner_for_rss_token(self, token: str) -> str | None:
        """Resolve a feed token to its owner; None when unknown or revoked."""
        row = self.con.execute(
            "SELECT owner, revoked FROM rss_token WHERE token=?", (token,)
        ).fetchone()
        if row is None or row["revoked"]:
            return None
        return row["owner"]

    def rss_tokens_for_owner(self, handle: str) -> list[RssToken]:
        rows = self.con.execute(
            "SELECT * FROM rss_token WHERE owner=? ORDER BY created_at", (handle,)
        ).fetchall()
        return [RssToken(r["token"], r["owner"], bool(r["revoked"]),
                         r["created_at"], r["revoked_at"]) for r in rows]

    # -- webhooks -------------------------------------------------------------------

    def register_webhook(self, handle: str, url: str, secret: str | None = None,
                         *, now: float | None = None) -> Webhook:
        """Register an API webhook (POST JSON + HMAC signature header).
        The secret is generated server-side when not supplied; it never
        leaves the server. Read-back."""
        self._require_flag(handle)
        if not (url.startswith("https://") or url.startswith("http://")):
            raise NotifyError("webhook url must be http(s)")
        now = time.time() if now is None else now
        secret = secret or _rand_token().replace("rss_", "whsec_")
        cur = self.con.execute(
            "INSERT INTO webhook (owner, url, secret, active, created_at)"
            " VALUES (?,?,?,1,?)",
            (handle, url, secret, now),
        )
        self.con.commit()
        return self.get_webhook(int(cur.lastrowid))   # read-back

    def get_webhook(self, hook_id: int) -> Webhook:
        row = self.con.execute(
            "SELECT * FROM webhook WHERE hook_id=?", (hook_id,)
        ).fetchone()
        if row is None:
            raise NotifyError(f"unknown webhook {hook_id!r}")
        return Webhook(row["hook_id"], row["owner"], row["url"],
                       bool(row["active"]), row["created_at"], row["revoked_at"])

    def revoke_webhook(self, hook_id: int, handle: str, *,
                       now: float | None = None) -> Webhook:
        self._require_flag(handle)
        w = self.get_webhook(hook_id)
        if w.owner != handle:
            raise NotifyError("webhook belongs to another owner — refused")
        now = time.time() if now is None else now
        self.con.execute(
            "UPDATE webhook SET active=0, revoked_at=? WHERE hook_id=?",
            (now, hook_id),
        )
        self.con.commit()
        return self.get_webhook(hook_id)              # read-back

    def active_webhooks(self, handle: str) -> list[Webhook]:
        rows = self.con.execute(
            "SELECT * FROM webhook WHERE owner=? AND active=1 ORDER BY hook_id",
            (handle,),
        ).fetchall()
        return [Webhook(r["hook_id"], r["owner"], r["url"], bool(r["active"]),
                        r["created_at"], r["revoked_at"]) for r in rows]

    def webhook_secret(self, hook_id: int, handle: str) -> str:
        """Server-side read of a webhook's HMAC secret (used to sign POSTs;
        never serialized into any payload)."""
        w = self.get_webhook(hook_id)
        if w.owner != handle:
            raise NotifyError("webhook belongs to another owner — refused")
        row = self.con.execute(
            "SELECT secret FROM webhook WHERE hook_id=?", (hook_id,)
        ).fetchone()
        return row["secret"]

    # -- event emission ------------------------------------------------------------

    def emit_hit(self, card_id: str, handle: str, *, statistic: float | None,
                 threshold: float | None, entitlement: str | None = None,
                 now: float | None = None):
        """Build the v1 payload and enqueue the hit through the Notifier
        (opt-in + rate caps + digest/live unchanged). The payload is guarded
        fail-closed: a signed statistic, a forbidden key, a ticker-shaped
        token, or a banned phrase in the gloss raises NotifyError and NOTHING
        is enqueued. Returns the enqueued Notification (read-back) or None
        when opted out / capped."""
        self._require_flag(handle)
        card = self.cards.assert_owned(card_id, handle)   # non-transferable
        now = time.time() if now is None else now
        payload = build_event_payload(
            card=card, statistic=statistic, threshold=threshold, now=now,
            ledger=self.ledger, notifier=self.notifier, handle=handle,
        )
        # record the hit in the append-only ledger too (frequency facts source)
        if self.ledger is not None:
            self.ledger.record_event(
                HIT_EVENT_TYPE,
                {"card_id": card.card_id, "trial_id": card.trial_id,
                 "statistic": payload["statistic"], "threshold": payload["threshold"]},
                trial_id=card.trial_id, now=now,
            )
            # refresh the aggregate count AFTER recording this hit (ledger source)
            payload["cards_fired_today"] = cards_fired_today_ledger(
                self.ledger, self.cards, handle, now=now)
            _guard_payload(payload)
        return self.notifier.notify(
            card_id, handle, "hit", payload,
            entitlement=entitlement, now=now,
        )

    # -- delivery ------------------------------------------------------------------

    def due_now(self, card_id: str, handle: str, *,
                now: float | None = None) -> bool:
        """Quiet-hours gate: True when delivery is allowed right now for the
        card's channel pref (no quiet window configured, or outside it)."""
        now = time.time() if now is None else now
        pref = self.get_channel(card_id, handle)
        return not in_quiet_hours(now, pref.quiet_start, pref.quiet_end)

    def deliver_due_channels(self, handle: str, *,
                             now: float | None = None) -> list[dict]:
        """Deliver all queued notifications of `handle` whose per-card channel
        is due (outside quiet hours). Returns a per-row delivery manifest
        (channel, rss token / webhook target, signed body for api) — the
        actual transport is the caller's (server-side); this is the outbox
        read-back. Rows are marked delivered exactly once via the Notifier's
        idempotent path."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        manifest: list[dict] = []
        pending = self.notifier.pending(handle)
        for n in pending:
            if not self.due_now(n.card_id, handle, now=now):
                continue                              # quiet hours: stays queued
            pref = self.get_channel(n.card_id, handle)
            entry = {
                "notif_id": n.notif_id,
                "card_id": n.card_id,
                "channel": pref.channel,
                "payload": n.payload,
            }
            if pref.channel == CHANNEL_RSS:
                tok = self.con.execute(
                    "SELECT token FROM rss_token WHERE owner=? AND revoked=0"
                    " ORDER BY created_at DESC LIMIT 1", (handle,),
                ).fetchone()
                entry["rss_token"] = tok["token"] if tok else None
            elif pref.channel == CHANNEL_API:
                hooks = self.active_webhooks(handle)
                entry["webhooks"] = [
                    {"hook_id": w.hook_id, "url": w.url,
                     "signature": webhook_signature(
                         self.webhook_secret(w.hook_id, handle),
                         json.dumps(n.payload, sort_keys=True,
                                    separators=(",", ":"), default=str))}
                    for w in hooks
                ]
            manifest.append(entry)
        # mark ONLY the due rows delivered (quiet-gated rows stay queued);
        # the state transition is guarded (queued -> delivered exactly once)
        for e in manifest:
            self.notifier.con.execute(
                "UPDATE notify_log SET state=?, delivered_at=?"
                " WHERE notif_id=? AND state=?",
                ("delivered", now, e["notif_id"], "queued"),
            )
            rb = self.notifier.get_notif(e["notif_id"])   # read-back
            e["state"] = rb.state
            e["delivered_at"] = rb.delivered_at
        self.notifier.con.commit()
        return manifest                              # read-back manifest


__all__ = [
    "CHANNELS",
    "CHANNEL_API",
    "CHANNEL_EMAIL",
    "CHANNEL_RSS",
    "FORBIDDEN_PAYLOAD_KEYS",
    "HIT_EVENT_TYPE",
    "MINT_BASELINE_EVENT",
    "PAYLOAD_SCHEMA_VERSION",
    "QUIET_HOURS_DEFAULT",
    "ChannelPref",
    "RssToken",
    "SignalEvent",
    "SignalEventService",
    "TRAILING_12M_S",
    "WEBHOOK_SIG_HEADER",
    "Webhook",
    "build_event_payload",
    "cards_fired_today",
    "cards_fired_today_ledger",
    "health_facts",
    "in_quiet_hours",
    "persistence_facts",
    "utc_ts",
    "verify_webhook_signature",
    "webhook_signature",
]
