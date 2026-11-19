"""v4.2 deck layer — bucket registry + deck ledger (spec v4.2 §4, #166).

Two kinds of decks, both behind the `decks_v4_2` flag (which also requires
`cards_v4_1` — see sift/flags.py):

- RANKED decks: deterministic membership from the fixed, enumerable bucket
  grid — card-count {5,10,20,50,100} x optional GICS sector split ("all" +
  each sector) x family mix (pairs, lead-lag; mixing allowed). Membership is
  top-K by card tier within the bucket, K = bucket size, fixed in advance.
  All buckets are FREE-TIER (not pay-gated). Ranked decks are the only decks
  that can receive an F-S rank (see deck_rank) and appear in the deck
  ranking.
- CUSTOM hand-picked decks: allowed as a personal sandbox; the S6 score is
  computed and shown back to that user as INFORMATIONAL ONLY (amber,
  unranked, unshareable) because arbitrary C(N,k) selections enable
  try-until-it-passes.

Dedupe on (scope, bucket config, vintage) like cards: re-registering the
identical key returns the existing deck (skip-list semantics) and the deck
ledger counts it exactly once — the cross-user pool strengthens the sample
instead of inflating the trial count.

Ledger: append-only deck trials + deck events in their own SQLite file
(like the card registry), with read-back on every write (hard rule: never
claim success without a read-back). Decks are non-transferable (owner_handle
on the row; assert_owned raises on a foreign handle).

Hard rules (v3 §12 + v4.2 §5): no tickers/direction/P&L in any artifact —
deck membership is stored as CARD IDs (HMAC commitments, never symbols);
sector is a GICS bucket label; the deck face/copy layer guards display.
Raw market data stays server-side. Flag off -> DeckError(flag_off),
fail-closed; kill-switch SIFT_DECKS_V4_2=0 (plus SIFT_CARDS_V4_1=0).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import flags
from sift.gauntlet import config as gcfg

# Deck ranking F D C B A S — DISTINCT from card tiers (bronze/silver/gold).
# Evidence strength of the PORTFOLIO claim. Beta: deck results are amber
# telemetry only, never performance claims (the rank is shown with the
# telemetry caveat; see sift/decks/copy.py).
DECK_RANKS = ("F", "D", "C", "B", "A", "S")   # index = strength
DECK_KINDS = ("ranked", "custom")               # custom = informational only

# GICS sector buckets for the optional sector split (labels only, no tickers).
GICS_SECTORS = (
    "energy", "materials", "industrials", "consumer_discretionary",
    "consumer_staples", "health_care", "financials", "information_technology",
    "communication_services", "utilities", "real_estate",
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS deck (
    deck_id            TEXT PRIMARY KEY,          -- HMAC commitment, public id
    canonical_config   TEXT NOT NULL,             -- canonical JSON of the bucket config
    config_hash        TEXT NOT NULL,             -- sha256 of canonical_config (internal dedupe key)
    scope_id           TEXT NOT NULL,
    size               INTEGER NOT NULL,          -- bucket size K (member cards)
    sector             TEXT NOT NULL,             -- GICS bucket label or "all"
    family_mix         TEXT NOT NULL,             -- canonical JSON list, e.g. ["lead_lag","pairs"]
    kind               TEXT NOT NULL DEFAULT 'ranked',   -- ranked|custom
    member_card_ids    TEXT NOT NULL,             -- canonical JSON list of card ids (HMAC ids, no symbols)
    rank               TEXT,                      -- F|D|C|B|A|S for ranked decks; NULL for custom
    owner_handle       TEXT NOT NULL,             -- non-transferable
    data_vintage_hash  TEXT NOT NULL,
    s6_pass            INTEGER,                   -- NULL until S6 run; 1 pass / 0 fail
    s6_detail_json     TEXT,
    created_at         REAL NOT NULL,
    UNIQUE (scope_id, config_hash, data_vintage_hash)
);
CREATE INDEX IF NOT EXISTS idx_deck_scope ON deck(scope_id);
CREATE INDEX IF NOT EXISTS idx_deck_owner ON deck(owner_handle);
CREATE INDEX IF NOT EXISTS idx_deck_kind ON deck(kind);
CREATE TABLE IF NOT EXISTS deck_event (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,   -- append-only; rows never deleted
    deck_id      TEXT REFERENCES deck(deck_id),
    event_type   TEXT NOT NULL,                        -- e.g. 'deck_trial', 'deck_rank', 'deck_kill'
    payload_json TEXT NOT NULL,                        -- canonical JSON, immutable
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deck_event_type ON deck_event(event_type);
"""


class DeckError(Exception):
    """Deck-registry misuse (unknown deck, flag off, transfer attempt, ...)."""


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def bucket_config(*, size: int, sector: str = gcfg.DECK_SECTOR_ALL,
                  family_mix=("pairs",)) -> dict:
    """One bucket of the fixed grid. `size` in DECK_BUCKET_SIZES, `sector`
    in ("all",) + GICS_SECTORS, family_mix a non-empty subset of DECK_FAMILIES
    (mixing allowed). Validated fail-closed: anything outside the grid raises
    DeckError — no free-form sizes."""
    if size not in gcfg.DECK_BUCKET_SIZES:
        raise DeckError(f"bucket size {size!r} not in the fixed grid {gcfg.DECK_BUCKET_SIZES}")
    if sector != gcfg.DECK_SECTOR_ALL and sector not in GICS_SECTORS:
        raise DeckError(f"sector {sector!r} not in ('all', GICS_SECTORS)")
    fams = tuple(sorted(set(family_mix)))
    if not fams or any(f not in gcfg.DECK_FAMILIES for f in fams):
        raise DeckError(f"family_mix {family_mix!r} must be a non-empty subset of {gcfg.DECK_FAMILIES}")
    return {"size": int(size), "sector": sector, "family_mix": list(fams)}


def enumerate_buckets() -> list[dict]:
    """The FULL fixed bucket grid: sizes x sectors x family mixes.

    Family mixes enumerated: the single-family buckets plus the mixed
    bucket (all families) — the grid is finite and enumerable, so the
    registry can enumerate every ranked deck config without free-form
    sizes. All buckets are free-tier."""
    sectors = (gcfg.DECK_SECTOR_ALL,) + GICS_SECTORS
    mixes = [(f,) for f in gcfg.DECK_FAMILIES] + [tuple(sorted(gcfg.DECK_FAMILIES))]
    out = []
    for size in gcfg.DECK_BUCKET_SIZES:
        for sector in sectors:
            for mix in mixes:
                out.append(bucket_config(size=size, sector=sector, family_mix=mix))
    return out


def config_hash_of(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def deck_dedupe_key(scope_id: str, config: dict, data_vintage_hash: str) -> str:
    """Dedupe on (scope, bucket config, vintage) like cards."""
    blob = canonical_json({
        "scope_id": scope_id,
        "config_hash": config_hash_of(config),
        "data_vintage_hash": data_vintage_hash,
    })
    return hashlib.sha256(blob.encode()).hexdigest()


def _default_secret() -> bytes:
    """Server-side HMAC secret for deck ids (same posture as cards)."""
    env = os.environ.get("SIFT_DECK_HMAC_SECRET")
    if env:
        return env.encode()
    envf = os.environ.get("SIFT_CARD_HMAC_SECRET")
    if envf:
        return envf.encode()
    path = Path(os.environ.get("SIFT_CARD_HMAC_SECRET_FILE", ".card-hmac-secret"))
    if path.exists():
        return path.read_bytes().strip()
    if os.environ.get("SIFT_ENV", "dev") == "production":
        raise RuntimeError(
            "no deck HMAC secret configured in production: set SIFT_DECK_HMAC_SECRET"
            " or SIFT_CARD_HMAC_SECRET — refusing to mint with an ephemeral secret"
        )
    import sys
    print("WARNING: sift decks using ephemeral dev HMAC secret "
          "(set SIFT_DECK_HMAC_SECRET for real deployments)", file=sys.stderr)
    return b"sift-dev-ephemeral-secret-not-for-production"


@dataclass(frozen=True)
class Deck:
    deck_id: str
    scope_id: str
    config_hash: str
    size: int
    sector: str
    family_mix: str
    kind: str
    member_card_ids: str
    rank: str | None
    owner_handle: str
    data_vintage_hash: str
    s6_pass: int | None
    created_at: float


def deck_rank_from_stats(dsr: float | None, t_net: float | None,
                         p_perm_corr: float | None, pbo: float | None,
                         min_trl_ok: bool = True) -> str:
    """Deck ranking F D C B A S — DISTINCT from card tiers. Pure function of
    the pre-registered, ledger-recorded S6 stats only (never payment, never
    chance). Missing stats never inflate: a None component fails its check.

    - S:  DSR >= 0.99 AND |t_net| >= 4.0 AND p_corr <= 0.01 AND PBO <= 0.5
    - A:  DSR >= 0.97 AND |t_net| >= 3.5 AND p_corr <= 0.02 AND PBO <= 0.5
    - B:  DSR >= 0.95 AND |t_net| >= tau AND p_corr <= 0.05 AND PBO <= 0.5
    - C:  S6 pass (net SR > 0, |t_net| >= tau, MinTRL <= T, PBO <= 0.5)
    - D:  S6 pass on the research window only (amber telemetry)
    - F:  no pass (or any missing stat)
    Beta: the rank is amber telemetry only — never a performance claim.
    """
    if dsr is None or t_net is None or p_perm_corr is None or pbo is None:
        return "F"
    if not min_trl_ok or pbo > gcfg.S6_PBO_MAX:
        return "F"
    if dsr >= 0.99 and abs(t_net) >= 4.0 and p_perm_corr <= 0.01:
        return "S"
    if dsr >= 0.97 and abs(t_net) >= 3.5 and p_perm_corr <= 0.02:
        return "A"
    if dsr >= 0.95 and abs(t_net) >= 3.0 and p_perm_corr <= 0.05:
        return "B"
    # C/D split is decided by the caller (production vs research window);
    # the pure function returns C for any pass below B.
    return "C"


class DeckRegistry:
    """Append-only deck registry + deck-trial ledger. Server-side only."""

    def __init__(self, db_path: str | Path, secret: bytes | None = None):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()
        self._secret = secret if secret is not None else _default_secret()

    def __enter__(self) -> "DeckRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- flag -----------------------------------------------------------------
    def _require_flag(self, handle: str | None = None) -> None:
        if not flags.decks_v4_2_enabled(handle):
            raise DeckError(
                "flag_off: decks_v4_2 is disabled (kill-switch SIFT_DECKS_V4_2=0"
                " or SIFT_CARDS_V4_1=0) — the deck layer is inert"
            )

    # -- id commitment ----------------------------------------------------------
    def deck_id_for(self, scope_id: str, config: dict, data_vintage_hash: str) -> str:
        """Public deck id = HMAC-SHA256(secret, canonical(scope, config, vintage)).
        Unforgeable without the server secret; unique per canonical deck config."""
        blob = canonical_json({"scope_id": scope_id, "config": config,
                               "vintage": data_vintage_hash})
        mac = hmac.new(self._secret, blob.encode(), hashlib.sha256)
        return "deck-" + mac.hexdigest()

    # -- registration -------------------------------------------------------------
    def register_ranked(self, *, scope_id: str, config: dict,
                        member_card_ids: list[str], owner_handle: str,
                        data_vintage_hash: str, now: float | None = None) -> Deck:
        """Register a RANKED deck (deterministic membership, top-K by card tier
        in bucket, K = bucket size). Dedupe on (scope, bucket config, vintage):
        the identical key returns the existing deck (skip-list semantics) and
        the ledger counts it exactly once. Read-back before returning."""
        self._require_flag(owner_handle)
        config = bucket_config(size=config["size"], sector=config.get("sector", gcfg.DECK_SECTOR_ALL),
                               family_mix=config.get("family_mix", ("pairs",)))
        K = config["size"]
        if len(member_card_ids) != K:
            raise DeckError(f"ranked deck membership must be exactly K={K} cards, got {len(member_card_ids)}")
        return self._register(kind="ranked", scope_id=scope_id, config=config,
                              member_card_ids=member_card_ids,
                              owner_handle=owner_handle,
                              data_vintage_hash=data_vintage_hash, now=now)

    def register_custom(self, *, scope_id: str, member_card_ids: list[str],
                        owner_handle: str, data_vintage_hash: str,
                        now: float | None = None) -> Deck:
        """Register a CUSTOM hand-picked deck: INFORMATIONAL ONLY (amber,
        unranked — rank stays NULL, unshareable). Membership is arbitrary
        (any size), which is exactly why it can never be ranked: arbitrary
        C(N,k) selections enable try-until-it-passes. Read-back before
        returning."""
        self._require_flag(owner_handle)
        if not member_card_ids:
            raise DeckError("custom deck needs at least one card")
        config = {"size": len(member_card_ids), "sector": gcfg.DECK_SECTOR_ALL,
                  "family_mix": ["custom"], "custom": True}
        return self._register(kind="custom", scope_id=scope_id, config=config,
                              member_card_ids=member_card_ids,
                              owner_handle=owner_handle,
                              data_vintage_hash=data_vintage_hash, now=now)

    def _register(self, *, kind: str, scope_id: str, config: dict,
                  member_card_ids: list[str], owner_handle: str,
                  data_vintage_hash: str, now: float | None = None) -> Deck:
        ch = config_hash_of(config)
        key = deck_dedupe_key(scope_id, config, data_vintage_hash)
        existing = self.con.execute(
            "SELECT * FROM deck WHERE scope_id=? AND config_hash=? AND data_vintage_hash=?",
            (scope_id, ch, data_vintage_hash),
        ).fetchone()
        if existing is not None:
            # dedupe/cached: same (scope, bucket config, vintage) -> same deck
            return self.get_deck(existing["deck_id"])   # read-back
        deck_id = self.deck_id_for(scope_id, config, data_vintage_hash)
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT INTO deck (deck_id, canonical_config, config_hash, scope_id,"
            " size, sector, family_mix, kind, member_card_ids, rank, owner_handle,"
            " data_vintage_hash, s6_pass, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (deck_id, canonical_json(config), ch, scope_id, int(config["size"]),
             config.get("sector", gcfg.DECK_SECTOR_ALL),
             canonical_json(config.get("family_mix", [])), kind,
             canonical_json(sorted(member_card_ids)),
             None if kind == "custom" else "F",   # ranked starts unranked-at-F until S6 sets it
             owner_handle, data_vintage_hash, None, now),
        )
        self.con.commit()
        self.record_event(deck_id, "deck_register",
                          {"kind": kind, "scope_id": scope_id, "config": config,
                           "dedupe_key": key, "K": len(member_card_ids)},
                          now=now)
        return self.get_deck(deck_id)               # read-back

    # -- S6 trial recording ---------------------------------------------------------
    def record_s6(self, deck_id: str, *, passed: bool, detail: dict,
                  handle: str | None = None, now: float | None = None) -> Deck:
        """Record the S6 gauntlet result for a deck (append-only event +
        one-shot column update). Custom decks may be scored but stay
        informational: rank is NEVER set for kind='custom'. Read-back before
        returning."""
        self._require_flag(handle)
        deck = self.get_deck(deck_id)
        now = time.time() if now is None else now
        self.con.execute(
            "UPDATE deck SET s6_pass=?, s6_detail_json=? WHERE deck_id=?",
            (1 if passed else 0, canonical_json(detail), deck_id),
        )
        self.con.commit()
        rank = None
        if deck.kind == "ranked" and passed:
            rank = deck_rank_from_stats(
                detail.get("dsr"), detail.get("t_net"),
                detail.get("p_perm_corr", detail.get("p_perm")), detail.get("pbo"),
                min_trl_ok=bool(detail.get("min_trl_ok", True)),
            )
            if rank == "F":
                # passed the S6 bar but below the B evidence bar -> C
                rank = "C"
            self.con.execute("UPDATE deck SET rank=? WHERE deck_id=?", (rank, deck_id))
            self.con.commit()
        self.record_event(
            deck_id, "deck_trial",
            {"passed": bool(passed), "rank": rank, "kind": deck.kind,
             "informational_only": deck.kind == "custom", "detail": detail},
            now=now,
        )
        return self.get_deck(deck_id)               # read-back

    # -- events (append-only, read-back) -------------------------------------------
    def record_event(self, deck_id: str, event_type: str, payload: dict,
                     now: float | None = None) -> int:
        """Append a deck event (deck_register, deck_trial, deck_rank,
        deck_kill, ...). Append-only: never updated or deleted. Returns
        event_id."""
        now = time.time() if now is None else now
        cur = self.con.execute(
            "INSERT INTO deck_event (deck_id, event_type, payload_json, created_at)"
            " VALUES (?,?,?,?)",
            (deck_id, event_type, canonical_json(payload), now),
        )
        self.con.commit()
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def events(self, event_type: str | None = None,
               deck_id: str | None = None) -> list[sqlite3.Row]:
        """Read append-only deck events, oldest first (read-back surface)."""
        q = "SELECT * FROM deck_event"
        clauses, args = [], []
        if event_type is not None:
            clauses.append("event_type=?")
            args.append(event_type)
        if deck_id is not None:
            clauses.append("deck_id=?")
            args.append(deck_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY event_id"
        return self.con.execute(q, args).fetchall()

    # -- reads ----------------------------------------------------------------
    def get_deck(self, deck_id: str) -> Deck:
        """Read-back of one deck. Raises on unknown id."""
        row = self.con.execute("SELECT * FROM deck WHERE deck_id=?", (deck_id,)).fetchone()
        if row is None:
            raise DeckError(f"unknown deck_id {deck_id!r}")
        return self._to_deck(row)

    def decks_for_owner(self, handle: str) -> list[Deck]:
        rows = self.con.execute(
            "SELECT * FROM deck WHERE owner_handle=? ORDER BY created_at, deck_id",
            (handle,),
        ).fetchall()
        return [self._to_deck(r) for r in rows]

    def all_decks(self) -> list[Deck]:
        rows = self.con.execute("SELECT * FROM deck ORDER BY created_at, deck_id").fetchall()
        return [self._to_deck(r) for r in rows]

    def decks_in_scope(self, scope_id: str) -> list[Deck]:
        """All registered deck trials in the scope across ALL users (kills
        included) — the N for the deck-stratum multiplicity correction."""
        rows = self.con.execute(
            "SELECT * FROM deck WHERE scope_id=? ORDER BY created_at, deck_id",
            (scope_id,),
        ).fetchall()
        return [self._to_deck(r) for r in rows]

    def stratum_n(self, scope_id: str) -> int:
        """N = count of all registered deck trials in the scope (all users,
        kills included)."""
        return self.con.execute(
            "SELECT COUNT(*) FROM deck WHERE scope_id=?", (scope_id,)
        ).fetchone()[0]

    def _to_deck(self, row: sqlite3.Row) -> Deck:
        return Deck(
            deck_id=row["deck_id"], scope_id=row["scope_id"],
            config_hash=row["config_hash"], size=row["size"],
            sector=row["sector"], family_mix=row["family_mix"],
            kind=row["kind"], member_card_ids=row["member_card_ids"],
            rank=row["rank"], owner_handle=row["owner_handle"],
            data_vintage_hash=row["data_vintage_hash"],
            s6_pass=row["s6_pass"], created_at=row["created_at"],
        )

    def assert_owned(self, deck_id: str, handle: str) -> Deck:
        """Non-transferability guard: decks never change owner."""
        deck = self.get_deck(deck_id)
        if deck.owner_handle != handle:
            raise DeckError(
                f"deck {deck_id} belongs to {deck.owner_handle!r}, not {handle!r} —"
                " decks are non-transferable"
            )
        return deck


__all__ = [
    "DECK_KINDS",
    "DECK_RANKS",
    "Deck",
    "DeckError",
    "DeckRegistry",
    "GICS_SECTORS",
    "bucket_config",
    "canonical_json",
    "config_hash_of",
    "deck_dedupe_key",
    "deck_rank_from_stats",
    "enumerate_buckets",
]
