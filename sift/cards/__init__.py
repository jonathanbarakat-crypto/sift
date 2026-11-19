"""v4.1 card registry: card minting on gauntlet survivors (mechanics item 1).

Design (v4.1 mechanics 1, behind the `cards_v4_1` flag):

- A card is minted ONLY when a trial reaches verdict=survivor AND its
  survivor_class is `validated-green` (S5 registry-depth pass). Killed
  trials and on-watch-amber survivors (the F-REGIME class — edge that does
  not survive full depth) mint nothing. F-DELIST runs are kill/amber paths
  too: nothing mints. A planted clean survivor (F-PAIR-style, full-window
  validated) mints exactly one card.
- The public card id is an HMAC commitment (server-side secret) of the
  canonical card spec (family, scope_id, config_hash, data_vintage_hash,
  trial_id, survivor_class) — `card-<hex>`. HMAC makes ids unforgeable and
  unique per canonical spec without revealing the spec; the secret never
  leaves the server (never logged, never committed, env var or file).
- Tier comes from the pre-registered RAW stats recorded in the ledger stage
  rows (S2 p_perm, S4 bar, S5 full-depth t_net) — see `tier_from_stats`. The
  mapping is a pure function of recorded, N-INDEPENDENT quantities only:
  the stratum-deflated DSR is NOT an input to the tier (owner decision
  2026-11-18, hermes-tasks #170); it is the MINT-TIME ADMISSION GATE only
  (CardRegistry.mint caps a sub-gate candidate at bronze) and is stored on
  the card as mint-time context (`mint_dsr` / `mint_sr_star` /
  `mint_n_stratum`, labeled "admission evidence at mint-time N").
- The low-level `decay()` primitive is downgrade-only: it may only move a
  tier DOWN (gold -> silver -> bronze), never up, and never re-randomizes.
  The BIDIRECTIONAL tier policy lives one layer up, in the registry decay
  job (sift/cards/decay.py): upgrades are possible ONLY through the
  pre-registered live-window rules there (fully out-of-sample trailing
  12m window, two consecutive qualifying annual windows at the mint-time
  bar, 90-day rate limit, seeded re-evaluation, append-only
  `card_upgrade` ledger event) — never by payment or chance. The registry
  exposes `set_tier()` for that gated path only; `decay()` stays down-only.
- Duplicate spec (same canonical spec minted again, e.g. a re-run that
  hits the skip-list and re-survives) -> a GHOST card: kind="ghost",
  numbered FIFO per canonical spec (ghost_no). Ghost handling proper
  (vault shifting) lives in sift/cards/vault.py (mechanics 2-3); this
  module owns minting + id commitment + tier + decay.
- Cards are non-transferable: the record carries owner_handle and any
  transfer attempt raises (mechanics hard rule).
- Read-back discipline (hard rule): after every mint, the card is read
  back from the registry (`get_card`) before success is claimed; the
  minter returns the read-back row, not just the id.
- The MINT-TIME HOLDOUT BASELINE (frozen-holdout t_net / annualized Sharpe /
  holdout bars) is persisted on the card row at mint (soundness #10,
  hermes-tasks #181): `baseline_t_net` / `baseline_sr_ann` /
  `baseline_holdout_bars`, NULL when the mint run had no holdout — never
  fabricated. Display/health only; never an input to tier, decay, or
  upgrade decisions.

Storage: its own SQLite file (like the trial ledger) with an append-only
card table + a ghost-counter table. Never destroyed, never randomized
(mechanics 2 applies to cards as registry records).
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

CARD_KINDS = ("regular", "ghost")
# v4.2 §3 (owner decision 2026-11-18, hermes-tasks #170): index = strength;
# decay moves down. Tiers are computed from RAW, N-INDEPENDENT stats only
# (tier_from_stats) — DSR is the mint-time admission gate, not a tier input.
# platinum = S4 bar + S5 pass + full-depth |t_net| >= 4.0 + p_perm <= 0.01
# (the pull/31 gold bar minus DSR); gold = S4 bar + p_perm <= 0.01 + full-depth
# |t_net| >= 3.0; silver = full-depth |t_net| >= 3.0 + p_perm <= 0.05;
# bronze = S0-S2 pass.
CARD_TIERS = ("bronze", "silver", "gold", "platinum")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS card (
    card_id            TEXT PRIMARY KEY,          -- HMAC commitment, public id
    canonical_spec     TEXT NOT NULL,             -- canonical JSON of the spec
    spec_hash          TEXT NOT NULL,             -- sha256 of canonical_spec (internal dedupe key)
    trial_id           TEXT NOT NULL,
    family             TEXT NOT NULL,
    scope_id           TEXT NOT NULL,
    survivor_class     TEXT NOT NULL,
    kind               TEXT NOT NULL DEFAULT 'regular',   -- regular|ghost
    ghost_no           INTEGER,                   -- FIFO slot per spec_hash (ghosts only)
    tier               TEXT NOT NULL,
    tier_updated_at    REAL NOT NULL,
    minted_bar         INTEGER NOT NULL DEFAULT 0,  -- bar index at mint (live-window anchor)
    owner_handle       TEXT NOT NULL,             -- non-transferable
    minted_at          REAL NOT NULL,
    decayed_from       TEXT,                      -- tier before last downgrade
    -- v4.2 §3: admission evidence at mint-time N (CONTEXT only — never an
    -- input to decay/re-evaluation; the tier is raw-stats only):
    mint_dsr           REAL,                      -- DSR at mint-time stratum N
    mint_sr_star       REAL,                      -- expected max Sharpe SR* at mint-time N
    mint_n_stratum     INTEGER,                   -- stratum N at mint
    -- soundness #10 (hermes-tasks #181): the MINT-TIME HOLDOUT BASELINE,
    -- persisted per-card at mint (the frozen-holdout live-baseline). All
    -- three are NULL when the run had no holdout (holdout_bars=0) — a
    -- missing baseline is recorded as NULL, never fabricated. Display/health
    -- only; never an input to tier, decay, or upgrade decisions.
    baseline_t_net        REAL,                   -- frozen-holdout full-window t_net
    baseline_sr_ann       REAL,                   -- frozen-holdout annualized Sharpe
    baseline_holdout_bars INTEGER,                -- holdout length in bars at mint
    UNIQUE (spec_hash, kind, ghost_no)
);
CREATE INDEX IF NOT EXISTS idx_card_spec ON card(spec_hash);
CREATE INDEX IF NOT EXISTS idx_card_owner ON card(owner_handle);
CREATE TABLE IF NOT EXISTS spec_ghost_counter (
    spec_hash   TEXT PRIMARY KEY,
    next_no     INTEGER NOT NULL DEFAULT 1
);
"""


class CardError(Exception):
    """Raised on card-registry misuse (unknown card, transfer attempt, ...)."""


def canonical_spec_json(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash_of(spec: dict) -> str:
    return hashlib.sha256(canonical_spec_json(spec).encode()).hexdigest()


def _default_secret() -> bytes:
    """Server-side HMAC secret. Env var first (deployment), else a local
    dev secret file (gitignored pattern: *.card-secret), else an ephemeral
    in-memory secret (dev/test only — ids won't survive a restart, and a
    loud warning is emitted so production never silently runs on it; tests
    set SIFT_CARD_HMAC_SECRET explicitly). Never logged, never committed."""
    env = os.environ.get("SIFT_CARD_HMAC_SECRET")
    if env:
        return env.encode()
    path = Path(os.environ.get("SIFT_CARD_HMAC_SECRET_FILE", ".card-hmac-secret"))
    if path.exists():
        return path.read_bytes().strip()
    if os.environ.get("SIFT_ENV", "dev") == "production":
        raise RuntimeError(
            "no card HMAC secret configured in production: set SIFT_CARD_HMAC_SECRET"
            " or SIFT_CARD_HMAC_SECRET_FILE — refusing to mint with an ephemeral secret"
        )
    import sys
    print("WARNING: sift cards using ephemeral dev HMAC secret "
          "(set SIFT_CARD_HMAC_SECRET for real deployments)", file=sys.stderr)
    return b"sift-dev-ephemeral-secret-not-for-production"


@dataclass(frozen=True)
class Card:
    card_id: str
    spec_hash: str
    trial_id: str
    family: str
    scope_id: str
    survivor_class: str
    kind: str
    ghost_no: int | None
    tier: str
    owner_handle: str
    minted_at: float
    minted_bar: int
    decayed_from: str | None
    # v4.2 §3: admission evidence at mint-time N, recorded as CONTEXT only
    # (dsr / sr_star / n_stratum of the mint-time stratum). Decay and
    # re-evaluation NEVER re-deflate by a current N — the tier is a function
    # of raw, N-independent stats only (tier_from_stats).
    mint_dsr: float | None = None
    mint_sr_star: float | None = None
    mint_n_stratum: int | None = None
    # soundness #10: the mint-time holdout baseline (frozen-holdout t_net /
    # annualized Sharpe / holdout bars), persisted at mint. None when the
    # mint run had no holdout — never fabricated. Display/health only.
    baseline_t_net: float | None = None
    baseline_sr_ann: float | None = None
    baseline_holdout_bars: int | None = None


def tier_from_stats(dsr: float | None, t_net_full: float | None,
                    p_perm: float | None, *, s4_pass: bool = False,
                    s5_pass: bool = False) -> str:
    """v4.2 §3 (owner decision 2026-11-18): tier from RAW, N-INDEPENDENT,
    pre-registered, ledger-recorded stats ONLY. `dsr` is accepted for
    signature compatibility but is NO LONGER an input to the tier — DSR is
    deflated by the stratum N (all trials in scope), so using it here made a
    card's label drift when OTHER users mined more. DSR's only remaining
    role is the MINT-TIME ADMISSION GATE (see sift.gauntlet.v42
    .mint_admission_gate and CardRegistry.mint), enforced separately.

    Bars (raw stats only; identical numeric thresholds where they map 1:1
    to the pre-#170 pull/31 bars):
    - platinum: s4_pass AND s5_pass AND full-depth |t_net| >= 4.0 AND
      p_perm <= 0.01   (the pull/31 gold bar minus DSR, plus S5 pass)
    - gold:     s4_pass AND p_perm <= 0.01 AND full-depth |t_net| >= 3.0
      (the S4 evidence bar plus the full-depth line)
    - silver:   full-depth |t_net| >= 3.0 AND p_perm <= 0.05
    - bronze:   S0-S2 pass (the minimum card tier) — anything below silver

    Missing stats never inflate: a None component fails its check.
    `dsr` is deliberately UNUSED in the decision — a different N changes
    DSR but never the tier.
    """
    if t_net_full is None or p_perm is None:
        return "bronze"
    if (s4_pass and s5_pass and abs(t_net_full) >= 4.0
            and p_perm <= 0.01):
        return "platinum"
    if s4_pass and p_perm <= 0.01 and abs(t_net_full) >= 3.0:
        return "gold"
    if abs(t_net_full) >= 3.0 and p_perm <= 0.05:
        return "silver"
    return "bronze"


def decay_tier(current: str, new: str) -> str:
    """Downgrade-only decay: the result is min(strength) — never an
    upgrade. Raises if `new` would be an upgrade (programmer error)."""
    if current not in CARD_TIERS or new not in CARD_TIERS:
        raise CardError(f"invalid tier {current!r} -> {new!r}")
    if CARD_TIERS.index(new) > CARD_TIERS.index(current):
        raise CardError(f"decay may only downgrade, got {current} -> {new}")
    return new


class CardRegistry:
    """Append-only card registry. The only writer is the server-side
    engine (mint on survivor) and the server-side decay job."""

    def __init__(self, db_path: str | Path, secret: bytes | None = None):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(_SCHEMA_SQL)
        # migration: older card DBs predate the minted_bar anchor column
        cols = {r[1] for r in self.con.execute("PRAGMA table_info(card)").fetchall()}
        if "minted_bar" not in cols:
            self.con.execute(
                "ALTER TABLE card ADD COLUMN minted_bar INTEGER NOT NULL DEFAULT 0"
            )
        # v4.2 §3 migration: mint-time admission-evidence context columns
        for col, decl in (("mint_dsr", "REAL"), ("mint_sr_star", "REAL"),
                          ("mint_n_stratum", "INTEGER")):
            if col not in cols:
                self.con.execute(f"ALTER TABLE card ADD COLUMN {col} {decl}")
        # soundness #10 migration: mint-time holdout baseline columns
        # (additive, idempotent — same pattern as minted_bar above)
        for col, decl in (("baseline_t_net", "REAL"),
                          ("baseline_sr_ann", "REAL"),
                          ("baseline_holdout_bars", "INTEGER")):
            if col not in cols:
                self.con.execute(f"ALTER TABLE card ADD COLUMN {col} {decl}")
        self.con.commit()
        self._secret = secret if secret is not None else _default_secret()

    def __enter__(self) -> "CardRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- id commitment ------------------------------------------------------

    def card_id_for(self, spec: dict) -> str:
        """Public card id = HMAC-SHA256(secret, canonical_spec), hex, prefixed.
        Unforgeable without the server secret; unique per canonical spec."""
        mac = hmac.new(self._secret, canonical_spec_json(spec).encode(), hashlib.sha256)
        return "card-" + mac.hexdigest()

    # -- minting --------------------------------------------------------------

    def mint(self, *, trial_id: str, family: str, scope_id: str,
             config_hash: str, data_vintage_hash: str,
             survivor_class: str, owner_handle: str,
             dsr: float | None, t_net_full: float | None,
             p_perm: float | None, now: float | None = None,
             minted_bar: int = 0, s4_pass: bool = False,
             s5_pass: bool = False,
             sr_star: float | None = None,
             n_stratum: int | None = None,
             baseline_t_net: float | None = None,
             baseline_sr_ann: float | None = None,
             baseline_holdout_bars: int | None = None) -> Card:
        """Mint a card for a gauntlet survivor. v4.2 §2-§3 (owner decision
        2026-11-18):

        - v4.2: ANY S0-S2 passer mints (bronze+). `survivor_class` may be
          'validated-green', 'on-watch-amber', or the v4.2 evidence classes
          'on-watch-amber-s4-evidence' / 'on-watch-amber-s3-evidence'; only
          verdict='killed' (survivor_class='none') refuses;
        - MINT-TIME ADMISSION GATE: `dsr` (DSR at mint-time N) is checked
          HERE and only here — a candidate that does not clear DSR >= 0.95
          at mint-time N mints at most BRONZE, even with strong raw stats
          (sift.gauntlet.v42.mint_admission_gate). The tier itself is
          computed from the raw, N-independent stats only
          (tier_from_stats), so the label never drifts when the pool grows;
        - `dsr`, `sr_star`, `n_stratum` are recorded on the card as
          "admission evidence at mint-time N" context — decay/re-evaluation
          NEVER re-deflates by a current N;
        - duplicate canonical spec -> ghost (kind='ghost', FIFO ghost_no);
        - `minted_bar` anchors the card's age in live bars for the
          pre-registered live-window upgrade rules (sift/cards/decay.py);
        - `s4_pass` / `s5_pass` carry the S4 score bar and the S5 full-depth
          pass into tier_from_stats (the raw evidence vector, v4.2 §3);
        - soundness #10: `baseline_t_net` / `baseline_sr_ann` /
          `baseline_holdout_bars` persist the MINT-TIME HOLDOUT BASELINE (the
          frozen-holdout live-baseline) on the card row. None when the mint
          run had no holdout (holdout_bars=0) — never fabricated. Display /
          health only; never an input to tier, decay, or upgrade.
        - read-back: the returned Card is re-read from the DB (hard rule:
          never claim success without read-back).
        """
        # v4.2: killed trials (survivor_class == 'none') mint nothing; every
        # S0-S2 passer mints at the tier its evidence supports (bronze min).
        from sift.gauntlet.v42 import SURVIVOR_CLASSES_V42
        if survivor_class not in SURVIVOR_CLASSES_V42 or survivor_class == "none":
            raise CardError(
                f"only gauntlet survivors (S0-S2 pass) mint cards, got {survivor_class!r}"
            )
        spec = {
            "family": family,
            "scope_id": scope_id,
            "config_hash": config_hash,
            "data_vintage_hash": data_vintage_hash,
            "trial_id": trial_id,
            "survivor_class": survivor_class,
        }
        sh = spec_hash_of(spec)
        card_id = self.card_id_for(spec)
        now = time.time() if now is None else now
        # v4.2 §3: tier from RAW, N-independent stats only (dsr deliberately
        # not passed into the decision); the mint-time DSR admission gate is
        # enforced separately and caps a sub-gate candidate at bronze.
        from sift.gauntlet.v42 import mint_admission_gate
        tier = tier_from_stats(dsr, t_net_full, p_perm,
                               s4_pass=s4_pass, s5_pass=s5_pass)
        if not mint_admission_gate(dsr):
            tier = "bronze"   # DSR < 0.95 at mint-time N -> max bronze

        existing = self.con.execute(
            "SELECT card_id FROM card WHERE spec_hash=? AND kind='regular'", (sh,)
        ).fetchone()
        if existing is not None:
            # duplicate spec -> ghost, numbered FIFO per canonical spec
            no = self._next_ghost_no(sh)
            self.con.execute(
                "INSERT INTO card (card_id, canonical_spec, spec_hash, trial_id,"
                " family, scope_id, survivor_class, kind, ghost_no, tier,"
                " tier_updated_at, minted_bar, owner_handle, minted_at,"
                " mint_dsr, mint_sr_star, mint_n_stratum,"
                " baseline_t_net, baseline_sr_ann, baseline_holdout_bars)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (card_id + f":g{no}", canonical_spec_json(spec), sh, trial_id,
                 family, scope_id, survivor_class, "ghost", no, tier,
                 now, int(minted_bar), owner_handle, now,
                 dsr, sr_star, n_stratum,
                 baseline_t_net, baseline_sr_ann, baseline_holdout_bars),
            )
            self.con.commit()
            return self.get_card(card_id + f":g{no}")   # read-back

        self.con.execute(
            "INSERT INTO card (card_id, canonical_spec, spec_hash, trial_id,"
            " family, scope_id, survivor_class, kind, ghost_no, tier,"
            " tier_updated_at, minted_bar, owner_handle, minted_at,"
            " mint_dsr, mint_sr_star, mint_n_stratum,"
            " baseline_t_net, baseline_sr_ann, baseline_holdout_bars)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (card_id, canonical_spec_json(spec), sh, trial_id,
             family, scope_id, survivor_class, "regular", None, tier,
             now, int(minted_bar), owner_handle, now,
             dsr, sr_star, n_stratum,
             baseline_t_net, baseline_sr_ann, baseline_holdout_bars),
        )
        self.con.commit()
        return self.get_card(card_id)                    # read-back

    def _next_ghost_no(self, spec_hash: str) -> int:
        row = self.con.execute(
            "SELECT next_no FROM spec_ghost_counter WHERE spec_hash=?", (spec_hash,)
        ).fetchone()
        no = row["next_no"] if row else 1
        self.con.execute(
            "INSERT INTO spec_ghost_counter (spec_hash, next_no) VALUES (?,?)"
            " ON CONFLICT(spec_hash) DO UPDATE SET next_no=excluded.next_no",
            (spec_hash, no + 1),
        )
        return no

    # -- reads ----------------------------------------------------------------

    def get_card(self, card_id: str) -> Card:
        """Read-back of one card. Raises on unknown id."""
        row = self.con.execute("SELECT * FROM card WHERE card_id=?", (card_id,)).fetchone()
        if row is None:
            raise CardError(f"unknown card_id {card_id!r}")
        return self._to_card(row)

    def cards_for_owner(self, handle: str) -> list[Card]:
        rows = self.con.execute(
            "SELECT * FROM card WHERE owner_handle=? ORDER BY minted_at, card_id",
            (handle,),
        ).fetchall()
        return [self._to_card(r) for r in rows]

    def all_cards(self) -> list[Card]:
        """Every card in the registry (regular + ghosts), oldest first.
        Read used by the decay re-evaluation pass (sift/cards/decay.py)."""
        rows = self.con.execute("SELECT * FROM card ORDER BY minted_at, card_id").fetchall()
        return [self._to_card(r) for r in rows]

    def ghosts_for_spec(self, spec_hash: str) -> list[Card]:
        """ACTIVE ghost cards for one canonical spec, FIFO order (mechanics
        3). Ghosts discarded from the FIFO carry ghost_no<=0 and are excluded
        here (they remain in the registry pool, never destroyed)."""
        rows = self.con.execute(
            "SELECT * FROM card WHERE spec_hash=? AND kind='ghost' AND ghost_no>0"
            " ORDER BY ghost_no",
            (spec_hash,),
        ).fetchall()
        return [self._to_card(r) for r in rows]

    def _to_card(self, row: sqlite3.Row) -> Card:
        keys = row.keys()
        return Card(
            card_id=row["card_id"], spec_hash=row["spec_hash"],
            trial_id=row["trial_id"], family=row["family"], scope_id=row["scope_id"],
            survivor_class=row["survivor_class"],
            kind=row["kind"], ghost_no=row["ghost_no"], tier=row["tier"],
            owner_handle=row["owner_handle"], minted_at=row["minted_at"],
            minted_bar=row["minted_bar"] if "minted_bar" in keys else 0,
            decayed_from=row["decayed_from"],
            mint_dsr=row["mint_dsr"] if "mint_dsr" in keys else None,
            mint_sr_star=row["mint_sr_star"] if "mint_sr_star" in keys else None,
            mint_n_stratum=row["mint_n_stratum"] if "mint_n_stratum" in keys else None,
            baseline_t_net=row["baseline_t_net"] if "baseline_t_net" in keys else None,
            baseline_sr_ann=row["baseline_sr_ann"] if "baseline_sr_ann" in keys else None,
            baseline_holdout_bars=(row["baseline_holdout_bars"]
                                   if "baseline_holdout_bars" in keys else None),
        )

    # -- decay + non-transferability ------------------------------------------

    def decay(self, card_id: str, new_tier: str, now: float | None = None) -> Card:
        """Downgrade-only decay primitive (mechanics 1). Records
        decayed_from; the row is updated in place but the card's history is
        kept in the returned record + the tier change is append-only in
        spirit (decayed_from). Read-back before returning.

        NOTE: this primitive can only move a tier DOWN. The bidirectional
        policy (pre-registered live-window upgrades) lives in
        sift/cards/decay.py::RegistryDecayJob and writes through
        `set_tier()` — never through here."""
        card = self.get_card(card_id)
        now = time.time() if now is None else now
        downgraded = decay_tier(card.tier, new_tier)
        self.con.execute(
            "UPDATE card SET tier=?, tier_updated_at=?, decayed_from=? WHERE card_id=?",
            (downgraded, now, card.tier if downgraded != card.tier else card.decayed_from,
             card_id),
        )
        self.con.commit()
        return self.get_card(card_id)

    def set_tier(self, card_id: str, new_tier: str, now: float | None = None) -> Card:
        """Gated tier write used ONLY by the registry decay job's
        pre-registered upgrade path (sift/cards/decay.py). Unlike `decay()`
        this may move a tier UP, but callers must have already satisfied
        every pre-registered rule (fully-OOS live window, two consecutive
        qualifying annual windows at the mint-time bar, 90-day rate limit,
        seeded evidence, append-only ledger event). The low-level
        `decay_tier` guard stays downgrade-only; this method exists so the
        registry's downgrade path keeps using `decay()` semantics
        (decayed_from) while the upgrade path records its own evidence.
        Read-back before returning."""
        if new_tier not in CARD_TIERS:
            raise CardError(f"invalid tier {new_tier!r}")
        card = self.get_card(card_id)
        now = time.time() if now is None else now
        self.con.execute(
            "UPDATE card SET tier=?, tier_updated_at=? WHERE card_id=?",
            (new_tier, now, card_id),
        )
        self.con.commit()
        return self.get_card(card_id)

    def assert_owned(self, card_id: str, handle: str) -> Card:
        """Non-transferability guard: cards never change owner; any path that
        needs the card for a user checks ownership here and raises otherwise."""
        card = self.get_card(card_id)
        if card.owner_handle != handle:
            raise CardError(
                f"card {card_id} belongs to {card.owner_handle!r}, not {handle!r} —"
                " cards are non-transferable"
            )
        return card


__all__ = [
    "CARD_KINDS",
    "CARD_TIERS",
    "Card",
    "CardError",
    "CardRegistry",
    "canonical_spec_json",
    "decay_tier",
    "spec_hash_of",
    "tier_from_stats",
]
