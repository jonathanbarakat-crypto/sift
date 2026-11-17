"""v4.1 mechanics item 7 — knowledge-reward badges (behind `cards_v4_1`).

Badges celebrate RESEARCH DISCIPLINE, never outcomes or money:

- kill_rate:        "43 trials, 41 killed" — from the trial registry ledger,
                    deterministic. Celebrates hypothesis discipline (v3 §6.1).
- prereg_streak:    consecutive PRE-REGISTERED trials (provenance
                    `user_initiated` — the trial_id is minted before any data
                    access, i.e. pre-registration before peeking) surviving
                    S0-S4. Rewards the PROCESS, not the outcome.
- peer_retest:      "re-tested by @user: 47/50 robust" as the status artifact
                    on registry entries (extends v3 §6.3). The per-user badge
                    aggregates the re-tests the user performed.
- scope_honesty:    narrow scopes declared honestly (scope_id != the
                    registry_depth full-depth scope) AND S5 actually judged the
                    trial at full depth (s5 row decided pass|fail).

Leaderboard: validation SURVIVAL RATE only (validated-green survivors /
decided trials) — never P&L, never win rates on trades (v3 §12.4 preserved).

Hard rules (v3 §12 + v4.1 mechanics 7):
- Every badge is a deterministic, read-only function of the trial registry +
  the peer_retest table. No randomness, no lotteries, nothing purchasable.
- No P&L, no win rates, no directional language, no tickers anywhere in a
  badge artifact: a fail-closed guard (like the card-face ticker guard in
  sift/cards/face.py) scans every emitted artifact and raises BadgeError on
  any forbidden field/term/ticker-like token. Badge detail carries integer
  counts only (floats are refused except whitelisted timestamp/rate keys).
- No confetti / trading hooks: artifacts carry `hooks: []` and nothing else
  engagement-shaped (FCA/IOSCO DEP guidance — hooks reward research
  discipline only).
- Opt-in display: a user sees badges only after opting in
  (set_visibility(handle, True)); opt-out is respected (badges_for -> []).
- Flag off -> BadgeError(flag_off) on EVERY path (fail closed). Kill-switch:
  env SIFT_CARDS_V4_1=0 (see sift/flags.py).
- Read-back discipline (hard rule): every write (visibility, re-test) is
  re-read from the DB (GET after write) before it is returned.

Storage: its own SQLite file (like the card registry). Badges are computed
read-only from the trial registry ledger at query time — nothing is cached
that could drift; the only writes are the opt-in flag and peer re-test rows.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import flags
from sift.registry import Ledger, LedgerError

BADGE_KINDS = ("kill_rate", "prereg_streak", "peer_retest", "scope_honesty")

FULL_DEPTH_SCOPE = "registry_depth"     # the S5 full-depth scope
PREREG_PROVENANCE = "user_initiated"     # pre-registered before peeking
GAUNTLET_STAGES = ("s0", "s1", "s2", "s3", "s4")   # S0-S4 survival (S5 separate)

# deterministic award thresholds (a badge is not a participation trophy)
KILL_RATE_MIN_TRIALS = 10
PREREG_STREAK_MIN = 3
PEER_RETEST_MIN = 1
SCOPE_HONESTY_MIN = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS badge_visibility (
    handle      TEXT PRIMARY KEY,
    opt_in      INTEGER NOT NULL DEFAULT 0,   -- opt-in display flag
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS peer_retest (
    trial_id      TEXT NOT NULL,
    tester_handle TEXT NOT NULL,
    n_retests     INTEGER NOT NULL,
    n_robust      INTEGER NOT NULL,
    recorded_at   REAL NOT NULL,
    PRIMARY KEY (trial_id, tester_handle)
);
CREATE INDEX IF NOT EXISTS idx_retest_trial ON peer_retest(trial_id);
"""


class BadgeError(Exception):
    """Badge misuse: flag off, forbidden field/term, unknown trial, ..."""


@dataclass(frozen=True)
class Badge:
    kind: str
    handle: str
    label: str
    detail: dict
    earned: bool
    earned_at: float | None


# -- fail-closed content guard (mirrors the card-face ticker guard) ----------

# Fields that must NEVER appear in a badge artifact (no P&L / win-rate /
# directional / market-content fields — v3 §12).
_FORBIDDEN_KEYS = frozenset({
    "pnl", "p&l", "profit", "loss", "win", "wins", "win_rate", "winrate",
    "winrate_pct", "sharpe", "sortino", "alpha", "beta", "buy", "sell",
    "long", "short", "leverage", "notional", "drawdown", "cagr", "equity",
    "return", "returns", "pct_return", "ticker", "tickers", "symbol",
    "symbols", "directional", "price", "prices", "holdings", "positions",
})

# Terms that must never appear in badge text (word-boundary, case-insensitive).
_FORBIDDEN_WORD_RE = re.compile(
    r"\b(pnl|p&l|profit|loss|win|wins|winrate|win\s*rate|sharpe|sortino|alpha|"
    r"beta|buy|sell|long|short|leverage|notional|drawdown|cagr|equity|"
    r"return|returns|ticker|tickers|symbol|symbols|directional|price|"
    r"prices)\b",
    re.IGNORECASE,
)

# Ticker-like tokens (same rule as the card face, v3 §12.3).
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b")
_GUARD_WHITELIST = {"SIFT", "JSON", "US", "ID", "NO", "TS", "QA"}

# Floats are refused in badge detail (counts only); these keys may carry one.
_FLOAT_OK_KEYS = frozenset({"earned_at", "updated_at", "recorded_at", "ts", "survival_rate"})


def _guard_text(text: str, mask: tuple[str, ...], path: str) -> None:
    """Fail closed: any forbidden term or ticker-like token in badge text
    aborts the emit. `mask` strings (handles) are masked before the ticker
    scan so a user handle like "AL" is not mistaken for a ticker."""
    t = text
    for m in mask:
        if m:
            t = t.replace(str(m), "<H>")
    for m in _FORBIDDEN_WORD_RE.finditer(t):
        raise BadgeError(
            f"{path}: forbidden term {m.group(0)!r} in badge text — refused"
            " (no P&L / win-rate / directional language in badges)"
        )
    for tok in _TICKER_RE.findall(t):
        if tok not in _GUARD_WHITELIST:
            raise BadgeError(
                f"{path}: ticker-like token {tok!r} in badge text — refused (v3 §12.3)"
            )


def _guard_obj(obj, mask: tuple[str, ...] = (), path: str = "badge") -> None:
    """Fail-closed structural guard on any emitted badge artifact: forbidden
    keys, forbidden terms in strings, ticker-like tokens, and floats outside
    the whitelisted timestamp/rate keys (badge detail = integer counts only,
    so no P&L-shaped number can ride along)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in _FORBIDDEN_KEYS:
                raise BadgeError(
                    f"{path}: forbidden field {k!r} — badges never carry"
                    " P&L / win-rate fields (fail closed)"
                )
            if isinstance(v, float) and key not in _FLOAT_OK_KEYS:
                raise BadgeError(
                    f"{path}: float value under {k!r} — badge detail carries"
                    " integer counts only (fail closed)"
                )
            _guard_obj(v, mask, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _guard_obj(v, mask, f"{path}[{i}]")
    elif isinstance(obj, str):
        _guard_text(obj, mask, path)


def badge_dict(b: Badge, mask: tuple[str, ...] = ()) -> dict:
    """Serialize a Badge to its public artifact dict and run the guard."""
    d = {
        "kind": b.kind,
        "handle": b.handle,
        "label": b.label,
        "detail": dict(b.detail),
        "earned": b.earned,
        "earned_at": b.earned_at,
    }
    _guard_obj(d, mask or (b.handle,))
    return d


class BadgeEngine:
    """Computes knowledge-reward badges read-only from the trial registry.

    Server-side only. Every public method fails closed with
    BadgeError(flag_off) when `cards_v4_1` is off for the handle.
    """

    def __init__(self, ledger: Ledger, db_path: str | Path):
        self.ledger = ledger
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "BadgeEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- flag ---------------------------------------------------------------

    def _require_flag(self, handle: str | None) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise BadgeError("flag_off: cards_v4_1 is disabled for this account")

    # -- opt-in visibility ----------------------------------------------------

    def set_visibility(self, handle: str, opt_in: bool,
                       now: float | None = None) -> dict:
        """Set the per-user opt-in display flag. Write + read-back (hard
        rule: GET after write). Flag off -> BadgeError(flag_off)."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT INTO badge_visibility (handle, opt_in, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(handle) DO UPDATE SET opt_in=excluded.opt_in,"
            " updated_at=excluded.updated_at",
            (handle, 1 if opt_in else 0, now),
        )
        self.con.commit()
        return self.visibility_row(handle)          # read-back

    def visibility_row(self, handle: str) -> dict:
        """Raw visibility row (no flag check — internal read helper)."""
        row = self.con.execute(
            "SELECT * FROM badge_visibility WHERE handle=?", (handle,)
        ).fetchone()
        if row is None:
            return {"handle": handle, "opt_in": False, "updated_at": None}
        return {"handle": row["handle"], "opt_in": bool(row["opt_in"]),
                "updated_at": row["updated_at"]}

    def visibility(self, handle: str) -> bool:
        """Whether badge display is opted in for `handle` (read-back row)."""
        self._require_flag(handle)
        return self.visibility_row(handle)["opt_in"]

    # -- read-only ledger helpers ---------------------------------------------

    def _trials_for(self, handle: str) -> list[sqlite3.Row]:
        return self.ledger.con.execute(
            "SELECT * FROM trial WHERE discoverer_handle=?"
            " ORDER BY registered_at, rowid",
            (handle,),
        ).fetchall()

    def _stage_status(self, trial_id: str, stage: str) -> str | None:
        row = self.ledger.con.execute(
            "SELECT status FROM stage_result WHERE trial_id=? AND stage=?",
            (trial_id, stage),
        ).fetchone()
        return row["status"] if row else None

    # -- badge computations (deterministic, read-only) -------------------------

    def _kill_rate(self, handle: str, now: float) -> Badge:
        rows = self._trials_for(handle)
        decided = [r for r in rows if r["verdict"] != "pending"]
        killed = [r for r in decided if r["verdict"] == "killed"]
        n, k = len(decided), len(killed)
        earned = n >= KILL_RATE_MIN_TRIALS
        return Badge(
            "kill_rate", handle, f"{n} trials, {k} killed",
            {"trials": n, "killed": k}, earned, now if earned else None,
        )

    def _prereg_streak(self, handle: str, now: float) -> Badge:
        """Consecutive pre-registered trials (provenance user_initiated —
        minted before any data access) surviving S0-S4, counting backwards
        from the latest DECIDED trial. Undecided (pending) trials are
        skipped; anything else breaks the streak."""
        streak = 0
        for r in reversed(self._trials_for(handle)):
            if r["verdict"] == "pending":
                continue
            if r["provenance"] != PREREG_PROVENANCE:
                break
            if all(self._stage_status(r["trial_id"], s) == "pass"
                   for s in GAUNTLET_STAGES):
                streak += 1
            else:
                break
        earned = streak >= PREREG_STREAK_MIN
        return Badge(
            "prereg_streak", handle, f"pre-registration streak: {streak}",
            {"streak": streak}, earned, now if earned else None,
        )

    def _peer_retest(self, handle: str, now: float) -> Badge:
        """Per-user aggregate of the re-tests this user performed (the
        per-entry status artifact lives on the trial, see retest_status)."""
        rows = self.con.execute(
            "SELECT n_retests, n_robust FROM peer_retest WHERE tester_handle=?"
            " ORDER BY recorded_at, rowid",
            (handle,),
        ).fetchall()
        n = sum(int(r["n_retests"]) for r in rows)
        rob = sum(int(r["n_robust"]) for r in rows)
        earned = n >= PEER_RETEST_MIN
        return Badge(
            "peer_retest", handle, f"peer re-tests: {rob}/{n} robust",
            {"retests": n, "robust": rob, "entries": len(rows)},
            earned, now if earned else None,
        )

    def _scope_honesty(self, handle: str, now: float) -> Badge:
        """Narrow scope declared honestly (scope_id != registry_depth) AND
        S5 judged at full depth (s5 row decided pass|fail, never left
        pending)."""
        n = 0
        for r in self._trials_for(handle):
            if r["scope_id"] == FULL_DEPTH_SCOPE:
                continue
            if r["verdict"] == "pending":
                continue
            if self._stage_status(r["trial_id"], "s5") in ("pass", "fail"):
                n += 1
        earned = n >= SCOPE_HONESTY_MIN
        return Badge(
            "scope_honesty", handle,
            f"scope-honest: {n} narrow-scope trials judged at full depth",
            {"full_depth_judged": n}, earned, now if earned else None,
        )

    def compute_badges(self, handle: str, now: float | None = None) -> list[dict]:
        """All four badge kinds computed read-only from the registry, each
        guard-checked. Returns artifact dicts with the `earned` flag.
        Flag off -> BadgeError(flag_off)."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        badges = (
            self._kill_rate(handle, now),
            self._prereg_streak(handle, now),
            self._peer_retest(handle, now),
            self._scope_honesty(handle, now),
        )
        return [badge_dict(b, mask=(handle,)) for b in badges]

    def badges_for(self, handle: str, now: float | None = None) -> list[dict]:
        """Badges DISPLAYED for `handle`: earned badges only, and only when
        the user opted in (opt-out respected -> []). Flag off ->
        BadgeError(flag_off)."""
        self._require_flag(handle)
        if not self.visibility_row(handle)["opt_in"]:
            return []
        return [b for b in self.compute_badges(handle, now=now) if b["earned"]]

    def badge_artifact(self, handle: str, now: float | None = None) -> dict:
        """The public badge artifact for a profile: opt-in state + earned
        badges + explicit empty hooks list (no confetti / trading hooks —
        hooks reward research discipline only, and there are none)."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        art = {
            "handle": handle,
            "display_opt_in": self.visibility_row(handle)["opt_in"],
            "badges": self.badges_for(handle, now=now),
            "hooks": [],
            "note": "SIFT badges reward research discipline only (v3 §12).",
        }
        _guard_obj(art, mask=(handle,))
        return art

    # -- peer re-test ladder (v3 §6.3 status artifact on registry entries) -----

    def record_retest(self, trial_id: str, tester_handle: str, *,
                      n_retests: int, n_robust: int,
                      now: float | None = None) -> dict:
        """Record a peer re-test of a decided trial: the status artifact
        "re-tested by @tester: R/N robust" on the registry entry. Write +
        read-back (GET after write). Flag off -> BadgeError(flag_off)."""
        self._require_flag(tester_handle)
        if not isinstance(n_retests, int) or isinstance(n_retests, bool):
            raise BadgeError("n_retests must be an int")
        if not isinstance(n_robust, int) or isinstance(n_robust, bool):
            raise BadgeError("n_robust must be an int")
        if n_retests < 1:
            raise BadgeError("n_retests must be >= 1")
        if not 0 <= n_robust <= n_retests:
            raise BadgeError(
                f"n_robust ({n_robust}) must be within [0, n_retests={n_retests}]"
            )
        try:
            trial = self.ledger.get(trial_id)
        except LedgerError as e:
            raise BadgeError(str(e)) from e
        if trial.verdict == "pending":
            raise BadgeError(
                f"cannot re-test {trial_id!r}: trial still pending — only"
                " decided registry entries carry a re-test status"
            )
        now = time.time() if now is None else now
        self.con.execute(
            "INSERT INTO peer_retest (trial_id, tester_handle, n_retests,"
            " n_robust, recorded_at) VALUES (?,?,?,?,?)"
            " ON CONFLICT(trial_id, tester_handle) DO UPDATE SET"
            " n_retests=excluded.n_retests, n_robust=excluded.n_robust,"
            " recorded_at=excluded.recorded_at",
            (trial_id, tester_handle, n_retests, n_robust, now),
        )
        self.con.commit()
        return self._retest_artifact(trial_id, tester_handle)   # read-back

    def _retest_artifact(self, trial_id: str, tester_handle: str) -> dict:
        row = self.con.execute(
            "SELECT * FROM peer_retest WHERE trial_id=? AND tester_handle=?",
            (trial_id, tester_handle),
        ).fetchone()
        if row is None:
            raise BadgeError(f"no re-test row for {trial_id!r}/{tester_handle!r}")
        art = {
            "trial_id": row["trial_id"],
            "tester": row["tester_handle"],
            "n_retests": int(row["n_retests"]),
            "n_robust": int(row["n_robust"]),
            "label": (f"re-tested by @{row['tester_handle']}: "
                      f"{int(row['n_robust'])}/{int(row['n_retests'])} robust"),
            "recorded_at": row["recorded_at"],
        }
        _guard_obj(art, mask=(row["tester_handle"],))
        return art

    def retest_status(self, trial_id: str, requester_handle: str) -> list[dict]:
        """The re-test status artifacts attached to one registry entry
        (v3 §6.3 ladder). Flag off -> BadgeError(flag_off)."""
        self._require_flag(requester_handle)
        try:
            self.ledger.get(trial_id)
        except LedgerError as e:
            raise BadgeError(str(e)) from e
        rows = self.con.execute(
            "SELECT tester_handle FROM peer_retest WHERE trial_id=?"
            " ORDER BY recorded_at, rowid",
            (trial_id,),
        ).fetchall()
        return [self._retest_artifact(trial_id, r["tester_handle"]) for r in rows]

    # -- leaderboard: validation survival rate ONLY (v3 §12.4) -----------------

    def leaderboard(self, requester_handle: str | None = None) -> list[dict]:
        """Leaderboard on VALIDATION SURVIVAL RATE only: validated-green
        survivors / decided trials per handle. Never P&L, never trade win
        rates (v3 §12.4). Deterministic ordering: rate desc, handle asc.
        Flag off -> BadgeError(flag_off)."""
        self._require_flag(requester_handle)
        rows = self.ledger.con.execute(
            "SELECT discoverer_handle, verdict, survivor_class FROM trial"
            " WHERE verdict != 'pending' AND discoverer_handle IS NOT NULL"
        ).fetchall()
        agg: dict[str, dict[str, int]] = {}
        for r in rows:
            h = r["discoverer_handle"]
            a = agg.setdefault(h, {"decided": 0, "validated_green": 0})
            a["decided"] += 1
            if r["verdict"] == "survivor" and r["survivor_class"] == "validated-green":
                a["validated_green"] += 1
        entries = [
            {
                "handle": h,
                "decided": a["decided"],
                "validated_green": a["validated_green"],
                "survival_rate": a["validated_green"] / a["decided"],
            }
            for h, a in agg.items()
        ]
        entries.sort(key=lambda e: (-e["survival_rate"], e["handle"]))
        for e in entries:
            _guard_obj(e, mask=(e["handle"],))
        return entries


__all__ = [
    "BADGE_KINDS",
    "Badge",
    "BadgeEngine",
    "BadgeError",
    "FULL_DEPTH_SCOPE",
    "GAUNTLET_STAGES",
    "KILL_RATE_MIN_TRIALS",
    "PEER_RETEST_MIN",
    "PREREG_PROVENANCE",
    "PREREG_STREAK_MIN",
    "SCOPE_HONESTY_MIN",
    "badge_dict",
]
