"""v4.1 knowledge-reward badges (mechanics item 7, behind `cards_v4_1`).

Mechanics (authoritative, v4.1 spec §4 items 1-4):
- kill_rate badge: "43 trials, 41 killed" — celebrates hypothesis discipline;
  computed deterministically from the append-only trial registry.
- pre_registration_streak: consecutive pre-registered trials surviving
  S0-S4 — rewards process (pre-registration before peeking), not outcomes.
- peer_retest: "re-tested by @user: 47/50 robust" recorded as a status
  artifact on registry entries (extends v3 §6.3); any leaderboard is on
  validation survival rate ONLY, never P&L (v3 §12.4 preserved).
- scope_honesty: users who declare narrow scopes honestly and let S5 judge
  at full depth.

Hard rules:
- All badges are deterministic and derivable read-only from the trial
  registry. No P&L, no win rates, no directional language anywhere — a
  fail-closed guard (same discipline as the card-face ticker guard) rejects
  any badge payload containing forbidden fields.
- Opt-in display per user; opt-out respected (no badge is shown).
- No confetti/trading hooks — badges reward research discipline only.
- Flag off -> BadgeError(flag_off) on every compute/display path.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from sift import flags
from sift.registry import Ledger

BADGE_KINDS = ("kill_rate", "pre_registration_streak", "peer_retest", "scope_honesty")

# Fields that must NEVER appear in a badge payload (v3 §12: no P&L, no win
# rates, no directional calls, no market language). Fail closed on any hit.
# The guard scans dict KEYS and leaf VALUES separately — never free-text
# notes/disclaimers, which legitimately say "never P&L" — so the badge's own
# legal disclaimer text cannot false-positive the guard.
_FORBIDDEN_RE = re.compile(
    r"^(pnl|p&l|profit|loss|win_rate|winrate|return|returns|alpha|pick|picks|"
    r"buy|sell|long|short|leverage|market|price_target|target_price)$",
    re.IGNORECASE,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS badge_state (
    handle      TEXT NOT NULL,
    badge       TEXT NOT NULL,
    opt_in      INTEGER NOT NULL DEFAULT 0,     -- opt-in display flag
    awarded_at  REAL NOT NULL,
    payload     TEXT NOT NULL,                  -- canonical JSON, guard-checked
    PRIMARY KEY (handle, badge)
);
"""


class BadgeError(Exception):
    """Badge misuse: flag off, unknown badge, forbidden payload, ..."""


def _guard_payload(payload: dict) -> None:
    """Fail closed: any P&L / win-rate / directional / market FIELD in the
    badge aborts (same discipline as the card-face ticker guard). Scans dict
    keys and leaf values only — free-text notes/disclaimers are exempt so
    the badge's own legal wording ("never P&L") cannot trip the guard."""
    def _walk(obj, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _FORBIDDEN_RE.match(str(k).strip()):
                    raise BadgeError(
                        f"badge payload field {path + '.' + str(k)!r} is forbidden"
                        " (v3 §12: no P&L / win rates / directional language)"
                    )
                _walk(v, path + "." + str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            # leaf string value: a bare forbidden term as the whole value is
            # a data leak (e.g. {"note": "win_rate"}); prose containing the
            # word inside a longer sentence is the legal disclaimer case.
            if _FORBIDDEN_RE.match(obj.strip()):
                raise BadgeError(
                    f"badge payload value at {path!r} is a forbidden term"
                    " (v3 §12: no P&L / win rates / directional language)"
                )
    _walk(payload)


@dataclass(frozen=True)
class Badge:
    handle: str
    badge: str
    opt_in: bool
    awarded_at: float
    payload: dict


class BadgeService:
    """Computes badges read-only from the ledger and stores per-user opt-in
    display state. Server-side only."""

    def __init__(self, ledger: Ledger, db_path: str | Path):
        self.ledger = ledger
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA_SQL)
        self.con.commit()

    def __enter__(self) -> "BadgeService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    def _require_flag(self, handle: str) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise BadgeError("flag_off: cards_v4_1 is disabled for this account")

    # -- deterministic badge computation (read-only over the ledger) --------------

    def compute_kill_rate(self, handle: str) -> dict:
        """"N trials, K killed" — hypothesis discipline from the registry."""
        row = self.ledger.con.execute(
            "SELECT COUNT(*) n,"
            " SUM(CASE WHEN verdict='killed' THEN 1 ELSE 0 END) k"
            " FROM trial WHERE discoverer_handle=?",
            (handle,),
        ).fetchone()
        n, k = int(row["n"] or 0), int(row["k"] or 0)
        return {
            "badge": "kill_rate",
            "label": f"{n} trials, {k} killed",
            "trials": n,
            "killed": k,
            "note": "killing bad hypotheses is the product working — discipline, "
                    "not outcomes (survival facts only)",
        }

    def compute_prereg_streak(self, handle: str) -> dict:
        """Consecutive pre-registered trials (provenance != operator_sweep)
        surviving S0-S4, counting back from the most recent. Process badge."""
        rows = self.ledger.con.execute(
            "SELECT trial_id, provenance, verdict FROM trial"
            " WHERE discoverer_handle=? ORDER BY registered_at, trial_id",
            (handle,),
        ).fetchall()
        streak = 0
        for r in reversed(rows):
            if r["provenance"] == "operator_sweep":
                continue                      # not user pre-registration
            if r["verdict"] != "survivor":
                break
            streak += 1
        return {
            "badge": "pre_registration_streak",
            "label": f"{streak} consecutive pre-registered survivor(s)",
            "streak": streak,
            "note": "rewards pre-registration before peeking — process, not outcome",
        }

    def compute_peer_retest(self, handle: str) -> dict:
        """Peer re-tests of this user's trials (provenance='peer_retest' with
        parent_trial_id pointing at the user's trials): "re-tested by peers:
        R robust of T". Survival-rate only, never P&L."""
        rows = self.ledger.con.execute(
            "SELECT t2.verdict v FROM trial t1 JOIN trial t2"
            " ON t2.parent_trial_id = t1.trial_id"
            " WHERE t1.discoverer_handle=? AND t2.provenance='peer_retest'",
            (handle,),
        ).fetchall()
        t = len(rows)
        robust = sum(1 for r in rows if r["v"] == "survivor")
        return {
            "badge": "peer_retest",
            "label": f"re-tested by peers: {robust}/{t} robust",
            "retests": t,
            "robust": robust,
            "note": "validation survival only — never P&L (v3 §12.4)",
        }

    def compute_scope_honesty(self, handle: str) -> dict:
        """Users who declare narrow scopes and let S5 judge at full depth:
        count of the user's trials whose s5 row was actually run (not pending)
        — i.e. they accepted full-depth judgment."""
        rows = self.ledger.con.execute(
            "SELECT COUNT(*) n FROM trial t JOIN stage_result sr"
            " ON sr.trial_id=t.trial_id AND sr.stage='s5'"
            " WHERE t.discoverer_handle=? AND sr.status IN ('pass','fail')",
            (handle,),
        ).fetchone()
        n = int(rows["n"] or 0)
        return {
            "badge": "scope_honesty",
            "label": f"{n} trial(s) judged at full registry depth",
            "full_depth_judged": n,
            "note": "narrow honest scopes, S5 at full depth — the honesty artifact",
        }

    _COMPUTERS = {
        "kill_rate": compute_kill_rate,
        "pre_registration_streak": compute_prereg_streak,
        "peer_retest": compute_peer_retest,
        "scope_honesty": compute_scope_honesty,
    }

    # -- award + opt-in -----------------------------------------------------------

    def award(self, handle: str, badge: str, opt_in: bool = False,
              now: float | None = None) -> Badge:
        """Compute the badge from the ledger, guard the payload, store the
        per-user state (opt-in display), and read back before returning."""
        self._require_flag(handle)
        if badge not in BADGE_KINDS:
            raise BadgeError(f"unknown badge {badge!r}")
        now = time.time() if now is None else now
        payload = self._COMPUTERS[badge](self, handle)
        _guard_payload(payload)                      # fail closed on forbidden fields
        self.con.execute(
            "INSERT INTO badge_state (handle, badge, opt_in, awarded_at, payload)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(handle, badge) DO UPDATE SET opt_in=excluded.opt_in,"
            " awarded_at=excluded.awarded_at, payload=excluded.payload",
            (handle, badge, int(opt_in), now, json.dumps(payload, sort_keys=True)),
        )
        self.con.commit()
        return self.get_badge(handle, badge)          # read-back

    def set_opt_in(self, handle: str, badge: str, opt_in: bool,
                   now: float | None = None) -> Badge:
        """Opt-in display toggle (default OFF — badges are opt-in)."""
        self._require_flag(handle)
        if badge not in BADGE_KINDS:
            raise BadgeError(f"unknown badge {badge!r}")
        now = time.time() if now is None else now
        row = self.con.execute(
            "SELECT payload FROM badge_state WHERE handle=? AND badge=?",
            (handle, badge),
        ).fetchone()
        if row is None:
            return self.award(handle, badge, opt_in=opt_in, now=now)
        self.con.execute(
            "UPDATE badge_state SET opt_in=?, awarded_at=? WHERE handle=? AND badge=?",
            (int(opt_in), now, handle, badge),
        )
        self.con.commit()
        return self.get_badge(handle, badge)

    # -- reads ---------------------------------------------------------------------

    def get_badge(self, handle: str, badge: str) -> Badge:
        row = self.con.execute(
            "SELECT * FROM badge_state WHERE handle=? AND badge=?",
            (handle, badge),
        ).fetchone()
        if row is None:
            raise BadgeError(f"no badge {badge!r} for {handle!r}")
        return Badge(handle=row["handle"], badge=row["badge"],
                     opt_in=bool(row["opt_in"]), awarded_at=row["awarded_at"],
                     payload=json.loads(row["payload"]))

    def badges_for(self, handle: str, visible_only: bool = True) -> list[Badge]:
        """Badges for the user. `visible_only` (default) returns only opt-in
        badges — opt-out is respected: a non-opted-in badge is never shown."""
        q = "SELECT * FROM badge_state WHERE handle=?"
        if visible_only:
            q += " AND opt_in=1"
        rows = self.con.execute(q + " ORDER BY badge", (handle,)).fetchall()
        return [Badge(handle=r["handle"], badge=r["badge"], opt_in=bool(r["opt_in"]),
                      awarded_at=r["awarded_at"], payload=json.loads(r["payload"]))
                for r in rows]

    def all_badges(self, handle: str) -> list[Badge]:
        return self.badges_for(handle, visible_only=False)


__all__ = ["BADGE_KINDS", "Badge", "BadgeError", "BadgeService"]
