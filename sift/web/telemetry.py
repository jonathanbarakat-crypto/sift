"""Gate-1 append-only telemetry event store & D30/D60/D90 Wilson CI reporting.

Implements docs/beta-plan.md §4–§6:
- One append-only SQLite event table (`telemetry_event`) keyed by opaque
  `user_id` (random lowercase UUID). Email NEVER enters this table or any
  analytics export — enforced fail-closed at write time AND by SQLite
  BEFORE UPDATE / BEFORE DELETE triggers.
- Minimal event set (§5):
    beta_signup, beta_invited, beta_accepted, account_created,
    session_login, trial_registered, trial_verdict, connection_saved,
    card_minted, card_discarded, ghost_promoted, share_grant, feedback_note
- Pre-registered Gate-1 active definition (§4):
    >= 1 NEW unique `trial_registered` event in (day0 + D - 30d, day0 + D].
    Skip-list cache hits never emit a duplicate `trial_registered`.
- Small-n reporting rules (§5–§6):
    - Every rate ships with raw k/n AND binomial Wilson 95% CI (z=1.96).
    - Qualitative band: majority_retained (k >= ceil(n/2)),
      minority_retained (k < n/3), inconclusive_in_between, or no_cohort.
    - No point-estimate pass line; D30 early stop-loss flagged when
      k < ceil(n/2).
    - Subgroup splits (card-holders vs non-holders) report counts +
      qualitative description only — never p-values or significance claims.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Self

from sift.web.compliance import ComplianceError, assert_no_email_pii

DAY_S = 86400.0

ALLOWED_EVENT_TYPES = frozenset({
    "beta_signup",
    "beta_invited",
    "beta_accepted",
    "account_created",
    "session_login",
    "trial_registered",
    "trial_verdict",
    "connection_saved",
    "card_minted",
    "card_discarded",
    "ghost_promoted",
    "share_grant",
    "feedback_note",
})

TELEMETRY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telemetry_event (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT,                      -- opaque UUID; NULL only for pre-account events
    event_type   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_type_ts
    ON telemetry_event(event_type, created_at);
CREATE INDEX IF NOT EXISTS idx_telemetry_user_type_ts
    ON telemetry_event(user_id, event_type, created_at);

CREATE TRIGGER IF NOT EXISTS trg_telemetry_no_update
BEFORE UPDATE ON telemetry_event
BEGIN
    SELECT RAISE(ABORT, 'telemetry_event is append-only (UPDATE refused)');
END;

CREATE TRIGGER IF NOT EXISTS trg_telemetry_no_delete
BEFORE DELETE ON telemetry_event
BEGIN
    SELECT RAISE(ABORT, 'telemetry_event is append-only (DELETE refused)');
END;
"""


class TelemetryError(Exception):
    """Telemetry store misuse (invalid event type, PII detected, ...)."""


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Binomial Wilson score interval (default 95%, z=1.96).

    Matches docs/beta-plan.md §0 benchmarks:
      n=30, k=15 -> (0.331, 0.669) -> [33%, 67%], width 34pp
      n=20, k=10 -> (0.299, 0.701) -> [30%, 70%], width 40pp
      n=10, k=5  -> (0.237, 0.763) -> [24%, 76%], width 53pp
    """
    if n <= 0:
        return (0.0, 0.0)
    if k < 0 or k > n:
        raise ValueError(f"invalid counts k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def qualitative_band(k: int, n: int) -> str:
    """Small-n qualitative retention band per docs/beta-plan.md §4 & §6."""
    if n <= 0:
        return "no_cohort"
    if k >= math.ceil(n / 2):
        return "majority_retained"
    if k < (n / 3.0):
        return "minority_retained"
    return "inconclusive_in_between"


def _rate_block(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_ci(k, n)
    rate = (k / n) if n > 0 else 0.0
    return {
        "k": int(k),
        "n": int(n),
        "rate": round(rate, 4),
        "wilson_95_ci": [round(lo, 4), round(hi, 4)],
        "ci_width_pp": round((hi - lo) * 100.0, 1) if n > 0 else 0.0,
        "qualitative_band": qualitative_band(k, n),
    }


class TelemetryStore:
    """Append-only Gate-1 telemetry event store backed by SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path), timeout=10.0)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode = WAL")
        self.con.execute("PRAGMA busy_timeout = 10000")
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(TELEMETRY_SCHEMA_SQL)
        self.con.commit()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
        now: float | None = None,
    ) -> int:
        """Append one event. Fail-closed on unknown event_type or email PII."""
        if event_type not in ALLOWED_EVENT_TYPES:
            raise TelemetryError(f"unknown telemetry event_type {event_type!r}")
        if user_id is not None:
            try:
                assert_no_email_pii(user_id, path="user_id")
            except ComplianceError as exc:
                raise TelemetryError(str(exc)) from exc
        data = dict(payload or {})
        try:
            assert_no_email_pii(data, path="payload")
        except ComplianceError as exc:
            raise TelemetryError(str(exc)) from exc
        ts = time.time() if now is None else float(now)
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        try:
            cur = self.con.execute(
                "INSERT INTO telemetry_event (user_id, event_type, payload_json, created_at)"
                " VALUES (?,?,?,?)",
                (user_id, event_type, blob, ts),
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    def export_events(
        self,
        *,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return PII-free analytics export rows (never includes email)."""
        sql = "SELECT event_id, user_id, event_type, payload_json, created_at FROM telemetry_event"
        params: list[Any] = []
        if event_type is not None:
            sql += " WHERE event_type=?"
            params.append(event_type)
        sql += " ORDER BY event_id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self.con.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = {
                "event_id": int(r["event_id"]),
                "user_id": r["user_id"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload_json"]),
                "created_at": float(r["created_at"]),
            }
            assert_no_email_pii(item, path=f"export[{r['event_id']}]")
            out.append(item)
        return out

    def count_events(self, event_type: str | None = None) -> int:
        if event_type is None:
            return int(self.con.execute("SELECT COUNT(*) FROM telemetry_event").fetchone()[0])
        return int(
            self.con.execute(
                "SELECT COUNT(*) FROM telemetry_event WHERE event_type=?",
                (event_type,),
            ).fetchone()[0]
        )


def gate1_report(
    store: TelemetryStore,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Compute the pre-registered Gate-1 D30/D60/D90 report (docs/beta-plan.md §4–§7).

    Uses ONLY the append-only telemetry store keyed by opaque `user_id` —
    never reads emails. Every rate is paired with its raw k/n, binomial
    Wilson 95% CI, and qualitative band. Subgroup splits (card-holders vs
    non-card-holders) report counts and qualitative summary only (never
    p-values or significance headlines on small cells).
    """
    as_of = time.time() if now is None else float(now)

    # Day-0 anchor per beta user: first account_created event with beta=True.
    acct_rows = store.con.execute(
        "SELECT user_id, payload_json, MIN(created_at) AS day0"
        " FROM telemetry_event"
        " WHERE event_type='account_created' AND user_id IS NOT NULL"
        " GROUP BY user_id"
    ).fetchall()
    cohort: dict[str, float] = {}
    for r in acct_rows:
        payload = json.loads(r["payload_json"])
        if payload.get("beta", True):
            cohort[str(r["user_id"])] = float(r["day0"])

    # Unique trial_registered events per user: (user_id, trial_id, created_at).
    # Deduplicating on (user_id, trial_id) guarantees skip-list re-runs can
    # never inflate activity even if a caller logged redundantly.
    trial_rows = store.con.execute(
        "SELECT user_id, payload_json, created_at FROM telemetry_event"
        " WHERE event_type='trial_registered' AND user_id IS NOT NULL"
        " ORDER BY created_at ASC"
    ).fetchall()
    user_trials: dict[str, list[tuple[str, float]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for r in trial_rows:
        uid = str(r["user_id"])
        if uid not in cohort:
            continue
        payload = json.loads(r["payload_json"])
        tid = str(payload.get("trial_id") or f"ev-{r['created_at']}")
        key = (uid, tid)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        user_trials.setdefault(uid, []).append((tid, float(r["created_at"])))

    # Card holders (minted >= 1 card) and users with >= 1 survivor verdict.
    card_holders = {
        str(r["user_id"])
        for r in store.con.execute(
            "SELECT DISTINCT user_id FROM telemetry_event"
            " WHERE event_type='card_minted' AND user_id IS NOT NULL"
        ).fetchall()
        if str(r["user_id"]) in cohort
    }
    survivor_users = set()
    for r in store.con.execute(
        "SELECT user_id, payload_json FROM telemetry_event"
        " WHERE event_type='trial_verdict' AND user_id IS NOT NULL"
    ).fetchall():
        uid = str(r["user_id"])
        if uid in cohort and json.loads(r["payload_json"]).get("verdict") == "survivor":
            survivor_users.add(uid)

    # Horizons D30 / D60 / D90
    horizons: dict[str, dict[str, Any]] = {}
    active_sets: dict[int, set[str]] = {}
    for d in (30, 60, 90):
        # Matured users = beta users whose day0 + D days <= as_of.
        matured = {uid for uid, d0 in cohort.items() if (d0 + d * DAY_S) <= as_of}
        active_matured: set[str] = set()
        for uid in matured:
            d0 = cohort[uid]
            win_lo = d0 + (d - 30) * DAY_S
            win_hi = d0 + d * DAY_S
            if any(win_lo < ts <= win_hi for _, ts in user_trials.get(uid, ())):
                active_matured.add(uid)
        active_sets[d] = active_matured
        entry = _rate_block(len(active_matured), len(matured))
        entry["horizon_days"] = d
        entry["total_cohort_n"] = len(cohort)
        entry["matured_n"] = len(matured)
        horizons[f"D{d}"] = entry

    # D30 early stop-loss check (docs/beta-plan.md §6.2): k < ceil(n/2)
    d30 = horizons["D30"]
    d30_stop_loss = bool(
        d30["matured_n"] > 0 and d30["k"] < math.ceil(d30["matured_n"] / 2)
    )

    # Secondary metrics (docs/beta-plan.md §4 & §7 — small-n rules enforced)
    since_30d = as_of - 30.0 * DAY_S
    logins_30d = int(
        store.con.execute(
            "SELECT COUNT(*) FROM telemetry_event"
            " WHERE event_type='session_login' AND created_at > ? AND created_at <= ?",
            (since_30d, as_of),
        ).fetchone()[0]
    )
    connections_saved = store.count_events("connection_saved")
    cards_minted = store.count_events("card_minted")
    cards_discarded = store.count_events("card_discarded")
    ghosts_promoted = store.count_events("ghost_promoted")
    share_grants = store.count_events("share_grant")

    # G1-a subgroup split at D90: counts + qualitative only (NEVER p-values)
    matured_d90 = {uid for uid, d0 in cohort.items() if (d0 + 90 * DAY_S) <= as_of}
    holders_d90 = matured_d90 & card_holders
    non_holders_d90 = matured_d90 - card_holders
    holders_active_d90 = holders_d90 & active_sets[90]
    non_holders_active_d90 = non_holders_d90 & active_sets[90]

    report = {
        "as_of_ts": as_of,
        "cohort_n": len(cohort),
        "reporting_rules": (
            "Small-n beta read (docs/beta-plan.md §5-§6): counts, Wilson 95% "
            "CIs, and qualitative bands only. No bare percentages, no "
            "point-estimate pass line, no significance tests on small cells."
        ),
        "horizons": horizons,
        "d30_early_stop_loss_triggered": d30_stop_loss,
        "secondary": {
            "logins_trailing_30d": logins_30d,
            "unique_trials_registered": len(seen_pairs),
            "connections_saved": connections_saved,
            "users_with_survivor": _rate_block(len(survivor_users), len(cohort)),
            "cards_minted": cards_minted,
            "cards_discarded": cards_discarded,
            "ghosts_promoted": ghosts_promoted,
            "share_grants": share_grants,
            "g1_a_d90_split_counts_only": {
                "card_holders": {
                    "matured_n": len(holders_d90),
                    "active_k": len(holders_active_d90),
                    "band": qualitative_band(
                        len(holders_active_d90), len(holders_d90)
                    ),
                },
                "non_card_holders": {
                    "matured_n": len(non_holders_d90),
                    "active_k": len(non_holders_active_d90),
                    "band": qualitative_band(
                        len(non_holders_active_d90), len(non_holders_d90)
                    ),
                },
                "note": "Counts and qualitative bands only — no p-values at small n.",
            },
        },
        "funnel_counts": {
            "beta_signup": store.count_events("beta_signup"),
            "beta_invited": store.count_events("beta_invited"),
            "beta_accepted": store.count_events("beta_accepted"),
            "account_created": len(cohort),
        },
    }
    assert_no_email_pii(report, path="gate1_report")
    return report


__all__ = [
    "ALLOWED_EVENT_TYPES",
    "DAY_S",
    "TelemetryError",
    "TelemetryStore",
    "gate1_report",
    "qualitative_band",
    "wilson_ci",
]
