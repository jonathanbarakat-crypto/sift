"""Gate-1 metric hooks for the v4.1 card layer (behind `cards_v4_1`).

The beta-cohort measurement pipeline (docs/beta-plan.md §5: append-only
event store keyed by opaque user_id, email as the only PII, aggregate-only
reporting) is NOT yet in this repo — there is no event store / auth DB here.
These hooks are therefore INSTRUMENTATION STUBS: pure, deterministic
computations over the stores that DO exist server-side (the trial ledger,
the card registry, the vault, the share audit log), returning the named
gate-1 metrics. When the measurement pipeline lands, each hook is replaced
by the same-named query over the event store; the signatures and return
shapes here are the contract.

Metrics (per the v4.1 card task, gate-1 set):
- d90_retention_split: D90 "active" (>=1 new unique trial in trailing 30d,
  per beta-plan §4) split card-holders vs non-card-holders. Activity is
  derived from the ledger (discoverer_handle -> trial registrations);
  "card-holder" = owner of >=1 card in the card registry.
- ghost_ratio: active ghosts / total cards (registry pool health).
- vault_turnover: discards per active vault card over the window
  (vault_entry state flips + registry_pool inserts).
- share_grant_d30_retention: D30 retention of accounts that received a
  share-for-capacity grant vs accounts that did not.
- anchored_vs_unanchored_hit_rate: survivor rate of anchored-run sampled
  candidates vs unanchored (plain) trials — from the ledger
  (parent_trial_id set + anchored run records vs not).

All hooks are read-only, fail closed on flag-off for the querying handle
when a handle is involved (MetricError(flag_off)), and never expose raw
data — aggregates only (v3 §12).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sift import flags
from sift.cards import CardRegistry
from sift.cards.share import ShareService
from sift.cards.vault import Vault
from sift.registry import Ledger

DAY_S = 24 * 3600


class MetricError(Exception):
    """Metric-hook misuse (flag off, missing store, ...)."""


@dataclass(frozen=True)
class RetentionSplit:
    cohort: str                 # "card_holders" | "non_card_holders"
    users: int
    active_d90: int
    rate_d90: float


@dataclass(frozen=True)
class GhostRatio:
    total_cards: int
    active_ghosts: int
    ghost_ratio: float


@dataclass(frozen=True)
class VaultTurnover:
    window_days: int
    active_vault_cards: int
    discards: int
    discards_per_card: float


@dataclass(frozen=True)
class ShareGrantRetention:
    granted_users: int
    granted_active_d30: int
    granted_rate_d30: float
    non_granted_users: int
    non_granted_active_d30: int
    non_granted_rate_d30: float


@dataclass(frozen=True)
class AnchoredHitRate:
    anchored_trials: int
    anchored_survivors: int
    anchored_hit_rate: float
    unanchored_trials: int
    unanchored_survivors: int
    unanchored_hit_rate: float


class Metrics:
    """Read-only gate-1 metric hooks over the server-side stores. When the
    measurement pipeline (event store keyed by opaque user_id) lands, these
    become queries over it; shapes stay the same."""

    def __init__(self, ledger: Ledger, cards: CardRegistry, vault: Vault,
                 shares: ShareService | None = None):
        self.ledger = ledger
        self.cards = cards
        self.vault = vault
        self.shares = shares

    def _require_flag(self, handle: str | None) -> None:
        if not flags.cards_v4_1_enabled(handle):
            raise MetricError("flag_off: cards_v4_1 is disabled for this account")

    # -- activity (the "active at D" definition, beta-plan §4) --------------------

    def _active_handles(self, day: int, now: float) -> set[str]:
        """Handles with >=1 NEW unique trial registered in the trailing 30d
        window ending at day D: (now - D - 30d, now - D]. Uses the ledger
        only (skip-list means re-runs are not new trials)."""
        end = now - day * DAY_S
        start = end - 30 * DAY_S
        rows = self.ledger.con.execute(
            "SELECT DISTINCT discoverer_handle h FROM trial"
            " WHERE discoverer_handle IS NOT NULL AND registered_at > ? AND registered_at <= ?",
            (start, end),
        ).fetchall()
        return {r["h"] for r in rows}

    def _all_handles(self) -> set[str]:
        rows = self.ledger.con.execute(
            "SELECT DISTINCT discoverer_handle h FROM trial WHERE discoverer_handle IS NOT NULL"
        ).fetchall()
        return {r["h"] for r in rows}

    def _card_holders(self) -> set[str]:
        rows = self.cards.con.execute(
            "SELECT DISTINCT owner_handle h FROM card"
        ).fetchall()
        return {r["h"] for r in rows}

    # -- hooks -----------------------------------------------------------------------

    def d90_retention_split(self, handle: str | None = None,
                            now: float | None = None) -> list[RetentionSplit]:
        """D90 retention split: card-holders vs non-card-holders. Active per
        beta-plan §4 (>=1 new unique trial in the trailing 30 days ending at
        D90). INSTRUMENTATION STUB over the ledger + card registry."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        active = self._active_handles(90, now)
        holders = self._card_holders()
        all_h = self._all_handles()
        out = []
        for cohort, members in (("card_holders", holders), ("non_card_holders", all_h - holders)):
            n = len(members)
            a = len(members & active)
            out.append(RetentionSplit(cohort=cohort, users=n, active_d90=a,
                                      rate_d90=(a / n) if n else 0.0))
        return out

    def ghost_ratio(self, handle: str | None = None) -> GhostRatio:
        """Active ghosts / total cards — registry pool health (mechanics 3)."""
        self._require_flag(handle)
        total = self.cards.con.execute("SELECT COUNT(*) c FROM card").fetchone()["c"]
        ghosts = self.cards.con.execute(
            "SELECT COUNT(*) c FROM card WHERE kind='ghost' AND ghost_no>0"
        ).fetchone()["c"]
        return GhostRatio(total_cards=total, active_ghosts=ghosts,
                          ghost_ratio=(ghosts / total) if total else 0.0)

    def vault_turnover(self, window_days: int = 30, handle: str | None = None,
                       now: float | None = None) -> VaultTurnover:
        """Discards per active vault card over the window — vault churn."""
        self._require_flag(handle)
        now = time.time() if now is None else now
        active = self.vault.con.execute(
            "SELECT COUNT(*) c FROM vault_entry WHERE state='vault'"
        ).fetchone()["c"]
        since = now - window_days * DAY_S
        discards = self.vault.con.execute(
            "SELECT COUNT(*) c FROM registry_pool WHERE discarded_at >= ?", (since,)
        ).fetchone()["c"]
        return VaultTurnover(window_days=window_days, active_vault_cards=active,
                             discards=discards,
                             discards_per_card=(discards / active) if active else 0.0)

    def share_grant_d30_retention(self, handle: str | None = None,
                                  now: float | None = None) -> ShareGrantRetention:
        """D30 retention of share-grant recipients vs non-recipients."""
        self._require_flag(handle)
        if self.shares is None:
            raise MetricError("share service not wired — no share-grant metric")
        now = time.time() if now is None else now
        active = self._active_handles(30, now)
        granted = {r["owner"] for r in self.shares.con.execute(
            "SELECT DISTINCT owner FROM share_grant WHERE outcome='granted'").fetchall()}
        all_h = self._all_handles()
        g_users = granted & all_h
        n_users = all_h - granted
        g_act = len(g_users & active)
        n_act = len(n_users & active)
        return ShareGrantRetention(
            granted_users=len(g_users), granted_active_d30=g_act,
            granted_rate_d30=(g_act / len(g_users)) if g_users else 0.0,
            non_granted_users=len(n_users), non_granted_active_d30=n_act,
            non_granted_rate_d30=(n_act / len(n_users)) if n_users else 0.0,
        )

    def anchored_vs_unanchored_hit_rate(self, handle: str | None = None) -> AnchoredHitRate:
        """Survivor rate of anchored-run sampled candidates (parent_trial_id
        set — anchored runs register with parent_trial_id=anchor) vs plain
        trials. Hit = survivor verdict. From the ledger only."""
        self._require_flag(handle)
        a = self.ledger.con.execute(
            "SELECT COUNT(*) n, SUM(CASE WHEN verdict='survivor' THEN 1 ELSE 0 END) s"
            " FROM trial WHERE parent_trial_id IS NOT NULL"
        ).fetchone()
        u = self.ledger.con.execute(
            "SELECT COUNT(*) n, SUM(CASE WHEN verdict='survivor' THEN 1 ELSE 0 END) s"
            " FROM trial WHERE parent_trial_id IS NULL"
        ).fetchone()
        an,asun = int(a["n"] or 0), int(a["s"] or 0)
        un,usu = int(u["n"] or 0), int(u["s"] or 0)
        return AnchoredHitRate(
            anchored_trials=an, anchored_survivors=asun,
            anchored_hit_rate=(asun / an) if an else 0.0,
            unanchored_trials=un, unanchored_survivors=usu,
            unanchored_hit_rate=(usu / un) if un else 0.0,
        )


__all__ = [
    "AnchoredHitRate",
    "GhostRatio",
    "MetricError",
    "Metrics",
    "RetentionSplit",
    "ShareGrantRetention",
    "VaultTurnover",
]
