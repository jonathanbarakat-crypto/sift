"""v4.2 registry decay re-evaluation (mechanics 7/8 — BIDIRECTIONAL, pre-registered).

Owner decision 2026-11-17: card-tier decay changes from downgrade-only to
**bidirectional with strict pre-registered upgrade rules** (v4.1 spec §2/§4 +
v3 §6.1):

- A card's tier is a STATIC function of gauntlet strength at mint
  (tier_from_stats). It is NEVER changed by payment or chance. On registry
  re-evaluation it can move DOWN (amber, from registry evidence, any time)
  or UP — but an upgrade only fires when ALL pre-registered rules hold:

  1. FULLY OUT-OF-SAMPLE LIVE WINDOW ONLY: the card's age must exceed
     12 months (WINDOW_BARS = 252 trading days), so the entire trailing
     window is bars the mint-time gauntlet never saw. Before 12m the
     window overlaps mint-time data (in-sample echo) — upgrades refused.
  2. PRE-REGISTERED THRESHOLDS ON RAW STATS, IDENTICAL TO THE MINT-TIME
     BAR IN WINDOW-LENGTH-INVARIANT TERMS (soundness #5 fix, hermes-tasks
     #181): an upgrade silver->gold requires the trailing-12m live
     re-evaluation to meet the SAME bar `tier_from_stats` uses for gold at
     mint. The mint bar is a FULL-DEPTH (MINT_T_FULL = 4032-bar) t of 4.0
     (platinum) / 3.0 (silver and gold's full-depth line), and t scales
     with sqrt(T), so on a window of T_win bars the IDENTICAL bar is
     |t_win| >= B * sqrt(T_win / MINT_T_FULL)  <=>  |sr_ann| >= B *
     sqrt(BARS_PER_YEAR / MINT_T_FULL) — annualized Sharpe, invariant to
     the window length (at 252 bars: platinum 1.0, silver 0.75). Comparing
     a 252-bar window's RAW t against the raw full-depth bar 4.0 would
     demand annSR ~4.0, ~4x stricter than mint — soundness finding #5:
     genuine edges essentially never upgraded. Per owner decision
     2026-11-18 (hermes-tasks #170) the bars stay RAW and N-INDEPENDENT:
     DSR is NOT part of the live-window bar — decay/re-evaluation NEVER
     re-deflates by a current stratum N. The mint-time DSR gate lives only
     at mint (CardRegistry.mint). p_perm rules unchanged (platinum/gold:
     p_perm <= 0.01, silver: <= 0.05). No operator discretion.
  3. SUSTAINED, NOT SPIKY: the qualifying stats must hold on the trailing
     12m AND the preceding 12m (two consecutive annual windows).
  4. RATE LIMIT: at most one tier change (up OR down) per card per 90 days.
  5. DETERMINISTIC + SEEDED re-evaluation (PCG64), same as S2.
  6. APPEND-ONLY LEDGER: every upgrade is written as a `card_upgrade` event
     with evidence (window bounds, stats, config hash, vintage); read-back
     via events(). Never by payment or chance — that invariant and its test
     are kept.
  7. PUBLIC explanation on upgrade, rarity language only (see
     `public_upgrade_explanation`) — covered by the banned-phrase test.
  8. Downgrade behavior is UNCHANGED: down-only still allowed any time from
     registry evidence (amber), still via `card_decay` events.

- HEALTH is the one N-SENSITIVE quantity, deliberately (owner decision
  2026-11-18, #170): health = the card's trailing-12m live-window Sharpe
  vs the expected max Sharpe of N_current random strategies over the SAME
  window (a current-window DSR-style deflation, `health_ratio` below),
  display-only, never a tier, downgrade-only until fully out-of-sample.
  See `health_ratio` / `health_facts`.

- The registry pass (`RegistryDecayJob.run_pass`) re-evaluates every card
  against fresh registry evidence (a `re_evaluator` callable supplied by
  the engine; default `default_re_evaluator` re-derives the tier from the
  card's own pre-registered RAW ledger stats — S2 p, S4 bar, S5 t). A LOWER
  proposal downgrades (written to the card row, decayed_from recorded,
  `card_decay` event appended, owner notified through the per-card
  channel). A HIGHER proposal is NOT applied blindly: it is checked against
  the pre-registered live window rules by `live_window_evaluator` (age > 12m,
  two consecutive qualifying annual windows, rate limit, seeded
  re-evaluation) and only then applied as a `card_upgrade` event.
- The amber transition is PUBLIC and EXPLAINED: `public_explanation`
  returns the public text shown on the card face / registry page. A card
  that can visibly die is the core honesty artifact (v3 §6.1). The green
  upgrade transition is public and explained too, with rarity language
  only ("re-validated on live data — tier raised").
- Cadence: monthly registry pass is the candidate (v4.1 spec §8 open
  question 5) — the job is cadence-agnostic; the caller schedules it.

Flag: behind `cards_v4_1`; flag off -> DecayError(flag_off) on the pass.
Kill-switch env: SIFT_CARDS_V4_1=0 (sift/flags.py).

Copy discipline (legal posture §6): rarity language only — never "value
appreciates". The public explanation texts live in
`public_explanation` / `public_upgrade_explanation` and are covered by
banned-phrase tests. Public cards show survival facts only.

Read-back discipline (hard rule): every transition returns the card
re-read from the DB, and the ledger event is read back via events().
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from sift import flags
from sift.cards import CARD_TIERS, Card, CardError, CardRegistry, decay_tier, tier_from_stats
from sift.cards.notify import Notifier, contains_banned
from sift.gauntlet import config as gcfg
from sift.gauntlet import stats as gstats
from sift.registry import Ledger

# The public event types appended to the trial ledger.
DECAY_EVENT = "card_decay"        # downgrade (amber) — unchanged behavior
UPGRADE_EVENT = "card_upgrade"    # pre-registered live-window upgrade (green)

# -- pre-registered upgrade constants (owner decision 2026-11-17) -------------
# 12 months of trading days = 252 bars (config.BARS_PER_YEAR).
WINDOW_BARS = gcfg.BARS_PER_YEAR            # 252: one annual window in bars
MIN_CARD_AGE_BARS = WINDOW_BARS             # card age must EXCEED this (rule 1)
RATE_LIMIT_S = 90 * 86400.0                # max one tier change per card per 90d (rule 4)
# RAW, N-INDEPENDENT live-window bars per tier (owner decision 2026-11-18,
# hermes-tasks #170): identical to `tier_from_stats` at mint, with DSR REMOVED
# (decay/re-evaluation never re-deflates by a current N). The numeric bars
# that map 1:1 to raw stats are kept verbatim from pull/31.
SILVER_T_NET, SILVER_P_PERM = 3.0, 0.05    # silver: |t_net_full| >= 3.0 AND p_perm <= 0.05
GOLD_P_PERM = 0.01                          # gold: p_perm <= 0.01 (S4 bar is a mint-time scope fact)
PLATINUM_T_NET, PLATINUM_P_PERM = 4.0, 0.01 # platinum: |t_net_full| >= 4.0 AND p_perm <= 0.01
# Deprecated aliases kept for callers/tests that referenced the pre-#170
# DSR-carrying bars; they now hold the RAW values (no DSR anywhere).
GOLD_DSR, GOLD_T_NET = 0.99, 4.0            # legacy: platinum bar's old names
SILVER_DSR = 0.95                           # legacy: the mint-time admission gate value
PLATINUM_DSR, PLATINUM_T_NET_ALIAS, PLATINUM_P_PERM = 0.99, 4.0, 0.01

# Mint-time FULL-DEPTH window length in bars (registry_depth scope — the same
# data tier_from_stats sees at mint via s5.statistic). The live-window t
# hurdle is derived from the full-depth anchors above by sqrt(T) scaling, so
# the live-window bar is IDENTICAL to the mint-time bar in invariant
# (annualized-Sharpe) terms, not in raw-t terms (soundness #5,
# hermes-tasks #181 finding #5 P1).
MINT_T_FULL = gcfg.SCOPE_T[gcfg.S5_SCOPE]   # 4032


def window_t_hurdle(t_net_full_bar: float, window_bars: int) -> float:
    """Window-length-INVARIANT form of a full-depth t bar (soundness #5).

    t_T = sr_ann * sqrt(T / BARS_PER_YEAR), so a full-depth bar
    |t_full| >= B at T = MINT_T_FULL is equivalent, on ANY window of length
    T_win, to

        |t_win| >= B * sqrt(T_win / MINT_T_FULL)
              <=> |sr_ann| >= B * sqrt(BARS_PER_YEAR / MINT_T_FULL)

    The annSR form is the fully window-length-invariant statement of the
    mint bar: platinum 4.0 -> annSR >= 1.0, silver 3.0 -> annSR >= 0.75, at
    ANY window length. Comparing a 252-bar window's RAW t against the raw
    full-depth bar 4.0 would demand annSR ~4.0 — ~4x stricter than mint —
    which is the defect (hermes-tasks #181 finding #5): genuine edges
    essentially never upgraded. At T_win == MINT_T_FULL the raw bar is
    returned verbatim, so full-depth windows are not weakened.
    """
    return t_net_full_bar * math.sqrt(max(int(window_bars), 1) / MINT_T_FULL)

# Default re-evaluator: re-derive the tier from the card's own
# pre-registered RAW ledger stats (S2 p_perm, S4 bar, S5 full-depth t_net).
# v4.2 §3 (#170): the S3 DSR row is read ONLY to record mint-time admission
# context — it is NOT passed to tier_from_stats, so the re-evaluated tier
# never drifts with the stratum N. The registry pass may move a tier DOWN on
# this evidence at any time; a HIGHER proposal is never applied directly —
# it must pass the pre-registered live-window rules (live_window_evaluator).


def default_re_evaluator(card: Card, ledger: Ledger) -> str:
    """Re-evaluate a card's tier from its pre-registered RAW ledger stats.
    Pure read of stage rows; returns one of CARD_TIERS (v4.2 incl. platinum).
    DSR is deliberately NOT an input (N-independence, #170).

    soundness #10 note: the stage rows are append-only and never change, so
    this fallback re-derives the MINT tier and can never downgrade on its own.
    The engine wires `live_re_evaluator(...)` (below) as the production
    re_evaluator so the registry pass re-evaluates against FRESH live
    evidence; this default remains the documented fail-safe fallback when no
    live series is wired (no-op, never a fabricated change)."""
    s2 = ledger.stage(card.trial_id, "s2")
    s4 = ledger.stage(card.trial_id, "s4")
    s5 = ledger.stage(card.trial_id, "s5")
    return tier_from_stats(
        dsr=None,                       # NOT a tier input (#170) — raw stats only
        t_net_full=s5["statistic"], p_perm=s2["p_value"],
        s4_pass=s4["status"] == "pass", s5_pass=s5["status"] == "pass",
    )


def live_re_evaluator(live_returns_fn, *, seed: int = 0,
                      s2_B: int = gcfg.S2_B) -> Callable[[Card, Ledger], str]:
    """Build a re_evaluator that re-derives a card's tier from FRESH live
    evidence (soundness #10): the trailing WINDOW_BARS live window of the
    card's post-mint return series, computed exactly as the mint-time
    gauntlet computes them (S2 block-permutation p + window t via
    `_window_stats`), mapped through the SAME pre-registered raw bars
    (`meets_tier_bar`) — platinum/gold/silver/bronze, fail-closed on
    missing stats.

    `live_returns_fn(card) -> np.ndarray | None` supplies the card's
    post-mint live net-return series (server-side; None -> the evaluator
    falls back to the static ledger stats via default_re_evaluator, i.e.
    no-op — never a fabricated tier).

    The proposal is a TIER PROPOSAL only: a LOWER proposal downgrades
    through the amber path; a HIGHER one still needs the pre-registered
    live-window upgrade rules (live_window_evaluator) — this factory does
    not relax them.
    """
    def _re(card: Card, ledger: Ledger) -> str:
        rets = live_returns_fn(card) if live_returns_fn is not None else None
        if rets is None:
            return default_re_evaluator(card, ledger)
        r = np.asarray(rets, float)
        r = r[np.isfinite(r)]
        if r.size < 2:
            return default_re_evaluator(card, ledger)
        n, v = ledger.stratum_stats(card.scope_id)
        end = int(r.size)
        start = max(0, end - WINDOW_BARS)
        ws = _window_stats(r, start=start, end=end,
                           n_stratum=n, v_stratum=v, seed=seed, s2_B=s2_B)
        # map the window evidence through the pre-registered bars, highest
        # tier whose bar the window meets (fail-closed on None)
        if meets_tier_bar(ws, "platinum"):
            return "platinum"
        if meets_tier_bar(ws, "gold"):
            return "gold"
        if meets_tier_bar(ws, "silver"):
            return "silver"
        return "bronze"
    return _re


def default_live_evaluator(live_returns_fn, *, n_stratum: int = 1,
                           v_stratum: float = 0.0, seed: int = 0,
                           s2_B: int = gcfg.S2_B) -> Callable[[Card], LiveWindowEvidence]:
    """Build the live_window_evaluator over a server-supplied live series
    (soundness #10): `live_returns_fn(card) -> np.ndarray | None` gives the
    card's post-mint live net-return series; `now_bar` is its current length.
    With no series for a card the evidence is INELIGIBLE (fail closed —
    upgrades refused), never fabricated."""
    def _ev(card: Card) -> LiveWindowEvidence:
        rets = live_returns_fn(card) if live_returns_fn is not None else None
        if rets is None:
            r = np.zeros(0)
        else:
            r = np.asarray(rets, float)
            r = r[np.isfinite(r)]
        return live_window_evaluator(
            card,
            live_returns=r,
            now_bar=int(r.size),
            n_stratum=n_stratum,
            v_stratum=v_stratum,
            seed=seed,
            s2_B=s2_B,
        )
    return _ev


# ---------------------------------------------------------------------------
# Live-window re-evaluation (the ONLY path that can raise a tier)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowStats:
    """The pre-registered RAW stats on one annual live window (v4.2 §3,
    #170: N-independent — `dsr` is kept ONLY as current-stratum context for
    the HEALTH ratio, never as a tier input)."""
    window_start_bar: int
    window_end_bar: int
    dsr: float | None          # current-stratum deflation — HEALTH context only
    t_net_full: float | None
    p_perm: float | None


def _window_stats(returns: np.ndarray, *, start: int, end: int,
                  n_stratum: int, v_stratum: float, seed: int,
                  s2_B: int) -> WindowStats:
    """Compute the pre-registered stats on bars [start, end) of the
    live window, exactly as the mint-time gauntlet computes them:

    - S2 block-permutation p on the window's spread returns (PCG64-seeded,
      same as S2: `block_permutation_null`);
    - S4 walk-forward net returns at full depth -> |t_net| (t_stat_sharpe);
    - S3-style DSR with the CURRENT stratum denominator — recorded as
      current-stratum context for the HEALTH ratio only; per #170 it is
      NEVER used by meets_tier_bar (the tier bar is raw, N-independent).

    Deterministic for a given (returns, window, seed, s2_B, stratum).

    NOTE on the t component (soundness #5): `t_net_full` here is the RAW t
    of the WINDOW (t_stat_sharpe on [start, end)), which scales with
    sqrt(T_window) and therefore must NOT be compared against the raw
    full-depth mint bar (e.g. PLATINUM_T_NET = 4.0 at MINT_T_FULL = 4032
    bars). meets_tier_bar rescales the bar by sqrt(T_window / MINT_T_FULL)
    via window_t_hurdle, making the comparison window-length invariant
    (equivalently: annualized Sharpe — platinum annSR >= 1.0, silver >=
    0.75).
    """
    r = np.asarray(returns, float)[start:end]
    r = r[np.isfinite(r)]
    T = int(r.size)
    if T < 2:
        return WindowStats(start, end, None, None, None)
    s2 = gstats.block_permutation_null(r, gstats.mean_reversion_stat,
                                       B=s2_B, seed=seed)
    p_perm = s2["p_perm"]
    # full-depth walk-forward net on the window: use the window returns as
    # the net series (costs already folded in by the caller when it built
    # `returns`); t_net = t_stat_sharpe(net).
    t_net = gstats.t_stat_sharpe(r) if T >= 2 else None
    sr_ann = gstats.sharpe_ratio(r) if T >= 2 else 0.0
    sr_bar = sr_ann / math.sqrt(gcfg.BARS_PER_YEAR)
    skew, kurt = gstats.skew_kurt(r) if T >= 4 else (0.0, 3.0)
    d = gstats.dsr(sr_bar, max(T, 2), max(n_stratum, 1), v_stratum,
                   skew=skew, kurt=kurt)
    return WindowStats(start, end, d["dsr"], t_net, p_perm)


def meets_tier_bar(ws: WindowStats, tier: str) -> bool:
    """Rule 2: does one annual window meet the pre-registered RAW bar for
    `tier`? The bar is IDENTICAL to `tier_from_stats` at mint, expressed in
    WINDOW-LENGTH-INVARIANT terms (soundness #5, hermes-tasks #181): the
    mint-time bar is a full-depth (MINT_T_FULL-bar) t of 4.0 (platinum) /
    3.0 (silver), and t scales with sqrt(T), so on a window of
    `ws.window_end_bar - ws.window_start_bar` bars the equivalent hurdle is

        platinum: |t_win| >= 4.0 * sqrt(T_win / MINT_T_FULL)  (1.0 at 252)
                  AND p_perm <= 0.01
        silver:   |t_win| >= 3.0 * sqrt(T_win / MINT_T_FULL)  (0.75 at 252)
                  AND p_perm <= 0.05
        gold:     p_perm <= 0.01 (unchanged; the S4 scope bar is a
                  mint-time fact, not re-derivable per live window)

    Equivalently in annualized Sharpe (window-length invariant):
    platinum |sr_ann| >= 1.0, silver |sr_ann| >= 0.75. At T_win ==
    MINT_T_FULL the raw bars 4.0 / 3.0 are used verbatim — no weakening.
    The bars stay RAW and N-INDEPENDENT per #170: `ws.dsr` is ignored here
    on purpose. Missing stats never qualify."""
    if ws.t_net_full is None or ws.p_perm is None:
        return False
    t_win = ws.window_end_bar - ws.window_start_bar
    if tier == "platinum":
        return (abs(ws.t_net_full) >= window_t_hurdle(PLATINUM_T_NET, t_win)
                and ws.p_perm <= PLATINUM_P_PERM)
    if tier == "gold":
        # gold's live-window bar is the raw p_perm <= 0.01 line; the S4
        # scope bar is mint-time only. (Same numeric spirit as pull/31's
        # gold bar minus DSR so the live-window bar never drifts with N.)
        return ws.p_perm <= GOLD_P_PERM
    if tier == "silver":
        return (abs(ws.t_net_full) >= window_t_hurdle(SILVER_T_NET, t_win)
                and ws.p_perm <= SILVER_P_PERM)
    return False


# ---------------------------------------------------------------------------
# Health — the deliberately N-SENSITIVE quantity (owner decision 2026-11-18,
# hermes-tasks #170). Display-only, never a tier, downgrade-only until the
# trailing window is fully out-of-sample.
# ---------------------------------------------------------------------------

def health_ratio(live_returns: np.ndarray, *, start: int, end: int,
                 n_stratum_current: int, v_stratum_current: float) -> float | None:
    """N-sensitive health ratio on one live window (v4.2 §3, #170):

    health = (card's live-window annualized Sharpe)
             / (expected max Sharpe of N_current random strategies over the
                SAME window)  — i.e. sr_hat / SR*(N_current, V̄_current).

    The denominator is a CURRENT-stratum DSR-style deflation, so when many
    new candidates are minted in the same regime the luck ceiling SR* rises
    with N_current and the health ratio DROPS for unchanged card
    performance; with unchanged N the ratio is stable. Returns None when
    the window has too few bars or the denominator is degenerate (never
    fabricated). Display-only: never a tier, never an upgrade input.
    """
    r = np.asarray(live_returns, float)[start:end]
    r = r[np.isfinite(r)]
    T = int(r.size)
    if T < 2:
        return None
    sr_ann = gstats.sharpe_ratio(r)
    sr_bar = sr_ann / math.sqrt(gcfg.BARS_PER_YEAR)
    n = max(int(n_stratum_current), 1)
    # #1/#16: v_stratum_current is PER-BAR SR^2 (ledger convention); floor it
    # with the PSR normal-theory per-bar variance (1 + sr_bar^2/2)/T, the same
    # floor stats.dsr applies, so sr_star shares per-bar units with sr_bar.
    v = max(float(v_stratum_current), (1.0 + sr_bar ** 2 / 2.0) / max(T, 2))
    sr_star = gstats.expected_max_sharpe(n, v)
    if not (sr_star > 0) or not math.isfinite(sr_star):
        return None
    return float(sr_bar / sr_star)


@dataclass(frozen=True)
class LiveWindowEvidence:
    """The full pre-registered evidence for an upgrade decision."""
    eligible: bool                       # all rules hold
    reason: str                          # refusal reason when not eligible
    target_tier: str | None              # the tier the evidence supports
    trailing: WindowStats | None         # trailing 12m window stats
    preceding: WindowStats | None        # preceding 12m window stats
    card_age_bars: int
    seed: int


def live_window_evaluator(
    card: Card,
    *,
    live_returns: np.ndarray,
    now_bar: int,
    n_stratum: int,
    v_stratum: float,
    seed: int = 0,
    s2_B: int = gcfg.S2_B,
    last_tier_change_at: float | None = None,
    now: float | None = None,
) -> LiveWindowEvidence:
    """Evaluate the pre-registered live-window upgrade rules for one card.

    All rules must hold (owner decision 2026-11-17):
      1. age: now_bar - card.minted_bar > WINDOW_BARS (fully OOS window);
      2. the trailing 12m window meets the mint-time bar for the next tier
         up (silver->gold bar / bronze->silver bar) — the SAME thresholds as
         `tier_from_stats`, in window-length-invariant (annualized-Sharpe)
         terms via window_t_hurdle: |t_win| >= B*sqrt(252/MINT_T_FULL), NOT
         the raw full-depth t (soundness #5);
      3. the SAME stats also hold on the preceding 12m (two consecutive
         annual windows — sustained, not spiky);
      4. rate limit: no tier change in the last 90 days;
      5. deterministic + seeded (PCG64) — the seed is recorded in evidence.

    `live_returns` is the card's post-mint live return series (server-side
    only); `now_bar` is the current bar index in that series;
    `card.minted_at`/`minted_bar` anchor the window. Returns the evidence;
    `eligible` is True only when every rule holds.
    """
    now = time.time() if now is None else now
    minted_bar = int(getattr(card, "minted_bar", 0) or 0)
    age_bars = int(now_bar) - minted_bar
    # rule 1: fully out-of-sample window only
    if age_bars <= MIN_CARD_AGE_BARS:
        return LiveWindowEvidence(
            eligible=False,
            reason=(f"in-sample echo: card age {age_bars} bars <= "
                    f"{MIN_CARD_AGE_BARS} (trailing window overlaps mint-time "
                    f"data) — upgrades refused"),
            target_tier=None, trailing=None, preceding=None,
            card_age_bars=age_bars, seed=seed,
        )
    # rule 4: rate limit (one tier change per card per 90 days)
    if last_tier_change_at is not None and (now - last_tier_change_at) < RATE_LIMIT_S:
        return LiveWindowEvidence(
            eligible=False,
            reason=(f"rate limit: last tier change "
                    f"{now - last_tier_change_at:.0f}s ago < {RATE_LIMIT_S:.0f}s "
                    f"(90d) — at most one tier change per card per 90 days"),
            target_tier=None, trailing=None, preceding=None,
            card_age_bars=age_bars, seed=seed,
        )
    # the next tier up from the card's current tier
    idx = CARD_TIERS.index(card.tier)
    if idx >= len(CARD_TIERS) - 1:
        return LiveWindowEvidence(
            eligible=False, reason="already at the top tier (platinum)",
            target_tier=None, trailing=None, preceding=None,
            card_age_bars=age_bars, seed=seed,
        )
    target = CARD_TIERS[idx + 1]
    # rules 2+3: two consecutive annual windows must meet the bar
    w = WINDOW_BARS
    end = int(now_bar)
    start = end - w
    if start < 0:
        return LiveWindowEvidence(
            eligible=False, reason="live window shorter than 252 bars",
            target_tier=None, trailing=None, preceding=None,
            card_age_bars=age_bars, seed=seed,
        )
    trailing = _window_stats(live_returns, start=start, end=end,
                             n_stratum=n_stratum, v_stratum=v_stratum,
                             seed=seed, s2_B=s2_B)
    preceding = _window_stats(live_returns, start=start - w, end=start,
                              n_stratum=n_stratum, v_stratum=v_stratum,
                              seed=seed, s2_B=s2_B)
    if not meets_tier_bar(trailing, target):
        return LiveWindowEvidence(
            eligible=False,
            reason=(f"trailing {w}m live window below the pre-registered "
                    f"{target} bar (window-length-invariant hurdle |t_win| >= "
                    f"{window_t_hurdle(PLATINUM_T_NET if target == 'platinum' else SILVER_T_NET, w):.4f} "
                    f"at {w} bars; |t_net| {trailing.t_net_full}, "
                    f"p_perm {trailing.p_perm})"),
            target_tier=target, trailing=trailing, preceding=preceding,
            card_age_bars=age_bars, seed=seed,
        )
    if not meets_tier_bar(preceding, target):
        return LiveWindowEvidence(
            eligible=False,
            reason=(f"preceding {w}m live window below the pre-registered "
                    f"{target} bar — not sustained across two consecutive "
                    f"annual windows"),
            target_tier=target, trailing=trailing, preceding=preceding,
            card_age_bars=age_bars, seed=seed,
        )
    return LiveWindowEvidence(
        eligible=True,
        reason=(f"re-validated on live data: trailing and preceding {w}m "
                f"windows both meet the pre-registered {target} bar on RAW "
                f"stats (fully out-of-sample, card age {age_bars} bars)"),
        target_tier=target, trailing=trailing, preceding=preceding,
        card_age_bars=age_bars, seed=seed,
    )


# ---------------------------------------------------------------------------
# Public explanations (rarity language only — banned-phrase fail-closed)
# ---------------------------------------------------------------------------

def _assert_no_banned(text: str) -> None:
    """Fail-closed guard shared with notify.copy_for_event: raises
    DecayError on the first banned phrase/word (substring scan over
    BANNED_PHRASES + whole-word scan over directional words) so every
    public explanation string is covered by the same grep discipline."""
    banned = contains_banned(text)
    if banned is not None:
        raise DecayError(
            f"public explanation contains banned phrase {banned!r} — refused"
        )


def public_explanation(card_id: str, from_tier: str, to_tier: str,
                       reason: str) -> str:
    """The PUBLIC explanation text attached to an amber downgrade (v3 §6.1:
    publicly visible with explanation). Rarity language only — fail closed
    on any banned (appreciation/urgency) phrase."""
    text = (
        f"Card {card_id} was re-evaluated by the registry pass and its rarity "
        f"tier moved from {from_tier} to {to_tier} (amber). Reason: {reason}. "
        f"Tiers are set by gauntlet strength at mint and move down on "
        f"re-evaluation; they can also move up only when pre-registered "
        f"live-window rules are met — never by payment or chance. This is a "
        f"survival fact, not a recommendation; SIFT does not trade."
    )
    _assert_no_banned(text)
    return text


def public_upgrade_explanation(card_id: str, from_tier: str, to_tier: str,
                               reason: str) -> str:
    """The PUBLIC explanation text attached to a green upgrade (rule 7).
    Rarity language only: "re-validated on live data — tier raised".
    Fail closed on any banned phrase (no "more valuable", "appreciated",
    "gained value", no profit/return numbers)."""
    text = (
        f"Card {card_id} was re-validated on live data — its rarity tier was "
        f"raised from {from_tier} to {to_tier} (green). Reason: {reason}. "
        f"The upgrade fired only because the pre-registered live-window rules "
        f"were met on two consecutive annual windows that are fully "
        f"out-of-sample; upgrades are never granted by payment or chance. "
        f"This is a survival fact, not a recommendation; SIFT does not trade."
    )
    _assert_no_banned(text)
    return text


class DecayError(Exception):
    """Decay misuse: flag off, upgrade attempt without eligibility, unknown card, ..."""


@dataclass(frozen=True)
class DecayResult:
    """One card's outcome of a registry re-evaluation pass."""
    card_id: str
    owner: str
    from_tier: str
    to_tier: str            # == from_tier when unchanged
    downgraded: bool
    upgraded: bool = False
    ledger_event_id: int | None = None
    notified: bool = False  # whether a notification was enqueued for the owner
    public_text: str | None = None  # public explanation when the tier moved


class RegistryDecayJob:
    """Registry re-evaluation pass (spec §8 Q5 candidate: monthly cadence).

    BIDIRECTIONAL with strict pre-registered upgrade rules:
    - a re-evaluator proposal LOWER than the current tier downgrades
      (unchanged amber behavior: card row + `card_decay` ledger event +
      per-card notification);
    - a proposal HIGHER than the current tier is NEVER applied directly.
      It is checked by `live_window_evaluator` against the pre-registered
      rules (age > 12m fully-OOS window, two consecutive annual windows at
      the mint-time bar in window-length-invariant terms, 90d rate limit,
      seeded PCG64) and only then
      applied as a `card_upgrade` ledger event + notification;
    - anything else is a no-op.
    """

    def __init__(self, cards: CardRegistry, ledger: Ledger,
                 notifier: Notifier | None = None,
                 re_evaluator: Callable[[Card, Ledger], str] | None = None,
                 live_evaluator: Callable[[Card], LiveWindowEvidence] | None = None):
        self.cards = cards
        self.ledger = ledger
        self.notifier = notifier
        self.re_evaluator = re_evaluator or default_re_evaluator
        # The live-window evaluator supplies the evidence an upgrade needs.
        # Default: None -> no live series wired, so upgrades are refused
        # (fail closed) unless the engine supplies one.
        self.live_evaluator = live_evaluator

    def _require_flag(self) -> None:
        if not flags.cards_v4_1_enabled():
            raise DecayError("flag_off: cards_v4_1 is disabled (global kill-switch)")

    def _last_tier_change_at(self, card_id: str) -> float | None:
        """Rule 4 evidence: the most recent tier-change event (decay or
        upgrade) on this card, from the append-only ledger. None if none."""
        rows = self.ledger.con.execute(
            "SELECT created_at FROM ledger_event WHERE event_type IN (?,?)"
            " AND json_extract(payload_json, '$.card_id') = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (DECAY_EVENT, UPGRADE_EVENT, card_id),
        ).fetchall()
        return float(rows[0]["created_at"]) if rows else None

    def evaluate_card(self, card: Card, now: float | None = None) -> DecayResult:
        """Re-evaluate ONE card. Downgrades any time from registry
        evidence; upgrades only via the pre-registered live-window rules.
        Read-back on every write."""
        self._require_flag()
        now = time.time() if now is None else now
        proposed = self.re_evaluator(card, self.ledger)
        if proposed not in CARD_TIERS:
            raise DecayError(f"re-evaluator returned invalid tier {proposed!r}")
        cur_idx, prop_idx = CARD_TIERS.index(card.tier), CARD_TIERS.index(proposed)
        downgraded = upgraded = False
        target = card.tier
        event_id: int | None = None
        notified = False
        public_text: str | None = None
        event_type: str | None = None
        payload_extra: dict = {}

        if prop_idx < cur_idx:
            # ---- downgrade (unchanged amber path, rule 8: any time) ----
            try:
                target = decay_tier(card.tier, proposed)
            except CardError as exc:
                raise DecayError(
                    f"decay may only downgrade, got {card.tier} -> {proposed}: {exc}"
                ) from exc
            downgraded = True
            event_type = DECAY_EVENT
            reason = (
                "registry re-evaluation against pre-registered RAW gauntlet "
                "stats (S5 full-depth t_net, S4 bar, S2 permutation p — "
                "N-independent, no DSR re-deflation) no longer supports the "
                f"{card.tier} rarity tier"
            )
            public_text = public_explanation(card.card_id, card.tier, target, reason)
            payload_extra = {"reason": reason}
        elif prop_idx > cur_idx:
            # ---- upgrade candidate: pre-registered live-window rules ONLY ----
            # A higher proposal is NEVER applied directly — never by payment
            # or chance. Without eligible live-window evidence it raises
            # (loud refusal, same invariant as before).
            if self.live_evaluator is None:
                raise DecayError(
                    f"upgrade refused: {card.tier} -> {proposed} has no "
                    "pre-registered live-window evidence wired — tiers are "
                    "never raised by payment or chance"
                )
            ev = self.live_evaluator(card)
            # rule 4: rate limit — max one tier change (up or down) per card
            # per 90 days; any prior change within 90d blocks the upgrade.
            last = self._last_tier_change_at(card.card_id)
            if last is not None and (now - last) < RATE_LIMIT_S:
                raise DecayError(
                    f"rate_limit: last tier change of {card.card_id} was "
                    f"{now - last:.0f}s ago < {RATE_LIMIT_S:.0f}s (90d) — at "
                    "most one tier change per card per 90 days"
                )
            if not ev.eligible or ev.target_tier != proposed:
                raise DecayError(
                    f"upgrade refused: {card.tier} -> {proposed} fails the "
                    f"pre-registered live-window rules — {ev.reason}"
                )
            target = proposed
            upgraded = True
            event_type = UPGRADE_EVENT
            reason = ev.reason
            public_text = public_upgrade_explanation(
                card.card_id, card.tier, target, reason)
            trial = self.ledger.get(card.trial_id)
            payload_extra = {
                "reason": reason,
                "evidence": {
                    "card_age_bars": ev.card_age_bars,
                    "seed": ev.seed,
                    "window_bars": WINDOW_BARS,
                    "trailing_window": (
                        {"start_bar": ev.trailing.window_start_bar,
                         "end_bar": ev.trailing.window_end_bar,
                         "dsr": ev.trailing.dsr,
                         "t_net_full": ev.trailing.t_net_full,
                         "p_perm": ev.trailing.p_perm}
                        if ev.trailing else None),
                    "preceding_window": (
                        {"start_bar": ev.preceding.window_start_bar,
                         "end_bar": ev.preceding.window_end_bar,
                         "dsr": ev.preceding.dsr,
                         "t_net_full": ev.preceding.t_net_full,
                         "p_perm": ev.preceding.p_perm}
                        if ev.preceding else None),
                    "config_hash": trial.config_hash,
                    "data_vintage_hash": trial.data_vintage_hash,
                },
            }
        # else: equal -> no-op

        if downgraded or upgraded:
            # write the tier change to the card row (read-back inside):
            # downgrades keep the decay() semantics (decayed_from recorded);
            # upgrades write through the gated set_tier() path.
            if downgraded:
                updated = self.cards.decay(card.card_id, target, now=now)
            else:
                updated = self.cards.set_tier(card.card_id, target, now=now)
            # append the transition to the trial ledger (append-only event)
            event_id = self.ledger.record_event(
                event_type,
                {
                    "card_id": card.card_id,
                    "owner": card.owner_handle,
                    "kind": card.kind,
                    "from_tier": card.tier,
                    "to_tier": target,
                    "public_explanation": public_text,
                    **payload_extra,
                },
                trial_id=card.trial_id,
                now=now,
            )
            # read-back: event via events(), card row via get_card
            evs = self.ledger.events(event_type)
            assert any(e["event_id"] == event_id for e in evs)
            assert updated.tier == target
            if downgraded:
                assert updated.decayed_from == card.tier
            # notify the owner via the per-card channel (opt-in + caps apply)
            if self.notifier is not None:
                n = self.notifier.notify(
                    card.card_id, card.owner_handle,
                    "upgrade" if upgraded else "decay",
                    {"from_tier": card.tier, "to_tier": target,
                     "public_explanation": public_text},
                    now=now,
                )
                notified = n is not None
        return DecayResult(
            card_id=card.card_id, owner=card.owner_handle,
            from_tier=card.tier, to_tier=target, downgraded=downgraded,
            upgraded=upgraded, ledger_event_id=event_id, notified=notified,
            public_text=public_text,
        )

    def run_pass(self, now: float | None = None) -> list[DecayResult]:
        """The registry re-evaluation pass over ALL cards (regular + ghost).
        Returns one DecayResult per card, in mint order. Flag off ->
        DecayError(flag_off)."""
        self._require_flag()
        now = time.time() if now is None else now
        results: list[DecayResult] = []
        for card in self.cards.all_cards():
            results.append(self.evaluate_card(card, now=now))
        return results


__all__ = [
    "DECAY_EVENT",
    "UPGRADE_EVENT",
    "MIN_CARD_AGE_BARS",
    "RATE_LIMIT_S",
    "WINDOW_BARS",
    "SILVER_T_NET",
    "SILVER_P_PERM",
    "GOLD_P_PERM",
    "GOLD_DSR",
    "GOLD_T_NET",
    "SILVER_DSR",
    "PLATINUM_T_NET",
    "PLATINUM_P_PERM",
    "PLATINUM_DSR",
    "MINT_T_FULL",
    "DecayError",
    "DecayResult",
    "LiveWindowEvidence",
    "RegistryDecayJob",
    "WindowStats",
    "default_live_evaluator",
    "default_re_evaluator",
    "health_ratio",
    "live_re_evaluator",
    "live_window_evaluator",
    "meets_tier_bar",
    "public_explanation",
    "public_upgrade_explanation",
    "window_t_hurdle",
]
