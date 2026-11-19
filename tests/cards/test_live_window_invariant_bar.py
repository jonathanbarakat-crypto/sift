"""TDD fixtures for soundness #5 — kanban t_3c4076e1.

The live-window upgrade bar in sift/cards/decay.py must be WINDOW-LENGTH
INVARIANT. The mint-time gold bar is |t_net| >= 4.0 on the FULL-DEPTH
(4032-bar, registry_depth) series. Since t_T = sr_ann * sqrt(T / BARS_PER_YEAR),
the SAME bar expressed on a window of length T_win is

    |t_win| >= GOLD_T_NET * sqrt(T_win / T_FULL)
            = |sr_ann| >= GOLD_T_NET * sqrt(BARS_PER_YEAR / T_FULL)   (gold = 1.0)

so a 252-bar live window uses |t_win| >= 1.0 (annSR >= 1.0), NOT |t_win| >= 4.0.
Silver: |t_win| >= 0.75 (annSR >= 0.75). DSR and p_perm rules are unchanged.

Fixtures:
1. A realistic edge meeting the mint GOLD bar at full depth (t_full == 4.0)
   has a trailing 252-bar window with raw t ~= 1.0-1.5: under the OLD raw-t
   comparison meets_tier_bar said False; under the invariant bar it says True
   (with DSR/p_perm at passing values, as in the verdict-record probe).
2. The bar is not weakened in invariant terms: a window just below the
   invariant gold hurdle fails even with a huge raw t on a long window; a
   252-bar window at the hurdle passes with raw t = 1.0 << 4.0.
3. A null (zero-drift) window still fails; missing stats never qualify.
4. End-to-end through live_window_evaluator: the t-rule no longer blocks a
   mint-gold edge's live window.
"""

import math
import os
import tempfile

import numpy as np

from sift.cards import CardRegistry, tier_from_stats
from sift.cards.decay import (
    GOLD_T_NET, SILVER_T_NET, WINDOW_BARS,
    WindowStats, _window_stats, live_window_evaluator, meets_tier_bar,
    window_t_hurdle,
)
from sift.gauntlet import stats as gstats
from sift.gauntlet.config import BARS_PER_YEAR
from sift.registry import Ledger

T_FULL = 4032  # registry_depth / pro_full bars (gauntlet config SCOPE_T)

# invariant thresholds equivalent to the mint-time full-depth t bars
GOLD_SR_ANN = GOLD_T_NET * math.sqrt(BARS_PER_YEAR / T_FULL)      # 1.0
SILVER_SR_ANN = SILVER_T_NET * math.sqrt(BARS_PER_YEAR / T_FULL)  # 0.75
GOLD_T_WIN = GOLD_T_NET * math.sqrt(WINDOW_BARS / T_FULL)         # 1.0 at 252 bars
SILVER_T_WIN = SILVER_T_NET * math.sqrt(WINDOW_BARS / T_FULL)     # 0.75


def _edge_meeting_mint_gold_bar(seed=11):
    """A realistic stationary edge whose FULL-DEPTH t is exactly the mint
    gold bar (|t_full| = GOLD_T_NET): zero-mean AR(1) shape + drift chosen
    so t_stat_sharpe over T_FULL bars == 4.0."""
    rng = np.random.Generator(np.random.PCG64(seed))
    e = rng.normal(0.0, 1.0, T_FULL)
    s = np.zeros(T_FULL)
    for i in range(1, T_FULL):
        s[i] = -0.35 * s[i - 1] + e[i]
    base = s - 0.5 * e
    sd = base.std(ddof=1)
    drift = GOLD_T_NET * sd / math.sqrt(T_FULL) - base.mean()
    r = base + drift
    assert abs(abs(gstats.t_stat_sharpe(r)) - GOLD_T_NET) < 1e-9
    return r


def _win(r, start, end, seed=424242, s2_B=200):
    return _window_stats(r, start=start, end=end, n_stratum=50,
                         v_stratum=0.01, seed=seed, s2_B=s2_B)


# -- fixture 1: mint-gold edge qualifies on a 252-bar window (invariant bar) ---

def test_mint_gold_edge_window_t_rule_passes_under_invariant_bar():
    r = _edge_meeting_mint_gold_bar()
    # mint-time bar met at full depth -> gold at mint
    assert tier_from_stats(0.999, gstats.t_stat_sharpe(r), 0.001) == "gold"
    w = _win(r, T_FULL - WINDOW_BARS, T_FULL)
    # the defect's raw numbers: window t ~= 1.2 << 4.0
    assert 0.0 < abs(w.t_net_full) < GOLD_T_NET
    # verdict-record probe scenario: DSR and p_perm pass; only the t bar was
    # blocking. Rebuild the WindowStats with passing dsr/p and the REAL
    # window t: under the invariant bar this must qualify for gold.
    ws = WindowStats(w.window_start_bar, w.window_end_bar, 0.999,
                     w.t_net_full, 0.001)
    assert meets_tier_bar(ws, "gold") is True
    # equivalence identity: t_win on 252 bars == t_full = t_win*sqrt(4032/252)
    assert abs(w.t_net_full) * math.sqrt(T_FULL / WINDOW_BARS) >= GOLD_T_NET


def test_invariant_bar_boundary_exact():
    # at exactly the invariant gold hurdle on a 252-bar window -> qualifies
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.999, GOLD_T_WIN, 0.001),
                          "gold") is True
    # just below -> fails (bar not weakened in invariant terms)
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.999, GOLD_T_WIN - 1e-9,
                                      0.001), "gold") is False
    # silver boundaries
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.95, SILVER_T_WIN, 0.5),
                          "silver") is True
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.95, SILVER_T_WIN - 1e-9,
                                      0.5), "silver") is False
    # DSR / p_perm rules unchanged
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.989, GOLD_T_WIN, 0.001),
                          "gold") is False
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.999, GOLD_T_WIN, 0.011),
                          "gold") is False
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.949, SILVER_T_WIN, 0.5),
                          "silver") is False


def test_bar_scales_with_window_length_not_fixed_raw_t():
    # the invariant is annSR: a full-depth window uses raw 4.0; a 252-bar
    # window uses 1.0; a 1008-bar window uses 2.0.
    assert window_t_hurdle(GOLD_T_NET, T_FULL) == GOLD_T_NET
    assert abs(window_t_hurdle(GOLD_T_NET, WINDOW_BARS) - GOLD_T_WIN) < 1e-12
    assert abs(window_t_hurdle(GOLD_T_NET, 1008) - 2.0) < 1e-12
    assert meets_tier_bar(WindowStats(0, T_FULL, 0.999, 4.0, 0.001), "gold") is True
    assert meets_tier_bar(WindowStats(0, T_FULL, 0.999, 3.99, 0.001), "gold") is False
    assert meets_tier_bar(WindowStats(0, 1008, 0.999, 2.0, 0.001), "gold") is True
    assert meets_tier_bar(WindowStats(0, 1008, 0.999, 1.99, 0.001), "gold") is False
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.999, 1.0, 0.001), "gold") is True
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.999, 0.99, 0.001), "gold") is False


# -- fixture 2: null window still fails; missing stats never qualify ------------

def test_null_window_still_fails():
    rng = np.random.Generator(np.random.PCG64(7))
    null = rng.normal(0.0, 0.01, WINDOW_BARS)
    w = _win(null, 0, WINDOW_BARS)
    assert meets_tier_bar(w, "gold") is False
    assert meets_tier_bar(w, "silver") is False


def test_missing_stats_never_qualify():
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, None, 9.0, 0.001), "gold") is False
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.999, None, 0.001), "gold") is False
    assert meets_tier_bar(WindowStats(0, WINDOW_BARS, 0.999, 9.0, None), "gold") is False


# -- fixture 3: end-to-end through live_window_evaluator ------------------------

def test_live_window_evaluator_t_rule_no_longer_blocks_mint_gold_edge():
    r = _edge_meeting_mint_gold_bar()
    # two consecutive annual windows carved from the same edge
    live = np.concatenate([r[T_FULL - 2 * WINDOW_BARS:T_FULL - WINDOW_BARS],
                           r[T_FULL - WINDOW_BARS:T_FULL]])
    tmp = tempfile.mkdtemp()
    cards = CardRegistry(os.path.join(tmp, "c.db"), secret=b"test-secret")
    ledger = Ledger(os.path.join(tmp, "l.db"))
    try:
        t = ledger.register("pairs", "free_5y_top1k", {"cfg": "x"}, "v1",
                            discoverer_handle="alice")
        ledger.record_stage(t.trial_id, "s2", "pass", p_value=0.04)
        ledger.record_stage(t.trial_id, "s3", "pass", dsr=0.96)
        ledger.record_stage(t.trial_id, "s5", "pass", statistic=3.2)
        card = cards.mint(
            trial_id=t.trial_id, family=t.family, scope_id=t.scope_id,
            config_hash=t.config_hash, data_vintage_hash=t.data_vintage_hash,
            survivor_class="validated-green", owner_handle="alice",
            dsr=0.96, t_net_full=3.2, p_perm=0.04, now=1_000_000.0, minted_bar=0,
        )
        assert card.tier == "silver"
        ev = live_window_evaluator(
            card, live_returns=live, now_bar=len(live),
            n_stratum=50, v_stratum=0.01, seed=424242, s2_B=200,
        )
        # both windows: raw t well below 4.0 but annSR above the invariant
        # gold hurdle -> the t-component of the bar must pass on both.
        for w in (ev.trailing, ev.preceding):
            assert w is not None and w.t_net_full is not None
            assert abs(w.t_net_full) < GOLD_T_NET          # raw t below old bar
            sr = gstats.sharpe_ratio(live[w.window_start_bar:w.window_end_bar])
            assert abs(sr) >= GOLD_SR_ANN                  # annSR above invariant bar
            ws = WindowStats(w.window_start_bar, w.window_end_bar,
                             0.999, w.t_net_full, 0.001)
            assert meets_tier_bar(ws, "gold") is True      # t-rule passes
    finally:
        cards.close()
        ledger.close()
