"""Soundness #5 (hermes-tasks #181 finding #5 P1): the live-window upgrade
bar must be WINDOW-LENGTH INVARIANT — identical to the mint-time bar in
annualized-Sharpe terms, not in raw t.

Defect on main: meets_tier_bar compared the RAW t of a 252-bar live window
against the FULL-DEPTH (4032-bar) mint bars (platinum 4.0 / silver 3.0).
t scales with sqrt(T): an edge with t_full = 4.0 at mint (annSR = 1.0,
platinum at mint) has t_win ~ 1.0-1.3 on its trailing 252-bar window, so it
FAILED the raw 4.0 bar — the upgrade bar was ~4x stricter than the mint bar
it claimed to be identical to; genuine edges essentially never upgraded.

Fix under test: window_t_hurdle() rescales the full-depth bar by
sqrt(T_win / MINT_T_FULL), i.e.

    |t_win| >= B * sqrt(T_win / MINT_T_FULL)
        <=>  |sr_ann| >= B * sqrt(BARS_PER_YEAR / MINT_T_FULL)

which is window-length invariant (annualized Sharpe). At 252 bars the
identical mint bars are |t_win| >= 1.0 (platinum 4.0) and >= 0.75 (silver
3.0); at T_win = MINT_T_FULL the raw bar is used verbatim — no weakening at
full depth. DSR stays out of the bar (#170 N-independence) and the p_perm
rules are unchanged.

Fixtures (TDD, written against the defect first):
- a realistic stationary edge meeting the mint platinum bar at full depth
  (t_full == 4.0, annSR == 1.0) QUALIFIES on a 252-bar window of the same
  series (probe pre-fix: t_win = 1.27 -> meets_tier_bar False);
- a null (pure-noise) window still fails every bar;
- boundary exactness and window-length scaling of the hurdle;
- full-depth windows use the raw bars verbatim (no weakening);
- equivalence of the t form and the annualized-Sharpe form;
- missing stats never qualify; unknown tier never qualifies;
- gold bar unchanged (p_perm only, #170);
- end-to-end through live_window_evaluator.
"""

import math

import numpy as np
import pytest

from sift.cards import CARD_TIERS, tier_from_stats
from sift.cards.decay import (
    GOLD_P_PERM,
    MINT_T_FULL,
    PLATINUM_P_PERM,
    PLATINUM_T_NET,
    SILVER_P_PERM,
    SILVER_T_NET,
    WINDOW_BARS,
    WindowStats,
    live_window_evaluator,
    meets_tier_bar,
    window_t_hurdle,
)
from sift.gauntlet import config as gcfg
from sift.gauntlet import stats as gstats

BPY = gcfg.BARS_PER_YEAR
assert MINT_T_FULL == gcfg.SCOPE_T[gcfg.S5_SCOPE] == 4032
assert WINDOW_BARS == BPY == 252
assert CARD_TIERS == ("bronze", "silver", "gold", "platinum")


def _edge(seed: int, n: int, t_target: float) -> np.ndarray:
    """A realistic stationary mean-reverting edge series whose FULL-SAMPLE
    t (t_stat_sharpe over all n bars) is calibrated to ~t_target, i.e. its
    annualized Sharpe is t_target / sqrt(n / BPY). At n = MINT_T_FULL this
    is exactly a card meeting the mint bar of t_target."""
    rng = np.random.Generator(np.random.PCG64(seed))
    phi = -0.35                      # mean reversion (the conservative direction)
    e = rng.normal(0.0, 1.0, n)
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + e[k]
    x = x - x.mean()
    sd = x.std(ddof=1)
    m = t_target * sd / math.sqrt(n)     # t = mean / (sd / sqrt(n))
    return m + x


def _strong_live_returns(n=2 * WINDOW_BARS + 40, seed=5):
    """Same construction as tests/cards/test_bidirectional_tiers.py: a
    strongly mean-reverting live series whose 252-bar windows have a high
    annualized Sharpe and a tiny permutation p."""
    rng = np.random.Generator(np.random.PCG64(seed))
    e = rng.normal(0, 0.01, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] + e[t]
    r = 0.002 - 0.8 * np.roll(s, 1) + e
    r[0] = 0.002 + e[0]
    return r


@pytest.fixture()
def ledger_cards(tmp_path):
    from sift.cards import CardRegistry
    from sift.registry import Ledger
    from tests.cards.test_minting import SECRET
    l = Ledger(tmp_path / "ledger.db")
    c = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    yield c, l
    c.close()
    l.close()


# ---------------------------------------------------------------------------
# 1. The headline fixture: mint-bar edge qualifies on the 252-bar window
# ---------------------------------------------------------------------------

def test_mint_platinum_bar_edge_qualifies_on_252_bar_window():
    """An edge meeting the mint PLATINUM bar at full depth (t_full == 4.0,
    annSR == 1.0 => tier_from_stats = platinum at mint) must QUALIFY on a
    252-bar window of the SAME series under the invariant bar.
    Pre-fix probe: t_win = 1.27 < raw bar 4.0 -> meets_tier_bar = False
    (the defect). Post-fix: |t_win| >= 4.0*sqrt(252/4032) = 1.0 -> True."""
    r_full = _edge(seed=11, n=MINT_T_FULL, t_target=4.0)
    t_full = gstats.t_stat_sharpe(r_full)
    assert abs(t_full - 4.0) < 1e-9                      # calibration exact
    assert abs(gstats.sharpe_ratio(r_full) - 1.0) < 1e-9  # annSR 1.0 at mint
    # the card IS platinum at mint
    assert tier_from_stats(dsr=0.999, t_net_full=t_full, p_perm=0.001,
                           s4_pass=True, s5_pass=True) == "platinum"
    # its trailing 252-bar live window (fully out-of-sample tail of the series)
    r_win = r_full[-WINDOW_BARS:]
    t_win = gstats.t_stat_sharpe(r_win)
    ws = WindowStats(MINT_T_FULL - WINDOW_BARS, MINT_T_FULL, 0.999, t_win, 0.001)
    # the defect: raw comparison against the full-depth bar fails
    assert t_win < PLATINUM_T_NET                          # pre-fix would fail
    assert meets_tier_bar(ws, "platinum") is True           # invariant bar: >= 1.0
    # silver bar (3.0 full depth -> 0.75 at 252) also met
    assert meets_tier_bar(ws, "silver") is True


def test_mint_silver_bar_edge_qualifies_on_252_bar_window():
    """Same for the silver mint bar (t_full == 3.0 <=> annSR 0.75): the
    252-bar window hurdle is 0.75, so the same-series window qualifies."""
    r_full = _edge(seed=23, n=MINT_T_FULL, t_target=3.0)
    t_full = gstats.t_stat_sharpe(r_full)
    assert abs(t_full - 3.0) < 1e-9
    assert tier_from_stats(dsr=0.999, t_net_full=t_full, p_perm=0.04) == "silver"
    r_win = r_full[-WINDOW_BARS:]
    t_win = gstats.t_stat_sharpe(r_win)
    ws = WindowStats(0, WINDOW_BARS, 0.999, t_win, 0.04)
    assert meets_tier_bar(ws, "silver") is True            # t_win ~ 0.75+
    assert meets_tier_bar(ws, "platinum") is False         # below 1.0 hurdle


# ---------------------------------------------------------------------------
# 2. Null window still fails; no weakening at the boundary
# ---------------------------------------------------------------------------

def test_null_window_still_fails_every_bar():
    rng = np.random.Generator(np.random.PCG64(7))
    r_null = rng.normal(0, 0.01, WINDOW_BARS)
    t_null = gstats.t_stat_sharpe(r_null)
    ws = WindowStats(0, WINDOW_BARS, 0.999, t_null, 0.5)   # p_perm 0.5 fails
    assert meets_tier_bar(ws, "silver") is False
    assert meets_tier_bar(ws, "gold") is False             # p_perm 0.5 > 0.01
    assert meets_tier_bar(ws, "platinum") is False
    # null t below every invariant hurdle (|t| < 0.75) with p passing
    ws2 = WindowStats(0, WINDOW_BARS, 0.999, 0.5, 0.001)
    assert meets_tier_bar(ws2, "silver") is False
    assert meets_tier_bar(ws2, "platinum") is False
    assert meets_tier_bar(ws2, "gold") is True              # gold = p_perm only
    # a genuine pure-noise window (drawn, not forced): t below the 0.75 hurdle
    r_null2 = rng.normal(0, 0.01, WINDOW_BARS)
    t_null2 = gstats.t_stat_sharpe(r_null2)
    if abs(t_null2) < 0.75:                                 # the null case
        ws3 = WindowStats(0, WINDOW_BARS, 0.999, t_null2, 0.001)
        assert meets_tier_bar(ws3, "silver") is False
        assert meets_tier_bar(ws3, "platinum") is False


def test_boundary_exactness_at_252_bars():
    # platinum hurdle at 252 = 4.0*sqrt(252/4032) = 1.0 exactly
    assert window_t_hurdle(PLATINUM_T_NET, WINDOW_BARS) == pytest.approx(1.0)
    assert window_t_hurdle(SILVER_T_NET, WINDOW_BARS) == pytest.approx(0.75)
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 1.0, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.0, 0.999, 0.01), "platinum")
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 0.75, 0.05), "silver")
    assert not meets_tier_bar(WindowStats(0, 252, 0.0, 0.749, 0.05), "silver")


def test_no_weakening_in_invariant_terms():
    """annSR 0.99 still fails the platinum bar (mint bar is annSR 1.0);
    a full-depth window uses the raw bar VERBATIM (4.0 / 3.0)."""
    # annSR 0.99 on a 252-bar window <=> t_win = 0.99
    assert not meets_tier_bar(WindowStats(0, 252, 0.0, 0.99, 0.01), "platinum")
    # full-depth window: raw bars verbatim — 3.99 fails platinum, 4.0 passes
    assert not meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.0, 3.99, 0.01), "platinum")
    assert meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.0, 4.0, 0.01), "platinum")
    assert not meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.0, 2.99, 0.05), "silver")
    assert meets_tier_bar(WindowStats(0, MINT_T_FULL, 0.0, 3.0, 0.05), "silver")
    # window_t_hurdle at full depth is the raw bar
    assert window_t_hurdle(PLATINUM_T_NET, MINT_T_FULL) == pytest.approx(4.0)
    assert window_t_hurdle(SILVER_T_NET, MINT_T_FULL) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. Window-length scaling + equivalence with the annualized-Sharpe form
# ---------------------------------------------------------------------------

def test_window_t_hurdle_scales_with_sqrt_t():
    assert window_t_hurdle(4.0, 252) == pytest.approx(1.0)
    assert window_t_hurdle(4.0, 1008) == pytest.approx(2.0)
    assert window_t_hurdle(4.0, 4032) == pytest.approx(4.0)
    assert window_t_hurdle(3.0, 252) == pytest.approx(0.75)
    # monotone in window length
    assert window_t_hurdle(4.0, 100) < window_t_hurdle(4.0, 400) < window_t_hurdle(4.0, 4032)


def test_t_form_equivalent_to_annualized_sharpe_form():
    """|t_win| >= B*sqrt(T_win/MINT_T_FULL)  <=>  |sr_ann| >= B*sqrt(BPY/MINT_T_FULL):
    the comparison is window-length invariant."""
    for B, tier, p in ((4.0, "platinum", 0.01), (3.0, "silver", 0.05)):
        ann_bar = B * math.sqrt(BPY / MINT_T_FULL)      # 1.0 / 0.75
        for T_win in (63, 252, 1008, 4032):
            r = _edge(seed=99, n=T_win, t_target=B * math.sqrt(T_win / BPY) * 1.05)
            t = gstats.t_stat_sharpe(r)
            sr_ann = gstats.sharpe_ratio(r)
            ws = WindowStats(0, T_win, 0.0, t, p)
            assert meets_tier_bar(ws, tier) == (abs(sr_ann) >= ann_bar - 1e-12)


# ---------------------------------------------------------------------------
# 4. Structural rules unchanged: p_perm, missing stats, unknown tier, gold
# ---------------------------------------------------------------------------

def test_gold_bar_unchanged_p_perm_only():
    """#170 N-independence kept: gold's live-window bar is still the raw
    p_perm <= 0.01 line (no t, no DSR)."""
    assert meets_tier_bar(WindowStats(0, 252, 0.0, 9.0, GOLD_P_PERM), "gold")
    assert meets_tier_bar(WindowStats(0, 252, 0.999, 0.0, GOLD_P_PERM), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, 0.011), "gold")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, None), "gold")


def test_missing_stats_never_qualify():
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, None, 0.001), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, None, 0.001), "silver")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, None), "silver")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, None), "platinum")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, None, None), "gold")


def test_unknown_tier_never_qualifies():
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, 0.001), "bronze")
    assert not meets_tier_bar(WindowStats(0, 252, 0.999, 9.0, 0.001), "diamond")


def test_dsr_still_not_part_of_the_bar():
    """#170 kept: the dsr field is health context only — it never changes
    the meets_tier_bar verdict."""
    for dsr in (None, 0.0, 0.5, 0.999):
        assert meets_tier_bar(WindowStats(0, 252, dsr, 1.0, 0.01), "platinum")
        assert not meets_tier_bar(WindowStats(0, 252, dsr, 0.99, 0.01), "platinum")


# ---------------------------------------------------------------------------
# 5. End-to-end through live_window_evaluator (252-bar windows)
# ---------------------------------------------------------------------------

def test_end_to_end_strong_live_series_qualifies(ledger_cards):
    cards, ledger = ledger_cards
    from tests.cards.test_bidirectional_tiers import _mint_silver
    card = _mint_silver(cards, ledger, minted_bar=0)
    rets = _strong_live_returns()
    ev = live_window_evaluator(
        card, live_returns=rets, now_bar=len(rets),
        n_stratum=50, v_stratum=0.01, seed=424242, s2_B=200,
    )
    assert ev.trailing is not None and ev.preceding is not None
    # both 252-bar windows meet the invariant t hurdles (the strong series
    # has annSR >> 1.0) — modulo the p_perm draw per tier
    assert ev.trailing.t_net_full is not None
    t_win = ev.trailing.t_net_full
    assert abs(t_win) >= window_t_hurdle(PLATINUM_T_NET, WINDOW_BARS)
    assert meets_tier_bar(ev.trailing, "platinum") == (
        ev.trailing.p_perm <= PLATINUM_P_PERM)
    assert meets_tier_bar(ev.preceding, "silver") == (
        abs(ev.preceding.t_net_full) >= window_t_hurdle(SILVER_T_NET, WINDOW_BARS)
        and ev.preceding.p_perm <= SILVER_P_PERM)
