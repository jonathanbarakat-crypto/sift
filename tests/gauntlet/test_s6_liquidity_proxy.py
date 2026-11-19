"""RED TEST first (soundness #8 remainder): synthetic panel where name A has
HIGH volatility but is LIQUID (continuous small moves, quotes every bar)
and name B has LOW volatility but is ILLIQUID (rare large jumps — few
prints, wide quoted spread). The S6 half-spread proxy must rank
cost(A) < cost(B); the current dispersion proxy ranks cost(A) > cost(B)
(it measures volatility, not liquidity).

Ground truth per the chosen proxy's definition: liquidity = quote/trade
continuity, measured from prices alone by the concentration of absolute
log-returns (participation ratio n_eff/T of |r_t|): liquid names move a
little every bar (n_eff/T ~ 0.6), illiquid names move in rare jumps
(n_eff/T ~ 0.03-0.08).
"""
import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.decks import DeckRegistry
from sift.decks.service import DeckService
from sift.gauntlet import config as gcfg
from sift.gauntlet import s6 as s6stats

SECRET = b"test-s6-liq-secret"
SCOPE = "free_5y_top1k"
VINTAGE = "v-test-1"
T = 600


def _liquid_series(rng, sigma, T=T):
    """Liquid name: every bar re-quotes, small continuous moves (GBM)."""
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, T)))


def _illiquid_series(rng, sigma, trade_prob, jump_mult=8.0, T=T):
    """Illiquid name: only trade_prob of bars print (the rest are stale
    quotes); prints are large jumps -> wide quoted half-spread."""
    rets = rng.normal(0.0, sigma * 0.1, T)
    trade = rng.random(T) < trade_prob
    rets[trade] = rng.normal(0.0, sigma * jump_mult, T)[trade]
    return 100.0 * np.exp(np.cumsum(rets))


def _mint_cards(cards, n=5, owner="alice"):
    out = []
    tiers = [(0.995, 4.5, 0.001), (0.96, 3.2, 0.01), (0.90, 2.5, 0.04)]
    for i in range(n):
        t = tiers[i % len(tiers)]
        card = cards.mint(
            trial_id=f"trial-x{i}", family="pairs", scope_id=SCOPE,
            config_hash=f"cfg{i}", data_vintage_hash=VINTAGE,
            survivor_class="validated-green", owner_handle=owner,
            dsr=t[0], t_net_full=t[1], p_perm=t[2], now=1000.0 + i,
        )
        out.append(card)
    return out


def test_liquidity_proxy_ranks_highvol_liquid_cheaper_than_lowvol_illiquid():
    rng = np.random.Generator(np.random.PCG64(42))
    A = _liquid_series(rng, 0.030)                    # high vol, liquid
    B = _illiquid_series(rng, 0.003, trade_prob=0.08)  # low vol, illiquid
    panel = np.column_stack([A, B])
    est = s6stats.estimate_half_spread_bps(panel)
    assert est["source"] == "data_estimate"
    cost_A, cost_B = est["per_name_bps"][0], est["per_name_bps"][1]
    # the required ranking: high-vol liquid cheaper than low-vol illiquid
    assert cost_A < cost_B, (
        f"proxy measures volatility not liquidity: cost_A={cost_A:.2f} "
        f">= cost_B={cost_B:.2f}")
    # and the liquidity (concentration) component alone must carry the signal
    liq_A = s6stats._liquidity_concentration_bps(A)
    liq_B = s6stats._liquidity_concentration_bps(B)
    assert liq_B > liq_A


def test_liquidity_concentration_invariant_to_vol_at_fixed_liquidity():
    """Pre-registered property: holding liquidity (trade frequency) fixed,
    the liquidity proxy must NOT grow with volatility."""
    rng = np.random.Generator(np.random.PCG64(7))
    liquid = [_illiquid_series(rng, s, trade_prob=1.0) for s in (0.003, 0.03)]
    illiquid = [_illiquid_series(rng, s, trade_prob=0.08) for s in (0.003, 0.03)]
    l_liq_lo, l_liq_hi = (s6stats._liquidity_concentration_bps(p) for p in liquid)
    l_ill_lo, l_ill_hi = (s6stats._liquidity_concentration_bps(p) for p in illiquid)
    assert l_liq_hi <= l_liq_lo * 1.25   # ~flat in vol, not monotone up
    assert l_ill_hi <= l_ill_lo * 1.25
    # and illiquid >> liquid at the SAME vol
    assert l_ill_lo > l_liq_lo * 1.5
    assert l_ill_hi > l_liq_hi * 1.5


def test_effective_half_spread_is_max_of_both_proxies_and_floor():
    """Charge max(liquidity proxy, dispersion proxy, floor) — never understated."""
    rng = np.random.Generator(np.random.PCG64(42))
    A = _liquid_series(rng, 0.030)
    B = _illiquid_series(rng, 0.003, trade_prob=0.08)
    est = s6stats.estimate_half_spread_bps(np.column_stack([A, B]))
    for j, p in enumerate((A, B)):
        disp = s6stats._dispersion_half_spread_bps(p)
        liq = s6stats._liquidity_concentration_bps(p)
        assert est["per_name_bps"][j] == pytest.approx(
            max(disp, liq, gcfg.S6_HALF_SPREAD_BPS_FLOOR), rel=1e-12)
    # the estimate records which proxy produced each name's cost
    assert est.get("proxy") == "max(liquidity_concentration, dispersion)"


def test_stage_detail_records_proxy_source(tmp_path):
    """The S6 stage detail must record which proxy produced the cost."""
    flags.reset()
    cards = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    decks = DeckRegistry(tmp_path / "decks.db", secret=SECRET)
    svc = DeckService(cards, decks)
    cs = _mint_cards(cards, n=5)
    d = svc.register_custom(scope_id=SCOPE, member_card_ids=[c.card_id for c in cs],
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    rng = np.random.Generator(np.random.PCG64(2))
    R = rng.normal(0.0005, 0.01, (300, 5))
    A = _liquid_series(rng, 0.030, T=300)
    B = _illiquid_series(rng, 0.003, trade_prob=0.08, T=300)
    prices = np.column_stack([A, B, A, B, A])
    res = svc.run_s6(d.deck_id, component_returns=R,
                     component_names=[{f"p{i}"} for i in range(5)],
                     prices=prices, owner_handle="alice", now=2.0)
    assert res["s6"]["cost_source"] == "data_estimate"
    assert res["s6"].get("half_spread_proxy") == "max(liquidity_concentration, dispersion)"
    cards.close()
    decks.close()
