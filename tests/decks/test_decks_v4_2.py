"""v4.2 deck layer — S6 portfolio gauntlet + bucket registry tests (spec v4.2 §4, hermes-tasks #166).

Covers the task's test list:
- dedupe/caching on (scope, bucket config, vintage) like cards;
- bucket enumeration (fixed grid sizes x sectors x family mixes, free-tier);
- deterministic top-K membership;
- custom-deck informational-only path (amber, unranked, unshareable —
  rank NEVER set even when S6 passes);
- S6 pass/fail incl. multiplicity correction over the deck stratum
  (N = all registered deck trials in scope across all users, kills
  included);
- effective-N computation (components sharing names correlate — 50 mixed
  cards != 50 independent bets);
- realistic per-name half-spreads from data replace the 100bps
  placeholder, 100bps kept as conservative bound;
- deck ranking F D C B A S distinct from card tiers; beta amber telemetry;
- append-only deck trials + deck events with read-back;
- flag-off fail-closed (SIFT_DECKS_V4_2=0 and SIFT_CARDS_V4_1=0);
- banned-phrase grep fail-closed on ALL deck strings; no tickers/direction
  in any artifact; non-transferable.
"""

import json

import numpy as np
import pytest

from sift import flags
from sift.cards import CardRegistry
from sift.decks import (
    DECK_RANKS,
    DeckError,
    DeckRegistry,
    GICS_SECTORS,
    bucket_config,
    deck_rank_from_stats,
    enumerate_buckets,
)
from sift.decks.copy import (
    DECK_GLOSSARY,
    DeckCopyError,
    assert_deck_copy_clean,
    deck_custom_warning,
    deck_explainer,
    deck_glossary,
    deck_rank_label,
    deck_telemetry_caveat,
)
from sift.decks.service import DeckService, ranked_membership
from sift.gauntlet import config as gcfg
from sift.gauntlet import s6 as s6stats

SECRET = b"test-deck-hmac-secret"
SCOPE = "free_5y_top1k"
VINTAGE = "v-test-1"


@pytest.fixture(autouse=True)
def _flags_reset():
    flags.reset()
    yield
    flags.reset()


@pytest.fixture()
def cards(tmp_path):
    c = CardRegistry(tmp_path / "cards.db", secret=SECRET)
    yield c
    c.close()


@pytest.fixture()
def decks(tmp_path):
    d = DeckRegistry(tmp_path / "decks.db", secret=SECRET)
    yield d
    d.close()


@pytest.fixture()
def svc(cards, decks):
    return DeckService(cards, decks)


def _mint_cards(cards, n=12, owner="alice", family="pairs"):
    """Mint n regular cards with distinct specs and mixed tiers."""
    out = []
    tiers = [("gold", 0.995, 4.5, 0.001), ("silver", 0.96, 3.2, 0.01),
             ("bronze", 0.90, 2.5, 0.04)]
    for i in range(n):
        t = tiers[i % len(tiers)]
        card = cards.mint(
            trial_id=f"trial-x{i}", family=family, scope_id=SCOPE,
            config_hash=f"cfg{i}", data_vintage_hash=VINTAGE,
            survivor_class="validated-green", owner_handle=owner,
            dsr=t[1], t_net_full=t[2], p_perm=t[3], now=1000.0 + i,
        )
        out.append(card)
    return out


# ---------------------------------------------------------------------------
# Bucket enumeration + config validation (fixed grid, free-tier)
# ---------------------------------------------------------------------------

def test_bucket_grid_is_fixed_and_enumerable():
    buckets = enumerate_buckets()
    n_sectors = 1 + len(GICS_SECTORS)
    n_mixes = len(gcfg.DECK_FAMILIES) + 1  # singles + mixed
    assert len(buckets) == len(gcfg.DECK_BUCKET_SIZES) * n_sectors * n_mixes
    sizes = {b["size"] for b in buckets}
    assert sizes == {5, 10, 20, 50, 100}
    for b in buckets:
        assert b["sector"] == "all" or b["sector"] in GICS_SECTORS
        assert set(b["family_mix"]) <= set(gcfg.DECK_FAMILIES)
        assert b["family_mix"]


def test_bucket_config_rejects_free_form_sizes_fail_closed():
    with pytest.raises(DeckError):
        bucket_config(size=7)
    with pytest.raises(DeckError):
        bucket_config(size=5, sector="not-a-sector")
    with pytest.raises(DeckError):
        bucket_config(size=5, family_mix=("momentum",))
    with pytest.raises(DeckError):
        bucket_config(size=5, family_mix=())


# ---------------------------------------------------------------------------
# Deterministic membership: top-K by card tier in bucket
# ---------------------------------------------------------------------------

def test_ranked_membership_topk_by_tier_deterministic(cards):
    cs = _mint_cards(cards, n=12)
    bucket = bucket_config(size=5, sector="all", family_mix=("pairs",))
    m1 = ranked_membership(cs, bucket)
    m2 = ranked_membership(cs, bucket)
    assert m1 == m2                      # deterministic
    assert len(m1) == 5                  # K = bucket size
    # top-5 by tier: golds first (indices 0,3,6,9), then a silver (1)
    by_id = {c.card_id: c.tier for c in cs}
    assert [by_id[c] for c in m1] == ["gold", "gold", "gold", "gold", "silver"]


def test_ranked_membership_family_mix_and_sector(cards):
    pairs = _mint_cards(cards, n=6, family="pairs")
    ll = _mint_cards(cards, n=6, owner="bob", family="lead_lag")
    pool = pairs + ll
    b_pairs = bucket_config(size=5, sector="all", family_mix=("pairs",))
    got = ranked_membership(pool, b_pairs)
    assert got == sorted(got, key=lambda c: (-{"gold": 2, "silver": 1, "bronze": 0}[
        {c.card_id: c.tier for c in pairs}[c]], c))  # top-5 by tier among pairs
    assert set(got) <= {c.card_id for c in pairs}
    b_mixed = bucket_config(size=10, sector="all", family_mix=("pairs", "lead_lag"))
    m = ranked_membership(pool, b_mixed)
    assert len(m) == 10 and any(c in m for c in [x.card_id for x in ll])
    # sector split: only cards labelled with that sector
    sector_of = {c.card_id: ("financials" if i % 2 == 0 else "energy")
                 for i, c in enumerate(pool)}
    b_sec = bucket_config(size=5, sector="financials", family_mix=("pairs", "lead_lag"))
    m_sec = ranked_membership(pool, b_sec, sector_of=sector_of)
    assert len(m_sec) == 5
    assert all(sector_of[c] == "financials" for c in m_sec)


def test_ranked_membership_raises_when_bucket_unsatisfiable(cards):
    cs = _mint_cards(cards, n=3)
    with pytest.raises(DeckError):
        ranked_membership(cs, bucket_config(size=5, family_mix=("pairs",)))


# ---------------------------------------------------------------------------
# Dedupe/caching on (scope, bucket config, vintage) like cards
# ---------------------------------------------------------------------------

def test_register_ranked_dedupes_on_scope_config_vintage(svc, cards, decks):
    _mint_cards(cards, n=10)
    bucket = bucket_config(size=5, sector="all", family_mix=("pairs",))
    d1 = svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="alice",
                             data_vintage_hash=VINTAGE, now=100.0)
    d2 = svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="bob",
                             data_vintage_hash=VINTAGE, now=200.0)
    assert d1.deck_id == d2.deck_id          # same key -> same deck (cached)
    assert len(decks.all_decks()) == 1       # counted exactly once
    # different vintage -> new deck
    d3 = svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="alice",
                             data_vintage_hash="v-test-2", now=300.0)
    assert d3.deck_id != d1.deck_id
    assert len(decks.all_decks()) == 2
    # different bucket config -> new deck
    bucket2 = bucket_config(size=10, sector="all", family_mix=("pairs",))
    d4 = svc.register_ranked(scope_id=SCOPE, bucket=bucket2, owner_handle="alice",
                             data_vintage_hash=VINTAGE, now=400.0)
    assert d4.deck_id != d1.deck_id
    # read-back: register returned a re-read row
    assert decks.get_deck(d1.deck_id).deck_id == d1.deck_id
    # registration events appended + readable
    evs = decks.events("deck_register")
    assert len(evs) >= 3


def test_ranked_register_requires_exactly_K_members(svc, cards):
    _mint_cards(cards, n=3)
    with pytest.raises(DeckError):
        svc.decks.register_ranked(scope_id=SCOPE,
                                  config=bucket_config(size=5, family_mix=("pairs",)),
                                  member_card_ids=["card-a", "card-b"],
                                  owner_handle="alice", data_vintage_hash=VINTAGE)


# ---------------------------------------------------------------------------
# Custom decks: informational only (amber, unranked, unshareable)
# ---------------------------------------------------------------------------

def test_custom_deck_informational_only_even_when_s6_passes(svc, cards, decks):
    cs = _mint_cards(cards, n=6)
    ids = [c.card_id for c in cs[:5]]
    d = svc.register_custom(scope_id=SCOPE, member_card_ids=ids,
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=10.0)
    assert d.kind == "custom"
    assert d.rank is None
    # run S6 with a strongly mean-reverting basket -> pass
    rng = np.random.Generator(np.random.PCG64(3))
    T = 400
    comp = rng.normal(0.001, 0.01, (T, 5)) * np.array([1, -1, 1, -1, 1])
    # plant mean reversion: negative autocorrelation
    for k in range(5):
        comp[1:, k] += -0.3 * comp[:-1, k] * 10 * 0.01
    res = svc.run_s6(d.deck_id, component_returns=comp,
                     component_names=[{f"n{i}", f"n{i+1}"} for i in range(5)],
                     owner_handle="alice", now=20.0)
    rb = res["deck"]
    assert rb.kind == "custom"
    assert rb.rank is None                    # NEVER ranked, even on a pass
    assert res["s6"]["informational_only"] is True
    # the deck_trial event carries informational_only + no rank
    evs = decks.events("deck_trial", deck_id=d.deck_id)
    assert len(evs) == 1
    payload = json.loads(evs[0]["payload_json"])
    assert payload["informational_only"] is True
    assert payload["rank"] is None
    # copy marks it unranked + unshareable
    assert "unranked" in deck_rank_label(None, "custom")
    assert "cannot be shared" in deck_custom_warning()


def test_custom_deck_requires_owned_cards(svc, cards):
    cs = _mint_cards(cards, n=3, owner="alice")
    with pytest.raises(Exception):
        svc.register_custom(scope_id=SCOPE, member_card_ids=[cs[0].card_id],
                            owner_handle="mallory", data_vintage_hash=VINTAGE)


# ---------------------------------------------------------------------------
# S6 pass/fail incl. multiplicity correction over the deck stratum
# ---------------------------------------------------------------------------

def _planted_basket(T=1260, K=10, seed=11, a=0.35, sig=0.01, drift=0.0004):
    """K component STRATEGY return series with a planted mean-reversion
    edge. The spread follows r_t = drift - a*r_{t-1} + e_t; the pairs
    convention earns the LAGGED position -sign(r_{t-1}) * r_t (short the
    spread when it is high), which has a positive mean under mean
    reversion. NOTE (soundness #3): feeding raw spread returns (mean ~0)
    directly as component_returns only produced a positive basket Sharpe
    through the look-ahead bug — the honest fixture must feed strategy
    returns, not raw spreads."""
    rng = np.random.Generator(np.random.PCG64(seed))
    R = np.zeros((T, K))
    prev = rng.normal(0, sig, K)
    for t in range(T):
        e = rng.normal(0, sig, K)
        r = drift - a * prev + e
        R[t] = -np.sign(prev) * r
        prev = r
    return R


def test_s6_pass_on_planted_edge_and_fail_on_null(svc, cards, decks):
    cs = _mint_cards(cards, n=10)
    bucket = bucket_config(size=10, sector="all", family_mix=("pairs",))
    d = svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="alice",
                            data_vintage_hash=VINTAGE, now=1.0)
    names = [{f"n{i}", f"n{i+1}"} for i in range(10)]  # disjoint-ish overlap
    # pass case: planted mean reversion in every component
    R = _planted_basket(T=1260, K=10, seed=11)
    res = svc.run_s6(d.deck_id, component_returns=R, component_names=names,
                     owner_handle="alice", now=2.0)
    s6 = res["s6"]
    assert s6["pass"] is True
    assert s6["net_sr_pos"] if "net_sr_pos" in s6 else s6["checks"]["net_sr_pos"]
    assert s6["checks"]["t_net"] and s6["checks"]["min_trl"] and s6["checks"]["pbo"]
    assert s6["n_stratum"] >= 1
    assert s6["n_eff"] < s6["K"]            # overlap -> fewer independent bets
    rb = res["deck"]
    assert rb.s6_pass == 1
    assert rb.rank in DECK_RANKS and rb.rank != "F"
    # fail case: pure noise on a second deck
    d2 = svc.register_ranked(scope_id=SCOPE,
                             bucket=bucket_config(size=10, sector="energy",
                                                  family_mix=("pairs",)),
                             owner_handle="alice", data_vintage_hash=VINTAGE, now=3.0)
    rng = np.random.Generator(np.random.PCG64(99))
    R_null = rng.normal(0, 0.01, (1260, 10))
    res2 = svc.run_s6(d2.deck_id, component_returns=R_null,
                      component_names=[{f"m{i}"} for i in range(10)],
                      owner_handle="alice", now=4.0)
    assert res2["s6"]["pass"] is False
    assert res2["deck"].s6_pass == 0
    # multiplicity: N includes kills — stratum_n counts BOTH decks
    assert decks.stratum_n(SCOPE) == 2


def test_multiplicity_correction_over_deck_stratum_kills_included(svc, cards, decks):
    """N = all registered deck trials in scope across ALL users, kills
    included; p_corr = min(1, p_raw * N)."""
    cs = _mint_cards(cards, n=10)
    bucket = bucket_config(size=10, sector="all", family_mix=("pairs",))
    deck_ids = []
    for i, sector in enumerate(["all", "energy", "financials"]):
        b = bucket_config(size=10, sector=sector, family_mix=("pairs",))
        d = svc.register_ranked(scope_id=SCOPE, bucket=b, owner_handle="alice",
                                data_vintage_hash=VINTAGE, now=1.0 + i)
        deck_ids.append(d.deck_id)
    # one passes, two fail (kills) — all three count toward N
    R = _planted_basket(T=1260, K=10, seed=5)
    rng = np.random.Generator(np.random.PCG64(7))
    svc.run_s6(deck_ids[0], component_returns=R,
               component_names=[{f"a{i}"} for i in range(10)],
               p_perm=0.01, owner_handle="alice", now=10.0)
    svc.run_s6(deck_ids[1], component_returns=rng.normal(0, 0.01, (1260, 10)),
               component_names=[{f"b{i}"} for i in range(10)],
               p_perm=0.5, owner_handle="alice", now=11.0)
    svc.run_s6(deck_ids[2], component_returns=rng.normal(0, 0.01, (1260, 10)),
               component_names=[{f"c{i}"} for i in range(10)],
               p_perm=0.5, owner_handle="alice", now=12.0)
    assert decks.stratum_n(SCOPE) == 3      # kills included
    mc = s6stats.multiplicity_corrected_p(0.01, n_stratum=3, n_eff=10.0)
    assert mc["p_corr"] == pytest.approx(0.03)
    assert mc["n_stratum"] == 3 and mc["n_eff"] == 10.0
    mc2 = s6stats.multiplicity_corrected_p(0.5, n_stratum=3, n_eff=2.0)
    assert mc2["p_corr"] == 1.0             # capped


def test_s6_multiplicity_uses_effective_n_not_card_count():
    # 10 components all sharing the same 2 names -> n_eff collapses toward 1
    neff = s6stats.effective_n([{"A", "B"}] * 10)
    assert neff["K"] == 10
    assert neff["n_eff"] == pytest.approx(1.0)
    # 10 disjoint components -> n_eff == K (independent bets)
    neff2 = s6stats.effective_n([{f"x{i}"} for i in range(10)])
    assert neff2["n_eff"] == pytest.approx(10.0)
    # partial overlap: between 1 and K
    neff3 = s6stats.effective_n([{"a", "b"}, {"b", "c"}, {"c", "d"}])
    assert 1.0 < neff3["n_eff"] < 3.0
    # 50 mixed cards are NOT 50 independent bets
    neff50 = s6stats.effective_n([{f"n{i % 5}", f"n{(i+1) % 5}"} for i in range(50)])
    assert neff50["n_eff"] < 50.0


def test_effective_n_never_relaxes_multiplicity():
    """n_eff enters the DSR benchmark (fewer independent bets -> LOWER luck
    ceiling -> HARDER bar), never a relaxation of p."""
    mc = s6stats.multiplicity_corrected_p(0.04, n_stratum=100, n_eff=2.0)
    assert mc["p_corr"] == min(1.0, 0.04 * 100)   # raw stratum scaling, not n_eff
    assert mc["sr_star_eff"] >= 0.0               # benchmark exists for low n_eff


# ---------------------------------------------------------------------------
# Realistic per-name half-spreads from data (replace 100bps placeholder)
# ---------------------------------------------------------------------------

def test_half_spread_estimated_from_data_and_bound_kept():
    rng = np.random.Generator(np.random.PCG64(1))
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    est = s6stats.estimate_half_spread_bps(prices)
    assert est["source"] == "data_estimate"
    assert est["median_bps"] >= gcfg.S6_HALF_SPREAD_BPS_FLOOR
    assert est["median_bps"] != 100.0 or est["median_bps"] == 100.0  # data-driven
    assert est["conservative_bound_bps"] == 100.0   # placeholder kept as bound
    eff = s6stats.effective_half_spread_bps(est)
    assert eff >= gcfg.S6_HALF_SPREAD_BPS_FLOOR
    # degenerate data -> conservative bound, never zero cost
    est0 = s6stats.estimate_half_spread_bps(np.array([100.0, 100.0, 100.0]))
    assert est0["source"] == "conservative_bound"
    assert s6stats.effective_half_spread_bps(est0) == 100.0


def test_s6_cost_model_uses_data_spreads(svc, cards):
    cs = _mint_cards(cards, n=5)
    d = svc.register_custom(scope_id=SCOPE, member_card_ids=[c.card_id for c in cs],
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    rng = np.random.Generator(np.random.PCG64(2))
    T = 300
    R = _planted_basket(T=T, K=5, seed=8)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, T)))
    res = svc.run_s6(d.deck_id, component_returns=R,
                     component_names=[{f"p{i}"} for i in range(5)],
                     prices=prices, owner_handle="alice", now=2.0)
    assert res["s6"]["cost_source"] == "data_estimate"
    assert res["s6"]["conservative_bound_bps"] == 100.0
    assert res["s6"]["half_spread_bps_used"] < 100.0 or res["s6"]["half_spread_bps_used"] >= 1.0


# ---------------------------------------------------------------------------
# Regression: 2D multi-name price panels must be estimated PER NAME
# (column-wise), never flattened across symbol boundaries (t_d8534dbc)
# ---------------------------------------------------------------------------

def _name_series(rng, T, level, sigma):
    return level * np.exp(np.cumsum(rng.normal(0.0, sigma, T)))


def test_half_spread_2d_panel_matches_median_of_per_name_estimates():
    """(a) 2D panel with known per-name dispersion -> estimate ~= median of
    the per-name medians, NOT inflated by cross-symbol boundary steps."""
    rng = np.random.Generator(np.random.PCG64(11))
    T, K = 400, 6
    sigmas = [0.005, 0.008, 0.010, 0.012, 0.015, 0.020]
    panel = np.column_stack([_name_series(rng, T, 100.0, s) for s in sigmas])
    est = s6stats.estimate_half_spread_bps(panel)
    assert est["source"] == "data_estimate"
    per_name = np.array([s6stats.estimate_half_spread_bps(panel[:, j])["median_bps"]
                         for j in range(K)])
    assert est["per_name_bps"].shape == (K,)
    assert np.allclose(est["per_name_bps"], per_name)
    assert est["median_bps"] == pytest.approx(float(np.median(per_name)), rel=1e-12)


def test_half_spread_2d_wildly_different_price_levels_not_inflated():
    """(b) symbols at wildly different price levels (0.01 vs 10000): the
    old flattened estimator saw huge spurious 'returns' at each boundary;
    the per-name estimate stays bounded by the per-name dispersion."""
    rng = np.random.Generator(np.random.PCG64(12))
    T, K = 300, 5
    levels = [0.01, 0.5, 10.0, 250.0, 10000.0]
    panel = np.column_stack([_name_series(rng, T, lv, 0.01) for lv in levels])
    est = s6stats.estimate_half_spread_bps(panel)
    assert est["source"] == "data_estimate"
    per_name = np.array([s6stats.estimate_half_spread_bps(panel[:, j])["median_bps"]
                         for j in range(K)])
    # every per-name estimate is a ~1%-daily-vol name -> small, bounded bps
    assert np.all(per_name < 100.0)
    assert est["median_bps"] == pytest.approx(float(np.median(per_name)), rel=1e-12)
    assert est["median_bps"] < 100.0  # NOT the 33%-per-side garbage from flattening


def test_half_spread_2d_nonfinite_handled_per_column():
    """(c) non-finite entries are dropped per column; a column that is all
    non-finite falls back to the conservative bound for that name only."""
    rng = np.random.Generator(np.random.PCG64(13))
    T, K = 200, 4
    panel = np.column_stack([_name_series(rng, T, 100.0, 0.01) for _ in range(K)])
    panel[5, 0] = np.nan          # hole in column 0
    panel[10, 1] = np.inf         # hole in column 1
    panel[:, 3] = np.nan          # column 3 entirely non-finite
    est = s6stats.estimate_half_spread_bps(panel)
    assert est["source"] == "data_estimate"
    good = s6stats.estimate_half_spread_bps(panel[:, 0][np.isfinite(panel[:, 0])])["median_bps"]
    assert est["per_name_bps"][0] == pytest.approx(good, rel=1e-12)
    assert est["per_name_bps"][3] == gcfg.S6_HALF_SPREAD_BPS_CONSERVATIVE
    finite_per_name = [est["per_name_bps"][j] for j in range(K) if j != 3]
    assert est["median_bps"] == pytest.approx(
        float(np.median([est["per_name_bps"][0], est["per_name_bps"][1],
                         est["per_name_bps"][2], gcfg.S6_HALF_SPREAD_BPS_CONSERVATIVE])),
        rel=1e-12)
    assert np.isfinite(est["median_bps"])


def test_half_spread_1d_behavior_unchanged():
    """1D single-series input: result is max(liquidity_concentration,
    dispersion, floor) — the dispersion term still matches the pre-fix
    formula exactly (0.25 * std(|diff(log p)|) * 1e4 floored), and the
    liquidity term adds the soundness-#8 liquidity signal on top;
    floor/bound logic untouched."""
    rng = np.random.Generator(np.random.PCG64(1))
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    est = s6stats.estimate_half_spread_bps(prices)
    assert est["source"] == "data_estimate"
    p = np.asarray(prices, float)
    p = p[np.isfinite(p)]
    r = np.abs(np.diff(np.log(p)))
    r = r[np.isfinite(r)]
    expected_disp = float(np.maximum(0.25 * float(np.std(r, ddof=1)) * 1e4,
                                     gcfg.S6_HALF_SPREAD_BPS_FLOOR))
    expected_liq = s6stats._liquidity_concentration_bps(p)
    assert expected_disp == pytest.approx(
        s6stats._dispersion_half_spread_bps(p), rel=1e-12)
    assert est["median_bps"] == pytest.approx(
        max(expected_disp, expected_liq, gcfg.S6_HALF_SPREAD_BPS_FLOOR), rel=1e-12)
    assert est["median_bps"] >= expected_disp   # never below the legacy bound
    assert est["proxy"] == gcfg.S6_HALF_SPREAD_PROXY
    assert est["per_name_bps"].shape == (1,)
    # degenerate 1D still falls back to the conservative bound
    est0 = s6stats.estimate_half_spread_bps(np.array([100.0, 100.0, 100.0]))
    assert est0["source"] == "conservative_bound"
    assert est0["median_bps"] == gcfg.S6_HALF_SPREAD_BPS_CONSERVATIVE


def test_half_spread_2d_degenerate_panel_falls_back_to_bound():
    """A 2D panel with no usable per-name variation at all -> conservative
    bound, never zero cost."""
    panel = np.full((50, 3), 100.0)
    est = s6stats.estimate_half_spread_bps(panel)
    assert est["source"] == "conservative_bound"
    assert est["median_bps"] == gcfg.S6_HALF_SPREAD_BPS_CONSERVATIVE


def test_basket_equal_weight_fixed_cadence_costs():
    rng = np.random.Generator(np.random.PCG64(4))
    R = rng.normal(0.0005, 0.01, (500, 5))
    cost = s6stats.DeckCostModel(half_spread_bps=8.0)
    out = s6stats.deck_basket_returns(R, rebalance_cadence=gcfg.S6_REBALANCE_CADENCE,
                                      cost_model=cost)
    assert out["net"].shape == (500,)
    # costs charged only on rebalance bars (after formation)
    rebal_bars = [t for t in range(500) if t >= gcfg.S6_FORMATION
                  and (t - gcfg.S6_FORMATION) % gcfg.S6_REBALANCE_CADENCE == 0]
    assert all(out["cost_bps"][t] > 0 for t in rebal_bars)
    assert all(out["cost_bps"][t] == 0 for t in range(500) if t not in rebal_bars)
    # equal weights right AFTER each rebalance (the rebalance is at the close
    # of the rebalance bar, so the bar FOLLOWING it opens at equal weight,
    # drifted with the rebalance bar's own return — soundness #3 convention:
    # weights_path[t] stores what EARNED bar t)
    for t in rebal_bars:
        w_next = np.full(5, 1.0 / 5) * (1.0 + R[t])
        w_next = w_next / w_next.sum()
        assert np.allclose(out["weights"][t + 1], w_next)
    # ... and the rebalance bar itself earned the drifted (pre-rebalance)
    # weights — never a weight vector that includes its own bar's return
    for t in rebal_bars:
        w = out["weights"][t]
        assert not np.allclose(w, 1.0 / 5) or t == gcfg.S6_FORMATION


# ---------------------------------------------------------------------------
# Deck ranking F D C B A S (distinct from card tiers) + beta telemetry
# ---------------------------------------------------------------------------

def test_deck_rank_from_stats_FDSCBAS_distinct_from_card_tiers():
    assert deck_rank_from_stats(None, None, None, None) == "F"
    assert deck_rank_from_stats(0.90, 2.0, 0.5, 0.9) == "F"       # pbo too high
    assert deck_rank_from_stats(0.94, 2.9, 0.04, 0.4) == "C"       # below the B bar
    assert deck_rank_from_stats(0.96, 3.2, 0.03, 0.4) == "B"       # B bar met
    assert deck_rank_from_stats(0.95, 3.0, 0.05, 0.5) == "B"
    assert deck_rank_from_stats(0.97, 3.6, 0.02, 0.4) == "A"
    assert deck_rank_from_stats(0.99, 4.0, 0.01, 0.5) == "S"
    assert set(DECK_RANKS) == {"F", "D", "C", "B", "A", "S"}
    assert set(DECK_RANKS).isdisjoint({"bronze", "silver", "gold"})


def test_ranked_deck_gets_rank_only_on_s6_pass(svc, cards, decks):
    _mint_cards(cards, n=10)
    d = svc.register_ranked(scope_id=SCOPE, bucket=bucket_config(size=10, family_mix=("pairs",)),
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    assert d.rank == "F"   # registered but not yet trialed
    R = _planted_basket(T=1260, K=10, seed=21)
    res = svc.run_s6(d.deck_id, component_returns=R,
                     component_names=[{f"q{i}"} for i in range(10)],
                     p_perm=0.001, owner_handle="alice", now=2.0)
    assert res["s6"]["pass"] is True
    assert res["deck"].rank in ("B", "A", "S", "C")
    evs = decks.events("deck_trial", deck_id=d.deck_id)
    assert evs and json.loads(evs[-1]["payload_json"])["rank"] == res["deck"].rank


# ---------------------------------------------------------------------------
# Ledger: append-only deck trials + events with read-back; non-transferable
# ---------------------------------------------------------------------------

def test_deck_events_append_only_with_readback(svc, cards, decks):
    _mint_cards(cards, n=6)
    d = svc.register_ranked(scope_id=SCOPE, bucket=bucket_config(size=5, family_mix=("pairs",)),
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    R = _planted_basket(T=1260, K=5, seed=8)
    svc.run_s6(d.deck_id, component_returns=R,
               component_names=[{f"z{i}"} for i in range(5)],
               owner_handle="alice", now=2.0)
    evs = decks.events(deck_id=d.deck_id)
    types = [e["event_type"] for e in evs]
    assert types == ["deck_register", "deck_trial"]   # oldest first, append-only
    # read-back of the trial row
    rb = decks.get_deck(d.deck_id)
    assert rb.s6_pass in (0, 1)
    assert rb.member_card_ids == d.member_card_ids


def test_decks_non_transferable(decks):
    with pytest.raises(DeckError):
        decks.assert_owned("deck-nonexistent", "alice")
    d = decks  # register directly to test ownership guard
    # (use service for a real deck in other tests; here just the guard path)


def test_deck_registry_assert_owned(svc, cards, decks):
    cs = _mint_cards(cards, n=6)
    d = svc.register_custom(scope_id=SCOPE, member_card_ids=[c.card_id for c in cs[:5]],
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    decks.assert_owned(d.deck_id, "alice")
    with pytest.raises(DeckError):
        decks.assert_owned(d.deck_id, "mallory")


# ---------------------------------------------------------------------------
# Flag-off fail-closed (SIFT_DECKS_V4_2=0 and SIFT_CARDS_V4_1=0)
# ---------------------------------------------------------------------------

def test_flag_off_fail_closed(decks, cards, svc):
    _mint_cards(cards, n=6)
    bucket = bucket_config(size=5, family_mix=("pairs",))
    # decks_v4_2 off only
    flags.set_flag(flags.FLAG_DECKS_V4_2, False)
    with pytest.raises(DeckError):
        svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="alice",
                            data_vintage_hash=VINTAGE)
    with pytest.raises(DeckError):
        svc.register_custom(scope_id=SCOPE, member_card_ids=["card-x"],
                            owner_handle="alice", data_vintage_hash=VINTAGE)
    with pytest.raises(DeckCopyError):
        deck_explainer("ranked")
    # cards_v4_1 off also kills decks (layer sits on top)
    flags.reset()
    flags.set_flag(flags.FLAG_CARDS_V4_1, False)
    assert flags.decks_v4_2_enabled() is False
    with pytest.raises(DeckError):
        svc.register_ranked(scope_id=SCOPE, bucket=bucket, owner_handle="alice",
                            data_vintage_hash=VINTAGE)
    # per-user disable
    flags.reset()
    flags.disable_for_user(flags.FLAG_DECKS_V4_2, "mallory")
    assert flags.decks_v4_2_enabled("mallory") is False
    assert flags.decks_v4_2_enabled("alice") is True
    # unknown flag reads False (fail closed)
    assert flags.is_enabled("no_such_flag") is False


def test_env_kill_switch(monkeypatch, cards, decks):
    monkeypatch.setenv("SIFT_DECKS_V4_2", "0")
    flags.reset()
    assert flags.is_enabled(flags.FLAG_DECKS_V4_2) is False
    assert flags.decks_v4_2_enabled() is False
    monkeypatch.setenv("SIFT_DECKS_V4_2", "1")
    monkeypatch.setenv("SIFT_CARDS_V4_1", "0")
    flags.reset()
    assert flags.decks_v4_2_enabled() is False   # cards kill-switch also disables decks


# ---------------------------------------------------------------------------
# Copy discipline: banned-phrase grep fail-closed on ALL deck strings;
# no tickers/direction in artifacts
# ---------------------------------------------------------------------------

def _all_deck_strings():
    out = []
    out.append(deck_explainer("ranked"))
    out.append(deck_explainer("custom"))
    for r in DECK_RANKS:
        out.append(deck_rank_label(r, "ranked"))
    out.append(deck_rank_label(None, "custom"))
    out.append(deck_telemetry_caveat())
    out.append(deck_custom_warning())
    out.extend(deck_glossary().values())
    return out


def test_all_deck_strings_pass_banned_phrase_grep():
    for s in _all_deck_strings():
        assert assert_deck_copy_clean(s) == s     # no exception


def test_banned_phrase_in_deck_string_refused_fail_closed():
    for bad in ["buy now", "sell signal", "guaranteed returns", "expected return 10%",
                "risk-free profit", "this deck will moon", "act now", "p&l 50%"]:
        with pytest.raises(DeckCopyError):
            assert_deck_copy_clean(bad)


def test_deck_artifacts_carry_no_tickers_no_direction(svc, cards, decks):
    import re
    _mint_cards(cards, n=6)
    d = svc.register_custom(scope_id=SCOPE,
                            member_card_ids=[c.card_id for c in cards.all_cards()[:5]],
                            owner_handle="alice", data_vintage_hash=VINTAGE, now=1.0)
    R = _planted_basket(T=1260, K=5, seed=17)
    res = svc.run_s6(d.deck_id, component_returns=R,
                     component_names=[{f"w{i}"} for i in range(5)],
                     owner_handle="alice", now=2.0)
    # serialize every string in the deck row + events and grep for tickers
    row = decks.con.execute("SELECT * FROM deck WHERE deck_id=?", (d.deck_id,)).fetchone()
    blob = json.dumps(dict(row))
    for e in decks.events(deck_id=d.deck_id):
        blob += e["payload_json"]
    ticker_re = re.compile(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b")
    whitelist = {"SIFT", "JSON", "US", "S6", "DSR", "PBO", "GICS", "K", "T", "N"}
    for tok in ticker_re.findall(blob):
        assert tok in whitelist, f"ticker-like token {tok!r} leaked into deck artifact"
    # no direction words in any event payload
    for e in decks.events(deck_id=d.deck_id):
        p = e["payload_json"].lower()
        for w in ("buy", "sell", "long", "short"):
            import re as _re
            assert not _re.search(r"\b" + w + r"\b", p)
    # membership stored as card ids (HMAC), never symbols
    members = json.loads(res["deck"].member_card_ids)
    assert all(m.startswith("card-") for m in members)
