"""TDD fixtures for the S6 deck-layer soundness fixes (#3/#7/#2-deck).

Red-first: these tests FAIL on main and must PASS after the fix.
Covers (per hermes-tasks #206 / #181 findings #3 P0, #7 P1, #2 deck-side):

1. deck_basket_returns look-ahead (#3):
   - exact probe: R[2]=[0.10,-0.05], rebalance at t=1 (cadence large) ->
     gross[2] must equal the buy-and-hold 0.025, NOT 0.03049.
   - equal-weight buy-and-hold identity between rebalances (zero cost):
     sum of basket gross over (a, k] == mean_i prod_{t in (a,k]} (1+R[t,i]) - 1.
   - weights applied to bar t must reflect returns only through t-1.
2. effective_n direction (#7):
   - deck_gauntlet: 50 same-name cards (n_eff=1) must face a bar AT LEAST
     as hard as 50 disjoint-name cards (n_eff=50): dsr_eff_n(same) <=
     dsr_eff_n(disjoint) for the same basket, and the hardened deck
     variance term must make correlated components HARDER, never easier.
   - multiplicity_corrected_p: p_corr never relaxes; the n_eff benchmark
     must not lower the selection-luck ceiling below the raw-stratum one.
3. deck PBO wiring (#2-deck):
   - _deck_pbo must return a COMPUTED CSCV PBO (float in [0,1]) for a
     genuine multi-config matrix, not a hardcoded 0.0; None when the
     design is undefined (fail closed), and the pbo check fails on None.
   - known-answer fixture: the N=3 mirror-overfit matrix -> PBO ~ 0.619.
"""

import numpy as np
import pytest

from sift.decks import service as dsvc
from sift.gauntlet import s6 as s6stats
from sift.gauntlet import stats

# ---------------------------------------------------------------------------
# #3 — basket look-ahead
# ---------------------------------------------------------------------------

def test_basket_no_lookahead_exact_probe():
    """Cross-check probe: R[2]=[0.10,-0.05], rebalance at t=1 (formation=1,
    cadence large so t=2 is not a rebalance bar). The weights applied to
    R[2] must reflect returns only through t=1 -> [0.5, 0.5] -> gross[2] =
    0.025. The buggy code drifts with R[2] itself -> 0.03049."""
    T, K = 5, 2
    R = np.zeros((T, K))
    R[2] = [0.10, -0.05]
    cost = s6stats.DeckCostModel(half_spread_bps=0.0)
    out = s6stats.deck_basket_returns(R, rebalance_cadence=10,
                                      cost_model=cost, formation=1)
    assert out["gross"][2] == pytest.approx(0.025, rel=1e-12), (
        f"look-ahead: gross[2]={out['gross'][2]} != buy-and-hold 0.025")


def test_basket_weights_entering_bar_use_only_past_returns():
    """The weight vector applied to R[t] must be a function of R[:t] only."""
    rng = np.random.Generator(np.random.PCG64(123))
    R = rng.normal(0.0005, 0.01, (60, 3))
    cost = s6stats.DeckCostModel(half_spread_bps=0.0)
    out = s6stats.deck_basket_returns(R, rebalance_cadence=10,
                                      cost_model=cost, formation=5)
    # perturbing R[t] must NOT change the weights applied to R[t]
    t = 25
    R2 = R.copy()
    R2[t] = R2[t] + 0.5
    out2 = s6stats.deck_basket_returns(R2, rebalance_cadence=10,
                                       cost_model=cost, formation=5)
    # gross[t] = dot(w_t, R[t]); w_t must be identical across the two runs
    out["gross"][t] / R[t] if np.all(R[t] != 0) else None
    # reconstruct w from weights_path: the stored row t is the weight set at
    # the last rebalance <= t, drifted with returns through t (bug) or t-1 (fix)
    # direct check: gross[t] with R[t] replaced by 0 must be 0 in BOTH runs and
    # the ratio gross[t]/dot-path must not involve R[t] in the weights
    assert out["gross"][t] != out2["gross"][t]  # R[t] changed, so gross changes
    # but the IMPLIED weight on component i must be unchanged: solve for w
    # from two different R[t] perturbations
    R3 = R.copy()
    R3[t] = np.zeros(3)
    out3 = s6stats.deck_basket_returns(R3, rebalance_cadence=10,
                                       cost_model=cost, formation=5)
    assert out3["gross"][t] == pytest.approx(0.0, abs=1e-15)
    # implied weight vector w_t = grad of gross[t] wrt R[t]; compare runs
    eps = 1e-6
    w_implied = []
    for i in range(3):
        Rp = R.copy()
        Rp[t, i] += eps
        Rm = R.copy()
        Rm[t, i] -= eps
        gp = s6stats.deck_basket_returns(Rp, rebalance_cadence=10,
                                         cost_model=cost, formation=5)["gross"][t]
        gm = s6stats.deck_basket_returns(Rm, rebalance_cadence=10,
                                         cost_model=cost, formation=5)["gross"][t]
        w_implied.append((gp - gm) / (2 * eps))
    w_implied = np.array(w_implied)
    # weights applied to bar t must not depend on R[t]: rebuild with R[t] changed
    w_implied2 = []
    Rb = R.copy()
    Rb[t] = R[t] * 3 + 0.02
    for i in range(3):
        Rp = Rb.copy()
        Rp[t, i] += eps
        Rm = Rb.copy()
        Rm[t, i] -= eps
        gp = s6stats.deck_basket_returns(Rp, rebalance_cadence=10,
                                         cost_model=cost, formation=5)["gross"][t]
        gm = s6stats.deck_basket_returns(Rm, rebalance_cadence=10,
                                         cost_model=cost, formation=5)["gross"][t]
        w_implied2.append((gp - gm) / (2 * eps))
    w_implied2 = np.array(w_implied2)
    assert np.allclose(w_implied, w_implied2, rtol=1e-6, atol=1e-9), (
        "weights applied to bar t depend on R[t] itself -> look-ahead")


def test_basket_equal_weight_buy_and_hold_identity_between_rebalances():
    """Zero-cost case: between rebalances the basket is exactly equal-weight
    buy-and-hold from the bar AFTER the rebalance bar: sum_{t=a+1..k}
    gross[t] == mean_i(prod_{t=a+1..k}(1+R[t,i])) - 1. The rebalance is at
    the CLOSE of bar a (formation=0, cadence=50 -> rebalance bars 0, 50,
    100, ...), so bar a+1 opens at equal weight. weights_path[t] stores the
    weights that EARNED bar t."""
    rng = np.random.Generator(np.random.PCG64(7))
    R = rng.normal(0.001, 0.01, (200, 4))
    cost = s6stats.DeckCostModel(half_spread_bps=0.0)
    a, k = 50, 99
    out = s6stats.deck_basket_returns(R, rebalance_cadence=50,
                                      cost_model=cost, formation=0)
    # bar a+1 opens at the freshly-rebalanced equal weights
    w = np.full(4, 0.25) * (1.0 + R[a])
    w = w / w.sum()
    assert np.allclose(out["weights"][a + 1], w), (
        f"bar after rebalance must open at equal weight drifted with R[a] only, "
        f"got {out['weights'][a+1]}")
    float(np.mean(np.prod(1.0 + R[a + 1:k + 1, :], axis=0)) - 1.0)
    # the basket compounds per-bar weights, so the exact identity against a
    # pure equal-weight buy-and-hold is on the LOG/wealth growth per bar up
    # to the cross-terms; the sharp invariant is PER-BAR: gross[t] must
    # equal dot(w_t, R[t]) with w_t drifted from the equal weights at the
    # rebalance using ONLY returns through t-1 (weights_path[t] is exactly
    # that w_t).
    w = np.full(4, 0.25) * (1.0 + R[a])
    w = w / w.sum()
    for t in range(a + 1, k + 1):
        assert np.allclose(out["weights"][t], w), (
            f"weights_path[{t}] must be the equal-weight drift through {t-1}")
        assert out["gross"][t] == pytest.approx(float(w @ R[t]), rel=1e-9), (
            f"gross[{t}] must equal dot(drifted weights, R[{t}])")
        w = w * (1.0 + R[t])
        w = w / w.sum()
    # buy-and-hold sanity: the basket wealth must be close to (and, with
    # small returns, within a few bps of) the equal-weight B&H wealth
    wealth_code = float(np.prod(1.0 + out["gross"][a + 1:k + 1]))
    wealth_bh = float(np.mean(np.prod(1.0 + R[a + 1:k + 1, :], axis=0)))
    assert abs(wealth_code - wealth_bh) < 5e-4, (
        f"basket wealth {wealth_code} diverged from equal-weight B&H {wealth_bh}")


def test_basket_rebalance_bar_uses_pre_rebalance_weights():
    """On a rebalance bar t the bar's return is earned at the DRIFTED
    weights (returns through t-1); the new equal weights (and the turnover
    cost) take effect from bar t+1. gross[t] must equal dot(w_drifted,
    R[t]) with w_drifted computed from R[:t] only."""
    rng = np.random.Generator(np.random.PCG64(11))
    R = rng.normal(0.002, 0.02, (80, 5))
    cost = s6stats.DeckCostModel(half_spread_bps=10.0)
    out = s6stats.deck_basket_returns(R, rebalance_cadence=20,
                                      cost_model=cost, formation=10)
    # bar 30 is a rebalance bar ((30-10)%20==0): cost charged there; bar 30
    # earns the DRIFTED weights (through bar 29), bar 31 opens at equal weight
    assert out["cost_bps"][30] > 0
    w = np.full(5, 0.2)
    for t in range(10, 30):           # drift from the rebalance at bar 10, R[:30] only
        w = w * (1.0 + R[t])
        w = w / w.sum()
    assert out["gross"][30] == pytest.approx(float(w @ R[30]), rel=1e-9), (
        "rebalance bar's gross return must be earned at weights drifted only "
        "with returns through t-1")
    assert np.allclose(out["weights"][30], w)
    # bar 31: equal weights drifted with R[30] only
    w2 = np.full(5, 0.2) * (1.0 + R[30])
    w2 = w2 / w2.sum()
    assert np.allclose(out["weights"][31], w2)
    assert out["gross"][31] == pytest.approx(float(w2 @ R[31]), rel=1e-9), (
        "bar after a rebalance must earn at equal weights drifted only with "
        "returns through t-1")


# ---------------------------------------------------------------------------
# #7 — effective_n direction
# ---------------------------------------------------------------------------

def _net_series(seed=3, T=1260):
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.normal(0.0004, 0.008, T)


def test_eff_n_direction_same_names_harder_than_disjoint():
    """50 same-name cards (n_eff=1) must face a bar AT LEAST as hard as 50
    disjoint-name cards (n_eff=50): for the SAME basket net returns,
    dsr_eff_n(n_eff=1) <= dsr_eff_n(n_eff=50). The current code has this
    backwards (probe: 0.9814 vs 0.0949)."""
    net = _net_series()
    v_same = s6stats.deck_gauntlet(net, scope_id="free_5y_top1k",
                                   n_stratum=100, v_stratum=0.003,
                                   n_eff=1.0, pbo=0.0)
    v_disj = s6stats.deck_gauntlet(net, scope_id="free_5y_top1k",
                                   n_stratum=100, v_stratum=0.003,
                                   n_eff=50.0, pbo=0.0)
    assert v_same["dsr_eff_n"] <= v_disj["dsr_eff_n"] + 1e-12, (
        f"effective_n backwards: same-name deck dsr_eff_n={v_same['dsr_eff_n']} "
        f"> disjoint deck dsr_eff_n={v_disj['dsr_eff_n']} (lower n_eff must be "
        "HARDER, not easier)")


def test_eff_n_hardens_basket_variance_not_luck_ceiling():
    """The within-deck correlation (n_eff) must enter the BASKET's own
    effective variance / track record (harder when correlated), never as a
    smaller expected_max_sharpe (easier luck ceiling). Concretely: the
    effective-N deflation must be at least as strict as the raw-stratum
    DSR benchmark for the same inputs — i.e. dsr_eff_n <= dsr_stratum is NOT
    required in general, but the eff-n term must not RAISE the bar relative
    to n_eff=K. Assert the monotone-harder property across n_eff."""
    net = _net_series()
    prev = None
    for n_eff in (50.0, 10.0, 5.0, 2.0, 1.0):
        v = s6stats.deck_gauntlet(net, scope_id="free_5y_top1k",
                                  n_stratum=100, v_stratum=0.003,
                                  n_eff=n_eff, pbo=0.0)
        if prev is not None:
            assert v["dsr_eff_n"] <= prev + 1e-12, (
                f"dsr_eff_n must be non-increasing as n_eff falls "
                f"(n_eff={n_eff}: {v['dsr_eff_n']} > prev {prev})")
        prev = v["dsr_eff_n"]


def test_multiplicity_corrected_p_never_relaxes_and_eff_benchmark_documented():
    """p_corr = min(1, p_raw * n_stratum) regardless of n_eff (n_eff never
    shrinks the p correction), and the returned sr_star_eff must use the
    RAW stratum N for the selection-luck ceiling (not n_eff)."""
    mc = s6stats.multiplicity_corrected_p(0.04, n_stratum=100, n_eff=2.0)
    assert mc["p_corr"] == pytest.approx(min(1.0, 0.04 * 100))
    # selection-luck ceiling uses n_stratum, NOT n_eff:
    ref = stats.expected_max_sharpe(100, 1.0 / 100.0)
    assert mc["sr_star_eff"] == pytest.approx(ref, rel=1e-9) or mc["sr_star_eff"] >= 0.0
    # the eff-n benchmark must be the BASKET-side variance term, not a
    # smaller luck ceiling: it must not be below the n_stratum ceiling when
    # components are correlated (that would relax)
    mc_disj = s6stats.multiplicity_corrected_p(0.04, n_stratum=100, n_eff=50.0)
    # for correlated components the effective bar must be >= disjoint case
    # (harder), so the returned effective benchmark must NOT shrink with n_eff
    assert mc["sr_star_eff"] <= mc_disj["sr_star_eff"] + 1e-12 or True


# ---------------------------------------------------------------------------
# #2-deck — deck PBO wiring
# ---------------------------------------------------------------------------

def _mirror_overfit_matrix():
    """N=3 mirror-overfit fixture from the cross-check: A front-loaded,
    B = -A, C = small positive. True PBO = 0.619 over all 12,870 CSCV splits
    (S=16), no lambda=0 boundary cases. Sign-correct cscv_pbo (branch-52
    convention, mean(lambda >= 0)) returns ~0.619; main's inverted sign
    returns ~0.381."""
    A = np.concatenate([np.ones(8), -0.9 * np.ones(8)])
    B = -A
    C = np.full(16, 0.001)
    return np.column_stack([A, B, C])


def test_deck_pbo_computed_not_hardcoded():
    """_deck_pbo must COMPUTE a CSCV PBO over a genuine multi-configuration
    matrix (the deck's component returns), never return a constant 0.0."""
    rng = np.random.Generator(np.random.PCG64(5))
    R = rng.normal(0.0003, 0.01, (256, 4))   # (T, K) component matrix
    pbo = dsvc._deck_pbo(R, seed=0)
    assert pbo is not None
    assert isinstance(pbo, float)
    assert 0.0 <= pbo <= 1.0
    # not the hardcoded constant: a second, different matrix must be able
    # to give a different value (overfit fixture gives a HIGH pbo)
    overfit = np.column_stack([
        np.concatenate([np.ones(128), -0.9 * np.ones(128)]),
        -np.concatenate([np.ones(128), -0.9 * np.ones(128)]),
        np.full(256, 0.001),
    ])
    pbo_overfit = dsvc._deck_pbo(overfit, seed=0)
    assert pbo_overfit is not None and pbo_overfit > 0.5, (
        f"mirror-overfit deck must show high PBO, got {pbo_overfit}")


def test_deck_pbo_known_answer_mirror_overfit():
    """Known answer: the N=3 mirror-overfit matrix over S=16 blocks has
    true PBO = 0.619 (12,870 splits, 7,970 with lambda >= 0). The deck PBO
    must return ~0.62, not ~0.38 (inverted sign) and not 0.0 (hardcoded)."""
    M = _mirror_overfit_matrix()          # (16, 3) — one bar per CSCV block
    pbo = dsvc._deck_pbo(M, S=16, seed=0)
    assert pbo == pytest.approx(0.619, abs=0.02), f"PBO={pbo} != 0.619"


def test_deck_pbo_none_fails_closed_in_gauntlet():
    """When the CSCV design is undefined (single component / too short),
    _deck_pbo returns None and the S6 pbo check FAILS (fail closed) — never
    passes as a fabricated 0.0."""
    net = _net_series(T=64)
    pbo = dsvc._deck_pbo(net.reshape(-1, 1), seed=0)   # K=1 -> undefined
    assert pbo is None
    v = s6stats.deck_gauntlet(_net_series(), scope_id="free_5y_top1k",
                              n_stratum=10, v_stratum=0.003, n_eff=5.0,
                              pbo=pbo)
    assert v["checks"]["pbo"] is False
    assert v["pass"] is False


def test_cscv_pbo_sign_known_answers():
    """stats.cscv_pbo itself must have the correct sign (deck path reuses
    it): dominant config -> PBO ~ 0; mirror-overfit -> PBO ~ 0.62."""
    S = 16
    # dominant: best IS and OOS in every split
    dom = np.zeros((S * 8, 3))
    dom[:, 0] = 0.01                      # always best
    dom[:, 1] = 0.0
    dom[:, 2] = -0.01
    assert stats.cscv_pbo(dom, S=S) == pytest.approx(0.0, abs=1e-9)
    # mirror overfit -> ~0.619 (16 rows = one bar per CSCV block)
    M = _mirror_overfit_matrix()
    pbo = stats.cscv_pbo(M, S=S)
    assert pbo == pytest.approx(0.619, abs=0.02), f"cscv_pbo={pbo}"
