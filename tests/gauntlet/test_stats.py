"""Unit tests for the pure gauntlet stat functions (S0-S4)."""

import math

import numpy as np
import pytest

from sift.gauntlet import config as cfg
from sift.gauntlet import stats


# -- normal helpers ---------------------------------------------------------

def test_norm_cdf_ppf_roundtrip():
    for p in (0.01, 0.05, 0.5, 0.95, 0.99):
        x = stats._norm_ppf(p)
        assert stats._norm_cdf(x) == pytest.approx(p, abs=1e-9)


# -- S0 ----------------------------------------------------------------------

def test_s0_calibration_recovers_ar1():
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    r = stats.calibrate_ar1_fixture(
        fx["prices"][:, 0], fx["prices"][:, 1],
        fx["beta_true"], fx["half_life_true"],
    )
    assert r.beta_ok
    assert r.half_life_ok
    assert r.passed


def test_s0_rejects_wrong_beta():
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    r = stats.calibrate_ar1_fixture(
        fx["prices"][:, 0], fx["prices"][:, 1], beta_true=5.0,
        half_life_true=fx["half_life_true"],
    )
    assert not r.beta_ok
    assert not r.passed


# -- S1 ----------------------------------------------------------------------

def test_s1_screen_threshold():
    rng = np.random.Generator(np.random.PCG64(1))
    noise = rng.normal(0, 0.01, 500)
    assert not stats.discovery_screen(noise)["pass"]
    effect = rng.normal(0.001, 0.005, 500)
    assert stats.discovery_screen(effect)["pass"]


# -- S2 ----------------------------------------------------------------------

def test_block_length_matches_design():
    assert stats.block_length(1260) == 11   # ceil(1260^(1/3))
    assert stats.block_length(4032) == 16   # ceil(4032^(1/3))
    assert stats.block_length(10) == 5      # min block


def test_permutation_pvalue_granularity():
    reps = np.zeros(1000)
    assert stats.permutation_pvalue(1.0, reps) == pytest.approx(1 / 1001)
    reps[::2] = 5.0
    assert stats.permutation_pvalue(1.0, reps) == pytest.approx(501 / 1001)


def test_s2_detects_signal_and_passes_null():
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    prices = fx["prices"]
    x, y = prices[:, 0], prices[:, 1]
    rx = np.diff(np.log(x)); ry = np.diff(np.log(y))
    rs = ry - rx
    # mean-reversion content: -corr(spread level t, next-bar spread return)
    stat = stats.mean_reversion_stat
    r = stats.block_permutation_null(rs, stat, B=200, seed=3)
    assert r["p_perm"] <= 0.05
    # pure noise: p not extreme
    rng = np.random.Generator(np.random.PCG64(42))
    noise = rng.normal(0, 0.01, 1260)
    r2 = stats.block_permutation_null(noise, stat, B=200, seed=4)
    assert r2["p_perm"] > 0.05


def test_s2_deterministic_given_seed():
    rng = np.random.Generator(np.random.PCG64(9))
    s = np.cumsum(rng.normal(0, 0.01, 500))
    a = stats.block_permutation_null(s, lambda v: v.mean(), B=50, seed=11)
    b = stats.block_permutation_null(s, lambda v: v.mean(), B=50, seed=11)
    assert a["p_perm"] == b["p_perm"]
    assert np.array_equal(a["replicates"], b["replicates"])


# -- S3 ----------------------------------------------------------------------

def test_expected_max_sharpe_monotone_in_N():
    v = 1 / 1260
    e12 = stats.expected_max_sharpe(12, v)
    e1000 = stats.expected_max_sharpe(1000, v)
    assert e1000 > e12 > 0


def test_dsr_at_or_below_benchmark_is_low():
    T = 1260
    d = stats.dsr(sr_hat=0.0, T=T, n_stratum=1000, v_stratum=1 / T)
    assert d["dsr"] < 0.5
    # a huge Sharpe clears the bar
    d2 = stats.dsr(sr_hat=0.5, T=T, n_stratum=1000, v_stratum=1 / T)
    assert d2["dsr"] > 0.95


def test_dsr_var_floor():
    # V floored at the normal-theory Var(SR_hat) = (1 + sr_hat^2/2)/T when the
    # stratum variance is smaller (#16: PSR variance convention, not 1/T)
    sr_hat = 0.1
    d = stats.dsr(sr_hat=sr_hat, T=1260, n_stratum=10, v_stratum=0.0)
    assert d["v_stratum"] == pytest.approx((1.0 + sr_hat ** 2 / 2.0) / 1260)


def test_bh_fdr_controls():
    rng = np.random.Generator(np.random.PCG64(5))
    p = rng.uniform(0, 1, 100)
    out = stats.bh_fdr(p, q=0.10)
    assert out["m"] == 100
    assert len(out["q_values"]) == 100
    # all-tiny p-values reject everything
    out2 = stats.bh_fdr(np.full(20, 0.001), q=0.10)
    assert len(out2["rejected"]) == 20


def test_min_trl_positive_and_grows_with_confidence():
    t1 = stats.min_trl(0.1, 0.0, 3.0, alpha=0.05)
    t2 = stats.min_trl(0.1, 0.0, 3.0, alpha=0.01)
    assert t1 > 1
    assert t2 > t1


# -- S4 ----------------------------------------------------------------------

def test_costs_reduce_returns():
    gross = np.array([0.0, 0.001, 0.001, -0.001])
    pos = np.array([0.0, 1.0, 1.0, 0.0])
    cm = stats.CostModel(half_spread_bps=100,
                         slippage_participation_threshold=1e9)  # spread only
    net = stats.apply_costs(gross, pos, cm)   # single-leg (no hedge): legacy path
    # soundness #9: cost is charged at the FILL bar t+1, not the decision bar
    assert net[1] == gross[1]                 # decision bar: no cost now
    assert net[2] < gross[2]                  # fill bar of the t=1 entry
    assert net[3] < gross[3]                  # fill bar of the t=2 exit
    # single-leg rate: |dpos| * 100/1e4 charged one bar later
    assert net[2] == pytest.approx(gross[2] - 1.0 * 100 / 1e4, abs=1e-12)
    assert net[3] == pytest.approx(gross[3] - 1.0 * 100 / 1e4, abs=1e-12)


def test_pairs_strategy_no_lookahead_and_costs():
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    out = stats.pairs_strategy_returns(fx["prices"])
    gross, net, pos = out["gross"], out["net"], out["positions"]
    assert gross.shape == net.shape == pos.shape
    assert np.all(net[1:] <= gross[1:] + 1e-12)
    # positions only after formation window
    assert np.all(pos[:60] == 0)


def test_walk_forward_runs():
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    out = stats.walk_forward_returns(fx["prices"], formation=60, step=120)
    assert out["net"].shape[0] == fx["prices"].shape[0]
    assert np.isfinite(out["net"]).all()


def test_cscv_pbo_bounds():
    rng = np.random.Generator(np.random.PCG64(0))
    R = rng.normal(0, 0.01, (256, 8))
    pbo = stats.cscv_pbo(R, S=16, seed=1)
    assert 0.0 <= pbo <= 1.0
    # direction: under the null of iid configs PBO concentrates near 0.5,
    # and a dominant config (best IS+OOS every split) has PBO exactly 0 —
    # see tests/gauntlet/test_pbo_soundness.py for the known-answer fixtures
    # (soundness #2: the old <= 0 comparison returned 1 - PBO here).


def test_cscv_pbo_direction_and_none():
    """Soundness #2: cscv_pbo must return PBO (overfit -> HIGH), not 1-PBO,
    and None (not 0.0) when the CSCV design is undefined (N<2 or T<S)."""
    S = 16
    # dominant config: best IS and OOS in every split -> true PBO = 0
    dom = np.zeros((S * 8, 3))
    dom[:, 0] = 0.01
    dom[:, 1] = 0.0
    dom[:, 2] = -0.01
    assert stats.cscv_pbo(dom, S=S) == pytest.approx(0.0, abs=1e-9)
    # mirror-overfit N=3 fixture: true PBO = 0.619 over 12,870 splits
    A = np.concatenate([np.ones(8), -0.9 * np.ones(8)])
    M = np.column_stack([A, -A, np.full(16, 0.001)])
    assert stats.cscv_pbo(M, S=S) == pytest.approx(0.619, abs=0.02)
    # undefined designs -> None (fail closed), never a fabricated 0.0
    assert stats.cscv_pbo(np.zeros((256, 1)), S=S) is None      # N < 2
    assert stats.cscv_pbo(np.zeros((8, 3)), S=S) is None         # T < S
    assert stats.cscv_pbo(np.zeros(256), S=S) is None            # not 2D


# -- config -------------------------------------------------------------------

def test_tau_cal_scope_matched():
    assert cfg.tau_cal("free_5y_top1k") == pytest.approx(3.5)
    assert cfg.tau_cal("pro_full") == pytest.approx(3.0)
    assert cfg.tau_cal("registry_depth") == pytest.approx(3.0)
