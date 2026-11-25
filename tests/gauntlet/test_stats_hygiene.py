"""TDD fixtures (RED first) for statistical hygiene: HAC/effective-T option,
bias-corrected skew/kurt, honest S0 real-data docs (soundness #12/#13/#14).

Conventions under test:
* Newey-West HAC t with Bartlett kernel, lag rule floor(4*(T/100)^(2/9))
  (Andrews 1991), min 1, capped at T-1.
* Effective sample size T_eff = T * g0 / S (g0 = gamma_0, S = long-run var).
* Bias-corrected sample skew G1 and RAW kurtosis G2 (excess + 3), adjusted
  Fisher-Pearson; degenerate guards preserved (n<4 or zero variance ->
  (0.0, 3.0)).
* PSR/DSR take an optional T_eff (effective sample size) in place of T-1;
  default keeps the iid sqrt(T-1) behavior.
"""

import math

import numpy as np
import pytest

from sift.gauntlet import stats

# ---------------------------------------------------------------- #12 HAC

def test_nw_lag_rule_matches_andrews():
    # floor(4*(T/100)^(2/9)), floored at 1
    assert stats.nw_lag(252) == math.floor(4 * (252 / 100) ** (2 / 9)) == 4
    assert stats.nw_lag(1260) == math.floor(4 * (1260 / 100) ** (2 / 9)) == 7
    assert stats.nw_lag(10) == max(1, math.floor(4 * (10 / 100) ** (2 / 9))) == 2


def test_hac_t_known_answer_planted_series():
    # Known answer from the independent explicit-loop reference (probe):
    # T=6 -> lag 2, Bartlett weights; g0=2.8888888889e-04, S=4.6913580247e-05
    r = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
    out = stats.t_stat_sharpe_hac(r)
    assert out["lag"] == 2
    assert out["gamma0"] == pytest.approx(2.8888888889e-04, rel=1e-9)
    assert out["long_run_var"] == pytest.approx(4.6913580247e-05, rel=1e-9)
    assert out["t"] == pytest.approx(2.3841582427, rel=1e-9)
    # differs from the iid t on this autocorrelated-looking series
    assert out["t"] != pytest.approx(stats.t_stat_sharpe(r), rel=1e-6)


def test_hac_t_iid_series_close_to_iid_t():
    rng = np.random.Generator(np.random.PCG64(2026))
    r = rng.normal(0.0005, 0.01, 1260)
    hac = stats.t_stat_sharpe_hac(r)
    iid = stats.t_stat_sharpe(r)
    assert hac["t"] == pytest.approx(iid, rel=0.15)
    assert hac["lag"] == stats.nw_lag(1260)


def test_hac_t_mc_direction_positive_autocorr():
    # 2000-rep MC, T=252, AR(1) phi=+0.5: iid t dispersion ~1.73 (theory
    # sqrt((1+phi)/(1-phi))); HAC t dispersion must be materially smaller and
    # near 1. At phi=-0.5 the iid t is conservative (~0.58); HAC stays ~1.
    def ar1(T, phi, rng):
        e = rng.normal(0, 0.01, T)
        s = np.zeros(T)
        for t in range(1, T):
            s[t] = phi * s[t - 1] + e[t]
        return s

    rng = np.random.Generator(np.random.PCG64(1234))
    ts_iid, ts_hac = [], []
    for _ in range(2000):
        r = ar1(252, 0.5, rng)
        ts_iid.append(stats.t_stat_sharpe(r))
        ts_hac.append(stats.t_stat_sharpe_hac(r)["t"])
    sd_iid = float(np.std(ts_iid))
    sd_hac = float(np.std(ts_hac))
    assert sd_iid == pytest.approx(1.732, abs=0.15)     # anticonservative iid
    assert sd_hac < 1.45                                # HAC removes the inflation
    assert sd_hac > 0.7                                 # not over-shrunk

    rng = np.random.Generator(np.random.PCG64(4321))
    ts_iid, ts_hac = [], []
    for _ in range(2000):
        r = ar1(252, -0.5, rng)
        ts_iid.append(stats.t_stat_sharpe(r))
        ts_hac.append(stats.t_stat_sharpe_hac(r)["t"])
    assert float(np.std(ts_iid)) == pytest.approx(0.577, abs=0.08)  # conservative iid
    assert 0.75 < float(np.std(ts_hac)) < 1.35


def test_effective_sample_size_ar1():
    # T_eff = T * gamma0 / S. On a planted AR(1) phi=0.5, T=252 the HAC
    # deflation must be substantial (theory T*(1-phi)/(1+phi) = 84); on iid
    # data T_eff ~ T.
    def ar1(T, phi, rng):
        e = rng.normal(0, 0.01, T)
        s = np.zeros(T)
        for t in range(1, T):
            s[t] = phi * s[t - 1] + e[t]
        return s

    rng = np.random.Generator(np.random.PCG64(7))
    r = ar1(252, 0.5, rng)
    t_eff = stats.effective_sample_size(r)
    assert 60.0 < t_eff < 160.0          # deflated well below 252 (theory ~84)
    assert t_eff < 252

    rng = np.random.Generator(np.random.PCG64(8))
    iid = rng.normal(0.0005, 0.01, 1260)
    t_eff_iid = stats.effective_sample_size(iid)
    assert t_eff_iid == pytest.approx(1260, rel=0.15)


def test_psr_dsr_accept_effective_T():
    # PSR with T_eff replaces sqrt(T-1) with sqrt(T_eff-1): on positive
    # autocorrelation (T_eff < T) the PSR must DROP vs the iid PSR at the
    # same sr_hat — i.e. the effective-T option is the conservative one
    # under positive autocorrelation.
    sr_hat, skew, kurt = 0.05, 0.0, 3.0
    T = 252
    p_iid = stats.psr(sr_hat, sr_star=0.0, T=T, skew=skew, kurt=kurt)
    p_eff = stats.psr(sr_hat, sr_star=0.0, T=T, skew=skew, kurt=kurt,
                      t_eff=100.0)
    assert p_eff < p_iid
    # dsr passes t_eff through and records it (against the DSR benchmark
    # sr_star, which is NOT rescaled — only the candidate's own evidence is)
    d = stats.dsr(sr_hat, T, 100, 1.0 / T, skew=skew, kurt=kurt, t_eff=100.0)
    # #16: dsr's default var_floor is now the PSR variance (1+sr_hat^2/2)/T,
    # so the effective V is max(1/T, floor) — recompute sr_star with THAT V:
    v_eff = max(1.0 / T, (1.0 + sr_hat ** 2 / 2.0) / T)
    sr_star = stats.expected_max_sharpe(100, v_eff)
    assert d["dsr"] == pytest.approx(
        stats.psr(sr_hat, sr_star, T, skew, kurt, t_eff=100.0), rel=1e-12)
    assert d["t_eff"] == 100.0
    # vs the same DSR WITHOUT t_eff (iid sqrt(T-1)): here sr_hat < sr_star so
    # shrinking the evidence moves PSR AWAY from 1 — i.e. toward 0 for a
    # positive-looking but under-evidenced candidate: the effective-T option
    # never inflates PSR above the iid value at fixed sr_hat > sr_star, and
    # always reduces the evidence term.
    d_iid = stats.dsr(sr_hat, T, 100, 1.0 / T, skew=skew, kurt=kurt)
    assert d["dsr"] != pytest.approx(d_iid["dsr"], rel=1e-9)
    big = stats.psr(0.5, 0.0, T, skew, kurt)
    big_eff = stats.psr(0.5, 0.0, T, skew, kurt, t_eff=100.0)
    assert big_eff < big  # positive autocorrelation: effective-T is conservative
    # default behavior unchanged (t_eff None -> sqrt(T-1), same sr_star)
    d0 = stats.dsr(sr_hat, T, 100, 1.0 / T, skew=skew, kurt=kurt)
    assert d0["dsr"] == pytest.approx(
        stats.psr(sr_hat, sr_star, T, skew, kurt), rel=1e-12)
    assert d0["t_eff"] is None


def test_hac_t_degenerate_guards():
    assert stats.t_stat_sharpe_hac(np.array([1.0]))["t"] == 0.0
    assert stats.t_stat_sharpe_hac(np.array([0.01, 0.01, 0.01]))["t"] == 0.0
    assert stats.effective_sample_size(np.array([0.01, 0.01])) == 0.0


# ------------------------------------------------------- #13 bias-corrected

def test_skew_kurt_bias_corrected_known_answer():
    # Known answer (independent adjusted-moment reference, probe):
    # n=8 -> raw (0.0065885587, 1.3742289294), corrected G1=0.0100397085,
    # raw kurtosis G2=1.8693136350 (RAW, not excess).
    r = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.005, -0.015])
    skew, kurt = stats.skew_kurt(r)
    assert skew == pytest.approx(0.0100397085, rel=1e-8)
    assert kurt == pytest.approx(1.8693136350, rel=1e-8)
    assert kurt < 3.0  # still raw kurtosis (not excess); this sample is platykurtic


def test_skew_kurt_mc_unbiased_normal():
    # 2000-rep MC on N(0,1), n=60: corrected raw kurtosis mean ~3.0
    # (the old estimator gave 2.797-2.80 at n=60 — downward bias).
    rng = np.random.Generator(np.random.PCG64(99))
    ks, ss = [], []
    for _ in range(2000):
        r = rng.normal(0, 0.01, 60)
        s, k = stats.skew_kurt(r)
        ks.append(k)
        ss.append(s)
    assert float(np.mean(ks)) == pytest.approx(3.0, abs=0.08)
    assert abs(float(np.mean(ss))) < 0.08


def test_skew_kurt_guards_preserved():
    assert stats.skew_kurt(np.array([0.01, 0.02, 0.03])) == (0.0, 3.0)      # n<4
    assert stats.skew_kurt(np.array([0.01, 0.01, 0.01, 0.01])) == (0.0, 3.0)  # zero var
    s, k = stats.skew_kurt(np.array([1.0, 2.0, 3.0, 4.0]))
    assert math.isfinite(s) and math.isfinite(k)


def test_psr_dsr_mintrl_consumers_consistent():
    # PSR expects RAW kurtosis: feeding excess (kurt-3) must change PSR the
    # same way as before the fix; consumers pass stats.skew_kurt's raw kurt
    # straight through.
    p_raw = stats.psr(0.05, 0.0, 252, skew=0.0, kurt=3.0)
    p_excess = stats.psr(0.05, 0.0, 252, skew=0.0, kurt=0.0)
    assert p_raw != pytest.approx(p_excess)
    # min_trl takes raw kurt too (formula uses (kurt-1))
    t_raw = stats.min_trl(0.1, 0.0, 3.0)
    t_excess = stats.min_trl(0.1, 0.0, 0.0)
    assert t_raw > t_excess  # raw kurt 3 -> (3-1)*sr^2/4 > 0 adds to MinTRL


# ------------------------------------------------------- #14 S0 doc-only

def test_s0_real_data_role_documented():
    # Doc-only finding: on real data S0's beta assertion is tautological
    # (beta_true = the pair's own levels OLS) and the genuine gate is the
    # finite-half-life (mean-reversion) check. The behavior must be:
    # explosive spread (phi>1) -> half_life=inf -> half_life_ok False ->
    # S0 FAILS (probe: phi=1.019 gives half_life=inf).
    rng = np.random.Generator(np.random.PCG64(5))
    e = rng.normal(0, 0.5, 252)
    s = np.zeros(252)
    for t in range(1, 252):
        s[t] = 1.019 * s[t - 1] + e[t]
    phi_hat, hl = stats.estimate_ar1(s)
    assert phi_hat > 1.0
    assert hl == math.inf
    # and the calibration gate fails on it when half_life_true is finite
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    x, y = fx["prices"][:, 0], fx["prices"][:, 1]
    res = stats.calibrate_ar1_fixture(
        x, y, beta_true=stats.ols_hedge_ratio(x, y),
        half_life_true=1e9, spread=s,
    )
    assert res.beta_ok                       # tautological beta recovery
    assert not res.half_life_ok              # genuine mean-reversion gate
    assert not res.passed
    # docstrings must state the real-data role honestly
    assert "tautolog" in stats.calibrate_ar1_fixture.__doc__.lower()
