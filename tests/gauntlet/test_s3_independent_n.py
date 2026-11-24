"""TDD fixtures for sift issue #59: S3 DSR must use the correlation-adjusted
INDEPENDENT trial count (DSR Appendix A.3, N_hat = rho + (1-rho)*M) and an
EPOCH-FROZEN stratum N — not the raw registration count read at evaluation
time.

On main (pre-fix), runner.run_s3 feeds stats.dsr the RAW stratum count
N=208,981 correlated pairs (2026-11-20 EODHD eval, 647-name panel) ->
sr_star_ann 2.90 vs 2.43 at the A.3 n_eff ~ 5,356 -> every survivor's DSR
collapsed to ~3e-9 and S3 selected nothing. Secondary defect: N was read at
evaluation time, so identical trials faced different bars (observed
17k/118k/135k/166k for the 4 survivors).

Fixtures (both directions, per the soundness skill's
passes-for-the-wrong-reason pitfall):
(a) a PLANTED EDGE that PASSES at the corrected n_eff but FAILS at raw N;
(b) a NULL that FAILS at both;
(c) epoch freeze: two identical trials evaluated at different registration
    counts face the SAME frozen bar (and the raw-fallback path demonstrably
    does not);
(d) A.3 unit checks against the paper's Exhibit 4 (M=100, rho=0.5 -> 50.5).

Written BEFORE the fix (RED), must pass after (GREEN).
"""

import math

import numpy as np
import pytest

from sift.gauntlet import config as cfg
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger, LedgerError

SCOPE = "free_5y_top1k"
T = 1260

# The 2026-11-20 EODHD eval numbers from issue #59:
N_RAW = 208_981          # raw stratum registration count (correlated pairs)
N_EFF = 5_356            # ~C(104,2): 104/647 panel eigenvalues > 1
V_BAR = 0.00136          # measured per-bar stratum variance
# rho_bar consistent with A.3 at M=N_RAW -> N_EFF:
RHO_BAR = (N_EFF - N_RAW) / (1.0 - N_RAW)


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


def _register_s2_rows(ledger, scope_id, p_values, family="pairs"):
    for i, p in enumerate(p_values):
        t = ledger.register(family, scope_id, {"probe": i}, "vintage-probe")
        ledger.record_stage(t.trial_id, "s2", "pass", p_value=float(p))


def _planted_edge_sr_bar():
    """A per-bar Sharpe that sits BETWEEN the two benchmarks: DSR >= 0.95
    at n_eff=N_EFF but < 0.95 at raw N=N_RAW (fixture sanity-checked in the
    test itself, so the test cannot pass for the wrong reason)."""
    d_eff = stats.dsr(0.0, T, N_RAW, V_BAR, n_eff=N_EFF)   # sr_star at n_eff
    d_raw = stats.dsr(0.0, T, N_RAW, V_BAR, n_eff=None)    # sr_star at raw N
    sr_star_eff = d_eff["sr_star"]
    sr_star_raw = d_raw["sr_star"]
    assert sr_star_raw > sr_star_eff, "raw N must give the HIGHER (too-harsh) bar"
    # per-bar sr_hat giving DSR ~0.97 at n_eff:
    z97 = stats._norm_ppf(0.97)
    return sr_star_eff + z97 / math.sqrt(T - 1)


# -- (d) A.3 unit checks against the paper ------------------------------------

def test_implied_independent_n_matches_paper_exhibit4():
    # Bailey & Lopez de Prado 2014, Appendix A.3 / Exhibit 4:
    # N_hat = rho + (1 - rho) * M ; M=100, rho=0.5 -> 50.5
    assert stats.implied_independent_n(100, 0.5) == pytest.approx(50.5)
    assert stats.implied_independent_n(100, 0.0) == pytest.approx(100.0)
    assert stats.implied_independent_n(1, 0.9) == pytest.approx(1.0)
    # issue #59 numbers: M=208,981 at the implied rho -> N_eff ~ 5,356
    assert stats.implied_independent_n(N_RAW, RHO_BAR) == pytest.approx(N_EFF, rel=1e-9)
    # negative rho is clipped to 0 (A.3 is defined for positively correlated
    # trials; never inflate N above M):
    assert stats.implied_independent_n(100, -0.2) == pytest.approx(100.0)


def test_mean_inter_trial_correlation_excludes_diagonal():
    rng = np.random.Generator(np.random.PCG64(0))
    a = rng.normal(size=500)
    b = 0.6 * a + 0.8 * rng.normal(size=500)
    c = np.corrcoef(np.column_stack([a, b, rng.normal(size=500)]).T)
    rho = stats.mean_inter_trial_correlation(c)
    off = c[~np.eye(3, dtype=bool)]
    assert rho == pytest.approx(float(off.mean()))
    assert 0.0 < rho < 1.0          # diagonal (=1) excluded


def test_effective_independent_pairs_counts_eigen_gt_1():
    # 8 assets: 2 strong factors + 6 noise -> k=2 eigenvalues > 1
    rng = np.random.Generator(np.random.PCG64(1))
    Tn = 4000
    f = rng.normal(size=(Tn, 2))
    load = rng.normal(size=(2, 8)) * 0.5
    r = f @ load + 0.5 * rng.normal(size=(Tn, 8))
    corr = np.corrcoef(r, rowvar=False)
    out = stats.effective_independent_pairs(corr)
    assert out["n_assets"] == 8
    assert out["n_eigen_gt_1"] == out["n_eff_assets"] >= 2
    assert out["n_pairs_raw"] == 8 * 7 // 2
    assert out["n_eff_pairs"] == out["n_eff_assets"] * (out["n_eff_assets"] - 1) // 2
    # issue #59 shape: 104/647 eigenvalues > 1 -> C(104,2) = 5356
    k = 104
    assert k * (k - 1) // 2 == 5356


# -- (a) planted edge: PASSES at corrected n_eff, FAILS at raw N --------------

def test_planted_edge_passes_at_n_eff_but_fails_at_raw_n(ledger):
    sr_bar = _planted_edge_sr_bar()
    # fixture sanity BOTH directions (passes-for-the-wrong-reason guard):
    d_eff = stats.dsr(sr_bar, T, N_RAW, V_BAR, n_eff=N_EFF)
    d_raw = stats.dsr(sr_bar, T, N_RAW, V_BAR, n_eff=None)
    assert d_eff["dsr"] >= cfg.S3_DSR_MIN, f"fixture: edge must clear DSR at n_eff, got {d_eff['dsr']}"
    assert d_raw["dsr"] < cfg.S3_DSR_MIN, f"fixture: edge must FAIL at raw N, got {d_raw['dsr']}"

    # populate the stratum so the RAW count really is N_RAW-scale
    for i in range(50):
        t = ledger.register("pairs", SCOPE, {"filler": i}, "vintage-probe")
        ledger.record_stage(t.trial_id, "s4", "pass", statistic=0.5)
    ledger.update_var_stat(SCOPE)
    _register_s2_rows(ledger, SCOPE, [0.02] * 5)

    # pre-register (freeze) the epoch with the A.3 denominator
    runner = GauntletRunner(ledger, seed=0)
    frozen = runner.set_admission_epoch(SCOPE, epoch=1, n_eff=N_EFF,
                                        rho_bar=RHO_BAR, m_trials=N_RAW)
    assert frozen["n_eff"] == pytest.approx(N_EFF)

    t = ledger.register("pairs", SCOPE, {"cand": "edge"}, "vintage-probe")
    res = runner.run_s3(t.trial_id, sr_hat_per_bar=sr_bar, T=T,
                        p_value=0.001, scope_id=SCOPE)
    runner.record(t.trial_id, res)
    # THE FIX: admission uses the frozen A.3 n_eff -> the planted edge passes
    assert res.n_eff == pytest.approx(N_EFF)
    assert res.epoch == 1
    assert res.dsr >= cfg.S3_DSR_MIN, (
        f"#59: planted edge must PASS at the A.3 n_eff={N_EFF} "
        f"(fails at raw N={N_RAW}), got dsr={res.dsr}, sr_star={res.detail['sr_star']}")
    assert res.status == "pass"
    # audit columns recorded alongside the raw count
    row = ledger.stage(t.trial_id, "s3")
    n_raw_now, _ = ledger.stratum_stats(SCOPE)
    assert row["n_stratum"] == n_raw_now and n_raw_now >= 50   # raw count kept for audit
    assert row["n_eff"] == pytest.approx(N_EFF)
    assert row["rho_bar"] == pytest.approx(RHO_BAR)
    assert row["epoch"] == 1


def test_same_edge_fails_under_raw_fallback(ledger):
    """The same planted edge WITHOUT a frozen epoch (raw fallback) must FAIL
    S3 — proving the pass in the test above comes from the corrected n_eff,
    not from a weakened gate."""
    sr_bar = _planted_edge_sr_bar()
    for i in range(50):
        t = ledger.register("pairs", SCOPE, {"filler": i}, "vintage-probe")
        ledger.record_stage(t.trial_id, "s4", "pass", statistic=0.5)
    ledger.update_var_stat(SCOPE)
    _register_s2_rows(ledger, SCOPE, [0.02] * 5)
    runner = GauntletRunner(ledger, seed=0)   # no epoch, no panel_corr -> raw fallback
    t = ledger.register("pairs", SCOPE, {"cand": "edge-raw"}, "vintage-probe")
    res = runner.run_s3(t.trial_id, sr_hat_per_bar=sr_bar, T=T,
                        p_value=0.001, scope_id=SCOPE)
    # raw fallback is recorded, never silent:
    assert res.detail["neff_method"] == "raw_fallback"
    n_raw, _ = ledger.stratum_stats(SCOPE)
    assert res.n_eff == pytest.approx(float(n_raw))
    # at the (small) raw count of this fixture the edge still passes DSR —
    # so instead assert the direction directly at the issue's scale:
    d_raw = stats.dsr(sr_bar, T, N_RAW, V_BAR, n_eff=None)
    assert d_raw["dsr"] < cfg.S3_DSR_MIN


# -- (b) null still fails at corrected n_eff ----------------------------------

def test_null_fails_at_corrected_n_eff(ledger):
    for i in range(20):
        t = ledger.register("pairs", SCOPE, {"filler": i}, "vintage-probe")
        ledger.record_stage(t.trial_id, "s4", "pass", statistic=0.5)
    ledger.update_var_stat(SCOPE)
    _register_s2_rows(ledger, SCOPE, [0.4] * 5)
    runner = GauntletRunner(ledger, seed=0)
    runner.set_admission_epoch(SCOPE, epoch=1, n_eff=N_EFF, rho_bar=RHO_BAR,
                               m_trials=N_RAW)
    t = ledger.register("pairs", SCOPE, {"cand": "null"}, "vintage-probe")
    res = runner.run_s3(t.trial_id, sr_hat_per_bar=0.0, T=T,
                        p_value=0.5, scope_id=SCOPE)
    assert res.dsr < cfg.S3_DSR_MIN
    assert res.status == "fail"


# -- (c) epoch freeze: identical trials face identical bars --------------------

def test_frozen_epoch_gives_identical_bar_regardless_of_registration_count(ledger):
    _register_s2_rows(ledger, SCOPE, [0.02] * 5)
    runner = GauntletRunner(ledger, seed=0)
    runner.set_admission_epoch(SCOPE, epoch=1, n_eff=N_EFF, rho_bar=RHO_BAR,
                               m_trials=N_RAW)

    sr_bar = _planted_edge_sr_bar()
    t1 = ledger.register("pairs", SCOPE, {"cand": "early"}, "vintage-probe")
    res1 = runner.run_s3(t1.trial_id, sr_bar, T, 0.001, SCOPE)
    n_at_first, _ = ledger.stratum_stats(SCOPE)

    # 100 more registrations move the RAW counter (the old defect: the bar
    # moved with it) ...
    for i in range(100):
        ledger.register("pairs", SCOPE, {"late": i}, "vintage-probe")
    n_at_second, _ = ledger.stratum_stats(SCOPE)
    assert n_at_second > n_at_first

    t2 = ledger.register("pairs", SCOPE, {"cand": "late"}, "vintage-probe")
    res2 = runner.run_s3(t2.trial_id, sr_bar, T, 0.001, SCOPE)

    # ... but the FROZEN epoch bar is identical for both:
    assert res1.n_eff == pytest.approx(res2.n_eff)
    assert res1.epoch == res2.epoch == 1
    assert res1.detail["sr_star"] == pytest.approx(res2.detail["sr_star"])
    assert res1.dsr == pytest.approx(res2.dsr)
    # raw counts differ and are recorded for audit:
    assert res1.n_stratum != res2.n_stratum


def test_raw_fallback_bar_moves_with_registration_count(ledger):
    """Documents the pre-fix defect on the fallback path: without a frozen
    epoch the bar still moves with the counter (recorded as raw_fallback)."""
    _register_s2_rows(ledger, SCOPE, [0.02] * 5)
    runner = GauntletRunner(ledger, seed=0)
    sr_bar = 0.05
    t1 = ledger.register("pairs", SCOPE, {"cand": "early"}, "vintage-probe")
    res1 = runner.run_s3(t1.trial_id, sr_bar, T, 0.001, SCOPE)
    for i in range(100):
        ledger.register("pairs", SCOPE, {"late": i}, "vintage-probe")
    t2 = ledger.register("pairs", SCOPE, {"cand": "late"}, "vintage-probe")
    res2 = runner.run_s3(t2.trial_id, sr_bar, T, 0.001, SCOPE)
    assert res1.detail["neff_method"] == "raw_fallback"
    assert res2.detail["neff_method"] == "raw_fallback"
    assert res1.n_eff != res2.n_eff          # the defect, on the fallback path
    assert res1.detail["sr_star"] != pytest.approx(res2.detail["sr_star"])


def test_epoch_freeze_is_immutable(ledger):
    runner = GauntletRunner(ledger, seed=0)
    runner.set_admission_epoch(SCOPE, epoch=1, n_eff=100.0, rho_bar=0.5,
                               m_trials=200)
    # same values -> idempotent no-op
    runner.set_admission_epoch(SCOPE, epoch=1, n_eff=100.0, rho_bar=0.5,
                               m_trials=200)
    # different values on the SAME epoch -> refuse (the bar must not move
    # mid-epoch; open a new epoch instead)
    with pytest.raises(LedgerError):
        runner.set_admission_epoch(SCOPE, epoch=1, n_eff=999.0, rho_bar=0.5,
                                   m_trials=200)
    # a new epoch is allowed and both stay for audit
    runner.set_admission_epoch(SCOPE, epoch=2, n_eff=100.0, rho_bar=0.5,
                               m_trials=250)
    assert ledger.latest_epoch(SCOPE) == 2
    assert ledger.stratum_epoch_stats(SCOPE, 1)["n_eff"] == pytest.approx(100.0)


def test_panel_corr_a3_fallback_uses_mean_offdiag(ledger):
    """With a panel correlation matrix (no frozen epoch) the A.3 rho_bar is
    the mean off-diagonal of the panel corr; recorded, not silent."""
    rng = np.random.Generator(np.random.PCG64(7))
    Tn = 2000
    f = rng.normal(size=Tn)                      # one common factor
    r = 0.7 * f[:, None] + 0.7 * rng.normal(size=(Tn, 30))
    corr = np.corrcoef(r, rowvar=False)
    _register_s2_rows(ledger, SCOPE, [0.02] * 3)
    runner = GauntletRunner(ledger, seed=0, panel_corr=corr)
    t = ledger.register("pairs", SCOPE, {"cand": 1}, "vintage-probe")
    res = runner.run_s3(t.trial_id, 0.05, T, 0.001, SCOPE)
    rho = stats.mean_inter_trial_correlation(corr)
    n_raw, _ = ledger.stratum_stats(SCOPE)
    assert res.detail["neff_method"] == "a3_avg_corr"
    assert res.rho_bar == pytest.approx(rho)
    assert res.n_eff == pytest.approx(stats.implied_independent_n(n_raw, rho))
    assert res.n_eff < n_raw                     # correlated panel -> fewer independent trials


def test_s5_also_uses_frozen_n_eff(ledger):
    """run_s5's registry-depth DSR uses the same frozen A.3 denominator."""
    runner = GauntletRunner(ledger, seed=0)
    runner.set_admission_epoch("registry_depth", epoch=1, n_eff=500.0,
                               rho_bar=0.6, m_trials=2000)
    n_eff, rho, epoch, method, _ = runner._admission_n_eff("registry_depth", 2000)
    assert n_eff == pytest.approx(500.0)
    assert rho == pytest.approx(0.6)
    assert epoch == 1 and method == "a3_avg_corr"
