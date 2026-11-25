"""TDD regression tests for soundness #1/#16 (DSR stratum-variance units +
var_floor), written BEFORE the fix — they must FAIL on the buggy code and
PASS after.

#1 P0: runner.run_s4 stores s4.statistic = sr_ann (ANNUALIZED) but
runner.run_s3 feeds a PER-BAR sr_hat into stats.dsr with the stratum V taken
(raw) from those annualized statistics -> V is annualized-SR^2 while sr_hat is
per-bar -> sr_star comes out annualized -> DSR collapses to ~0 in a
populated stratum and a genuine edge fails S3.

#16 P2: dsr's default var_floor is 1/T; the PSR variance formula gives
(1 + sr_hat^2/2)/T. Align the floor and document the convention.

Fixtures:
(a) genuine edge PASSES S3 end-to-end through Ledger+GauntletRunner in a
    POPULATED stratum (calibrated honestly: at N=200, V_bar~0.003 the edge
    needs sr_bar >= ~0.22, annSR ~3.5, to clear sr_star~0.154 at DSR 0.95);
    a modest annSR-0.79 edge (sr_bar=0.05) SHOULD still fail — that is DSR
    working, not the bug;
(b) null still fails S3;
(c) unit-consistency: sr_star is in per-bar range when sr_hat is per-bar
    (runtime guard asserts it too);
(d) existing null-fails tests keep passing for the right reason.
"""

import math

import numpy as np
import pytest

from sift.gauntlet import config as cfg
from sift.gauntlet import stats
from sift.gauntlet.runner import GauntletRunner
from sift.registry import Ledger, sample_variance

SCOPE = "free_5y_top1k"
T = 1260  # free-scope window (bars)


@pytest.fixture()
def ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.db")
    yield led
    led.close()


def _populate_stratum(ledger, sr_anns, scope_id=SCOPE):
    """Record one trial per annualized SR as an s4 statistic — exactly what
    GauntletRunner.run_s4 stores (statistic=sr_ann, annualized units)."""
    for i, sr_ann in enumerate(sr_anns):
        t = ledger.register("pairs", scope_id, {"i": i}, "vintage-probe")
        ledger.record_stage(t.trial_id, "s4", "pass", statistic=float(sr_ann))
    ledger.update_var_stat(scope_id)
    return ledger.stratum_stats(scope_id)


# -- (c) unit consistency at the stats layer --------------------------------

def test_dsr_sr_star_in_per_bar_units_when_sr_hat_per_bar():
    """#1: with a per-bar sr_hat and a per-bar V, sr_star must be per-bar
    scale (same units as sr_hat), not annualized (~sqrt(252)x bigger)."""
    v_bar = 0.003  # per-bar variance ~ annualized 0.756/252
    d = stats.dsr(sr_hat=0.05, T=T, n_stratum=200, v_stratum=v_bar,
                  skew=0.0, kurt=3.0)
    # E[max] at N=200 is ~2.76; per-bar sr_star ~ sqrt(0.003)*2.76 ~ 0.15
    assert d["sr_star"] == pytest.approx(math.sqrt(v_bar) * 2.7635, rel=0.02)
    assert d["sr_star"] < 1.0          # per-bar scale, NOT annualized (>= ~2.4)
    assert d["sr_star"] / 0.05 < 10    # same-units comparable with sr_hat


def test_dsr_var_floor_is_psr_variance_formula():
    """#16: default var_floor must be the normal-theory Var(SR_hat) =
    (1 + sr_hat^2/2)/T, not the plain 1/T (PSR's denominator convention)."""
    sr_hat, t = 0.1, 252
    d = stats.dsr(sr_hat=sr_hat, T=t, n_stratum=10, v_stratum=0.0,
                  skew=0.0, kurt=3.0)
    assert d["v_stratum"] == pytest.approx((1.0 + sr_hat ** 2 / 2.0) / t)
    # explicit override still honored (backwards compat)
    d2 = stats.dsr(sr_hat=sr_hat, T=t, n_stratum=10, v_stratum=0.0,
                   skew=0.0, kurt=3.0, var_floor=1.0 / t)
    assert d2["v_stratum"] == pytest.approx(1.0 / t)


def test_dsr_floor_monotone_in_sr_hat():
    """#16: the floor grows with |sr_hat| (PSR variance), so a stronger
    candidate gets a slightly HIGHER floor — check the direction."""
    t = 252
    d_small = stats.dsr(sr_hat=0.0, T=t, n_stratum=10, v_stratum=0.0)
    d_big = stats.dsr(sr_hat=0.5, T=t, n_stratum=10, v_stratum=0.0)
    assert d_big["v_stratum"] > d_small["v_stratum"]


# -- ledger stratum variance units -------------------------------------------

def test_update_var_stat_stores_per_bar_variance(ledger):
    """#1: s4.statistic is stored ANNUALIZED (runner.run_s4), so
    update_var_stat must divide by BARS_PER_YEAR before sample_variance —
    V must come out in PER-BAR SR^2 units, matching the candidate's per-bar
    sr_hat (same convention as decks/service.py::_deck_stratum_variance)."""
    rng = np.random.Generator(np.random.PCG64(42))
    sr_anns = rng.normal(0.5, 1.0, 200)          # annualized SRs
    n, v = _populate_stratum(ledger, sr_anns)
    assert n == 200
    v_bar_expected = sample_variance([float(s) / math.sqrt(cfg.BARS_PER_YEAR)
                                      for s in sr_anns])
    assert v == pytest.approx(v_bar_expected, rel=1e-9)
    # and NOT the annualized variance (the bug):
    assert v == pytest.approx(sample_variance([float(s) for s in sr_anns])
                              / cfg.BARS_PER_YEAR, rel=1e-9)
    assert v < sample_variance([float(s) for s in sr_anns])  # per-bar << annualized


# -- (a) genuine edge PASSES S3 in a POPULATED stratum (end-to-end) ----------

def test_genuine_edge_passes_s3_in_populated_stratum(ledger):
    """#1 end-to-end: Ledger + GauntletRunner, stratum populated with 200
    annualized SRs ~ N(0.5,1) (PCG64(42)). A GENUINE strong edge
    (sr_bar = 0.25, annSR ~ 3.96) must PASS S3 (DSR >= 0.95).

    Calibration (honest): at N=200, V_bar ~ 0.0031 -> sr_star ~ 0.154, so
    DSR 0.95 at T=1260 needs sr_bar >= ~0.22 (annSR ~ 3.5)."""
    rng = np.random.Generator(np.random.PCG64(42))
    sr_anns = rng.normal(0.5, 1.0, 200)
    n, v = _populate_stratum(ledger, sr_anns)
    assert n == 200 and v > 0
    runner = GauntletRunner(ledger, seed=42)
    sr_bar = 0.25                      # per-bar edge, annSR ~ 3.96
    res = runner.run_s3("cand-strong", sr_bar, T, p_value=0.01, scope_id=SCOPE)
    assert res.detail["sr_star"] < 1.0          # per-bar units (guard)
    assert res.dsr >= cfg.S3_DSR_MIN, (
        f"genuine edge failed S3: dsr={res.dsr}, sr_star={res.detail['sr_star']}, "
        f"v={res.v_stratum}")
    assert res.status == "pass"


def test_modest_edge_still_fails_s3_in_populated_stratum(ledger):
    """#1 honest calibration: a MODEST edge (sr_bar=0.05, annSR ~ 0.79)
    SHOULD still fail S3 in a populated stratum — that is DSR working as
    designed, not the bug. (Before the fix it failed for the WRONG reason:
    sr_star=2.44 annualized vs per-bar sr_hat.)"""
    rng = np.random.Generator(np.random.PCG64(42))
    sr_anns = rng.normal(0.5, 1.0, 200)
    _populate_stratum(ledger, sr_anns)
    runner = GauntletRunner(ledger, seed=42)
    res = runner.run_s3("cand-modest", 0.05, T, p_value=0.01, scope_id=SCOPE)
    assert res.detail["sr_star"] == pytest.approx(0.154, abs=0.02)  # per-bar
    assert res.dsr < cfg.S3_DSR_MIN
    assert res.status == "fail"


# -- (b) null still fails S3 --------------------------------------------------

def test_null_still_fails_s3(ledger):
    """(b): under the null (sr_bar ~ 0) DSR must stay below 0.95 in a
    populated stratum."""
    rng = np.random.Generator(np.random.PCG64(42))
    sr_anns = rng.normal(0.5, 1.0, 200)
    _populate_stratum(ledger, sr_anns)
    runner = GauntletRunner(ledger, seed=42)
    res = runner.run_s3("cand-null", 0.0, T, p_value=0.5, scope_id=SCOPE)
    assert res.dsr < cfg.S3_DSR_MIN
    assert res.status == "fail"


# -- (c) mint-time runtime guard: sr_star and sr_hat share units -------------

def test_run_s3_guard_rejects_unit_mismatched_sr_star(ledger):
    """The mint-time guard must refuse to pass S3/S5 when sr_star is not in
    the same (per-bar) units as sr_hat — e.g. an annualized sr_star (~2.4)
    against a per-bar sr_hat (0.05) must raise, not silently pass/fail."""
    rng = np.random.Generator(np.random.PCG64(42))
    sr_anns = rng.normal(0.5, 1.0, 200)
    _populate_stratum(ledger, sr_anns)
    runner = GauntletRunner(ledger, seed=42)
    res = runner.run_s3("cand-guard", 0.05, T, p_value=0.01, scope_id=SCOPE)
    # after the fix sr_star is per-bar; the guard asserts this at mint time:
    stats.assert_sr_units_consistent(res.detail["sr_star"], res.detail["sr_hat"],
                                     bars_per_year=cfg.BARS_PER_YEAR)
    # and a deliberately annualized sr_star must be caught by the guard:
    with pytest.raises(AssertionError):
        stats.assert_sr_units_consistent(
            res.detail["sr_star"] * math.sqrt(cfg.BARS_PER_YEAR),
            res.detail["sr_hat"], bars_per_year=cfg.BARS_PER_YEAR)
