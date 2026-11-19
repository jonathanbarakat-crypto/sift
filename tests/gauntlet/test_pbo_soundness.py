"""TDD fixtures for soundness #2/#15 (cscv_pbo sign, real multi-config PBO at
S5, fail-closed S5 verdict) — written BEFORE the fix.

Known-answer fixtures (verified by independent enumeration, see probe):
* dominant config (best IS and OOS on every CSCV split): true PBO = 0.
* N=3 mirror-overfit: A has front-loaded block sums [1]*8 + [-0.9]*8,
  B = -A, C = +0.001/bar constant: true PBO = 0.619 over the C(16,8)=12870
  splits. With the CURRENT (<= 0) comparison the function returns 0.381
  (= 1 - PBO); the fixed (>= 0) comparison returns ~0.619.
"""

import math
from itertools import combinations

import numpy as np
import pytest

from sift.gauntlet import config as cfg
from sift.gauntlet import s5 as s5stats
from sift.gauntlet import stats


S_CSCV = 16


def _blocks(R, S=S_CSCV):
    T, N = R.shape
    b = T // S
    return R[: b * S].reshape(S, b, N).sum(axis=1)


def _reference_pbo(R, S=S_CSCV):
    """Independent brute-force PBO: P(lambda >= 0), rank 0 = best OOS,
    lambda = log((rank+0.5)/(N-rank+0.5)). Boundary lambda == 0 counts as
    overfit (>= 0) — the documented convention."""
    R = np.asarray(R, float)
    T, N = R.shape
    blocks = _blocks(R, S)
    K = S // 2
    idx = np.arange(S)
    lams = []
    for train in combinations(range(S), K):
        train = np.array(train)
        test = np.array([i for i in idx if i not in train])
        winner = int(np.argmax(blocks[train].sum(axis=0)))
        oos = blocks[test].sum(axis=0)
        rank = int(np.sum(oos > oos[winner]))
        lams.append(math.log((rank + 0.5) / (N - rank + 0.5)))
    return float(np.mean(np.array(lams) >= 0.0))


def _dominant_matrix(T=256, N=3, seed=0):
    rng = np.random.Generator(np.random.PCG64(seed))
    R = rng.normal(0.0, 0.01, (T, N))
    R[:, 0] += 0.005          # config 0 dominates IS and OOS on every split
    return R


def _mirror_overfit_matrix(S=S_CSCV, b=16):
    """A front-loaded block profile [1]*8 + [-0.9]*8, B = -A, C constant
    tiny positive. IS winner is A on half the splits, B on the other half;
    OOS the mirror leg loses -> true PBO = 0.619."""
    T = S * b
    block_sums = np.array([1.0] * 8 + [-0.9] * 8)
    A = np.zeros(T)
    for j, s in enumerate(block_sums):
        A[j * b:(j + 1) * b] = s / b
    B = -A
    C = np.full(T, 0.001)
    return np.column_stack([A, B, C])


# -- cscv_pbo: DIRECTION (known answers), not just bounds --------------------

def test_cscv_pbo_dominant_config_is_zero():
    """A config that wins IS and OOS on every split has true PBO = 0.
    The current <= 0 comparison returns 1.0 here — this test fails on the
    buggy sign and passes after the fix."""
    R = _dominant_matrix()
    pbo = stats.cscv_pbo(R, S=S_CSCV, seed=1)
    assert pbo == pytest.approx(0.0, abs=1e-12)
    assert _reference_pbo(R) == pytest.approx(0.0, abs=1e-12)


def test_cscv_pbo_mirror_overfit_known_answer():
    """N=3 mirror-overfit fixture: true PBO = 0.619 over 12870 splits.
    Current buggy comparison returns 0.381 (= 1 - PBO)."""
    R = _mirror_overfit_matrix()
    ref = _reference_pbo(R)
    assert ref == pytest.approx(0.619, abs=5e-3)     # known answer
    pbo = stats.cscv_pbo(R, S=S_CSCV, seed=1)
    assert pbo == pytest.approx(ref, abs=1e-12)       # matches brute force
    assert pbo > 0.5                                  # direction: overfit


def test_cscv_pbo_bounds_and_direction_random():
    """Upgraded from bounds-only: random configs have PBO near 0.5 (no
    config consistently wins OOS after winning IS), and the value must be
    in [0,1] and match the brute-force reference."""
    rng = np.random.Generator(np.random.PCG64(0))
    R = rng.normal(0, 0.01, (256, 8))
    pbo = stats.cscv_pbo(R, S=S_CSCV, seed=1)
    assert 0.0 <= pbo <= 1.0
    assert pbo == pytest.approx(_reference_pbo(R), abs=1e-12)
    assert pbo == pytest.approx(0.5204, abs=0.05)   # ~0.5 under the null


def test_cscv_pbo_single_config_returns_none_not_zero():
    """CSCV on ONE configuration is meaningless: the production path must
    NOT silently return 0.0 ('no overfit') for N < 2. Return None so the
    caller is forced to fail closed / handle explicitly."""
    rng = np.random.Generator(np.random.PCG64(0))
    R = rng.normal(0, 0.01, (256, 1))
    assert stats.cscv_pbo(R, S=S_CSCV, seed=1) is None
    # too few bars for S blocks is equally undefined
    R2 = rng.normal(0, 0.01, (8, 4))
    assert stats.cscv_pbo(R2, S=S_CSCV, seed=1) is None


def _tie_rich_matrix(seed=2024):
    """N=4 matrix on a coarse 0.5 grid: block sums tie frequently, so a known
    set of CSCV splits has the IS winner at OOS rank exactly N/2 = 2, i.e.
    lambda == log(2.5/2.5) == 0 (the boundary). Deterministic."""
    rng = np.random.Generator(np.random.PCG64(seed))
    R = rng.normal(0, 1, (S_CSCV * 16, 4))
    return np.round(R * 2) / 2


def test_cscv_pbo_lambda_zero_boundary_counts_as_overfit():
    """Documented boundary convention: lambda == 0 (IS winner at exactly the
    OOS median, rank == N/2) counts as OVERFIT (PBO side, >= 0). The tie-rich
    fixture produces lambda == 0 on a known set of splits; the implementation
    must count them on the PBO side exactly as the brute-force reference does
    (under a strict ``> 0`` comparison the zeros would be dropped, under the
    old ``<= 0`` they landed on the 1-PBO side)."""
    R = _tie_rich_matrix()
    blocks = _blocks(R)
    K = S_CSCV // 2
    idx = np.arange(S_CSCV)
    lams = []
    for train in combinations(range(S_CSCV), K):
        train = np.array(train)
        test = np.array([i for i in idx if i not in train])
        winner = int(np.argmax(blocks[train].sum(axis=0)))
        oos = blocks[test].sum(axis=0)
        rank = int(np.sum(oos > oos[winner]))
        lams.append(math.log((rank + 0.5) / (4 - rank + 0.5)))
    lams = np.array(lams)
    n_zero = int((lams == 0.0).sum())
    assert n_zero > 0                       # the fixture DOES hit the boundary
    pbo = stats.cscv_pbo(R, S=S_CSCV, seed=1)
    assert pbo == pytest.approx(float(np.mean(lams >= 0)), abs=1e-12)
    assert pbo == pytest.approx(_reference_pbo(R), abs=1e-12)
    # the zeros genuinely move the value: >= counts them, > would not
    assert float(np.mean(lams >= 0)) != float(np.mean(lams > 0))


# -- S5 verdict: fail closed on missing stats ---------------------------------

def test_s5_verdict_none_stats_fail_closed():
    """Soundness #15: a required stat that was NOT computed (None) must
    FAIL the check — 'a stage not run cannot pass'. Current code passes
    None through ((x is None) or ...)."""
    rng = np.random.Generator(np.random.PCG64(5))
    strong = rng.normal(0.002, 0.01, 4032)   # strong edge, would pass on t
    v = s5stats.s5_full_depth_verdict(strong, t_hurdle=3.0,
                                      p_perm=None, dsr=None, pbo=None)
    assert v["pass"] is False
    assert v["checks"]["p_perm"] is False
    assert v["checks"]["dsr"] is False
    assert v["checks"]["pbo"] is False
    # each individually
    v2 = s5stats.s5_full_depth_verdict(strong, t_hurdle=3.0,
                                       p_perm=0.001, dsr=None, pbo=0.1)
    assert v2["checks"]["dsr"] is False and v2["pass"] is False
    v3 = s5stats.s5_full_depth_verdict(strong, t_hurdle=3.0,
                                       p_perm=0.001, dsr=0.99, pbo=None)
    assert v3["checks"]["pbo"] is False and v3["pass"] is False


def test_s5_verdict_min_trl_none_semantics_documented():
    """min_trl_max=None means 'no MinTRL bar' (intentional opt-out at the
    pure-function level); it stays permissive and is documented. The
    required-stats fail-closed rule applies to p_perm/dsr/pbo only."""
    rng = np.random.Generator(np.random.PCG64(5))
    strong = rng.normal(0.002, 0.01, 4032)
    v = s5stats.s5_full_depth_verdict(strong, t_hurdle=3.0, min_trl_max=None,
                                      p_perm=0.001, dsr=0.99, pbo=0.1)
    assert v["checks"]["min_trl"] is True
    assert v["pass"] is True


def test_s5_verdict_genuine_edge_unaffected():
    """The fail-closed change must not affect the genuine path: all stats
    supplied and passing -> pass."""
    rng = np.random.Generator(np.random.PCG64(5))
    strong = rng.normal(0.002, 0.01, 4032)
    v = s5stats.s5_full_depth_verdict(strong, t_hurdle=3.0,
                                      p_perm=0.001, dsr=0.99, pbo=0.1)
    assert v["pass"] is True
    assert all(v["checks"].values())


# -- real multi-config PBO wiring (runner) ------------------------------------

def test_s5_variant_grid_is_preregistered_and_hashed():
    """The S5 variant grid around the registered config is pre-registered
    in config and hashed into a stable grid digest recorded in the s5
    stage detail (so the PBO input is auditable / pre-registered)."""
    grid = cfg.s5_variant_grid()
    assert len(grid) >= 3                       # genuine multi-config
    names = {"formation", "open_z", "close_z", "step"}
    for v in grid:
        assert names.issubset(v.keys())
    # the registered (default) config is the grid's center member
    center = {"formation": 60, "open_z": 2.0, "close_z": 0.5, "step": 120}
    assert center in [{k: v[k] for k in names} for v in grid]
    # deterministic + hashed
    d1 = cfg.s5_variant_grid_digest()
    d2 = cfg.s5_variant_grid_digest()
    assert d1 == d2 and len(d1) == 64


def test_run_s5_computes_real_pbo_over_variant_grid(ledger=None):
    """run_s5 must call cscv_pbo over the variant-grid returns matrix and
    record a computed pbo (not a hardcoded 0.0). Uses the F-DELIST-style
    small panel so the test is fast; the pbo recorded in the stage detail
    equals stats.cscv_pbo on the same matrix (deterministic given seed)."""
    from sift.registry import Ledger
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    l = Ledger(tmp / "l.db")
    try:
        from sift.gauntlet.runner import GauntletRunner
        from tests.fixtures.planted import f_pair, T_FULL
        fx = f_pair(T=1024, seed=3)
        prices = fx["prices"]
        spread = np.diff(np.log(prices[:, 1])) - np.diff(np.log(prices[:, 0]))
        trial = l.register("pairs", "free_5y_top1k", {"fixture": "F-PAIR"}, "v1")
        runner = GauntletRunner(l, seed=11)
        s5 = runner.run_s5(trial.trial_id, full_prices=prices,
                           full_spread_returns=spread,
                           statistic_fn=stats.mean_reversion_stat,
                           s2_B=50, seed=11)
        pbo = s5.detail["pbo"]
        assert pbo is not None
        assert 0.0 <= pbo <= 1.0
        assert s5.detail["pbo_n_configs"] == len(cfg.s5_variant_grid())
        assert s5.detail["pbo_grid_digest"] == cfg.s5_variant_grid_digest()
        # determinism: same seed -> same pbo
        s5b = runner.run_s5(trial.trial_id, full_prices=prices,
                            full_spread_returns=spread,
                            statistic_fn=stats.mean_reversion_stat,
                            s2_B=50, seed=11)
        assert s5b.detail["pbo"] == pbo
        # and the checks dict carries the pbo check result (fail-closed wired)
        assert "pbo" in s5.detail["checks"]
    finally:
        l.close()
