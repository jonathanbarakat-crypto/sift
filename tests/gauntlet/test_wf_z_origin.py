"""Regression fixtures for soundness #4: walk_forward_returns z-score origin
reset at block boundaries (hermes-tasks #181 cross-check, finding #4 P0).

Defect: the traded block's spread was cumsum(rs) restarting at 0 each block,
while mu/sd came from the formation window's cumsum — different origins, so z
at the first traded bar of each block was (0 - mu)/sd purely from the reset.

Fixtures (TDD, written before the fix):
  (a) z-continuity at block boundaries — no reset jump;
  (b) trending pair: no RESET-DRIVEN boundary entries vs the continuous-
      spread reference;
  (c) existing shape test kept (test_walk_forward_runs);
  (d) cointegrated planted pair: walk-forward behaviour unchanged within
      tolerance.
"""

import numpy as np
import pytest

from sift.gauntlet import stats

FORMATION, STEP = 60, 120
OPEN_Z, CLOSE_Z = 2.0, 0.5


def trending_pair(seed=7, n_blocks=6, formation=FORMATION, step=STEP):
    """Seeded trending (non-cointegrated) pair: two drifting random walks."""
    T = formation + n_blocks * step
    rng = np.random.Generator(np.random.PCG64(seed))
    rx = rng.normal(0.0008, 0.01, T)
    ry = rng.normal(0.0006, 0.01, T)
    x = 100.0 * np.exp(np.cumsum(rx))
    y = 100.0 * np.exp(np.cumsum(ry))
    return np.column_stack([x, y])


def block_layout(T, formation=FORMATION, step=STEP):
    """[(block_start, block_end), ...] as produced by walk_forward_returns."""
    blocks = []
    t = formation
    while t < T:
        end = min(t + step, T)
        blocks.append((t, end))
        t = end
    return blocks


def _block_params(prices, t, formation=FORMATION):
    """h, mu, sd, train_spread for the block starting at t (bars <= t only)."""
    w = slice(max(0, t - formation), t)
    x, y = prices[w, 0], prices[w, 1]
    rx = np.zeros(len(x)); rx[1:] = np.diff(np.log(x))
    ry = np.zeros(len(y)); ry[1:] = np.diff(np.log(y))
    vx = rx.var(ddof=1)
    h = float(np.cov(rx, ry, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    train_spread = np.cumsum(ry - h * rx)
    mu = float(train_spread.mean()) if len(train_spread) else 0.0
    sd = float(train_spread.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return h, mu, sd, train_spread


def wf_anchored_reference(prices, formation=FORMATION, step=STEP,
                          open_z=OPEN_Z, close_z=CLOSE_Z):
    """Anchored reference implementation of walk_forward_returns (fix on):
    spread = train_spread[-1] + cumsum(rs). Exposes z and positions."""
    prices = np.asarray(prices, float)
    T = prices.shape[0]
    net = np.zeros(T)
    z = np.full(T, np.nan)
    pos_all = np.zeros(T)
    for (t, end) in block_layout(T, formation, step):
        h, mu, sd, train_spread = _block_params(prices, t, formation)
        block = prices[t:end]
        bx = np.zeros(len(block)); bx[1:] = np.diff(np.log(block[:, 0]))
        by = np.zeros(len(block)); by[1:] = np.diff(np.log(block[:, 1]))
        rs = by - h * bx
        spread = float(train_spread[-1]) + np.cumsum(rs)
        pos = np.zeros(len(block))
        for k in range(1, len(block)):
            zk = (spread[k] - mu) / sd
            z[t + k] = zk
            if pos[k - 1] == 0:
                if zk > open_z:
                    pos[k] = -1.0
                elif zk < -open_z:
                    pos[k] = 1.0
            else:
                pos[k] = 0.0 if abs(zk) < close_z else pos[k - 1]
        gross = np.zeros(len(block))
        gross[1:] = pos[:-1] * rs[1:]
        net[t:end] = stats.apply_costs(gross, pos, stats.CostModel())
        pos_all[t:end] = pos
    return {"net": net, "z": z, "positions": pos_all}


def wf_reset_positions(prices, formation=FORMATION, step=STEP,
                       open_z=OPEN_Z, close_z=CLOSE_Z):
    """PRE-FIX behaviour (spread resets to cumsum(rs) from 0 each block),
    kept here only so the fixtures can prove they detect the defect."""
    prices = np.asarray(prices, float)
    T = prices.shape[0]
    pos_all = np.zeros(T)
    for (t, end) in block_layout(T, formation, step):
        h, mu, sd, _ = _block_params(prices, t, formation)
        block = prices[t:end]
        bx = np.zeros(len(block)); bx[1:] = np.diff(np.log(block[:, 0]))
        by = np.zeros(len(block)); by[1:] = np.diff(np.log(block[:, 1]))
        rs = by - h * bx
        spread = np.cumsum(rs)  # the bug
        pos = np.zeros(len(block))
        for k in range(1, len(block)):
            zk = (spread[k] - mu) / sd
            if pos[k - 1] == 0:
                if zk > open_z:
                    pos[k] = -1.0
                elif zk < -open_z:
                    pos[k] = 1.0
            else:
                pos[k] = 0.0 if abs(zk) < close_z else pos[k - 1]
        pos_all[t:end] = pos
    return pos_all


def continuous_reference(prices, formation=FORMATION, step=STEP,
                         open_z=OPEN_Z, close_z=CLOSE_Z):
    """Continuous-spread reference: ONE cumsum of the hedge-adjusted spread
    return over the whole sample (per-block hedge from the formation window),
    z scored against the same per-block mu/sd — the no-reset reference."""
    prices = np.asarray(prices, float)
    T = prices.shape[0]
    z = np.full(T, np.nan)
    pos_all = np.zeros(T)
    for (t, end) in block_layout(T, formation, step):
        h, mu, sd, _ = _block_params(prices, t, formation)
        xa, ya = prices[:end, 0], prices[:end, 1]
        rxa = np.zeros(end); rxa[1:] = np.diff(np.log(xa))
        rya = np.zeros(end); rya[1:] = np.diff(np.log(ya))
        spread = np.cumsum(rya - h * rxa)
        for k in range(t + 1, end):
            zk = (spread[k] - mu) / sd
            z[k] = zk
            if pos_all[k - 1] == 0:
                if zk > open_z:
                    pos_all[k] = -1.0
                elif zk < -open_z:
                    pos_all[k] = 1.0
            else:
                pos_all[k] = 0.0 if abs(zk) < close_z else pos_all[k - 1]
    return {"z": z, "positions": pos_all}


def first_boundary_entries(positions, T, formation=FORMATION, step=STEP,
                           max_k=3):
    """Blocks whose FIRST entry (|z|>open_z with pos==0) happens within the
    first max_k bars of the block — i.e. boundary-driven entries."""
    out = []
    for (t, end) in block_layout(T, formation, step):
        for k in range(1, min(max_k, end - t) + 1):
            if positions[t + k] != 0 and positions[t + k - 1] == 0:
                out.append((t, k))
                break
    return out


# (a) z-continuity at block boundaries --------------------------------------

def test_wf_z_not_reset_jump_at_block_boundaries():
    """z at the first traded bar of each block must NOT equal the reset value
    (0 - mu)/sd; it must equal the anchored continuation
    (train_spread[-1] + rs[1] - mu)/sd."""
    prices = trending_pair()
    T = prices.shape[0]
    ref = wf_anchored_reference(prices)
    for (t, end) in block_layout(T):
        h, mu, sd, train_spread = _block_params(prices, t)
        block = prices[t:end]
        bx = np.zeros(len(block)); bx[1:] = np.diff(np.log(block[:, 0]))
        by = np.zeros(len(block)); by[1:] = np.diff(np.log(block[:, 1]))
        rs = by - h * bx
        z_reset = (0.0 - mu) / sd                       # the buggy value
        z_anchored = (float(train_spread[-1]) + rs[1] - mu) / sd
        z_ref = float(ref["z"][t + 1])
        assert np.isfinite(z_ref)
        assert z_ref == pytest.approx(z_anchored, abs=1e-12)
        # unless the anchor happens to coincide with 0, z must differ from reset
        if abs(float(train_spread[-1]) + rs[1]) > 1e-9:
            assert z_ref != pytest.approx(z_reset, abs=1e-9)


def test_wf_shipped_matches_anchored_reference():
    """The shipped walk_forward_returns net must reproduce the anchored
    reference exactly (same loop, fix on)."""
    prices = trending_pair()
    shipped = stats.walk_forward_returns(prices, formation=FORMATION, step=STEP)
    ref = wf_anchored_reference(prices)
    np.testing.assert_allclose(shipped["net"], ref["net"], atol=1e-12)


# (b) no reset-driven boundary entries on a trending pair -------------------

def test_wf_trending_pair_no_reset_driven_boundary_entries():
    """On the seeded trending pair, the FIXED implementation must not produce
    MORE boundary entries (first entry within the first 3 bars of a block)
    than the continuous-spread reference, and the PRE-FIX reset variant must
    produce more than the fixed one — proving the fixture detects the reset."""
    prices = trending_pair()
    T = prices.shape[0]
    fixed_pos = wf_anchored_reference(prices)["positions"]
    ref_pos = continuous_reference(prices)["positions"]
    reset_pos = wf_reset_positions(prices)
    n_fixed = len(first_boundary_entries(fixed_pos, T))
    n_ref = len(first_boundary_entries(ref_pos, T))
    n_reset = len(first_boundary_entries(reset_pos, T))
    assert n_fixed <= n_ref
    assert n_reset > n_fixed


# (c) existing shape test is kept in test_stats.py::test_walk_forward_runs ---

def test_wf_shape_and_finiteness_kept():
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    out = stats.walk_forward_returns(fx["prices"], formation=60, step=120)
    assert out["net"].shape[0] == fx["prices"].shape[0]
    assert np.isfinite(out["net"]).all()
    assert out["T"] == fx["prices"].shape[0]


# (d) cointegrated planted pair: behaviour unchanged within tolerance --------

def _wf_prefix(prices, formation=FORMATION, step=STEP,
               open_z=OPEN_Z, close_z=CLOSE_Z):
    """Pre-fix walk_forward_returns (spread resets to 0 each block)."""
    prices = np.asarray(prices, float)
    T = prices.shape[0]
    net = np.zeros(T)
    t = formation
    while t < T:
        end = min(t + step, T)
        h, mu, sd, _ = _block_params(prices, t, formation)
        block = prices[t:end]
        bx = np.zeros(len(block)); bx[1:] = np.diff(np.log(block[:, 0]))
        by = np.zeros(len(block)); by[1:] = np.diff(np.log(block[:, 1]))
        rs = by - h * bx
        spread = np.cumsum(rs)
        pos = np.zeros(len(block))
        for k in range(1, len(block)):
            z = (spread[k] - mu) / sd
            if pos[k - 1] == 0:
                if z > open_z:
                    pos[k] = -1.0
                elif z < -open_z:
                    pos[k] = 1.0
            else:
                pos[k] = 0.0 if abs(z) < close_z else pos[k - 1]
        gross = np.zeros(len(block))
        gross[1:] = pos[:-1] * rs[1:]
        net[t:end] = stats.apply_costs(gross, pos, stats.CostModel())
        t = end
    return {"net": net, "T": T}


def test_wf_cointegrated_pair_unchanged_within_tolerance():
    """F-PAIR walk-forward must stay close to the pre-fix values (the fix
    removes a spurious boundary discontinuity; on a genuinely cointegrated
    pair the effect is small): SR and net sum within tolerance, and the pair
    must still trade — the fix must not silently kill the strategy."""
    from tests.fixtures.planted import f_pair
    fx = f_pair()
    prices = fx["prices"]
    new = stats.walk_forward_returns(prices, formation=FORMATION, step=STEP)
    old = _wf_prefix(prices)
    sr_new = stats.sharpe_ratio(new["net"])
    sr_old = stats.sharpe_ratio(old["net"])
    assert sr_new == pytest.approx(sr_old, rel=0.25)
    assert new["net"].sum() == pytest.approx(old["net"].sum(), rel=0.5)
    # still trades: nonzero net bars in the OOS region
    assert np.any(np.abs(new["net"][FORMATION:]) > 0)
