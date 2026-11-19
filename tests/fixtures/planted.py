"""Planted fixtures for acceptance tests (design doc §5).

F-PAIR: two synthetic series with an AR(1) cointegrating spread, known
beta* and half-life lambda*. Must pass the gauntlet (S0-S4 at the free scope).
F-NULL: 200 independent random walks. Must fail S2/S3 (type-I control).
F-REGIME: a pair with a real edge ONLY in years 0-7 of a 16-year window
(AR(1) spread inside, random walk outside). The 5-yr in-edge window must
show t > 2; the full 16-yr window must attenuate (|t_full| < 3.0) so S5
kills it -> on-watch-amber, never validated-green.
F-DELIST: a universe where 20% of names delist mid-window; the pipeline must
not crash and must not use post-delist prices.
All fixtures are deterministic (PCG64 seed recorded in the ledger row).
"""

from __future__ import annotations

import math

import numpy as np

FIXTURE_SEED = 7
NULL_SEED = 99
T_5Y = 1260   # free scope window length in daily bars
T_FULL = 4032  # registry_depth window: 16y of daily bars (2010-2025)
T_EDGE = 7 * 252  # F-REGIME: edge present in the first 7 years (1764 bars)
REGIME_SEED = 7
# F-REGIME parameters (tuned so the 5y in-edge window passes the screen with
# t > 2 and the full window attenuates; see tests/gauntlet/test_s5.py):
REGIME_PHI = 0.70
REGIME_SIG_IN = 6.0    # AR(1) innovation vol inside the edge (log-space, x100)
REGIME_SIG_OUT = 16.0  # random-walk innovation vol outside the edge (dilutes)


def f_pair(T: int = T_5Y, seed: int = FIXTURE_SEED, beta: float = 1.0,
           phi: float = 0.7, sig_x: float = 0.01, sig_s: float = 12.0,
           s0: float = 0.0) -> dict:
    """AR(1) cointegrated pair in LOG space: log x is a random walk with vol
    sig_x, the planted log-spread s_t = phi s_{t-1} + e_t with innovation vol
    sig_s, and y = x * exp(s). Returns prices (T,2), the levels-OLS beta of
    its own prices (the quantity S0 asserts), half-life lambda*, and the
    planted spread (in log units).

    Parameters are chosen so the planted mean-reversion survives the HONEST
    two-leg cost model (soundness #9: both legs charged at the 100 bps
    placeholder per leg, fill-bar timing, live slippage) at the free scope.
    The pair is constructed in log space because the strategy hedges on log
    returns; in level space (y = beta*x + s) the log transform leaves a
    residual of the common factor in the hedge residual and the walk-forward
    t stays below the S4 bar for any spread vol. Recalibrated 2026-11-19 from
    the level-space sig_s=3.2 version, which only passed under the old
    one-leg decision-bar cost model that understated two-leg costs ~2-3x
    (probe on main 8c1a1b0: one-leg 1000 bps vs two-leg 2948 bps on the same
    pair). No threshold was weakened — S4_T_NET_5Y=3.5 etc. are unchanged;
    the fixture's edge is simply strong enough for honest costs."""
    rng = np.random.Generator(np.random.PCG64(seed))
    lx = np.cumsum(rng.normal(0, sig_x, T))
    e = rng.normal(0, sig_s, T)
    s = np.zeros(T)
    s[0] = s0
    for t in range(1, T):
        s[t] = phi * s[t - 1] + e[t]
    x = 100.0 * np.exp(lx)
    y = x * np.exp(s / 100.0)            # s is in percent -> log units s/100
    prices = np.column_stack([x, y])
    xc, yc = x - x.mean(), y - y.mean()
    beta_true = float(np.dot(xc, yc) / np.dot(xc, xc))
    half_life = -math.log(2.0) / math.log(phi)
    return {"prices": prices, "beta_true": beta_true,
            "phi": phi, "half_life_true": half_life, "spread": s / 100.0, "seed": seed}


def f_null(T: int = T_5Y, n: int = 200, seed: int = NULL_SEED,
           sig: float = 0.01) -> np.ndarray:
    """200 independent random walks, shape (T, n): pure noise universe."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.cumprod(1 + rng.normal(0, sig, (T, n)), axis=0)


def f_null_spread_returns(T: int = T_5Y, seed: int = NULL_SEED,
                          sig: float = 0.01) -> np.ndarray:
    """Log returns of one random walk (a candidate 'spread' series under the
    null), length T."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.normal(0, sig, T)


def f_regime(T: int = T_FULL, t_edge: int = T_EDGE, seed: int = REGIME_SEED,
             phi: float = REGIME_PHI, sig_in: float = REGIME_SIG_IN,
             sig_out: float = REGIME_SIG_OUT, sig_x: float = 0.01,
             scale: float = 100.0) -> dict:
    """F-REGIME fixture: a pair with a REAL edge only in years 0-7 of a
    16-year window (design doc §5 F-REGIME: coupling 2010-2016, absent 2017+).

    Construction (log-space so prices stay strictly positive):
      log px = cumsum(N(0, sig_x))                       — common random walk
      s_t   = phi s_{t-1} + e_t·sig_in  for t < t_edge   — AR(1) spread (edge)
      s_t   = s_{t-1}      + e_t·sig_out for t >= t_edge — random walk (no edge)
      log py = log px + s / scale

    The registered free scope is the trailing 5y AS OF THE END OF THE EDGE
    (bars [t_edge-1260, t_edge)) — fully inside the edge, so the 5y window
    shows the planted mean reversion (t > 2). Re-evaluated at registry_depth
    (the full T bars) the out-of-edge random walk dilutes the edge: full-depth
    SR well below scope SR and |t_full| < 3.0, so S5 must kill it.

    S0 note: the OLS hedge ratio is estimated on LEVELS over the 5y window;
    in log-space construction log py = log px + s/scale the level hedge is
    exp(s/scale)-ish, so `beta_true` passed to S0 is the expected level hedge
    over the scope window (computed here from the fixture itself) rather than
    exactly 1.0. The AR(1) half-life assertion uses the planted spread in
    log space.

    Returns the full-window prices (T,2), the scope-window prices (1260,2),
    the planted spread, the edge boundary, and the scope-window level hedge
    `beta_true` for S0.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    lx = np.cumsum(rng.normal(0, sig_x, T))
    e = rng.normal(0, 1.0, T)
    s = np.zeros(T)
    for t in range(1, T):
        if t < t_edge:
            s[t] = phi * s[t - 1] + e[t] * sig_in
        else:
            s[t] = s[t - 1] + e[t] * sig_out
    px = 100.0 * np.exp(lx)
    py = px * np.exp(s / scale)
    prices_full = np.column_stack([px, py])
    t_edge = min(t_edge, T)
    prices_scope = prices_full[max(0, t_edge - T_5Y): t_edge]
    beta_true = float(np.dot(prices_scope[:, 0] - prices_scope[:, 0].mean(),
                             prices_scope[:, 1] - prices_scope[:, 1].mean())
                      / np.dot(prices_scope[:, 0] - prices_scope[:, 0].mean(),
                               prices_scope[:, 0] - prices_scope[:, 0].mean()))
    return {"prices_full": prices_full, "prices_scope": prices_scope,
            "spread": s, "t_edge": t_edge, "T": T, "seed": seed,
            "half_life_true": -math.log(2.0) / math.log(phi) if 0 < phi < 1 else math.inf,
            "beta_true": beta_true}


def f_delist(T: int = T_FULL, n: int = 50, delist_frac: float = 0.20,
             seed: int = 4242, sig: float = 0.01) -> dict:
    """F-DELIST fixture: a universe of `n` random walks where `delist_frac`
    of the names delist mid-window (design doc §5 F-DELIST, vendor probe #2).

    A delisted name's price series STOPS at its delist bar; the cells after
    are NaN (never a fabricated price). Returns prices (T, n) with NaN after
    each delist, the delist_idx array (first invalid bar per column; T for
    names that never delist), and the count of delisted names.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    r = rng.normal(0, sig, (T, n))
    prices = 100.0 * np.cumprod(1 + r, axis=0)
    n_delist = int(round(delist_frac * n))
    delist_idx = np.full(n, T, dtype=int)
    if n_delist > 0:
        # deterministic spread of delist points across the middle of the window
        which = rng.permutation(n)[:n_delist]
        for j in which:
            d = int(rng.integers(T // 4, 3 * T // 4))
            delist_idx[j] = d
            prices[d:, j] = np.nan
    return {"prices": prices, "delist_idx": delist_idx,
            "n_delisted": n_delist, "n": n, "T": T, "seed": seed}
