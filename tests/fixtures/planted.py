"""Planted fixtures for acceptance tests (design doc §5).

F-PAIR: two synthetic series with an AR(1) cointegrating spread, known
beta* and half-life lambda*. Must pass the gauntlet (S0-S4 at the free scope).
F-NULL: 200 independent random walks. Must fail S2/S3 (type-I control).
All fixtures are deterministic (PCG64 seed recorded in the ledger row).
"""

from __future__ import annotations

import math

import numpy as np

FIXTURE_SEED = 7
NULL_SEED = 99
T_5Y = 1260  # free scope window length in daily bars


def f_pair(T: int = T_5Y, seed: int = FIXTURE_SEED, beta: float = 1.0,
           phi: float = 0.7, sig_x: float = 0.10, sig_s: float = 3.2,
           s0: float = 0.0) -> dict:
    """AR(1) cointegrated pair: x random walk, y = beta*x + s with
    s_t = phi s_{t-1} + e_t. Returns prices (T,2), beta*, half-life lambda*,
    and the planted spread.

    Parameters are chosen so the planted mean-reversion is strong enough to
    survive the cost model (100 bps half-spread placeholder) at the free
    scope: phi=0.7 (half-life ~1.9 bars), spread innovation vol 3.2 vs
    common-factor vol 0.10, so the spread dominates the pair's returns."""
    rng = np.random.Generator(np.random.PCG64(seed))
    x = 100.0 + np.cumsum(rng.normal(0, sig_x, T))
    s = np.zeros(T)
    s[0] = s0
    e = rng.normal(0, sig_s, T)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + e[t]
    y = beta * x + s
    half_life = -math.log(2.0) / math.log(phi)
    return {"prices": np.column_stack([x, y]), "beta_true": beta,
            "phi": phi, "half_life_true": half_life, "spread": s, "seed": seed}


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
