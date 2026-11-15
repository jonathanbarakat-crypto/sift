"""Planted fixtures for gauntlet acceptance tests (deterministic, PCG64-seeded).

- F-PAIR: two synthetic series with an AR(1) cointegrating spread and known
  beta*, phi* (half-life lambda*). MUST pass the gauntlet (S0-S4).
- F-NULL: 200 independent random walks. MUST fail S2/S3 (type-I control).
"""

from __future__ import annotations

import numpy as np

F_PAIR_BETA = 2.0
F_PAIR_PHI = 0.85
F_PAIR_SIG_EPS = 0.5
F_PAIR_SIG_X = 1.5
F_PAIR_T = 1260  # 5y of daily bars — the free-tier scope length
F_PAIR_SEED = 20261115

F_NULL_N = 200
F_NULL_T = 250
F_NULL_SIG = 1.5


def make_f_pair(T: int = F_PAIR_T, seed: int = F_PAIR_SEED):
    """AR(1) cointegrated pair: x random walk, y = beta*x + s, s AR(1)(phi)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    x = 100.0 + np.cumsum(F_PAIR_SIG_X * rng.standard_normal(T))
    s = np.zeros(T)
    eps = F_PAIR_SIG_EPS * rng.standard_normal(T)
    for t in range(1, T):
        s[t] = F_PAIR_PHI * s[t - 1] + eps[t]
    y = F_PAIR_BETA * x + s
    return x.tolist(), y.tolist(), F_PAIR_BETA, F_PAIR_PHI


def make_f_null(n: int = F_NULL_N, T: int = F_NULL_T, seed0: int = 424242):
    """n independent random-walk pairs (no cointegration): must fail S2/S3."""
    pairs = []
    for i in range(n):
        rng = np.random.Generator(np.random.PCG64(seed0 + i))
        x = 100.0 + np.cumsum(F_NULL_SIG * rng.standard_normal(T))
        y = 100.0 + np.cumsum(F_NULL_SIG * rng.standard_normal(T))
        pairs.append((x.tolist(), y.tolist()))
    return pairs
