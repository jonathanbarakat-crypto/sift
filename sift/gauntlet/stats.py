"""Pure statistics for the SIFT gauntlet (stages S0-S4).

All functions are pure: no I/O, no global state, no ledger access. The stage
runner (`sift.gauntlet.runner`) wires these into trials and records results.

Grounding (see FL research/design/2026-11-15-sift-gauntlet-s5-design.md and
sift-research scope v3 SS6-SS8):

- S1 screen: |t| >= 2.0 (pre-registered discovery screen).
- S2: block-permutation null, block length b = ceil(T^(1/3)) (min 5),
  B = 1000 replicates, p = (1 + #{T_b >= T_obs}) / (B + 1), PCG64-seeded.
- S3: DSR = PSR(SR_hat; SR*, T, g3, g4) with SR* = E[max_N] the expected
  maximum Sharpe under the null over N trials with per-bar variance Vbar
  (Bailey & Lopez de Prado 2014, JPM 40(5):94-107), plus BH-FDR q-values
  (Benjamini & Hochberg 1995). PSR uses skew/kurtosis of returns; MinTRL
  (Bailey & Lopez de Prado 2012, J. Risk 15(2):3-44).
- S4: cost-aware walk-forward net Sharpe, |t_net| >= tau_cal (scope-matched;
  tau_cal ~= 3.5 pre-registered for the 5y free scope), MinTRL <= T_scope.

Units: Sharpe statistics are per-bar unless annualized explicitly
(annualization factor sqrt(252) for daily bars).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

__all__ = [
    "EULER_MASCHERONI",
    "ANNUALIZATION_DAILY",
    "block_length",
    "permutation_pvalue",
    "block_permutation_stats",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "min_trl",
    "benjamini_hochberg",
    "sharpe_ratio",
    "t_statistic",
    "mean_skew_kurtosis",
    "ols_hedge_ratio",
    "half_life",
    "pair_spread_strategy_returns",
    "walk_forward_net_returns",
    "StageResult",
    "SCOPE_TAU_CAL",
    "DEFAULT_SCREEN_T",
    "DEFAULT_PERM_B",
    "DEFAULT_PERM_P",
    "DEFAULT_DSR",
    "DEFAULT_BH_Q",
]

ND = NormalDist()

EULER_MASCHERONI = 0.5772156649015328606
ANNUALIZATION_DAILY = math.sqrt(252.0)

# Pre-registered defaults (FL gauntlet design doc SS2-SS3; v3 SS8.1, SS14 #5).
DEFAULT_SCREEN_T = 2.0          # S1: |t| >= 2.0 to proceed
DEFAULT_PERM_B = 1000           # S2: B = 1000 block-permuted replicates
DEFAULT_PERM_P = 0.05           # S2: p_perm <= 0.05
DEFAULT_DSR = 0.95              # S3: DSR >= 0.95 (scope-stratified)
DEFAULT_BH_Q = 0.10             # S3: BH-FDR q = 0.10 within (family x scope)
MIN_BLOCK_LEN = 5

# Scope-matched S4 hurdle. tau_cal = tau_S5 / (f_min * sqrt(T_full / T_scope))
# = 3.0 / (0.5 * sqrt(4032/1260)) = 3.354 -> pre-registered as ~= 3.5 for the
# 5-year free scope (free_5y_top1k, T=1260). S5/full-depth keeps tau = 3.0.
SCOPE_TAU_CAL = {"free_5y_top1k": 3.5, "pro_full": 3.0, "registry_depth": 3.0}


@dataclass(frozen=True)
class StageResult:
    """One stage's outcome. `statistic`/`threshold` are recorded to the ledger."""

    stage: str
    passed: bool
    statistic: float
    threshold: float
    detail: dict


# ---------------------------------------------------------------------------
# S1 / S2 building blocks
# ---------------------------------------------------------------------------

def t_statistic(returns: list[float]) -> float:
    """t-statistic of the mean of `returns` against zero."""
    n = len(returns)
    if n < 2:
        return 0.0
    m = sum(returns) / n
    v = sum((r - m) ** 2 for r in returns) / (n - 1)
    if v <= 0.0:
        return 0.0
    return m / math.sqrt(v / n)


def _hac_t_stat(returns: list[float], max_lag: int = 1) -> float:
    """Newey-West HAC t-statistic (lag-1 Bartlett kernel) of the mean.

    Strategy returns are autocorrelated (positions persist), so the iid
    t-stat is anticonservative; the HAC version is the S1/S2 statistic.
    """
    n = len(returns)
    if n < 3:
        return t_statistic(returns)
    m = sum(returns) / n
    d = [r - m for r in returns]
    g0 = sum(x * x for x in d) / n
    if g0 <= 0.0:
        return 0.0
    lrv = g0
    for lag in range(1, min(max_lag, n - 1) + 1):
        g = sum(d[t] * d[t - lag] for t in range(lag, n)) / n
        lrv += 2.0 * (1.0 - lag / (n + 1.0)) * g
    if lrv <= 0.0:
        return 0.0
    return m / math.sqrt(lrv / n)


def sharpe_ratio(returns: list[float]) -> float:
    """Per-bar Sharpe ratio (mean / std, ddof=1); 0.0 if undefined."""
    n = len(returns)
    if n < 2:
        return 0.0
    m = sum(returns) / n
    v = sum((r - m) ** 2 for r in returns) / (n - 1)
    if v <= 0.0:
        return 0.0
    return m / math.sqrt(v)


def mean_skew_kurtosis(returns: list[float]) -> tuple[float, float]:
    """Sample skewness (g3) and excess kurtosis (g4) of `returns`.

    Used by PSR/MinTRL (Bailey & Lopez de Prado 2012/2014). Falls back to
    (0, 0) (normal) for short samples.
    """
    n = len(returns)
    if n < 4:
        return 0.0, 0.0
    m = sum(returns) / n
    d = [r - m for r in returns]
    m2 = sum(x * x for x in d) / n
    if m2 <= 0.0:
        return 0.0, 0.0
    m3 = sum(x ** 3 for x in d) / n
    m4 = sum(x ** 4 for x in d) / n
    g3 = m3 / m2 ** 1.5
    g4 = m4 / m2 ** 2 - 3.0
    return g3, g4


def block_length(T: int) -> int:
    """Block-permutation block length b = ceil(T^(1/3)), floored at 5."""
    if T <= 0:
        raise ValueError("T must be positive")
    return max(MIN_BLOCK_LEN, math.ceil(T ** (1.0 / 3.0)))


def _block_shuffle(series: list[float], b: int, rng) -> list[float]:
    """Shuffle whole blocks of length `b` (last block may be short)."""
    T = len(series)
    nblocks = math.ceil(T / b)
    blocks = [series[i * b:(i + 1) * b] for i in range(nblocks)]
    rng.shuffle(blocks)
    out: list[float] = []
    for blk in blocks:
        out.extend(blk)
    return out[:T]


def permutation_pvalue(t_obs: float, t_perm: list[float]) -> float:
    """p = (1 + #{T_b >= T_obs}) / (B + 1) — two-sided uses |.| at the call site."""
    B = len(t_perm)
    if B <= 0:
        raise ValueError("need at least one permutation replicate")
    extreme = sum(1 for t in t_perm if abs(t) >= abs(t_obs))
    return (1.0 + extreme) / (B + 1.0)


def block_permutation_stats(
    effect_fn,
    series_x: list[float],
    series_y: list[float],
    B: int = DEFAULT_PERM_B,
    seed: int = 0,
    T_obs: float = 0.0,
) -> dict:
    """S2 block-permutation null (independent block-shuffle of both legs).

    `effect_fn(x, y)` maps the two (possibly block-shuffled) price series to
    the stage's test statistic. Each replicate independently block-shuffles
    x and y (block length b = ceil(T^(1/3)), min 5), which destroys any
    cross-sectional cointegration while preserving within-block time
    dependence of each leg. p = (1 + #{|T_b| >= |T_obs|}) / (B + 1), PCG64.
    Returns dict with p_perm, B, block length, seed, and replicate stats.
    """
    import numpy as np

    T = len(series_x)
    if len(series_y) != T:
        raise ValueError("series must have equal length")
    b = block_length(T)
    rng = np.random.Generator(np.random.PCG64(seed))
    nblocks = math.ceil(T / b)
    t_perm = []
    for _ in range(B):
        perm = list(range(nblocks))
        rng.shuffle(perm)
        x2: list[float] = []
        for i in perm:
            x2.extend(series_x[i * b:(i + 1) * b])
        perm = list(range(nblocks))
        rng.shuffle(perm)
        y2: list[float] = []
        for i in perm:
            y2.extend(series_y[i * b:(i + 1) * b])
        t_perm.append(float(effect_fn(x2[:T], y2[:T])))
    p = permutation_pvalue(T_obs, t_perm)
    return {
        "p_perm": p,
        "B": B,
        "block_len": b,
        "seed": seed,
        "t_obs": float(T_obs),
        "t_perm": t_perm,
    }


# ---------------------------------------------------------------------------
# S3 — DSR / PSR / MinTRL / BH-FDR
# ---------------------------------------------------------------------------

def expected_max_sharpe(n_trials: int, vbar: float) -> float:
    """SR* = E[max_N] = sqrt(Vbar) * [(1-g)Phi^-1(1-1/N) + g Phi^-1(1-1/(N e))].

    Per-bar units; `vbar` is the empirical variance of the stratum's Sharpe
    estimates (floored at 1/T by the caller). Bailey & Lopez de Prado 2014.
    N=1 is the pre-registration corollary: a single trial faces almost no
    deflation, so the benchmark collapses toward zero.
    """
    if n_trials < 1:
        raise ValueError("n_trials >= 1 required")
    if n_trials == 1:
        return 0.0
    N = float(n_trials)
    return math.sqrt(vbar) * (
        (1.0 - EULER_MASCHERONI) * ND.inv_cdf(1.0 - 1.0 / N)
        + EULER_MASCHERONI * ND.inv_cdf(1.0 - 1.0 / (N * math.e))
    )


def probabilistic_sharpe_ratio(
    sr_hat: float, sr_star: float, T: int, g3: float = 0.0, g4: float = 0.0
) -> float:
    """PSR(SR_hat; SR*, T, g3, g4) = Phi((SR-SR*)sqrt(T-1) / sqrt(1 - g3 SR + (g4-1) SR^2/4))."""
    denom_sq = 1.0 - g3 * sr_hat + (g4 - 1.0) * sr_hat * sr_hat / 4.0
    denom = math.sqrt(max(denom_sq, 1e-12))
    return ND.cdf((sr_hat - sr_star) * math.sqrt(T - 1) / denom)


def deflated_sharpe_ratio(
    sr_hat: float, n_trials: int, vbar: float, T: int, g3: float = 0.0, g4: float = 0.0
) -> float:
    """DSR = PSR(SR_hat; E[max_N], T, g3, g4) with scope-stratified N and Vbar."""
    sr_star = expected_max_sharpe(n_trials, vbar)
    return probabilistic_sharpe_ratio(sr_hat, sr_star, T, g3, g4)


def min_trl(sr_hat: float, g3: float = 0.0, g4: float = 0.0, alpha: float = 0.05) -> float:
    """MinTRL = 1 + (1 - g3 SR + (g4-1) SR^2/4) * (Phi^-1(1-alpha)/SR)^2  (bars).

    Bailey & Lopez de Prado 2012. Minimum number of bars needed to reject
    SR* = 0 at confidence 1-alpha given the observed SR's skew/kurtosis.
    """
    if sr_hat <= 0.0:
        return math.inf
    factor = 1.0 - g3 * sr_hat + (g4 - 1.0) * sr_hat * sr_hat / 4.0
    return 1.0 + factor * (ND.inv_cdf(1.0 - alpha) / sr_hat) ** 2


def benjamini_hochberg(pvalues: list[float], q: float = DEFAULT_BH_Q) -> list[float]:
    """BH-FDR adjusted q-values, same order as input.

    Reject H0(i) for the largest k with p_(k) <= (k/m) q (Benjamini &
    Hochberg 1995). Applied within each (family x scope) stratum.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    qvals = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvalues[i] * m / rank)
        prev = val
        qvals[i] = val
    return qvals


# ---------------------------------------------------------------------------
# Pairs machinery (S0 golden fixture + S1/S2 effect for the pairs family)
# ---------------------------------------------------------------------------

def ols_hedge_ratio(x: list[float], y: list[float]) -> float:
    """OLS hedge ratio beta of y on x (Engle-Granger step 1)."""
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("x and y must be equal length >= 2")
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    var = sum((x[i] - mx) ** 2 for i in range(n))
    if var <= 0.0:
        raise ValueError("x has zero variance")
    return cov / var


def half_life(spread: list[float]) -> float:
    """Half-life of a mean-reverting AR(1) spread: log(0.5)/log(phi_hat).

    phi_hat comes from the level regression ds_t = (phi-1) * s_{t-1} + e_t
    with an intercept (equivalently, both sides de-meaned) — the correct
    AR(1) estimator; a no-intercept regression on the raw level is biased.
    """
    n = len(spread)
    if n < 3:
        return math.nan
    ds = [spread[t] - spread[t - 1] for t in range(1, n)]
    sp1 = [spread[t - 1] for t in range(1, n)]
    md = sum(ds) / len(ds)
    ms = sum(sp1) / len(sp1)
    denom = sum((v - ms) ** 2 for v in sp1)
    if denom <= 0.0:
        return math.nan
    a = sum((sp1[i] - ms) * (ds[i] - md) for i in range(len(ds))) / denom
    phi = 1.0 + a
    if not (0.0 < phi < 1.0):
        return math.nan
    return math.log(0.5) / math.log(phi)


def pair_spread_strategy_returns(
    x: list[float],
    y: list[float],
    half_spread: float = 0.0,
) -> dict:
    """Long-short return series of the pairs strategy on a candidate pair.

    Hedge ratio re-estimated by OLS on the full window (S1/S2 use the
    discovery window; S4's walk-forward re-estimates per block — see
    `walk_forward_net_returns`). Position = -z-score of the spread at t-1
    (long the underpriced leg), scaled to unit spread-vol; transaction cost
    `half_spread` (fraction of price) charged on |Δh| turnover, normalized
    by the mean price level so returns stay in return units.

    Returns dict with `returns`, `t_stat`, `sharpe_per_bar`, `spread`,
    `beta`, `spread_mean`, `spread_std`.
    """
    T = len(x)
    if len(y) != T or T < 3:
        raise ValueError("x and y must be equal length >= 3")
    beta = ols_hedge_ratio(x, y)
    spread = [y[i] - beta * x[i] for i in range(T)]
    ms = sum(spread) / T
    var = sum((v - ms) ** 2 for v in spread) / T
    sd = math.sqrt(var) if var > 0 else 0.0
    xb = sum(x) / T
    rets: list[float] = []
    h_prev = 0.0
    if sd > 0 and xb > 0:
        for t in range(1, T):
            z = (spread[t - 1] - ms) / sd
            h = -z  # units of spread
            u = (spread[t] - spread[t - 1]) / sd
            gross = h * u * (sd / xb)  # in units of price return on x leg
            cost = half_spread * (sd / xb) * abs(h - h_prev)
            rets.append(gross - cost)
            h_prev = h
    # residual autocorrelation inflates the iid t-stat; report the
    # Newey-West (lag-1) HAC t-stat as the stage statistic, which is what
    # the permutation null (S2) then calibrates exactly.
    t = _hac_t_stat(rets)
    return {
        "returns": rets,
        "t_stat": t,
        "sharpe_per_bar": sharpe_ratio(rets),
        "beta": beta,
        "spread": spread,
        "spread_mean": ms,
        "spread_std": sd,
    }


def walk_forward_net_returns(
    x: list[float],
    y: list[float],
    half_spread: float = 0.01,
    n_blocks: int = 8,
) -> dict:
    """S4 cost-aware walk-forward backtest of the pairs candidate.

    The window is split into `n_blocks` contiguous walk-forward blocks; the
    hedge ratio and spread z-score are re-estimated on each block's own
    history (expanding up to that block) so no future data leaks into a
    position. Net of `half_spread` per unit turnover. Entry at next bar,
    no same-day fill (position set at t-1 earns the t return — no look-ahead).

    Returns dict with net `returns`, `t_net`, `sharpe_per_bar` (net),
    `gross_sharpe_per_bar`, `n_blocks`, `turnover`.
    """
    T = len(x)
    if len(y) != T or T < n_blocks * 2:
        raise ValueError("window too short for walk-forward blocks")
    edges = [round(T * i / n_blocks) for i in range(n_blocks + 1)]
    rets: list[float] = []
    gross: list[float] = []
    turnover = 0.0
    h_prev = 0.0
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        if hi - lo < 2:
            continue
        xb = sum(x[:hi]) / hi
        beta = ols_hedge_ratio(x[:hi], y[:hi])
        sp = [y[i] - beta * x[i] for i in range(hi)]
        ms = sum(sp) / hi
        var = sum((v - ms) ** 2 for v in sp) / hi
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd <= 0 or xb <= 0:
            continue
        for t in range(max(lo, 1), hi):
            z = (sp[t - 1] - ms) / sd
            h = -z
            u = (sp[t] - sp[t - 1]) / sd
            g = h * u * (sd / xb)
            c = half_spread * (sd / xb) * abs(h - h_prev)
            gross.append(g)
            rets.append(g - c)
            turnover += abs(h - h_prev)
            h_prev = h
    t_net = _hac_t_stat(rets)
    return {
        "returns": rets,
        "t_net": t_net,
        "sharpe_per_bar": sharpe_ratio(rets),
        "gross_sharpe_per_bar": sharpe_ratio(gross),
        "n_blocks": n_blocks,
        "turnover": turnover,
    }
