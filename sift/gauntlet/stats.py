"""Pure statistical functions for the SIFT gauntlet stages S0-S4.

Design source: FL branch docs/sift-gauntlet-s5-design,
research/design/2026-11-15-sift-gauntlet-s5-design.md §2 (per-stage spec),
§3 (scope-matched calibration), §5 (planted fixtures); SIFT v3 scope doc §8.1.

Conventions
-----------
* All functions are pure: numpy arrays in, plain floats/dicts out. No I/O,
  no ledger access, no randomness except where a seed is passed explicitly
  (PCG64 via ``numpy.random.Generator(np.random.PCG64(seed))``).
* Returns are per-bar unless noted; annualization uses ``bars_per_year``
  (default 252 daily bars).
* Serial correlation (#12): ``t_stat_sharpe`` / ``psr`` / the sqrt(252)
  annualization assume IID returns. Under POSITIVE autocorrelation the iid
  t is ANTICONSERVATIVE (std of t ~ sqrt((1+phi)/(1-phi)) > 1); under
  mean-reversion (negative autocorrelation) it is conservative. The
  mean-reversion pairs path keeps the iid variants (conservative direction);
  positive-autocorrelation exposures (e.g. deck baskets, pull/37) must use
  ``t_stat_sharpe_hac`` / the ``t_eff`` option of ``psr``/``dsr`` — every
  switch is explicit, never silent.
* Pre-registered constants live in ``sift/gauntlet/config.py`` and are hashed
  into each trial's config_hash by the runner.

Hard rules honored here (v3 §12): nothing in this module touches market data
ingestion or the network; the ledger supplies scope-stratified denominators;
public-facing output of stages is survival facts only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EULER_GAMMA = 0.5772156649015328606065120900824024310421  # Euler–Mascheroni


# --------------------------------------------------------------------------
# Normal distribution helpers (no scipy dependency)
# --------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


# --------------------------------------------------------------------------
# S0 — harness calibration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationResult:
    beta_hat: float
    beta_true: float
    half_life_hat: float
    half_life_true: float
    beta_ok: bool
    half_life_ok: bool
    passed: bool


def estimate_ar1(series: np.ndarray) -> tuple[float, float]:
    """OLS AR(1) on a demeaned series: s_t = c + phi s_{t-1} + e_t.
    Returns (phi, half_life) with half_life = -ln2 / ln(phi) for 0 < phi < 1.
    A degenerate (constant) series returns (0.0, inf) — no mean reversion."""
    s = np.asarray(series, dtype=float)
    s_lag, s_next = s[:-1], s[1:]
    s_lag_c = s_lag - s_lag.mean()
    denom = float(np.dot(s_lag_c, s_lag_c))
    if denom <= 0:
        return 0.0, math.inf
    phi = float(np.dot(s_lag_c, s_next - s_next.mean()) / denom)
    half_life = -math.log(2.0) / math.log(phi) if 0.0 < phi < 1.0 else math.inf
    return phi, half_life


def ols_hedge_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """OLS hedge ratio beta in y = alpha + beta x + e (levels)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        raise ValueError("degenerate x")
    return float(np.dot(xc, y - y.mean()) / denom)


def calibrate_ar1_fixture(
    x: np.ndarray,
    y: np.ndarray,
    beta_true: float,
    half_life_true: float,
    beta_tol: float = 0.05,
    hl_rel_tol: float = 0.20,
    spread: np.ndarray | None = None,
) -> CalibrationResult:
    """S0 golden fixture: an AR(1) cointegrated pair must reproduce the known
    hedge ratio and half-life within pre-registered tolerances
    (|beta_hat - beta*| <= 0.05, half-life within +/-20%).

    Both are estimated on the *true-beta* hedge convention of the fixture
    check: beta* is the planted cointegrating coefficient, so the spread is
    y - beta* x (levels) and its AR(1) half-life is the assertion; beta_hat
    is the OLS hedge on levels and must recover beta*.

    `spread` (optional) is the planted spread series itself (used by
    log-space fixtures where the level hedge is not exactly beta*); when
    given, the half-life assertion runs on it directly.

    REAL-DATA ROLE (finding #14, doc-only — no behavior change): on real
    data the caller (first_run) sets beta_true = the pair's OLS hedge on
    levels and half_life_true = the measured half-life, so the beta
    assertion is TAUTOLOGICAL (beta_hat - beta_true = 0.0 exactly) and the
    genuine gate is the finite-half-life (mean-reversion) check: an
    explosive spread (phi_hat > 1 -> half-life = inf) FAILS S0. The planted
    AR(1) fixture (F-PAIR) remains the real calibration of both assertions."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    beta_hat = ols_hedge_ratio(x, y)
    sp = np.asarray(spread, float) if spread is not None else y - beta_true * x
    _, hl_hat = estimate_ar1(sp)
    beta_ok = abs(beta_hat - beta_true) <= beta_tol
    hl_ok = math.isfinite(hl_hat) and abs(hl_hat - half_life_true) <= hl_rel_tol * half_life_true
    return CalibrationResult(
        beta_hat=beta_hat, beta_true=beta_true,
        half_life_hat=hl_hat, half_life_true=half_life_true,
        beta_ok=beta_ok, half_life_ok=hl_ok, passed=beta_ok and hl_ok,
    )


def mean_reversion_stat(series: np.ndarray) -> float:
    """The candidate's S2 predictive statistic on a spread-return series:
    -corr(spread level at t, spread return at t+1). Positive under mean
    reversion (high spread predicts a down next bar). This is the statistic
    block-permuted in S2; block permutation preserves within-block dependence
    so the null stays valid under AR(1) serial correlation."""
    s = np.asarray(series, float)
    level = np.cumsum(s)
    if s.size < 3 or s.std(ddof=1) == 0:
        return 0.0
    return -float(np.corrcoef(level[:-1], s[1:])[0, 1])


# --------------------------------------------------------------------------
# S1 — discovery screen: t-stat of mean excess return (cohort - control)
# --------------------------------------------------------------------------

def t_stat_mean(x: np.ndarray) -> float:
    """t-statistic of the mean of x against zero (two-sided)."""
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return 0.0
    sd = x.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(x.mean() / (sd / math.sqrt(n)))


def discovery_screen(excess_returns: np.ndarray, threshold: float = 2.0) -> dict:
    """S1: |t| >= threshold proceeds (pre-registered screen, discovery only)."""
    t = t_stat_mean(excess_returns)
    return {"t": t, "threshold": threshold, "pass": abs(t) >= threshold}


# --------------------------------------------------------------------------
# S2 — block-permutation null
# --------------------------------------------------------------------------

def block_length(T: int, min_block: int = 5) -> int:
    """Block day-shuffle length b = ceil(T^(1/3)), floored at min_block.
    T=1260 -> 11; T=4032 -> 16 (design doc §2 S2)."""
    b = math.ceil(T ** (1.0 / 3.0))
    return max(min_block, b)


def _block_permute(series: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle contiguous blocks of length b (last block may be short)."""
    T = series.shape[0]
    n_blocks = math.ceil(T / b)
    starts = np.arange(n_blocks) * b
    perm = rng.permutation(n_blocks)
    out = np.empty_like(series)
    pos = 0
    for p in perm:
        seg = series[starts[p]: starts[p] + b]
        out[pos: pos + len(seg)] = seg
        pos += len(seg)
    return out


def permutation_pvalue(
    observed: float,
    replicates: np.ndarray,
    B: int | None = None,
) -> float:
    """p = (1 + #{T_b >= T_obs}) / (B + 1) — granularity 1/(B+1)."""
    reps = np.asarray(replicates, float)
    n = reps.size if B is None else B
    extreme = int(np.sum(reps >= observed))
    return (1.0 + extreme) / (n + 1.0)


def block_permutation_null(
    series: np.ndarray,
    statistic_fn,
    B: int = 1000,
    seed: int = 0,
    block_len: int | None = None,
) -> dict:
    """S2: B block-permuted replicates of `statistic_fn` (PCG64-seeded).

    The null is exchangeability of time blocks (no event-conditional
    predictability); plain day-shuffle is anticonservative under serial
    correlation, so we shuffle blocks of length b = ceil(T^(1/3)) (min 5).
    Returns observed stat, p_perm = (1+#extreme)/(B+1), seed, B, block length.
    """
    series = np.asarray(series, float)
    T = series.shape[0]
    b = block_len if block_len is not None else block_length(T)
    rng = np.random.Generator(np.random.PCG64(seed))
    obs = float(statistic_fn(series))
    reps = np.empty(B)
    for i in range(B):
        reps[i] = float(statistic_fn(_block_permute(series, b, rng)))
    p = permutation_pvalue(obs, reps, B)
    return {"observed": obs, "p_perm": p, "B": B, "seed": seed,
            "block_len": b, "replicates": reps}


# --------------------------------------------------------------------------
# S3 — DSR (primary) + BH-FDR, scope-stratified denominators
# --------------------------------------------------------------------------

def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max_N] under the null (Bailey & Lopez de Prado 2014):
    SR* = sqrt(V) * [ (1-gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e)) ].
    Per-bar units. N = trials in the same scope stratum (incl. kills);
    V = empirical variance of the stratum's Sharpe estimates."""
    if n_trials < 1:
        raise ValueError("N must be >= 1")
    g = EULER_GAMMA
    n_eff = max(n_trials, 2)  # Phi^{-1}(1-1/N) needs N >= 2 numerically
    return math.sqrt(var_sharpe) * (
        (1 - g) * _norm_ppf(1 - 1.0 / n_eff)
        + g * _norm_ppf(1 - 1.0 / (n_eff * math.e))
    )


def psr(sr_hat: float, sr_star: float, T: int, skew: float, kurt: float,
        t_eff: float | None = None) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012):
    PSR = Phi( (sr_hat - sr_star) * sqrt(T-1) / sqrt(1 - skew*sr_hat + (kurt-1)*sr_hat^2/4) ).
    kurt is the (non-excess, RAW) kurtosis; the term (kurt - 1) * sr^2 / 4
    uses the raw 4th standardized moment as in the paper.

    ``t_eff`` (effective sample size, e.g. from ``effective_sample_size``)
    replaces T under the sqrt: sqrt(t_eff - 1). Use it for series with
    POSITIVE serial correlation, where the iid sqrt(T-1) is
    anticonservative (#12). Default None keeps the iid sqrt(T-1)."""
    if T < 2:
        raise ValueError("T must be >= 2")
    n_eff = T if t_eff is None else float(t_eff)
    if n_eff < 2:
        raise ValueError("t_eff must be >= 2")
    denom = math.sqrt(1.0 - skew * sr_hat + (kurt - 1.0) * sr_hat ** 2 / 4.0)
    if denom <= 0:
        return 0.0 if sr_hat < sr_star else 1.0
    return _norm_cdf((sr_hat - sr_star) * math.sqrt(n_eff - 1) / denom)


def dsr(
    sr_hat: float,
    T: int,
    n_stratum: int,
    v_stratum: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    var_floor: float | None = None,
    t_eff: float | None = None,
) -> dict:
    """Deflated Sharpe Ratio at the candidate's registered scope.

    The deflation benchmark uses the scope-stratified denominator:
    N = n_stratum, V = max(v_stratum, var_floor) with var_floor = 1/T
    (theoretical minimum variance of a Sharpe estimate). ``t_eff`` is the
    optional effective sample size forwarded to ``psr`` (serial-correlation
    option, #12); None keeps the iid sqrt(T-1). Returns the DSR,
    the benchmark SR*, and the inputs used (all recorded in the ledger).
    """
    v = v_stratum
    floor = (1.0 / T) if var_floor is None else var_floor
    if v < floor:
        v = floor
    sr_star = expected_max_sharpe(n_stratum, v)
    d = psr(sr_hat, sr_star, T, skew, kurt, t_eff=t_eff)
    return {"dsr": d, "sr_star": sr_star, "sr_hat": sr_hat, "T": T,
            "n_stratum": n_stratum, "v_stratum": v, "skew": skew, "kurt": kurt,
            "t_eff": t_eff}


def bh_fdr(p_values: np.ndarray, q: float = 0.10) -> dict:
    """Benjamini-Hochberg within one (family x scope) stratum.
    Reject H0(i) for the largest k with p_(k) <= (k/m) q. Returns per-trial
    q-values and the rejection set (indices)."""
    p = np.asarray(p_values, float)
    m = p.size
    if m == 0:
        return {"q_values": np.array([]), "rejected": np.array([], dtype=int), "m": 0}
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresholds
    k = int(np.max(np.nonzero(below)[0])) + 1 if below.any() else 0
    rejected_sorted = order[:k]
    # BH q-value per hypothesis: q_i = min_{j >= rank(i)} m/j * p_(j)
    qvals_sorted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
    q_values = np.empty(m)
    q_values[order] = np.minimum(qvals_sorted, 1.0)
    return {"q_values": q_values, "rejected": np.sort(rejected_sorted), "m": m}


def min_trl(sr: float, skew: float, kurt: float, alpha: float = 0.05) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado 2012):
    MinTRL = 1 + (1 - g3*SR + (g4-1)*SR^2/4) * (Phi^{-1}(1-alpha) / SR)^2."""
    if sr <= 0:
        return math.inf
    z = _norm_ppf(1 - alpha)
    return 1.0 + (1 - skew * sr + (kurt - 1.0) * sr ** 2 / 4.0) * (z / sr) ** 2


def skew_kurt(returns: np.ndarray) -> tuple[float, float]:
    """Bias-corrected sample skewness (G1, adjusted Fisher-Pearson) and RAW
    (non-excess) kurtosis (G2 = excess + 3) of a return series.

    Convention (#13): the plain standardized-moment estimators
    mean(z^3)/mean(z^4) are downward-biased (MC on N(0,1): mean raw kurt
    2.80 at n=60, 2.95 at n=252 vs true 3.0), which feeds PSR/DSR/MinTRL
    slightly anticonservative for positive SR at short windows. We return
    the standard bias-corrected forms:
        G1 = sqrt(n(n-1))/(n-2) * g1,   g1 = m3 / m2^1.5
        G2 = (n-1)/((n-2)(n-3)) * ((n+1) g2 - 3(n-1)) + 3,  g2 = m4 / m2^2
    G2 is RAW kurtosis (excess + 3): PSR expects the raw 4th standardized
    moment in its (kurt - 1) * sr^2 / 4 term — do NOT pass excess kurtosis.
    Degenerate guards unchanged: n < 4 or zero variance -> (0.0, 3.0)."""
    r = np.asarray(returns, float)
    n = r.size
    if n < 4:
        return 0.0, 3.0
    m = r.mean()
    d = r - m
    m2 = float(np.dot(d, d)) / n
    if m2 <= 0:
        return 0.0, 3.0
    m3 = float(np.mean(d ** 3))
    m4 = float(np.mean(d ** 4))
    g1 = m3 / m2 ** 1.5
    g2 = m4 / m2 ** 2
    skew = math.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
    kurt = ((n - 1.0) / ((n - 2.0) * (n - 3.0))
            * ((n + 1.0) * g2 - 3.0 * (n - 1.0)) + 3.0)
    return float(skew), float(kurt)


def sharpe_ratio(returns: np.ndarray, bars_per_year: int = 252) -> float:
    """Annualized Sharpe from per-bar returns (rf = 0, per the design doc's
    per-bar SR convention; risk-free is folded into the cost model)."""
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * math.sqrt(bars_per_year))


def t_stat_sharpe(returns: np.ndarray) -> float:
    """t-stat of the mean return: t = SR_per_bar * sqrt(T). ASSUMES IID
    returns (#12): under positive autocorrelation this is anticonservative
    (std of t ~ sqrt((1+phi)/(1-phi))); under mean reversion it is
    conservative. The pairs gauntlet path keeps this iid variant (the pairs
    family is mean-reversion, the conservative direction); positive-
    autocorrelation exposures must use ``t_stat_sharpe_hac`` explicitly."""
    r = np.asarray(returns, float)
    n = r.size
    if n < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / (sd / math.sqrt(n)))


# --------------------------------------------------------------------------
# Serial-correlation-aware (HAC) variants — soundness finding #12.
# Lag rule: Andrews (1991) automatic lag L = floor(4 * (T/100)^(2/9)),
# floored at 1 and capped at T-1. Bartlett kernel weights 1 - k/(L+1).
# --------------------------------------------------------------------------

def nw_lag(T: int) -> int:
    """Newey-West automatic lag floor(4*(T/100)^(2/9)) (Andrews 1991),
    floored at 1 (T=252 -> 4, T=1260 -> 6)."""
    if T < 2:
        return 1
    return max(1, min(int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))), T - 1))


def _hac_long_run_var(r: np.ndarray, lag: int) -> tuple[float, float]:
    """Bartlett-kernel long-run variance S and gamma_0 of a demeaned series.
    S = g0 + 2 * sum_{k=1..L} (1 - k/(L+1)) g_k, with g_k = mean(d_t d_{t-k})."""
    T = r.size
    d = r - r.mean()
    g0 = float(np.dot(d, d)) / T
    S = g0
    for k in range(1, lag + 1):
        gk = float(np.dot(d[:-k], d[k:])) / T
        S += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return g0, S


def t_stat_sharpe_hac(returns: np.ndarray, lag: int | None = None) -> dict:
    """Newey-West/HAC t-stat of the mean return (Bartlett kernel):
    t = mean(r) / sqrt(S / T), S = long-run variance with automatic lag
    ``nw_lag(T)`` unless `lag` is given. Valid under AR(1) serial
    correlation; under positive autocorrelation it removes the iid t's
    anticonservatism (MC, T=252, phi=+0.5: iid t dispersion ~1.73 vs HAC
    ~1). Returns the t, the lag used, gamma_0, the long-run variance, and
    the effective sample size T_eff = T * g0 / S."""
    r = np.asarray(returns, float)
    T = r.size
    if T < 2:
        return {"t": 0.0, "lag": 0, "gamma0": 0.0, "long_run_var": 0.0,
                "t_eff": 0.0}
    L = nw_lag(T) if lag is None else max(1, min(int(lag), T - 1))
    g0, S = _hac_long_run_var(r, L)
    if S <= 0:
        return {"t": 0.0, "lag": L, "gamma0": g0, "long_run_var": S,
                "t_eff": 0.0}
    t = float(r.mean() / math.sqrt(S / T))
    t_eff = float(T * g0 / S)
    return {"t": t, "lag": L, "gamma0": g0, "long_run_var": S, "t_eff": t_eff}


def effective_sample_size(returns: np.ndarray, lag: int | None = None) -> float:
    """Effective sample size T_eff = T * gamma_0 / S (HAC). T_eff < T under
    positive autocorrelation, T_eff > T under mean reversion. Feed T_eff to
    ``psr``/``dsr`` via their ``t_eff`` argument for positive-autocorrelation
    series (#12). Degenerate series (T < 2 or S <= 0) -> 0.0."""
    out = t_stat_sharpe_hac(returns, lag=lag)
    return out["t_eff"]


# --------------------------------------------------------------------------
# S4 — cost-aware walk-forward backtest
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    half_spread_bps: float = 100.0            # placeholder, FL open item #6
    slippage_participation_threshold: float = 0.05
    slippage_bps_per_pct: float = 5.0
    max_participation: float = 0.10
    risk_capital_per_trade: float = 0.01
    max_open_positions: int = 20


def apply_costs(strat_returns: np.ndarray, positions: np.ndarray,
                cost_model: CostModel) -> np.ndarray:
    """Net-of-cost strategy returns. Cost is charged on position changes at
    the bar where the change occurs: |Δpos| * half_spread_bps / 1e4.
    Entry is next-bar (positions are already lagged by the caller)."""
    strat = np.asarray(strat_returns, float).copy()
    pos = np.asarray(positions, float)
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat -= dpos * cost_model.half_spread_bps / 1e4
    return strat


def pairs_strategy_returns(
    prices: np.ndarray,
    formation: int = 60,
    open_z: float = 2.0,
    close_z: float = 0.5,
    cost_model: CostModel | None = None,
) -> dict:
    """Pairs strategy on a two-column price matrix [x, y] (log-return hedge,
    z-scored spread, entry at next bar — no same-day fill, no look-ahead).

    Returns per-bar strategy returns (gross), positions, hedge ratios, and
    the z-score series. Mean-reversion convention: short the spread when z is
    high (pos = -1), long when z is low (pos = +1).
    """
    if cost_model is None:
        cost_model = CostModel()
    prices = np.asarray(prices, float)
    T = prices.shape[0]
    x, y = prices[:, 0], prices[:, 1]
    rx = np.zeros(T); rx[1:] = np.diff(np.log(x))
    ry = np.zeros(T); ry[1:] = np.diff(np.log(y))
    hedge = np.full(T, np.nan)
    for t in range(formation, T):
        w = slice(t - formation, t)
        vx = rx[w].var(ddof=1)
        if vx > 0:
            hedge[t] = np.cov(rx[w], ry[w], ddof=1)[0, 1] / vx
    h = np.nan_to_num(hedge)
    rs = ry - h * rx                      # spread return of bar t
    spread = np.cumsum(rs)                # spread level at close of bar t
    z = np.full(T, np.nan)
    for t in range(formation, T):
        w = spread[t - formation:t]
        sd = w.std(ddof=1)
        if sd > 0:
            z[t] = (spread[t] - w.mean()) / sd
    pos = np.zeros(T)                     # pos[t] decided at close of bar t
    for t in range(formation, T - 1):
        if pos[t - 1] == 0:
            if z[t] > open_z:
                pos[t] = -1.0
            elif z[t] < -open_z:
                pos[t] = 1.0
        else:
            pos[t] = 0.0 if abs(z[t]) < close_z else pos[t - 1]
    strat = np.zeros(T)
    strat[1:] = pos[:-1] * rs[1:]         # bar t earns pos[t-1] * rs[t]
    net = apply_costs(strat, pos, cost_model)
    return {"gross": strat, "net": net, "positions": pos, "hedge": h, "z": z}


def walk_forward_returns(
    prices: np.ndarray,
    formation: int = 60,
    step: int = 120,
    open_z: float = 2.0,
    close_z: float = 0.5,
    cost_model: CostModel | None = None,
) -> dict:
    """S4 walk-forward: re-estimate the hedge on each rolling formation window
    and trade the following step block with parameters estimated at its start
    (no look-ahead). Concatenates per-bar net returns of the OOS blocks; bars
    before the first full formation window stay zero."""
    if cost_model is None:
        cost_model = CostModel()
    prices = np.asarray(prices, float)
    T = prices.shape[0]
    net = np.zeros(T)
    t = formation
    while t < T:
        end = min(t + step, T)
        w = slice(max(0, t - formation), t)
        x, y = prices[w, 0], prices[w, 1]
        rx = np.zeros(len(x)); rx[1:] = np.diff(np.log(x))
        ry = np.zeros(len(y)); ry[1:] = np.diff(np.log(y))
        vx = rx.var(ddof=1)
        h = float(np.cov(rx, ry, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
        block = prices[t:end]
        bx = np.zeros(len(block)); bx[1:] = np.diff(np.log(block[:, 0]))
        by = np.zeros(len(block)); by[1:] = np.diff(np.log(block[:, 1]))
        rs = by - h * bx
        train_spread = np.cumsum(ry - h * rx)
        mu = float(train_spread.mean()) if len(train_spread) else 0.0
        sd = float(train_spread.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
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
        net_block = apply_costs(gross, pos, cost_model)
        net[t:end] = net_block
        t = end
    return {"net": net, "T": T}


def cscv_pbo(
    returns_matrix: np.ndarray,
    S: int = 16,
    seed: int = 0,
) -> float | None:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2016/17).

    `returns_matrix` is (T, N) strategy return series (one column per
    configuration). Split the T bars into S=2*K contiguous blocks; for each
    of the C(S, K) combinations, pick the IS winner on the K training blocks
    and rank it OOS on the complementary K blocks; lambda = logit of the
    OOS rank (rank 0 = best); PBO = P(lambda >= 0) estimated by the
    fraction of combinations where the IS winner's OOS rank is in the
    BOTTOM half (lambda >= 0 means the IS winner underperforms the OOS
    median — the overfitting signature).

    Sign convention (soundness #2): the comparison is ``lambda >= 0``.
    The previous ``lambda <= 0`` returned 1 - PBO (a dominant config,
    true PBO = 0, returned 1.0). Boundary convention: lambda == 0 (IS
    winner at exactly the OOS median, rank == N/2) counts as OVERFIT
    (PBO side), matching Bailey et al.'s P(lambda <= 0) on the negated
    logit with the +0.5 smoothing.

    Returns None when the CSCV design is undefined (N < 2 configurations
    or T < S blocks): CSCV on a single configuration is meaningless, and
    silently returning 0.0 would advertise protection that was never
    computed. Callers must treat None as "not computed" (fail closed).
    """
    R = np.asarray(returns_matrix, float)
    if R.ndim != 2:
        return None
    T, N = R.shape
    if T < S or N < 2:
        return None
    b = T // S
    blocks = R[: b * S].reshape(S, b, N).sum(axis=1)  # (S, N) block returns
    K = S // 2
    idx = np.arange(S)
    # enumerate combinations deterministically
    from itertools import combinations
    combos = []
    for train in combinations(range(S), K):
        train = np.array(train)
        test = np.array([i for i in idx if i not in train])
        is_ret = blocks[train].sum(axis=0)
        winner = int(np.argmax(is_ret))
        oos = blocks[test].sum(axis=0)
        # rank of the IS winner among ALL configs OOS (0 = best). Compared
        # with a strict > on float sums, so a genuine tie is NOT counted as
        # "beaten" — the +0.5 smoothing in lambda then places the boundary.
        rank = int(np.sum(oos > oos[winner]))
        lam = math.log((rank + 0.5) / (N - rank + 0.5)) if 0 <= rank < N else 0.0
        combos.append(lam)
    if not combos:
        return None
    return float(np.mean(np.array(combos) >= 0.0))
