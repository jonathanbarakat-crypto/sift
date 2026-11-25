"""Pure statistical primitives and planted fixtures for the `lead_lag` family.

Canonical spec: `docs/lead-lag-family-spec.md` (and
`sift-research/main:docs/lead-lag-family-spec.md`, owner decisions locked
2026-11-25).

Conventions
-----------
* Pure functions: numpy arrays in, plain floats/dicts out. Zero I/O, zero
  ledger access, zero network. Deterministic randomness via explicit `PCG64`
  seeds only.
* Directionality: ordered pair ``(leader -> follower)`` represented as a
  2-column price matrix ``prices[:, 0] = leader``, ``prices[:, 1] = follower``.
* Zero look-ahead:
  - Cross-sectional demeaning against the active PIT universe equal-weight
    mean return ``r_M,t``.
  - Leader AR(1) pre-whitening on non-overlapping lag ``R_A,t-W^(W)`` fitted
    strictly inside each estimation window.
  - Follower own-lag control ``b_t = R_B,t^(W)`` included in every regression.
  - Positions decided at close of bar ``k`` earn return starting at bar ``k+1``
    with costs charged on fill bar ``k+1``.
* HAC mandatory: S1/S2 use Newey-West HAC SE on ``gamma_hat`` with lag
  ``max(H - 1, stats.nw_lag(n))``; S4/S5 use ``stats.t_stat_sharpe_hac`` and
  ``stats.effective_sample_size`` (``T_eff``).
* Compliance: ``latest_lead_lag_abs_z`` returns magnitude ``|z|`` ONLY — never
  a sign, never ``gamma_hat``, never direction.
"""

from __future__ import annotations

import math

import numpy as np

from sift.gauntlet import config as cfg
from sift.gauntlet import s5 as s5stats
from sift.gauntlet import stats

# --------------------------------------------------------------------------
# Return preparation & confounder controls (§1.2)
# --------------------------------------------------------------------------

def log_returns(prices_1d: np.ndarray) -> np.ndarray:
    """Per-bar log-returns of a 1D price series; index 0 is NaN, and any bar
    with non-finite or non-positive endpoint price is NaN (never zero-filled)."""
    p = np.asarray(prices_1d, float)
    T = p.size
    out = np.full(T, np.nan, dtype=float)
    if T < 2:
        return out
    valid = np.isfinite(p) & (p > 0.0)
    both = valid[1:] & valid[:-1]
    if both.any():
        out[1:][both] = np.log(p[1:][both]) - np.log(p[:-1][both])
    return out


def demean_returns(
    r_leg: np.ndarray,
    r_market: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-sectionally demean a log-return series against the active PIT
    equal-weight market return ``r_M,t``."""
    r = np.asarray(r_leg, float).copy()
    if r_market is None:
        return r
    rm = np.asarray(r_market, float)
    if rm.size != r.size:
        raise ValueError(
            f"market_returns length {rm.size} does not match leg length {r.size}"
        )
    return r - rm


def cumulative_returns(r: np.ndarray, W: int) -> np.ndarray:
    """Trailing W-bar cumulative log-return ``R_t^(W) = sum_{k=0}^{W-1} r_{t-k}``.

    An entry is NaN unless all W constituent returns are finite (zero look-ahead).
    """
    arr = np.asarray(r, float)
    w = int(W)
    if w < 1:
        raise ValueError(f"window W must be >= 1, got {W!r}")
    T = arr.size
    if w == 1:
        return arr.copy()
    out = np.full(T, np.nan, dtype=float)
    if T < w:
        return out
    finite = np.isfinite(arr)
    clean = np.where(finite, arr, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(clean)])
    cf = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    sums = cs[w:] - cs[:-w]
    counts = cf[w:] - cf[:-w]
    ok = counts == w
    out[w - 1:][ok] = sums[ok]
    return out


def forward_cumulative_returns(r: np.ndarray, H: int) -> np.ndarray:
    """Forward H-bar cumulative return ``y_{t+1}^(H) = sum_{h=1}^{H} r_{t+h}``.

    Aligned at decision bar ``t`` using strictly future bars ``t+1 .. t+H``.
    Entries with ``t + H >= T`` or any non-finite bar in ``t+1 .. t+H`` are NaN.
    """
    arr = np.asarray(r, float)
    h = int(H)
    if h < 1:
        raise ValueError(f"horizon H must be >= 1, got {H!r}")
    T = arr.size
    out = np.full(T, np.nan, dtype=float)
    if T <= h:
        return out
    trailing = cumulative_returns(arr, h)
    # trailing[t + h] = sum_{j=t+1}^{t+h} r[j]
    out[:-h] = trailing[h:]
    return out


def prewhiten_leader(
    r_a_tilde: np.ndarray,
    W: int,
    *,
    phi_hat: float | None = None,
) -> tuple[np.ndarray, float]:
    """Leader AR(1) pre-whitening over horizon W on non-overlapping lags:

        R_{A,t}^(W) = c + phi_A * R_{A, t-W}^(W) + u_t
        x_t = R_{A,t}^(W) - phi_hat_A * R_{A, t-W}^(W)

    Returns ``(x, phi_hat_A)``. When ``phi_hat`` is supplied (OOS block), it is
    applied directly without re-estimating on future data.
    """
    w = int(W)
    R_a = cumulative_returns(r_a_tilde, w)
    T = R_a.size
    R_prev = np.full(T, np.nan, dtype=float)
    if T > w:
        R_prev[w:] = R_a[:-w]
    if phi_hat is None:
        valid = np.isfinite(R_a) & np.isfinite(R_prev)
        if int(valid.sum()) >= 5:
            prev_v = R_prev[valid]
            cur_v = R_a[valid]
            prev_c = prev_v - prev_v.mean()
            denom = float(np.dot(prev_c, prev_c))
            if denom > 1e-18:
                phi = float(np.dot(prev_c, cur_v - cur_v.mean()) / denom)
            else:
                phi = 0.0
        else:
            phi = 0.0
    else:
        phi = float(phi_hat)
    x = R_a - phi * R_prev
    return x, phi


def build_regression_arrays(
    r_a_tilde: np.ndarray,
    r_b_tilde: np.ndarray,
    lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
    hold_bars: int | None = None,
    *,
    phi_a: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Assemble finite aligned ``(y, x, b, phi_a, valid_mask)`` for §1.2:

        y_{t+1}^(H) = alpha + gamma * x_t + phi * b_t + eps_{t+1}

    where ``x_t = R_{A,t}^(W) - phi_A_hat * R_{A,t-W}^(W)`` is the pre-whitened
    leader W-bar innovation, ``b_t = R_{B,t}^(W)`` is the follower's own-lag
    control, and ``y_{t+1}^(H)`` is the follower's forward H-bar return
    starting at bar ``t+1``.
    """
    W = int(lag_bars)
    H = W if hold_bars is None else int(hold_bars)
    ra = np.asarray(r_a_tilde, float)
    rb = np.asarray(r_b_tilde, float)
    if ra.size != rb.size:
        raise ValueError(
            f"leader length {ra.size} does not match follower length {rb.size}"
        )
    x_full, phi = prewhiten_leader(ra, W, phi_hat=phi_a)
    b_full = cumulative_returns(rb, W)
    y_full = forward_cumulative_returns(rb, H)
    mask = np.isfinite(x_full) & np.isfinite(b_full) & np.isfinite(y_full)
    return y_full[mask], x_full[mask], b_full[mask], phi, mask


def _build_iut_regression_arrays(
    r_a_tilde: np.ndarray,
    r_b_tilde: np.ndarray,
    lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
    hold_bars: int | None = None,
    *,
    phi_a: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Assemble paired ``(y_pred, y_struct, x, b, phi_a, mask)`` strictly inside
    an estimation window for the Berger (1982) Intersection-Union Test (IUT):

    * ``y_pred = R_{B,t+1..t+H}^(H)`` tests causal Granger predictability
      (rejects reverse direction ``B -> A`` without collider bias).
    * ``y_struct = y_pred - beta0_hat * R_{A,t+1..t+H}^(H)`` removes the
      contemporaneous-loading × leader-AR(1) confounder (``F-LEADLAG-CONTEMP``:
      ``r_{B,t+1} = beta0 * r_{A,t+1} + ...`` with ``gamma* = 0``).

    Both targets use only bars strictly inside the historical estimation window;
    OOS execution remains 100% causal (bars ``<= k`` only).
    """
    W = int(lag_bars)
    H = W if hold_bars is None else int(hold_bars)
    ra = np.asarray(r_a_tilde, float)
    rb = np.asarray(r_b_tilde, float)
    if ra.size != rb.size:
        raise ValueError(
            f"leader length {ra.size} does not match follower length {rb.size}"
        )
    x_full, phi = prewhiten_leader(ra, W, phi_hat=phi_a)
    b_full = cumulative_returns(rb, W)
    y_pred = forward_cumulative_returns(rb, H)
    y_lead = forward_cumulative_returns(ra, H)

    contemp_mask = np.isfinite(x_full) & np.isfinite(b_full)
    if int(contemp_mask.sum()) >= 5:
        xc = x_full[contemp_mask] - float(np.mean(x_full[contemp_mask]))
        bc = b_full[contemp_mask] - float(np.mean(b_full[contemp_mask]))
        vx = float(np.dot(xc, xc))
        beta0_hat = float(np.dot(xc, bc) / vx) if vx > 1e-18 else 0.0
    else:
        beta0_hat = 0.0

    y_struct = y_pred - beta0_hat * y_lead
    mask = (
        np.isfinite(x_full)
        & np.isfinite(b_full)
        & np.isfinite(y_pred)
        & np.isfinite(y_struct)
    )
    return y_pred[mask], y_struct[mask], x_full[mask], b_full[mask], phi, mask


# --------------------------------------------------------------------------
# Exact Newey-West HAC partial regression & fast FWL permutation kernel
# --------------------------------------------------------------------------

def _fwl_basis_and_y(y: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis Q_b for [1, b] and residualized target y_perp."""
    n = y.size
    Z = np.column_stack([np.ones(n, dtype=float), b])
    Q_b, _ = np.linalg.qr(Z, mode="reduced")
    y_perp = y - Q_b @ (Q_b.T @ y)
    return Q_b, y_perp


def _fwl_hac_gamma_t(
    x: np.ndarray,
    Q_b: np.ndarray,
    y_perp: np.ndarray,
    lag: int,
) -> tuple[float, float, float]:
    """Exact OLS gamma_hat, Newey-West HAC SE, and t_HAC via FWL projection.

    Algebraically identical to extracting entry (1, 1) of
    ``(X'X)^{-1} S_NW (X'X)^{-1}`` for ``X = [1, x, b]``: by Frisch-Waugh-Lovell
    the gamma row of ``(X'X)^{-1} X'`` is ``x_perp' / (x_perp' x_perp)``, so
    the exact scalar score is ``s_t = x_perp[t] * eps[t]``.
    """
    x_perp = x - Q_b @ (Q_b.T @ x)
    denom = float(np.dot(x_perp, x_perp))
    if not math.isfinite(denom) or denom <= 1e-18:
        return 0.0, math.inf, 0.0
    gamma = float(np.dot(x_perp, y_perp) / denom)
    eps = y_perp - gamma * x_perp
    s = x_perp * eps
    S = float(np.dot(s, s))
    L = min(max(int(lag), 0), s.size - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        S += 2.0 * w * float(np.dot(s[k:], s[:-k]))
    if not math.isfinite(S) or S <= 0.0:
        return gamma, math.inf, 0.0
    var_gamma = S / (denom * denom)
    if not math.isfinite(var_gamma) or var_gamma <= 0.0:
        return gamma, math.inf, 0.0
    se = math.sqrt(var_gamma)
    return gamma, se, float(gamma / se)


def _nw_hac_from_perp(
    x_perp: np.ndarray,
    denom: float,
    y_perp: np.ndarray,
    lag: int,
) -> tuple[float, float, float]:
    """Compute ``(gamma, se, t_hac)`` given pre-projected ``x_perp`` and ``denom``."""
    gamma = float(np.dot(x_perp, y_perp) / denom)
    eps = y_perp - gamma * x_perp
    s = x_perp * eps
    S = float(np.dot(s, s))
    L = min(max(int(lag), 0), s.size - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        S += 2.0 * w * float(np.dot(s[k:], s[:-k]))
    if not math.isfinite(S) or S <= 0.0:
        return gamma, math.inf, 0.0
    var_gamma = S / (denom * denom)
    if not math.isfinite(var_gamma) or var_gamma <= 0.0:
        return gamma, math.inf, 0.0
    se = math.sqrt(var_gamma)
    return gamma, se, float(gamma / se)


def _iut_hac_gamma_t(
    x: np.ndarray,
    Q_b: np.ndarray,
    y_perp_pred: np.ndarray,
    y_perp_struct: np.ndarray,
    lag: int,
) -> tuple[float, float, float]:
    """Berger (1982) Intersection-Union Test combining:

    1. Causal Granger prediction (``y_perp_pred``)
    2. Contemporaneous-orthogonalized structural transmission (``y_perp_struct``)

    Requires both estimates to share the same sign and returns the conservative
    smaller-``|t_HAC|`` tuple ``(gamma, se, t_hac)`` (preserving exact level
    ``<= alpha`` under the union null without multiplicity inflation).
    """
    x_perp = x - Q_b @ (Q_b.T @ x)
    denom = float(np.dot(x_perp, x_perp))
    if not math.isfinite(denom) or denom <= 1e-18:
        return 0.0, math.inf, 0.0
    gp = float(np.dot(x_perp, y_perp_pred) / denom)
    gs = float(np.dot(x_perp, y_perp_struct) / denom)
    if (gp * gs) <= 0.0:
        return 0.0, math.inf, 0.0
    gp, sp, tp = _nw_hac_from_perp(x_perp, denom, y_perp_pred, lag)
    gs, ss, ts = _nw_hac_from_perp(x_perp, denom, y_perp_struct, lag)
    if abs(tp) <= abs(ts):
        return gp, sp, tp
    return gs, ss, ts


def ols_hac_gamma(
    y: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    *,
    hold_bars: int = 1,
    lag: int | None = None,
) -> dict:
    """Fit ``y = alpha + gamma * x + phi * b + eps`` and compute Newey-West
    HAC standard error & t-stat for ``gamma`` at lag ``max(H - 1, nw_lag(n))``."""
    yv = np.asarray(y, float)
    xv = np.asarray(x, float)
    bv = np.asarray(b, float)
    n = yv.size
    if n < 5:
        return {
            "alpha": 0.0,
            "gamma": 0.0,
            "phi": 0.0,
            "se_gamma": math.inf,
            "t_hac": 0.0,
            "n": int(n),
            "lag": 0,
        }
    L = (
        max(int(hold_bars) - 1, stats.nw_lag(n))
        if lag is None
        else max(0, int(lag))
    )
    L = min(L, n - 1)
    X = np.column_stack([np.ones(n, dtype=float), xv, bv])
    coef, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    Q_b, y_perp = _fwl_basis_and_y(yv, bv)
    gamma, se, t_hac = _fwl_hac_gamma_t(xv, Q_b, y_perp, L)
    return {
        "alpha": float(coef[0]),
        "gamma": float(gamma),
        "phi": float(coef[2]),
        "se_gamma": float(se),
        "t_hac": float(t_hac),
        "n": int(n),
        "lag": int(L),
    }


def _ols_iut_hac_gamma(
    y_pred: np.ndarray,
    y_struct: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    *,
    hold_bars: int = 1,
    lag: int | None = None,
) -> dict:
    """IUT Newey-West HAC fit combining causal Granger and structural targets."""
    yp = np.asarray(y_pred, float)
    ys = np.asarray(y_struct, float)
    xv = np.asarray(x, float)
    bv = np.asarray(b, float)
    n = yp.size
    if n < 5:
        return {
            "alpha": 0.0,
            "gamma": 0.0,
            "phi": 0.0,
            "se_gamma": math.inf,
            "t_hac": 0.0,
            "n": int(n),
            "lag": 0,
        }
    L = (
        max(int(hold_bars) - 1, stats.nw_lag(n))
        if lag is None
        else max(0, int(lag))
    )
    L = min(L, n - 1)
    X = np.column_stack([np.ones(n, dtype=float), xv, bv])
    coef, _, _, _ = np.linalg.lstsq(X, yp, rcond=None)
    Q_b, y_perp_p = _fwl_basis_and_y(yp, bv)
    y_perp_s = ys - Q_b @ (Q_b.T @ ys)
    gamma, se, t_hac = _iut_hac_gamma_t(xv, Q_b, y_perp_p, y_perp_s, L)
    return {
        "alpha": float(coef[0]),
        "gamma": float(gamma),
        "phi": float(coef[2]),
        "se_gamma": float(se),
        "t_hac": float(t_hac),
        "n": int(n),
        "lag": int(L),
    }


# --------------------------------------------------------------------------
# S0, S1, S2 — calibration/sanity, discovery screen, block-permutation null
# --------------------------------------------------------------------------

def s0_lead_lag_sanity(
    prices: np.ndarray,
    *,
    market_returns: np.ndarray | None = None,
    lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
    hold_bars: int | None = None,
    formation: int = cfg.LEAD_LAG_DEFAULT_FORMATION,
    step: int = cfg.LEAD_LAG_STEP,
    delist_idx: np.ndarray | None = None,
    gamma_true: float | None = None,
    gamma_tol: float = cfg.LEAD_LAG_S0_GAMMA_TOL,
) -> dict:
    """S0 gate for `lead_lag` (spec §3 S0):

    Real-data sanity checks:
    1. Valid overlapping finite return bars >= max(formation + step, 120).
    2. Non-degenerate leader and follower return std > 1e-8.
    3. Stale-quote zero-return fraction < 0.25 on BOTH legs.

    When `gamma_true` is supplied (planted synthetic fixture calibration),
    additionally verifies ``|gamma_hat - gamma_true| <= gamma_tol``.
    """
    P = np.asarray(prices, float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("lead_lag prices must be a 2-column (T, 2) array [leader, follower]")
    W = int(lag_bars)
    H = W if hold_bars is None else int(hold_bars)
    mask = s5stats.delist_mask(P, delist_idx)
    r = s5stats.valid_returns(P, delist_idx, mask)
    ra_raw, rb_raw = r[:, 0], r[:, 1]
    both = np.isfinite(ra_raw) & np.isfinite(rb_raw)
    valid_bars = int(both.sum())
    min_bars = max(int(formation) + int(step), 120)

    if valid_bars >= 2:
        ra_v = ra_raw[both]
        rb_v = rb_raw[both]
        sigma_a = float(np.std(ra_v, ddof=1))
        sigma_b = float(np.std(rb_v, ddof=1))
        stale_a = float(np.mean(np.abs(ra_v) <= 1e-12))
        stale_b = float(np.mean(np.abs(rb_v) <= 1e-12))
    else:
        sigma_a = 0.0
        sigma_b = 0.0
        stale_a = 1.0
        stale_b = 1.0

    bars_ok = valid_bars >= min_bars
    var_ok = (sigma_a > cfg.LEAD_LAG_MIN_SIGMA) and (sigma_b > cfg.LEAD_LAG_MIN_SIGMA)
    stale_ok = (
        (stale_a < cfg.LEAD_LAG_STALE_QUOTE_MAX)
        and (stale_b < cfg.LEAD_LAG_STALE_QUOTE_MAX)
    )

    ra_tilde = demean_returns(ra_raw, market_returns)
    rb_tilde = demean_returns(rb_raw, market_returns)
    yp, ys, x, b, phi_a, _ = _build_iut_regression_arrays(ra_tilde, rb_tilde, W, H)
    fit = _ols_iut_hac_gamma(yp, ys, x, b, hold_bars=H)

    gamma_ok = True
    if gamma_true is not None:
        gamma_ok = (
            math.isfinite(fit["gamma"])
            and abs(fit["gamma"] - float(gamma_true)) <= float(gamma_tol)
        )

    passed = bool(bars_ok and var_ok and stale_ok and gamma_ok and fit["n"] >= 20)
    return {
        "passed": passed,
        "bars_ok": bool(bars_ok),
        "var_ok": bool(var_ok),
        "stale_ok": bool(stale_ok),
        "gamma_ok": bool(gamma_ok),
        "valid_bars": valid_bars,
        "min_bars": min_bars,
        "sigma_leader": sigma_a,
        "sigma_follower": sigma_b,
        "stale_ratio_leader": stale_a,
        "stale_ratio_follower": stale_b,
        "lag_bars": W,
        "hold_bars": H,
        "gamma_hat": fit["gamma"],
        "gamma_true": gamma_true,
        "t_hac": fit["t_hac"],
        "phi_leader": phi_a,
        "n_obs": fit["n"],
    }


def s1_lead_lag_screen(
    r_a_tilde: np.ndarray,
    r_b_tilde: np.ndarray,
    *,
    lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
    hold_bars: int | None = None,
    threshold: float = cfg.S1_THRESHOLD,
) -> dict:
    """S1 discovery screen: ``|t_HAC(gamma_hat)| >= threshold`` (two-sided)."""
    W = int(lag_bars)
    H = W if hold_bars is None else int(hold_bars)
    yp, ys, x, b, phi_a, _ = _build_iut_regression_arrays(r_a_tilde, r_b_tilde, W, H)
    fit = _ols_iut_hac_gamma(yp, ys, x, b, hold_bars=H)
    t_hac = float(fit["t_hac"])
    passed = math.isfinite(t_hac) and abs(t_hac) >= float(threshold)
    return {
        "pass": bool(passed),
        "t": t_hac,
        "abs_t": abs(t_hac) if math.isfinite(t_hac) else 0.0,
        "gamma": fit["gamma"],
        "se_gamma": fit["se_gamma"],
        "phi_leader": phi_a,
        "phi_follower": fit["phi"],
        "n": fit["n"],
        "lag": fit["lag"],
        "lag_bars": W,
        "hold_bars": H,
    }


def s2_lead_lag_permutation(
    r_a_tilde: np.ndarray,
    r_b_tilde: np.ndarray,
    *,
    lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
    hold_bars: int | None = None,
    B: int = cfg.S2_B,
    seed: int = 0,
) -> dict:
    """S2 block-permutation null for `lead_lag` (spec §3 S2):

    Holds follower ``y`` and own-lag control ``b_t`` fixed while block-permuting
    the pre-whitened leader series ``x_t`` using contiguous blocks of length
    ``b = stats.block_length(n)`` under ``PCG64(seed)``. Test statistic is the
    IUT ``|t_HAC(gamma)|``; ``p_perm = (1 + #{T_b >= T_obs}) / (B + 1)``.
    """
    W = int(lag_bars)
    H = W if hold_bars is None else int(hold_bars)
    yp, ys, x, b, phi_a, _ = _build_iut_regression_arrays(r_a_tilde, r_b_tilde, W, H)
    n = yp.size
    if n < 10:
        return {
            "observed": 0.0,
            "p_perm": 1.0,
            "B": int(B),
            "block_len": cfg.S2_MIN_BLOCK,
            "seed": int(seed),
            "gamma": 0.0,
            "t_hac": 0.0,
            "n": int(n),
            "lag": 0,
        }
    L = min(max(H - 1, stats.nw_lag(n)), n - 1)
    Q_b, y_perp_p = _fwl_basis_and_y(yp, b)
    y_perp_s = ys - Q_b @ (Q_b.T @ ys)
    gamma_obs, _, t_obs_signed = _iut_hac_gamma_t(x, Q_b, y_perp_p, y_perp_s, L)
    t_obs = abs(t_obs_signed) if math.isfinite(t_obs_signed) else 0.0
    b_len = stats.block_length(n)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    ge = 0
    for _ in range(int(B)):
        x_perm = stats._block_permute(x, b_len, rng)
        _, _, t_b = _iut_hac_gamma_t(x_perm, Q_b, y_perp_p, y_perp_s, L)
        if math.isfinite(t_b) and abs(t_b) >= t_obs:
            ge += 1
    p_perm = (1.0 + ge) / (int(B) + 1.0)
    return {
        "observed": float(t_obs),
        "p_perm": float(p_perm),
        "B": int(B),
        "block_len": int(b_len),
        "seed": int(seed),
        "gamma": float(gamma_obs),
        "t_hac": float(t_obs_signed),
        "phi_leader": float(phi_a),
        "n": int(n),
        "lag": int(L),
    }


# --------------------------------------------------------------------------
# S4 — cost-aware walk-forward (market-neutralized gating + Roll costs)
# --------------------------------------------------------------------------

def lead_lag_walk_forward(
    prices: np.ndarray,
    *,
    market_returns: np.ndarray | None = None,
    lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
    hold_bars: int | None = None,
    formation: int = cfg.LEAD_LAG_DEFAULT_FORMATION,
    open_z: float = cfg.LEAD_LAG_DEFAULT_OPEN_Z,
    step: int = cfg.LEAD_LAG_STEP,
    cost_model: stats.CostModel | None = None,
    half_spreads: str | float | np.ndarray = "roll",
) -> dict:
    """Cost-aware walk-forward for `lead_lag` (spec §3 S4, Decision 2A):

    * Non-overlapping OOS blocks of length ``step`` starting at ``formation``.
    * All parameters (``phi_A``, ``gamma_hat``, ``sgn(gamma_hat)``, ``mu_x``,
      ``sigma_x``, follower Roll half-spread ``hs_B``) estimated strictly on
      the formation window ``[start - formation, start)`` (zero look-ahead).
    * Position decided at close of bar ``k`` earns return starting at bar ``k+1``.
    * Primary (gating) series: market-neutralized follower return
      ``p_k * r_tilde_{B,k+1}`` charged with follower Roll half-spread +
      ``1.0`` bps market-hedge leg floor + live participation slippage via
      ``stats.apply_costs``.
    * Diagnostic series: unhedged follower return ``p_k * r_{B,k+1}`` charged
      with follower Roll half-spread + slippage.
    * Magnitude-only ``current_abs_z`` exposed for live monitoring (never sign).
    """
    if cost_model is None:
        cost_model = stats.CostModel.from_config()
    P = np.asarray(prices, float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("lead_lag prices must be a 2-column (T, 2) array [leader, follower]")
    T = P.shape[0]
    W = int(lag_bars)
    H = W if hold_bars is None else int(hold_bars)
    form = int(formation)
    st = int(step)
    oz = float(open_z)

    ra_raw = log_returns(P[:, 0])
    rb_raw = log_returns(P[:, 1])
    ra_tilde = demean_returns(ra_raw, market_returns)
    rb_tilde = demean_returns(rb_raw, market_returns)
    R_a = cumulative_returns(ra_tilde, W)

    net = np.zeros(T, dtype=float)
    gross = np.zeros(T, dtype=float)
    raw_net = np.zeros(T, dtype=float)
    raw_gross = np.zeros(T, dtype=float)
    positions = np.zeros(T, dtype=float)
    z_all = np.full(T, np.nan, dtype=float)
    hs_b_all = np.full(T, cost_model.half_spread_bps, dtype=float)

    t = form
    while t < T:
        end = min(t + st, T)
        lo = max(0, t - form)
        hi = t

        # Strictly in-sample formation window [lo:hi] (bars < t only)
        yp_f, ys_f, x_f, b_f, phi_a, _ = _build_iut_regression_arrays(
            ra_tilde[lo:hi], rb_tilde[lo:hi], W, H
        )
        fit = _ols_iut_hac_gamma(yp_f, ys_f, x_f, b_f, hold_bars=H)
        gamma_hat = float(fit["gamma"])
        x_form, _ = prewhiten_leader(ra_tilde[lo:hi], W, phi_hat=phi_a)
        x_fin = x_form[np.isfinite(x_form)]
        if x_fin.size >= 5:
            mu_x = float(x_fin.mean())
            sd_x = float(x_fin.std(ddof=1))
        else:
            mu_x = 0.0
            sd_x = 0.0

        if (
            not math.isfinite(gamma_hat)
            or abs(gamma_hat) <= 1e-12
            or not math.isfinite(sd_x)
            or sd_x <= 1e-12
        ):
            sign_g = 0.0
        else:
            sign_g = 1.0 if gamma_hat > 0.0 else -1.0

        if isinstance(half_spreads, str) and half_spreads == "roll":
            hs_b, _ = stats.roll_half_spread_bps_window(P[:, 1], lo, hi)
        elif isinstance(half_spreads, str):
            hs_b = float(cost_model.half_spread_bps)
        else:
            hs_arr = np.asarray(half_spreads, float)
            hs_b = float(hs_arr) if hs_arr.ndim == 0 else float(hs_arr[t])

        blen = end - t
        pos_block = np.zeros(blen, dtype=float)
        hold_rem = 0
        for j in range(blen):
            k = t + j
            if k - W >= 0 and np.isfinite(R_a[k]) and np.isfinite(R_a[k - W]):
                x_k = float(R_a[k] - phi_a * R_a[k - W])
                z_k = (x_k - mu_x) / sd_x if sd_x > 1e-12 else math.nan
            else:
                z_k = math.nan
            z_all[k] = z_k
            hs_b_all[k] = hs_b
            if math.isfinite(z_k) and abs(z_k) >= oz and sign_g != 0.0:
                pos_block[j] = sign_g * (1.0 if z_k > 0.0 else -1.0)
                hold_rem = H
            elif hold_rem > 1 and j > 0:
                pos_block[j] = pos_block[j - 1]
                hold_rem -= 1
            else:
                pos_block[j] = 0.0
                hold_rem = 0

        # Bar k+1 earns pos[k] * r_tilde_{B, k+1} (no same-bar fill)
        rb_tilde_blk = np.nan_to_num(rb_tilde[t:end], nan=0.0)
        rb_raw_blk = np.nan_to_num(rb_raw[t:end], nan=0.0)
        g_blk = np.zeros(blen, dtype=float)
        rg_blk = np.zeros(blen, dtype=float)
        if blen >= 2:
            g_blk[1:] = pos_block[:-1] * rb_tilde_blk[1:]
            rg_blk[1:] = pos_block[:-1] * rb_raw_blk[1:]

        # Primary costs: follower Roll hs_B + 1.0 bps market-hedge leg floor
        n_blk = stats.apply_costs(
            g_blk,
            pos_block,
            cost_model,
            hedge=np.ones(blen, dtype=float),
            half_spread_bps_x=cfg.LEAD_LAG_INDEX_HEDGE_BPS,
            half_spread_bps_y=hs_b,
        )
        # Diagnostic unhedged costs: follower leg only
        rn_blk = stats.apply_costs(
            rg_blk,
            pos_block,
            cost_model,
            hedge=np.zeros(blen, dtype=float),
            half_spread_bps_x=0.0,
            half_spread_bps_y=hs_b,
        )

        positions[t:end] = pos_block
        gross[t:end] = g_blk
        net[t:end] = n_blk
        raw_gross[t:end] = rg_blk
        raw_net[t:end] = rn_blk
        t = end

    abs_z = np.abs(z_all)
    fin_z = abs_z[np.isfinite(abs_z)]
    current_abs_z = float(fin_z[-1]) if fin_z.size > 0 else None

    return {
        "net": net,
        "gross": gross,
        "raw_net": raw_net,
        "raw_gross": raw_gross,
        "positions": positions,
        "abs_z": abs_z,
        "current_abs_z": current_abs_z,
        "half_spread_b_bps": hs_b_all,
        "T": int(T),
    }


def latest_lead_lag_abs_z(
    prices: np.ndarray,
    *,
    market_returns: np.ndarray | None = None,
    lag_bars: int = cfg.LEAD_LAG_DEFAULT_LAG,
    formation: int = cfg.LEAD_LAG_DEFAULT_FORMATION,
) -> float | None:
    """Magnitude-only ``|z_k|`` of the pre-whitened leader innovation at the
    most recent bar (strictly trailing formation window, zero look-ahead,
    NEVER returns a sign or slope)."""
    P = np.asarray(prices, float)
    if P.ndim != 2 or P.shape[1] != 2:
        return None
    T = P.shape[0]
    W = int(lag_bars)
    form = int(formation)
    if T < max(form + 2, 2 * W + 10):
        return None
    ra_raw = log_returns(P[:, 0])
    ra_tilde = demean_returns(ra_raw, market_returns)
    R_a = cumulative_returns(ra_tilde, W)

    # Fit phi_A, mu_x, sigma_x strictly on bars < T - 1
    hi = T - 1
    lo = max(0, hi - form)
    x_form, phi_a = prewhiten_leader(ra_tilde[lo:hi], W)
    x_fin = x_form[np.isfinite(x_form)]
    if x_fin.size < 5:
        return None
    sd_x = float(x_fin.std(ddof=1))
    if not math.isfinite(sd_x) or sd_x <= 1e-12:
        return None
    mu_x = float(x_fin.mean())
    if not (np.isfinite(R_a[hi]) and np.isfinite(R_a[hi - W])):
        return None
    x_now = float(R_a[hi] - phi_a * R_a[hi - W])
    z_now = (x_now - mu_x) / sd_x
    if not math.isfinite(z_now):
        return None
    return float(abs(z_now))


# --------------------------------------------------------------------------
# Pre-registered synthetic fixtures (spec §3 S0, deterministic PCG64)
# --------------------------------------------------------------------------

def _prices_from_returns(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Exponentiate cumulative log-returns starting at 100.0 (shape (T, 2))."""
    ra = np.asarray(r_a, float)
    rb = np.asarray(r_b, float)
    T = ra.size
    pa = 100.0 * np.exp(np.cumsum(np.nan_to_num(ra, nan=0.0)))
    pb = 100.0 * np.exp(np.cumsum(np.nan_to_num(rb, nan=0.0)))
    # Preserve leading bar = 100.0 so log_returns reproduces r[1:]
    pa = np.concatenate([[100.0], pa[:-1]]) if T >= 2 else pa
    pb = np.concatenate([[100.0], pb[:-1]]) if T >= 2 else pb
    return np.column_stack([pa, pb])


def f_leadlag(
    T: int = s5stats.T_FULL,
    *,
    gamma_true: float = 0.30,
    lag_true: int = 1,
    beta0: float = 0.4,
    phi_b: float = -0.1,
    sigma_a: float = 0.015,
    sigma_eps: float = 0.009,
    seed: int = 101,
) -> dict:
    """`F-LEADLAG`: planted directed lead-lag (A -> B) with contemporaneous
    co-movement and follower AR(1) autocorrelation."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    L = int(lag_true)
    r_a = rng.normal(0.0, sigma_a, size=int(T))
    eps = rng.normal(0.0, sigma_eps, size=int(T))
    r_b = np.zeros(int(T), dtype=float)
    for t in range(int(T)):
        lag_term = gamma_true * r_a[t - L] if t >= L else 0.0
        own_term = phi_b * r_b[t - 1] if t >= 1 else 0.0
        r_b[t] = beta0 * r_a[t] + lag_term + own_term + eps[t]
    prices = _prices_from_returns(r_a, r_b)
    return {
        "prices": prices,
        "market_returns": np.zeros(int(T), dtype=float),
        "gamma_true": float(gamma_true),
        "lag_true": L,
        "beta0": float(beta0),
        "phi_b": float(phi_b),
        "seed": int(seed),
        "fixture": "F-LEADLAG",
    }


def f_leadlag_contemp(
    T: int = s5stats.T_FULL,
    *,
    beta0: float = 0.8,
    phi_a: float = 0.25,
    phi_b: float = 0.25,
    sigma: float = 0.015,
    seed: int = 202,
) -> dict:
    """`F-LEADLAG-CONTEMP`: strong contemporaneous beta + strong AR(1) in both
    legs, true gamma* = 0. Tests §1.2 confounder controls."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    u_a = rng.normal(0.0, sigma, size=int(T))
    u_b = rng.normal(0.0, sigma, size=int(T))
    r_a = np.zeros(int(T), dtype=float)
    r_b = np.zeros(int(T), dtype=float)
    for t in range(int(T)):
        r_a[t] = (phi_a * r_a[t - 1] if t >= 1 else 0.0) + u_a[t]
        r_b[t] = beta0 * r_a[t] + (phi_b * r_b[t - 1] if t >= 1 else 0.0) + u_b[t]
    prices = _prices_from_returns(r_a, r_b)
    return {
        "prices": prices,
        "market_returns": np.zeros(int(T), dtype=float),
        "gamma_true": 0.0,
        "lag_true": 1,
        "beta0": float(beta0),
        "phi_a": float(phi_a),
        "phi_b": float(phi_b),
        "seed": int(seed),
        "fixture": "F-LEADLAG-CONTEMP",
    }


def f_leadlag_regime(
    T: int = s5stats.T_FULL,
    *,
    split: int = 2016,
    gamma_active: float = 0.30,
    lag_true: int = 1,
    beta0: float = 0.4,
    phi_b: float = -0.1,
    sigma_a: float = 0.015,
    sigma_eps: float = 0.009,
    seed: int = 303,
) -> dict:
    """`F-LEADLAG-REGIME`: edge active on trailing 5y sub-window (or first half
    [0, split) per spec) and dead (gamma=0) on the rest, mirroring F-REGIME.

    Spec table: gamma* = 0.30 for bars [0, split), gamma* = 0 for [split, T).
    Wait: callers testing a trailing 5y window (or leading active window) can
    slice ``prices[:1260]`` (active 5y) vs full ``prices`` (4032, attenuated).
    """
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    L = int(lag_true)
    r_a = rng.normal(0.0, sigma_a, size=int(T))
    eps = rng.normal(0.0, sigma_eps, size=int(T))
    r_b = np.zeros(int(T), dtype=float)
    for t in range(int(T)):
        g_t = gamma_active if t < int(split) else 0.0
        lag_term = g_t * r_a[t - L] if t >= L else 0.0
        own_term = phi_b * r_b[t - 1] if t >= 1 else 0.0
        r_b[t] = beta0 * r_a[t] + lag_term + own_term + eps[t]
    prices = _prices_from_returns(r_a, r_b)
    return {
        "prices": prices,
        "market_returns": np.zeros(int(T), dtype=float),
        "gamma_active": float(gamma_active),
        "split": int(split),
        "lag_true": L,
        "seed": int(seed),
        "fixture": "F-LEADLAG-REGIME",
    }


def f_leadlag_delist(
    T: int = s5stats.T_FULL,
    *,
    delist_bar: int = 1500,
    delist_leg: str = "follower",
    gamma_true: float = 0.30,
    lag_true: int = 1,
    beta0: float = 0.4,
    phi_b: float = -0.1,
    sigma_a: float = 0.015,
    sigma_eps: float = 0.009,
    seed: int = 404,
) -> dict:
    """`F-LEADLAG-DELIST`: planted edge where leader or follower delists at
    ``delist_bar`` (trailing NaN post-delist)."""
    base = f_leadlag(
        T=T,
        gamma_true=gamma_true,
        lag_true=lag_true,
        beta0=beta0,
        phi_b=phi_b,
        sigma_a=sigma_a,
        sigma_eps=sigma_eps,
        seed=seed,
    )
    prices = base["prices"].copy()
    d = max(0, min(int(T), int(delist_bar)))
    if delist_leg == "leader":
        prices[d:, 0] = np.nan
        delist_idx = np.array([d, int(T)], dtype=int)
    else:
        prices[d:, 1] = np.nan
        delist_idx = np.array([int(T), d], dtype=int)
    base["prices"] = prices
    base["delist_idx"] = delist_idx
    base["delist_bar"] = d
    base["delist_leg"] = delist_leg
    base["fixture"] = "F-LEADLAG-DELIST"
    return base


def f_leadlag_null(
    T: int = s5stats.T_FREE_5Y,
    *,
    sigma: float = 0.015,
    seed: int = 505,
) -> dict:
    """`F-LEADLAG-NULL`: independent white-noise leader and follower (true gamma* = 0)."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    r_a = rng.normal(0.0, sigma, size=int(T))
    r_b = rng.normal(0.0, sigma, size=int(T))
    prices = _prices_from_returns(r_a, r_b)
    return {
        "prices": prices,
        "market_returns": np.zeros(int(T), dtype=float),
        "gamma_true": 0.0,
        "lag_true": 1,
        "seed": int(seed),
        "fixture": "F-LEADLAG-NULL",
    }


__all__ = [
    "log_returns",
    "demean_returns",
    "cumulative_returns",
    "forward_cumulative_returns",
    "prewhiten_leader",
    "build_regression_arrays",
    "ols_hac_gamma",
    "s0_lead_lag_sanity",
    "s1_lead_lag_screen",
    "s2_lead_lag_permutation",
    "lead_lag_walk_forward",
    "latest_lead_lag_abs_z",
    "f_leadlag",
    "f_leadlag_contemp",
    "f_leadlag_regime",
    "f_leadlag_delist",
    "f_leadlag_null",
]
