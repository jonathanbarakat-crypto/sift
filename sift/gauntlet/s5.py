"""S5 registry-depth validation + delist-safe price panel helpers (v3 §8.1-8.4).

Design source: FL research/design/2026-11-15-sift-gauntlet-s5-design.md §2 (S5
spec), §5 (F-REGIME / F-DELIST fixtures), §7 (ledger rows with
scope_id=registry_depth); SIFT v3 scope doc §8.1-§8.4.

Conventions
-----------
* Pure functions: numpy arrays in, plain floats/dicts out. No I/O, no ledger
  access, no network (v3 §12: server-side only; nothing here touches raw data
  egress — these operate on arrays the server already holds).
* S5 re-evaluates at scope ``registry_depth`` (full window) with the FULL-DEPTH
  hurdle t >= 3.0 (S4_T_NET_FULL) vs the free scope's 3.5 (tau_cal) — S5's own
  thresholds are never weakened (design doc §3.2 rule 2).
* Delisted names: a panel may carry NaN after a name's last bar. No statistic
  in this module ever reads a post-delist cell: masking happens up front and
  every reduction runs on the valid mask only.
"""

from __future__ import annotations

import math

import numpy as np

# Full-depth window (16y of daily bars, 2010-2025 per FL configs/periods.yaml)
T_FULL = 4032
# Free-tier registered scope (trailing 5y)
T_FREE_5Y = 1260
# S5 full-depth t hurdle (Harvey-Liu-Zhu) — kept in config; repeated here only
# as the module-level default used by the pure functions below.
S5_T_FULL = 3.0


# --------------------------------------------------------------------------
# Delist-safe panels (F-DELIST)
# --------------------------------------------------------------------------

def delist_mask(prices: np.ndarray, delist_idx: np.ndarray | None = None) -> np.ndarray:
    """Boolean (T, N) validity mask: True where the bar exists (pre-delist).

    `delist_idx[i]` is the first INVALID (post-delist) bar index for column i;
    None or T means "never delisted / valid to the end". NaN cells are always
    invalid. This is the single choke point through which every statistic in
    this module reads the panel — post-delist prices are never used.
    """
    P = np.asarray(prices, float)
    if P.ndim == 1:
        P = P[:, None]
    T, N = P.shape
    mask = np.isfinite(P)
    if delist_idx is not None:
        di = np.asarray(delist_idx, int)
        if di.shape[0] != N:
            raise ValueError(
                f"delist_idx has {di.shape[0]} entries but the panel has {N} columns")
        for j in range(N):
            d = int(di[j])
            d = max(0, min(T, d))
            mask[d:, j] = False
    return mask


def valid_spread_returns(spread_returns: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask a spread-return series (length T-1; entry k is the return from bar
    k to bar k+1) so that ONLY returns whose BOTH endpoint bars are valid in
    both legs survive. Returns with either endpoint post-delist/NaN are set to
    NaN — never zero-filled, never kept. This is the alignment-correct choke
    point for the S2 permutation series at registry depth."""
    sr = np.asarray(spread_returns, float)
    m = np.asarray(mask, bool)
    if m.ndim == 2:
        row_ok = m.all(axis=1)          # bar valid in every leg
    else:
        row_ok = m
    # return k (bar k -> k+1) needs row_ok[k] AND row_ok[k+1]
    pair_ok = row_ok[:-1] & row_ok[1:]
    out = np.where(pair_ok[: sr.shape[0]], sr, np.nan)
    return out


def leading_valid_window(mask: np.ndarray) -> int:
    """Length of the LEADING prefix of bars that are valid in every leg — the
    only window S5's walk-forward may run on (a count of all-valid rows is NOT
    a prefix and would include post-delist rows)."""
    m = np.asarray(mask, bool)
    row_ok = m.all(axis=1) if m.ndim == 2 else m
    if not row_ok.any():
        return 0
    first_bad = int(np.argmin(row_ok))   # argmin of bool -> first False
    return first_bad if not row_ok[first_bad] else int(row_ok.shape[0])


def valid_returns(prices: np.ndarray, delist_idx: np.ndarray | None = None,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """Log returns with post-delist cells set to NaN (never zero-filled — a
    zero would fabricate a 0% return and bias every downstream statistic)."""
    P = np.asarray(prices, float)
    if P.ndim == 1:
        P = P[:, None]
    r = np.full(P.shape, np.nan)
    r[1:] = np.log(P[1:]) - np.log(P[:-1])
    if mask is None:
        mask = delist_mask(P, delist_idx)
    # a return into bar t is valid only if both t-1 and t are valid
    both = mask[1:] & mask[:-1]
    r[1:][~both] = np.nan
    return r


def panel_stats_valid(prices: np.ndarray, delist_idx: np.ndarray | None = None) -> dict:
    """Per-column count of valid bars and valid return bars — the F-DELIST
    assertion target: no column contributes post-delist data."""
    P = np.asarray(prices, float)
    if P.ndim == 1:
        P = P[:, None]
    mask = delist_mask(P, delist_idx)
    r = valid_returns(P, delist_idx, mask)
    return {
        "valid_bars": mask.sum(axis=0).astype(int),
        "valid_returns": np.isfinite(r).sum(axis=0).astype(int),
        "n_cols": P.shape[1],
        "T": P.shape[0],
    }


# --------------------------------------------------------------------------
# S5 — registry-depth re-evaluation (pure statistics)
# --------------------------------------------------------------------------

def s5_full_depth_verdict(
    net_returns: np.ndarray,
    *,
    t_hurdle: float = S5_T_FULL,
    net_sr_min: float = 0.0,
    min_trl_max: int | None = T_FULL,
    p_perm: float | None = None,
    p_perm_max: float = 0.05,
    dsr: float | None = None,
    dsr_min: float = 0.95,
    pbo: float | None = None,
    pbo_max: float = 0.5,
    mask: np.ndarray | None = None,
    use_hac: bool = False,
) -> dict:
    """S5 verdict at registry_depth scope (full window), full-strength
    thresholds (design doc §2 S5 / §3.2 rule 2 — never weakened, never opt-in):

      * net SR > 0 after costs
      * |t_net| >= t_hurdle (3.0 full-depth vs 3.5 free-scope tau_cal)
      * MinTRL <= T_full (4032), and when `use_hac=True`, MinTRL <= T_eff
      * p_perm <= 0.05 (block permutation at full depth, b=ceil(T^(1/3))=16)
      * DSR >= 0.95 with the registry_depth stratum denominator
      * PBO <= 0.5 (CSCV, S=16, over the pre-registered variant grid)

    Every input is computed on the full window with the delist mask applied
    (post-delist cells excluded). Returns pass/fail per criterion plus the
    overall verdict; the survivor class is assigned by the caller (runner):
    pass -> validated-green, fail -> on-watch-amber (v3 §8.4).

    FAIL CLOSED (soundness #15): the required statistics p_perm, dsr and
    pbo are None -> the check FAILS. "A stage not run cannot pass": a None
    stat means the statistic was never computed, and passing on it would
    let an unrun check masquerade as a pass. (Previously ``(x is None) or
    ...`` let None through.) ``min_trl_max=None`` is different and stays
    permissive on purpose: it is the documented opt-out meaning "no MinTRL
    bar at this scope", not a missing statistic.
    """
    r = np.asarray(net_returns, float)
    if mask is not None:
        m = np.asarray(mask, bool)
        r = r[m[: r.shape[0]]] if m.ndim == 1 else r[np.broadcast_to(m, r.shape)]
    r = r[np.isfinite(r)]
    T = r.size
    sr_ann = 0.0
    t_net = 0.0
    t_eff = float(T)
    skew, kurt = 0.0, 3.0
    min_trl = math.inf
    if T >= 2 and r.std(ddof=1) > 0:
        from sift.gauntlet import stats as _stats  # local import to stay pure-ish
        sr_ann = _stats.sharpe_ratio(r)
        if use_hac:
            hac = _stats.t_stat_sharpe_hac(r)
            t_net = float(hac["t"])
            t_eff = float(hac["t_eff"])
        else:
            t_net = _stats.t_stat_sharpe(r)
            t_eff = float(T)
        skew, kurt = _stats.skew_kurt(r)
        sr_bar = sr_ann / math.sqrt(252.0)
        min_trl = _stats.min_trl(sr_bar, skew, kurt, alpha=0.05)
    min_trl_ok = (min_trl_max is None) or (min_trl <= min_trl_max)
    if use_hac:
        min_trl_ok = min_trl_ok and (min_trl <= t_eff)
    checks = {
        "net_sr_positive": sr_ann > net_sr_min,
        "t_full_depth": abs(t_net) >= t_hurdle,
        "min_trl": bool(min_trl_ok),
        # fail closed: a required statistic that is None (never computed)
        # FAILS its check — a stage not run cannot pass (soundness #15).
        "p_perm": (p_perm is not None) and (p_perm <= p_perm_max),
        "dsr": (dsr is not None) and (dsr >= dsr_min),
        "pbo": (pbo is not None) and (pbo <= pbo_max),
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "checks": checks,
        "t_net": t_net,
        "t_hurdle": t_hurdle,
        "sr_ann": sr_ann,
        "T": T,
        "t_eff": t_eff,
        "use_hac": bool(use_hac),
        "min_trl": min_trl,
        "skew": skew,
        "kurt": kurt,
        "p_perm": p_perm,
        "dsr": dsr,
        "pbo": pbo,
    }


def attenuation_ratio(t_scope: float, t_full: float, T_scope: int = T_FREE_5Y,
                      T_full: int = T_FULL) -> float:
    """Empirical persistence fraction f_hat = t_full / (t_scope *
    sqrt(T_full/T_scope)) (design doc §3.1: t_full ≈ f · sqrt(T_F/T_scope) ·
    t_scope). F-REGIME has the edge in 7 of 16 years, so f_hat ≈ 7/16 ≈ 0.44
    is expected (i.e. t_full ≈ 0.78 × t_scope before noise)."""
    denom = t_scope * math.sqrt(T_full / T_scope)
    if denom == 0:
        return math.nan
    return t_full / denom
