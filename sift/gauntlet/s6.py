"""S6 deck (portfolio) gauntlet statistics — spec v4.2 §4 (hermes-tasks #166).

Pure functions, same conventions as sift/gauntlet/stats.py: numpy in,
plain floats/dicts out, no I/O, no ledger access, randomness only via an
explicit PCG64 seed.

Contents
--------
- `estimate_half_spread_bps(prices)` — REALISTIC per-name half-spread
  computed from the data (proxy: absolute log-return dispersion scaled to
  round-trip bps, floored at S6_HALF_SPREAD_BPS_FLOOR). This replaces the
  100 bps placeholder; `S6_HALF_SPREAD_BPS_CONSERVATIVE` (100 bps) is kept
  as the conservative bound used when the data estimate is unavailable.
- `deck_basket_returns(...)` — equal-weight basket of component spread
  returns with a FIXED pre-registered rebalance cadence, cost-aware
  walk-forward (per-name half-spreads from the data; turnover charged on
  rebalance bars only, entry next-bar, no look-ahead).
- `effective_n(name_sets)` — EFFECTIVE N (independent bets), NOT the card
  count: components sharing names correlate. Computed from the name-overlap
  graph as N_eff = (sum w_i)^2 / sum_i sum_j w_i w_j corr_ij with unit
  weights and corr_ij = |names_i ∩ names_j| / |names_i ∪ names_j|
  (Jaccard overlap as the correlation proxy), floored at S6_EFF_N_MIN.
- `multiplicity_corrected_p(p_raw, n_eff)` — deck-stratum multiplicity
  correction over N = all registered deck trials in scope (kills included):
  p_corr = min(1, p_raw * N_stratum) on the RAW stratum count; EFFECTIVE N
  (independent bets) hardens the deck's OWN evidence in `deck_gauntlet`
  (inflated effective variance + shrunk effective track record), never the
  selection-luck ceiling and never a p-value relaxation.
- `deck_gauntlet(...)` — the S6 pass/fail computation: net SR > 0,
  |t_net| >= tau_cal(scope), MinTRL <= T, PBO <= 0.5 at depth, all
  multiplicity-corrected over the deck stratum. Returns the full evidence
  dict that the deck ledger records.

Hard rules (v3 §12 + v4.2 §5): nothing here touches ingestion or the
network; inputs are server-side price panels / name sets supplied by the
caller; public-facing output is survival facts only (this module returns
statistics — the deck copy layer guards what is displayed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sift.gauntlet import config as cfg
from sift.gauntlet import stats


# --------------------------------------------------------------------------
# Realistic per-name half-spreads from the data (replaces the 100 bps placeholder)
# --------------------------------------------------------------------------

def _estimate_half_spread_1d(p: np.ndarray) -> float:
    """Single-name half-spread in bps from one price series (1D path —
    unchanged semantics): k * std(|diff(log p)|, ddof=1) * 1e4, floored at
    S6_HALF_SPREAD_BPS_FLOOR; non-finite entries dropped. Returns NaN when
    there is no usable evidence (too few points or zero dispersion), so the
    caller can fall back to the conservative bound."""
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if p.size < 3:
        return float("nan")
    r = np.abs(np.diff(np.log(p)))
    r = r[np.isfinite(r)]
    if r.size == 0 or float(np.std(r, ddof=1)) == 0.0:
        return float("nan")
    k = 0.25
    return float(np.maximum(k * float(np.std(r, ddof=1)) * 1e4,
                            cfg.S6_HALF_SPREAD_BPS_FLOOR))


def estimate_half_spread_bps(prices: np.ndarray,
                             bars_per_year: int = cfg.BARS_PER_YEAR) -> dict:
    """Per-name half-spread estimate in bps, computed FROM THE DATA.

    Proxy: the quoted half-spread is estimated as a fraction of the name's
    daily log-return dispersion — half_spread_bps ≈ k * sigma_daily * 1e4
    with k = 0.25 (a name's tradable half-spread is a fraction of its daily
    range; conservative vs the old flat 100 bps placeholder for liquid
    names, and it GROWS for illiquid names instead of staying flat).
    Floored at S6_HALF_SPREAD_BPS_FLOOR bps so costs are never assumed
    away. Returns per-name bps array + median (the deck-level cost input)
    + the conservative bound for comparison.

    Input may be a 1D single-name series or a 2D (T, K) per-name panel
    (bars x names, one column per name — e.g. DeckService.run_s6's
    `prices`). 2D panels are estimated PER NAME (column-wise): diff(log)
    is taken within each column so the step across a symbol boundary is
    never treated as a return. The deck-level figure is the MEDIAN of the
    per-name estimates. Non-finite entries are dropped per column; a name
    with no usable evidence falls back to the conservative bound for that
    name only. 1D input behaves exactly as before (single-element
    per_name_bps).
    """
    p = np.asarray(prices, float)
    if p.ndim == 2:
        raw = np.array([_estimate_half_spread_1d(p[:, j]) for j in range(p.shape[1])])
        if not np.any(np.isfinite(raw)):
            # no name has usable dispersion evidence -> conservative bound,
            # never assume zero cost
            med = cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE
            return {"per_name_bps": np.array([med]), "median_bps": med,
                    "conservative_bound_bps": cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE,
                    "source": "conservative_bound"}
        # a name with no usable evidence falls back to the conservative
        # bound for that name only
        per_name = np.where(np.isfinite(raw), raw,
                            cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE)
        med = float(np.median(per_name))
        return {"per_name_bps": per_name, "median_bps": med,
                "conservative_bound_bps": cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE,
                "source": "data_estimate"}
    # 1D path — unchanged
    v = _estimate_half_spread_1d(p)
    if not np.isfinite(v):
        med = cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE
        return {"per_name_bps": np.array([med]), "median_bps": med,
                "conservative_bound_bps": cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE,
                "source": "conservative_bound"}
    med = v
    return {"per_name_bps": np.array([med]), "median_bps": med,
            "conservative_bound_bps": cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE,
            "source": "data_estimate"}


def effective_half_spread_bps(estimate: dict) -> float:
    """The cost actually charged: the data estimate, but never below the
    floor; the 100 bps placeholder remains available as the conservative
    bound (callers may run both and report the stricter). Costs are never
    understated: max(data_estimate, floor)."""
    med = float(estimate.get("median_bps", cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE))
    return max(med, cfg.S6_HALF_SPREAD_BPS_FLOOR)


# --------------------------------------------------------------------------
# Equal-weight basket, fixed rebalance cadence, cost-aware walk-forward
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckCostModel:
    """S6 cost model: per-name half-spreads from the data (dict name->bps
    or a scalar median), conservative bound kept for reporting."""
    half_spread_bps: float            # effective (data) half-spread, bps
    conservative_bound_bps: float = cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE
    per_name_bps: dict | None = None  # optional per-name mapping

    def half_spread_for(self, name: str) -> float:
        if self.per_name_bps is not None and name in self.per_name_bps:
            return max(float(self.per_name_bps[name]),
                       cfg.S6_HALF_SPREAD_BPS_FLOOR)
        return max(self.half_spread_bps, cfg.S6_HALF_SPREAD_BPS_FLOOR)


def deck_basket_returns(
    component_returns: np.ndarray,        # (T, K) per-bar component (spread) returns
    *,
    rebalance_cadence: int = cfg.S6_REBALANCE_CADENCE,
    cost_model: DeckCostModel | None = None,
    formation: int = cfg.S6_FORMATION,
) -> dict:
    """S6 equal-weight basket with a FIXED pre-registered rebalance cadence.

    - At each rebalance bar t (t % cadence == 0, after the formation window)
      the basket is reset to equal weight 1/K across the K components, at the
      CLOSE of bar t (turnover cost charged on bar t; the new equal weights
      are what bar t+1 opens with).
    - Between rebalances weights drift with the components' returns.
    - Costs are charged on the rebalance bar on the turnover actually
      traded: sum_i |w_i_new - w_i_drifted| * half_spread_i / 1e4 (per-name
      realistic half-spreads; the 100 bps placeholder stays as the
      conservative bound reported alongside).
    - Entry is next-bar, NO LOOK-AHEAD (soundness #3): bar t earns
      gross[t] = dot(w_t, R[t]) where w_t was drifted with returns only
      through t-1 (the rebalance at the close of a bar does not let that
      bar's own return tilt its weights); the drift with R[t] happens
      AFTER gross[t] is computed. Weights applied to a bar are never a
      function of that bar's return. weights_path[t] stores the weights
      that EARNED bar t (pre-rebalance on a rebalance bar).

    Returns per-bar gross/net basket returns, the weight path, turnover,
    and the total cost in bps.
    """
    R = np.asarray(component_returns, float)
    if R.ndim != 2:
        raise ValueError("component_returns must be (T, K)")
    T, K = R.shape
    if K == 0:
        raise ValueError("empty deck")
    if cost_model is None:
        cost_model = DeckCostModel(half_spread_bps=cfg.S6_HALF_SPREAD_BPS_CONSERVATIVE)
    if rebalance_cadence < 1:
        raise ValueError("rebalance_cadence must be >= 1")

    names = [f"c{i}" for i in range(K)]  # names resolved by the caller; cost lookup is positional
    gross = np.zeros(T)
    net = np.zeros(T)
    w = np.full(K, 1.0 / K)
    weights_path = np.zeros((T, K))
    turnover = np.zeros(T)
    cost_bps = np.zeros(T)
    last_rebal = -1
    for t in range(T):
        # NO LOOK-AHEAD (soundness #3): bar t earns the weights as they stand
        # at the START of bar t — i.e. drifted with returns only through
        # t-1. The previous code drifted w with R[t+1] at the end of bar t
        # and then applied the drifted w to R[t+1], so the weights applied to
        # a bar were tilted by that bar's own return (probe: R[2]=[0.10,-0.05],
        # rebalance at t=1 -> gross[2]=0.03049 instead of the buy-and-hold
        # 0.025). gross[t] is computed BEFORE this bar's drift.
        gross[t] = float(np.dot(w, R[t])) if t > 0 else 0.0
        net[t] = gross[t] - cost_bps[t] / 1e4
        weights_path[t] = w
        # rebalance at the CLOSE of bar t: turnover cost is charged on bar t,
        # and the new equal weights take effect from bar t+1 (weights
        # entering bar t+1 reflect returns only through t — still no
        # look-ahead).
        if t >= formation and (t - formation) % rebalance_cadence == 0:
            new_w = np.full(K, 1.0 / K)
            to = float(np.sum(np.abs(new_w - w)))
            # per-name half-spread: positional lookup when per_name given
            if cost_model.per_name_bps is not None:
                hs = np.array([cost_model.half_spread_for(names[i]) for i in range(K)])
            else:
                hs = np.full(K, cost_model.half_spread_for("c0"))
            cost = float(np.sum(np.abs(new_w - w) * hs / 1e4))
            turnover[t] = to
            cost_bps[t] = cost * 1e4
            w = new_w
            last_rebal = t
        # drift: w_i <- w_i * (1 + r_i) / sum_j w_j (1 + r_j), using the
        # CURRENT bar's return AFTER gross[t] was computed
        if t + 1 < T:
            w = w * (1.0 + R[t])
            s = w.sum()
            if s > 0:
                w = w / s
            else:
                w = np.full(K, 1.0 / K)
    # post-pass: bars strictly between rebalances must satisfy the
    # equal-weight buy-and-hold identity (weights drift only with past
    # returns); asserted in tests/decks/test_s6_soundness.py
    return {"gross": gross, "net": net, "weights": weights_path,
            "turnover": turnover, "cost_bps": cost_bps, "T": T, "K": K,
            "rebalance_cadence": rebalance_cadence,
            "total_cost_bps": float(cost_bps.sum()),
            "conservative_bound_bps": cost_model.conservative_bound_bps}


# --------------------------------------------------------------------------
# Effective N (independent bets) — components sharing names correlate
# --------------------------------------------------------------------------

def effective_n(name_sets: list[set] | list[frozenset]) -> dict:
    """EFFECTIVE N = independent bets, NOT the card count.

    Components sharing names correlate: 50 mixed cards are NOT 50
    independent bets (spec v4.2 §4). With unit weights,
        N_eff = (Σ w_i)^2 / Σ_i Σ_j w_i w_j ρ_ij,
    with ρ_ij = Jaccard(names_i, names_j) = |i∩j| / |i∪j| (1 iff identical
    name sets, 0 iff disjoint) as the correlation proxy. N_eff = K when all
    components are disjoint, and -> 1 as all components share the same
    names. Floored at S6_EFF_N_MIN.

    Returns {"K": card count, "n_eff": effective independent bets, "overlap":
    mean pairwise Jaccard, "pairwise": matrix}.
    """
    sets = [frozenset(s) for s in name_sets]
    K = len(sets)
    if K == 0:
        return {"K": 0, "n_eff": 0.0, "overlap": 0.0, "pairwise": np.zeros((0, 0))}
    rho = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                rho[i, j] = 1.0
            else:
                union = len(sets[i] | sets[j])
                rho[i, j] = (len(sets[i] & sets[j]) / union) if union > 0 else 0.0
    denom = float(np.sum(rho))           # Σ_i Σ_j ρ_ij  (w_i = 1)
    n_eff = (float(K) ** 2) / denom if denom > 0 else float(K)
    n_eff = max(n_eff, cfg.S6_EFF_N_MIN)
    off = rho[~np.eye(K, dtype=bool)]
    return {"K": K, "n_eff": float(n_eff),
            "overlap": float(off.mean()) if off.size else 0.0,
            "pairwise": rho}


# --------------------------------------------------------------------------
# Multiplicity correction over the deck stratum
# --------------------------------------------------------------------------

def multiplicity_corrected_p(p_raw: float, n_stratum: int, n_eff: float) -> dict:
    """Deck-stratum multiplicity correction.

    N = all registered deck trials in the scope across ALL users, kills
    included (raw stratum count from the deck ledger). The selection-luck
    deflation scales with the RAW stratum N (how many deck trials were
    tried), and n_eff NEVER relaxes it:

        p_corr = min(1, p_raw * N_stratum)          (Bonferroni-style bound)

    n_eff (independent bets) enters only through the WITHIN-DECK
    correlation: correlated components share names, so the deck's own
    evidence is worth n_eff bets, not K. That makes the deck-side bar
    HARDER when components correlate (less diversification), never easier —
    see ``deck_gauntlet``'s ``v_eff`` term (soundness #7: the previous code
    fed n_eff into ``expected_max_sharpe`` as the luck-ceiling N, which
    made LOW n_eff an EASIER bar — exactly backwards).

    ``sr_star_eff`` returned here is the raw-stratum selection-luck ceiling
    (expected_max_sharpe(n_stratum, 1/n_stratum)) for reference; it is NOT
    n_eff-scaled.
    """
    p = float(p_raw)
    n_stratum = max(int(n_stratum), 1)
    n_eff = max(float(n_eff), cfg.S6_EFF_N_MIN)
    p_corr = min(1.0, p * n_stratum)
    sr_star_eff = stats.expected_max_sharpe(n_stratum, 1.0 / n_stratum)
    return {"p_raw": p, "p_corr": p_corr, "n_stratum": n_stratum,
            "n_eff": n_eff, "sr_star_eff": sr_star_eff}


# --------------------------------------------------------------------------
# S6 gauntlet verdict
# --------------------------------------------------------------------------

def deck_gauntlet(
    net_returns: np.ndarray,
    *,
    scope_id: str,
    p_perm: float | None = None,
    n_stratum: int = 1,
    v_stratum: float = 0.0,
    n_eff: float = 1.0,
    pbo: float | None = 0.0,
    n_cards: int | None = None,
    bars_per_year: int = cfg.BARS_PER_YEAR,
) -> dict:
    """S6 pass/fail on a deck's net basket returns.

    Pass bar (pre-registered, never weakened): net SR > 0,
    |t_net| >= tau_cal(scope), MinTRL <= T, PBO <= 0.5 at depth —
    multiplicity-corrected over the deck stratum (N = n_stratum, all
    registered deck trials in scope across all users, kills included;
    n_eff = effective independent bets for the within-deck correlation).

    The multiplicity correction enters through the DSR denominator:
    sr_star uses N = n_stratum (raw trials). EFFECTIVE N enters the
    BASKET's own effective variance, never the luck ceiling (soundness
    #7): correlated components (low n_eff) share names and returns, so the
    basket's evidence is worth n_eff independent bets, not K. We deflate
    the stratum variance by the diversification ratio K/n_eff when
    computing the within-deck benchmark,
        v_eff = v_stratum * (K / n_eff)   (>= v_stratum, harder when correlated),
    and the effective-N bar is dsr_eff = psr(sr_bar, sr_star, T_eff, ...)
    with the track record shrunk to the effective sample size
        T_eff = T * n_eff / K             (<= T, harder when correlated),
    both with the SAME selection-luck ceiling sr_star = expected_max_sharpe(
    n_stratum, v_stratum) as the stratum DSR. A deck with correlated
    components (low n_eff) therefore faces a HARDER bar on both axes —
    '50 mixed cards' cannot launder one lucky pair. The previous code fed
    n_eff into expected_max_sharpe (lower luck ceiling for fewer bets ->
    EASIER bar), which was backwards.
    """
    r = np.asarray(net_returns, float)
    r = r[np.isfinite(r)]
    T = int(r.size)
    sr_ann = stats.sharpe_ratio(r, bars_per_year=bars_per_year) if T >= 2 else 0.0
    t_net = stats.t_stat_sharpe(r) if T >= 2 else 0.0
    skew, kurt = stats.skew_kurt(r) if T >= 4 else (0.0, 3.0)
    sr_bar = sr_ann / math.sqrt(bars_per_year) if T >= 2 else 0.0
    min_trl = stats.min_trl(sr_bar, skew, kurt, alpha=0.05) if sr_bar > 0 else math.inf
    tau = cfg.tau_cal(scope_id)
    T_scope = cfg.scope_T(scope_id)

    # DSR with the deck-stratum denominator: N = raw registered deck trials
    # (kills included); the candidate's own stats feed V like S3/S5.
    d = stats.dsr(sr_bar, max(T, 2), max(int(n_stratum), 1), v_stratum,
                  skew=skew, kurt=kurt)
    # Effective-N multiplicity (soundness #7, corrected direction): the
    # selection-luck ceiling sr_star stays on the RAW stratum N; n_eff
    # hardens the deck's OWN evidence — inflated effective variance
    # (correlated components diversify less) and a shrunk effective track
    # record. K = card count (defaults to n_eff -> neutral when unknown).
    n_eff = max(float(n_eff), cfg.S6_EFF_N_MIN)
    K_cards = max(float(n_cards), 1.0) if n_cards else n_eff
    div = max(K_cards / n_eff, 1.0)          # >= 1: correlation never helps
    # effective stratum variance for the within-deck benchmark: the stratum
    # V (per-bar SR^2 of recorded deck trials) inflated by the
    # diversification ratio; floored at the PSR normal-theory variance
    # (1 + sr_bar^2/2)/T, same convention as stats.dsr's var_floor (#16).
    v_floor = (1.0 + sr_bar ** 2 / 2.0) / max(T, 2)
    v_eff = max(max(v_stratum, 0.0) * div, v_floor)
    T_eff = max(T * (n_eff / K_cards), 2.0)  # <= T: less independent evidence
    sr_star_eff = stats.expected_max_sharpe(max(int(n_stratum), 1), v_eff)
    dsr_eff = stats.psr(sr_bar, sr_star_eff, max(int(round(T_eff)), 2), skew, kurt)

    p_corr = None
    if p_perm is not None:
        p_corr = multiplicity_corrected_p(p_perm, n_stratum, n_eff)["p_corr"]

    checks = {
        "net_sr_pos": bool(sr_ann > cfg.S6_NET_SR_MIN),
        "t_net": bool(abs(t_net) >= tau),
        "min_trl": bool(min_trl <= T_scope),
        "pbo": bool(pbo is not None and pbo <= cfg.S6_PBO_MAX),
        "dsr_stratum": bool(d["dsr"] >= cfg.S3_DSR_MIN),
        "dsr_eff_n": bool(dsr_eff >= cfg.S3_DSR_MIN),
        "p_perm_corr": (True if p_corr is None else bool(p_corr <= cfg.S2_P_MAX)),
    }
    passed = all(checks.values())
    return {
        "pass": passed, "checks": checks,
        "sr_ann": sr_ann, "t_net": t_net, "tau": tau,
        "min_trl": min_trl, "T": T, "T_scope": T_scope,
        "pbo": pbo, "skew": skew, "kurt": kurt,
        "dsr": d["dsr"], "sr_star": d["sr_star"],
        "dsr_eff_n": dsr_eff, "sr_star_eff_n": sr_star_eff,
        "v_eff": v_eff, "T_eff": T_eff, "diversification": div,
        "n_stratum": int(n_stratum), "n_eff": n_eff,
        "p_perm": p_perm, "p_perm_corr": p_corr,
        "scope_id": scope_id,
    }


__all__ = [
    "DeckCostModel",
    "deck_basket_returns",
    "deck_gauntlet",
    "effective_half_spread_bps",
    "effective_n",
    "estimate_half_spread_bps",
    "multiplicity_corrected_p",
]
