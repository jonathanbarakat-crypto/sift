"""Pre-registered gauntlet thresholds (hashed into each trial's config_hash).

Source: FL research/design/2026-11-15-sift-gauntlet-s5-design.md §2-§3 and
SIFT v3 scope doc §8.1. The free 5-yr scope hurdle is the scope-matched
tau_cal = 3.5 (= 3.0 / (f_min * sqrt(T_F/T_scope)) with f_min = 0.5,
sqrt(4032/1260) ~ 1.79); S5's own thresholds stay at full strength and are
never weakened (S5 wired in sift/gauntlet/runner.py::run_s5, task 3/8).
"""

from __future__ import annotations

BARS_PER_YEAR = 252

# Scope table (design doc §1): (scope_id -> window length in bars)
SCOPE_T = {
    "free_5y_top1k": 1260,     # trailing 5 years, ~1260 daily bars
    "pro_full": 4032,          # full history 2010-01-01..2025-12-31
    "registry_depth": 4032,    # S5 validation scope (same data as pro_full)
    # RESEARCH-only scope (yfinance/Yahoo 5y feed, task t_bb9ebe51): same ~5y
    # bar count as free_5y_top1k but a DIFFERENT vendor with no delisted list
    # — the research_ prefix makes it impossible to confuse with production
    # scopes; S4's hurdle matches the 5y window.
    "research_5y_top1k": 1260,
}

# S5 registry-depth validation (v3 §8.1-§8.4, design doc §2 S5 / §3.2):
# automatic on every S0-S4 survivor, NEVER opt-in, thresholds at full strength.
# Full-depth t hurdle is 3.0 (Harvey-Liu-Zhu) vs the free scope's tau_cal=3.5;
# survivors that fail full-depth become on-watch-amber, not validated-green.
S5_SCOPE = "registry_depth"
S5_T_FULL = 3.0             # = S4_T_NET_FULL, full-depth hurdle — S5's own bar, never weakened (design doc §3.2 rule 2)
S5_PBO_MAX = 0.5            # = S4_PBO_MAX, PBO <= 0.5 at registry depth (CSCV, S=16, over S5_VARIANT_GRID — computed, never hardcoded; None fails closed)

# S5 CSCV variant grid (soundness #2): CSCV over a SINGLE configuration is
# meaningless, so S5 computes PBO over a pre-registered grid of walk-forward
# variants around the registered config (formation / open_z / close_z /
# step). The registered config (60 / 2.0 / 0.5 / 120) is the grid center.
# The grid is fixed here (pre-registered) and hashed by
# `s5_variant_grid_digest()` into the s5 stage detail, so the PBO input is
# auditable and cannot be tuned after seeing the outcome.
S5_CSCV_S = 16              # CSCV blocks (ceil(4032^(1/3)) = 16, matches S2 block convention)
S5_VARIANT_GRID = (
    {"formation": 60,  "open_z": 2.0, "close_z": 0.5, "step": 120},   # registered (center)
    {"formation": 40,  "open_z": 2.0, "close_z": 0.5, "step": 120},
    {"formation": 80,  "open_z": 2.0, "close_z": 0.5, "step": 120},
    {"formation": 60,  "open_z": 1.5, "close_z": 0.5, "step": 120},
    {"formation": 60,  "open_z": 2.5, "close_z": 0.5, "step": 120},
    {"formation": 60,  "open_z": 2.0, "close_z": 0.3, "step": 120},
    {"formation": 60,  "open_z": 2.0, "close_z": 0.8, "step": 120},
    {"formation": 60,  "open_z": 2.0, "close_z": 0.5, "step": 60},
    {"formation": 60,  "open_z": 2.0, "close_z": 0.5, "step": 252},
)

# S1 discovery screen
S1_THRESHOLD = 2.0            # |t| >= 2.0 to proceed

# S2 block-permutation null
S2_B = 1000                   # replicates
S2_P_MAX = 0.05               # p_perm <= 0.05
S2_MIN_BLOCK = 5

# S3 multiplicity
S3_DSR_MIN = 0.95             # DSR >= 0.95 at the registered scope
S3_BH_Q = 0.10                # BH-FDR q at family level

# S4 cost-aware walk-forward
S4_T_NET_5Y = 3.5             # scope-matched tau_cal for the 5-yr free scope
S4_T_NET_FULL = 3.0           # full-depth hurdle (Harvey-Liu-Zhu), used at pro_full
S4_NET_SR_MIN = 0.0           # net SR > 0 after costs
S4_PBO_MAX = 0.5              # PBO <= 0.5 at registry depth (S5, CSCV S=16 over the pre-registered S5_VARIANT_GRID)

# S0 calibration tolerances
S0_BETA_TOL = 0.05
S0_HL_REL_TOL = 0.20

# Cost model (FL configs/backtest.yaml; half_spread 100bps is a flagged placeholder)
COSTS = {
    "half_spread_bps": 100.0,
    "slippage_participation_threshold": 0.05,
    "slippage_bps_per_pct": 5.0,
    "max_participation": 0.10,
    "risk_capital_per_trade": 0.01,
    "max_open_positions": 20,
    # documented participation proxy denominator (no ADV feed server-side):
    # one unit of position = risk_capital_per_trade of capital against a
    # unit-ADV reference (soundness #9)
    "adv_proxy": 1.0,
}

# S4 per-name data-driven half-spread floor (Roll estimator, soundness #9):
# costs are never assumed below this; the 100 bps COSTS placeholder stays as
# the conservative FALLBACK when a name has no usable data evidence.
S4_HALF_SPREAD_BPS_FLOOR = 1.0

# ---------------------------------------------------------------------------
# S6 deck (portfolio) gauntlet — spec v4.2 §4 (hermes-tasks #166)
# ---------------------------------------------------------------------------
# Fixed, enumerable bucket grid (no free-form sizes):
#   card-count {5,10,20,50,100} x optional GICS sector split ("all" +
#   each GICS sector) x family mix (pairs, lead-lag; mixing allowed).
DECK_BUCKET_SIZES = (5, 10, 20, 50, 100)
DECK_SECTOR_ALL = "all"
DECK_FAMILIES = ("pairs", "lead_lag")

# S6 pass bar (pre-registered; same shape as S4/S5, never weakened):
#   net SR > 0, |t_net| >= tau_cal(scope), MinTRL <= T, PBO <= 0.5 at depth —
#   multiplicity-corrected over the deck stratum (N = all registered deck
#   trials in scope across ALL users, kills included).
S6_NET_SR_MIN = 0.0          # net SR > 0 after costs
S6_PBO_MAX = 0.5             # PBO <= 0.5 at depth (CSCV, S=16)
S6_REBALANCE_CADENCE = 21    # fixed rebalance cadence in bars (monthly), pre-registered
S6_FORMATION = 60            # formation window for the per-name hedge/z stats
S6_STEP = 120                # walk-forward step (same convention as S4)

# Cost model for S6: realistic per-name half-spreads computed FROM THE DATA
# (median absolute log-return-scaled quoted spread proxy) replace the 100 bps
# placeholder; 100 bps is kept as the conservative BOUND (costs are never
# understated: effective half-spread = max(data estimate, floor), and the
# 100 bps placeholder stays as an upper safety check in tests).
S6_HALF_SPREAD_BPS_FLOOR = 1.0     # never assume below 1 bps half-spread
S6_HALF_SPREAD_BPS_CONSERVATIVE = 100.0   # the old placeholder, kept as bound

# Effective N (independent bets): components sharing names correlate, so
# N_eff is computed from the name-overlap graph of the deck's components
# (see sift.decks.effective_n), NOT from the card count. 50 mixed cards are
# NOT 50 independent bets (spec v4.2 §4).
S6_EFF_N_MIN = 1.0


def tau_cal(scope_id: str) -> float:
    """Scope-matched t hurdle for S4: tau_cal ~ 3.5 for a 5-yr scope,
    3.0 at full depth (design doc §3.2 table). Research-grade 5y scopes
    (``research_*``) share the 5y hurdle but are never production scopes."""
    if scope_id == "free_5y_top1k" or scope_id.startswith("research_"):
        return S4_T_NET_5Y
    return S4_T_NET_FULL


def s5_variant_grid() -> tuple[dict, ...]:
    """The pre-registered S5 CSCV variant grid (see S5_VARIANT_GRID above).
    Returned as a tuple of plain dicts, stable order — the PBO input for
    S5's CSCV (soundness #2)."""
    return tuple(dict(v) for v in S5_VARIANT_GRID)


def s5_variant_grid_digest() -> str:
    """SHA-256 over the canonical JSON of the variant grid — recorded in
    the s5 stage detail so the exact PBO input is auditable / pre-registered
    (hashed alongside the trial's config_hash)."""
    import hashlib
    import json
    canon = json.dumps([dict(v) for v in S5_VARIANT_GRID],
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def scope_T(scope_id: str) -> int:
    if scope_id in SCOPE_T:
        return SCOPE_T[scope_id]
    # research_* scopes carry their window in the id (research_5y_top500 -> 5y);
    # fall back to the 5y bar count so S5/scope_T never crash on a research id,
    # while the research_ prefix keeps it clearly non-production.
    if scope_id.startswith("research_"):
        return SCOPE_T["research_5y_top1k"]
    return SCOPE_T[scope_id]
