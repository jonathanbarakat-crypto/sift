"""Pre-registered gauntlet thresholds (hashed into each trial's config_hash).

Source: FL research/design/2026-11-15-sift-gauntlet-s5-design.md §2-§3 and
SIFT v3 scope doc §8.1. The free 5-yr scope hurdle is the scope-matched
tau_cal = 3.5 (= 3.0 / (f_min * sqrt(T_F/T_scope)) with f_min = 0.5,
sqrt(4032/1260) ~ 1.79); S5's own thresholds stay at full strength and are
never weakened — S5 wiring belongs to task 3/8.
"""

from __future__ import annotations

BARS_PER_YEAR = 252

# Scope table (design doc §1): (scope_id -> window length in bars)
SCOPE_T = {
    "free_5y_top1k": 1260,     # trailing 5 years, ~1260 daily bars
    "pro_full": 4032,          # full history 2010-01-01..2025-12-31
    "registry_depth": 4032,    # S5 validation scope (same data as pro_full)
}

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
S4_PBO_MAX = 0.5              # registry-depth only (task 3/8), kept here for reference

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
}


def tau_cal(scope_id: str) -> float:
    """Scope-matched t hurdle for S4: tau_cal ~ 3.5 for the 5-yr free scope,
    3.0 at full depth (design doc §3.2 table)."""
    return S4_T_NET_5Y if scope_id == "free_5y_top1k" else S4_T_NET_FULL


def scope_T(scope_id: str) -> int:
    return SCOPE_T[scope_id]
