"""SIFT gauntlet package — staged statistical admission (v3 SS8).

Stages (FL research/design/2026-11-15-sift-gauntlet-s5-design.md):
  S0 calibration (harness certification)  S1 discovery screen (|t| >= 2.0)
  S2 block-permutation null               S3 DSR + BH-FDR (scope-stratified)
  S4 cost-aware walk-forward              S5 registry-depth validation (task 3/8)

Pure statistics live in `sift.gauntlet.stats`; the ledger protocol and the
stage runner live in `sift.gauntlet.runner`. Hard rules (v3 SS12): server-side
mining only, no raw data to clients, scope-stratified denominators, S5 never
opt-in, public cards show survival facts only, SIFT does not trade.
"""

from sift.gauntlet import stats
from sift.gauntlet.runner import (
    DEFAULT_COST_PARAMS,
    GauntletConfig,
    Ledger,
    SQLiteLedger,
    StageRow,
    Trial,
    run_gauntlet_s0_s4,
    stage_s0_calibrate,
    stage_s1_screen,
    stage_s2_permutation,
    stage_s3_multiplicity,
    stage_s4_walk_forward,
)

__all__ = [
    "stats",
    "DEFAULT_COST_PARAMS",
    "GauntletConfig",
    "Ledger",
    "SQLiteLedger",
    "StageRow",
    "Trial",
    "run_gauntlet_s0_s4",
    "stage_s0_calibrate",
    "stage_s1_screen",
    "stage_s2_permutation",
    "stage_s3_multiplicity",
    "stage_s4_walk_forward",
]
