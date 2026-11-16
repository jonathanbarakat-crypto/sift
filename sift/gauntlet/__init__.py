"""The SIFT gauntlet: pre-registered stages S0-S4 (+ S5, wired by task 3/8).

Stage map (FL research/design/2026-11-15-sift-gauntlet-s5-design.md §2):
  S0 harness calibration · S1 discovery screen · S2 block-permutation null ·
  S3 scope-stratified DSR + BH-FDR · S4 cost-aware walk-forward ·
  S5 registry-depth validation (automatic, never opt-in — out of scope here).
"""

from sift.gauntlet import stats  # noqa: F401  (pure stat functions)
from sift.gauntlet.runner import GauntletRunner, StageResult  # noqa: F401
