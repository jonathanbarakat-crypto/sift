"""SIFT simulated-user harness — infrastructure validation ONLY.

SIMULATED — infrastructure validation, not user-behavior evidence.

Every report this package emits carries the SIM_HEADER banner (see
`header.py`): simulated traffic says nothing about how real users behave
(spec §8) and must never be cited as user-behavior evidence.

Determinism: seeded RNG (PCG64) + a fixed injected clock — the same seed
produces the same report (see `clock.py`, `report.py`). Simulated users are
synthetic: synthetic handles, synthetic instruments (never real tickers), a
test DB only (`SIFT_ENV=test`), never production data.

Hard rules honored here (v3 §12 + v4.1):
- sims run against a test DB / ephemeral env only (`SIFT_ENV=test`);
- no real tickers / raw holdings in sim fixtures — synthetic instruments only;
- card-layer sims are behind `cards_v4_1` and fail closed when off
  (kill-switch `SIFT_CARDS_V4_1=0` respected — see tests/sim/test_killswitch.py);
- no tickers / directional calls / P&L / leaderboards / profit-expectation or
  "value" language anywhere in this package or its reports.
"""

SIMULATED = True  # marker: this package is a simulator, never evidence

__all__ = ["SIMULATED"]
