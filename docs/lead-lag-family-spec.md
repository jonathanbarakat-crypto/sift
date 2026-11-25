# SIFT — Build-Reference Spec: Cross-Stock Lead–Lag (`lead_lag`)

Canonical pre-registration spec: `sift-research/main:docs/lead-lag-family-spec.md`
(owner decisions locked 2026-11-25). Cross-family mining modes policy:
`sift-research/main:docs/mining-modes-policy.md`. Base cards/decks & legal rules:
`docs/cards-decks-spec-v4.2.md` (§5, §5.1).

## 1. Locked Parameters & Scope Isolation

- **Family ID:** `lead_lag` (directed ordered pair `leader -> follower`, distinct from reverse).
- **Pre-registered parameter grid:**
  - `lag_bars` (`W`) ∈ `{1, 2, 5}`; `hold_bars` (`H`) = `lag_bars` in v1.
  - `formation` ∈ `{60, 120, 252}` (default `120`).
  - `open_z` ∈ `{1.0, 1.5, 2.0}` (default `1.5`, symmetric two-sided `|z| >= open_z`).
  - `step = 120` (varied only inside the 9-point pre-registered `S5_VARIANT_GRID_LEAD_LAG`).
- **Mining modes (zero hidden release pre-screening):** Direct (`A -> B`), Anchored
  (`A -> ?` or `? -> B`, seeded `PCG64` uniform draw, `anchor_id_hmac` recorded),
  Uniform Random (`? -> ?`, seeded `PCG64` uniform draw).
- **Multiplicity stratum isolation:** `stratum_stats`, `stratum_epoch`, DSR, and BH-FDR
  are strictly keyed by `(family, scope_id)` (`sift/registry/__init__.py`,
  `sift/gauntlet/runner.py`). `lead_lag` never pools into `pairs` strata.

## 2. Statistical Controls & Gauntlet Gates

- **Confounder controls (strictly inside each window, zero look-ahead):**
  1. Cross-sectional demeaning against equal-weight active PIT mean return `r_M,t`.
  2. Leader AR(1) pre-whitening on non-overlapping lag `R_A,t-W^(W)` → innovation `x_t`.
  3. Follower own-lag control `b_t = R_B,t^(W)` in `y_{t+1}^(H) = alpha + gamma * x_t + phi * b_t + eps`.
  4. In-window Berger (1982) Intersection-Union Test combining causal Granger prediction (`y_pred`) and contemporaneous-orthogonalized structural transmission (`y_struct = y_pred - beta0_hat * R_{A,t+1..t+H}^(H)`) strictly inside estimation/formation windows (OOS remains 100% causal), eliminating both `F-LEADLAG-CONTEMP` and reverse-direction collider bias without multiplicity inflation.
- **S0:** 5 pre-registered synthetic fixtures (`F-LEADLAG`, `F-LEADLAG-CONTEMP`,
  `F-LEADLAG-REGIME`, `F-LEADLAG-DELIST`, `F-LEADLAG-NULL`) + real-data finite-bar,
  non-degenerate variance (`sigma > 1e-8`), and stale-quote (`< 0.25`) sanity gate.
- **S1:** Newey–West HAC `|t_HAC(gamma_hat)| >= 2.0` (`lag = max(H - 1, nw_lag(T))`).
- **S2:** Block-permutation null (`B = 1000`, `PCG64`, block `b = max(5, ceil(T^(1/3)))`)
  permuting pre-whitened leader innovation `x_t` while holding follower `y` and `b_t` fixed;
  pass iff `p_perm <= 0.05`.
- **S4 (scored, market-neutralized follower return gating):** Non-overlapping OOS walk-forward;
  primary return `p_k * r_tilde_{B,k+1}` with follower Roll half-spread (`>= 1.0` bps floor,
  `100.0` bps conservative fallback) + `1.0` bps market-hedge leg floor; HAC `t_stat_sharpe_hac`
  and `T_eff = effective_sample_size` (`net SR > 0`, `|t_net| >= tau_cal(scope)`,
  `MinTRL <= T_eff`).
- **S3:** Mint-time admission gate `DSR >= 0.95` + `BH-FDR q = 0.10` strictly within
  `("lead_lag", scope_id)` using frozen A.3 `n_eff`.
- **S5 (automatic, full strength):** Re-runs S2, S4 (HAC), CSCV PBO (`S=16` over 9-point
  `S5_VARIANT_GRID_LEAD_LAG`), and S3 at `registry_depth` (`T = 4032`, delist-masked,
  `|t_net| >= 3.0`, `PBO <= 0.5`).

## 3. Compliance & Kill-Switch

- Public/general APIs, card faces, PNGs, SSE streams, telemetry, and off-platform alerts
  carry **zero tickers, zero slopes (`gamma_hat`), zero signs (`sgn(gamma_hat)`,
  `sgn(z_k)`), and zero P&L** — magnitude-only `current_abs_z` (`|z_k|`).
- Owner-only Private Spec (`v4.2 §5.1`) exposes only `[{"role": "leader", "symbol": ...},
  {"role": "follower", "symbol": ...}]`, non-price parameters, survival facts, and
  magnitude-only `|z_k|` — never `gamma_hat`, never sign/direction, never raw prices.
- Feature flag: `family_lead_lag`, kill-switch env `SIFT_FAMILY_LEAD_LAG=0` (fail-closed).
