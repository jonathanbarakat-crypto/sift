# SIFT

Server-side cross-stock pattern mining with a pre-registered statistical
gauntlet. **SIFT does not trade** — signals only, never execution.

Hard rules (v3 scope doc §12): server-side mining only; no raw data to any
client; registry-depth validation (S5) is automatic and never opt-in;
DSR denominators are scope-stratified; public cards show survival facts only.

## Layout

- `sift/registry/` — append-only trial ledger (task 1/8 schema)
- `sift/gauntlet/` — stages S0-S5: `stats.py` (pure stat functions),
  `s5.py` (registry-depth validation + delist-safe panel helpers),
  `config.py` (pre-registered thresholds), `runner.py` (stage wiring + ledger)
- `sift/engine/`, `sift/data/`, `sift/web/` — scaffolded by task 1/8

## Gauntlet stages

S0 calibration → S1 discovery screen (|t|≥2.0) → S2 block-permutation null
(PCG64, block ⌈T^1/3⌉, p=(1+#extreme)/(B+1)) → S3 scope-stratified DSR≥0.95 +
BH-FDR → S4 cost-aware walk-forward (|t_net| ≥ tau_cal: 3.5 at the 5-yr free
scope, 3.0 at full depth) → **S5 registry-depth validation**.

S5 (v3 §8.1-§8.4) runs **automatically on every S0-S4 survivor — never
opt-in**: the candidate is re-evaluated at `registry_depth` scope (full
window, full universe incl. later-delisted names) at full-strength thresholds
(t ≥ 3.0 full-depth vs 3.5 free-scope tau_cal, p_perm ≤ 0.05, DSR ≥ 0.95 with
the registry_depth stratum, PBO ≤ 0.5). Pass → `validated-green`; fail →
`on-watch-amber` (never validated-green). Post-delist prices are masked out
everywhere (F-DELIST). Acceptance fixtures: `F-REGIME` (edge only in years 0-7
of 16 — the 5y in-edge window shows t>2, the full window attenuates so S5
kills it) and `F-DELIST` (20% of a universe delists mid-window; the pipeline
must not crash and must not use post-delist prices). See
`tests/gauntlet/test_s5.py`.

## Tests

```
python -m pytest tests/
```
