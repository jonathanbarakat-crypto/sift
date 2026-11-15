# SIFT

Server-side cross-stock pattern mining with a staged statistical gauntlet
(scope doc v3, jonathanbarakat-crypto/sift-research `docs/sift-scope-v3`).

- All mining runs **server-side**; clients receive derived outputs only —
  never raw market data (v3 §4/§12.2).
- **SIFT does not trade** — signals only, never execution (§12.1).
- Gauntlet stages S0–S4 are implemented in `sift/gauntlet` (S5 registry-depth
  validation is a separate build task and is never opt-in).
- The trial ledger is scope-stratified: DSR denominators never pool across
  scopes; public cards show survival facts only.

## Layout

- `sift/gauntlet/stats.py` — pure statistics (S1 screen t, S2 block-permutation
  null p=(1+#extreme)/(B+1) with block ⌈T^(1/3)⌉ and PCG64 seed, S3 DSR/PSR/
  MinTRL/BH-FDR, S4 walk-forward net Sharpe).
- `sift/gauntlet/runner.py` — stage runner wiring stats into pre-registered
  trials and recording stage rows into the ledger (`Ledger` protocol +
  reference `SQLiteLedger`; binds to the task-1/8 registry when merged).
- `tests/` — unit tests + planted fixtures F-PAIR (AR(1) cointegrated pair,
  must pass) and F-NULL (200 independent random walks, must fail S2/S3).

## Test

```
pip install -r requirements.txt
python -m pytest tests/
```
