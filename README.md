# SIFT

Server-side cross-stock pattern mining with a pre-registered statistical
gauntlet. **SIFT does not trade** — signals only, never execution.

Hard rules (v3 scope doc §12): server-side mining only; no raw data to any
client; registry-depth validation (S5) is automatic and never opt-in;
DSR denominators are scope-stratified; public cards show survival facts only.

## Layout

- `sift/registry/` — append-only trial ledger (task 1/8 schema)
- `sift/gauntlet/` — stages S0-S4: `stats.py` (pure stat functions),
  `config.py` (pre-registered thresholds), `runner.py` (stage wiring + ledger)
- `sift/engine/`, `sift/data/`, `sift/web/` — scaffolded by task 1/8

## Tests

```
python -m pytest tests/
```
