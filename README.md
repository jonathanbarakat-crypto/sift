# SIFT

SIFT mines cross-stock relationship patterns server-side and tests them against
a staged, pre-registered statistical gauntlet. **SIFT does not trade** — it
produces signals only; users execute with their own methods.

All mining runs server-side. Clients receive derived outputs only — never raw
market data. Every trial is recorded in an append-only, scope-stratified trial
registry (the honesty ledger), which is the denominator for all
multiplicity-corrected statistics.

## Layout

- `sift/engine/` — server-side mining engine (sessions, seeded runs)
- `sift/gauntlet/` — staged gauntlet S0–S5 statistics and stage runner
- `sift/registry/` — append-only trial registry ledger (SQLite now,
  Postgres-compatible schema)
- `sift/data/` — data ingestion, versioned warehouse, PIT universe
- `sift/web/` — public pages (survival facts only)

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/
```

## Non-negotiables (scope doc v3 §12)

1. Signals, never execution.
2. No raw data reaches any user device — derived outputs only.
3. No tickers + directional calls in any SIFT-published content.
4. No win rates, no P&L leaderboards, no picks.
5. Registry-depth validation (S5) is automatic — never opt-in.
6. Scope is a first-class field on every trial; DSR denominators stratify by
   scope and are never pooled across scopes.
7. Public cards show survival facts only.
