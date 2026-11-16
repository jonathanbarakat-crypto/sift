# SIFT

Server-side cross-stock pattern mining with a pre-registered statistical
gauntlet. **SIFT does not trade** — signals only, never execution. No raw
market data is ever served to clients — browsers receive derived outputs only
(v3 scope doc §3, §4.1, §12). Every trial is recorded in an append-only,
scope-stratified trial registry (the honesty ledger), which is the
denominator for all multiplicity-corrected statistics.

Hard rules (v3 scope doc §12): server-side mining only; no raw data to any
client; registry-depth validation (S5) is automatic and never opt-in;
DSR denominators are scope-stratified; public cards show survival facts only.

## Layout

- `sift/registry/` — append-only trial ledger (SQLite now, Postgres-compatible
  schema)
- `sift/gauntlet/` — stages S0–S5: `stats.py` (pure stat functions),
  `s5.py` (registry-depth validation + delist-safe panel helpers),
  `config.py` (pre-registered thresholds), `runner.py` (stage wiring + ledger)
- `sift/engine/` — server-side mining engine (first trial run)
- `sift/data/`, `sift/web/` — scaffolded packages (ingest/landing live at
  repo-root `data/` and `web/`)
- `data/` — server-side EODHD ingest layer + versioned warehouse (gate 1)
- `web/` — static landing page + email-only beta waitlist (gate 1)

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/
```

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

## Data pipeline (gate 1)

`data/` is the server-side ingest layer (v3 scope doc §3.2: versioned
warehouse, data vintage = content hash; §4.5/§12.8: survivorship-bias-free
data at every tier):

- `data/eodhd_client.py` — EODHD free-tier client. Hard limits encoded:
  20 API calls/day, past-1-year EOD only, live quotes included. Token comes
  from `--api-token` / `$EODHD_TOKEN` / `pass eodhd/api-token` and is never
  printed, logged, or committed. Deep-history (pre-1-year) probes raise a
  clear `DeepHistoryBlocked` error — expected on the free tier, never
  bypassed.
- `data/ingest_eodhd.py` — CLI: fetch exchanges-list once (cached), then
  per-ticker daily EOD for US stocks → one parquet per ticker under
  `data/warehouse/eod/US/`, plus the data-vintage content hash
  (`data/warehouse/VINTAGE`). Resumable: cached parquet covering the
  requested window is skipped without spending a call; one call per ticker
  per run max; `--max-calls` (default 19, under the hard 20/day) is enforced
  before every HTTP request and the daily count is persisted in
  `state/calls.json` so reruns don't re-spend calls.
- `data/universe.py` — CLI: point-in-time top-N (default 1,000 = free-tier
  universe) by average dollar volume over the available window as of
  `--asof`. **No survivorship bias**: names whose series ended before the
  as-of date stay in the ranking/listing with `ended_before_asof` /
  `delisted` flags and their `last_date`.
- `data/warehouse.py` — parquet IO + data-vintage hash (sha256 over all
  warehouse data files).

```bash
python3 -m data.ingest_eodhd --warehouse data/warehouse --max-calls 19
python3 -m data.ingest_eodhd --tickers AAPL MSFT --refresh-live
python3 -m data.ingest_eodhd --exchanges-list-only
python3 -m data.ingest_eodhd --probe-deep-history AAPL   # expected to fail on free tier
python3 -m data.universe --warehouse data/warehouse --n 1000 --asof 2026-11-15
python3 -m pytest tests/ -q
```

`data/warehouse/` (raw vendor data + any tokens) is **gitignored** — this
repo is public; raw data lives only on the server under the internal
commercial license.

## Non-negotiables (scope doc v3 §12)

1. Signals, never execution.
2. No raw data reaches any user device — derived outputs only.
3. No tickers + directional calls in any SIFT-published content.
4. No win rates, no P&L leaderboards, no picks.
5. Registry-depth validation (S5) is automatic — never opt-in.
6. Scope is a first-class field on every trial; DSR denominators stratify by
   scope and are never pooled across scopes.
7. Public cards show survival facts only.
