# SIFT

SIFT mines cross-stock relationship patterns **server-side** and tests them
against a pre-registered, multiplicity-aware statistical gauntlet.
**SIFT does not trade.** No raw market data is ever served to clients —
browsers receive derived outputs only (v3 scope doc §3, §4.1, §12).

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
# deps: pandas, pyarrow, requests, pytest
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
