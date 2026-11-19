# Research feeds — yfinance (Yahoo) vs EODHD (production)

**EODHD is the production vendor.** The yfinance/Yahoo feed added by
`data/yf_client.py` + `data/ingest_yf.py` is **RESEARCH-ONLY** and must never
become the production data path.

Source of the constraint: `docs/vendor-licensing.md` (pull/26), which records
the EODHD Terms & Conditions fetched and read 2026-11-17. Yahoo's own terms do
**not** authorize redistribution of Yahoo data in a commercial product, and
the yfinance scraping path has **no SLA**.

| | EODHD (production) | yfinance / Yahoo (research-only) |
|---|---|---|
| ToS redistribution | Commercial/B2B license required for display rights; server-side ingest on personal plan is the current basis | Not authorized for production redistribution |
| Delisted-name lists | Available on paid plans (`delisted=1`) | **None** — active-only, survivorship-biased (v3 §4.5/§12.8) |
| SLA | Paid plans: SLAs via sales | None |
| Depth | Free tier: past 1y EOD; paid: 30+ years | `period="5y"` etc., keyless |
| Cost | 20 calls/day free tier | Free, but rate-limited (429s) |

## How the code enforces research-only

1. **Vintage marking** — `data/ingest_yf.py` writes
   `vendor: "yfinance"` + `research_grade: true` + a `research_*` scope id
   into `vintage.json`/manifest. Downstream trials therefore label the scope
   honestly (e.g. `research_5y_top500`), never `free_1y_top1k` or `pro_full`.
2. **Universe honesty** — `data/universe.py` detects a research vintage and
   marks `delist_idx_available=False` + `survivorship_bias: "BIASED …"` in the
   universe manifest; the ranked table carries `delist_idx_available=False` on
   every row, so the S5 delist-mask logic cannot silently treat research data
   as survivorship-free.
3. **Engine labeling** — `sift/engine/first_run.py --source yf` derives a
   `research_`-prefixed scope id from source+window and records
   `research_grade: true` in the measurements JSON. S5 registry_depth still
   requires the production full-window feed: on a yf 5y vintage, s5 rows stay
   pending/amber exactly as on free EODHD.
4. **Rate-limit friendliness** — Yahoo 429s: `YfClient` retries with
   exponential backoff + full jitter, and `ingest_yf --sleep` pauses between
   tickers; a persistently throttled ticker is recorded as an error and the
   run resumes on the next invocation (per-ticker parquet = resumable).

## Intended use

Initial testing with ~5 years of daily history (verified from the sandbox
2026-11-17: `yf.Ticker('AAPL').history(period='5y')` → 1255 daily bars
2021-11-17..2026-11-17, OHLCV + Dividends + Splits) so S0–S4 can be exercised
on deeper-than-1y data before paying for EODHD full history. S5 stays pending.

Run it server-side only (v3 §12 — raw data never leaves the server, the
warehouse is gitignored):

```
# --top is DEPRECATED: it cut the alphabetically sorted seed list, yielding an
# A/B-only universe (hermes-tasks #182). Ingest the full seed (or a seeded-random
# --max-ingest N cost cap); data/universe.py ranks by point-in-time avg dollar
# volume, excludes warrants/units/rights, and applies quality floors
# (min-bars / min avg dollar volume), then cuts top-N.
python3 -m data.ingest_yf --warehouse data/warehouse --period 5y --seed-list sec --sleep 1.5
python3 -m data.universe --warehouse data/warehouse --n 500
python3 -m sift.engine.first_run --warehouse data/warehouse \
    --universe data/warehouse/universe/top500_pit_<asof>.parquet \
    --source yf --window 5y --max-pairs 2000
```

Any new vendor: record its terms in `docs/vendor-licensing.md` **before**
writing its client.
