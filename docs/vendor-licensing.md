# Vendor licensing: EODHD terms and how SIFT stays inside them

Source: EODHD Terms & Conditions, https://eodhd.com/financial-apis/terms-conditions,
help center (personal-use / commercial / storage / plans), and
https://eodhd.com/pricing — re-fetched and read 2026-11-20.
This file records the constraint; it is not legal advice.

## What the terms say

- **Non-Professional (personal / free) users** may store, manipulate, and analyze
  the data for *private, non-commercial* purposes. They are **prohibited** from:
  sharing account access with others (incl. within groups), and from
  "selling, reselling, retransmitting, **redistributing, displaying**, or
  granting access to the Information or Services, whether in its original or
  repackaged form."
- **Professional User** = any individual/company using the data "as part of a
  business activity, or on behalf of a group, organization, or any other
  individual or business." SIFT is a Professional User once it serves users.
- EODHD sells explicit **Commercial / B2B plans** (Startups / Teams / Business:
  custom volumes, dedicated support, SLAs) via sales@eodhistoricaldata.com.
  Redistribution/display rights for a product are a B2B-license matter, not
  included in the personal plans (Free; **EOD Historical Data — All World**
  $19.99/mo, 30+ yrs, delisted tickers, bulk API, 100k calls/day).
- The ToS disclaims accuracy and states market-maker-sourced prices (CFDs,
  crypto, forex) are indicative and "not appropriate for trading purposes" —
  consistent with our "SIFT does not trade" positioning.
- Subscriptions are non-refundable; minimum one month.

## What the help center adds (2026-11-20) — read this before the beta

- `/help/personal-use`: private research, backtesting, paper trading, and
  **automated private pipelines are explicitly personal use** — allowed on a
  personal plan as long as data *and results* are not provided to anyone else.
- `/help/personal-use` on derived outputs: publishing **derived results**
  (rankings, signals, backtest statistics) to others is NOT automatically
  permitted — "Do not assume that transforming the data, aggregating it or
  making the source values difficult to reconstruct automatically permits
  publication." Route to sales with examples of the outputs and audience.
- `/help/commercial`: any display or distribution of data to others — in apps,
  websites, products, **free / non-profit / invite-only / registered-user
  services** — is commercial use. "The change in use, not the first payment
  from a customer, determines whether a personal plan is sufficient."
- `/help/storage`: downloaded data and derived results may be kept (incl.
  after cancellation) for private non-commercial use; retention supports
  pinning the data vintage.
- Delisted tickers (26k+ US from 2000) come with the EOD plan via the
  Exchanges API (`delisted=1`) — needed for S5 registry depth.

## How SIFT stays inside the limits (current, pre-beta design)

1. **Raw data never leaves the server** (v3 §12.2): ingest writes parquet to the
   server-side warehouse (`data/warehouse/`, gitignored); clients never receive
   raw prices or tickers.
2. **Aggregates only in user-facing artifacts**: card faces and discovery pages
   show anonymized aggregate diagrams + survival stats (v3 §12.3); the
   `_guard_no_tickers` fail-closed check runs on every rendered artifact.
3. **Signals-only, no execution** (v3 §12.1): we never display EODHD-derived
   prices/charts to users, never show buy/sell signals tied to raw quotes in a
   way that redistributes the data. Hit notifications reference the user's own
   saved connection, not vendor data.
4. **One API key per machine/operator** (terms prohibit sharing account access
   within groups): token lives in `pass eodhd/api-token` per host, never printed,
   logged, or committed.
5. **Server-side mining + personal use of results is private analysis** —
   explicitly allowed on personal plans (help/personal-use).

## When we need the B2B license — GATE for the beta cohort

Personal EOD plan covers: ingest, server-side mining, and *your own* use of the
results. It does **not** clearly cover serving cards/signals/derived stats to
other users — even anonymized, aggregated, free, or invite-only (help/commercial
+ help/personal-use derived-outputs clause). Therefore:

- **Beta (n≈10–30) requires written confirmation from sales@eodhistoricaldata.com
  on derived-series display rights (§4.4 licensing email) BEFORE any user
  receives an artifact.** Until that confirmation exists, the beta does not run
  on EODHD data; research-grade vintages stay labeled as such.
- Any display of EODHD-derived raw data to users (price charts, quote widgets,
  per-ticker UI) is redistribution/display and requires the Commercial/B2B plan.

## Other vendors

- **yfinance / Yahoo scraping (RETIRED)**: not authorized by Yahoo ToS; no
  delisted lists (survivorship-bias risk, v3 §4.5/§12.8), no SLA. The
  research-only `data/yf_client.py` / `data/ingest_yf.py` modules and
  `yfinance` dependency have been removed; fail-closed `research_grade`
  guards remain in `data/universe.py` and `sift/engine/first_run.py` for any
  legacy warehouses.
- Any new vendor: record its terms here before wiring its client.
