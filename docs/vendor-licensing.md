# Vendor licensing: EODHD terms and how SIFT stays inside them

Source: EODHD Terms & Conditions, https://eodhd.com/financial-apis/terms-conditions
(fetched and read 2026-11-17). Pricing: https://eodhd.com/pricing.
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
  included in the personal EOD plans (Sandbox free, Historian €19.99/mo, etc.).
- EODHD disclaims accuracy and states the data is "not appropriate for trading
  purposes" — consistent with our "SIFT does not trade" positioning.
- Subscriptions are non-refundable; minimum one month.

## How SIFT stays inside the limits (current design)

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
5. **Server-side mining is private analysis** — allowed on personal plans.

## When we would need the B2B license

If the product ever **displays EODHD-derived raw data to users** — e.g. price
charts on discovery pages, live quote widgets, per-ticker data in the UI — that
is redistribution/display and requires the Commercial/B2B plan. Until then,
the personal EOD tier (Historian, 30+ yrs, 100k calls/day) covers server-side
ingest + mining.

## Other vendors

- **yfinance / Yahoo scraping**: not authorized by Yahoo ToS. Research/pre-validation
  use only, never the production feed; no delisted lists (survivorship-bias risk,
  v3 §4.5/§12.8), no SLA. If used for beta research, label data vintage as
  research-grade in the warehouse.
- Any new vendor: record its terms here before wiring its client.
