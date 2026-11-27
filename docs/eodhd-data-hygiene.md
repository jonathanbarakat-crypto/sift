# EODHD Warehouse Data Hygiene — Audit & Fixes Applied

Recorded: **2026-11-27**. Covers the clean paid-plan EODHD rebuild at
`data/warehouse` (`2000-01-01` .. `2026-11-26`, `asof = 2026-11-25`) and the
pipeline/screen fixes applied after inspecting the full 50,960-symbol US
active + delisted archive (`91,756,754` bars).

---

## 1. Background: Why the Prior (`2026-11-20`) Warehouse Was Deleted & Rebuilt

An earlier evaluation warehouse built on `2026-11-20` (commit `9efd197`, before
v4.3 and PRs #66, #69, #70) failed current spec compliance on five grounds:
- **0 / 33,453 delisted US common stocks ingested** (active-only, survivorship-biased
  despite claiming `delist_idx_available: true`).
- **0 `/api/splits/` files cached**; stored `dollar_volume` used raw `close * volume`.
- **Ad-hoc `manifest.json`** missing `top_path` and `data_quality_screen`, causing
  `sift.web.mining._resolve_warehouse_universe()` to fail closed (`None`) in both
  `production` and `development`.
- **Non-compliant universe parquets** missing 5 of 11 `RANKED_COLUMNS` and lacking
  `quarantined_<asof>.parquet` (ranking corrupt `PROSY.US` #1 at \$102.6B/day).
- **Stale `VINTAGE` hash** mismatched against `warehouse_vintage_hash()`.

The old warehouse was removed and rebuilt cleanly from scratch under `--plan paid`
across all `17,505` active + `33,455` delisted = **50,960** unique US common stocks
(`49,890` non-empty EOD parquets + `49,890` `/api/splits/` files, `1,070` empty
delisted series, zero API errors).

---

## 2. Vendor Data Corruptions & Pipeline Bugs Identified and Fixed

Forensic inspection of the rebuilt Top-1,000 uncovered **6 systemic failure classes**
(majority upstream EODHD vendor defects; two bugs in our own handling). All fixes
were implemented with general, zero-hardcoding rules:

| # | Class | Root Cause (Vendor vs Ours) | Concrete Examples Observed | Fix Applied |
|---|---|---|---|---|
| 1 | **Foreign-exchange feeds leaked into US delisted list** | **Vendor:** Non-US stocks quoted in GBX pence / RUB / BRL / EUR / SEK leaked into `US-delisted.json` labeled `Country: USA`, `Currency: USD`, `Exchange: NYSE/NASDAQ/US`. Also multi-year ticker-reuse gaps stitched across 15–19 years. | `HSBA.US` (LSE in GBX pence, \$27.1B/day, missed 11/11 UK Boxing Days), `ISF-CL` (LSE, missed 13/13 Dec 26), `PLZL`/`NLMK`/`FEES` (MOEX in RUB, missed 100% May 1), `ASMI`, `CMIG3`, `EOAN`, `UCG`, `LUNE`, `BDO`, `VPK`, `VPS`; ticker-reuse holes `WORK`, `SLW`, `BFT`. | Added `foreign_calendar_mismatch` in `data/dollar_volume.py` (`_check_us_foreign_calendar`): flags US-exchange symbols missing or zero-volume on `>= 80%` of US trading sessions falling on major non-US holidays (Tue–Fri Dec 26 Boxing Day or Mon–Fri May 1 Labour Day) across `>= 3` occurrences. |
| 2 | **Unexecuted future announced splits applied retroactively** | **Ours:** EODHD `/api/splits/` returns announced upcoming splits (`split_date > last_bar_date`) whose bars/volume are not yet restated; `cumulative_split_factors()` applied them anyway. | `LAB.US` (`2026-11-29` `1:35` -> inflated 35x to \$53.7B peak), `CAN.US` (`2026-12-04` `1:25` -> inflated 25x to \$32.8B peak). | Added `_filter_executed_splits()` in `data/dollar_volume.py` to ignore splits strictly after `vintage_end` (`max(full_df["date"])`). `LAB` dropped to \$5.0M/day (rank `#5309`); `CAN` to \$33.5M/day (`#1871`). |
| 3 | **Duplicate zero-gap splits & inverted split ex-dates** | **Vendor + Ours:** EODHD recorded duplicate splits with 0 bars between them or applied splits backwards; `check_split_volume_consistency()` skipped zero-gap splits and had no ex-date price continuity check. | `JDSU.US` (two `1:8` splits on `2006-10-17` & `2006-11-14` with 0 bars between -> fake `-88%` crash in `adjusted_close` and \$67.1B DV), `RHT.US` (`2000-01-10` 2:1 inverted -> `+327%` fake jump), `AMCC.US` (`3.8x` jump), `RNAI.US` (`3.8x` jump). | Added `_group_split_events()` (merges consecutive zero-bar-gap splits) and `check_split_price_continuity()` (`split_price_mismatch`, `MAX_SPLIT_EXDATE_PRICE_MULT = 2.5`) in `data/dollar_volume.py`. |
| 4 | **Unadjusted & partially-restated reverse-split volumes** | **Vendor + Ours:** EODHD left pre-split volume unadjusted or only adjusted the final ~20 bars before a large reverse split; prior screen had an upper cap (`unadj_score <= 1.40`) and only inspected +/-20 bars. | `HPR.US` (`1:50`, `score=0.55`), `AMCC.US` (`score=1.54`), `CXRXF.US` (`1:300`, volume adjusted only on final ~20 bars while 2015–2017 stayed unadjusted at \$80.1B/day), `USU`, `BTUUQ`, `SPN`, `ASNA`, `VVUS`, `TTPH`. | Hardened `check_split_volume_consistency()` in `data/dollar_volume.py`: dropped upper cap on reverse-split `unadj_score >= 0.50` (`log_jump <= -log(2.5)`), and added full pre-vs-post segment check for large reverse splits (`r <= 0.10`, `> 15x` median DV shift). |
| 5 | **Pre-2018 unrecorded delisted reverse-split & print spikes** | **Vendor:** Dot-com and pre-2018 delisted tickers with `splits = []`, reverse-split-multiplied `close == adjusted_close`, and unadjusted volume printing \$30B–\$94B single days (flat \$250B cap was too loose for 2000–2017). | `FCOM1` (\$52.6B), `EXAP` (\$77.9B), `FIBR1` (\$68.7B), `FIRE1` (\$94.2B), `GMGC` (\$39.6B), `DVINA` (\$36.3B), `NAVI1` (\$31.7B), `AMSI` (\$49.7B), `BLUE1` (\$41.4B), `SOBI` (\$75.5B), `CMPC` (\$52.2B), `BBRY` (\$48.8B), `IAC`/`PPLI`/`MTCH` (`2003-06-09` 4.35B-share glitch). | Replaced flat cap with era-aware `_era_dollar_volume_caps()` in `data/dollar_volume.py`: **\$25B** (`< 2010`), **\$45B** (`2010–2017`), **\$250B** (`2018+`). Zero false positives on real US mega-caps (`AAPL`, `MSFT`, `TSLA`, `NVDA`, `BRK-A`, `AMC`, `SPCX`, `SNDK`). |
| 6 | **Non-equities tagged `Common Stock` + operational OOM/midnight bugs** | **Vendor + Ours:** EODHD tagged Eurex/CBOT futures, warrants, preferred series, ETFs, and when-issued lines as `Common Stock` (often under `Exchange == "US"`). `data/universe.py` concatenated all 91.8M rows in RAM (OOM on 4 GB VM); `EodhdClient` didn't reset `calls_made` across UTC midnight. | `BAC-WS-A`, `GM-WS-B`, `STI-WS-A/B`, `ORCL-PD`, `VYLR-WI`, `FDXF-WI`, `FGBLZ26`, `FMWOM26`, `FXXPM26`, `TYM1`, `KLS`, `PHP`, `BRSC`, `MUZ`. | Hardened `data/security_type.py` (all dash/dot suffix segments, `-WI`, `-P[A-Z]`, futures regex, plus `is_common_stock_metadata()` filtering `Exchange == "US"` and non-equity `Name` markers); added streaming `build_ranked_table_from_warehouse()` in `data/universe.py`; fixed `_rollover_if_new_utc_day()` in `data/eodhd_client.py`. |

---

## 3. Files Modified & Test Verification

- `data/security_type.py` — multi-segment suffix checks, preferred series `-P[A-Z]`, `-WI`, futures code regex, `is_common_stock_metadata()`.
- `data/dollar_volume.py` — `_filter_executed_splits()`, `_group_split_events()`, `check_split_price_continuity()`, hardened `check_split_volume_consistency()`, `_era_dollar_volume_caps()`, `_check_us_foreign_calendar()`.
- `data/universe.py` — `load_symbol_metadata()`, streaming `build_ranked_table_from_warehouse()`, updated manifest thresholds, `vintage.json` preservation of `plan`/`vendor`/`window`.
- `data/eodhd_client.py` — `_rollover_if_new_utc_day()` daily call-counter reset.
- `tests/test_dollar_volume_basis.py` — 6 new regression tests covering all fixes above.
- **Test result:** Full repository suite `557 passed, 0 failed`.

---

## 4. Rebuilt Warehouse Metrics (`asof = 2026-11-25`)

- **Vintage hash:** `6ab8e75215966af1...` (`VINTAGE` matches `vintage.json`)
- **Bars scanned:** `91,756,754` across `49,890` non-empty symbols (`45,247` common stocks, `0` missing splits)
- **Non-common excluded:** `4,643` (up from `3,944`)
- **Quarantined:** `23,110` common stocks (`3,914` post-floors) written to `universe/quarantined_2026-11-25.parquet`:
  - `foreign_calendar_mismatch`: `11,749` | `unsplit_price_scale_break`: `11,712` | `sub_dime_raw_price`: `7,973` | `split_volume_mismatch`: `5,432` | `adj_split_divergence`: `5,132` | `split_price_mismatch`: `4,957` | `non_finite_or_non_positive_price`: `3,630` | `eodhd_sentinel_price`: `827` | `implausible_bar_dollar_volume`: `592`
- **Clean rankable universe:** `9,240` symbols (`4,958` delisted retained)
- **Top-1,000 PIT listing:** `751` active + `249` delisted (`TWX`, `SQ`, `TWTR`, `ROKU`, `YHOO`, `AABA`, `KVUE`, `VRTS1`, `ESV`, `VRX`, `FOXA1`, `BRCM`, ...); `BRK-A` preserved (`#133`), `WMT` (`#49`)
- **`PairCatalog` (`_resolve_warehouse_universe`):** passes in both `production` (`require_pit_universe=True`) and `development` (`mode=pit_manifest`, `selected=1000`, `forbidden=50960`, `quarantined_excl=23110`).

---

## 5. Follow-Up Hardening Tracked for PR #2

1. Exclude noisy US OTC tiers (`PINK`, `OTCGREY`, `OTCCE`, `OTCBB`) while retaining `OTCMKTS`/`OTCQX`/`OTCQB`.
2. Add internal dormancy / ticker-reuse gap screen (`> 90` calendar days between consecutive bars).
3. Detect copied pre-spinoff history clusters (`IAC`/`PPLI`/`MTCH`) and block intra-cluster pair formation.
4. Expand US foreign-calendar holidays (Easter Monday, UK Bank Holidays) + single-bar volume print-glitch guard.
5. Add multi-exchange currency/sub-unit normalization registry, `--adv-window-bars` trailing PIT ADV option, independent cross-source audit script, Bulk API `--bulk-update`, and automated post-ingest validation gate.
