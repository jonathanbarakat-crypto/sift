"""Regression tests for split-consistent dollar volume and data-quality screens.

Covers:
  * EODHD ``/api/splits/`` parsing and cumulative split-factor math
  * Forward-split invariance (AAPL-shaped 7:1 + 4:1)
  * Zero dividend deflation (high-dividend payers like WMT/T/KO)
  * Strict PIT invariance across post-asof splits and dividends
  * Reverse-split consistent volume (GE 1:8 passes) vs unadjusted-volume bug
    (SOS.US 1:50 / 1:15 quarantined — never ranked #2)
  * Per-bar sentinel detection (even a single 999999.9999 bar among 2,000 bars)
  * Unsplit scale-break & implausible single-bar DV screening (PROSY caught)
  * Zero false positives on BRK-A (~$750k nominal) and heavily forward-split
    early NVDA (sub-dime adjusted_close, normal raw close)
  * Fail-closed missing-splits detection when splits cache is absent
  * Pre-floor vs post-floor quarantine accounting and stale file cleanup
  * EODHD ingest ``/api/splits/`` caching and ``--splits-only`` backfill CLI

Fully synthetic: no network, no external warehouse, no API token.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from data import dollar_volume as dv
from data import ingest_eodhd, universe
from data import warehouse as wh
from tests.test_ingest_eodhd import FakeResponse, FakeSession, std_routes


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bars_from_triples(
    triples: list[tuple[float, float, float]],
    start: str = "2020-01-02",
) -> list[dict]:
    """Build vendor-shaped daily rows from (close, adjusted_close, volume) triples."""
    days = pd.bdate_range(start, periods=len(triples))
    return [
        {
            "date": days[i].date().isoformat(),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "adjusted_close": a,
            "volume": int(v),
        }
        for i, (c, a, v) in enumerate(triples)
    ]


def _write_eod(root: str, symbol: str, rows: list[dict], *, stale_close_vol: bool = True) -> None:
    df = pd.DataFrame(rows)
    df["dollar_volume"] = df["close"] * df["volume"] if stale_close_vol else 0.0
    df["symbol"] = f"{symbol}.US"
    df["exchange"] = "US"
    wh.write_parquet_atomic(df[wh.EOD_COLUMNS], wh.eod_parquet_path(root, "US", symbol))


def _write_splits(root: str, symbol: str, splits: list[dict]) -> None:
    wh.write_json_atomic(splits, wh.splits_json_path(root, "US", symbol))


# --------------------------------------------------------------------------
# 1. split parsing & forward-split invariance
# --------------------------------------------------------------------------
def test_parse_split_ratio_handles_forward_reverse_and_spinoff_floats():
    assert dv.parse_split_ratio("7.000000/1.000000") == pytest.approx(7.0)
    assert dv.parse_split_ratio("1.000000/50.000000") == pytest.approx(0.02)
    assert dv.parse_split_ratio("1281.000000/1000.000000") == pytest.approx(1.281)
    assert dv.parse_split_ratio("bad") is None
    assert dv.parse_split_ratio("1/0") is None
    assert dv.parse_split_ratio("-2/1") is None


def test_forward_split_dollar_volume_invariant_and_dividend_unbiased():
    """AAPL-shaped 7:1 (2014-06-09) + 4:1 (2020-08-31) with dividends in adjusted_close.

    close * volume inflates pre-2014 by 28x; adjusted_close * volume deflates
    by cumulative dividends (~12.5%). split_adjusted_close * volume recovers
    exact traded dollar volume on every bar.
    """
    splits = [
        {"date": "2014-06-09", "split": "7.000000/1.000000"},
        {"date": "2020-08-31", "split": "4.000000/1.000000"},
    ]
    # True traded dollar volume = $100M on every bar:
    #   2014-06-06 (pre 7:1 & 4:1, S=28): raw close 560, raw vol 178_571.43 -> EODHD vol 5_000_000
    #   2020-08-28 (pre 4:1, S=4):        raw close 400, raw vol 250_000    -> EODHD vol 1_000_000
    #   2020-08-31 (post all splits, S=1): raw close 100, raw vol 1_000_000 -> EODHD vol 1_000_000
    # adjusted_close carries dividend deflation (0.85x in 2014, 0.95x in 2020-08-28, 1.0x on 08-31)
    rows = [
        {"date": "2014-06-06", "open": 560.0, "high": 560.0, "low": 560.0,
         "close": 560.0, "adjusted_close": 20.0 * 0.85, "volume": 5_000_000},
        {"date": "2020-08-28", "open": 400.0, "high": 400.0, "low": 400.0,
         "close": 400.0, "adjusted_close": 100.0 * 0.95, "volume": 1_000_000},
        {"date": "2020-08-31", "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "adjusted_close": 100.0, "volume": 1_000_000},
    ]
    df = ingest_eodhd._bars_to_frame(rows, "AAPL.US", "US", splits=splits)
    assert df["dollar_volume"].tolist() == pytest.approx([100_000_000.0] * 3)
    # Neither close*volume nor adjusted_close*volume matches true dollar volume:
    assert (df["close"] * df["volume"]).iloc[0] == pytest.approx(2_800_000_000.0)
    assert (df["adjusted_close"] * df["volume"]).iloc[0] == pytest.approx(85_000_000.0)


def test_high_dividend_payer_has_zero_dividend_deflation(tmp_path):
    """WMT/T/KO-shaped dividend payer (0 splits, 2.5x cumulative dividend factor).

    adjusted_close*volume understated these names by 1.5x–3x; split_adjusted_close*volume
    preserves their exact historical dollar volume.
    """
    root = str(tmp_path / "wh")
    # 40 bars where adjusted_close climbs smoothly from 0.40*close to 1.00*close (dividends only)
    triples = [
        (50.0, 50.0 * (0.40 + 0.60 * i / 39.0), 2_000_000.0)
        for i in range(40)
    ]
    _write_eod(root, "DIVCO", _bars_from_triples(triples, start="2020-01-02"))
    _write_splits(root, "DIVCO", [])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root),
        "2026-11-20",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert quarantine.empty
    assert list(ranked["symbol"]) == ["DIVCO.US"]
    # Exact $100M/day — NOT ~$70M/day from adjusted_close*volume
    assert float(ranked.loc[0, "avg_dollar_volume"]) == pytest.approx(100_000_000.0)


# --------------------------------------------------------------------------
# 2. strict PIT invariance across post-asof splits and dividends
# --------------------------------------------------------------------------
def test_pit_invariance_across_post_asof_splits_and_dividends(tmp_path):
    """Ranking as of 2020-01-31 must be IDENTICAL on a 2020 vintage vs a 2026 vintage
    where the stock later split 4:1 and paid 30% cumulative dividends.
    """
    # Vintage A (as downloaded in Jan 2020: pre-split raw close 400, raw vol 1_000_000)
    root_2020 = str(tmp_path / "wh2020")
    pre_2020 = _bars_from_triples([(400.0, 400.0, 1_000_000.0)] * 15, start="2020-01-02")
    _write_eod(root_2020, "TECH", pre_2020)
    _write_splits(root_2020, "TECH", [])

    # Vintage B (as downloaded in 2026: 4:1 split on 2020-08-31 + post-2020 dividends
    # deflating adjusted_close by 0.70x; EODHD multiplied pre-split volume by 4x)
    root_2026 = str(tmp_path / "wh2026")
    pre_in_2026 = _bars_from_triples([(400.0, 100.0 * 0.70, 4_000_000.0)] * 15, start="2020-01-02")
    post_in_2026 = _bars_from_triples([(100.0, 100.0, 4_000_000.0)] * 15, start="2020-08-31")
    _write_eod(root_2026, "TECH", pre_in_2026 + post_in_2026)
    _write_splits(root_2026, "TECH", [{"date": "2020-08-31", "split": "4.000000/1.000000"}])

    r2020, q2020 = universe.build_ranked_table(
        universe.load_all_bars(root_2020), "2020-01-31",
        splits_by_symbol=universe.load_all_splits(root_2020),
    )
    r2026, q2026 = universe.build_ranked_table(
        universe.load_all_bars(root_2026), "2020-01-31",
        splits_by_symbol=universe.load_all_splits(root_2026),
    )
    assert q2020.empty and q2026.empty
    assert float(r2020.loc[0, "avg_dollar_volume"]) == pytest.approx(400_000_000.0)
    assert float(r2026.loc[0, "avg_dollar_volume"]) == pytest.approx(
        float(r2020.loc[0, "avg_dollar_volume"])
    )


# --------------------------------------------------------------------------
# 3. reverse-split handling: GE (consistent) passes, SOS (unadjusted vol) quarantined
# --------------------------------------------------------------------------
def test_reverse_split_with_consistent_volume_passes(tmp_path):
    """GE-shaped 1:8 reverse split where EODHD properly divided pre-split volume by 8."""
    root = str(tmp_path / "wh")
    # Pre-split: raw close $13, raw vol 80M -> EODHD split-adjusted vol 10M, adj_close $104
    # Post-split: raw close $104, vol 10M -> true DV = $1.04B/day throughout
    pre = _bars_from_triples([(13.0, 104.0, 10_000_000.0)] * 20, start="2021-07-01")
    post = _bars_from_triples([(104.0, 104.0, 10_000_000.0)] * 20, start="2021-08-02")
    _write_eod(root, "GE", pre + post)
    _write_splits(root, "GE", [{"date": "2021-08-02", "split": "1.000000/8.000000"}])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2021-10-01",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert quarantine.empty
    assert list(ranked["symbol"]) == ["GE.US"]
    assert float(ranked.loc[0, "avg_dollar_volume"]) == pytest.approx(1_040_000_000.0)


def test_reverse_split_with_unadjusted_volume_sos_is_quarantined(tmp_path):
    """SOS.US-shaped reverse-split bug: 1:50 (2022-07-06) & 1:15 (2024-11-19).

    EODHD reverse-split-adjusted price (750x) but left pre-split volume unadjusted,
    so split_adjusted_close * volume (and adjusted_close * volume) would inflate
    SOS to ~$13B+/day and rank it above TSLA. The consistency check must quarantine it.
    """
    root = str(tmp_path / "wh")
    # SOS: true DV ~$10M/day, raw vol NOT divided by 50 or 15 in EODHD:
    #   2021 (S = 1/750): close $0.20, adj $150, unadjusted vol 50_000_000 (raw DV $10M; adj DV $7.5B!)
    #   2023 (S = 1/15):  close $10.0, adj $150, unadjusted vol  1_000_000 (raw DV $10M; adj DV $150M!)
    #   2025 (S = 1):     close $150,  adj $150, vol               66_667  (raw DV $10M)
    sos_rows = (
        _bars_from_triples([(0.20, 150.0, 50_000_000.0)] * 25, start="2021-06-01")
        + _bars_from_triples([(10.0, 150.0, 1_000_000.0)] * 25, start="2022-07-06")
        + _bars_from_triples([(150.0, 150.0, 66_667.0)] * 25, start="2024-11-19")
    )
    tsla_rows = _bars_from_triples([(250.0, 250.0, 40_000_000.0)] * 50, start="2024-01-02")
    _write_eod(root, "SOS", sos_rows)
    _write_splits(root, "SOS", [
        {"date": "2022-07-06", "split": "1.000000/50.000000"},
        {"date": "2024-11-19", "split": "1.000000/15.000000"},
    ])
    _write_eod(root, "TSLA", tsla_rows)
    _write_splits(root, "TSLA", [])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2026-11-20",
        min_bars=20, min_avg_dollar_volume=1_000_000.0,
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert list(ranked["symbol"]) == ["TSLA.US"]
    assert list(quarantine["symbol"]) == ["SOS.US"]
    assert "split_volume_mismatch" in str(quarantine.loc[0, "reason_codes"])
    assert bool(quarantine.loc[0, "passes_quality_floors"]) is True


def test_forward_split_with_moderate_volume_surge_not_falsely_quarantined(tmp_path):
    """A 2:1 forward split with properly adjusted volume and a 2x post-split
    trading-activity surge must NOT be falsely flagged as unadjusted volume.
    """
    root = str(tmp_path / "wh")
    pre = _bars_from_triples([(200.0, 100.0, 1_000_000.0)] * 25, start="2021-01-04")
    post = _bars_from_triples([(100.0, 100.0, 2_000_000.0)] * 25, start="2021-02-10")
    _write_eod(root, "SURGE", pre + post)
    _write_splits(root, "SURGE", [{"date": "2021-02-10", "split": "2.000000/1.000000"}])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2021-06-01",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert quarantine.empty
    assert list(ranked["symbol"]) == ["SURGE.US"]


# --------------------------------------------------------------------------
# 4. per-bar screens: sparse sentinel, PROSY, BRK-A & NVDA preservation
# --------------------------------------------------------------------------
def test_single_bar_sentinel_caught_even_when_mean_is_in_band(tmp_path):
    """1 sentinel bar (999999.9999) among 1,000 normal $50 bars has mean ~$1,049
    (would pass a [0.1, 10000] mean screen) yet must be caught per-bar.
    """
    root = str(tmp_path / "wh")
    triples = [(50.0, 50.0, 1_000_000.0)] * 999 + [(50.0, 999_999.9999, 1_000_000.0)]
    _write_eod(root, "SPARSE", _bars_from_triples(triples, start="2020-01-02"))
    _write_splits(root, "SPARSE", [])
    _write_eod(root, "AAPL", _bars_from_triples([(200.0, 200.0, 30_000_000.0)] * 50, start="2024-01-02"))
    _write_splits(root, "AAPL", [])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2026-11-20",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert list(ranked["symbol"]) == ["AAPL.US"]
    assert list(quarantine["symbol"]) == ["SPARSE.US"]
    assert "eodhd_sentinel_price" in str(quarantine.loc[0, "reason_codes"])


def test_prosy_scale_artifact_quarantined_while_brka_and_early_nvda_preserved(tmp_path):
    """PROSY ($900k -> $7 cliff) is quarantined; BRK-A (~$750k nominal, smooth)
    and early NVDA (480x forward splits -> adjusted_close $0.05, raw close $24)
    are legitimate and MUST stay ranked.
    """
    root = str(tmp_path / "wh")
    # PROSY: $900k -> $7 cliff + $900B/day bar DV
    prosy = _bars_from_triples([(900_000.0, 900_000.0, 1_000_000.0)] * 20
                               + [(7.0, 7.0, 1_000_000.0)] * 10, start="2020-01-02")
    # BRK-A: smooth ~$750k share price, 1,500 shares/day -> $1.125B ADV
    brka = _bars_from_triples([(750_000.0 + 1_000.0 * i, 750_000.0 + 1_000.0 * i, 1_500.0)
                               for i in range(30)], start="2024-01-02")
    # NVDA early slice: raw close $3.20, 40:1 split -> split_adj_close $0.08 (< $0.10!),
    # split-adjusted volume 40M -> true DV = $3.2M/day
    nvda_rows = (
        _bars_from_triples([(3.20, 0.08, 40_000_000.0)] * 25, start="2020-01-02")
        + _bars_from_triples([(0.08, 0.08, 40_000_000.0)] * 5, start="2020-03-02")
    )
    _write_eod(root, "PROSY", prosy)
    _write_splits(root, "PROSY", [])
    _write_eod(root, "BRK-A", brka)
    _write_splits(root, "BRK-A", [])
    _write_eod(root, "NVDA", nvda_rows)
    _write_splits(root, "NVDA", [{"date": "2020-03-02", "split": "40.000000/1.000000"}])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2026-11-20",
        min_bars=20, min_avg_dollar_volume=1_000_000.0,
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert set(ranked["symbol"]) == {"BRK-A.US", "NVDA.US"}
    assert list(quarantine["symbol"]) == ["PROSY.US"]
    q_codes = set(str(quarantine.loc[0, "reason_codes"]).split(","))
    assert "unsplit_price_scale_break" in q_codes
    assert "implausible_bar_dollar_volume" in q_codes
    nvda_adv = float(ranked.loc[ranked["symbol"] == "NVDA.US", "avg_dollar_volume"].iloc[0])
    assert nvda_adv == pytest.approx(3_200_000.0)


# --------------------------------------------------------------------------
# 5. fail-closed missing splits & stale column override
# --------------------------------------------------------------------------
def test_missing_splits_table_quarantines_split_symbol_but_keeps_non_split(tmp_path):
    """When /api/splits/ has not been fetched yet:
      * a symbol with a split-like jump in adjusted_close/close fails closed (quarantined)
      * a non-split symbol with smooth dividends stays ranked accurately on close*volume
    """
    root = str(tmp_path / "wh")
    split_rows = (
        _bars_from_triples([(400.0, 100.0, 4_000_000.0)] * 15, start="2020-08-03")
        + _bars_from_triples([(100.0, 100.0, 4_000_000.0)] * 15, start="2020-08-31")
    )
    plain_rows = _bars_from_triples(
        [(50.0, 50.0 * (0.95 + 0.05 * i / 29.0), 1_000_000.0) for i in range(30)],
        start="2020-08-03",
    )
    _write_eod(root, "SPLIT", split_rows)
    _write_eod(root, "PLAIN", plain_rows)
    # Deliberately NO splits/ directory

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2020-10-10",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert list(ranked["symbol"]) == ["PLAIN.US"]
    assert list(quarantine["symbol"]) == ["SPLIT.US"]
    assert "missing_splits_table" in str(quarantine.loc[0, "reason_codes"])


def test_universe_recomputes_from_primitives_overriding_stale_stored_column(tmp_path):
    """A pre-fix warehouse storing dollar_volume = close*volume is re-ranked accurately
    once splits are cached — without re-ingesting EOD parquets.
    """
    root = str(tmp_path / "wh")
    # SPLIT: 4:1 split -> true ADV $400k, stale stored close*volume says $1.0M
    split_rows = (
        _bars_from_triples([(400.0, 100.0, 4_000.0)] * 15, start="2020-08-03")
        + _bars_from_triples([(100.0, 100.0, 4_000.0)] * 15, start="2020-08-31")
    )
    # PLAIN: no split -> true ADV $700k (beats SPLIT's $400k, loses to SPLIT's stale $1.0M)
    plain_rows = _bars_from_triples([(70.0, 70.0, 10_000.0)] * 30, start="2020-08-03")
    _write_eod(root, "SPLIT", split_rows, stale_close_vol=True)
    _write_splits(root, "SPLIT", [{"date": "2020-08-31", "split": "4.000000/1.000000"}])
    _write_eod(root, "PLAIN", plain_rows, stale_close_vol=True)
    _write_splits(root, "PLAIN", [])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2020-10-10",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert quarantine.empty
    assert list(ranked["symbol"]) == ["PLAIN.US", "SPLIT.US"]
    assert float(ranked.loc[0, "avg_dollar_volume"]) == pytest.approx(700_000.0)
    assert float(ranked.loc[1, "avg_dollar_volume"]) == pytest.approx(400_000.0)


# --------------------------------------------------------------------------
# 6. CLI manifest accounting (pre-floor vs post-floor) & stale quarantine cleanup
# --------------------------------------------------------------------------
def test_cli_reports_pre_and_post_floor_quarantines_and_cleans_stale_file(tmp_path):
    root = str(tmp_path / "wh")
    # AAPL: clean, passes floors
    _write_eod(root, "AAPL", _bars_from_triples([(200.0, 200.0, 20_000_000.0)] * 30, start="2026-09-01"))
    _write_splits(root, "AAPL", [])
    # PROSY: artifact that WOULD pass floors (30 bars, huge DV)
    _write_eod(root, "PROSY", _bars_from_triples([(900_000.0, 900_000.0, 1_000_000.0)] * 20
                                                 + [(7.0, 7.0, 1_000_000.0)] * 10, start="2026-09-01"))
    _write_splits(root, "PROSY", [])
    # TINYBAD: 5-bar penny shell with sentinel -> quarantined, would NOT pass min_bars=20
    _write_eod(root, "TINYBAD", _bars_from_triples([(0.02, 999_999.9999, 100.0)] * 5, start="2026-09-01"))
    _write_splits(root, "TINYBAD", [])

    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-20",
                        "--min-bars", "20", "--min-avg-dollar-volume", "1000000"])
    assert rc == 0
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["dollar_volume_basis"] == "split_adjusted_close * volume"
    assert manifest["splits_coverage"]["symbols_with_splits_file"] == 3
    assert manifest["splits_coverage"]["symbols_without_splits_file"] == 0
    dq = manifest["data_quality_screen"]
    assert dq["quarantined_total"] == 2
    assert dq["quarantined_would_pass_floors"] == 1
    q_path = dq["quarantined_path"]
    assert q_path and os.path.exists(q_path)
    qdf = wh.read_parquet(q_path)
    assert list(qdf["symbol"]) == ["PROSY.US", "TINYBAD.US"]  # post-floor first
    assert list(qdf["passes_quality_floors"]) == [True, False]

    # Remove the two bad symbols and rerun: stale quarantined file must be deleted
    os.remove(wh.eod_parquet_path(root, "US", "PROSY"))
    os.remove(wh.eod_parquet_path(root, "US", "TINYBAD"))
    rc2 = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-20",
                         "--min-bars", "20", "--min-avg-dollar-volume", "1000000"])
    assert rc2 == 0
    assert not os.path.exists(q_path)
    manifest2 = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest2["data_quality_screen"]["quarantined_total"] == 0
    assert manifest2["data_quality_screen"]["quarantined_path"] is None


def test_empty_warehouse_cli_succeeds(tmp_path):
    root = str(tmp_path / "empty_wh")
    os.makedirs(root, exist_ok=True)
    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-20"])
    assert rc == 0
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["symbols_ranked"] == 0
    assert manifest["dollar_volume_basis"] == "split_adjusted_close * volume"


# --------------------------------------------------------------------------
# 7. ingest_eodhd /api/splits/ integration + --splits-only backfill
# --------------------------------------------------------------------------
def test_paid_ingest_fetches_splits_and_writes_consistent_dollar_volume(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    aapl_rows = (
        _bars_from_triples([(400.0, 100.0, 4_000_000.0)] * 10, start="2020-08-14")
        + _bars_from_triples([(100.0, 100.0, 4_000_000.0)] * 10, start="2020-08-31")
    )
    routes = std_routes(
        eod_map={"AAPL.US": aapl_rows},
        delisted_403=False,
        splits_map={"AAPL.US": [{"date": "2020-08-31", "split": "4.000000/1.000000"}]},
    )
    sess = FakeSession(routes=routes)
    monkeypatch.setattr(ingest_eodhd, "_SESSION", sess)
    monkeypatch.setenv("SIFT_TEST_TOKEN", "test-token")

    rc = ingest_eodhd.main([
        "--warehouse", root, "--plan", "paid", "--tickers", "AAPL",
        "--from", "2020-08-14", "--to", "2020-09-14",
    ])
    assert rc == 0
    df = wh.read_parquet(wh.eod_parquet_path(root, "US", "AAPL"))
    assert df["dollar_volume"].tolist() == pytest.approx([400_000_000.0] * 20)
    cached_splits = wh.read_json(wh.splits_json_path(root, "US", "AAPL"))
    assert cached_splits == [{"date": "2020-08-31", "split": "4.000000/1.000000"}]

    # Re-run: both parquet and splits are cached -> 0 additional calls
    n_before = len(sess.requests)
    rc2 = ingest_eodhd.main([
        "--warehouse", root, "--plan", "paid", "--tickers", "AAPL",
        "--from", "2020-08-14", "--to", "2020-09-14",
    ])
    assert rc2 == 0
    assert len(sess.requests) == n_before


def test_splits_only_backfills_existing_warehouse_resumably(tmp_path, monkeypatch):
    root = str(tmp_path / "wh")
    _write_eod(root, "AAPL", _bars_from_triples([(400.0, 100.0, 4_000_000.0)] * 5, start="2020-08-20"))
    _write_eod(root, "MSFT", _bars_from_triples([(200.0, 200.0, 2_000_000.0)] * 5, start="2020-08-20"))

    routes = [
        (lambda u, p: u.endswith("/splits/AAPL.US"),
         FakeResponse(200, [{"date": "2020-08-31", "split": "4.000000/1.000000"}])),
        (lambda u, p: u.endswith("/splits/MSFT.US"), FakeResponse(200, [])),
    ]
    sess = FakeSession(routes=routes)
    monkeypatch.setattr(ingest_eodhd, "_SESSION", sess)
    monkeypatch.setenv("SIFT_TEST_TOKEN", "test-token")

    # First run with max_calls=1: backfills AAPL only, stops before MSFT
    rc1 = ingest_eodhd.main(["--warehouse", root, "--plan", "paid", "--splits-only", "--max-calls", "1"])
    assert rc1 == 0
    assert len(sess.requests) == 1
    assert wh.read_json(wh.splits_json_path(root, "US", "AAPL")) is not None
    assert wh.read_json(wh.splits_json_path(root, "US", "MSFT")) is None

    # Second run with budget 5: skips cached AAPL (0 calls), fetches MSFT (1 call)
    rc2 = ingest_eodhd.main(["--warehouse", root, "--plan", "paid", "--splits-only", "--max-calls", "5"])
    assert rc2 == 0
    assert len(sess.requests) == 2
    assert wh.read_json(wh.splits_json_path(root, "US", "MSFT")) == []
    # Never touched /eod/
    assert all("/splits/" in url for url, _ in sess.requests)


def test_sos_unadjusted_reverse_split_quarantined_even_for_pre_split_asof(tmp_path):
    """Historical PIT run (asof=2021-12-31) on a 2026 vintage must STILL detect
    that EODHD left SOS's 2021 volume unadjusted for its 2022/2024 reverse splits
    and quarantine SOS instead of inflating 2021 DV by 750x.
    """
    root = str(tmp_path / "wh")
    sos_rows = (
        _bars_from_triples([(0.20, 150.0, 50_000_000.0)] * 25, start="2021-06-01")
        + _bars_from_triples([(10.0, 150.0, 1_000_000.0)] * 25, start="2022-07-06")
        + _bars_from_triples([(150.0, 150.0, 66_667.0)] * 25, start="2024-11-19")
    )
    _write_eod(root, "SOS", sos_rows)
    _write_splits(root, "SOS", [
        {"date": "2022-07-06", "split": "1.000000/50.000000"},
        {"date": "2024-11-19", "split": "1.000000/15.000000"},
    ])
    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2021-12-31",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert ranked.empty
    assert list(quarantine["symbol"]) == ["SOS.US"]
    assert "split_volume_mismatch" in str(quarantine.loc[0, "reason_codes"])


def test_adj_split_divergence_catches_unrecorded_split_but_allows_special_dividend(tmp_path):
    """Cached splits=[] missing a real 2:1 split that adjusted_close reflects -> quarantined.
    Meanwhile a 30% special cash dividend (D steps 0.70 -> 1.00, 1.43x < 1.75x) -> passes.
    """
    root = str(tmp_path / "wh")
    # OMITTED: 2:1 split reflected in adjusted_close, but splits file erroneously says []
    omitted = (
        _bars_from_triples([(200.0, 100.0, 2_000_000.0)] * 15, start="2022-01-03")
        + _bars_from_triples([(100.0, 100.0, 2_000_000.0)] * 15, start="2022-02-01")
    )
    # SPDIV: 30% special cash dividend on 2022-02-01, no splits
    spdiv = (
        _bars_from_triples([(100.0, 70.0, 1_000_000.0)] * 15, start="2022-01-03")
        + _bars_from_triples([(70.0, 70.0, 1_000_000.0)] * 15, start="2022-02-01")
    )
    _write_eod(root, "OMITTED", omitted)
    _write_splits(root, "OMITTED", [])
    _write_eod(root, "SPDIV", spdiv)
    _write_splits(root, "SPDIV", [])

    ranked, quarantine = universe.build_ranked_table(
        universe.load_all_bars(root), "2022-04-01",
        splits_by_symbol=universe.load_all_splits(root),
    )
    assert list(ranked["symbol"]) == ["SPDIV.US"]
    assert list(quarantine["symbol"]) == ["OMITTED.US"]
    assert "adj_split_divergence" in str(quarantine.loc[0, "reason_codes"])
