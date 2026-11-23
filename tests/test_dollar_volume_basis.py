"""Regression tests for the dollar-volume price basis (hermes-tasks #234).

EODHD's ``volume`` is split-adjusted (restated to the current share basis)
while ``close`` is the raw traded price, so ``close * volume`` mixes price
bases and inflates historical dollar volume by the cumulative split factor
(AAPL 2014 7:1, 2020 4:1). The consistent basis is ``adjusted_close * volume``
— both fields on the vendor's adjusted basis (verified against an independent
vendor; see the PR description).

Also covers the price-scale sanity screen: a symbol whose mean raw close over
the as-of window falls outside the USD band is a vendor scale artifact, not a
rankable liquid name — it is quarantined (excluded from the ranking, written
out for review) instead of being silently ranked.

Fully synthetic: no network, no warehouse, no API token.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from data import ingest_eodhd, universe, warehouse as wh


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_symbol(root: str, symbol: str, rows: list[dict]) -> None:
    """Write one eod parquet for ``symbol`` built from vendor-shaped rows."""
    df = pd.DataFrame(rows)
    df["dollar_volume"] = df["close"] * df["volume"]  # stale pre-fix value
    df["symbol"] = f"{symbol}.US"
    df["exchange"] = "US"
    os.makedirs(os.path.join(root, "eod", "US"), exist_ok=True)
    wh.write_parquet_atomic(df[wh.EOD_COLUMNS],
                            os.path.join(root, "eod", "US", f"{symbol}.parquet"))


def _bars(rows_close_adj_vol: list[tuple[float, float, float]], start: str = "2026-10-01"):
    """Vendor-shaped daily rows from (close, adjusted_close, volume) triples."""
    days = pd.bdate_range(start, periods=len(rows_close_adj_vol))
    return [
        {"date": days[i].date().isoformat(), "open": c, "high": c, "low": c,
         "close": c, "adjusted_close": a, "volume": int(v)}
        for i, (c, a, v) in enumerate(rows_close_adj_vol)
    ]


# --------------------------------------------------------------------------
# 1. the computation itself — planted 4:1 split, dollar volume invariant
# --------------------------------------------------------------------------
def test_dollar_volume_is_invariant_across_a_planted_split():
    """Pre-split raw close 400 -> 4:1 split -> adjusted 100, volume x4.

    Vendor fields either side of the split describe the SAME economic trading
    value, so dollar volume must not jump by the split factor. close*volume
    jumps 4x (1.6M -> 400k); adjusted_close*volume does not (400k -> 400k).
    """
    rows = [
        {"date": "2020-08-28", "open": 400.0, "high": 400.0, "low": 400.0,
         "close": 400.0, "adjusted_close": 100.0, "volume": 4_000},
        {"date": "2020-08-31", "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "adjusted_close": 100.0, "volume": 4_000},
    ]
    df = ingest_eodhd._bars_to_frame(rows, "X.US", "US")
    assert df["dollar_volume"].tolist() == pytest.approx([400_000.0, 400_000.0])
    # the pre-fix computation is exactly what must NOT come back
    assert (df["close"] * df["volume"]).tolist() != pytest.approx(df["dollar_volume"].tolist())


# --------------------------------------------------------------------------
# 2. the ranking reads the primitives, not a stale stored column
# --------------------------------------------------------------------------
def test_universe_ranking_uses_adjusted_close_volume_not_the_stored_column(tmp_path):
    """A warehouse written by the pre-fix ingest carries dollar_volume = close*volume.

    The builder must recompute from the primitives so an existing warehouse is
    re-ranked correctly without a re-ingest.
    """
    root = str(tmp_path / "wh")
    # SPLIT: raw close 400 / adjusted 100 / volume 4_000 -> true adv 400_000;
    # the stale stored column says close*volume = 1_600_000 (the defect).
    split_rows = _bars([(400.0, 100.0, 4_000.0)] * 3)
    # PLAIN: no split -> both bases agree at 10 * 100_000 = 1_000_000.
    plain_rows = _bars([(10.0, 10.0, 100_000.0)] * 3)
    for rows, sym in ((split_rows, "SPLIT"), (plain_rows, "PLAIN")):
        df = pd.DataFrame(rows)
        df["dollar_volume"] = df["close"] * df["volume"]  # stale, pre-fix value
        df["symbol"] = f"{sym}.US"
        df["exchange"] = "US"
        os.makedirs(os.path.join(root, "eod", "US"), exist_ok=True)
        wh.write_parquet_atomic(df[wh.EOD_COLUMNS],
                                os.path.join(root, "eod", "US", f"{sym}.parquet"))

    ranked, quarantine = universe.build_ranked_table(universe.load_all_bars(root), "2026-10-10")
    assert list(ranked["symbol"]) == ["PLAIN.US", "SPLIT.US"]  # fixed basis: 1.0M > 0.4M
    assert float(ranked.loc[0, "avg_dollar_volume"]) == pytest.approx(1_000_000.0)
    assert float(ranked.loc[1, "avg_dollar_volume"]) == pytest.approx(400_000.0)
    assert quarantine.empty


# --------------------------------------------------------------------------
# 3. price-scale sanity screen — vendor artifact quarantined, not ranked
# --------------------------------------------------------------------------
def test_price_scale_screen_quarantines_vendor_scale_artifact(tmp_path):
    """PROSY-shaped series: raw close ~$900k for 20 bars, then $7 for 10.

    Mean close is ~$600k — far outside [0.1, 10000] USD — and the stale
    dollar volume would rank it #1. It must be excluded from the ranking and
    reported for review.
    """
    root = str(tmp_path / "wh")
    prosy = _bars([(900_000.0, 900_000.0, 1_000_000.0)] * 20
                  + [(7.0, 7.0, 1_000_000.0)] * 10)
    aapl = _bars([(350.0, 350.0, 40_000_000.0)] * 30)
    _write_symbol(root, "PROSY", prosy)
    _write_symbol(root, "AAPL", aapl)

    ranked, quarantine = universe.build_ranked_table(universe.load_all_bars(root), "2026-11-20",
                                                     min_bars=1, min_avg_dollar_volume=0.0)
    assert "PROSY.US" not in set(ranked["symbol"])
    assert list(ranked["symbol"]) == ["AAPL.US"]

    assert [str(s) for s in quarantine["symbol"]] == ["PROSY.US"]
    assert float(quarantine.loc[0, "mean_close"]) == pytest.approx(
        (20 * 900_000.0 + 10 * 7.0) / 30)
    assert "close" in str(quarantine.loc[0, "reason"])


def test_price_scale_screen_quarantines_adjusted_close_sentinel(tmp_path):
    """Vendor missing-data sentinel (999999.9999) stuffed into adjusted_close.

    Raw close stays in band, so a close-only screen would rank this symbol #1
    on adjusted_close*volume; the screen must also check the adjusted series.
    """
    root = str(tmp_path / "wh")
    prosy = _bars([(900_000.0, 900_000.0, 1_000_000.0)] * 20
                  + [(7.0, 7.0, 1_000_000.0)] * 10)
    otc = _bars([(1.4, 999_999.9999, 5_000_000.0)] * 20 + [(1.2, 1.2, 50_000.0)] * 10)
    aapl = _bars([(350.0, 350.0, 40_000_000.0)] * 30)
    _write_symbol(root, "PROSY", prosy)
    _write_symbol(root, "OTCX", otc)
    _write_symbol(root, "AAPL", aapl)

    ranked, quarantine = universe.build_ranked_table(universe.load_all_bars(root), "2026-11-20",
                                                     min_bars=1, min_avg_dollar_volume=0.0)
    assert list(ranked["symbol"]) == ["AAPL.US"]
    assert set(quarantine["symbol"]) == {"PROSY.US", "OTCX.US"}
    assert "adjusted_close" in str(quarantine.loc[quarantine["symbol"] == "OTCX.US",
                                                  "reason"].iloc[0])


def test_price_scale_screen_keeps_a_normal_series(tmp_path):
    """AAPL-shaped series (close ~$350) must pass the screen untouched."""
    root = str(tmp_path / "wh")
    _write_symbol(root, "AAPL", _bars([(350.0, 350.0, 40_000_000.0)] * 30))
    ranked, quarantine = universe.build_ranked_table(universe.load_all_bars(root), "2026-11-20",
                                                     min_bars=1, min_avg_dollar_volume=0.0)
    assert list(ranked["symbol"]) == ["AAPL.US"]
    assert quarantine.empty


def test_universe_cli_writes_quarantine_table_and_manifest(tmp_path):
    """End to end: the CLI quarantines the artifact and records the basis."""
    root = str(tmp_path / "wh")
    _write_symbol(root, "PROSY", _bars([(900_000.0, 900_000.0, 1_000_000.0)] * 20
                                       + [(7.0, 7.0, 1_000_000.0)] * 10))
    _write_symbol(root, "AAPL", _bars([(350.0, 350.0, 40_000_000.0)] * 30))
    rc = universe.main(["--warehouse", root, "--n", "10", "--asof", "2026-11-20",
                        "--min-bars", "1", "--min-avg-dollar-volume", "0"])
    assert rc == 0
    quar = wh.read_parquet(os.path.join(root, "universe", "quarantined_2026-11-20.parquet"))
    assert list(quar["symbol"]) == ["PROSY.US"]
    ranked = wh.read_parquet(os.path.join(root, "universe", "ranked_2026-11-20.parquet"))
    assert list(ranked["symbol"]) == ["AAPL.US"]
    manifest = wh.read_json(os.path.join(root, "universe", "manifest.json"))
    assert manifest["dollar_volume_basis"] == "adjusted_close * volume"
    assert manifest["price_scale_screen"]["quarantined"] == 1
    assert manifest["price_scale_screen"]["quarantined_path"].endswith(
        "quarantined_2026-11-20.parquet")
