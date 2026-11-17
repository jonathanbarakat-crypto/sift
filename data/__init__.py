"""SIFT data pipeline (server-side only).

v3 scope doc section 3.2/4: all raw vendor data lives server-side under an
internal commercial license; nothing here is ever served to a client. The
on-disk warehouse under ``data/warehouse/`` is therefore gitignored — this
repository is public and must never contain raw bars or API tokens.

Modules:
  eodhd_client   — thin EODHD free-tier client (call budget, sanitized errors)
  ingest_eodhd   — CLI: exchanges-list once, per-ticker daily EOD -> parquet
  yf_client      — RESEARCH-ONLY yfinance (Yahoo) client (retry/backoff+jitter on 429)
  ingest_yf      — CLI: per-ticker daily bars -> parquet, research_grade vintage
  universe       — CLI: point-in-time top-N universe by avg dollar volume
  warehouse      — parquet IO + data-vintage content hash
"""
