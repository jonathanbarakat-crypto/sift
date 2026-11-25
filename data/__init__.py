"""SIFT data pipeline (server-side only).

v3 scope doc section 3.2/4: all raw vendor data lives server-side under an
internal commercial license; nothing here is ever served to a client. The
on-disk warehouse under ``data/warehouse/`` is therefore gitignored — this
repository is public and must never contain raw bars or API tokens.

Modules:
  dollar_volume  — split-consistent dollar volume (/api/splits/) & per-bar quality screens
  eodhd_client   — plan-aware EODHD client (free/paid call budget, sanitized errors)
  ingest_eodhd   — CLI: exchanges-list once, per-ticker daily EOD + splits -> warehouse
  security_type  — US ticker common-stock classifier (excludes warrants/units/rights/preferred)
  universe       — CLI: point-in-time top-N universe by avg dollar volume
  warehouse      — parquet IO + data-vintage content hash
"""
