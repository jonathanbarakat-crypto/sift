"""Shared SQLite connection and transaction helpers for SIFT.

All server-side stores (trial Ledger, CardRegistry, Vault, ShareService,
Notifier, SignalEventService, DeckRegistry) share a single SQLite database
file by default through `CardLayer`, configured consistently with:

- `PRAGMA busy_timeout = 30000` (applied first so concurrent opens wait)
- `PRAGMA journal_mode = WAL` (concurrent readers + serialized writers)
- `PRAGMA foreign_keys = ON`
- `row_factory = sqlite3.Row`

Cross-store atomic mutations use `transaction(con)` (top-level
`BEGIN IMMEDIATE` to prevent WAL reader-to-writer upgrade deadlocks;
nested `SAVEPOINT`s) paired with `maybe_commit(con)` inside individual
store methods so standalone calls still auto-commit while grouped calls
share a single atomic transaction boundary.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_BUSY_TIMEOUT_MS = 30_000

_TX_LOCK = threading.Lock()
_TX_DEPTH: dict[int, int] = {}


def connect(
    db_path: str | Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open a SQLite connection configured with WAL mode, foreign keys,
    and a busy timeout. Safe for concurrent multi-connection access."""
    path_str = str(db_path)
    is_memory = path_str == ":memory:" or path_str.startswith("file::memory:")
    if not is_memory:
        p = Path(db_path)
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
    timeout_s = max(busy_timeout_ms, 0) / 1000.0
    con = sqlite3.connect(path_str, timeout=timeout_s)
    con.row_factory = sqlite3.Row
    # Apply busy_timeout BEFORE any PRAGMA that may acquire a write lock.
    con.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    if not is_memory:
        cur_mode = con.execute("PRAGMA journal_mode").fetchone()
        if not cur_mode or str(cur_mode[0]).lower() != "wal":
            con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA foreign_keys = ON")
    return con


def in_managed_transaction(con: sqlite3.Connection) -> bool:
    """True when `con` is currently inside a `transaction(con)` block."""
    with _TX_LOCK:
        return _TX_DEPTH.get(id(con), 0) > 0


def maybe_commit(con: sqlite3.Connection) -> None:
    """Commit `con` unless an enclosing `transaction(con)` block owns the
    transaction boundary."""
    if not in_managed_transaction(con):
        con.commit()


@contextmanager
def transaction(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Atomic transaction context manager for a SQLite connection.

    Uses `BEGIN IMMEDIATE` at the outermost level (when not already in a
    transaction) so concurrent WAL writers respect `busy_timeout` instead
    of failing immediately on a deferred read-to-write lock upgrade.
    Nested blocks use `SAVEPOINT`s for partial rollback.
    """
    key = id(con)
    with _TX_LOCK:
        depth = _TX_DEPTH.get(key, 0)
        _TX_DEPTH[key] = depth + 1
    use_savepoint = depth > 0 or con.in_transaction
    sp_name = f"sift_sp_{depth + 1}"
    try:
        if use_savepoint:
            con.execute(f"SAVEPOINT {sp_name}")
        else:
            con.execute("BEGIN IMMEDIATE")
        try:
            yield con
        except BaseException:
            if use_savepoint:
                con.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
                con.execute(f"RELEASE SAVEPOINT {sp_name}")
            else:
                con.rollback()
            raise
        else:
            if use_savepoint:
                con.execute(f"RELEASE SAVEPOINT {sp_name}")
            else:
                con.commit()
    finally:
        with _TX_LOCK:
            rem = _TX_DEPTH.get(key, 1) - 1
            if rem <= 0:
                _TX_DEPTH.pop(key, None)
            else:
                _TX_DEPTH[key] = rem


def db_path_of(con: sqlite3.Connection) -> Path | None:
    """Return the filesystem path of the primary database on `con`, or None
    for an in-memory connection."""
    row = con.execute("PRAGMA database_list").fetchone()
    if row is not None and row[2]:
        return Path(row[2])
    return None


__all__ = [
    "DEFAULT_BUSY_TIMEOUT_MS",
    "connect",
    "db_path_of",
    "in_managed_transaction",
    "maybe_commit",
    "transaction",
]
