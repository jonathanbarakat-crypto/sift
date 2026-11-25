"""Saved connections store for beta users (`connection_saved` aha event).

Wires an owner's minted card to a monitored saved connection without adding
any new card or notification mechanics:
- Ownership is enforced via `CardLayer.cards.assert_owned(card_id, user_id)`
  (cards remain strictly non-transferable).
- Optional monitoring notifications use `CardLayer.notify.set_prefs(...)`
  in beta free-tier `digest` mode only (never paid live tier).
- Labels are sanitized through `sanitize_user_label` (fail-closed on tickers,
  banned phrases, and emails).
- Soft-remove preserves append-only auditability.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from sift.cards.notify import MODE_DIGEST, NotifyError
from sift.cards.service import CardLayer
from sift.web.compliance import sanitize_user_label

CONNECTIONS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS saved_connection (
    connection_id  TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    card_id        TEXT NOT NULL,
    trial_id       TEXT NOT NULL,
    label          TEXT NOT NULL,
    notify_enabled INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL,
    removed_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_saved_conn_user
    ON saved_connection(user_id, removed_at);
"""


class ConnectionError(Exception):
    """Saved-connection misuse (ownership, duplicate, compliance, ...)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = int(status_code)


@dataclass(frozen=True)
class SavedConnection:
    connection_id: str
    user_id: str
    card_id: str
    trial_id: str
    label: str
    notify_enabled: bool
    created_at: float


class ConnectionStore:
    """SQLite store for user saved connections."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.con.executescript(CONNECTIONS_SCHEMA_SQL)
        self.con.commit()

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save(
        self,
        *,
        card_layer: CardLayer,
        user_id: str,
        card_id: str,
        label: str | None = None,
        notify_enabled: bool = False,
        now: float | None = None,
    ) -> tuple[SavedConnection, bool]:
        """Save an owned card as a monitored connection.

        Returns `(connection, newly_created)`. Re-saving an already-active
        card for the same owner updates label/prefs idempotently and returns
        `newly_created=False` so telemetry does not double-count.
        """
        card = card_layer.cards.assert_owned(card_id, user_id)
        default_label = f"saved {card.family} connection ({card.scope_id})"
        clean_label = sanitize_user_label(label if label else default_label)
        ts = time.time() if now is None else float(now)

        # Optional per-card digest monitoring pref (beta = free digest only).
        if notify_enabled:
            try:
                card_layer.notify.set_pref(
                    card.card_id, user_id, True, mode=MODE_DIGEST
                )
            except NotifyError:
                notify_enabled = False

        existing = self.con.execute(
            "SELECT * FROM saved_connection"
            " WHERE user_id=? AND card_id=? AND removed_at IS NULL",
            (user_id, card.card_id),
        ).fetchone()
        if existing is not None:
            self.con.execute(
                "UPDATE saved_connection SET label=?, notify_enabled=?"
                " WHERE connection_id=?",
                (clean_label, 1 if notify_enabled else 0, existing["connection_id"]),
            )
            self.con.commit()
            return self.get(str(existing["connection_id"]), user_id), False

        conn_id = f"conn-{uuid.uuid4().hex[:12]}"
        self.con.execute(
            "INSERT INTO saved_connection"
            " (connection_id, user_id, card_id, trial_id, label, notify_enabled, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                conn_id,
                user_id,
                card.card_id,
                card.trial_id,
                clean_label,
                1 if notify_enabled else 0,
                ts,
            ),
        )
        self.con.commit()
        return self.get(conn_id, user_id), True

    def get(self, connection_id: str, user_id: str) -> SavedConnection:
        row = self.con.execute(
            "SELECT * FROM saved_connection"
            " WHERE connection_id=? AND user_id=? AND removed_at IS NULL",
            (connection_id, user_id),
        ).fetchone()
        if row is None:
            raise ConnectionError("Saved connection not found.", 404)
        return self._row_to_conn(row)

    def list_for_user(self, user_id: str) -> list[SavedConnection]:
        rows = self.con.execute(
            "SELECT * FROM saved_connection"
            " WHERE user_id=? AND removed_at IS NULL"
            " ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [self._row_to_conn(r) for r in rows]

    def remove(
        self, connection_id: str, user_id: str, *, now: float | None = None
    ) -> None:
        ts = time.time() if now is None else float(now)
        cur = self.con.execute(
            "UPDATE saved_connection SET removed_at=?"
            " WHERE connection_id=? AND user_id=? AND removed_at IS NULL",
            (ts, connection_id, user_id),
        )
        self.con.commit()
        if cur.rowcount != 1:
            raise ConnectionError("Saved connection not found.", 404)

    @staticmethod
    def _row_to_conn(row: sqlite3.Row) -> SavedConnection:
        return SavedConnection(
            connection_id=str(row["connection_id"]),
            user_id=str(row["user_id"]),
            card_id=str(row["card_id"]),
            trial_id=str(row["trial_id"]),
            label=str(row["label"]),
            notify_enabled=bool(row["notify_enabled"]),
            created_at=float(row["created_at"]),
        )


__all__ = [
    "ConnectionError",
    "ConnectionStore",
    "SavedConnection",
]
