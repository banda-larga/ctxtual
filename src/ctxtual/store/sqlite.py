"""
SQLite-backed store — persistent, zero extra dependencies (uses stdlib sqlite3).

Suitable for long-running agents or scenarios where workspace data must survive
process restarts.  Each workspace maps to rows in two tables:

  - ``cf_meta``:  workspace metadata (JSON-serialised WorkspaceMeta)
  - ``cf_data``:  key/value payloads (JSON-serialised values)

All values are serialised with :mod:`json`.  For non-JSON-serialisable types
(e.g. numpy arrays) subclass and override :meth:`_encode` / :meth:`_decode`.
"""

import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ctxtual.store.base import BaseStore
from ctxtual.types import WorkspaceMeta

_DDL = """\
CREATE TABLE IF NOT EXISTS cf_meta (
    workspace_id    TEXT PRIMARY KEY,
    workspace_type  TEXT NOT NULL,
    created_at      REAL NOT NULL,
    last_accessed_at REAL NOT NULL DEFAULT 0,
    producer_fn     TEXT NOT NULL DEFAULT '',
    producer_kwargs TEXT NOT NULL DEFAULT '{}',
    item_count      INTEGER NOT NULL DEFAULT 0,
    ttl             REAL,
    data_shape      TEXT NOT NULL DEFAULT '',
    extra           TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS cf_data (
    workspace_id TEXT NOT NULL,
    key          TEXT NOT NULL,
    value        TEXT NOT NULL,
    PRIMARY KEY (workspace_id, key),
    FOREIGN KEY (workspace_id) REFERENCES cf_meta(workspace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cf_items (
    workspace_id TEXT NOT NULL,
    idx          INTEGER NOT NULL,
    value        TEXT NOT NULL,
    PRIMARY KEY (workspace_id, idx),
    FOREIGN KEY (workspace_id) REFERENCES cf_meta(workspace_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cf_meta_type ON cf_meta(workspace_type);
CREATE INDEX IF NOT EXISTS idx_cf_items_ws ON cf_items(workspace_id);

CREATE VIRTUAL TABLE IF NOT EXISTS cf_fts USING fts5(
    workspace_id,
    idx,
    content,
    tokenize='porter unicode61'
);
"""

# Migration: add columns that may not exist in older databases
_MIGRATIONS = [
    "ALTER TABLE cf_meta ADD COLUMN last_accessed_at REAL NOT NULL DEFAULT 0",
    "ALTER TABLE cf_meta ADD COLUMN ttl REAL",
    "ALTER TABLE cf_meta ADD COLUMN data_shape TEXT NOT NULL DEFAULT ''",
]


class SQLiteStore(BaseStore):
    """
    Persistent store backed by a local SQLite database.

    Args:
        path:          File path for the database.  Pass ``":memory:"`` for an
                       in-process ephemeral DB (useful for testing the SQLite
                       codepath without touching disk).
        busy_timeout:  Milliseconds to wait when the database is locked by
                       another connection.  Defaults to 5 000 ms.

    Thread safety is handled via an :class:`threading.RLock`. Each thread
    gets its own :class:`sqlite3.Connection` (stored in
    :class:`threading.local`).
    """

    def __init__(
        self,
        path: str | Path = "ctx.db",
        *,
        busy_timeout: int = 5000,
    ) -> None:
        self._path = str(path)
        self._busy_timeout = busy_timeout
        self._local = threading.local()
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._ensure_schema()

    # Connection management

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout}")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn().executescript(_DDL)
            # Apply migrations for older databases
            for migration in _MIGRATIONS:
                with contextlib.suppress(sqlite3.OperationalError):
                    self._conn().execute(migration)
            self._conn().commit()

    def close(self) -> None:
        """Close the current thread's database connection (if any)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _commit(self) -> None:
        """Commit only when not inside a ``transaction()`` block."""
        if self._tx_depth == 0:
            self._conn().commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Group multiple operations into a single atomic commit.

        Nested calls are safe — only the outermost block commits.
        """
        with self._lock:
            self._tx_depth += 1
            try:
                yield
                if self._tx_depth == 1:
                    self._conn().commit()
            except BaseException:
                if self._tx_depth == 1:
                    self._conn().rollback()
                raise
            finally:
                self._tx_depth -= 1

    # Encoding helpers

    def _encode(self, value: Any) -> str:
        """Serialise a value to a JSON string for storage."""
        return json.dumps(value, default=str)

    def _decode(self, raw: str) -> Any:
        """Deserialise a JSON string back into a Python object."""
        return json.loads(raw)

    # FTS helpers

    @staticmethod
    def _fts_content(item: Any) -> str:
        """Extract searchable text from an item for FTS indexing."""
        if isinstance(item, dict):
            parts = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
            return " ".join(parts)
        return str(item)

    def _fts_reindex(self, workspace_id: str) -> None:
        """Rebuild the FTS index for a workspace from cf_items."""
        conn = self._conn()
        conn.execute(
            "DELETE FROM cf_fts WHERE workspace_id = ?", (workspace_id,)
        )
        rows = conn.execute(
            "SELECT idx, value FROM cf_items WHERE workspace_id = ? ORDER BY idx",
            (workspace_id,),
        ).fetchall()
        if rows:
            conn.executemany(
                "INSERT INTO cf_fts (workspace_id, idx, content) VALUES (?, ?, ?)",
                [
                    (workspace_id, r["idx"], self._fts_content(self._decode(r["value"])))
                    for r in rows
                ],
            )

    # Lifecycle

    def init_workspace(self, meta: WorkspaceMeta) -> None:
        with self._lock:
            self._conn().execute(
                """
                INSERT INTO cf_meta
                    (workspace_id, workspace_type, created_at, last_accessed_at,
                     producer_fn, producer_kwargs, item_count, ttl, data_shape, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    item_count       = excluded.item_count,
                    last_accessed_at = excluded.last_accessed_at,
                    ttl              = excluded.ttl,
                    data_shape       = excluded.data_shape,
                    extra            = excluded.extra
                """,
                (
                    meta.workspace_id,
                    meta.workspace_type,
                    meta.created_at,
                    meta.last_accessed_at,
                    meta.producer_fn,
                    json.dumps(meta.producer_kwargs, default=str),
                    meta.item_count,
                    meta.ttl,
                    meta.data_shape,
                    json.dumps(meta.extra, default=str),
                ),
            )
            self._commit()

    def drop_workspace(self, workspace_id: str) -> None:
        with self._lock:
            self._conn().execute(
                "DELETE FROM cf_fts WHERE workspace_id = ?", (workspace_id,)
            )
            self._conn().execute(
                "DELETE FROM cf_items WHERE workspace_id = ?", (workspace_id,)
            )
            self._conn().execute(
                "DELETE FROM cf_data WHERE workspace_id = ?", (workspace_id,)
            )
            self._conn().execute(
                "DELETE FROM cf_meta WHERE workspace_id = ?", (workspace_id,)
            )
            self._commit()

    # Metadata

    def get_meta(self, workspace_id: str) -> WorkspaceMeta | None:
        with self._lock:
            row = (
                self._conn()
                .execute(
                    "SELECT * FROM cf_meta WHERE workspace_id = ?",
                    (workspace_id,),
                )
                .fetchone()
            )
        if row is None:
            return None
        # data_shape may not exist in older databases before migration
        try:
            ds = row["data_shape"]
        except (IndexError, KeyError):
            ds = ""
        meta = WorkspaceMeta(
            workspace_id=row["workspace_id"],
            workspace_type=row["workspace_type"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            producer_fn=row["producer_fn"],
            producer_kwargs=json.loads(row["producer_kwargs"]),
            item_count=row["item_count"],
            ttl=row["ttl"],
            data_shape=ds,
            extra=json.loads(row["extra"]),
        )
        return meta

    def list_workspaces(self, workspace_type: str | None = None) -> list[str]:
        with self._lock:
            if workspace_type:
                rows = (
                    self._conn()
                    .execute(
                        "SELECT workspace_id FROM cf_meta "
                        "WHERE workspace_type = ? ORDER BY created_at",
                        (workspace_type,),
                    )
                    .fetchall()
                )
            else:
                rows = (
                    self._conn()
                    .execute("SELECT workspace_id FROM cf_meta ORDER BY created_at")
                    .fetchall()
                )
        return [r["workspace_id"] for r in rows]

    # Data I/O

    def set(self, workspace_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn().execute(
                """
                INSERT INTO cf_data (workspace_id, key, value) VALUES (?, ?, ?)
                ON CONFLICT(workspace_id, key) DO UPDATE SET value = excluded.value
                """,
                (workspace_id, key, self._encode(value)),
            )
            self._commit()

    def get(self, workspace_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = (
                self._conn()
                .execute(
                    "SELECT value FROM cf_data WHERE workspace_id = ? AND key = ?",
                    (workspace_id, key),
                )
                .fetchone()
            )
        if row is None:
            return default
        return self._decode(row["value"])

    def delete_key(self, workspace_id: str, key: str) -> None:
        with self._lock:
            self._conn().execute(
                "DELETE FROM cf_data WHERE workspace_id = ? AND key = ?",
                (workspace_id, key),
            )
            self._commit()

    # ---- Bulk helpers (per-row storage for list payloads) ----

    def set_items(self, workspace_id: str, items: Any) -> None:  # noqa: C901
        with self._lock:
            conn = self._conn()
            if isinstance(items, list):
                conn.execute(
                    "DELETE FROM cf_items WHERE workspace_id = ?", (workspace_id,)
                )
                conn.execute(
                    "DELETE FROM cf_data "
                    "WHERE workspace_id = ? AND key = 'items'",
                    (workspace_id,),
                )
                if items:
                    conn.executemany(
                        "INSERT INTO cf_items "
                        "(workspace_id, idx, value) VALUES (?, ?, ?)",
                        [
                            (workspace_id, i, self._encode(item))
                            for i, item in enumerate(items)
                        ],
                    )
                count = len(items)
            else:
                # Non-list payloads (dicts for kv_reader, etc.) use cf_data
                conn.execute(
                    "DELETE FROM cf_items WHERE workspace_id = ?", (workspace_id,)
                )
                conn.execute(
                    "INSERT INTO cf_data (workspace_id, key, value) "
                    "VALUES (?, 'items', ?) "
                    "ON CONFLICT(workspace_id, key) "
                    "DO UPDATE SET value = excluded.value",
                    (workspace_id, self._encode(items)),
                )
                try:
                    count = len(items)
                except TypeError:
                    count = 1
            conn.execute(
                "UPDATE cf_meta SET item_count = ?, last_accessed_at = ? "
                "WHERE workspace_id = ?",
                (count, time.time(), workspace_id),
            )
            self._fts_reindex(workspace_id)
            self._commit()

    def get_items(self, workspace_id: str) -> Any:
        with self._lock:
            rows = (
                self._conn()
                .execute(
                    "SELECT value FROM cf_items "
                    "WHERE workspace_id = ? ORDER BY idx",
                    (workspace_id,),
                )
                .fetchall()
            )
            if rows:
                return [self._decode(r["value"]) for r in rows]
            # Fall back to cf_data for non-list or legacy data
            return self.get(workspace_id, "items", default=[])

    # ---- Query methods (SQL-pushed) ----

    def get_page(self, workspace_id: str, offset: int, limit: int) -> list:
        with self._lock:
            rows = (
                self._conn()
                .execute(
                    "SELECT value FROM cf_items "
                    "WHERE workspace_id = ? ORDER BY idx "
                    "LIMIT ? OFFSET ?",
                    (workspace_id, limit, offset),
                )
                .fetchall()
            )
            if rows:
                return [self._decode(r["value"]) for r in rows]
            # Fall back for non-list or legacy data
            items = self.get(workspace_id, "items", default=[])
            if isinstance(items, list):
                return items[offset : offset + limit]
            return [items] if offset == 0 and limit > 0 else []

    def search_items(
        self,
        workspace_id: str,
        query: str,
        *,
        fields: list[str] | None = None,
        max_results: int = 20,
        case_sensitive: bool = False,
    ) -> list:
        with self._lock:
            if not self._has_cf_items(workspace_id):
                return super().search_items(
                    workspace_id,
                    query,
                    fields=fields,
                    max_results=max_results,
                    case_sensitive=case_sensitive,
                )

            conn = self._conn()

            # Use FTS5 with bm25 ranking for non-case-sensitive, non-field
            # queries.  FTS5 handles tokenization, stemming, and relevance.
            if not case_sensitive and not fields:
                # Escape FTS5 special chars and join terms with implicit AND
                fts_query = " ".join(
                    f'"{t}"' for t in query.split() if t
                )
                if not fts_query:
                    return []
                rows = conn.execute(
                    "SELECT f.idx, bm25(cf_fts) AS rank "
                    "FROM cf_fts f "
                    "WHERE f.workspace_id = ? "
                    "AND cf_fts MATCH ? "
                    "ORDER BY rank "
                    "LIMIT ?",
                    (workspace_id, fts_query, max_results),
                ).fetchall()
                if rows:
                    indices = [r["idx"] for r in rows]
                    placeholders = ",".join("?" for _ in indices)
                    item_rows = conn.execute(
                        f"SELECT idx, value FROM cf_items "
                        f"WHERE workspace_id = ? AND idx IN ({placeholders})",
                        [workspace_id, *indices],
                    ).fetchall()
                    by_idx = {r["idx"]: self._decode(r["value"]) for r in item_rows}
                    return [by_idx[i] for i in indices if i in by_idx]
                # FTS returned nothing — fall through to LIKE for substring
                # matching (handles partial-word queries FTS5 may miss)

            # Fallback: LIKE/GLOB substring search with relevance scoring
            if case_sensitive:
                pattern = f"*{query}*"
                op = "GLOB"
            else:
                pattern = f"%{query}%"
                op = "LIKE"

            if fields:
                conditions = " OR ".join(
                    f"json_extract(value, '$.' || ?) {op} ?" for _ in fields
                )
                params: list[Any] = [workspace_id]
                for f in fields:
                    params.extend([f, pattern])
                sql = (
                    f"SELECT idx, value FROM cf_items "
                    f"WHERE workspace_id = ? AND ({conditions}) "
                    f"ORDER BY idx"
                )
            else:
                params = [workspace_id, pattern]
                sql = (
                    f"SELECT idx, value FROM cf_items "
                    f"WHERE workspace_id = ? AND value {op} ? "
                    f"ORDER BY idx"
                )

            rows = conn.execute(sql, params).fetchall()
            # Score and rank matches
            q = query if case_sensitive else query.lower()
            terms = q.split()
            from ctxtual.store.base import _score_item

            scored = []
            for r in rows:
                item = self._decode(r["value"])
                score = _score_item(item, terms, fields, case_sensitive)
                if score > 0:
                    scored.append((score, r["idx"], item))
            scored.sort(key=lambda t: (-t[0], t[1]))
            return [item for _, _, item in scored[:max_results]]

    def filter_items(
        self,
        workspace_id: str,
        field: str,
        value: Any,
        operator: str = "eq",
    ) -> list:
        with self._lock:
            if not self._has_cf_items(workspace_id):
                return super().filter_items(workspace_id, field, value, operator)

            op_map = {
                "eq": "=",
                "ne": "!=",
                "lt": "<",
                "lte": "<=",
                "gt": ">",
                "gte": ">=",
            }

            if operator in op_map:
                sql = (
                    "SELECT value FROM cf_items "
                    "WHERE workspace_id = ? "
                    "AND json_extract(value, '$.' || ?) IS NOT NULL "
                    f"AND json_extract(value, '$.' || ?) {op_map[operator]} ? "
                    "ORDER BY idx"
                )
                rows = (
                    self._conn()
                    .execute(sql, (workspace_id, field, field, value))
                    .fetchall()
                )
            elif operator == "contains":
                sql = (
                    "SELECT value FROM cf_items "
                    "WHERE workspace_id = ? "
                    "AND json_extract(value, '$.' || ?) LIKE ? "
                    "ORDER BY idx"
                )
                rows = (
                    self._conn()
                    .execute(sql, (workspace_id, field, f"%{value}%"))
                    .fetchall()
                )
            elif operator == "startswith":
                sql = (
                    "SELECT value FROM cf_items "
                    "WHERE workspace_id = ? "
                    "AND CAST(json_extract(value, '$.' || ?) AS TEXT) LIKE ? "
                    "ORDER BY idx"
                )
                rows = (
                    self._conn()
                    .execute(sql, (workspace_id, field, f"{value}%"))
                    .fetchall()
                )
            else:
                return super().filter_items(workspace_id, field, value, operator)

            return [self._decode(r["value"]) for r in rows]

    def sort_items(
        self,
        workspace_id: str,
        field: str,
        *,
        descending: bool = False,
        limit: int = 100,
    ) -> list:
        with self._lock:
            if not self._has_cf_items(workspace_id):
                return super().sort_items(
                    workspace_id, field, descending=descending, limit=limit
                )

            direction = "DESC" if descending else "ASC"
            sql = (
                "SELECT value FROM cf_items "
                "WHERE workspace_id = ? "
                "AND json_extract(value, '$.' || ?) IS NOT NULL "
                f"ORDER BY json_extract(value, '$.' || ?) {direction} "
                "LIMIT ?"
            )
            rows = (
                self._conn()
                .execute(sql, (workspace_id, field, field, limit))
                .fetchall()
            )
            return [self._decode(r["value"]) for r in rows]

    # ---- Mutation methods (efficient per-row operations) ----

    def append_items(self, workspace_id: str, new_items: list) -> int:
        with self._lock:
            conn = self._conn()
            if not self._has_cf_items(workspace_id):
                # Check if legacy cf_data holds a list — migrate first
                existing = self.get(workspace_id, "items", default=[])
                if not isinstance(existing, list):
                    raise TypeError(
                        f"Cannot append to workspace '{workspace_id}': "
                        f"payload is {type(existing).__name__}, not list."
                    )
                # Migrate legacy data to cf_items, then append
                existing.extend(new_items)
                self.set_items(workspace_id, existing)
                return len(existing)

            row = conn.execute(
                "SELECT COALESCE(MAX(idx), -1) AS max_idx "
                "FROM cf_items WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            start_idx = row["max_idx"] + 1

            if new_items:
                conn.executemany(
                    "INSERT INTO cf_items (workspace_id, idx, value) VALUES (?, ?, ?)",
                    [
                        (workspace_id, start_idx + i, self._encode(item))
                        for i, item in enumerate(new_items)
                    ],
                )
                conn.executemany(
                    "INSERT INTO cf_fts (workspace_id, idx, content) VALUES (?, ?, ?)",
                    [
                        (workspace_id, start_idx + i, self._fts_content(item))
                        for i, item in enumerate(new_items)
                    ],
                )
            new_count = start_idx + len(new_items)
            conn.execute(
                "UPDATE cf_meta SET item_count = ?, last_accessed_at = ? "
                "WHERE workspace_id = ?",
                (new_count, time.time(), workspace_id),
            )
            self._commit()
            return new_count

    def update_item(self, workspace_id: str, index: int, item: Any) -> None:
        with self._lock:
            if not self._has_cf_items(workspace_id):
                return super().update_item(workspace_id, index, item)

            conn = self._conn()
            affected = conn.execute(
                "UPDATE cf_items SET value = ? WHERE workspace_id = ? AND idx = ?",
                (self._encode(item), workspace_id, index),
            ).rowcount
            if affected == 0:
                count = self.count_items(workspace_id)
                raise IndexError(
                    f"Index {index} out of range for workspace '{workspace_id}' "
                    f"with {count} items."
                )
            conn.execute(
                "UPDATE cf_meta SET last_accessed_at = ? WHERE workspace_id = ?",
                (time.time(), workspace_id),
            )
            # Update FTS index for the changed item
            conn.execute(
                "DELETE FROM cf_fts WHERE workspace_id = ? AND idx = ?",
                (workspace_id, index),
            )
            conn.execute(
                "INSERT INTO cf_fts (workspace_id, idx, content) VALUES (?, ?, ?)",
                (workspace_id, index, self._fts_content(item)),
            )
            self._commit()

    def patch_item(
        self, workspace_id: str, index: int, fields: dict[str, Any]
    ) -> None:
        with self._lock:
            if not self._has_cf_items(workspace_id):
                return super().patch_item(workspace_id, index, fields)

            conn = self._conn()
            row = conn.execute(
                "SELECT value FROM cf_items WHERE workspace_id = ? AND idx = ?",
                (workspace_id, index),
            ).fetchone()
            if row is None:
                count = self.count_items(workspace_id)
                raise IndexError(
                    f"Index {index} out of range for workspace '{workspace_id}' "
                    f"with {count} items."
                )
            existing = self._decode(row["value"])
            if not isinstance(existing, dict):
                raise TypeError(
                    f"Cannot patch item at index {index}: "
                    f"item is {type(existing).__name__}, not dict."
                )
            existing.update(fields)
            conn.execute(
                "UPDATE cf_items SET value = ? WHERE workspace_id = ? AND idx = ?",
                (self._encode(existing), workspace_id, index),
            )
            conn.execute(
                "UPDATE cf_meta SET last_accessed_at = ? WHERE workspace_id = ?",
                (time.time(), workspace_id),
            )
            # Update FTS index for the patched item
            conn.execute(
                "DELETE FROM cf_fts WHERE workspace_id = ? AND idx = ?",
                (workspace_id, index),
            )
            conn.execute(
                "INSERT INTO cf_fts (workspace_id, idx, content) VALUES (?, ?, ?)",
                (workspace_id, index, self._fts_content(existing)),
            )
            self._commit()

    def delete_items(self, workspace_id: str, indices: list[int]) -> int:
        with self._lock:
            if not self._has_cf_items(workspace_id):
                return super().delete_items(workspace_id, indices)

            conn = self._conn()
            if indices:
                placeholders = ",".join("?" for _ in indices)
                conn.execute(
                    f"DELETE FROM cf_items "
                    f"WHERE workspace_id = ? AND idx IN ({placeholders})",
                    [workspace_id, *indices],
                )
            # Re-index remaining rows to maintain contiguous indices
            rows = conn.execute(
                "SELECT rowid, idx FROM cf_items "
                "WHERE workspace_id = ? ORDER BY idx",
                (workspace_id,),
            ).fetchall()
            for new_idx, row in enumerate(rows):
                if row["idx"] != new_idx:
                    conn.execute(
                        "UPDATE cf_items SET idx = ? WHERE rowid = ?",
                        (new_idx, row["rowid"]),
                    )
            new_count = len(rows)
            conn.execute(
                "UPDATE cf_meta SET item_count = ?, last_accessed_at = ? "
                "WHERE workspace_id = ?",
                (new_count, time.time(), workspace_id),
            )
            self._fts_reindex(workspace_id)
            self._commit()
            return new_count

    # ---- Internal helpers ----

    def _has_cf_items(self, workspace_id: str) -> bool:
        """Check whether workspace has per-row data in cf_items."""
        return (
            self._conn()
            .execute(
                "SELECT 1 FROM cf_items WHERE workspace_id = ? LIMIT 1",
                (workspace_id,),
            )
            .fetchone()
        ) is not None

    def __repr__(self) -> str:
        return f"SQLiteStore(path={self._path!r})"
