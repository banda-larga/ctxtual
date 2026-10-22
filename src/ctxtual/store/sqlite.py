"""
SQLite-backed store — persistent, zero extra dependencies (uses stdlib sqlite3).

Suitable for long-running agents or scenarios where workspace data must survive
process restarts.  Tables:

  - ``cf_meta``:  one row per workspace (metadata)
  - ``cf_items``: one row per item of a list payload, so paging, filtering
    and sorting can be pushed down to SQL
  - ``cf_data``:  key/value payloads (non-list payloads and auxiliary keys)
  - ``cf_fts``:   FTS5 index, one row per (item, top-level field)

All values are serialised with :mod:`json`.  For non-JSON-serialisable types
(e.g. numpy arrays) subclass and override :meth:`_encode` / :meth:`_decode`.
"""

import contextlib
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ctxtual.store.base import BaseStore, query_terms
from ctxtual.types import WorkspaceMeta

# Bump when the on-disk layout changes; see ``_migrate``.
_SCHEMA_VERSION = 2

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
    extra           TEXT NOT NULL DEFAULT '{}',
    session_id      TEXT
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
"""

# ``wsk`` holds a single opaque token per workspace so a MATCH can be
# narrowed to one workspace through the full-text index itself.  Everything
# except ``content`` is UNINDEXED, so user queries can never match
# workspace ids, item indices or field names.
_FTS_DDL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS cf_fts USING fts5(
    wsk,
    workspace_id UNINDEXED,
    idx UNINDEXED,
    field UNINDEXED,
    content,
    tokenize='porter unicode61'
);
"""

# Columns added after the first release; applied idempotently.
_COLUMN_MIGRATIONS = [
    "ALTER TABLE cf_meta ADD COLUMN last_accessed_at REAL NOT NULL DEFAULT 0",
    "ALTER TABLE cf_meta ADD COLUMN ttl REAL",
    "ALTER TABLE cf_meta ADD COLUMN data_shape TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE cf_meta ADD COLUMN session_id TEXT",
]

# bm25 column weights: only ``content`` contributes to relevance.
_BM25 = "bm25(cf_fts, 0.0, 0.0, 0.0, 0.0, 1.0)"


def _wsk(workspace_id: str) -> str:
    """Single alphanumeric FTS token identifying a workspace."""
    return "w" + hashlib.sha1(workspace_id.encode()).hexdigest()[:20]


def _fts_phrase(text: str) -> str:
    """Quote *text* as an FTS5 string (doubling quotes, dropping NULs)."""
    return '"' + text.replace("\x00", "").replace('"', '""') + '"'


def _leaf_text(value: Any) -> list[str]:
    """Collect searchable scalar text from a (possibly nested) value."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _leaf_text(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _leaf_text(v)]
    return []


class SQLiteStore(BaseStore):
    """
    Persistent store backed by a local SQLite database.

    Args:
        path:          File path for the database.  Pass ``":memory:"`` for an
                       in-process ephemeral DB (useful for testing the SQLite
                       codepath without touching disk).
        busy_timeout:  Milliseconds to wait when the database is locked by
                       another connection.  Defaults to 5 000 ms.

    Each thread gets its own :class:`sqlite3.Connection`; a store-wide
    :class:`threading.RLock` serialises access.  :meth:`close` closes the
    connections of every thread.

    ``":memory:"`` gives this store its own private in-memory database that
    is shared by all of its threads and lives until :meth:`close`.
    """

    def __init__(
        self,
        path: str | Path = "ctx.db",
        *,
        busy_timeout: int = 5000,
    ) -> None:
        self._path = str(path)
        if self._path == ":memory:":
            # A plain ":memory:" connection per thread would give each thread
            # its own empty database; a named shared-cache DB is shared.
            self._uri = (
                f"file:ctxtual-{id(self):x}-{time.time_ns()}?mode=memory&cache=shared"
            )
        else:
            self._uri = None
        self._busy_timeout = busy_timeout
        self._local = threading.local()
        self._lock = threading.RLock()
        self._tx_depth = 0
        # thread ident -> (thread, connection); dead threads are pruned
        self._connections: dict[int, tuple[threading.Thread, sqlite3.Connection]] = {}
        self._ensure_schema()

    # Connection management

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            if self._uri is not None:
                conn = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
            else:
                conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._lock:
                self._prune_dead_connections()
                current = threading.current_thread()
                self._connections[current.ident or id(current)] = (current, conn)
        return conn

    def _prune_dead_connections(self) -> None:
        """Close connections owned by threads that have exited."""
        for key, (thread, conn) in list(self._connections.items()):
            if not thread.is_alive():
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
                del self._connections[key]

    def _ensure_schema(self) -> None:
        # One IMMEDIATE transaction, so processes opening the same database
        # concurrently run the migration one at a time.
        with self.transaction():
            conn = self._conn()
            _execute_script(conn, _DDL)
            for migration in _COLUMN_MIGRATIONS:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(migration)
            self._migrate_fts(conn)

    def _migrate_fts(self, conn: sqlite3.Connection) -> None:
        """Create the FTS table, rebuilding it if it predates schema v2."""
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'cf_fts'"
        ).fetchone()
        if exists and version < _SCHEMA_VERSION:
            # v1 indexed workspace_id/idx as searchable text, so queries
            # matched ids and indices.  Rebuild from cf_items.
            conn.execute("DROP TABLE cf_fts")
            exists = None
        _execute_script(conn, _FTS_DDL)
        if not exists:
            ws_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT workspace_id FROM cf_items"
                ).fetchall()
            ]
            for ws_id in ws_ids:
                self._fts_reindex(ws_id)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def close(self) -> None:
        """Close every connection opened by this store, across all threads."""
        with self._lock:
            for _, conn in self._connections.values():
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._connections.clear()
            # A fresh thread-local drops stale handles in every thread.
            self._local = threading.local()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Group multiple operations into a single atomic commit.

        Nested calls are safe — only the outermost block commits, and any
        exception rolls the whole block back.  Every mutating method runs in
        one, so a failed write never leaves a transaction open.
        """
        with self._lock:
            conn = self._conn()
            if self._tx_depth == 0 and not conn.in_transaction:
                # Take the write lock up front so check-then-write sequences
                # are atomic across processes, not only across threads.
                conn.execute("BEGIN IMMEDIATE")
            self._tx_depth += 1
            try:
                yield
                if self._tx_depth == 1:
                    conn.commit()
            except BaseException:
                if self._tx_depth == 1:
                    conn.rollback()
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
    def _fts_fields(item: Any) -> list[tuple[str, str]]:
        """Split an item into ``(field, text)`` rows for FTS indexing."""
        if isinstance(item, dict):
            rows = []
            for key, value in item.items():
                text = " ".join(_leaf_text(value))
                if text:
                    rows.append((str(key), text))
            return rows
        text = " ".join(_leaf_text(item))
        return [("", text)] if text else []

    def _fts_insert(self, workspace_id: str, start: int, items: list[Any]) -> None:
        wsk = _wsk(workspace_id)
        rows = [
            (wsk, workspace_id, start + i, field, text)
            for i, item in enumerate(items)
            for field, text in self._fts_fields(item)
        ]
        if rows:
            self._conn().executemany(
                "INSERT INTO cf_fts (wsk, workspace_id, idx, field, content) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def _fts_delete(self, workspace_id: str, index: int | None = None) -> None:
        """Remove FTS rows for a workspace (or a single item of it)."""
        sql = (
            "DELETE FROM cf_fts WHERE rowid IN ("
            "SELECT rowid FROM cf_fts WHERE cf_fts MATCH ?"
        )
        params: list[Any] = [f"wsk : {_fts_phrase(_wsk(workspace_id))}"]
        if index is not None:
            sql += " AND idx = ?"
            params.append(index)
        self._conn().execute(sql + ")", params)

    def _fts_reindex(self, workspace_id: str) -> None:
        """Rebuild the FTS index for a workspace from cf_items."""
        self._fts_delete(workspace_id)
        rows = (
            self._conn()
            .execute(
                "SELECT idx, value FROM cf_items WHERE workspace_id = ? ORDER BY idx",
                (workspace_id,),
            )
            .fetchall()
        )
        wsk = _wsk(workspace_id)
        fts_rows = [
            (wsk, workspace_id, r["idx"], field, text)
            for r in rows
            for field, text in self._fts_fields(self._decode(r["value"]))
        ]
        if fts_rows:
            self._conn().executemany(
                "INSERT INTO cf_fts (wsk, workspace_id, idx, field, content) "
                "VALUES (?, ?, ?, ?, ?)",
                fts_rows,
            )

    # Lifecycle

    def init_workspace(self, meta: WorkspaceMeta) -> None:
        """Insert or fully replace the metadata row (same as MemoryStore)."""
        with self.transaction():
            self._conn().execute(
                """
                INSERT INTO cf_meta
                    (workspace_id, workspace_type, created_at, last_accessed_at,
                     producer_fn, producer_kwargs, item_count, ttl, data_shape,
                     extra, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO UPDATE SET
                    workspace_type   = excluded.workspace_type,
                    created_at       = excluded.created_at,
                    last_accessed_at = excluded.last_accessed_at,
                    producer_fn      = excluded.producer_fn,
                    producer_kwargs  = excluded.producer_kwargs,
                    item_count       = excluded.item_count,
                    ttl              = excluded.ttl,
                    data_shape       = excluded.data_shape,
                    extra            = excluded.extra,
                    session_id       = excluded.session_id
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
                    meta.session_id,
                ),
            )

    def drop_workspace(self, workspace_id: str) -> None:
        with self.transaction():
            self._fts_delete(workspace_id)
            self._conn().execute(
                "DELETE FROM cf_items WHERE workspace_id = ?", (workspace_id,)
            )
            self._conn().execute(
                "DELETE FROM cf_data WHERE workspace_id = ?", (workspace_id,)
            )
            self._conn().execute(
                "DELETE FROM cf_meta WHERE workspace_id = ?", (workspace_id,)
            )

    def touch(self, workspace_id: str) -> None:
        with self.transaction():
            self._conn().execute(
                "UPDATE cf_meta SET last_accessed_at = ? WHERE workspace_id = ?",
                (time.time(), workspace_id),
            )

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
        return WorkspaceMeta(
            workspace_id=row["workspace_id"],
            workspace_type=row["workspace_type"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            producer_fn=row["producer_fn"],
            producer_kwargs=json.loads(row["producer_kwargs"]),
            item_count=row["item_count"],
            ttl=row["ttl"],
            data_shape=row["data_shape"],
            extra=json.loads(row["extra"]),
            session_id=row["session_id"],
        )

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
        with self.transaction():
            self._conn().execute(
                """
                INSERT INTO cf_data (workspace_id, key, value) VALUES (?, ?, ?)
                ON CONFLICT(workspace_id, key) DO UPDATE SET value = excluded.value
                """,
                (workspace_id, key, self._encode(value)),
            )

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
        with self.transaction():
            self._conn().execute(
                "DELETE FROM cf_data WHERE workspace_id = ? AND key = ?",
                (workspace_id, key),
            )

    # ---- Bulk helpers (per-row storage for list payloads) ----

    def set_items(self, workspace_id: str, items: Any) -> None:
        with self.transaction():
            conn = self._conn()
            conn.execute("DELETE FROM cf_items WHERE workspace_id = ?", (workspace_id,))
            self._fts_delete(workspace_id)
            if isinstance(items, list):
                conn.execute(
                    "DELETE FROM cf_data WHERE workspace_id = ? AND key = 'items'",
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
                    self._fts_insert(workspace_id, 0, items)
                count = len(items)
            else:
                # Non-list payloads (dicts for kv_reader, strings, ...) use cf_data
                conn.execute(
                    "INSERT INTO cf_data (workspace_id, key, value) "
                    "VALUES (?, 'items', ?) "
                    "ON CONFLICT(workspace_id, key) "
                    "DO UPDATE SET value = excluded.value",
                    (workspace_id, self._encode(items)),
                )
                count = 1 if isinstance(items, str) else _safe_len(items)
            conn.execute(
                "UPDATE cf_meta SET item_count = ?, last_accessed_at = ? "
                "WHERE workspace_id = ?",
                (count, time.time(), workspace_id),
            )

    def get_items(self, workspace_id: str) -> Any:
        with self._lock:
            rows = (
                self._conn()
                .execute(
                    "SELECT value FROM cf_items WHERE workspace_id = ? ORDER BY idx",
                    (workspace_id,),
                )
                .fetchall()
            )
            if rows:
                return [self._decode(r["value"]) for r in rows]
            # Non-list payloads and empty lists live in cf_data
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
            if self._has_cf_items(workspace_id):
                return []  # offset past the end
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
        """
        Ranked full-text search.

        Case-insensitive queries of plain words use FTS5: each term is
        matched as a prefix (after Porter stemming), terms are OR-ed, and
        results are ranked by bm25 summed over the matching fields.  The
        substring scoring of :meth:`BaseStore.search_items` (as in
        MemoryStore) is used instead when FTS finds nothing (e.g. a fragment
        from the middle of a word), when a term contains symbols ("C++",
        "e-mail") that the tokenizer would drop, or for case-sensitive
        searches.
        """
        with self._lock:
            if not case_sensitive and self._has_cf_items(workspace_id):
                hits = self._fts_search(workspace_id, query, fields, max_results)
                if hits:
                    return hits
            return super().search_items(
                workspace_id,
                query,
                fields=fields,
                max_results=max_results,
                case_sensitive=case_sensitive,
            )

    def _fts_search(
        self,
        workspace_id: str,
        query: str,
        fields: list[str] | None,
        max_results: int,
    ) -> list:
        terms = query_terms(query)
        if not terms or max_results <= 0 or not all(t.isalnum() for t in terms):
            # The tokenizer drops symbols, so "C++" and "C#" would both
            # become "c"; such queries use exact substring scoring instead.
            return []
        content_expr = " OR ".join(f"{_fts_phrase(t)}*" for t in terms)
        match = (
            f"wsk : {_fts_phrase(_wsk(workspace_id))} AND content : ({content_expr})"
        )
        params: list[Any] = [match]
        field_clause = ""
        if fields:
            field_clause = f" AND field IN ({','.join('?' for _ in fields)})"
            params.extend(fields)
        params.append(max_results)
        conn = self._conn()
        # bm25() can't be used inside an aggregate directly; LIMIT -1 keeps
        # SQLite from flattening the subquery.
        rows = conn.execute(
            "SELECT idx, SUM(score) AS score FROM ("
            f"  SELECT idx, {_BM25} AS score FROM cf_fts"
            f"  WHERE cf_fts MATCH ?{field_clause} LIMIT -1"
            ") GROUP BY idx ORDER BY score, idx LIMIT ?",
            params,
        ).fetchall()
        if not rows:
            return []
        indices = [r["idx"] for r in rows]
        placeholders = ",".join("?" for _ in indices)
        item_rows = conn.execute(
            f"SELECT idx, value FROM cf_items "
            f"WHERE workspace_id = ? AND idx IN ({placeholders})",
            [workspace_id, *indices],
        ).fetchall()
        by_idx = {r["idx"]: self._decode(r["value"]) for r in item_rows}
        return [by_idx[i] for i in indices if i in by_idx]

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
                    "AND instr(json_extract(value, '$.' || ?), ?) > 0 "
                    "ORDER BY idx"
                )
                rows = (
                    self._conn()
                    .execute(sql, (workspace_id, field, str(value)))
                    .fetchall()
                )
            elif operator == "startswith":
                sql = (
                    "SELECT value FROM cf_items "
                    "WHERE workspace_id = ? "
                    "AND substr(CAST(json_extract(value, '$.' || ?) AS TEXT), 1, ?) = ? "
                    "ORDER BY idx"
                )
                prefix = str(value)
                rows = (
                    self._conn()
                    .execute(sql, (workspace_id, field, len(prefix), prefix))
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
                f"ORDER BY json_extract(value, '$.' || ?) {direction}, idx "
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
        with self.transaction():
            conn = self._conn()
            if not self._has_cf_items(workspace_id):
                # Empty list or non-list payload stored in cf_data
                existing = self.get(workspace_id, "items", default=[])
                if not isinstance(existing, list):
                    raise TypeError(
                        f"Cannot append to workspace '{workspace_id}': "
                        f"payload is {type(existing).__name__}, not list."
                    )
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
                self._fts_insert(workspace_id, start_idx, new_items)
            new_count = start_idx + len(new_items)
            conn.execute(
                "UPDATE cf_meta SET item_count = ?, last_accessed_at = ? "
                "WHERE workspace_id = ?",
                (new_count, time.time(), workspace_id),
            )
            return new_count

    def update_item(self, workspace_id: str, index: int, item: Any) -> None:
        with self.transaction():
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
            self._fts_delete(workspace_id, index)
            self._fts_insert(workspace_id, index, [item])

    def patch_item(self, workspace_id: str, index: int, fields: dict[str, Any]) -> None:
        with self.transaction():
            if not self._has_cf_items(workspace_id):
                return super().patch_item(workspace_id, index, fields)

            row = (
                self._conn()
                .execute(
                    "SELECT value FROM cf_items WHERE workspace_id = ? AND idx = ?",
                    (workspace_id, index),
                )
                .fetchone()
            )
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
            self.update_item(workspace_id, index, existing)

    def delete_items(self, workspace_id: str, indices: list[int]) -> int:
        with self.transaction():
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
                "SELECT rowid, idx FROM cf_items WHERE workspace_id = ? ORDER BY idx",
                (workspace_id,),
            ).fetchall()
            for new_idx, row in enumerate(rows):
                if row["idx"] != new_idx:
                    conn.execute(
                        "UPDATE cf_items SET idx = ? WHERE rowid = ?",
                        (new_idx, row["rowid"]),
                    )
            new_count = len(rows)
            if new_count == 0:
                # Keep the workspace a (now empty) list rather than
                # falling back to whatever cf_data held.
                conn.execute(
                    "INSERT INTO cf_data (workspace_id, key, value) "
                    "VALUES (?, 'items', '[]') "
                    "ON CONFLICT(workspace_id, key) DO UPDATE SET value = '[]'",
                    (workspace_id,),
                )
            conn.execute(
                "UPDATE cf_meta SET item_count = ?, last_accessed_at = ? "
                "WHERE workspace_id = ?",
                (new_count, time.time(), workspace_id),
            )
            self._fts_reindex(workspace_id)
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


def _execute_script(conn: sqlite3.Connection, script: str) -> None:
    """Run ``;``-separated DDL without ``executescript``'s implicit COMMIT."""
    for statement in script.split(";"):
        if statement.strip():
            conn.execute(statement)


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 1
