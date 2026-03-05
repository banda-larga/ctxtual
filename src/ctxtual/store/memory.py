"""
In-memory store — fast, zero dependencies, not persistent.

Suitable for single-process agents and testing.  Data is lost when the
process exits.  Thread-safe via a reentrant lock.
"""

import copy
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ctxtual.store.base import BaseStore
from ctxtual.types import WorkspaceMeta


class MemoryStore(BaseStore):
    """
    Stores everything in nested dicts protected by a reentrant lock.

    Args:
        max_workspaces: Optional cap on the number of live workspaces.
                        When exceeded the oldest workspace is evicted
                        (LRU by ``created_at``).

    Internal structure::

        _meta:  { workspace_id -> WorkspaceMeta }
        _data:  { workspace_id -> { key -> value } }
    """

    def __init__(self, *, max_workspaces: int | None = None) -> None:
        self._meta: dict[str, WorkspaceMeta] = {}
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.max_workspaces = max_workspaces

    # Lifecycle

    def init_workspace(self, meta: WorkspaceMeta) -> None:
        with self._lock:
            self._meta[meta.workspace_id] = meta
            self._data.setdefault(meta.workspace_id, {})
            self._maybe_evict()

    def drop_workspace(self, workspace_id: str) -> None:
        with self._lock:
            self._meta.pop(workspace_id, None)
            self._data.pop(workspace_id, None)

    # Metadata

    def get_meta(self, workspace_id: str) -> WorkspaceMeta | None:
        with self._lock:
            meta = self._meta.get(workspace_id)
            # Return a defensive copy to prevent external mutation of
            # internal state outside the lock.
            return copy.deepcopy(meta) if meta is not None else None

    def list_workspaces(self, workspace_type: str | None = None) -> list[str]:
        with self._lock:
            ids = sorted(self._meta.keys(), key=lambda k: self._meta[k].created_at)
            if workspace_type:
                ids = [k for k in ids if self._meta[k].workspace_type == workspace_type]
            return ids

    # Data I/O

    def set(self, workspace_id: str, key: str, value: Any) -> None:
        with self._lock:
            if workspace_id not in self._data:
                self._data[workspace_id] = {}
            self._data[workspace_id][key] = value

    def get(self, workspace_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(workspace_id, {}).get(key, default)

    def delete_key(self, workspace_id: str, key: str) -> None:
        with self._lock:
            self._data.get(workspace_id, {}).pop(key, None)

    # Eviction

    def _maybe_evict(self) -> None:
        """Drop the oldest workspace when ``max_workspaces`` is exceeded."""
        if self.max_workspaces is None:
            return
        while len(self._meta) > self.max_workspaces:
            oldest_id = min(self._meta, key=lambda k: self._meta[k].created_at)
            self._meta.pop(oldest_id, None)
            self._data.pop(oldest_id, None)

    def sweep_expired(self) -> list[str]:
        """Remove all workspaces that have exceeded their TTL."""
        with self._lock:
            dropped: list[str] = []
            for ws_id, meta in list(self._meta.items()):
                if meta.is_expired:
                    self._meta.pop(ws_id, None)
                    self._data.pop(ws_id, None)
                    dropped.append(ws_id)
            return dropped

    # Transaction support

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the lock for the entire block, preventing interleaving."""
        with self._lock:
            yield

    # Debug

    def __repr__(self) -> str:
        with self._lock:
            types = sorted({m.workspace_type for m in self._meta.values()})
            return f"MemoryStore(workspaces={len(self._meta)}, " f"types={types})"
