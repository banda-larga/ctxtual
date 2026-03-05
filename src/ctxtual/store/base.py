"""
Abstract base class for ctx storage backends.

A Store is a multi-workspace key-value system. Each workspace is identified
by a string workspace_id and contains:
  - A primary payload under the key "items" (the bulk data from a producer)
  - Arbitrary sub-keys for toolset-specific derived data
  - A WorkspaceMeta record for introspection
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Self

from ctxtual.types import WorkspaceMeta

# Query helpers (used by default query-method implementations)


def _make_hashable(value: Any) -> Any:
    """Convert a value to something safe for set membership."""
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    if isinstance(value, list):
        return tuple(value)
    return value


def _item_matches(
    item: Any,
    query: str,
    fields: list[str] | None,
    case_sensitive: bool,
) -> bool:
    """Check whether an item matches a substring query."""
    return any(query in _c for _c in _item_candidates(item, fields, case_sensitive))


def _item_candidates(
    item: Any,
    fields: list[str] | None,
    case_sensitive: bool,
) -> list[str]:
    """Extract searchable text segments from an item."""
    if isinstance(item, dict):
        raw = (
            [str(item.get(f, "")) for f in fields]
            if fields
            else [str(v) for v in item.values()]
        )
    elif isinstance(item, str):
        raw = [item]
    else:
        raw = [str(item)]
    return raw if case_sensitive else [c.lower() for c in raw]


def _score_item(
    item: Any,
    terms: list[str],
    fields: list[str] | None,
    case_sensitive: bool,
) -> float:
    """Score an item by how many query terms match and how often.

    Returns 0.0 if no terms match.  Higher = more relevant.
    """
    candidates = _item_candidates(item, fields, case_sensitive)
    text = " ".join(candidates)
    if not text:
        return 0.0

    score = 0.0
    terms_matched = 0
    for term in terms:
        count = text.count(term)
        if count > 0:
            terms_matched += 1
            # Log-dampened term frequency
            score += 1.0 + math.log(count) if count > 1 else 1.0

    if terms_matched == 0:
        return 0.0

    # Bonus for matching all query terms (phrase-like)
    if terms_matched == len(terms):
        score *= 1.5

    return score


def _apply_op(actual: Any, expected: Any, op: str) -> bool:
    """Apply a comparison operator. Returns False on type errors."""
    try:
        if op == "eq":
            return actual == expected  # type: ignore[no-any-return]
        if op == "ne":
            return actual != expected  # type: ignore[no-any-return]
        if op == "lt":
            return actual < expected  # type: ignore[no-any-return]
        if op == "lte":
            return actual <= expected  # type: ignore[no-any-return]
        if op == "gt":
            return actual > expected  # type: ignore[no-any-return]
        if op == "gte":
            return actual >= expected  # type: ignore[no-any-return]
        if op == "contains":
            return expected in actual  # type: ignore[no-any-return]
        if op == "startswith":
            return str(actual).startswith(str(expected))
    except TypeError:
        return False
    return False


class BaseStore(ABC):
    """
    Minimal interface all store backends must implement.

    Workspaces are created via ``init_workspace`` and removed via
    ``drop_workspace``.  Data is stored as key/value pairs scoped to a
    workspace_id, with "items" being the canonical key for the primary
    payload.
    """

    # Lifecycle

    @abstractmethod
    def init_workspace(self, meta: WorkspaceMeta) -> None:
        """
        Register metadata for a workspace.
        Called by the producer decorator before writing items.
        If the workspace already exists, update the metadata (upsert).
        """

    @abstractmethod
    def drop_workspace(self, workspace_id: str) -> None:
        """Remove a workspace and all its data."""

    # Metadata

    @abstractmethod
    def get_meta(self, workspace_id: str) -> WorkspaceMeta | None:
        """Return workspace metadata, or ``None`` if not found."""

    @abstractmethod
    def list_workspaces(self, workspace_type: str | None = None) -> list[str]:
        """
        Return all workspace_ids, optionally filtered by type.
        Results are ordered oldest-first.
        """

    # Data I/O

    @abstractmethod
    def set(self, workspace_id: str, key: str, value: Any) -> None:
        """
        Store a value under ``(workspace_id, key)``.
        Overwrites any existing value for the same key.
        """

    @abstractmethod
    def get(self, workspace_id: str, key: str, default: Any = None) -> Any:
        """Retrieve a stored value. Returns *default* if not found."""

    @abstractmethod
    def delete_key(self, workspace_id: str, key: str) -> None:
        """Delete a specific key within a workspace (leaves the workspace alive)."""

    # Bulk helpers (default implementations using the primitives above)

    def set_items(self, workspace_id: str, items: Any) -> None:
        """
        Store the primary payload under the canonical ``'items'`` key.

        Also updates ``item_count`` in the workspace metadata so it stays
        in sync with the stored data.
        """
        self.set(workspace_id, "items", items)
        meta = self.get_meta(workspace_id)
        if meta is not None:
            try:
                meta.item_count = len(items)
            except TypeError:
                meta.item_count = 1
            meta.touch()
            self.init_workspace(meta)

    def get_items(self, workspace_id: str) -> Any:
        """Retrieve the primary payload (the ``'items'`` key)."""
        return self.get(workspace_id, "items", default=[])

    def workspace_exists(self, workspace_id: str) -> bool:
        """Check whether a workspace is present in the store."""
        return self.get_meta(workspace_id) is not None

    # TTL & cleanup

    def sweep_expired(self) -> list[str]:
        """
        Remove all workspaces that have exceeded their TTL.

        Returns the list of workspace_ids that were dropped.
        """
        dropped: list[str] = []
        for ws_id in self.list_workspaces():
            meta = self.get_meta(ws_id)
            if meta is not None and meta.is_expired:
                self.drop_workspace(ws_id)
                dropped.append(ws_id)
        return dropped

    def clear(self) -> None:
        """Remove **all** workspaces.  Useful for testing or reset."""
        for ws_id in list(self.list_workspaces()):
            self.drop_workspace(ws_id)

    # Query methods — override in subclasses for optimised implementations.
    # Defaults load all items and process in Python (correct but slow for
    # large datasets stored on disk).

    def count_items(self, workspace_id: str) -> int:
        """Return the number of items without loading the payload."""
        meta = self.get_meta(workspace_id)
        return meta.item_count if meta is not None else 0

    def get_page(self, workspace_id: str, offset: int, limit: int) -> list:
        """Return a slice of items ``[offset : offset+limit]``."""
        items = self.get_items(workspace_id)
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
        """Search workspace items with relevance scoring.

        Splits the query into terms, scores each item by term-frequency,
        and returns results sorted best-match-first.  Falls back to
        substring matching when a single term is used.
        """
        items = self.get_items(workspace_id)
        q = query if case_sensitive else query.lower()
        terms = q.split()

        if not terms:
            return []

        scored: list[tuple[float, int, Any]] = []
        for idx, item in enumerate(items):
            score = _score_item(item, terms, fields, case_sensitive)
            if score > 0:
                scored.append((score, idx, item))

        # Sort by score descending, then original order for ties
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [item for _, _, item in scored[:max_results]]

    def filter_items(
        self,
        workspace_id: str,
        field: str,
        value: Any,
        operator: str = "eq",
    ) -> list:
        """Filter items by a field value using an operator."""
        items = self.get_items(workspace_id)
        return [
            item
            for item in items
            if isinstance(item, dict) and _apply_op(item.get(field), value, operator)
        ]

    def sort_items(
        self,
        workspace_id: str,
        field: str,
        *,
        descending: bool = False,
        limit: int = 100,
    ) -> list:
        """Sort items by a field and return the top *limit* results."""
        items = self.get_items(workspace_id)
        sortable = [
            i for i in items if isinstance(i, dict) and i.get(field) is not None
        ]
        try:
            sorted_items = sorted(sortable, key=lambda x: x[field], reverse=descending)
        except TypeError:
            sorted_items = sorted(
                sortable, key=lambda x: str(x[field]), reverse=descending
            )
        return sorted_items[:limit]

    def distinct_field_values(
        self,
        workspace_id: str,
        field: str,
        *,
        max_values: int = 50,
    ) -> list:
        """Return distinct values for a given field across all items."""
        items = self.get_items(workspace_id)
        seen: set[Any] = set()
        values: list[Any] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            val = item.get(field)
            if val is None:
                continue
            if isinstance(val, list):
                for v in val:
                    hashable = _make_hashable(v)
                    if hashable not in seen and len(values) < max_values:
                        seen.add(hashable)
                        values.append(v)
            else:
                hashable = _make_hashable(val)
                if hashable not in seen and len(values) < max_values:
                    seen.add(hashable)
                    values.append(val)
        return values

    # Mutation methods — modify items in-place without full rewrite.
    # Override in subclasses for optimised implementations.

    def append_items(self, workspace_id: str, new_items: list) -> int:
        """
        Append items to a list workspace.

        Returns the new total item count.

        Raises:
            TypeError: if the existing payload is not a list.
        """
        existing = self.get_items(workspace_id)
        if not isinstance(existing, list):
            raise TypeError(
                f"Cannot append to workspace '{workspace_id}': "
                f"payload is {type(existing).__name__}, not list."
            )
        existing.extend(new_items)
        self.set_items(workspace_id, existing)
        return len(existing)

    def update_item(self, workspace_id: str, index: int, item: Any) -> None:
        """
        Replace a single item at the given index.

        Raises:
            TypeError:  if the existing payload is not a list.
            IndexError: if the index is out of range.
        """
        existing = self.get_items(workspace_id)
        if not isinstance(existing, list):
            raise TypeError(
                f"Cannot update item in workspace '{workspace_id}': "
                f"payload is {type(existing).__name__}, not list."
            )
        if not (0 <= index < len(existing)):
            raise IndexError(
                f"Index {index} out of range for workspace '{workspace_id}' "
                f"with {len(existing)} items."
            )
        existing[index] = item
        self.set_items(workspace_id, existing)

    def patch_item(self, workspace_id: str, index: int, fields: dict[str, Any]) -> None:
        """
        Merge *fields* into the dict item at the given index.

        Raises:
            TypeError:  if the payload is not a list or the item is not a dict.
            IndexError: if the index is out of range.
        """
        existing = self.get_items(workspace_id)
        if not isinstance(existing, list):
            raise TypeError(
                f"Cannot patch item in workspace '{workspace_id}': "
                f"payload is {type(existing).__name__}, not list."
            )
        if not (0 <= index < len(existing)):
            raise IndexError(
                f"Index {index} out of range for workspace '{workspace_id}' "
                f"with {len(existing)} items."
            )
        if not isinstance(existing[index], dict):
            raise TypeError(
                f"Cannot patch item at index {index}: "
                f"item is {type(existing[index]).__name__}, not dict."
            )
        existing[index].update(fields)
        self.set_items(workspace_id, existing)

    def delete_items(self, workspace_id: str, indices: list[int]) -> int:
        """
        Remove items at the given indices.

        Returns the new total item count.

        Raises:
            TypeError: if the existing payload is not a list.
        """
        existing = self.get_items(workspace_id)
        if not isinstance(existing, list):
            raise TypeError(
                f"Cannot delete items from workspace '{workspace_id}': "
                f"payload is {type(existing).__name__}, not list."
            )
        to_remove = set(indices)
        new_items = [item for i, item in enumerate(existing) if i not in to_remove]
        self.set_items(workspace_id, new_items)
        return len(new_items)

    # Transaction support

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Context manager for grouping multiple operations atomically.

        Within a transaction block, stores that support it will defer commits
        and hold locks to prevent interleaving from other threads.

        Default implementation is a no-op (for backward compatibility with
        custom stores).  Override in subclasses for real atomicity.
        """
        yield

    # Context-manager protocol

    def close(self) -> None:  # noqa: B027
        """Release any resources held by the store.  Override in subclasses."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
