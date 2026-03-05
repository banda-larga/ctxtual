"""Tests for concurrency safety (Fix #6)."""

import threading

import pytest

from ctxtual import Forge, MemoryStore, SQLiteStore
from ctxtual.types import WorkspaceMeta
from ctxtual.utils import paginator

# ═══════════════════════════════════════════════════════════════════════════
# Forge-level thread safety
# ═══════════════════════════════════════════════════════════════════════════


class TestForgeConcurrency:
    def test_concurrent_producer_calls(self):
        """Multiple threads calling producers simultaneously don't corrupt state."""
        forge = Forge(store=MemoryStore())
        paginator(forge, "docs")

        @forge.producer(workspace_type="docs")
        def make_docs(query: str) -> list[dict]:
            return [{"q": query, "i": i} for i in range(10)]

        results: list[dict] = []
        errors: list[Exception] = []

        def worker(q: str) -> None:
            try:
                ref = make_docs(query=q)
                results.append(ref)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"q{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        assert len(results) == 20
        # Each workspace should have 10 items
        for ref in results:
            wid = ref["workspace_id"]
            items = forge.store.get_items(wid)
            assert len(items) == 10

    def test_concurrent_toolset_registration(self):
        """Concurrent toolset() calls don't lose toolsets."""
        forge = Forge(store=MemoryStore())
        errors: list[Exception] = []

        def register(name: str) -> None:
            try:
                forge.toolset(name)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register, args=(f"ts_{i}",)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(forge._toolsets) == 20

    def test_concurrent_dispatch_and_produce(self):
        """Dispatch and produce can run concurrently without crashes."""
        forge = Forge(store=MemoryStore())
        pager = paginator(forge, "data")

        @forge.producer(workspace_type="data", toolsets=[pager])
        def make_data(query: str) -> list[dict]:
            return [{"v": i} for i in range(5)]

        # Create initial workspace
        ref = make_data(query="init")
        wid = ref["workspace_id"]

        errors: list[Exception] = []

        def dispatcher() -> None:
            try:
                for _ in range(10):
                    forge.dispatch_tool_call(
                        "data_paginate",
                        {"workspace_id": wid, "page": 0, "size": 2},
                    )
            except Exception as e:
                errors.append(e)

        def producer() -> None:
            try:
                for i in range(10):
                    make_data(query=f"concurrent_{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=dispatcher)
        t2 = threading.Thread(target=producer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors


# ═══════════════════════════════════════════════════════════════════════════
# MemoryStore concurrency
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryStoreConcurrency:
    def test_get_meta_returns_defensive_copy(self):
        """get_meta returns a copy — external mutation doesn't affect internal state."""
        store = MemoryStore()
        meta = WorkspaceMeta(workspace_id="copy_test", workspace_type="test")
        store.init_workspace(meta)

        retrieved = store.get_meta("copy_test")
        assert retrieved is not None
        original_time = retrieved.last_accessed_at

        # Mutate the retrieved copy
        retrieved.last_accessed_at = 999999.0
        retrieved.item_count = 42

        # Internal state should be unchanged
        internal = store.get_meta("copy_test")
        assert internal is not None
        assert internal.last_accessed_at == original_time
        assert internal.item_count == 0  # original default

    def test_transaction_prevents_interleaving(self):
        """Operations within transaction() are atomic."""
        store = MemoryStore()
        meta = WorkspaceMeta(workspace_id="tx_test", workspace_type="test")
        store.init_workspace(meta)

        barrier = threading.Barrier(2)
        results: list[list] = []

        def writer() -> None:
            with store.transaction():
                store.set_items("tx_test", [1, 2, 3])
                barrier.wait(timeout=2)
                # Other thread tries to read while we hold the lock
                store.set_items("tx_test", [4, 5, 6])

        def reader() -> None:
            barrier.wait(timeout=2)
            # This should block until writer's transaction completes
            items = store.get_items("tx_test")
            results.append(items)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Reader should see [4, 5, 6] (final state), not [1, 2, 3]
        assert len(results) == 1
        assert results[0] == [4, 5, 6]

    def test_concurrent_append_items(self):
        """Multiple threads appending items don't lose data."""
        store = MemoryStore()
        meta = WorkspaceMeta(
            workspace_id="append_test", workspace_type="test", item_count=0
        )
        store.init_workspace(meta)
        store.set_items("append_test", [])

        errors: list[Exception] = []

        def appender(start: int) -> None:
            try:
                with store.transaction():
                    store.append_items("append_test", [start, start + 1])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=appender, args=(i * 2,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        items = store.get_items("append_test")
        assert len(items) == 20  # 10 threads × 2 items each


# ═══════════════════════════════════════════════════════════════════════════
# SQLiteStore concurrency
# ═══════════════════════════════════════════════════════════════════════════


class TestSQLiteStoreConcurrency:
    def test_transaction_atomicity(self):
        """init_workspace + set_items within transaction() are atomic."""
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="atomic_test", workspace_type="test", item_count=3
        )

        with store.transaction():
            store.init_workspace(meta)
            store.set_items("atomic_test", [1, 2, 3])

        items = store.get_items("atomic_test")
        assert items == [1, 2, 3]

    def test_transaction_rollback_on_error(self):
        """Failed transaction rolls back all changes."""
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="rollback_test", workspace_type="test", item_count=3
        )
        store.init_workspace(meta)
        store.set_items("rollback_test", [1, 2, 3])

        with pytest.raises(ValueError, match="boom"), store.transaction():
            store.set_items("rollback_test", [10, 20, 30])
            raise ValueError("boom")

        # Data should be unchanged (rolled back)
        items = store.get_items("rollback_test")
        assert items == [1, 2, 3]

    def test_nested_transactions(self):
        """Nested transaction blocks only commit at outermost level."""
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="nested_test", workspace_type="test", item_count=0
        )

        with store.transaction():
            store.init_workspace(meta)
            with store.transaction():
                store.set_items("nested_test", [1, 2])
            # Inner transaction did NOT commit yet

        # Now outer committed
        items = store.get_items("nested_test")
        assert items == [1, 2]

    def test_commit_suppressed_inside_transaction(self):
        """Individual operations don't commit when inside a transaction block."""
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="suppress_test", workspace_type="test", item_count=5
        )

        with store.transaction():
            store.init_workspace(meta)
            store.set_items("suppress_test", [1, 2, 3, 4, 5])
            # _tx_depth should be > 0, suppressing per-op commits

        assert store._tx_depth == 0
        assert store.get_items("suppress_test") == [1, 2, 3, 4, 5]

    def test_operations_without_transaction_still_commit(self):
        """Without transaction(), each operation auto-commits as before."""
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="auto_test", workspace_type="test", item_count=3
        )
        store.init_workspace(meta)
        store.set_items("auto_test", [10, 20, 30])

        # Each op committed immediately
        assert store.get_items("auto_test") == [10, 20, 30]
