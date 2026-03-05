"""
Tests for production-readiness features:
  - TTL & workspace expiry
  - Error boundaries in tools (safe mode)
  - Async producer / consumer support
  - Payload size limits (max_items)
  - Schema export for LLM integration
  - Context manager protocol
  - Store.clear() and sweep_expired()
  - MemoryStore max_workspaces eviction
  - Forge.dispatch_tool_call()
  - WorkspaceRef.to_compact()
"""

import asyncio
import time

import pytest

from ctxtual import Forge, MemoryStore, SQLiteStore, WorkspaceExpiredError
from ctxtual.exceptions import PayloadTooLargeError
from ctxtual.forge import ConsumerContext
from ctxtual.toolset import ToolSet
from ctxtual.types import WorkspaceMeta, WorkspaceRef

# TTL & expiry


class TestTTL:
    def test_workspace_meta_is_expired(self) -> None:
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            created_at=time.time() - 100,
            ttl=50,
        )
        assert meta.is_expired is True

    def test_workspace_meta_not_expired(self) -> None:
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            ttl=3600,
        )
        assert meta.is_expired is False

    def test_workspace_meta_no_ttl_never_expires(self) -> None:
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            created_at=0.0,  # very old
            ttl=None,
        )
        assert meta.is_expired is False

    def test_memory_store_expired_workspace_still_visible(self) -> None:
        """get_meta returns the meta even if expired — callers check is_expired."""
        store = MemoryStore()
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            created_at=time.time() - 100,
            ttl=1,
        )
        store.init_workspace(meta)
        store.set_items("ws_1", [1, 2, 3])
        # get_meta returns the meta — it's the caller's job to check is_expired
        retrieved = store.get_meta("ws_1")
        assert retrieved is not None
        assert retrieved.is_expired is True

    def test_sqlite_store_expired_workspace_still_visible(self) -> None:
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            created_at=time.time() - 100,
            ttl=1,
        )
        store.init_workspace(meta)
        store.set_items("ws_1", [1, 2, 3])
        retrieved = store.get_meta("ws_1")
        assert retrieved is not None
        assert retrieved.is_expired is True
        store.close()

    def test_producer_with_ttl(self) -> None:
        forge = Forge(store=MemoryStore())

        @forge.producer(workspace_type="data", ttl=3600)
        def load() -> list:
            return [1, 2, 3]

        result = load()
        meta = forge.workspace_meta(result["workspace_id"])
        assert meta is not None
        assert meta.ttl == 3600

    def test_producer_inherits_default_ttl(self) -> None:
        forge = Forge(store=MemoryStore(), default_ttl=600)

        @forge.producer(workspace_type="data")
        def load() -> list:
            return [1]

        result = load()
        meta = forge.workspace_meta(result["workspace_id"])
        assert meta is not None
        assert meta.ttl == 600

    def test_producer_ttl_overrides_default(self) -> None:
        forge = Forge(store=MemoryStore(), default_ttl=600)

        @forge.producer(workspace_type="data", ttl=60)
        def load() -> list:
            return [1]

        result = load()
        meta = forge.workspace_meta(result["workspace_id"])
        assert meta is not None
        assert meta.ttl == 60

    def test_producer_ttl_none_overrides_default(self) -> None:
        forge = Forge(store=MemoryStore(), default_ttl=600)

        @forge.producer(workspace_type="data", ttl=None)
        def load() -> list:
            return [1]

        result = load()
        meta = forge.workspace_meta(result["workspace_id"])
        assert meta is not None
        assert meta.ttl is None

    def test_sweep_expired(self) -> None:
        store = MemoryStore()
        # Fresh workspace
        store.init_workspace(
            WorkspaceMeta(workspace_id="fresh", workspace_type="data", ttl=3600)
        )
        # Expired workspace
        store.init_workspace(
            WorkspaceMeta(
                workspace_id="stale",
                workspace_type="data",
                created_at=time.time() - 1000,
                ttl=1,
            )
        )
        dropped = store.sweep_expired()
        assert "stale" in dropped
        assert "fresh" not in dropped
        assert store.get_meta("stale") is None
        assert store.get_meta("fresh") is not None

    def test_forge_sweep_expired(self) -> None:
        forge = Forge(store=MemoryStore())
        forge.store.init_workspace(
            WorkspaceMeta(
                workspace_id="old",
                workspace_type="data",
                created_at=time.time() - 100,
                ttl=1,
            )
        )
        dropped = forge.sweep_expired()
        assert "old" in dropped

    def test_tool_on_expired_workspace_raises(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @ts.tool
        def read(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        meta = WorkspaceMeta(
            workspace_id="ws_old",
            workspace_type="data",
            created_at=time.time() - 100,
            ttl=1,
        )
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_old", [1, 2, 3])

        with pytest.raises(WorkspaceExpiredError):
            read("ws_old")


# Error boundaries


class TestErrorBoundaries:
    def test_safe_tool_catches_runtime_errors(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @ts.tool
        def bad_tool(workspace_id: str) -> list:
            raise ValueError("something broke")

        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)

        result = bad_tool("ws_1")
        assert "error" in result
        assert "ValueError" in result["error"]
        assert result["tool"] == "bad_tool"

    def test_safe_tool_propagates_workspace_errors(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @ts.tool
        def read(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        # Workspace doesn't exist — should still raise
        from ctxtual.exceptions import WorkspaceNotFoundError

        with pytest.raises(WorkspaceNotFoundError):
            read("nonexistent")

    def test_unsafe_tool_propagates_all_errors(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = ToolSet("data", safe=False)
        forge.register_toolset(ts)

        @ts.tool
        def bad_tool(workspace_id: str) -> list:
            raise ValueError("should propagate")

        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)

        with pytest.raises(ValueError, match="should propagate"):
            bad_tool("ws_1")

    def test_safe_tool_includes_workspace_id_in_error(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @ts.tool
        def divide(workspace_id: str, divisor: int = 0) -> float:
            return 1 / divisor

        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)

        result = divide("ws_1", divisor=0)
        assert result["workspace_id"] == "ws_1"


# Async support


class TestAsyncSupport:
    def test_async_producer(self) -> None:
        forge = Forge(store=MemoryStore())

        @forge.producer(workspace_type="data")
        async def async_load() -> list:
            await asyncio.sleep(0)
            return [1, 2, 3]

        result = asyncio.get_event_loop().run_until_complete(async_load())
        assert result["status"] == "workspace_ready"
        assert result["item_count"] == 3
        ws_id = result["workspace_id"]
        assert forge.store.get_items(ws_id) == [1, 2, 3]

    def test_async_producer_with_transform(self) -> None:
        forge = Forge(store=MemoryStore())

        @forge.producer(
            workspace_type="data",
            transform=lambda items: [x * 10 for x in items],
        )
        async def async_load() -> list:
            return [1, 2, 3]

        result = asyncio.get_event_loop().run_until_complete(async_load())
        ws_id = result["workspace_id"]
        assert forge.store.get_items(ws_id) == [10, 20, 30]

    def test_async_consumer(self) -> None:
        forge = Forge(store=MemoryStore())
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_1", [10, 20, 30])

        @forge.consumer(workspace_type="data")
        async def async_process(
            workspace_id: str, forge_ctx: ConsumerContext
        ) -> list:
            await asyncio.sleep(0)
            return forge_ctx.get_items()

        result = asyncio.get_event_loop().run_until_complete(async_process("ws_1"))
        assert result == [10, 20, 30]

    def test_async_producer_with_key(self) -> None:
        forge = Forge(store=MemoryStore())

        @forge.producer(workspace_type="data", key="data_{category}")
        async def async_load(category: str) -> list:
            return [1]

        result = asyncio.get_event_loop().run_until_complete(async_load("sales"))
        assert result["workspace_id"] == "data_sales"


# Payload size limits


class TestMaxItems:
    def test_max_items_enforced(self) -> None:
        forge = Forge(store=MemoryStore(), max_items=10)

        @forge.producer(workspace_type="data")
        def load() -> list:
            return list(range(100))

        with pytest.raises(PayloadTooLargeError) as exc_info:
            load()
        assert exc_info.value.count == 100
        assert exc_info.value.limit == 10

    def test_max_items_allows_within_limit(self) -> None:
        forge = Forge(store=MemoryStore(), max_items=100)

        @forge.producer(workspace_type="data")
        def load() -> list:
            return list(range(50))

        result = load()
        assert result["item_count"] == 50

    def test_max_items_none_means_unlimited(self) -> None:
        forge = Forge(store=MemoryStore())  # max_items=None

        @forge.producer(workspace_type="data")
        def load() -> list:
            return list(range(100_000))

        result = load()
        assert result["item_count"] == 100_000


# Schema export


class TestSchemaExport:
    def test_toolset_to_tool_schemas(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("papers")

        @ts.tool
        def paginate(workspace_id: str, page: int = 0, size: int = 10) -> list:
            """Return a page of papers."""
            return []

        schemas = ts.to_tool_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "paginate"
        assert "Return a page of papers." in schema["function"]["description"]
        params = schema["function"]["parameters"]
        assert "workspace_id" in params["properties"]
        assert "page" in params["properties"]
        assert "size" in params["properties"]
        assert "workspace_id" in params["required"]
        assert "page" not in params["required"]

    def test_toolset_schemas_with_workspace_id(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @ts.tool
        def read(workspace_id: str) -> list:
            """Read data."""
            return []

        schemas = ts.to_tool_schemas(workspace_id="ws_abc")
        assert "ws_abc" in schemas[0]["function"]["description"]

    def test_bound_toolset_schemas(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @ts.tool
        def read(workspace_id: str) -> list:
            """Read."""
            return []

        bound = ts.bind("ws_123")
        schemas = bound.to_tool_schemas()
        assert "ws_123" in schemas[0]["function"]["description"]

    def test_forge_get_all_tool_schemas(self) -> None:
        forge = Forge(store=MemoryStore())
        ts1 = forge.toolset("papers")
        ts2 = forge.toolset("employees")

        @ts1.tool
        def search_papers(workspace_id: str, query: str) -> list:
            """Search papers."""
            return []

        @ts2.tool
        def list_employees(workspace_id: str) -> list:
            """List employees."""
            return []

        schemas = forge.get_all_tool_schemas()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"search_papers", "list_employees"}

    def test_dispatch_tool_call(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_1", [10, 20, 30])

        @ts.tool
        def get_all(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        result = forge.dispatch_tool_call("get_all", {"workspace_id": "ws_1"})
        assert result == [10, 20, 30]

    def test_dispatch_tool_call_unknown(self) -> None:
        forge = Forge(store=MemoryStore())
        result = forge.dispatch_tool_call("not_real", {})
        assert "error" in result
        assert "not_real" in result["error"]
        assert "suggested_action" in result


# Context manager protocol


class TestContextManager:
    def test_forge_context_manager(self) -> None:
        with Forge(store=MemoryStore()) as forge:

            @forge.producer(workspace_type="data", key="ws_1")
            def load() -> list:
                return [1]

            load()
            assert forge.workspace_meta("ws_1") is not None
        # After __exit__, store.close() was called (no crash)

    def test_memory_store_context_manager(self) -> None:
        with MemoryStore() as store:
            meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
            store.init_workspace(meta)
        # No crash

    def test_sqlite_store_context_manager(self) -> None:
        with SQLiteStore(":memory:") as store:
            meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
            store.init_workspace(meta)
            assert store.get_meta("ws_1") is not None
        # Connection closed after __exit__


# Store.clear()


class TestStoreClear:
    def test_memory_store_clear(self) -> None:
        store = MemoryStore()
        for i in range(5):
            meta = WorkspaceMeta(
                workspace_id=f"ws_{i}", workspace_type="data", created_at=float(i)
            )
            store.init_workspace(meta)
            store.set_items(f"ws_{i}", [i])
        assert len(store.list_workspaces()) == 5
        store.clear()
        assert len(store.list_workspaces()) == 0

    def test_sqlite_store_clear(self) -> None:
        store = SQLiteStore(":memory:")
        for i in range(5):
            meta = WorkspaceMeta(
                workspace_id=f"ws_{i}", workspace_type="data", created_at=float(i)
            )
            store.init_workspace(meta)
        assert len(store.list_workspaces()) == 5
        store.clear()
        assert len(store.list_workspaces()) == 0
        store.close()

    def test_forge_clear(self) -> None:
        forge = Forge(store=MemoryStore())

        @forge.producer(workspace_type="data", key="ws_1")
        def load() -> list:
            return [1]

        load()
        assert len(forge.list_workspaces()) == 1
        forge.clear()
        assert len(forge.list_workspaces()) == 0


# MemoryStore max_workspaces eviction


class TestMemoryStoreEviction:
    def test_max_workspaces_evicts_oldest(self) -> None:
        store = MemoryStore(max_workspaces=3)
        for i in range(5):
            meta = WorkspaceMeta(
                workspace_id=f"ws_{i}", workspace_type="data", created_at=float(i)
            )
            store.init_workspace(meta)
            store.set_items(f"ws_{i}", [i])

        ws_ids = store.list_workspaces()
        assert len(ws_ids) == 3
        # Oldest (ws_0, ws_1) should be evicted
        assert "ws_0" not in ws_ids
        assert "ws_1" not in ws_ids
        assert "ws_4" in ws_ids

    def test_max_workspaces_none_means_unlimited(self) -> None:
        store = MemoryStore(max_workspaces=None)
        for i in range(100):
            meta = WorkspaceMeta(
                workspace_id=f"ws_{i}", workspace_type="data", created_at=float(i)
            )
            store.init_workspace(meta)
        assert len(store.list_workspaces()) == 100


# WorkspaceRef.to_compact()


class TestWorkspaceRefCompact:
    def test_to_compact_with_tools(self) -> None:
        ref = WorkspaceRef(
            workspace_id="papers_abc",
            workspace_type="papers",
            item_count=100,
            available_tools=["paginate", "search"],
        )
        c = ref.to_compact()
        assert "papers_abc" in c
        assert "100" in c
        assert "paginate" in c
        assert "search" in c

    def test_to_compact_no_tools(self) -> None:
        ref = WorkspaceRef(
            workspace_id="ws_1",
            workspace_type="data",
            item_count=0,
        )
        c = ref.to_compact()
        assert "none" in c


# WorkspaceMeta.touch()


class TestWorkspaceMetaTouch:
    def test_touch_updates_last_accessed_at(self) -> None:
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            last_accessed_at=0.0,
        )
        assert meta.last_accessed_at == 0.0
        meta.touch()
        assert meta.last_accessed_at > 0.0
        assert meta.last_accessed_at <= time.time()


# SQLite TTL persistence


class TestSQLiteTTL:
    def test_ttl_stored_and_retrieved(self) -> None:
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            ttl=300,
        )
        store.init_workspace(meta)
        retrieved = store.get_meta("ws_1")
        assert retrieved is not None
        assert retrieved.ttl == 300
        store.close()

    def test_ttl_none_stored_and_retrieved(self) -> None:
        store = SQLiteStore(":memory:")
        meta = WorkspaceMeta(
            workspace_id="ws_1",
            workspace_type="data",
            ttl=None,
        )
        store.init_workspace(meta)
        retrieved = store.get_meta("ws_1")
        assert retrieved is not None
        assert retrieved.ttl is None
        store.close()


# Integration: full flow with production features


class TestProductionIntegration:
    def test_full_flow_with_ttl_and_safe_tools(self) -> None:
        """End-to-end: produce with TTL, use safe tools, sweep expired."""
        forge = Forge(store=MemoryStore(), default_ttl=3600)
        ts = forge.toolset("items")

        @ts.tool
        def get_all(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        @ts.tool
        def bad_tool(workspace_id: str) -> dict:
            raise RuntimeError("oops")

        @forge.producer(workspace_type="items", toolsets=[ts])
        def load() -> list:
            return [{"name": "A"}, {"name": "B"}]

        result = load()
        ws_id = result["workspace_id"]

        # Normal tool works
        items = get_all(ws_id)
        assert len(items) == 2

        # Bad tool returns error dict instead of crashing
        err = bad_tool(ws_id)
        assert "error" in err
        assert "RuntimeError" in err["error"]

        # Schema export works
        schemas = forge.get_all_tool_schemas(workspace_id=ws_id)
        assert len(schemas) >= 2

        # Dispatch works
        dispatched = forge.dispatch_tool_call("get_all", {"workspace_id": ws_id})
        assert dispatched == items

    def test_schema_dispatch_roundtrip(self) -> None:
        """Simulate what happens in a real OpenAI function-calling loop."""
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @ts.tool
        def count_items(workspace_id: str) -> dict:
            """Count items in workspace."""
            items = ts.store.get_items(workspace_id)
            return {"count": len(items)}

        @forge.producer(workspace_type="data", toolsets=[ts], key="ws_main")
        def load() -> list:
            return list(range(42))

        result = load()
        ws_id = result["workspace_id"]

        # 1. Get schemas (what you'd pass to the LLM)
        schemas = forge.get_all_tool_schemas(workspace_id=ws_id)
        assert any(s["function"]["name"] == "count_items" for s in schemas)

        # 2. Simulate LLM response
        tool_call_name = "count_items"
        tool_call_args = {"workspace_id": ws_id}

        # 3. Dispatch
        output = forge.dispatch_tool_call(tool_call_name, tool_call_args)
        assert output == {"count": 42}
