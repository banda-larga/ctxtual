"""Tests for Ctx — producer, consumer, workspace management."""

import pytest

from ctxtual import Ctx, MemoryStore
from ctxtual.exceptions import WorkspaceNotFoundError
from ctxtual.ctx import ConsumerContext
from ctxtual.toolset import ToolSet
from ctxtual.types import WorkspaceMeta, WorkspaceRef

# Producer tests


class TestProducer:
    def test_basic_producer(self, ctx: Ctx, sample_papers: list) -> None:
        ts = ctx.toolset("papers")

        @ts.tool
        def paginate(workspace_id: str, page: int = 0) -> list:
            return ts.store.get_items(workspace_id)

        @ctx.producer(workspace_type="papers", toolsets=[ts])
        def fetch_papers(query: str) -> list:
            return sample_papers

        result = fetch_papers("ml")
        assert result["status"] == "workspace_ready"
        assert result["item_count"] == len(sample_papers)
        assert "workspace_id" in result
        ws_id = result["workspace_id"]
        assert ws_id.startswith("papers_")
        assert ctx.store.get_items(ws_id) == sample_papers

    def test_producer_with_transform(self, ctx: Ctx) -> None:
        @ctx.producer(
            workspace_type="nums",
            transform=lambda items: [x * 2 for x in items],
        )
        def get_numbers() -> list:
            return [1, 2, 3]

        result = get_numbers()
        ws_id = result["workspace_id"]
        assert ctx.store.get_items(ws_id) == [2, 4, 6]

    def test_producer_with_key_template(self, ctx: Ctx) -> None:
        @ctx.producer(
            workspace_type="data",
            key="data_{category}",
        )
        def fetch_data(category: str) -> list:
            return [1, 2, 3]

        result = fetch_data("sales")
        assert result["workspace_id"] == "data_sales"

    def test_producer_with_key_callable(self, ctx: Ctx) -> None:
        @ctx.producer(
            workspace_type="data",
            key=lambda kw: f"custom_{kw['name']}_{kw['year']}",
        )
        def fetch_data(name: str, year: int) -> list:
            return []

        result = fetch_data("report", 2024)
        assert result["workspace_id"] == "custom_report_2024"

    def test_producer_with_meta(self, ctx: Ctx) -> None:
        @ctx.producer(
            workspace_type="data",
            meta={"source": "api_v2", "region": "eu"},
        )
        def fetch_data() -> list:
            return [1]

        result = fetch_data()
        assert result["metadata"] == {"source": "api_v2", "region": "eu"}
        ws_meta = ctx.workspace_meta(result["workspace_id"])
        assert ws_meta is not None
        assert ws_meta.extra == {"source": "api_v2", "region": "eu"}

    def test_producer_notify_false(self, ctx: Ctx) -> None:
        @ctx.producer(workspace_type="data", notify=False)
        def fetch() -> list:
            return [1, 2]

        result = fetch()
        assert isinstance(result, WorkspaceRef)
        assert result.item_count == 2

    def test_producer_default_notify_false(self) -> None:
        ctx = Ctx(store=MemoryStore(), default_notify=False)

        @ctx.producer(workspace_type="data")
        def fetch() -> list:
            return [1]

        result = fetch()
        assert isinstance(result, WorkspaceRef)

    def test_producer_stores_metadata(self, ctx: Ctx) -> None:
        @ctx.producer(workspace_type="data")
        def load(query: str, limit: int = 100) -> list:
            return list(range(limit))

        result = load("test", limit=50)
        meta = ctx.workspace_meta(result["workspace_id"])
        assert meta is not None
        assert meta.producer_fn == "load"
        assert meta.producer_kwargs == {"query": "test", "limit": 50}
        assert meta.item_count == 50

    def test_producer_idempotent_key(self, ctx: Ctx) -> None:
        call_count = 0

        @ctx.producer(workspace_type="data", key="fixed_key")
        def load() -> list:
            nonlocal call_count
            call_count += 1
            return [call_count]

        result1 = load()
        result2 = load()
        assert result1["workspace_id"] == "fixed_key"
        assert result2["workspace_id"] == "fixed_key"
        # Second call overwrites
        assert ctx.store.get_items("fixed_key") == [2]

    def test_producer_requires_kwargs(self) -> None:
        ctx = Ctx()
        with pytest.raises(TypeError):

            @ctx.producer
            def bad_producer() -> list:
                return []

    def test_producer_multiple_toolsets(self, ctx: Ctx) -> None:
        ts1 = ctx.toolset("data")
        ts2 = ToolSet("data_extra", enforce_type=False)

        @ts1.tool
        def tool_a(workspace_id: str) -> str:
            return "a"

        @ts2.tool
        def tool_b(workspace_id: str) -> str:
            return "b"

        @ctx.producer(workspace_type="data", toolsets=[ts1, ts2])
        def load() -> list:
            return [1]

        result = load()
        tool_strs = result["available_tools"]
        tool_names = [t.split("(")[0] for t in tool_strs]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names

    def test_producer_deduplicates_toolsets(self, ctx: Ctx) -> None:
        ts = ctx.toolset("data")

        @ts.tool
        def my_tool(workspace_id: str) -> str:
            return "x"

        @ctx.producer(workspace_type="data", toolsets=[ts, ts, ts])
        def load() -> list:
            return [1]

        result = load()
        # Should only appear once
        assert len(result["available_tools"]) == 1


# Consumer tests


class TestConsumer:
    def test_consumer_receives_context(self, ctx: Ctx) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_1", [10, 20, 30])

        @ctx.consumer(workspace_type="data")
        def process(workspace_id: str, forge_ctx: ConsumerContext) -> list:
            return forge_ctx.get_items()

        result = process("ws_1")
        assert result == [10, 20, 30]

    def test_consumer_emit(self, ctx: Ctx) -> None:
        meta = WorkspaceMeta(workspace_id="ws_src", workspace_type="data")
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_src", [1, 2, 3, 4, 5])

        derived_ts = ctx.toolset("filtered")

        @ctx.consumer(
            workspace_type="data",
            produces="filtered",
            produces_toolsets=[derived_ts],
        )
        def filter_evens(workspace_id: str, forge_ctx: ConsumerContext) -> dict:
            items = forge_ctx.get_items()
            evens = [x for x in items if x % 2 == 0]
            return forge_ctx.emit(evens)

        result = filter_evens("ws_src")
        assert result["status"] == "workspace_ready"
        assert result["item_count"] == 2
        derived_id = result["workspace_id"]
        assert ctx.store.get_items(derived_id) == [2, 4]

    def test_consumer_validates_workspace(self, ctx: Ctx) -> None:
        @ctx.consumer(workspace_type="data")
        def process(workspace_id: str, forge_ctx: ConsumerContext) -> None:
            pass

        with pytest.raises(WorkspaceNotFoundError):
            process("nonexistent")

    def test_consumer_no_default_needed(self, ctx: Ctx) -> None:
        """forge_ctx works without ``= None`` default."""
        meta = WorkspaceMeta(workspace_id="ws_nd", workspace_type="data")
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_nd", [1, 2])

        @ctx.consumer(workspace_type="data")
        def read(workspace_id: str, forge_ctx: ConsumerContext) -> list:
            return forge_ctx.get_items()

        assert read("ws_nd") == [1, 2]

    def test_consumer_by_annotation(self, ctx: Ctx) -> None:
        """Context detected by ConsumerContext annotation, not just name."""
        meta = WorkspaceMeta(workspace_id="ws_ann", workspace_type="data")
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_ann", ["a", "b"])

        @ctx.consumer(workspace_type="data")
        def read(workspace_id: str, ctx: ConsumerContext) -> list:
            return ctx.get_items()

        assert read("ws_ann") == ["a", "b"]

    def test_consumer_direct_call(self, ctx: Ctx) -> None:
        """Consumer's __wrapped__ can be called directly with an explicit context."""
        meta = WorkspaceMeta(workspace_id="ws_dc", workspace_type="data")
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_dc", [10, 20])

        @ctx.consumer(workspace_type="data")
        def read(workspace_id: str, forge_ctx: ConsumerContext) -> list:
            return forge_ctx.get_items()

        ctx = ConsumerContext(ctx, "ws_dc")
        result = read.__wrapped__("ws_dc", forge_ctx=ctx)
        assert result == [10, 20]

    def test_consumer_context_defaults(self) -> None:
        """ConsumerContext can be constructed with minimal args for testing."""
        ctx = Ctx(store=MemoryStore())
        ctx = ConsumerContext(ctx)
        assert ctx.input_workspace_id == ""
        assert ctx._output_type is None
        assert ctx._output_toolsets == []


# Workspace management tests


class TestWorkspaceManagement:
    def test_list_workspaces(self, ctx: Ctx) -> None:
        @ctx.producer(workspace_type="papers", key="ws_p")
        def load_papers() -> list:
            return [1]

        @ctx.producer(workspace_type="employees", key="ws_e")
        def load_employees() -> list:
            return [2]

        load_papers()
        load_employees()

        all_ws = ctx.list_workspaces()
        assert "ws_p" in all_ws
        assert "ws_e" in all_ws

        papers = ctx.list_workspaces("papers")
        assert papers == ["ws_p"]

    def test_drop_workspace(self, ctx: Ctx) -> None:
        @ctx.producer(workspace_type="data", key="ws_drop")
        def load() -> list:
            return [1, 2, 3]

        load()
        assert ctx.workspace_meta("ws_drop") is not None
        ctx.drop_workspace("ws_drop")
        assert ctx.workspace_meta("ws_drop") is None

    def test_forge_repr(self, ctx: Ctx) -> None:
        r = repr(ctx)
        assert "Ctx" in r
        assert "MemoryStore" in r

    def test_toolset_retrieval(self, ctx: Ctx) -> None:
        ts1 = ctx.toolset("papers")
        ts2 = ctx.toolset("papers")
        assert ts1 is ts2  # Same instance


# Self-describing tools tests


class TestSystemPrompt:
    def test_system_prompt_with_preamble(self, ctx: Ctx) -> None:
        prompt = ctx.system_prompt(preamble="You are a helpful assistant.")
        assert "You are a helpful assistant." in prompt
        assert "workspace" in prompt.lower()

    def test_system_prompt_without_preamble(self, ctx: Ctx) -> None:
        prompt = ctx.system_prompt()
        assert "workspace" in prompt.lower()
        # Should not start with empty line
        assert not prompt.startswith("\n")

    def test_system_prompt_is_concise(self, ctx: Ctx) -> None:
        prompt = ctx.system_prompt(preamble="You are a research bot.")
        # Should be much shorter than the ~40-line manual prompts
        assert len(prompt) < 600


class TestProducerSchemas:
    def test_get_producer_schemas_basic(self, ctx: Ctx) -> None:
        @ctx.producer(workspace_type="data", key="k")
        def load_data(query: str, limit: int = 10) -> list:
            """Load data from the database."""
            return []

        schemas = ctx.get_producer_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "load_data"
        assert "Load data" in schema["function"]["description"]
        assert "workspace notification" in schema["function"]["description"]
        props = schema["function"]["parameters"]["properties"]
        assert "query" in props
        assert "limit" in props
        assert schema["function"]["parameters"]["required"] == ["query"]

    def test_get_producer_schemas_multiple(self, ctx: Ctx) -> None:
        @ctx.producer(workspace_type="a", key="a")
        def producer_a(x: str) -> list:
            return []

        @ctx.producer(workspace_type="b", key="b")
        def producer_b(y: int) -> list:
            return []

        schemas = ctx.get_producer_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "producer_a" in names
        assert "producer_b" in names

    def test_get_tools_combines_all(self, ctx: Ctx) -> None:
        ts = ctx.toolset("data")

        @ts.tool
        def read(workspace_id: str) -> list:
            return []

        @ctx.producer(workspace_type="data", toolsets=[ts], key="k")
        def fetch(q: str) -> list:
            return []

        tools = ctx.get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "fetch" in names
        assert "read" in names


class TestDispatchProducers:
    def test_dispatch_handles_producers(self, ctx: Ctx) -> None:
        @ctx.producer(workspace_type="data", key="ws_{q}")
        def search(q: str) -> list:
            return [1, 2, 3]

        result = ctx.dispatch_tool_call("search", {"q": "test"})
        assert result["status"] == "workspace_ready"
        assert result["workspace_id"] == "ws_test"
        assert result["item_count"] == 3

    def test_dispatch_handles_consumers(self, ctx: Ctx) -> None:
        ts = ctx.toolset("data")

        @ts.tool
        def count(workspace_id: str) -> int:
            items = ts.store.get_items(workspace_id)
            return len(items)

        from ctxtual.types import WorkspaceMeta

        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_1", [10, 20, 30])

        result = ctx.dispatch_tool_call("count", {"workspace_id": "ws_1"})
        assert result == 3

    def test_dispatch_unknown_returns_error(self, ctx: Ctx) -> None:
        result = ctx.dispatch_tool_call("nonexistent", {})
        assert "error" in result
        assert "not found" in result["error"]
        assert "suggested_action" in result
        assert "available_tools" in result


class TestWorkspaceRefNextSteps:
    def test_to_dict_includes_next_steps(self) -> None:
        ref = WorkspaceRef(
            workspace_id="ws_1",
            workspace_type="data",
            item_count=100,
            available_tools=["paginate", "search"],
            tool_descriptions={
                "paginate": "Page through results",
                "search": "Search within results",
            },
        )
        d = ref.to_dict()
        assert "next_steps" in d
        assert len(d["next_steps"]) == 2
        assert any("paginate" in s for s in d["next_steps"])
        assert any("search" in s for s in d["next_steps"])

    def test_to_dict_fallback_without_descriptions(self) -> None:
        ref = WorkspaceRef(
            workspace_id="ws_1",
            workspace_type="data",
            item_count=10,
            available_tools=["paginate"],
        )
        d = ref.to_dict()
        assert "next_steps" in d
        # Fallback should still mention the tool
        assert any("paginate" in s for s in d["next_steps"])

    def test_to_dict_no_metadata_when_empty(self) -> None:
        ref = WorkspaceRef(
            workspace_id="ws_1",
            workspace_type="data",
            item_count=10,
        )
        d = ref.to_dict()
        assert "metadata" not in d
