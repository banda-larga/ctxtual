"""Tests for LLM-friendly error recovery (Fix #20).

Validates that error responses include contextual hints, available
workspaces, and suggested next actions.
"""

from ctxtual import Ctx, MemoryStore
from ctxtual.exceptions import (
    WorkspaceExpiredError,
    WorkspaceNotFoundError,
    WorkspaceTypeMismatchError,
)
from ctxtual.types import WorkspaceMeta
from ctxtual.utils import kv_reader, paginator

# ═══════════════════════════════════════════════════════════════════════════
# Exception to_llm_dict()
# ═══════════════════════════════════════════════════════════════════════════


class TestExceptionLLMDicts:
    def test_workspace_not_found_basic(self):
        exc = WorkspaceNotFoundError("ws_bad")
        d = exc.to_llm_dict()
        assert "error" in d
        assert "ws_bad" in d["error"]
        assert "suggested_action" in d

    def test_workspace_not_found_with_available(self):
        exc = WorkspaceNotFoundError("ws_bad", available=["ws_1", "ws_2"])
        d = exc.to_llm_dict()
        assert d["available_workspaces"] == ["ws_1", "ws_2"]
        assert "ws_bad" in d["error"]

    def test_workspace_expired_basic(self):
        exc = WorkspaceExpiredError("ws_old")
        d = exc.to_llm_dict()
        assert "expired" in d["error"]
        assert "suggested_action" in d

    def test_workspace_expired_with_producer(self):
        exc = WorkspaceExpiredError("ws_old", producer_fn="load_papers")
        d = exc.to_llm_dict()
        assert "load_papers" in d["suggested_action"]

    def test_workspace_type_mismatch_basic(self):
        exc = WorkspaceTypeMismatchError("ws_1", "papers", "employees")
        d = exc.to_llm_dict()
        assert "papers" in d["error"]
        assert "employees" in d["error"]
        assert d["expected_type"] == "papers"
        assert d["actual_type"] == "employees"
        assert "suggested_action" in d

    def test_workspace_type_mismatch_with_matching(self):
        exc = WorkspaceTypeMismatchError(
            "ws_1", "papers", "employees", matching_workspaces=["ws_p1", "ws_p2"]
        )
        d = exc.to_llm_dict()
        assert d["workspaces_of_correct_type"] == ["ws_p1", "ws_p2"]


# ═══════════════════════════════════════════════════════════════════════════
# dispatch_tool_call error recovery
# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchErrorRecovery:
    def test_unknown_tool_returns_suggestions(self):
        ctx = Ctx(store=MemoryStore())
        pager = paginator(ctx, "docs")

        @ctx.producer(workspace_type="docs", toolsets=[pager])
        def load() -> list:
            return [{"title": "Doc 1"}]

        load()

        result = ctx.dispatch_tool_call("nonexistent_tool", {})
        assert "error" in result
        assert "nonexistent_tool" in result["error"]
        assert "available_tools" in result
        assert len(result["available_tools"]) > 0
        assert "suggested_action" in result

    def test_workspace_not_found_via_dispatch(self):
        ctx = Ctx(store=MemoryStore())
        pager = paginator(ctx, "docs")

        @ctx.producer(workspace_type="docs", toolsets=[pager])
        def load() -> list:
            return [{"title": "Doc 1"}]

        load()

        result = ctx.dispatch_tool_call(
            "docs_paginate", {"workspace_id": "ws_nonexistent", "page": 0}
        )
        assert "error" in result
        assert "ws_nonexistent" in result["error"]
        assert "suggested_action" in result

    def test_workspace_type_mismatch_via_dispatch(self):
        ctx = Ctx(store=MemoryStore())
        papers_pager = paginator(ctx, "papers")
        employees_pager = paginator(ctx, "employees")

        @ctx.producer(workspace_type="papers", toolsets=[papers_pager])
        def load_papers() -> list:
            return [{"title": "Paper 1"}]

        @ctx.producer(workspace_type="employees", toolsets=[employees_pager])
        def load_employees() -> list:
            return [{"name": "Alice"}]

        papers_ref = load_papers()
        load_employees()

        # Use a papers workspace_id with the employees tool
        result = ctx.dispatch_tool_call(
            "employees_paginate",
            {"workspace_id": papers_ref["workspace_id"], "page": 0},
        )
        assert "error" in result
        assert "suggested_action" in result


# ═══════════════════════════════════════════════════════════════════════════
# Utils error messages
# ═══════════════════════════════════════════════════════════════════════════


class TestUtilsErrorMessages:
    def test_get_item_out_of_range(self):
        ctx = Ctx(store=MemoryStore())
        pager = paginator(ctx, "docs")
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="docs", item_count=3)
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_1", [{"a": 1}, {"a": 2}, {"a": 3}])

        result = pager.tools["docs_get_item"]("ws_1", index=99)
        assert "error" in result
        assert "valid_range" in result
        assert "suggested_action" in result
        assert "docs_paginate" in result["suggested_action"]

    def test_kv_missing_key(self):
        ctx = Ctx(store=MemoryStore())
        kv = kv_reader(ctx, "config")
        meta = WorkspaceMeta(
            workspace_id="ws_cfg", workspace_type="config", item_count=2
        )
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_cfg", {"host": "localhost", "port": 5432})

        result = kv.tools["config_get_value"]("ws_cfg", key="missing_key")
        assert "error" in result
        assert "missing_key" in result["error"]
        assert "available_keys" in result
        assert "host" in result["available_keys"]
        assert "port" in result["available_keys"]
        assert "suggested_action" in result

    def test_kv_non_dict_workspace(self):
        ctx = Ctx(store=MemoryStore())
        kv = kv_reader(ctx, "data")
        meta = WorkspaceMeta(
            workspace_id="ws_list", workspace_type="data", item_count=3
        )
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_list", [1, 2, 3])

        result = kv.tools["data_get_value"]("ws_list", key="anything")
        assert "error" in result
        assert "not a dict" in result["error"]
        assert "suggested_action" in result
