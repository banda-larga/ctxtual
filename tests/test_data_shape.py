"""Tests for producer-consumer data shape validation (Fix #18).

Validates that:
- ToolSets can declare expected data_shape
- WorkspaceMeta/WorkspaceRef track data_shape
- Producers log warnings when shape mismatches toolset expectation
- Tool calls return LLM-friendly errors on shape mismatch
"""

import logging

from ctxtual import Forge, MemoryStore
from ctxtual.toolset import ToolSet
from ctxtual.types import WorkspaceMeta
from ctxtual.utils import filter_set, kv_reader, paginator, text_search


class TestDataShapeTracking:
    """data_shape propagates through the system."""

    def test_list_producer_sets_shape(self):
        forge = Forge(store=MemoryStore())
        pager = paginator(forge, "docs")

        @forge.producer(workspace_type="docs", toolsets=[pager])
        def load():
            return [{"title": "A"}, {"title": "B"}]

        ref = load()
        assert ref["data_shape"] == "list"

    def test_dict_producer_sets_shape(self):
        forge = Forge(store=MemoryStore())
        kv = kv_reader(forge, "config")

        @forge.producer(workspace_type="config", toolsets=[kv])
        def load():
            return {"host": "localhost", "port": 5432}

        ref = load()
        assert ref["data_shape"] == "dict"

    def test_scalar_producer_sets_shape(self):
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("data")

        @forge.producer(workspace_type="data", toolsets=[ts])
        def load():
            return 42

        ref = load()
        assert ref["data_shape"] == "scalar"

    def test_meta_stores_shape(self):
        forge = Forge(store=MemoryStore())
        pager = paginator(forge, "docs")

        @forge.producer(workspace_type="docs", toolsets=[pager])
        def load():
            return [{"x": 1}]

        ref = load()
        meta = forge.store.get_meta(ref["workspace_id"])
        assert meta.data_shape == "list"

    def test_toolset_data_shape_attribute(self):
        forge = Forge(store=MemoryStore())
        assert paginator(forge, "a").data_shape == "list"
        assert text_search(forge, "b").data_shape == "list"
        assert filter_set(forge, "c").data_shape == "list"
        assert kv_reader(forge, "d").data_shape == "dict"

    def test_custom_toolset_no_shape(self):
        ts = ToolSet("custom")
        assert ts.data_shape is None


class TestShapeMismatchWarning:
    """Producers warn when toolset shape != payload shape."""

    def test_dict_with_list_toolset_warns(self, caplog):
        forge = Forge(store=MemoryStore())
        pager = paginator(forge, "stuff")

        @forge.producer(workspace_type="stuff", toolsets=[pager])
        def load():
            return {"key": "value"}

        with caplog.at_level(logging.WARNING, logger="ctx"):
            load()

        assert "dict" in caplog.text and "list" in caplog.text

    def test_list_with_kv_toolset_warns(self, caplog):
        forge = Forge(store=MemoryStore())
        kv = kv_reader(forge, "stuff")

        @forge.producer(workspace_type="stuff", toolsets=[kv])
        def load():
            return [{"a": 1}]

        with caplog.at_level(logging.WARNING, logger="ctx"):
            load()

        assert "list" in caplog.text and "dict" in caplog.text

    def test_matching_shape_no_warning(self, caplog):
        forge = Forge(store=MemoryStore())
        pager = paginator(forge, "docs")

        @forge.producer(workspace_type="docs", toolsets=[pager])
        def load():
            return [{"title": "A"}]

        with caplog.at_level(logging.WARNING, logger="ctx"):
            load()

        assert "data shape" not in caplog.text.lower() or caplog.text == ""


class TestShapeMismatchAtToolCall:
    """Tool execution returns LLM-friendly error on shape mismatch."""

    def test_paginate_on_dict_workspace(self):
        forge = Forge(store=MemoryStore())
        paginator(forge, "stuff")  # registers toolset

        # Manually store dict data with data_shape="dict"
        meta = WorkspaceMeta(
            workspace_id="ws_dict",
            workspace_type="stuff",
            item_count=2,
            data_shape="dict",
        )
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_dict", {"a": 1, "b": 2})

        result = forge.dispatch_tool_call(
            "stuff_paginate", {"workspace_id": "ws_dict", "page": 0}
        )
        assert "error" in result
        assert "shape" in result["error"].lower() or "mismatch" in result["error"].lower()
        assert result["expected_shape"] == "list"
        assert result["actual_shape"] == "dict"
        assert "suggested_action" in result

    def test_kv_on_list_workspace(self):
        forge = Forge(store=MemoryStore())
        kv_reader(forge, "stuff")  # registers toolset

        meta = WorkspaceMeta(
            workspace_id="ws_list",
            workspace_type="stuff",
            item_count=3,
            data_shape="list",
        )
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_list", [1, 2, 3])

        result = forge.dispatch_tool_call(
            "stuff_get_keys", {"workspace_id": "ws_list"}
        )
        assert "error" in result
        assert result["expected_shape"] == "dict"
        assert result["actual_shape"] == "list"

    def test_no_shape_on_meta_skips_validation(self):
        """Legacy workspaces without data_shape should not trigger validation."""
        forge = Forge(store=MemoryStore())
        paginator(forge, "docs")  # registers toolset

        # Legacy: data_shape="" (empty string)
        meta = WorkspaceMeta(
            workspace_id="ws_old",
            workspace_type="docs",
            item_count=2,
            data_shape="",
        )
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_old", [{"x": 1}, {"x": 2}])

        result = forge.dispatch_tool_call(
            "docs_paginate", {"workspace_id": "ws_old", "page": 0}
        )
        # Should succeed — no shape validation for legacy workspaces
        assert "error" not in result
        assert "result" in result  # wrapped in hint envelope

    def test_custom_toolset_no_shape_skips_validation(self):
        """ToolSets without data_shape should work with any payload."""
        forge = Forge(store=MemoryStore())
        ts = forge.toolset("mytype")

        @ts.tool(name="mytype_read")
        def read(workspace_id: str):
            return ts.store.get_items(workspace_id)

        @forge.producer(workspace_type="mytype", toolsets=[ts])
        def load():
            return {"key": "val"}

        ref = load()
        result = forge.dispatch_tool_call(
            "mytype_read", {"workspace_id": ref["workspace_id"]}
        )
        # No shape enforcement — should succeed
        assert result == {"key": "val"}
