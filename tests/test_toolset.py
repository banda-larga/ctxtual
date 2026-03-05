"""Tests for ToolSet and BoundToolSet."""

import pytest

from ctxtual import Forge
from ctxtual.exceptions import WorkspaceNotFoundError, WorkspaceTypeMismatchError
from ctxtual.toolset import BoundToolSet, ToolSet
from ctxtual.types import WorkspaceMeta


class TestToolSetRegistration:
    def test_register_tool(self, forge: Forge) -> None:
        ts = forge.toolset("papers")

        @ts.tool
        def paginate(workspace_id: str, page: int = 0) -> list:
            return []

        assert "paginate" in ts.tool_names
        assert ts.tools["paginate"] is paginate

    def test_register_with_custom_name(self, forge: Forge) -> None:
        ts = forge.toolset("papers")

        @ts.tool(name="custom_page")
        def paginate(workspace_id: str) -> list:
            return []

        assert "custom_page" in ts.tool_names
        assert "paginate" not in ts.tool_names

    def test_get_tool(self, forge: Forge) -> None:
        ts = forge.toolset("data")

        @ts.tool
        def my_tool(workspace_id: str) -> str:
            return "ok"

        assert ts.get_tool("my_tool") is my_tool

    def test_get_tool_missing(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        with pytest.raises(KeyError, match="not_here"):
            ts.get_tool("not_here")


class TestToolSetValidation:
    def test_validates_workspace_exists(self, forge: Forge) -> None:
        ts = forge.toolset("papers")

        @ts.tool
        def read(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        with pytest.raises(WorkspaceNotFoundError):
            read("nonexistent_ws")

    def test_validates_workspace_type(self, forge: Forge) -> None:
        ts = forge.toolset("papers")  # expects type "papers"

        @ts.tool
        def read(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        # Create workspace of a different type
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="employees")
        forge.store.init_workspace(meta)

        with pytest.raises(WorkspaceTypeMismatchError):
            read("ws_1")

    def test_skip_validation(self, forge: Forge) -> None:
        ts = forge.toolset("papers")

        @ts.tool(validate_workspace=False)
        def raw_read(workspace_id: str) -> str:
            return "no validation"

        # Should not raise even though workspace doesn't exist
        assert raw_read("anything") == "no validation"

    def test_enforce_type_false(self, forge: Forge) -> None:
        ts = forge.toolset("any_type", enforce_type=False)

        @ts.tool
        def read(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        # Create workspace of type "papers" — should not raise
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="papers")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_1", [1, 2, 3])

        assert read("ws_1") == [1, 2, 3]

    def test_tool_receives_all_args(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_1", list(range(100)))

        @ts.tool
        def paginate(workspace_id: str, page: int = 0, size: int = 10) -> list:
            items = ts.store.get_items(workspace_id)
            start = page * size
            return items[start : start + size]

        result = paginate("ws_1", page=2, size=5)
        assert result == [10, 11, 12, 13, 14]


class TestToolSetStoreAccess:
    def test_store_not_attached_raises(self) -> None:
        ts = ToolSet("orphan")
        with pytest.raises(RuntimeError, match="not been attached"):
            _ = ts.store

    def test_store_attached_via_forge(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        assert ts.store is forge.store


class TestBoundToolSet:
    def test_bind_pre_fills_workspace_id(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_1", [10, 20, 30])

        @ts.tool
        def get_all(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        bound = ts.bind("ws_1")
        assert isinstance(bound, BoundToolSet)
        assert bound.workspace_id == "ws_1"
        assert bound.get_all() == [10, 20, 30]

    def test_bound_tool_names(self, forge: Forge) -> None:
        ts = forge.toolset("data")

        @ts.tool
        def tool_a(workspace_id: str) -> None:
            pass

        @ts.tool
        def tool_b(workspace_id: str) -> None:
            pass

        bound = ts.bind("ws_1")
        assert set(bound.tool_names) == {"tool_a", "tool_b"}

    def test_bound_repr(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        bound = ts.bind("ws_1")
        r = repr(bound)
        assert "BoundToolSet" in r
        assert "ws_1" in r


class TestToolSetRepr:
    def test_repr(self, forge: Forge) -> None:
        ts = forge.toolset("papers")

        @ts.tool
        def paginate(workspace_id: str) -> list:
            return []

        r = repr(ts)
        assert "ToolSet" in r
        assert "papers" in r
        assert "paginate" in r


class TestOutputHint:
    def test_output_hint_on_dict_result(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_1", [1, 2, 3])

        @ts.tool(
            output_hint="Call next_tool(workspace_id='{workspace_id}') to continue."
        )
        def my_tool(workspace_id: str) -> dict:
            return {"count": 3}

        result = my_tool("ws_1")
        # Hint is in envelope, original data is in "result"
        assert result["result"] == {"count": 3}
        assert "ws_1" in result["_hint"]
        assert "next_tool" in result["_hint"]

    def test_output_hint_on_list_result(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_1", ["a", "b"])

        @ts.tool(output_hint="Use get_value() to read each item.")
        def list_items(workspace_id: str) -> list:
            return ["key1", "key2"]

        result = list_items("ws_1")
        # List is preserved in "result", not wrapped in {"data": ...}
        assert result["result"] == ["key1", "key2"]
        assert "get_value()" in result["_hint"]

    def test_no_output_hint_by_default(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        forge.store.init_workspace(meta)

        @ts.tool
        def plain_tool(workspace_id: str) -> dict:
            return {"result": "ok"}

        result = plain_tool("ws_1")
        assert "_hint" not in result
        assert result == {"result": "ok"}

    def test_output_hint_workspace_id_placeholder(self, forge: Forge) -> None:
        ts = forge.toolset("data")
        meta = WorkspaceMeta(workspace_id="ws_abc", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_abc", [])

        @ts.tool(output_hint="page(workspace_id='{workspace_id}', page=1)")
        def tool_with_placeholder(workspace_id: str) -> dict:
            return {"items": []}

        result = tool_with_placeholder("ws_abc")
        assert "ws_abc" in result["_hint"]
        assert "{workspace_id}" not in result["_hint"]
        # Original data is preserved in "result" key
        assert result["result"] == {"items": []}
