"""Tests for core types: WorkspaceMeta, WorkspaceRef."""

import time

from ctxtual.types import WorkspaceMeta, WorkspaceRef


class TestWorkspaceMeta:
    def test_defaults(self) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="papers")
        assert meta.workspace_id == "ws_1"
        assert meta.workspace_type == "papers"
        assert meta.producer_fn == ""
        assert meta.producer_kwargs == {}
        assert meta.item_count == 0
        assert meta.extra == {}
        assert isinstance(meta.created_at, float)
        assert meta.created_at <= time.time()

    def test_full_construction(self) -> None:
        meta = WorkspaceMeta(
            workspace_id="ws_abc",
            workspace_type="employees",
            created_at=1000.0,
            producer_fn="load_employees",
            producer_kwargs={"dept": "eng"},
            item_count=42,
            extra={"source": "hr_db"},
        )
        assert meta.item_count == 42
        assert meta.extra["source"] == "hr_db"
        assert meta.created_at == 1000.0


class TestWorkspaceRef:
    def test_to_dict_structure(self) -> None:
        ref = WorkspaceRef(
            workspace_id="papers_abc",
            workspace_type="papers",
            item_count=100,
            producer_fn="search",
            available_tools=["paginate", "search"],
            metadata={"source": "arxiv"},
        )
        d = ref.to_dict()
        assert d["status"] == "workspace_ready"
        assert d["workspace_id"] == "papers_abc"
        assert d["workspace_type"] == "papers"
        assert d["item_count"] == 100
        assert "papers_abc" in d["message"]
        assert len(d["available_tools"]) == 2
        assert "paginate(workspace_id='papers_abc')" in d["available_tools"]
        assert d["metadata"] == {"source": "arxiv"}

    def test_to_dict_no_tools(self) -> None:
        ref = WorkspaceRef(
            workspace_id="ws_1",
            workspace_type="data",
            item_count=0,
        )
        d = ref.to_dict()
        assert d["available_tools"] == []

    def test_repr(self) -> None:
        ref = WorkspaceRef(
            workspace_id="ws_1",
            workspace_type="data",
            item_count=5,
        )
        r = repr(ref)
        assert "ws_1" in r
        assert "data" in r
        assert "5" in r

    def test_item_count_formatted(self) -> None:
        ref = WorkspaceRef(
            workspace_id="ws_big",
            workspace_type="logs",
            item_count=10_432,
        )
        d = ref.to_dict()
        assert "10,432" in d["message"]
