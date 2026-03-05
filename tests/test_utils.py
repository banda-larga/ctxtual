"""Tests for built-in utility ToolSets (paginator, text_search, filter_set, kv_reader)."""

import pytest

from ctxtual import Forge, MemoryStore
from ctxtual.types import WorkspaceMeta
from ctxtual.utils import filter_set, kv_reader, paginator, text_search


@pytest.fixture
def papers_forge(sample_papers: list[dict]) -> tuple[Forge, str]:
    """Forge with a pre-populated papers workspace."""
    forge = Forge(store=MemoryStore())
    ws_id = "papers_test"
    meta = WorkspaceMeta(
        workspace_id=ws_id, workspace_type="papers", item_count=len(sample_papers)
    )
    forge.store.init_workspace(meta)
    forge.store.set_items(ws_id, sample_papers)
    return forge, ws_id


# Paginator


class TestPaginator:
    def test_paginate_first_page(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = paginator(forge, "papers")
        resp = ts.tools["papers_paginate"](ws_id, page=0, size=2)
        result = resp["result"]  # unwrap hint envelope
        assert len(result["items"]) == 2
        assert result["page"] == 0
        assert result["total"] == 5
        assert result["total_pages"] == 3
        assert result["has_next"] is True
        assert result["has_prev"] is False

    def test_paginate_last_page(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = paginator(forge, "papers")
        resp = ts.tools["papers_paginate"](ws_id, page=2, size=2)
        result = resp["result"]  # unwrap hint envelope
        assert len(result["items"]) == 1  # 5 items, page 2 of size 2
        assert result["has_next"] is False
        assert result["has_prev"] is True

    def test_count(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = paginator(forge, "papers")
        result = ts.tools["papers_count"](ws_id)
        assert result["count"] == 5

    def test_get_item(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = paginator(forge, "papers")
        item = ts.tools["papers_get_item"](ws_id, index=0)
        assert item["title"] == "Attention Is All You Need"

    def test_get_item_out_of_range(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = paginator(forge, "papers")
        result = ts.tools["papers_get_item"](ws_id, index=999)
        assert "error" in result

    def test_get_slice(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = paginator(forge, "papers")
        items = ts.tools["papers_get_slice"](ws_id, start=1, end=3)
        assert len(items) == 2
        assert (
            items[0]["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"
        )


# Text Search


class TestTextSearch:
    def test_search_by_title(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = text_search(forge, "papers", fields=["title"])
        result = ts.tools["papers_search"](ws_id, query="transformer")
        assert result["total_matches"] >= 1
        titles = [m["title"] for m in result["matches"]]
        assert any("Transformer" in t for t in titles)

    def test_search_case_insensitive(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = text_search(forge, "papers", fields=["title"])
        result = ts.tools["papers_search"](ws_id, query="BERT")
        assert result["total_matches"] >= 1

    def test_search_no_results(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = text_search(forge, "papers", fields=["title"])
        result = ts.tools["papers_search"](ws_id, query="xyznonexistent")
        assert result["total_matches"] == 0
        assert result["matches"] == []

    def test_search_max_results(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = text_search(forge, "papers", fields=["abstract"])
        result = ts.tools["papers_search"](ws_id, query="language", max_results=1)
        assert len(result["matches"]) == 1

    def test_search_all_fields(self, papers_forge) -> None:
        """With fields=None, should search all dict values."""
        forge, ws_id = papers_forge
        ts = text_search(forge, "papers")  # no fields specified
        result = ts.tools["papers_search"](ws_id, query="Vaswani")
        assert result["total_matches"] >= 1

    def test_field_values(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = text_search(forge, "papers")
        result = ts.tools["papers_field_values"](ws_id, field="year")
        assert 2017 in result["distinct_values"]
        assert 2020 in result["distinct_values"]

    def test_field_values_list_field(self, papers_forge) -> None:
        """authors is a list — values should be flattened."""
        forge, ws_id = papers_forge
        ts = text_search(forge, "papers")
        result = ts.tools["papers_field_values"](ws_id, field="authors")
        assert "Vaswani" in result["distinct_values"]
        assert "Devlin" in result["distinct_values"]


# Filter Set


class TestFilterSet:
    def test_filter_eq(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = filter_set(forge, "papers")
        result = ts.tools["papers_filter_by"](ws_id, field="year", value=2020, operator="eq")
        assert result["count"] == 2
        assert all(p["year"] == 2020 for p in result["results"])

    def test_filter_gte(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = filter_set(forge, "papers")
        result = ts.tools["papers_filter_by"](ws_id, field="year", value=2020, operator="gte")
        assert result["count"] == 3  # 2020, 2020, 2022
        assert all(p["year"] >= 2020 for p in result["results"])

    def test_filter_contains(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = filter_set(forge, "papers")
        result = ts.tools["papers_filter_by"](
            ws_id, field="title", value="Transformer", operator="contains"
        )
        assert result["count"] >= 1

    def test_filter_no_match(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = filter_set(forge, "papers")
        result = ts.tools["papers_filter_by"](ws_id, field="year", value=1990, operator="eq")
        assert result["count"] == 0

    def test_sort_by_ascending(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = filter_set(forge, "papers")
        result = ts.tools["papers_sort_by"](ws_id, field="year", descending=False)
        years = [p["year"] for p in result]
        assert years == sorted(years)

    def test_sort_by_descending(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = filter_set(forge, "papers")
        result = ts.tools["papers_sort_by"](ws_id, field="citations", descending=True)
        citations = [p["citations"] for p in result]
        assert citations == sorted(citations, reverse=True)

    def test_sort_by_limit(self, papers_forge) -> None:
        forge, ws_id = papers_forge
        ts = filter_set(forge, "papers")
        result = ts.tools["papers_sort_by"](ws_id, field="citations", descending=True, limit=2)
        assert len(result) == 2
        assert result[0]["citations"] >= result[1]["citations"]


# KV Reader


class TestKVReader:
    def test_get_keys(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = kv_reader(forge, "config")
        meta = WorkspaceMeta(workspace_id="ws_cfg", workspace_type="config")
        forge.store.init_workspace(meta)
        forge.store.set_items(
            "ws_cfg", {"name": "app", "version": "1.0", "debug": True}
        )

        result = ts.tools["config_get_keys"]("ws_cfg")
        # output_hint wraps in {"result": [...], "_hint": "..."}
        keys = result["result"] if isinstance(result, dict) and "result" in result else result
        assert set(keys) == {"name", "version", "debug"}

    def test_get_value(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = kv_reader(forge, "config")
        meta = WorkspaceMeta(workspace_id="ws_cfg", workspace_type="config")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_cfg", {"name": "app", "version": "1.0"})

        assert ts.tools["config_get_value"]("ws_cfg", key="name") == "app"
        assert ts.tools["config_get_value"]("ws_cfg", key="version") == "1.0"

    def test_get_value_missing_key(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = kv_reader(forge, "config")
        meta = WorkspaceMeta(workspace_id="ws_cfg", workspace_type="config")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_cfg", {"name": "app"})

        result = ts.tools["config_get_value"]("ws_cfg", key="missing")
        assert "error" in result

    def test_get_keys_non_dict(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = kv_reader(forge, "data")
        meta = WorkspaceMeta(workspace_id="ws_list", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_list", [1, 2, 3])

        result = ts.tools["data_get_keys"]("ws_list")
        # output_hint wraps in {"result": [...], "_hint": "..."}
        keys = result["result"] if isinstance(result, dict) and "result" in result else result
        assert keys == []

    def test_get_value_non_dict(self) -> None:
        forge = Forge(store=MemoryStore())
        ts = kv_reader(forge, "data")
        meta = WorkspaceMeta(workspace_id="ws_list", workspace_type="data")
        forge.store.init_workspace(meta)
        forge.store.set_items("ws_list", [1, 2, 3])

        result = ts.tools["data_get_value"]("ws_list", key="anything")
        assert "error" in result
