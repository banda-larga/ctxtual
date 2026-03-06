"""End-to-end integration tests — full producer → consumer → derived flows."""

from ctxtual import Ctx, MemoryStore
from ctxtual.ctx import ConsumerContext
from ctxtual.store.sqlite import SQLiteStore
from ctxtual.utils import filter_set, paginator, text_search


class TestEndToEnd:
    def test_full_flow_memory(self) -> None:
        """Producer stores, consumer reads, everything works with MemoryStore."""
        ctx = Ctx(store=MemoryStore())
        pager = paginator(ctx, "items")
        search = text_search(ctx, "items", fields=["name"])

        @ctx.producer(workspace_type="items", toolsets=[pager, search])
        def load_items() -> list:
            return [
                {"name": "Widget A", "price": 10},
                {"name": "Widget B", "price": 20},
                {"name": "Gadget C", "price": 30},
            ]

        result = load_items()
        assert result["status"] == "workspace_ready"
        assert result["item_count"] == 3
        ws_id = result["workspace_id"]

        # Paginate — output_hint wraps in {"result": {...}, "_hint": "..."}
        page_resp = pager.tools["items_paginate"](ws_id, page=0, size=2)
        page = page_resp["result"]
        assert len(page["items"]) == 2
        assert page["has_next"] is True

        # Search
        hits = search.tools["items_search"](ws_id, query="gadget")
        assert hits["total_matches"] == 1
        assert hits["matches"][0]["name"] == "Gadget C"

    def test_full_flow_sqlite(self, tmp_path) -> None:
        """Same flow with SQLiteStore."""
        store = SQLiteStore(tmp_path / "e2e.db")
        ctx = Ctx(store=store)
        pager = paginator(ctx, "items")

        @ctx.producer(workspace_type="items", toolsets=[pager])
        def load_items() -> list:
            return [{"id": i} for i in range(50)]

        result = load_items()
        ws_id = result["workspace_id"]
        assert result["item_count"] == 50

        page_resp = pager.tools["items_paginate"](ws_id, page=4, size=10)
        page = page_resp["result"]
        assert len(page["items"]) == 10
        assert page["items"][0]["id"] == 40

        store.close()

    def test_derived_workspace_flow(self) -> None:
        """Consumer produces a derived workspace from filtered results."""
        ctx = Ctx(store=MemoryStore())
        items_pager = paginator(ctx, "items")
        filtered_pager = paginator(ctx, "filtered")

        @ctx.producer(workspace_type="items", toolsets=[items_pager])
        def load() -> list:
            return [
                {"name": "A", "category": "x"},
                {"name": "B", "category": "y"},
                {"name": "C", "category": "x"},
                {"name": "D", "category": "x"},
            ]

        result = load()
        src_id = result["workspace_id"]

        @ctx.consumer(
            workspace_type="items",
            produces="filtered",
            produces_toolsets=[filtered_pager],
        )
        def filter_category(
            workspace_id: str,
            category: str,
            forge_ctx: ConsumerContext,
        ) -> dict:
            items = forge_ctx.get_items()
            filtered = [i for i in items if i["category"] == category]
            return forge_ctx.emit(filtered)

        derived = filter_category(src_id, category="x")
        assert derived["status"] == "workspace_ready"
        assert derived["item_count"] == 3

        derived_id = derived["workspace_id"]
        page_resp = filtered_pager.tools["filtered_paginate"](derived_id, page=0, size=10)
        page = page_resp["result"]
        assert len(page["items"]) == 3

    def test_multiple_producers_multiple_types(self) -> None:
        """Multiple workspace types coexist."""
        ctx = Ctx(store=MemoryStore())

        @ctx.producer(workspace_type="papers", key="papers_1")
        def load_papers() -> list:
            return [{"title": "Paper 1"}]

        @ctx.producer(workspace_type="employees", key="employees_1")
        def load_employees() -> list:
            return [{"name": "Alice"}, {"name": "Bob"}]

        load_papers()
        load_employees()

        assert ctx.list_workspaces("papers") == ["papers_1"]
        assert ctx.list_workspaces("employees") == ["employees_1"]
        assert len(ctx.list_workspaces()) == 2

    def test_overwrite_with_deterministic_key(self) -> None:
        """Re-running a producer with a deterministic key overwrites."""
        ctx = Ctx(store=MemoryStore())

        call_count = 0

        @ctx.producer(workspace_type="data", key="data_fixed")
        def load() -> list:
            nonlocal call_count
            call_count += 1
            return [call_count]

        load()
        load()
        assert ctx.store.get_items("data_fixed") == [2]
        # Still only 1 workspace
        assert ctx.list_workspaces() == ["data_fixed"]

    def test_workspace_with_transform_and_filter(self) -> None:
        """Transform at produce time, filter at consume time."""
        ctx = Ctx(store=MemoryStore())
        filt = filter_set(ctx, "data")

        @ctx.producer(
            workspace_type="data",
            toolsets=[filt],
            transform=lambda items: [
                {**i, "normalized_price": i["price"] / 100} for i in items
            ],
        )
        def load() -> list:
            return [
                {"name": "A", "price": 1000},
                {"name": "B", "price": 500},
                {"name": "C", "price": 2000},
            ]

        result = load()
        ws_id = result["workspace_id"]

        # Items should have the normalized_price field
        items = ctx.store.get_items(ws_id)
        assert all("normalized_price" in i for i in items)

        # Filter expensive items
        expensive = filt.tools["data_filter_by"](
            ws_id, field="price", value=1000, operator="gte"
        )
        assert expensive["count"] == 2
