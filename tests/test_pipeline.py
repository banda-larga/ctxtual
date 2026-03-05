"""Tests for the pipeline engine and pipeline utility ToolSet."""

import pytest

from ctxtual import Forge, MemoryStore
from ctxtual.pipeline import PipelineEngine, PipelineError, compute_aggregates
from ctxtual.utils import pipeline

# ── Sample data ───────────────────────────────────────────────────────────

PAPERS = [
    {"id": 1, "title": "Deep Learning Basics", "author": "Alice", "year": 2021, "citations": 150, "tags": ["ml", "deep-learning"]},
    {"id": 2, "title": "Transformer Architecture", "author": "Bob", "year": 2022, "citations": 300, "tags": ["ml", "transformers"]},
    {"id": 3, "title": "Quantum Computing 101", "author": "Carol", "year": 2023, "citations": 50, "tags": ["quantum"]},
    {"id": 4, "title": "ML in Production", "author": "Alice", "year": 2023, "citations": 200, "tags": ["ml", "production"]},
    {"id": 5, "title": "Reinforcement Learning", "author": "Bob", "year": 2021, "citations": 120, "tags": ["ml", "rl"]},
    {"id": 6, "title": "Graph Neural Networks", "author": "Dave", "year": 2022, "citations": 80, "tags": ["ml", "graphs"]},
    {"id": 7, "title": "NLP with Transformers", "author": "Eve", "year": 2023, "citations": 250, "tags": ["ml", "nlp", "transformers"]},
    {"id": 8, "title": "Computer Vision Survey", "author": "Alice", "year": 2022, "citations": 180, "tags": ["ml", "cv"]},
    {"id": 9, "title": "Quantum ML Intersection", "author": "Carol", "year": 2023, "citations": 30, "tags": ["quantum", "ml"]},
    {"id": 10, "title": "Ethics in AI", "author": "Frank", "year": 2021, "citations": 90, "tags": ["ethics", "ml"]},
]


@pytest.fixture()
def engine():
    return PipelineEngine()


@pytest.fixture()
def forge_with_pipeline():
    forge = Forge(store=MemoryStore())
    pipe_ts = pipeline(forge, "papers")

    @forge.producer(workspace_type="papers", toolsets=[pipe_ts])
    def load_papers():
        return list(PAPERS)

    ref = load_papers()
    return forge, ref, pipe_ts


# ═══════════════════════════════════════════════════════════════════════════
# PipelineEngine unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineEngineFilter:
    def test_equality_filter(self, engine):
        result = engine.execute(PAPERS, [{"filter": {"author": "Alice"}}])
        assert len(result) == 3
        assert all(p["author"] == "Alice" for p in result)

    def test_gt_filter(self, engine):
        result = engine.execute(PAPERS, [{"filter": {"citations": {"$gt": 100}}}])
        assert all(p["citations"] > 100 for p in result)
        assert len(result) == 6

    def test_gte_lte_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"citations": {"$gte": 100, "$lte": 200}}}]
        )
        assert all(100 <= p["citations"] <= 200 for p in result)

    def test_in_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"year": {"$in": [2021, 2023]}}}]
        )
        assert all(p["year"] in (2021, 2023) for p in result)

    def test_nin_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"author": {"$nin": ["Alice", "Bob"]}}}]
        )
        assert all(p["author"] not in ("Alice", "Bob") for p in result)

    def test_ne_filter(self, engine):
        result = engine.execute(PAPERS, [{"filter": {"author": {"$ne": "Alice"}}}])
        assert all(p["author"] != "Alice" for p in result)

    def test_contains_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"title": {"$contains": "ML"}}}]
        )
        assert all("ML" in p["title"] for p in result)

    def test_startswith_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"title": {"$startswith": "Quantum"}}}]
        )
        assert all(p["title"].startswith("Quantum") for p in result)

    def test_regex_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"title": {"$regex": r"^(Deep|Graph)"}}}]
        )
        assert len(result) == 2

    def test_exists_filter(self, engine):
        data = [{"a": 1, "b": 2}, {"a": 3}, {"b": 4}]
        result = engine.execute(data, [{"filter": {"b": {"$exists": True}}}])
        assert len(result) == 2

    def test_or_filter(self, engine):
        result = engine.execute(
            PAPERS,
            [{"filter": {"$or": [{"author": "Alice"}, {"author": "Bob"}]}}],
        )
        assert all(p["author"] in ("Alice", "Bob") for p in result)
        assert len(result) == 5

    def test_and_filter(self, engine):
        result = engine.execute(
            PAPERS,
            [{"filter": {"$and": [{"author": "Alice"}, {"year": 2023}]}}],
        )
        assert len(result) == 1
        assert result[0]["title"] == "ML in Production"

    def test_not_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"$not": {"author": "Alice"}}}]
        )
        assert len(result) == 7

    def test_dot_notation(self, engine):
        data = [
            {"name": "x", "meta": {"score": 10}},
            {"name": "y", "meta": {"score": 5}},
            {"name": "z", "meta": {"score": 20}},
        ]
        result = engine.execute(
            data, [{"filter": {"meta.score": {"$gte": 10}}}]
        )
        assert len(result) == 2


class TestPipelineEngineSearch:
    def test_search_string(self, engine):
        result = engine.execute(PAPERS, [{"search": "quantum"}])
        assert len(result) == 2

    def test_search_dict(self, engine):
        result = engine.execute(
            PAPERS, [{"search": {"query": "transformer"}}]
        )
        assert len(result) == 2

    def test_search_with_fields(self, engine):
        result = engine.execute(
            PAPERS, [{"search": {"query": "alice", "fields": ["author"]}}]
        )
        assert len(result) == 3

    def test_search_no_match(self, engine):
        result = engine.execute(PAPERS, [{"search": "nonexistent"}])
        assert len(result) == 0


class TestPipelineEngineSort:
    def test_sort_asc(self, engine):
        result = engine.execute(
            PAPERS, [{"sort": {"field": "citations"}}]
        )
        citations = [p["citations"] for p in result]
        assert citations == sorted(citations)

    def test_sort_desc(self, engine):
        result = engine.execute(
            PAPERS, [{"sort": {"field": "citations", "order": "desc"}}]
        )
        citations = [p["citations"] for p in result]
        assert citations == sorted(citations, reverse=True)

    def test_sort_string_shorthand(self, engine):
        result = engine.execute(PAPERS, [{"sort": "author"}])
        authors = [p["author"] for p in result]
        assert authors == sorted(authors)

    def test_multi_sort(self, engine):
        result = engine.execute(
            PAPERS,
            [{"sort": [{"field": "year"}, {"field": "citations", "order": "desc"}]}],
        )
        # Within each year, citations should be descending
        for year in [2021, 2022, 2023]:
            year_papers = [p for p in result if p["year"] == year]
            cites = [p["citations"] for p in year_papers]
            assert cites == sorted(cites, reverse=True)

    def test_sort_with_none_values(self, engine):
        data = [{"a": 3}, {"a": None}, {"a": 1}]
        result = engine.execute(data, [{"sort": {"field": "a"}}])
        # None sorts first
        assert result[0]["a"] is None


class TestPipelineEngineProjection:
    def test_select(self, engine):
        result = engine.execute(
            PAPERS, [{"select": ["title", "author"]}]
        )
        for item in result:
            assert set(item.keys()) == {"title", "author"}

    def test_exclude(self, engine):
        result = engine.execute(
            PAPERS, [{"exclude": ["tags", "citations"]}]
        )
        for item in result:
            assert "tags" not in item
            assert "citations" not in item
            assert "title" in item


class TestPipelineEngineSlicing:
    def test_limit(self, engine):
        result = engine.execute(PAPERS, [{"limit": 3}])
        assert len(result) == 3

    def test_skip(self, engine):
        result = engine.execute(PAPERS, [{"skip": 8}])
        assert len(result) == 2

    def test_slice(self, engine):
        result = engine.execute(PAPERS, [{"slice": [2, 5]}])
        assert len(result) == 3
        assert result[0] == PAPERS[2]

    def test_sample_int(self, engine):
        result = engine.execute(PAPERS, [{"sample": 3}])
        assert len(result) == 3

    def test_sample_with_seed(self, engine):
        r1 = engine.execute(PAPERS, [{"sample": {"n": 5, "seed": 42}}])
        r2 = engine.execute(PAPERS, [{"sample": {"n": 5, "seed": 42}}])
        assert r1 == r2

    def test_sample_capped_at_length(self, engine):
        result = engine.execute(PAPERS, [{"sample": 999}])
        assert len(result) == len(PAPERS)


class TestPipelineEngineUnique:
    def test_unique_by_field(self, engine):
        result = engine.execute(PAPERS, [{"unique": "author"}])
        authors = [p["author"] for p in result]
        assert len(authors) == len(set(authors))
        # Should keep first occurrence
        assert result[0]["author"] == "Alice"

    def test_unique_by_year(self, engine):
        result = engine.execute(PAPERS, [{"unique": "year"}])
        assert len(result) == 3  # 2021, 2022, 2023


class TestPipelineEngineFlatten:
    def test_flatten_list_field(self, engine):
        result = engine.execute(PAPERS, [{"flatten": "tags"}])
        # Each paper's tags expanded into separate items
        total_tags = sum(len(p["tags"]) for p in PAPERS)
        assert len(result) == total_tags
        # Each result has a single tag string, not a list
        for item in result:
            assert isinstance(item["tags"], str)


class TestPipelineEngineGroupBy:
    def test_group_by_count(self, engine):
        result = engine.execute(
            PAPERS,
            [{"group_by": {"field": "year", "metrics": {"n": "count"}}}],
        )
        assert isinstance(result, list)
        year_counts = {g["year"]: g["n"] for g in result}
        assert year_counts[2021] == 3
        assert year_counts[2022] == 3
        assert year_counts[2023] == 4

    def test_group_by_sum_mean(self, engine):
        result = engine.execute(
            PAPERS,
            [
                {
                    "group_by": {
                        "field": "author",
                        "metrics": {
                            "n": "count",
                            "total_cites": "sum:citations",
                            "avg_cites": "mean:citations",
                        },
                    }
                }
            ],
        )
        alice = next(g for g in result if g["author"] == "Alice")
        assert alice["n"] == 3
        assert alice["total_cites"] == 150 + 200 + 180
        assert alice["avg_cites"] == pytest.approx((150 + 200 + 180) / 3)

    def test_group_by_min_max(self, engine):
        result = engine.execute(
            PAPERS,
            [
                {
                    "group_by": {
                        "field": "year",
                        "metrics": {
                            "min_cites": "min:citations",
                            "max_cites": "max:citations",
                        },
                    }
                }
            ],
        )
        y2023 = next(g for g in result if g["year"] == 2023)
        assert y2023["min_cites"] == 30
        assert y2023["max_cites"] == 250

    def test_group_by_values(self, engine):
        result = engine.execute(
            PAPERS,
            [
                {
                    "group_by": {
                        "field": "year",
                        "metrics": {"authors": "values:author"},
                    }
                }
            ],
        )
        y2021 = next(g for g in result if g["year"] == 2021)
        assert set(y2021["authors"]) == {"Alice", "Bob", "Frank"}

    def test_group_by_median(self, engine):
        result = engine.execute(
            PAPERS,
            [
                {
                    "group_by": {
                        "field": "year",
                        "metrics": {"med": "median:citations"},
                    }
                }
            ],
        )
        # 2021: [90, 120, 150] → median = 120
        y2021 = next(g for g in result if g["year"] == 2021)
        assert y2021["med"] == 120


class TestPipelineEngineCount:
    def test_count(self, engine):
        result = engine.execute(PAPERS, [{"count": True}])
        assert result == {"count": 10}

    def test_count_after_filter(self, engine):
        result = engine.execute(
            PAPERS, [{"filter": {"author": "Alice"}}, {"count": True}]
        )
        assert result == {"count": 3}


class TestPipelineEngineCompound:
    """Multi-step pipelines — the core use case."""

    def test_search_filter_sort_limit(self, engine):
        """The canonical example: compound ops in one call."""
        result = engine.execute(
            PAPERS,
            [
                {"search": {"query": "ml", "fields": ["tags"]}},
                {"filter": {"year": {"$gte": 2022}}},
                {"sort": {"field": "citations", "order": "desc"}},
                {"limit": 3},
            ],
        )
        assert len(result) == 3
        assert result[0]["citations"] >= result[1]["citations"]
        assert all(p["year"] >= 2022 for p in result)

    def test_filter_select_unique(self, engine):
        result = engine.execute(
            PAPERS,
            [
                {"filter": {"year": {"$gte": 2022}}},
                {"select": ["author"]},
                {"unique": "author"},
            ],
        )
        authors = [r["author"] for r in result]
        assert len(authors) == len(set(authors))

    def test_flatten_group_count(self, engine):
        """Expand tags, group by tag, count per tag."""
        result = engine.execute(
            PAPERS,
            [
                {"flatten": "tags"},
                {"group_by": {"field": "tags", "metrics": {"count": "count"}}},
                {"sort": {"field": "count", "order": "desc"}},
            ],
        )
        # "ml" should be the most common tag
        assert result[0]["tags"] == "ml"
        assert result[0]["count"] >= 7

    def test_filter_group_aggregate(self, engine):
        """Filter to 2023, group by author, compute stats."""
        result = engine.execute(
            PAPERS,
            [
                {"filter": {"year": 2023}},
                {
                    "group_by": {
                        "field": "author",
                        "metrics": {
                            "papers": "count",
                            "total_cites": "sum:citations",
                        },
                    }
                },
                {"sort": {"field": "total_cites", "order": "desc"}},
            ],
        )
        # Carol has 2 papers in 2023
        carol = next(g for g in result if g["author"] == "Carol")
        assert carol["papers"] == 2


class TestPipelineEngineErrors:
    def test_unknown_operation(self, engine):
        with pytest.raises(PipelineError, match="Unknown operation"):
            engine.execute(PAPERS, [{"magic": True}])

    def test_invalid_step_format(self, engine):
        with pytest.raises(PipelineError, match="exactly one key"):
            engine.execute(PAPERS, [{"filter": {}, "sort": {}}])

    def test_steps_not_list(self, engine):
        with pytest.raises(PipelineError, match="steps must be a list"):
            engine.execute(PAPERS, "not a list")

    def test_filter_on_non_list(self, engine):
        with pytest.raises(PipelineError, match="filter requires list"):
            engine.execute({"key": "value"}, [{"filter": {"key": "value"}}])

    def test_unknown_comparison_op(self, engine):
        with pytest.raises(PipelineError, match="Unknown comparison"):
            engine.execute(PAPERS, [{"filter": {"year": {"$magic": 5}}}])

    def test_unknown_metric_spec(self, engine):
        with pytest.raises(PipelineError, match="Unknown metric"):
            engine.execute(
                PAPERS, [{"group_by": {"field": "year", "metrics": {"x": "bad"}}}]
            )

    def test_unknown_agg_function(self, engine):
        with pytest.raises(PipelineError, match="Unknown aggregation"):
            engine.execute(
                PAPERS,
                [{"group_by": {"field": "year", "metrics": {"x": "magic:citations"}}}],
            )


# ═══════════════════════════════════════════════════════════════════════════
# compute_aggregates standalone function
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeAggregates:
    def test_flat_aggregation(self):
        result = compute_aggregates(
            PAPERS,
            {"n": "count", "total": "sum:citations", "avg": "mean:citations"},
        )
        assert result["n"] == 10
        assert result["total"] == sum(p["citations"] for p in PAPERS)
        assert result["avg"] == pytest.approx(result["total"] / 10)

    def test_grouped_aggregation(self):
        result = compute_aggregates(
            PAPERS,
            {"n": "count", "total": "sum:citations"},
            group_by="year",
        )
        assert isinstance(result, list)
        y2023 = next(g for g in result if g["year"] == 2023)
        assert y2023["n"] == 4


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline utility ToolSet integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineToolSet:
    def test_pipe_tool_exists(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        assert "papers_pipe" in ts.tool_names
        assert "papers_aggregate" in ts.tool_names

    def test_pipe_basic(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        wid = ref["workspace_id"]
        result = forge.dispatch_tool_call(
            "papers_pipe",
            {
                "workspace_id": wid,
                "steps": [
                    {"filter": {"author": "Alice"}},
                    {"sort": {"field": "citations", "order": "desc"}},
                ],
            },
        )
        assert result["count"] == 3
        assert result["items"][0]["citations"] >= result["items"][1]["citations"]

    def test_pipe_with_limit(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        wid = ref["workspace_id"]
        result = forge.dispatch_tool_call(
            "papers_pipe",
            {
                "workspace_id": wid,
                "steps": [
                    {"sort": {"field": "citations", "order": "desc"}},
                    {"limit": 3},
                    {"select": ["title", "citations"]},
                ],
            },
        )
        assert result["count"] == 3
        for item in result["items"]:
            assert set(item.keys()) == {"title", "citations"}

    def test_pipe_count_terminal(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        wid = ref["workspace_id"]
        result = forge.dispatch_tool_call(
            "papers_pipe",
            {
                "workspace_id": wid,
                "steps": [{"filter": {"year": 2023}}, {"count": True}],
            },
        )
        assert result["count"] == 4

    def test_pipe_save_as(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        wid = ref["workspace_id"]
        result = forge.dispatch_tool_call(
            "papers_pipe",
            {
                "workspace_id": wid,
                "steps": [{"filter": {"author": "Alice"}}],
                "save_as": "alice_papers",
            },
        )
        assert result["status"] == "pipeline_result_saved"
        assert result["workspace_id"] == "alice_papers"
        assert result["item_count"] == 3

        # Saved workspace is browsable
        saved = forge.store.get_items("alice_papers")
        assert len(saved) == 3

    def test_pipe_error_handling(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        wid = ref["workspace_id"]
        result = forge.dispatch_tool_call(
            "papers_pipe",
            {
                "workspace_id": wid,
                "steps": [{"nonexistent_op": True}],
            },
        )
        assert "error" in result
        assert "available_operations" in result

    def test_pipe_on_dict_workspace(self):
        """Pipeline on dict workspace returns helpful error."""
        forge = Forge(store=MemoryStore())
        pipe_ts = pipeline(forge, "config")

        @forge.producer(workspace_type="config", toolsets=[pipe_ts])
        def load_config():
            return {"key": "value"}

        ref = load_config()
        result = forge.dispatch_tool_call(
            "config_pipe",
            {
                "workspace_id": ref["workspace_id"],
                "steps": [{"filter": {"key": "value"}}],
            },
        )
        # Either the data_shape toolset validation catches it, or the
        # pipe tool itself catches it — both return error dicts
        assert "error" in result

    def test_aggregate_basic(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        wid = ref["workspace_id"]
        result = forge.dispatch_tool_call(
            "papers_aggregate",
            {"workspace_id": wid},
        )
        assert result["result"]["count"] == 10

    def test_aggregate_grouped(self, forge_with_pipeline):
        forge, ref, ts = forge_with_pipeline
        wid = ref["workspace_id"]
        result = forge.dispatch_tool_call(
            "papers_aggregate",
            {
                "workspace_id": wid,
                "group_by": "year",
                "metrics": {
                    "papers": "count",
                    "avg_citations": "mean:citations",
                },
            },
        )
        assert "groups" in result
        y2023 = next(g for g in result["groups"] if g["year"] == 2023)
        assert y2023["papers"] == 4

    def test_schema_generated(self, forge_with_pipeline):
        """Pipeline tools appear in tool schemas."""
        forge, _, _ = forge_with_pipeline
        schemas = forge.get_all_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "papers_pipe" in names
        assert "papers_aggregate" in names

    def test_pipe_schema_has_good_description(self, forge_with_pipeline):
        forge, _, _ = forge_with_pipeline
        schemas = forge.get_all_tool_schemas()
        pipe_schema = next(s for s in schemas if s["function"]["name"] == "papers_pipe")
        desc = pipe_schema["function"]["description"]
        # Should mention the key operations
        assert "filter" in desc.lower()
        assert "sort" in desc.lower()
        assert "group_by" in desc.lower()

    def test_coexists_with_paginator(self):
        """Pipeline can share a ToolSet with paginator."""
        from ctxtual.utils import paginator

        forge = Forge(store=MemoryStore())
        pager = paginator(forge, "data")
        pipe = pipeline(forge, "data")

        # Same ToolSet, different tools
        assert pager is pipe
        assert "data_paginate" in pager.tool_names
        assert "data_pipe" in pager.tool_names
        assert "data_aggregate" in pager.tool_names


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases and real-world patterns
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineRealWorldPatterns:
    """Patterns that an LLM would actually use."""

    def test_top_n_pattern(self, engine):
        """Classic 'top N by metric' — would normally be 2+ tool calls."""
        result = engine.execute(
            PAPERS,
            [
                {"sort": {"field": "citations", "order": "desc"}},
                {"limit": 5},
                {"select": ["title", "author", "citations"]},
            ],
        )
        assert len(result) == 5
        assert result[0]["citations"] == 300

    def test_faceted_search_pattern(self, engine):
        """Search + facet analysis — would normally be 3+ calls."""
        result = engine.execute(
            PAPERS,
            [
                {"search": {"query": "ml", "fields": ["tags"]}},
                {
                    "group_by": {
                        "field": "year",
                        "metrics": {"n": "count", "avg_cites": "mean:citations"},
                    }
                },
                {"sort": {"field": "avg_cites", "order": "desc"}},
            ],
        )
        assert isinstance(result, list)
        assert all("n" in g and "avg_cites" in g for g in result)

    def test_outlier_detection_pattern(self, engine):
        """Find items that are statistical outliers."""
        result = engine.execute(
            PAPERS,
            [
                {"filter": {"citations": {"$gt": 200}}},
                {"sort": {"field": "citations", "order": "desc"}},
                {"select": ["title", "author", "citations"]},
            ],
        )
        assert all(r["citations"] > 200 for r in result)
        assert len(result) == 2  # Transformer (300), NLP (250)

    def test_dedup_and_count_pattern(self, engine):
        """Unique values + count — common exploration pattern."""
        result = engine.execute(
            PAPERS,
            [
                {"flatten": "tags"},
                {"unique": "tags"},
                {"count": True},
            ],
        )
        assert result["count"] > 0  # Number of unique tags

    def test_empty_result_pipeline(self, engine):
        """Pipeline that filters everything out."""
        result = engine.execute(
            PAPERS,
            [
                {"filter": {"year": 2099}},
                {"count": True},
            ],
        )
        assert result == {"count": 0}

    def test_identity_pipeline(self, engine):
        """Empty steps = return original data."""
        result = engine.execute(PAPERS, [])
        assert result == PAPERS

    def test_nested_field_pipeline(self, engine):
        """Pipeline with dot-notation nested fields."""
        data = [
            {"name": "A", "stats": {"score": 10, "rank": 3}},
            {"name": "B", "stats": {"score": 20, "rank": 1}},
            {"name": "C", "stats": {"score": 15, "rank": 2}},
        ]
        result = engine.execute(
            data,
            [
                {"filter": {"stats.score": {"$gte": 15}}},
                {"sort": {"field": "stats.rank"}},
            ],
        )
        assert len(result) == 2
        assert result[0]["name"] == "B"  # rank 1
