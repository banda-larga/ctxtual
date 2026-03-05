"""Tests for LLM-facing interface quality.

Validates that tool results, system prompts, WorkspaceRef notifications,
schemas, and error messages are production-ready for real LLM consumption.
"""

import json

import pytest

from ctxtual import Forge, MemoryStore
from ctxtual.utils import (
    filter_set,
    kv_reader,
    paginator,
    pipeline,
    text_search,
)

# ── Fixtures ─────────────────────────────────────────────────────────

PAPERS = [
    {
        "title": "Attention Is All You Need",
        "year": 2017,
        "authors": ["Vaswani", "Shazeer", "Parmar"],
        "citations": 98000,
        "field": "NLP",
        "abstract": "We propose a new architecture based solely on attention mechanisms.",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "year": 2019,
        "authors": ["Devlin", "Chang"],
        "citations": 67000,
        "field": "NLP",
        "abstract": "We introduce BERT, a new language representation model.",
    },
    {
        "title": "GPT-4 Technical Report",
        "year": 2023,
        "authors": ["OpenAI"],
        "citations": 5000,
        "field": "LLM",
        "abstract": "We report the development of GPT-4, a large-scale multimodal model.",
    },
]


@pytest.fixture
def forge():
    return Forge(store=MemoryStore())


@pytest.fixture
def full_forge(forge):
    """Forge with all toolsets registered."""
    pager = paginator(forge, "papers")
    search = text_search(forge, "papers", fields=["title", "abstract"])
    filt = filter_set(forge, "papers")
    pipe = pipeline(forge, "papers")

    @forge.producer(workspace_type="papers", toolsets=[pager, search, filt, pipe])
    def search_papers(query: str, limit: int = 10000):
        """Search the academic papers database by keyword."""
        return PAPERS

    return forge, search_papers


# ── WorkspaceRef: Data Preview ───────────────────────────────────────


class TestWorkspaceRefPreview:
    """The LLM must know field names immediately — no paginate-to-discover."""

    def test_ref_includes_item_schema_with_fields(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        assert "item_schema" in ref
        props = ref["item_schema"]["properties"]
        assert list(props.keys()) == ["title", "year", "authors", "citations", "field", "abstract"]

    def test_ref_includes_item_schema(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        assert "item_schema" in ref
        schema = ref["item_schema"]
        assert schema["type"] == "object"
        assert "title" in schema["properties"]
        assert "year" in schema["properties"]

    def test_item_schema_infers_types(self, forge):
        pager = paginator(forge, "docs")

        @forge.producer(workspace_type="docs", toolsets=[pager])
        def fetch_docs():
            return [{"content": "x" * 200, "id": 1, "score": 0.5, "tags": ["a"], "active": True}]

        ref = fetch_docs()
        schema = ref["item_schema"]
        assert schema["properties"]["content"]["type"] == "string"
        assert schema["properties"]["id"]["type"] == "integer"
        assert schema["properties"]["score"]["type"] == "number"
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["active"]["type"] == "boolean"

    def test_item_schema_detects_optional_fields(self, forge):
        pager = paginator(forge, "docs")

        @forge.producer(workspace_type="docs", toolsets=[pager])
        def fetch_docs():
            return [
                {"id": 1, "name": "Alice", "email": "a@b.com"},
                {"id": 2, "name": "Bob"},  # no email
            ]

        ref = fetch_docs()
        schema = ref["item_schema"]
        # id and name are in all items → required
        assert "id" in schema["required"]
        assert "name" in schema["required"]
        # email is only in one item → not required
        assert "email" not in schema.get("required", [])

    def test_item_schema_nested_dict_is_object(self, forge):
        pager = paginator(forge, "docs")

        @forge.producer(workspace_type="docs", toolsets=[pager])
        def fetch_docs():
            return [{"meta": {"score": 0.95, "tags": ["a"]}, "id": 1}]

        ref = fetch_docs()
        schema = ref["item_schema"]
        assert schema["properties"]["meta"]["type"] == "object"

    def test_dict_workspace_has_schema(self, forge):
        kv = kv_reader(forge, "config")

        @forge.producer(workspace_type="config", toolsets=[kv])
        def fetch_config():
            return {"host": "localhost", "port": 5432, "debug": True}

        ref = fetch_config()
        assert "item_schema" in ref
        schema = ref["item_schema"]
        assert schema["properties"]["host"]["type"] == "string"
        assert schema["properties"]["port"]["type"] == "integer"
        assert schema["properties"]["debug"]["type"] == "boolean"

    def test_empty_list_no_schema(self, forge):
        pager = paginator(forge, "empty")

        @forge.producer(workspace_type="empty", toolsets=[pager])
        def fetch_empty():
            return []

        ref = fetch_empty()
        assert ref.get("item_schema") is None

    def test_list_of_non_dicts_no_schema(self, forge):
        pager = paginator(forge, "nums")

        @forge.producer(workspace_type="nums", toolsets=[pager])
        def fetch_nums():
            return [1, 2, 3, 4, 5]

        ref = fetch_nums()
        assert ref.get("item_schema") is None

    def test_ref_is_json_serializable(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        # Must not raise
        serialized = json.dumps(ref, default=str)
        assert len(serialized) > 100


# ── System Prompt Quality ────────────────────────────────────────────


class TestSystemPrompt:
    """System prompt must give the LLM useful guidance, not a generic blurb."""

    def test_includes_preamble(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt(preamble="You are a research assistant.")
        assert "You are a research assistant." in prompt

    def test_includes_workspace_pattern(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "workspace" in prompt.lower()
        assert "workspace_id" in prompt

    def test_includes_producer_names(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "search_papers" in prompt

    def test_includes_producer_description(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "academic papers" in prompt.lower()

    def test_includes_pipeline_syntax_when_registered(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "Pipeline" in prompt
        assert "$gte" in prompt
        assert "filter" in prompt
        assert "sort" in prompt

    def test_no_pipeline_section_without_pipeline(self, forge):
        pager = paginator(forge, "items")

        @forge.producer(workspace_type="items", toolsets=[pager])
        def fetch():
            return [{"x": 1}]

        prompt = forge.system_prompt()
        assert "Pipeline" not in prompt

    def test_includes_error_recovery_guidance(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "error" in prompt.lower()
        assert "suggested_action" in prompt

    def test_includes_search_mention(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "search" in prompt.lower()

    def test_includes_filter_mention(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "filter" in prompt.lower()

    def test_prompt_is_concise(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        # Should be informative but not bloated
        assert len(prompt) < 3000, f"System prompt too long: {len(prompt)} chars"
        assert len(prompt) > 200, f"System prompt too short: {len(prompt)} chars"

    def test_lists_exploration_tools(self, full_forge):
        forge, _ = full_forge
        prompt = forge.system_prompt()
        assert "papers_paginate" in prompt or "*_paginate" in prompt


# ── Schema Quality for Complex Parameters ────────────────────────────


class TestPipelineSchemaQuality:
    """LLM needs structured schema guidance for complex params like steps."""

    def test_steps_has_examples(self, full_forge):
        forge, _ = full_forge
        schemas = forge.get_all_tool_schemas()
        pipe_schema = next(
            s for s in schemas if s["function"]["name"] == "papers_pipe"
        )
        steps_prop = pipe_schema["function"]["parameters"]["properties"]["steps"]
        assert "examples" in steps_prop
        examples = steps_prop["examples"]
        assert len(examples) >= 2

    def test_steps_items_has_description(self, full_forge):
        forge, _ = full_forge
        schemas = forge.get_all_tool_schemas()
        pipe_schema = next(
            s for s in schemas if s["function"]["name"] == "papers_pipe"
        )
        steps_prop = pipe_schema["function"]["parameters"]["properties"]["steps"]
        assert "items" in steps_prop
        assert "description" in steps_prop["items"]
        assert "operation" in steps_prop["items"]["description"].lower()

    def test_steps_examples_are_valid_pipelines(self, full_forge):
        """Every schema example must be a real, executable pipeline."""
        forge, search_papers = full_forge
        ref = search_papers("test")
        ws_id = ref["workspace_id"]

        schemas = forge.get_all_tool_schemas()
        pipe_schema = next(
            s for s in schemas if s["function"]["name"] == "papers_pipe"
        )
        examples = pipe_schema["function"]["parameters"]["properties"]["steps"]["examples"]

        for i, example in enumerate(examples):
            result = forge.dispatch_tool_call("papers_pipe", {
                "workspace_id": ws_id,
                "steps": example,
            })
            assert "error" not in result, f"Example {i} failed: {result}"

    def test_metrics_has_examples(self, full_forge):
        forge, _ = full_forge
        schemas = forge.get_all_tool_schemas()
        agg_schema = next(
            s for s in schemas if s["function"]["name"] == "papers_aggregate"
        )
        metrics_prop = agg_schema["function"]["parameters"]["properties"]["metrics"]
        assert "examples" in metrics_prop
        assert len(metrics_prop["examples"]) >= 1

    def test_schema_extra_merges_with_auto_schema(self, forge):
        """schema_extra should add to, not replace, auto-generated fields."""
        ts = forge.toolset("test")

        @ts.tool(
            name="test_tool",
            validate_workspace=False,
            schema_extra={"count": {"minimum": 0, "maximum": 100}},
        )
        def my_tool(workspace_id: str, count: int = 10) -> dict:
            """Do something.

            Args:
                count: Number of items to return.
            """
            return {}

        schema = ts.to_tool_schemas()[0]
        count_prop = schema["function"]["parameters"]["properties"]["count"]
        assert count_prop["type"] == "integer"  # From auto-gen
        assert count_prop["minimum"] == 0  # From schema_extra
        assert count_prop["maximum"] == 100  # From schema_extra
        assert "description" in count_prop  # From docstring


# ── save_as Returns Proper WorkspaceRef ──────────────────────────────


class TestSaveAsNotification:
    """Pipeline save_as must return a self-describing WorkspaceRef."""

    def test_save_as_has_available_tools(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        ws_id = ref["workspace_id"]

        result = forge.dispatch_tool_call("papers_pipe", {
            "workspace_id": ws_id,
            "steps": [{"filter": {"year": {"$gte": 2019}}}],
            "save_as": "recent",
        })
        assert "available_tools" in result
        assert any("recent" in t for t in result["available_tools"])

    def test_save_as_has_schema_with_projected_fields(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        ws_id = ref["workspace_id"]

        result = forge.dispatch_tool_call("papers_pipe", {
            "workspace_id": ws_id,
            "steps": [{"select": ["title", "year"]}],
            "save_as": "projected",
        })
        assert "item_schema" in result
        assert set(result["item_schema"]["properties"].keys()) == {"title", "year"}

    def test_save_as_has_item_schema(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        ws_id = ref["workspace_id"]

        result = forge.dispatch_tool_call("papers_pipe", {
            "workspace_id": ws_id,
            "steps": [{"limit": 2}],
            "save_as": "top2",
        })
        assert "item_schema" in result
        schema = result["item_schema"]
        assert schema["type"] == "object"
        assert "title" in schema["properties"]

    def test_save_as_has_next_steps(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        ws_id = ref["workspace_id"]

        result = forge.dispatch_tool_call("papers_pipe", {
            "workspace_id": ws_id,
            "steps": [{"limit": 1}],
            "save_as": "one",
        })
        assert "next_steps" in result
        assert len(result["next_steps"]) > 0

    def test_save_as_workspace_is_browsable(self, full_forge):
        """The saved workspace must work with paginate/search/etc."""
        forge, search_papers = full_forge
        ref = search_papers("test")
        ws_id = ref["workspace_id"]

        forge.dispatch_tool_call("papers_pipe", {
            "workspace_id": ws_id,
            "steps": [{"filter": {"year": {"$gte": 2019}}}],
            "save_as": "browsable",
        })

        # Should be browsable with all the registered tools
        page = forge.dispatch_tool_call("papers_paginate", {
            "workspace_id": "browsable",
            "page": 0,
            "size": 10,
        })
        assert "result" in page  # hint envelope
        assert page["result"]["total"] == 2

    def test_save_as_status_field(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        ws_id = ref["workspace_id"]

        result = forge.dispatch_tool_call("papers_pipe", {
            "workspace_id": ws_id,
            "steps": [{"limit": 1}],
            "save_as": "saved",
        })
        assert result["status"] == "pipeline_result_saved"


# ── End-to-End: Can the LLM Actually Use This? ──────────────────────


class TestEndToEndLLMWorkflow:
    """Simulate what an LLM would do: read notification → use tools."""

    def test_full_workflow_without_paginate_first(self, full_forge):
        """LLM can construct a pipeline query directly from the notification."""
        forge, search_papers = full_forge

        # Step 1: LLM calls producer
        ref = search_papers("transformers")

        # Step 2: LLM reads schema from notification — no paginate needed
        assert "item_schema" in ref
        props = ref["item_schema"]["properties"]
        assert "year" in props
        assert "citations" in props

        ws_id = ref["workspace_id"]

        # Step 3: LLM constructs pipeline using field knowledge
        result = forge.dispatch_tool_call("papers_pipe", {
            "workspace_id": ws_id,
            "steps": [
                {"filter": {"year": {"$gte": 2019}}},
                {"sort": {"field": "citations", "order": "desc"}},
                {"select": ["title", "year", "citations"]},
            ],
        })
        assert "error" not in result
        assert result["count"] == 2
        assert result["items"][0]["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"

    def test_system_prompt_plus_schemas_is_complete(self, full_forge):
        """System prompt + tool schemas give the LLM everything it needs."""
        forge, _ = full_forge
        prompt = forge.system_prompt(preamble="You are a research assistant.")
        schemas = forge.get_tools()

        # Prompt mentions the producer
        assert "search_papers" in prompt

        # Schemas include producer + consumer tools
        tool_names = [s["function"]["name"] for s in schemas]
        assert "search_papers" in tool_names
        assert "papers_pipe" in tool_names
        assert "papers_paginate" in tool_names

        # Pipe schema has examples
        pipe_schema = next(s for s in schemas if s["function"]["name"] == "papers_pipe")
        assert "examples" in pipe_schema["function"]["parameters"]["properties"]["steps"]

    def test_error_includes_recovery_guidance(self, full_forge):
        """When LLM makes a mistake, error tells it what to do."""
        forge, search_papers = full_forge
        search_papers("test")

        # Wrong workspace_id
        result = forge.dispatch_tool_call("papers_paginate", {
            "workspace_id": "nonexistent",
        })
        assert "error" in result
        assert "suggested_action" in result
        assert "available_workspaces" in result

    def test_schemas_are_json_serializable(self, full_forge):
        """All schemas must be serializable for API calls."""
        forge, _ = full_forge
        schemas = forge.get_tools()
        serialized = json.dumps(schemas, default=str)
        assert len(serialized) > 500

    def test_notification_is_json_serializable(self, full_forge):
        forge, search_papers = full_forge
        ref = search_papers("test")
        serialized = json.dumps(ref, default=str)
        roundtrip = json.loads(serialized)
        assert roundtrip["workspace_id"] == ref["workspace_id"]
        assert roundtrip["item_schema"] == ref["item_schema"]
