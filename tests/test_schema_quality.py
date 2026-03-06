"""Tests for schema quality improvements (Fix #10)."""

import inspect
from typing import Any, Literal, Optional, Union  # noqa: UP035

from ctxtual import Ctx, MemoryStore
from ctxtual.toolset import (
    _build_param_description,
    _extract_param_descriptions,
    _python_type_to_json_schema,
)
from ctxtual.utils import filter_set, paginator

# ═══════════════════════════════════════════════════════════════════════════
# Type mapping
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeMapping:
    def test_basic_str(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_basic_int(self):
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_basic_float(self):
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_basic_bool(self):
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_basic_list(self):
        assert _python_type_to_json_schema(list) == {"type": "array"}

    def test_basic_dict(self):
        assert _python_type_to_json_schema(dict) == {"type": "object"}

    def test_empty_annotation(self):
        assert _python_type_to_json_schema(inspect.Parameter.empty) == {
            "type": "string"
        }

    def test_any(self):
        assert _python_type_to_json_schema(Any) == {}

    def test_none_type(self):
        assert _python_type_to_json_schema(type(None)) == {"type": "null"}

    def test_optional_str(self):
        schema = _python_type_to_json_schema(Optional[str])  # noqa: UP045
        assert schema == {"type": "string"}

    def test_optional_int(self):
        schema = _python_type_to_json_schema(Optional[int])  # noqa: UP045
        assert schema == {"type": "integer"}

    def test_union_str_int(self):
        schema = _python_type_to_json_schema(Union[str, int])  # noqa: UP007
        assert "anyOf" in schema
        types = [s["type"] for s in schema["anyOf"]]
        assert "string" in types
        assert "integer" in types

    def test_pipe_union(self):
        schema = _python_type_to_json_schema(str | int)
        assert "anyOf" in schema

    def test_pipe_optional(self):
        schema = _python_type_to_json_schema(str | None)
        assert schema == {"type": "string"}

    def test_literal_strings(self):
        schema = _python_type_to_json_schema(Literal["eq", "ne", "gt"])
        assert schema == {"type": "string", "enum": ["eq", "ne", "gt"]}

    def test_literal_ints(self):
        schema = _python_type_to_json_schema(Literal[1, 2, 3])
        assert schema == {"type": "integer", "enum": [1, 2, 3]}

    def test_literal_mixed(self):
        schema = _python_type_to_json_schema(Literal["a", 1])
        assert schema == {"enum": ["a", 1]}

    def test_list_of_str(self):
        schema = _python_type_to_json_schema(list[str])
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_list_of_dict(self):
        schema = _python_type_to_json_schema(list[dict])
        assert schema == {"type": "array", "items": {"type": "object"}}

    def test_dict_str_any(self):
        schema = _python_type_to_json_schema(dict[str, Any])
        assert schema == {"type": "object"}

    def test_tuple(self):
        schema = _python_type_to_json_schema(tuple[str, int])
        assert schema["type"] == "array"
        assert "prefixItems" in schema

    def test_set(self):
        schema = _python_type_to_json_schema(set[str])
        assert schema == {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
        }

    def test_unknown_type_fallback(self):
        class CustomType:
            pass

        schema = _python_type_to_json_schema(CustomType)
        assert schema == {"type": "string"}


# ═══════════════════════════════════════════════════════════════════════════
# Docstring extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestDocstringExtraction:
    def test_google_style(self):
        def my_func(query: str, limit: int = 10):
            """Search for items.

            Args:
                query: The search query string to use.
                limit: Maximum number of results.
            """

        descs = _extract_param_descriptions(my_func)
        assert descs["query"] == "The search query string to use."
        assert descs["limit"] == "Maximum number of results."

    def test_google_style_multiline(self):
        def my_func(data: list):
            """Process data.

            Args:
                data: The input data list that
                    can contain any number of items.
            """

        descs = _extract_param_descriptions(my_func)
        assert "input data list" in descs["data"]
        assert "items" in descs["data"]

    def test_sphinx_style(self):
        def my_func(name: str, count: int):
            """Do something.

            :param name: The name to look up.
            :param count: How many items.
            """

        descs = _extract_param_descriptions(my_func)
        assert descs["name"] == "The name to look up."
        assert descs["count"] == "How many items."

    def test_no_docstring(self):
        def my_func(x: int):
            pass

        assert _extract_param_descriptions(my_func) == {}

    def test_no_params_section(self):
        def my_func(x: int):
            """Just a summary."""

        assert _extract_param_descriptions(my_func) == {}

    def test_google_with_returns_section(self):
        def my_func(query: str):
            """Search.

            Args:
                query: What to search for.

            Returns:
                A list of results.
            """

        descs = _extract_param_descriptions(my_func)
        assert descs["query"] == "What to search for."
        assert "Returns" not in descs


# ═══════════════════════════════════════════════════════════════════════════
# Parameter description builder
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildParamDescription:
    def test_docstring_takes_priority(self):
        param = inspect.Parameter(
            "query", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
        )
        desc = _build_param_description("query", param, {"query": "Custom desc."})
        assert desc == "Custom desc."

    def test_docstring_with_default(self):
        param = inspect.Parameter(
            "page", inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=int, default=0
        )
        desc = _build_param_description("page", param, {"page": "Page number."})
        assert "Page number." in desc
        assert "Defaults to 0" in desc

    def test_docstring_already_mentions_default(self):
        param = inspect.Parameter(
            "page", inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=int, default=0
        )
        desc = _build_param_description(
            "page", param, {"page": "Page number. Default is 0."}
        )
        # Should not duplicate "default"
        assert desc.count("efault") == 1

    def test_well_known_param(self):
        param = inspect.Parameter(
            "workspace_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
        )
        desc = _build_param_description("workspace_id", param, {})
        assert "workspace" in desc.lower()
        assert "identifier" in desc.lower()

    def test_well_known_with_default(self):
        param = inspect.Parameter(
            "size", inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=int, default=10
        )
        desc = _build_param_description("size", param, {})
        assert "items per page" in desc.lower()
        assert "Defaults to 10" in desc

    def test_fallback_with_annotation(self):
        param = inspect.Parameter(
            "user_name", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
        )
        desc = _build_param_description("user_name", param, {})
        assert "user name" in desc
        assert "str" in desc

    def test_fallback_with_default(self):
        param = inspect.Parameter(
            "verbose", inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=bool, default=False
        )
        desc = _build_param_description("verbose", param, {})
        assert "Defaults to False" in desc


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end: schema output quality
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaOutputQuality:
    def test_consumer_schema_has_meaningful_descriptions(self):
        """ToolSet schemas use docstring params, not just the param name."""
        ctx = Ctx(store=MemoryStore())
        ts = ctx.toolset("test")

        @ts.tool
        def search(workspace_id: str, query: str, limit: int = 10) -> list:
            """Search items.

            Args:
                workspace_id: The workspace to search in.
                query: Full-text search query.
                limit: Max results to return.
            """
            return []

        schemas = ts.to_tool_schemas()
        props = schemas[0]["function"]["parameters"]["properties"]

        assert props["workspace_id"]["description"] == "The workspace to search in."
        assert props["query"]["description"] == "Full-text search query."
        assert "Max results" in props["limit"]["description"]

    def test_producer_schema_has_meaningful_descriptions(self):
        ctx = Ctx(store=MemoryStore())

        @ctx.producer(workspace_type="data")
        def fetch(query: str, top_k: int = 100) -> list:
            """Fetch data from the database.

            Args:
                query: SQL-like query string.
                top_k: Number of top results.
            """
            return []

        schemas = ctx.get_producer_schemas()
        props = schemas[0]["function"]["parameters"]["properties"]

        assert props["query"]["description"] == "SQL-like query string."
        assert "top results" in props["top_k"]["description"].lower()

    def test_well_known_params_get_good_descriptions(self):
        """Even without docstrings, well-known param names get good descriptions."""
        ctx = Ctx(store=MemoryStore())
        ts = ctx.toolset("test")

        @ts.tool
        def paginate(workspace_id: str, page: int = 0, size: int = 10) -> list:
            """Paginate items."""
            return []

        schemas = ts.to_tool_schemas()
        props = schemas[0]["function"]["parameters"]["properties"]

        assert "workspace" in props["workspace_id"]["description"].lower()
        assert "page" in props["page"]["description"].lower()
        assert "items per page" in props["size"]["description"].lower()

    def test_default_values_in_schema(self):
        """Default values appear in the JSON Schema."""
        ctx = Ctx(store=MemoryStore())
        ts = ctx.toolset("test")

        @ts.tool
        def search(workspace_id: str, page: int = 0, size: int = 10) -> list:
            """Search."""
            return []

        schemas = ts.to_tool_schemas()
        props = schemas[0]["function"]["parameters"]["properties"]

        assert props["page"].get("default") == 0
        assert props["size"].get("default") == 10
        # workspace_id has no default → no "default" key
        assert "default" not in props["workspace_id"]

    def test_complex_type_annotations(self):
        """Complex annotations produce richer schemas."""
        ctx = Ctx(store=MemoryStore())
        ts = ctx.toolset("test")

        @ts.tool
        def advanced(
            workspace_id: str,
            tags: list[str] = [],  # noqa: B006
            mode: Literal["fast", "thorough"] = "fast",
            extra: dict[str, Any] | None = None,
        ) -> list:
            """Advanced tool."""
            return []

        schemas = ts.to_tool_schemas()
        props = schemas[0]["function"]["parameters"]["properties"]

        assert props["tags"]["type"] == "array"
        assert props["tags"]["items"] == {"type": "string"}

        assert props["mode"]["type"] == "string"
        assert props["mode"]["enum"] == ["fast", "thorough"]

        # Optional[dict] → object (None is stripped)
        assert props["extra"]["type"] == "object"

    def test_builtin_paginator_schema_quality(self):
        """Built-in paginator produces high-quality schemas."""
        ctx = Ctx(store=MemoryStore())
        pager = paginator(ctx, "docs")
        schemas = pager.to_tool_schemas()

        # Find the paginate tool schema
        paginate_schema = next(
            s for s in schemas if s["function"]["name"] == "docs_paginate"
        )
        props = paginate_schema["function"]["parameters"]["properties"]

        # Descriptions should be meaningful, NOT just the param name
        assert props["workspace_id"]["description"] != "workspace_id"
        assert props["page"]["description"] != "page"
        assert props["size"]["description"] != "size"

        # Should contain useful information
        assert "workspace" in props["workspace_id"]["description"].lower()
        assert "page" in props["page"]["description"].lower()

    def test_builtin_filter_set_schema_quality(self):
        """Built-in filter_set has good operator description."""
        ctx = Ctx(store=MemoryStore())
        filt = filter_set(ctx, "items")
        schemas = filt.to_tool_schemas()

        filter_schema = next(
            s for s in schemas if s["function"]["name"] == "items_filter_by"
        )
        props = filter_schema["function"]["parameters"]["properties"]

        # operator should explain available values
        assert "eq" in props["operator"]["description"]
        assert props["operator"].get("default") == "eq"
