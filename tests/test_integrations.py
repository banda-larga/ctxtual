"""Tests for ctx.integrations adapters."""

import json

import pytest

from ctxtual import Ctx, MemoryStore
from ctxtual.utils import paginator, text_search

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def forge_with_data():
    """Ctx with a producer, a workspace, and some data."""
    ctx = Ctx(store=MemoryStore())
    pager = paginator(ctx, "docs")
    searcher = text_search(ctx, "docs", fields=["title"])

    @ctx.producer(workspace_type="docs", toolsets=[pager, searcher])
    def find_docs(query: str) -> list[dict]:
        """Search for documents."""
        return [{"title": f"Doc {i}: {query}"} for i in range(20)]

    ref = find_docs(query="testing")
    return ctx, ref


# ── Mock objects that mimic SDK response shapes ──────────────────────────


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls = tool_calls
        self.content = content


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    """Mimics openai.types.chat.ChatCompletion."""

    def __init__(self, choices):
        self.choices = choices


class _AnthropicTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _AnthropicToolUse:
    def __init__(self, id, name, input_data):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input_data


class _AnthropicResponse:
    """Mimics anthropic.types.Message."""

    def __init__(self, content):
        self.content = content


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI adapter tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAIAdapter:
    def test_to_openai_tools(self, forge_with_data):
        from ctxtual.integrations.openai import to_openai_tools

        ctx, _ = forge_with_data
        tools = to_openai_tools(ctx)
        assert isinstance(tools, list)
        assert len(tools) > 0
        # Every tool has the OpenAI function-calling shape
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]
            assert "parameters" in t["function"]

    def test_has_tool_calls_sdk_true(self, forge_with_data):
        from ctxtual.integrations.openai import has_tool_calls

        tc = _ToolCall("call_1", "find_docs", '{"query": "x"}')
        resp = _Response([_Choice(_Message(tool_calls=[tc]))])
        assert has_tool_calls(resp) is True

    def test_has_tool_calls_sdk_false(self, forge_with_data):
        from ctxtual.integrations.openai import has_tool_calls

        resp = _Response([_Choice(_Message(tool_calls=None))])
        assert has_tool_calls(resp) is False

    def test_has_tool_calls_dict_true(self, forge_with_data):
        from ctxtual.integrations.openai import has_tool_calls

        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "find_docs",
                                    "arguments": '{"query": "x"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        assert has_tool_calls(resp) is True

    def test_has_tool_calls_dict_false(self, forge_with_data):
        from ctxtual.integrations.openai import has_tool_calls

        resp = {"choices": [{"message": {"content": "hello"}}]}
        assert has_tool_calls(resp) is False

    def test_has_tool_calls_garbage(self):
        from ctxtual.integrations.openai import has_tool_calls

        assert has_tool_calls(42) is False
        assert has_tool_calls(None) is False
        assert has_tool_calls("hello") is False

    def test_handle_tool_calls_sdk(self, forge_with_data):
        from ctxtual.integrations.openai import handle_tool_calls

        ctx, ref = forge_with_data
        wid = ref["workspace_id"]

        tc = _ToolCall(
            "call_1",
            "docs_paginate",
            json.dumps({"workspace_id": wid, "page": 0, "size": 5}),
        )
        resp = _Response([_Choice(_Message(tool_calls=[tc]))])
        msgs = handle_tool_calls(ctx, resp)

        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "call_1"
        data = json.loads(msgs[0]["content"])
        assert "result" in data  # hint envelope
        assert "items" in data["result"]

    def test_handle_tool_calls_dict(self, forge_with_data):
        from ctxtual.integrations.openai import handle_tool_calls

        ctx, ref = forge_with_data
        wid = ref["workspace_id"]

        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {
                                    "name": "docs_paginate",
                                    "arguments": json.dumps(
                                        {
                                            "workspace_id": wid,
                                            "page": 0,
                                            "size": 3,
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        msgs = handle_tool_calls(ctx, resp)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        data = json.loads(msgs[0]["content"])
        assert len(data["result"]["items"]) == 3

    def test_handle_tool_calls_error(self, forge_with_data):
        """Unknown tool returns an error dict instead of raising."""
        from ctxtual.integrations.openai import handle_tool_calls

        ctx, _ = forge_with_data
        tc = _ToolCall("call_e", "nonexistent_tool", '{"x": 1}')
        resp = _Response([_Choice(_Message(tool_calls=[tc]))])
        msgs = handle_tool_calls(ctx, resp)

        assert len(msgs) == 1
        data = json.loads(msgs[0]["content"])
        assert "error" in data

    def test_handle_tool_calls_producer(self, forge_with_data):
        """Dispatching a producer through the adapter."""
        from ctxtual.integrations.openai import handle_tool_calls

        ctx, _ = forge_with_data
        tc = _ToolCall(
            "call_p", "find_docs", json.dumps({"query": "via adapter"})
        )
        resp = _Response([_Choice(_Message(tool_calls=[tc]))])
        msgs = handle_tool_calls(ctx, resp)

        assert len(msgs) == 1
        data = json.loads(msgs[0]["content"])
        assert data["status"] == "workspace_ready"

    def test_max_content_length(self, forge_with_data):
        from ctxtual.integrations.openai import handle_tool_calls

        ctx, ref = forge_with_data
        wid = ref["workspace_id"]

        tc = _ToolCall(
            "call_t",
            "docs_paginate",
            json.dumps({"workspace_id": wid, "page": 0, "size": 20}),
        )
        resp = _Response([_Choice(_Message(tool_calls=[tc]))])
        msgs = handle_tool_calls(ctx, resp, max_content_length=100)

        assert len(msgs[0]["content"]) <= 100

    def test_multiple_tool_calls(self, forge_with_data):
        from ctxtual.integrations.openai import handle_tool_calls

        ctx, ref = forge_with_data
        wid = ref["workspace_id"]

        tc1 = _ToolCall(
            "call_a",
            "docs_paginate",
            json.dumps({"workspace_id": wid, "page": 0, "size": 2}),
        )
        tc2 = _ToolCall(
            "call_b",
            "docs_paginate",
            json.dumps({"workspace_id": wid, "page": 1, "size": 2}),
        )
        resp = _Response([_Choice(_Message(tool_calls=[tc1, tc2]))])
        msgs = handle_tool_calls(ctx, resp)
        assert len(msgs) == 2
        assert msgs[0]["tool_call_id"] == "call_a"
        assert msgs[1]["tool_call_id"] == "call_b"


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic adapter tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicAdapter:
    def test_to_anthropic_tools(self, forge_with_data):
        from ctxtual.integrations.anthropic import to_anthropic_tools

        ctx, _ = forge_with_data
        tools = to_anthropic_tools(ctx)
        assert isinstance(tools, list)
        assert len(tools) > 0
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "input_schema" in t
            # Must NOT have the OpenAI wrapper
            assert "function" not in t
            assert "type" not in t

    def test_schema_shape(self, forge_with_data):
        """Anthropic schemas have input_schema with properties/required."""
        from ctxtual.integrations.anthropic import to_anthropic_tools

        ctx, _ = forge_with_data
        tools = to_anthropic_tools(ctx)
        for t in tools:
            schema = t["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_has_tool_use_sdk_true(self, forge_with_data):
        from ctxtual.integrations.anthropic import has_tool_use

        resp = _AnthropicResponse(
            [_AnthropicToolUse("tu_1", "find_docs", {"query": "x"})]
        )
        assert has_tool_use(resp) is True

    def test_has_tool_use_sdk_false(self, forge_with_data):
        from ctxtual.integrations.anthropic import has_tool_use

        resp = _AnthropicResponse([_AnthropicTextBlock("Hello")])
        assert has_tool_use(resp) is False

    def test_has_tool_use_dict(self, forge_with_data):
        from ctxtual.integrations.anthropic import has_tool_use

        resp = {
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "x", "input": {}}
            ]
        }
        assert has_tool_use(resp) is True

        resp_text = {"content": [{"type": "text", "text": "hi"}]}
        assert has_tool_use(resp_text) is False

    def test_has_tool_use_garbage(self):
        from ctxtual.integrations.anthropic import has_tool_use

        assert has_tool_use(42) is False
        assert has_tool_use(None) is False

    def test_handle_tool_use_sdk(self, forge_with_data):
        from ctxtual.integrations.anthropic import handle_tool_use

        ctx, ref = forge_with_data
        wid = ref["workspace_id"]

        resp = _AnthropicResponse(
            [
                _AnthropicTextBlock("Let me check"),
                _AnthropicToolUse(
                    "tu_1",
                    "docs_paginate",
                    {"workspace_id": wid, "page": 0, "size": 3},
                ),
            ]
        )
        results = handle_tool_use(ctx, resp)

        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert results[0]["tool_use_id"] == "tu_1"
        assert "is_error" not in results[0]  # no error → key absent
        data = json.loads(results[0]["content"])
        assert len(data["result"]["items"]) == 3

    def test_handle_tool_use_dict(self, forge_with_data):
        from ctxtual.integrations.anthropic import handle_tool_use

        ctx, ref = forge_with_data
        wid = ref["workspace_id"]

        resp = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "docs_paginate",
                    "input": {"workspace_id": wid, "page": 0, "size": 2},
                }
            ]
        }
        results = handle_tool_use(ctx, resp)
        assert len(results) == 1
        data = json.loads(results[0]["content"])
        assert len(data["result"]["items"]) == 2

    def test_handle_tool_use_error(self, forge_with_data):
        from ctxtual.integrations.anthropic import handle_tool_use

        ctx, _ = forge_with_data
        resp = _AnthropicResponse(
            [_AnthropicToolUse("tu_e", "bad_tool", {"x": 1})]
        )
        results = handle_tool_use(ctx, resp)
        assert len(results) == 1
        assert results[0]["is_error"] is True

    def test_handle_tool_use_producer(self, forge_with_data):
        from ctxtual.integrations.anthropic import handle_tool_use

        ctx, _ = forge_with_data
        resp = _AnthropicResponse(
            [_AnthropicToolUse("tu_p", "find_docs", {"query": "anthropic"})]
        )
        results = handle_tool_use(ctx, resp)
        assert len(results) == 1
        data = json.loads(results[0]["content"])
        assert data["status"] == "workspace_ready"

    def test_multiple_tool_use(self, forge_with_data):
        from ctxtual.integrations.anthropic import handle_tool_use

        ctx, ref = forge_with_data
        wid = ref["workspace_id"]

        resp = _AnthropicResponse(
            [
                _AnthropicToolUse(
                    "tu_a", "docs_paginate",
                    {"workspace_id": wid, "page": 0, "size": 2},
                ),
                _AnthropicToolUse(
                    "tu_b", "docs_paginate",
                    {"workspace_id": wid, "page": 1, "size": 2},
                ),
            ]
        )
        results = handle_tool_use(ctx, resp)
        assert len(results) == 2
        assert results[0]["tool_use_id"] == "tu_a"
        assert results[1]["tool_use_id"] == "tu_b"


# ═══════════════════════════════════════════════════════════════════════════
# LangChain adapter tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLangChainAdapter:
    def test_import_error_without_langchain(self):
        """to_langchain_tools raises ImportError when langchain-core is missing."""
        # This test works because langchain-core is not in our deps
        from ctxtual.integrations.langchain import to_langchain_tools

        ctx = Ctx(store=MemoryStore())
        with pytest.raises(ImportError, match="langchain-core"):
            to_langchain_tools(ctx)

    def test_json_schema_to_pydantic(self):
        """Internal helper builds a working Pydantic model from JSON Schema."""
        from ctxtual.integrations.langchain import _json_schema_to_pydantic

        try:
            from pydantic import Field, create_model
        except ImportError:
            pytest.skip("pydantic not installed")

        schema = {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string", "description": "Workspace"},
                "page": {"type": "integer", "description": "Page number"},
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["workspace_id"],
        }
        model = _json_schema_to_pydantic("test_tool", schema, create_model, Field)
        assert model.__name__ == "TestToolInput"

        # Required field works
        instance = model(workspace_id="abc")
        assert instance.workspace_id == "abc"

        # Optional fields have defaults
        assert instance.page is None
