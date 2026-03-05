"""Tests for text/scalar data support.

Covers: chunk_text, split_sections, split_markdown_sections transforms,
text_content toolset, scalar _count fix, and shape mismatch guard.
"""


import pytest

from ctxtual import (
    Forge,
    MemoryStore,
    chunk_text,
    split_markdown_sections,
    split_sections,
)
from ctxtual.utils import paginator, pipeline, text_content

SAMPLE_TEXT = (
    "# Introduction\n"
    "This is a document about AI. It covers several topics.\n\n"
    "# Machine Learning\n"
    "Machine learning is a subset of AI. It uses data to learn patterns.\n\n"
    "# Deep Learning\n"
    "Deep learning uses neural networks with many layers.\n\n"
    "# Conclusion\n"
    "AI is transforming the world."
)


@pytest.fixture
def forge():
    return Forge(store=MemoryStore())


# ── chunk_text transform ────────────────────────────────────────────


class TestChunkText:
    def test_basic_chunking(self):
        transform = chunk_text(chunk_size=50, overlap=10)
        result = transform("A" * 100)
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)
        assert result[0]["chunk_index"] == 0
        assert result[0]["char_offset"] == 0
        assert len(result[0]["text"]) == 50

    def test_overlap(self):
        transform = chunk_text(chunk_size=50, overlap=10)
        result = transform("A" * 100)
        # With chunk_size=50, overlap=10, step=40
        # Chunks: [0:50], [40:90], [80:100]
        assert len(result) == 3
        assert result[1]["char_offset"] == 40
        assert result[2]["char_offset"] == 80

    def test_no_overlap(self):
        transform = chunk_text(chunk_size=50, overlap=0)
        result = transform("A" * 100)
        assert len(result) == 2
        assert result[0]["char_offset"] == 0
        assert result[1]["char_offset"] == 50

    def test_short_string(self):
        transform = chunk_text(chunk_size=1000, overlap=100)
        result = transform("short")
        assert len(result) == 1
        assert result[0]["text"] == "short"

    def test_empty_string(self):
        transform = chunk_text(chunk_size=100, overlap=10)
        assert transform("") == []

    def test_passthrough_non_string(self):
        transform = chunk_text(chunk_size=100, overlap=10)
        data = [{"a": 1}]
        assert transform(data) is data

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="positive"):
            chunk_text(chunk_size=0)
        with pytest.raises(ValueError, match="non-negative"):
            chunk_text(chunk_size=100, overlap=-1)
        with pytest.raises(ValueError, match="less than"):
            chunk_text(chunk_size=100, overlap=100)

    def test_with_producer(self, forge):
        pager = paginator(forge, "doc")

        @forge.producer(
            workspace_type="doc",
            toolsets=[pager],
            transform=chunk_text(chunk_size=50, overlap=10),
        )
        def read_doc():
            return "A" * 200

        ref = read_doc()
        assert ref["data_shape"] == "list"
        assert ref["item_count"] > 1
        assert "chunk_index" in ref["item_schema"]["properties"]
        assert "text" in ref["item_schema"]["properties"]
        assert "char_offset" in ref["item_schema"]["properties"]


# ── split_sections transform ────────────────────────────────────────


class TestSplitSections:
    def test_paragraph_split(self):
        transform = split_sections(separator="\n\n")
        result = transform("Para 1\n\nPara 2\n\nPara 3")
        assert len(result) == 3
        assert result[0]["text"] == "Para 1"
        assert result[1]["text"] == "Para 2"
        assert result[2]["section_index"] == 2

    def test_custom_separator(self):
        transform = split_sections(separator="---")
        result = transform("Part A---Part B---Part C")
        assert len(result) == 3

    def test_min_length_filter(self):
        transform = split_sections(separator="\n\n", min_length=5)
        result = transform("OK\n\nThis is longer\n\nHi\n\nAnother long section")
        # "OK" (2 chars) and "Hi" (2 chars) should be filtered
        assert len(result) == 2

    def test_empty_string(self):
        transform = split_sections()
        assert transform("") == []

    def test_passthrough_non_string(self):
        transform = split_sections()
        data = [1, 2, 3]
        assert transform(data) is data

    def test_char_offsets(self):
        transform = split_sections(separator="\n\n")
        result = transform("Hello\n\nWorld")
        assert result[0]["char_offset"] == 0
        assert result[1]["char_offset"] == 7  # After "Hello\n\n"


# ── split_markdown_sections transform ────────────────────────────────


class TestSplitMarkdownSections:
    def test_basic_markdown(self):
        transform = split_markdown_sections()
        result = transform(SAMPLE_TEXT)
        assert isinstance(result, list)
        headings = [s["heading"] for s in result]
        assert "Introduction" in headings
        assert "Machine Learning" in headings
        assert "Deep Learning" in headings
        assert "Conclusion" in headings

    def test_heading_levels(self):
        transform = split_markdown_sections()
        text = "# H1\nBody1\n## H2\nBody2\n### H3\nBody3"
        result = transform(text)
        levels = {s["heading"]: s["level"] for s in result}
        assert levels["H1"] == 1
        assert levels["H2"] == 2
        assert levels["H3"] == 3

    def test_preamble(self):
        transform = split_markdown_sections()
        text = "Some preamble text.\n\n# First Section\nBody"
        result = transform(text)
        assert result[0]["heading"] == "(preamble)"
        assert result[0]["level"] == 0

    def test_no_headers(self):
        transform = split_markdown_sections()
        result = transform("Just plain text with no headers.")
        assert len(result) == 1
        assert result[0]["heading"] == "(document)"

    def test_empty_string(self):
        transform = split_markdown_sections()
        assert transform("") == []

    def test_passthrough_non_string(self):
        transform = split_markdown_sections()
        data = {"key": "val"}
        assert transform(data) is data

    def test_with_producer_and_pipeline(self, forge):
        pager = paginator(forge, "md")
        pipe = pipeline(forge, "md")

        @forge.producer(
            workspace_type="md",
            toolsets=[pager, pipe],
            transform=split_markdown_sections(),
        )
        def read_markdown():
            return SAMPLE_TEXT

        ref = read_markdown()
        assert ref["data_shape"] == "list"
        assert "heading" in ref["item_schema"]["properties"]
        assert "level" in ref["item_schema"]["properties"]
        assert "text" in ref["item_schema"]["properties"]

        # Pipeline works on sections
        ws_id = ref["workspace_id"]
        result = forge.dispatch_tool_call("md_pipe", {
            "workspace_id": ws_id,
            "steps": [
                {"filter": {"level": 1}},
                {"select": ["heading"]},
            ],
        })
        assert "error" not in result
        assert result["count"] >= 2


# ── text_content toolset ────────────────────────────────────────────


class TestTextContent:
    def test_read_page(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "A" * 5000

        ref = fetch_doc()
        ws_id = ref["workspace_id"]

        result = forge.dispatch_tool_call("doc_read_page", {
            "workspace_id": ws_id,
            "page": 0,
            "chars_per_page": 1000,
        })
        # Unwrap hint envelope
        data = result.get("result", result)
        assert len(data["text"]) == 1000
        assert data["page"] == 0
        assert data["total_pages"] == 5
        assert data["has_next"] is True
        assert data["has_prev"] is False

    def test_read_last_page(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "A" * 5000

        ref = fetch_doc()
        ws_id = ref["workspace_id"]
        result = forge.dispatch_tool_call("doc_read_page", {
            "workspace_id": ws_id,
            "page": 4,
            "chars_per_page": 1000,
        })
        data = result.get("result", result)
        assert data["has_next"] is False
        assert data["has_prev"] is True

    def test_page_out_of_range(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "Hello"

        ref = fetch_doc()
        ws_id = ref["workspace_id"]
        result = forge.dispatch_tool_call("doc_read_page", {
            "workspace_id": ws_id,
            "page": 99,
        })
        data = result.get("result", result)
        assert "error" in data

    def test_search_in_text(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return SAMPLE_TEXT

        ref = fetch_doc()
        ws_id = ref["workspace_id"]
        result = forge.dispatch_tool_call("doc_search_in_text", {
            "workspace_id": ws_id,
            "query": "neural networks",
        })
        assert result["match_count"] == 1
        assert "neural networks" in result["matches"][0]["context"].lower()

    def test_search_case_insensitive(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "Hello World"

        ref = fetch_doc()
        ws_id = ref["workspace_id"]
        result = forge.dispatch_tool_call("doc_search_in_text", {
            "workspace_id": ws_id,
            "query": "HELLO",
        })
        assert result["match_count"] == 1

    def test_search_no_results(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "Hello World"

        ref = fetch_doc()
        ws_id = ref["workspace_id"]
        result = forge.dispatch_tool_call("doc_search_in_text", {
            "workspace_id": ws_id,
            "query": "nonexistent",
        })
        assert result["match_count"] == 0

    def test_get_length(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "Hello\nWorld\n\nThird line"

        ref = fetch_doc()
        ws_id = ref["workspace_id"]
        result = forge.dispatch_tool_call("doc_get_length", {
            "workspace_id": ws_id,
        })
        assert result["chars"] == len("Hello\nWorld\n\nThird line")
        assert result["words"] == 4
        assert result["lines"] == 4

    def test_ref_shows_scalar_shape(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "Some text"

        ref = fetch_doc()
        assert ref["data_shape"] == "scalar"
        assert ref["item_count"] == 1


# ── Scalar count fix ────────────────────────────────────────────────


class TestScalarCount:
    def test_string_count_is_one(self, forge):
        reader = text_content(forge, "doc")

        @forge.producer(workspace_type="doc", toolsets=[reader])
        def fetch_doc():
            return "A" * 10000

        ref = fetch_doc()
        assert ref["item_count"] == 1  # Not 10000!

    def test_list_count_is_length(self, forge):
        pager = paginator(forge, "items")

        @forge.producer(workspace_type="items", toolsets=[pager])
        def fetch_items():
            return [1, 2, 3]

        ref = fetch_items()
        assert ref["item_count"] == 3


# ── Shape mismatch guard ────────────────────────────────────────────


class TestShapeMismatchGuard:
    def test_scalar_with_list_toolset_hides_tools(self, forge):
        """List tools should NOT appear in ref when data is scalar."""
        pager = paginator(forge, "doc")  # expects list
        reader = text_content(forge, "doc")  # expects scalar

        @forge.producer(workspace_type="doc", toolsets=[pager, reader])
        def fetch_doc():
            return "Some text"

        ref = fetch_doc()
        tool_names = [t.split("(")[0] for t in ref["available_tools"]]
        # List tools should be filtered out
        assert "doc_paginate" not in tool_names
        assert "doc_count" not in tool_names
        # Scalar tools should be present
        assert "doc_read_page" in tool_names
        assert "doc_search_in_text" in tool_names

    def test_list_with_dict_toolset_hides_tools(self, forge):
        """Dict tools should NOT appear in ref when data is list."""
        from ctxtual.utils import kv_reader

        pager = paginator(forge, "mixed")
        kv = kv_reader(forge, "mixed")

        @forge.producer(workspace_type="mixed", toolsets=[pager, kv])
        def fetch_data():
            return [{"a": 1}]

        ref = fetch_data()
        tool_names = [t.split("(")[0] for t in ref["available_tools"]]
        # List tools should be present
        assert "mixed_paginate" in tool_names
        # Dict tools should be filtered out
        assert "mixed_get_keys" not in tool_names

    def test_matching_shape_shows_all_tools(self, forge):
        pager = paginator(forge, "items")
        pipe = pipeline(forge, "items")

        @forge.producer(workspace_type="items", toolsets=[pager, pipe])
        def fetch_items():
            return [{"x": 1}]

        ref = fetch_items()
        tool_names = [t.split("(")[0] for t in ref["available_tools"]]
        assert "items_paginate" in tool_names
        assert "items_pipe" in tool_names
