"""
Tests for the MCP integration (``ctxtual.mcp``) against real MCP transports.

Requires the ``mcp`` extra; skipped otherwise.
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport  # noqa: E402
from fastmcp.tools.base import ToolResult  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402
from mcp_types import ImageContent, TextContent  # noqa: E402

from ctxtual.mcp import Offload, create_offload_proxy, expand_env  # noqa: E402
from ctxtual.mcp import proxy as proxy_module  # noqa: E402
from ctxtual.mcp.preview import (  # noqa: E402
    PREVIEW_MAX_CHARS,
    build_preview,
    parse_text_payload,
    unwrap_structured,
)

FIXTURES = Path(__file__).parent / "fixtures"
UPSTREAM = str(FIXTURES / "upstream_server.py")
OAUTH_UPSTREAM = str(FIXTURES / "oauth_server.py")
INSTR_UPSTREAM = str(FIXTURES / "instructions_server.py")


def text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"nothing listening on {port}")


async def await_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError(f"nothing listening on {port}")


@pytest.fixture(scope="module")
def http_upstream():
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, UPSTREAM, "http", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(5)


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CTXTUAL_HOME", str(tmp_path / "ctxtual-home"))


# ── a server using the middleware directly ────────────────────────────────


def _server(**offload_kwargs) -> FastMCP:
    mcp = FastMCP("test")

    @mcp.tool
    def rows(n: int = 500) -> list[dict]:
        """Rows."""
        return [{"id": i, "name": f"row {i}", "group": i % 7, "pad": "x" * 60} for i in range(n)]

    @mcp.tool
    def wrapped(n: int = 500) -> dict:
        """An object wrapping a list."""
        return {"items": [{"id": i, "pad": "y" * 60} for i in range(n)], "total": n}

    @mcp.tool
    def config() -> dict:
        """A big object without a dominant list."""
        return {f"key_{i}": {"value": i, "text": "z" * 80} for i in range(200)}

    @mcp.tool
    def doc() -> str:
        """Long text."""
        return ("lorem ipsum " * 2000) + " NEEDLE " + ("dolor sit " * 2000)

    @mcp.tool
    def listing() -> dict:
        """Text wrapped in a single-key object (like the filesystem server)."""
        return {"content": "\n".join(f"[FILE] report_{i}.txt" for i in range(2000))}

    @mcp.tool
    def jsonl() -> str:
        """JSON Lines log output."""
        return "\n".join(json.dumps({"ts": i, "level": ["info", "warn"][i % 2], "msg": "m" * 30})
                         for i in range(400))

    @mcp.tool
    def essays() -> list[dict]:
        """Items with a long field."""
        return [{"id": i, "author": {"name": "Ada"}, "body": f"essay {i} " + "w" * 11_000}
                for i in range(3)]

    @mcp.tool
    def manual() -> str:
        """A markdown document."""
        parts = []
        for i in range(12):
            parts.append(f"# Chapter {i}\n\n" + f"text about chapter {i}. " * 300)
            parts.append("```bash\n# not a heading\nrun --it\n```\n")
        parts.append("The streamable transport keeps one session per client.\n")
        return "\n".join(parts)

    @mcp.tool
    def tiny() -> dict:
        """Small."""
        return {"ok": True}

    @mcp.tool
    def image() -> ToolResult:
        """Big text plus an image."""
        return ToolResult(content=[
            TextContent(type="text", text="t" * 20_000),
            ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png"),
        ])

    @mcp.tool
    def boom() -> str:
        """Fails with a long message."""
        raise ValueError("start " + "e" * 30_000 + " end")

    mcp.add_middleware(Offload(**offload_kwargs))
    return mcp


class TestNotice:
    async def test_large_list_is_stored_with_a_concrete_example(self):
        async with Client(_server()) as c:
            r = await c.call_tool("rows", {"n": 500})
            notice = json.loads(text(r))
            assert notice["result_id"].startswith("rows_")
            assert notice["items"] == 500
            assert notice["fields"] == {"id": "integer", "name": "string",
                                        "group": "integer", "pad": "string"}
            assert len(notice["preview"]) == 3
            how = notice["how_to_explore"]
            assert how.startswith("rows returned 500 items")
            # the example uses the result's real fields
            assert '{"sort": {"field": "group", "order": "desc"}}' in how
            assert "result_search(result_id, query)" in how and "narrower arguments" in how
            assert len(text(r)) < 2_500 and r.structured_content == notice

    async def test_text_notice(self):
        async with Client(_server()) as c:
            notice = json.loads(text(await c.call_tool("doc", {})))
            assert notice["chars"] > 40_000
            assert "pages of 4,000 chars" in notice["how_to_explore"]

    async def test_small_results_and_non_text_pass_through(self):
        async with Client(_server()) as c:
            r = await c.call_tool("tiny", {})
            assert json.loads(text(r)) == {"ok": True} and r.structured_content == {"ok": True}
            r = await c.call_tool("image", {})
            assert [b.type for b in r.content] == ["text", "image"]

    async def test_payload_shapes(self):
        async with Client(_server()) as c:
            wrapped = json.loads(text(await c.call_tool("wrapped", {"n": 500})))
            assert wrapped["items"] == 500
            assert wrapped["other_fields"] == {"list_field": "items", "total": 500}
            listing = json.loads(text(await c.call_tool("listing", {})))
            assert listing["preview"].startswith("[FILE] report_0.txt\n")
            config = json.loads(text(await c.call_tool("config", {})))
            assert "chars" in config  # an object without a dominant list → text
            logs = json.loads(text(await c.call_tool("jsonl", {})))
            assert logs["items"] == 400 and logs["fields"]["level"] == "string"

    async def test_large_errors_are_trimmed(self):
        async with Client(_server()) as c:
            r = await c.call_tool("boom", {}, raise_on_error=False)
            msg = text(r)
            assert r.is_error and len(msg) < 9_000
            assert "start" in msg and msg.rstrip().endswith("end") and "omitted" in msg

    async def test_each_call_gets_its_own_result(self):
        async with Client(_server()) as c:
            a = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            b = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            assert a != b
            listing = json.loads(text(await c.call_tool("result_list", {})))
            assert [r["result_id"] for r in listing["stored_results"]] == [b, a]
            assert listing["stored_results"][0] == {
                "result_id": b, "tool": "rows", "arguments": {"n": 500}, "items": 500}


class TestDocuments:
    async def test_markdown_outline(self):
        async with Client(_server()) as c:
            notice = json.loads(text(await c.call_tool("manual", {})))
            outline = notice["outline"]
            assert outline[0] == "0: Chapter 0" and len(outline) == 12
            assert not any("not a heading" in line for line in outline)
            page = int(outline[5].split(":")[0])
            read = json.loads(text(await c.call_tool(
                "result_read", {"result_id": notice["result_id"], "page": page})))
            assert "Chapter 5" in read["text"] or "chapter 5" in read["text"]
            assert len(text(await c.call_tool("manual", {}))) < 2_000

    async def test_multi_word_search_ranks_passages(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("manual", {})))["result_id"]
            found = json.loads(text(await c.call_tool(
                "result_search", {"result_id": rid, "query": "streamable session client"})))
            assert found["match"] == "words" and "note" in found
            assert found["passages"][0]["words"] == ["client", "session", "streamable"]
            exact = json.loads(text(await c.call_tool(
                "result_search", {"result_id": rid, "query": "one session per"})))
            assert exact["match"] == "phrase" and exact["returned"] == 1


class TestTools:
    async def test_tool_list_is_small_and_read_only(self):
        async with Client(_server()) as c:
            tools = {t.name: t for t in await c.list_tools()}
            injected = sorted(n for n in tools if n.startswith("result_"))
            assert injected == ["result_list", "result_query", "result_read", "result_search"]
            assert all(tools[n].annotations.read_only_hint for n in injected)
            size = sum(len(json.dumps({"d": tools[n].description, "s": tools[n].input_schema}))
                       for n in injected)
            assert size < 4_000  # every request carries these definitions
            for n in injected:
                assert "Args:" not in tools[n].description  # no duplicated parameter docs
            assert tools["rows"].output_schema is None

    async def test_search_items_with_index_and_fields(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            found = json.loads(text(await c.call_tool(
                "result_search", {"result_id": rid, "query": "row 123", "fields": ["id", "name"]})))
            assert found["hits"][0] == {"index": 123, "item": {"id": 123, "name": "row 123"}}
            assert found["returned"] <= 10 and found["matching_any_word"] >= found["returned"]
            assert found["matching_all_words"] >= 1 and found["max_results"] == 10

    async def test_search_text_passages(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("doc", {})))["result_id"]
            found = json.loads(text(await c.call_tool(
                "result_search", {"result_id": rid, "query": "NEEDLE"})))
            assert found["returned"] == 1 and "NEEDLE" in found["passages"][0]["text"]
            assert found["passages"][0]["page"] == found["passages"][0]["offset"] // 4000

    async def test_read_pages_items_and_long_fields(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            page = json.loads(text(await c.call_tool(
                "result_read", {"result_id": rid, "page": 2, "page_size": 5})))
            assert [i["index"] for i in page["items"]] == [10, 11, 12, 13, 14]
            assert page["pages"] == 100 and page["has_more"]
            one = json.loads(text(await c.call_tool("result_read", {"result_id": rid, "index": 499})))
            assert one["item"]["id"] == 499

            rid = json.loads(text(await c.call_tool("essays", {})))["result_id"]
            listed = json.loads(text(await c.call_tool("result_read", {"result_id": rid})))
            body = listed["items"][0]["item"]["body"]
            # long values are cut, with the exact call to read them
            assert len(body) < 500 and "result_read(result_id, index=0, field='body')" in body
            full = json.loads(text(await c.call_tool(
                "result_read", {"result_id": rid, "index": 0, "field": "body", "page": 1})))
            assert full["pages"] == 3 and full["page"] == 1 and len(full["text"]) == 4000
            nested = json.loads(text(await c.call_tool(
                "result_read", {"result_id": rid, "index": 0, "field": "author.name"})))
            assert nested["value"] == "Ada"

    async def test_read_text(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("doc", {})))["result_id"]
            page = json.loads(text(await c.call_tool("result_read", {"result_id": rid})))
            assert page["text"].startswith("lorem ipsum") and page["has_more"]

    async def test_query(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            grouped = json.loads(text(await c.call_tool("result_query", {
                "result_id": rid,
                "steps": [{"group_by": {"field": "group", "metrics": {"n": "count"}}},
                          {"sort": {"field": "group"}}],
            })))
            assert [g["n"] for g in grouped["rows"]][:2] == [72, 72]
            everything = json.loads(text(await c.call_tool(
                "result_query", {"result_id": rid, "steps": []})))
            assert everything["returned"] == 50 and everything["total"] == 500
            assert "save_as" in everything["note"]
            count = json.loads(text(await c.call_tool(
                "result_query", {"result_id": rid, "steps": [{"count": True}]})))
            assert count["result"] == {"count": 500}

    async def test_query_save_as(self):
        async with Client(_server()) as c:
            tools = {t.name for t in await c.list_tools()}
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            saved = json.loads(text(await c.call_tool("result_query", {
                "result_id": rid, "steps": [{"filter": {"group": 1}}], "save_as": "g1"})))
            assert saved["result_id"] == "saved_g1" and saved["items"] == 72
            assert saved["how_to_explore"].startswith("Stored 72 items from result_query")
            mentioned = set(re.findall(r"\b(result_[a-z_]+)\(", saved["how_to_explore"]))
            assert mentioned and mentioned <= tools
            page = json.loads(text(await c.call_tool("result_read", {"result_id": "saved_g1"})))
            assert page["total"] == 72

    async def test_errors_are_fixable(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            for call, needle in [
                ({"result_id": "nope"}, "No stored result"),
                ({"result_id": rid, "index": 9999}, "out of range"),
                ({"result_id": rid, "index": 0, "field": "missing"}, "no field"),
                ({"result_id": rid, "page": 999}, "past the end"),
            ]:
                r = await c.call_tool("result_read", call, raise_on_error=False)
                body = json.loads(text(r))
                assert r.is_error and needle in body["error"] and body["suggested_action"]
            bad = await c.call_tool("result_query", {"result_id": rid, "steps": [{"where": {}}]},
                                    raise_on_error=False)
            assert bad.is_error and "available_operations" in json.loads(text(bad))
            on_text = json.loads(text(await c.call_tool("doc", {})))["result_id"]
            r = await c.call_tool("result_query", {"result_id": on_text, "steps": []},
                                  raise_on_error=False)
            assert r.is_error and "result_search" in json.loads(text(r))["suggested_action"]
            unknown = json.loads(text(await c.call_tool(
                "result_read", {"result_id": "nope"}, raise_on_error=False)))
            assert rid in unknown["stored_results"]

    async def test_output_is_bounded(self):
        async with Client(_server(max_output_chars=3_000)) as c:
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            r = await c.call_tool("result_read", {"result_id": rid, "page_size": 50},
                                  raise_on_error=False)
            assert r.is_error and "fields=" in json.loads(text(r))["suggested_action"]
            ok = await c.call_tool("result_read", {"result_id": rid, "page_size": 50,
                                                   "fields": ["id"]})
            assert len(json.loads(text(ok))["items"]) == 50


class TestConfiguration:
    async def test_include_exclude_and_thresholds(self):
        async with Client(_server(exclude=["rows"])) as c:
            assert isinstance(json.loads(text(await c.call_tool("rows", {"n": 500}))), list)
            assert "result_id" in text(await c.call_tool("wrapped", {"n": 500}))
        async with Client(_server(include=["wrap*"])) as c:
            assert isinstance(json.loads(text(await c.call_tool("rows", {"n": 500}))), list)
        async with Client(_server(thresholds={"rows": 10_000_000})) as c:
            assert isinstance(json.loads(text(await c.call_tool("rows", {"n": 500}))), list)
            assert "result_id" in text(await c.call_tool("wrapped", {"n": 500}))

    async def test_prefix_tools_and_ttl(self):
        async with Client(_server(prefix="big", tools=["search", "read"], ttl=None)) as c:
            names = {t.name for t in await c.list_tools()}
            assert {"big_search", "big_read"} <= names
            assert not names & {"big_query", "big_list", "result_search"}
            notice = json.loads(text(await c.call_tool("rows", {"n": 500})))
            assert "big_search" in notice["how_to_explore"]
            assert "big_query" not in notice["how_to_explore"]
        with pytest.raises(ValueError, match="unknown tools"):
            Offload(tools=["search", "nope"])

    async def test_idle_results_expire(self):
        server = _server(ttl=0.2)
        async with Client(server) as c:
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            await asyncio.sleep(0.1)
            await c.call_tool("result_read", {"result_id": rid})  # use keeps it alive
            await asyncio.sleep(0.15)
            assert not (await c.call_tool("result_read", {"result_id": rid})).is_error
            await asyncio.sleep(0.3)
            r = await c.call_tool("result_read", {"result_id": rid}, raise_on_error=False)
            assert r.is_error and "expired" in text(r)

    async def test_eviction_by_count(self):
        async with Client(_server(max_results=2)) as c:
            ids = [json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
                   for _ in range(3)]
            listing = json.loads(text(await c.call_tool("result_list", {})))
            assert [r["result_id"] for r in listing["stored_results"]] == ids[:0:-1]

    async def test_instructions_are_appended(self):
        mcp = FastMCP("own", instructions="OWN-GUIDANCE")
        mcp.add_middleware(Offload())
        async with Client(mcp) as c:
            assert c.instructions.startswith("OWN-GUIDANCE\n\nLarge tool results:")
        mcp = FastMCP("quiet")
        mcp.add_middleware(Offload(instructions=False))
        async with Client(mcp) as c:
            assert not c.instructions


class TestRedesignReview:
    """Findings of the review of the 0.4 redesign."""

    def _mcp(self, tool_fn, **offload):
        mcp = FastMCP("t")
        mcp.tool(tool_fn)
        mcp.add_middleware(Offload(**offload))
        return mcp

    async def test_stored_result_survives_its_own_eviction(self):
        def big() -> list[dict]:
            return [{"i": i, "pad": "x" * 100} for i in range(300)]

        async with Client(self._mcp(big, max_total_chars=10_000)) as c:
            rid = json.loads(text(await c.call_tool("big", {})))["result_id"]
            assert not (await c.call_tool("result_read", {"result_id": rid})).is_error
        with pytest.raises(ValueError):
            Offload(max_results=0)

    async def test_dotted_keys_lists_and_nulls_are_readable(self):
        def odd() -> list:
            return [{"a.b": "v" * 3000, "deep": {"x.y": {"z": "w" * 3000}}, "n": None},
                    ["s" * 10_000 + "END", 0], "t" * 10_000] * 3

        async with Client(self._mcp(odd)) as c:
            rid = json.loads(text(await c.call_tool("odd", {})))["result_id"]
            first = json.loads(text(await c.call_tool("result_read", {"result_id": rid, "index": 0})))
            for key in ("a.b", "deep.x.y.z"):
                assert f"field='{key}'" in json.dumps(first)
                got = await c.call_tool("result_read", {"result_id": rid, "index": 0, "field": key})
                assert not got.is_error
            null = json.loads(text(await c.call_tool(
                "result_read", {"result_id": rid, "index": 0, "field": "n"})))
            assert null["value"] is None
            listing = json.loads(text(await c.call_tool("result_read", {"result_id": rid})))
            assert "field='0'" in json.dumps(listing["items"][1])
            assert "result_read(result_id, index=2)" in json.dumps(listing["items"][2])
            tail = json.loads(text(await c.call_tool(
                "result_read", {"result_id": rid, "index": 1, "field": "0", "page": 2})))
            assert tail["text"].endswith("END")

    async def test_result_list_and_query_fit_the_budget(self):
        def wide(note: str = "") -> list[dict]:
            return [{"k": i, **{f"f{j}": "y" * 450 for j in range(4)}} for i in range(300)]

        async with Client(self._mcp(wide)) as c:
            for _ in range(30):
                await c.call_tool("wide", {"note": "n" * 1000})
            listing = await c.call_tool("result_list", {})
            assert not listing.is_error
            rid = json.loads(text(listing))["stored_results"][0]["result_id"]
            rows = json.loads(text(await c.call_tool(
                "result_query", {"result_id": rid, "steps": [{"sort": "k"}]})))
            assert rows["returned"] < 50 and "rows shown" in rows["note"]
            assert rows["rows"][0]["index"] == 0  # rows can be opened

    async def test_query_rows_carry_index_even_after_select(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            out = json.loads(text(await c.call_tool("result_query", {"result_id": rid, "steps": [
                {"filter": {"group": 3}}, {"select": ["id"]}, {"limit": 2}]})))
            assert out["rows"] == [{"index": 3, "row": {"id": 3}}, {"index": 10, "row": {"id": 10}}]
            grouped = json.loads(text(await c.call_tool("result_query", {"result_id": rid, "steps": [
                {"group_by": {"field": "group"}}]})))
            assert "index" not in grouped["rows"][0]
            single = json.loads(text(await c.call_tool(
                "result_query", {"result_id": rid, "steps": {"limit": 1}})))
            assert single["returned"] == 1  # a single step dict is accepted

    async def test_invalid_arguments_get_the_fixable_shape(self):
        async with Client(_server()) as c:
            for name, args in [("result_read", {"result_id": None}),
                               ("result_read", {"result_id": "x", "page": "two"}),
                               ("result_search", {"result_id": "x", "query": 5}),
                               ("result_query", {"result_id": "x", "steps": 3})]:
                r = await c.call_tool(name, args, raise_on_error=False)
                body = json.loads(text(r))
                assert r.is_error and body["suggested_action"], (name, body)
                assert "pydantic" not in json.dumps(body)
            rid = json.loads(text(await c.call_tool("rows", {"n": 500})))["result_id"]
            ok = json.loads(text(await c.call_tool(
                "result_read", {"result_id": rid, "page": "2", "fields": "id"})))
            assert ok["page"] == 2 and ok["items"][0]["item"] == {"id": 20}

    def test_passages(self):
        from ctxtual.mcp.offload import _passages

        text_ = ("filler " * 200) + "alpha" + "." * 280 + "beta" + (" tail" * 50)
        found = _passages(text_, "alpha beta", 3)
        top = found["passages"][0]
        assert top["words"] == ["alpha", "beta"] and "beta" in top["text"] and "alpha" in top["text"]
        rare = _passages(("the cat " * 30000) + " zebra rarefind word", "rarefind the zebra", 3)
        assert "rarefind" in rare["passages"][0]["words"]
        assert "rarefind" in rare["passages"][0]["text"]
        uni = _passages("İ" * 5000 + " NEEDLE here", "needle", 3)
        assert uni["passages"][0]["offset"] == 5001 and "NEEDLE" in uni["passages"][0]["text"]
        repeated = _passages("error: x " * 2000, "error", 10)
        assert repeated["passages"][0].get("occurrences", 1) > 1
        assert repeated["occurrences"] == 2000

    def test_outline_edge_cases(self):
        from ctxtual.mcp.offload import _outline

        doc = "# My Project\n" + "".join(f"## Section {i}\n\ntext\n" for i in range(40))
        out = _outline(doc)
        assert out[0] == "0: My Project" and len(out) > 3
        assert _outline("# Setup\n## Advanced Setup\n## Other\n") == [
            "0: Setup", "  0: Advanced Setup", "  0: Other"]
        fenced = "```\n~~~\n# not\n```\n# A\n# B\n# C\n"
        assert _outline(fenced) == ["0: A", "0: B", "0: C"]
        indented = "    ```\n# A\n# B\n## F#\n"
        assert _outline(indented) == ["0: A", "0: B", "  0: F#"]

    def test_examples_skip_dotted_field_names(self):
        from ctxtual.mcp.offload import _example_steps

        rows = [{"meta.v": i, "kind": ["a", "b"][i % 2]} for i in range(50)]
        steps = json.loads(_example_steps(rows, {"meta.v": "integer", "kind": "string"}))
        assert steps == [{"filter": {"kind": "a"}}, {"limit": 10}]

    async def test_descriptions_and_notices_follow_the_tool_set(self):
        async with Client(_server(tools=["search"])) as c:
            desc = {t.name: t.description for t in await c.list_tools()}
            assert "result_read" not in desc["result_search"]
            notice = json.loads(text(await c.call_tool("doc", {})))
            assert "narrower arguments" not in notice["how_to_explore"]  # doc() has none
        with pytest.raises(ValueError):
            Offload(tools=[])
        assert Offload(tools=["Search"]).tools == ("search",)

    async def test_read_on_text_rejects_list_arguments(self):
        async with Client(_server()) as c:
            rid = json.loads(text(await c.call_tool("config", {})))["result_id"]
            r = await c.call_tool("result_read", {"result_id": rid, "index": 1},
                                  raise_on_error=False)
            assert r.is_error and "is a text" in json.loads(text(r))["error"]

    async def test_not_found_lists_only_live_results(self):
        async with Client(_server(ttl=0.2)) as c:
            await c.call_tool("rows", {"n": 500})
            await asyncio.sleep(0.4)
            r = json.loads(text(await c.call_tool(
                "result_read", {"result_id": "nope"}, raise_on_error=False)))
            assert "stored_results" not in r

    async def test_nested_wrappers_stay_queryable(self):
        def api() -> dict:
            return {"data": {"items": [{"id": i, "v": "z" * 50} for i in range(400)]},
                    "meta": {"page": 1}}

        async with Client(self._mcp(api)) as c:
            notice = json.loads(text(await c.call_tool("api", {})))
            assert notice["items"] == 400
            assert notice["other_fields"]["list_field"] == "data.items"

    def test_trim_and_duration(self):
        from ctxtual.mcp.offload import _duration, _trim

        assert _trim("e" * 350, 300) == "e" * 350 or len(_trim("e" * 350, 300)) < 350
        assert _duration(90) == "2 minutes" and _duration(0.5) == "1 second"
        assert _duration(3600) == "1 hour"

    def test_cli_rejects_bad_tool_options(self, capsys):
        from ctxtual.cli import main

        assert main(["wrap", "--tools", "search,bogus", "--", "x"]) == 2
        assert main(["wrap", "--tools", ",,", "--", "x"]) == 2
        assert main(["wrap", "--tool-threshold", "=5", "--", "x"]) == 2
        assert main(["wrap", "--max-results", "0", "--", "x"]) == 2

    def test_custom_name_is_kept(self):
        async def run():
            transport = StdioTransport(sys.executable, [
                "-m", "ctxtual", "wrap", "--name", "my-name", "--", sys.executable, INSTR_UPSTREAM])
            async with Client(transport) as c:
                return c.server_info.name, c.instructions

        name, instructions = asyncio.run(run())
        assert name == "my-name" and instructions.startswith("UPSTREAM-GUIDANCE")

    def test_clash_rename_is_reflected_in_instructions(self, monkeypatch):
        monkeypatch.setenv("UPSTREAM_CLASH", "1")

        async def run():
            transport = StdioTransport(sys.executable, ["-m", "ctxtual", "wrap", "--",
                                                        sys.executable, UPSTREAM],
                                       env={**os.environ})
            async with Client(transport) as c:
                return c.instructions

        instructions = asyncio.run(run())
        assert "ctxtual_result_search" in instructions
        assert "result_list shows" not in instructions.replace("ctxtual_result_list", "")

    async def test_eviction_is_per_user(self):
        from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

        auth = StaticTokenVerifier({
            "a": {"client_id": "alice", "sub": "alice", "scopes": []},
            "b": {"client_id": "bob", "sub": "bob", "scopes": []},
        })
        port = free_port()
        proxy = create_offload_proxy([sys.executable, UPSTREAM], auth=auth,
                                     shared_session=False, max_results=2)
        task = asyncio.create_task(proxy.run_async(
            transport="http", host="127.0.0.1", port=port, show_banner=False))
        url = f"http://127.0.0.1:{port}/mcp"

        def client(token):
            return Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"}))

        try:
            await await_port(port)
            async with client("a") as a:
                alice = json.loads(text(await a.call_tool("list_issues", {"n": 600})))["result_id"]
            async with client("b") as b:
                for _ in range(3):
                    await b.call_tool("list_issues", {"n": 600})
            async with client("a") as a:
                assert not (await a.call_tool("result_read", {"result_id": alice})).is_error
        finally:
            task.cancel()


class TestReviewRegressions:
    async def test_data_with_an_error_field_is_not_a_tool_error(self):
        proxy = create_offload_proxy([sys.executable, UPSTREAM])
        async with Client(proxy) as c:
            rid = json.loads(text(await c.call_tool("log_rows", {})))["result_id"]
            item = await c.call_tool("result_read", {"result_id": rid, "index": 1})
            assert not item.is_error
            assert json.loads(text(item))["item"]["error"] == "timeout"

    async def test_large_text_with_small_structured_content_keeps_the_text(self):
        proxy = create_offload_proxy([sys.executable, UPSTREAM])
        async with Client(proxy) as c:
            notice = json.loads(text(await c.call_tool("text_with_summary", {})))
            assert notice["chars"] > 40_000
            found = json.loads(text(await c.call_tool(
                "result_search", {"result_id": notice["result_id"], "query": "NEEDLE-7"})))
            assert found["returned"] == 1

    async def test_clashing_upstream_tool_is_not_hidden(self, monkeypatch):
        monkeypatch.setenv("UPSTREAM_CLASH", "1")
        proxy = create_offload_proxy([sys.executable, UPSTREAM])
        async with Client(proxy) as c:
            names = [t.name for t in await c.list_tools()]
            assert "result_list" in names and "ctxtual_result_list" in names
            assert json.loads(text(await c.call_tool("result_list", {}))) == {"from": "upstream"}
            notice = json.loads(text(await c.call_tool("list_issues", {"n": 800})))
            assert "ctxtual_result_search" in notice["how_to_explore"]
            page = json.loads(text(await c.call_tool(
                "ctxtual_result_read", {"result_id": notice["result_id"], "page_size": 2})))
            assert page["total"] == 800

    async def test_unreachable_upstream_is_reported(self):
        proxy = create_offload_proxy(f"http://127.0.0.1:{free_port()}/mcp")
        async with Client(proxy) as c:
            with pytest.raises(MCPError, match="failed to connect"):
                await c.list_tools()

    def test_sse_url_is_detected_by_the_cli(self):
        port = free_port()
        upstream = subprocess.Popen([sys.executable, UPSTREAM, "sse", str(port)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_for_port(port)

            async def run():
                transport = StdioTransport(
                    sys.executable, ["-m", "ctxtual", "wrap", f"http://127.0.0.1:{port}/sse"])
                async with Client(transport) as c:
                    return json.loads(text(await c.call_tool("list_issues", {"n": 300})))["items"]

            assert asyncio.run(run()) == 300
        finally:
            upstream.terminate()
            upstream.wait(5)

    def test_store_file_is_private(self, tmp_path):
        db = tmp_path / "results.db"

        async def run():
            transport = StdioTransport(
                sys.executable,
                ["-m", "ctxtual", "wrap", "--store", str(db), "--", sys.executable, UPSTREAM])
            async with Client(transport) as c:
                await c.call_tool("list_issues", {"n": 300})

        asyncio.run(run())
        assert db.stat().st_mode & 0o077 == 0


class TestWrappedServerMetadata:
    def test_upstream_instructions_and_name_are_kept(self):
        async def run():
            transport = StdioTransport(
                sys.executable, ["-m", "ctxtual", "wrap", "--", sys.executable, INSTR_UPSTREAM])
            async with Client(transport) as c:
                return c.instructions, c.server_info.name

        instructions, name = asyncio.run(run())
        assert instructions.startswith("UPSTREAM-GUIDANCE: always pass org=acme.\n\n")
        assert "result_search" in instructions
        assert name == "with-instructions"


class TestPreview:
    def test_parse_and_unwrap(self):
        assert parse_text_payload('[1, 2]') == [1, 2]
        assert parse_text_payload("not json") == "not json"
        assert parse_text_payload("{broken") == "{broken"
        assert parse_text_payload('{"a": 1}\n{"a": 2}\n') == [{"a": 1}, {"a": 2}]
        assert parse_text_payload('{"a": 1}\nplain line') == '{"a": 1}\nplain line'
        assert unwrap_structured({"result": [1]}) == [1]
        assert unwrap_structured({"result": 1, "x": 2}) == {"result": 1, "x": 2}

    def test_preview_is_bounded(self):
        huge = [{"blob": "q" * 50_000, "nested": list(range(1000))}] * 5
        preview = build_preview(huge)
        assert len(json.dumps(preview)) <= PREVIEW_MAX_CHARS + 100
        assert preview[0]["blob"].endswith("[50,000 chars]")
        wide = {f"k{i}": "v" * 100 for i in range(500)}
        assert len(json.dumps(build_preview(wide))) <= PREVIEW_MAX_CHARS + 100

    def test_example_steps_use_category_fields(self):
        from ctxtual.mcp.offload import _example_steps

        rows = [{"id": i, "title": f"t{i}", "state": ["open", "closed"][i % 2], "n": i}
                for i in range(100)]
        steps = json.loads(_example_steps(rows, {"id": "integer", "title": "string",
                                                 "state": "string", "n": "integer"}))
        assert steps == [{"filter": {"state": "open"}},
                         {"sort": {"field": "n", "order": "desc"}}, {"limit": 10}]


class TestProxyStdio:
    async def test_env_state_and_offload(self, monkeypatch):
        monkeypatch.setenv("CTX_TEST_SECRET", "from-client-config")
        proxy = create_offload_proxy([sys.executable, UPSTREAM])
        async with Client(proxy) as c:
            env = json.loads(text(await c.call_tool("env_value", {"name": "CTX_TEST_SECRET"})))
            # The SDK would only pass PATH/HOME/…; the wrapper forwards all.
            assert env["value"] == "from-client-config"
            hits = [json.loads(text(await c.call_tool("hit", {}))) for _ in range(3)]
            assert [h["hits"] for h in hits] == [1, 2, 3]  # one upstream process
            assert len({h["pid"] for h in hits}) == 1
            notice = json.loads(text(await c.call_tool("list_issues", {"n": 1000})))
            assert notice["items"] == 1000

    async def test_cli_wrap_over_stdio(self):
        transport = StdioTransport(
            sys.executable,
            ["-m", "ctxtual", "wrap", "--threshold", "5000", "--", sys.executable, UPSTREAM],
            env={**os.environ, "CTX_TEST_SECRET": "cli"},
        )
        async with Client(transport) as c:
            notice = json.loads(text(await c.call_tool("list_issues", {"n": 300})))
            assert notice["items"] == 300
            found = json.loads(text(await c.call_tool(
                "result_query",
                {"result_id": notice["result_id"],
                 "steps": [{"filter": {"id": 42}}, {"select": ["id"]}]})))
            assert found["rows"] == [{"index": 42, "row": {"id": 42}}]
            env = json.loads(text(await c.call_tool("env_value", {"name": "CTX_TEST_SECRET"})))
            assert env["value"] == "cli"


class TestProxyHttp:
    async def test_results_over_one_mebibyte(self, http_upstream):
        # The MCP SDK rejects SSE events over 1 MiB unless told otherwise.
        proxy = create_offload_proxy(http_upstream)
        async with Client(proxy) as c:
            notice = json.loads(text(await c.call_tool("list_issues", {"n": 3000})))
            assert notice["items"] == 3000

    async def test_configured_headers_reach_upstream_caller_headers_do_not(self, http_upstream):
        port = free_port()
        proxy = create_offload_proxy(http_upstream, headers={"X-Api-Key": "upstream-key"})
        task = asyncio.create_task(proxy.run_async(
            transport="http", host="127.0.0.1", port=port, show_banner=False))
        try:
            await await_port(port)
            front = StreamableHttpTransport(
                f"http://127.0.0.1:{port}/mcp",
                headers={"Authorization": "Bearer FRONT", "X-Leak": "front"},
            )
            async with Client(front) as c:
                seen = json.loads(text(await c.call_tool("seen_headers", {})))["headers"]
            assert seen.get("x-api-key") == "upstream-key"
            assert "authorization" not in seen
            assert "x-leak" not in seen
        finally:
            task.cancel()

    async def test_results_are_isolated_per_principal(self):
        from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

        auth = StaticTokenVerifier({
            "tok-a": {"client_id": "alice", "sub": "alice", "scopes": []},
            "tok-b": {"client_id": "bob", "sub": "bob", "scopes": []},
        })
        port = free_port()
        proxy = create_offload_proxy([sys.executable, UPSTREAM], auth=auth, shared_session=False)
        task = asyncio.create_task(proxy.run_async(
            transport="http", host="127.0.0.1", port=port, show_banner=False))
        url = f"http://127.0.0.1:{port}/mcp"

        def client(token):
            return Client(StreamableHttpTransport(url, headers={"Authorization": f"Bearer {token}"}))

        try:
            await await_port(port)
            async with client("tok-a") as a:
                ws = json.loads(text(await a.call_tool("list_issues", {"n": 1000})))["result_id"]
            async with client("tok-b") as b:
                r = await b.call_tool("result_read", {"result_id": ws}, raise_on_error=False)
                assert r.is_error and "No stored result" in text(r)
                assert ws not in json.dumps(json.loads(text(r)).get("stored_results", []))
                assert json.loads(text(await b.call_tool("result_list", {})))["stored_results"] == []
            async with client("tok-a") as a:
                page = json.loads(text(await a.call_tool("result_read", {"result_id": ws})))
                assert page["total"] == 1000
        finally:
            task.cancel()


class TestCliHttpFront:
    def test_rejects_foreign_origin_and_host(self):
        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, "-m", "ctxtual", "wrap", "--http", "--port", str(port),
             "--", sys.executable, UPSTREAM],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            wait_for_port(port)

            def post(headers):
                body = json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "probe", "version": "1"}},
                }).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/mcp", data=body, method="POST",
                    headers={"Accept": "application/json, text/event-stream",
                             "Content-Type": "application/json", **headers},
                )
                try:
                    return urllib.request.urlopen(req, timeout=10).status
                except urllib.error.HTTPError as exc:
                    return exc.code

            assert post({}) == 200
            assert post({"Origin": f"http://localhost:{port}"}) == 200
            assert post({"Origin": "https://evil.example"}) == 403
            assert post({"Host": "evil.example"}) == 421  # DNS rebinding
        finally:
            proc.terminate()
            proc.wait(5)

    def test_refuses_non_loopback_bind(self):
        r = subprocess.run(
            [sys.executable, "-m", "ctxtual", "wrap", "--http", "--host", "0.0.0.0",
             "https://example.com/mcp"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
        assert "refusing to listen" in r.stderr


class TestOAuth:
    def test_login_once_then_reuse_token(self, monkeypatch, tmp_path):
        import webbrowser

        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, OAUTH_UPSTREAM, str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        opened = []
        browser_errors = []

        def follow(url):
            # FastMCP opens the browser before its callback server listens;
            # a person takes seconds, this stand-in retries instead.
            for _ in range(50):
                try:
                    urllib.request.urlopen(url, timeout=10).read()
                    return
                except urllib.error.URLError as exc:
                    if not isinstance(exc.reason, ConnectionRefusedError):
                        browser_errors.append(repr(exc))
                        return
                    time.sleep(0.1)
            browser_errors.append("callback server never started")

        def browser(url, *args, **kwargs):
            # Stand-in for the user: follow the authorization redirect.
            opened.append(url)
            threading.Thread(target=follow, args=(url,), daemon=True).start()
            return True

        monkeypatch.setattr(webbrowser, "open", browser)

        async def call():
            proxy = create_offload_proxy(f"http://127.0.0.1:{port}/mcp", oauth=True)
            async with Client(proxy) as c:
                r = await asyncio.wait_for(c.call_tool("list_issues", {"n": 400}), 60)
                return json.loads(text(r))

        try:
            wait_for_port(port)
            try:
                assert asyncio.run(call())["items"] == 400
            except TimeoutError:
                pytest.fail(f"login did not complete: opened={opened} errors={browser_errors}")
            assert len(opened) == 1
            assert asyncio.run(call())["items"] == 400
            assert len(opened) == 1  # token was reused, no second login
            oauth_dir = Path(os.environ["CTXTUAL_HOME"]) / "oauth"
            assert oauth_dir.stat().st_mode & 0o077 == 0
        finally:
            proc.terminate()
            proc.wait(5)

    async def test_oauth_is_harmless_without_auth(self, http_upstream):
        proxy = create_offload_proxy(http_upstream, oauth=True)
        async with Client(proxy) as c:
            assert json.loads(text(await c.call_tool("small", {}))) == {"ok": True, "value": 7}


class TestHelpers:
    def test_expand_env(self):
        assert expand_env("Bearer ${TOK}", {"TOK": "abc"}) == "Bearer abc"
        assert expand_env("plain", {}) == "plain"
        with pytest.raises(ValueError, match="MISSING"):
            expand_env("${MISSING}", {})

    def test_sse_hook_changes_nothing_by_default(self, monkeypatch):
        calls = []

        def fake(url, **kwargs):
            calls.append(kwargs)

        fake.__ctxtual_hook__ = False
        monkeypatch.setattr(proxy_module._fastmcp_http, "streamable_http_client", fake)
        proxy_module._install_sse_limit_hook()
        hooked = proxy_module._fastmcp_http.streamable_http_client
        hooked("http://x")
        assert calls[-1] == {}  # other FastMCP clients keep the SDK default
        token = proxy_module._max_sse_event.set(123)
        try:
            hooked("http://x")
        finally:
            proxy_module._max_sse_event.reset(token)
        assert calls[-1] == {"max_sse_event_size": 123}
