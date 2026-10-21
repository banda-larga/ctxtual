"""
Tests for the MCP integration (``ctxtual.mcp``) against real MCP transports.

Requires the ``mcp`` extra; skipped otherwise.
"""

import asyncio
import json
import os
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


class TestOffloadMiddleware:
    async def test_large_list_is_stored_with_preview(self):
        async with Client(_server()) as c:
            r = await c.call_tool("rows", {"n": 500})
            notice = json.loads(text(r))
            assert notice["item_count"] == 500
            assert notice["stored_result"].startswith("rows_")
            assert notice["fields"] == {"id": "integer", "name": "string",
                                        "group": "integer", "pad": "string"}
            assert len(notice["preview"]) == 3
            assert "result_search" in notice["how_to_explore"]
            assert "Kept for 1 hour" in notice["how_to_explore"]
            assert len(text(r)) < 3_000
            assert r.structured_content == notice

    async def test_exploration_tools(self):
        async with Client(_server()) as c:
            ws = json.loads(text(await c.call_tool("rows", {"n": 500})))["stored_result"]
            found = json.loads(text(await c.call_tool(
                "result_search", {"workspace_id": ws, "query": "row 123"})))
            assert found["matches"][0]["id"] == 123
            piped = json.loads(text(await c.call_tool("result_pipe", {
                "workspace_id": ws,
                "steps": [{"group_by": {"field": "group", "metrics": {"n": "count"}}},
                          {"sort": {"field": "group"}}],
            })))
            assert [g["n"] for g in piped["items"]][:2] == [72, 72]
            page = json.loads(text(await c.call_tool(
                "result_paginate", {"workspace_id": ws, "page": 2, "size": 5})))
            assert [i["id"] for i in page["result"]["items"]] == [10, 11, 12, 13, 14]
            item = json.loads(text(await c.call_tool(
                "result_get_item", {"workspace_id": ws, "index": 499})))
            assert item["id"] == 499

    async def test_injected_tools_are_listed_read_only(self):
        async with Client(_server()) as c:
            tools = {t.name: t for t in await c.list_tools()}
            injected = sorted(n for n in tools if n.startswith("result_"))
            assert injected == sorted([
                "result_paginate", "result_get_item", "result_search", "result_pipe",
                "result_read_page", "result_search_in_text", "result_list",
            ])
            assert all(tools[n].annotations.read_only_hint for n in injected)
            assert "kept for 1 hour" in tools["result_search"].description
            # Results may be replaced by a notice: no outputSchema to violate.
            assert tools["rows"].output_schema is None

    async def test_small_results_pass_through(self):
        async with Client(_server()) as c:
            r = await c.call_tool("tiny", {})
            assert json.loads(text(r)) == {"ok": True}
            assert r.structured_content == {"ok": True}

    async def test_object_wrapping_a_list_is_unpacked(self):
        async with Client(_server()) as c:
            notice = json.loads(text(await c.call_tool("wrapped", {"n": 500})))
            assert notice["item_count"] == 500
            assert notice["other_fields"] == {"list_field": "items", "total": 500}

    async def test_other_objects_become_searchable_text(self):
        async with Client(_server()) as c:
            notice = json.loads(text(await c.call_tool("config", {})))
            ws = notice["stored_result"]
            hits = json.loads(text(await c.call_tool(
                "result_search_in_text", {"workspace_id": ws, "query": "key_150"})))
            assert hits["match_count"] == 1

    async def test_single_key_object_is_unwrapped(self):
        async with Client(_server()) as c:
            notice = json.loads(text(await c.call_tool("listing", {})))
            assert notice["preview"].startswith("[FILE] report_0.txt\n[FILE] report_1.txt")
            hits = json.loads(text(await c.call_tool(
                "result_search_in_text",
                {"workspace_id": notice["stored_result"], "query": "report_1999.txt"})))
            assert hits["match_count"] == 1

    async def test_long_text(self):
        async with Client(_server()) as c:
            notice = json.loads(text(await c.call_tool("doc", {})))
            assert notice["chars"] > 40_000
            hits = json.loads(text(await c.call_tool(
                "result_search_in_text",
                {"workspace_id": notice["stored_result"], "query": "needle"})))
            assert hits["match_count"] == 1
            page = json.loads(text(await c.call_tool(
                "result_read_page", {"workspace_id": notice["stored_result"]})))
            assert page["result"]["text"].startswith("lorem ipsum")

    async def test_non_text_content_passes_through(self):
        async with Client(_server()) as c:
            r = await c.call_tool("image", {})
            assert [b.type for b in r.content] == ["text", "image"]

    async def test_large_errors_are_trimmed(self):
        async with Client(_server()) as c:
            r = await c.call_tool("boom", {}, raise_on_error=False)
            assert r.is_error
            msg = text(r)
            assert len(msg) < 9_000
            assert "start" in msg and msg.rstrip().endswith("end")
            assert "omitted" in msg

    async def test_each_call_gets_its_own_workspace(self):
        # An earlier result the model still refers to must not change under it.
        async with Client(_server()) as c:
            a = json.loads(text(await c.call_tool("rows", {"n": 500})))["stored_result"]
            b = json.loads(text(await c.call_tool("rows", {"n": 500})))["stored_result"]
            assert a != b
            listing = json.loads(text(await c.call_tool("result_list", {})))
            assert listing["count"] == 2
            assert {r["arguments"]["n"] for r in listing["stored_results"]} == {500}

    async def test_bad_workspace_is_a_tool_error(self):
        async with Client(_server()) as c:
            r = await c.call_tool("result_paginate", {"workspace_id": "nope"}, raise_on_error=False)
            assert r.is_error and "not found" in text(r)

    async def test_include_exclude_and_threshold(self):
        async with Client(_server(exclude=["rows"])) as c:
            r = await c.call_tool("rows", {"n": 500})
            assert isinstance(json.loads(text(r)), list)
            assert "stored_result" in text(await c.call_tool("wrapped", {"n": 500}))
        async with Client(_server(include=["wrap*"])) as c:
            assert isinstance(json.loads(text(await c.call_tool("rows", {"n": 500}))), list)
        async with Client(_server(threshold=10_000_000)) as c:
            assert isinstance(json.loads(text(await c.call_tool("rows", {"n": 500}))), list)

    async def test_prefix_and_ttl(self):
        async with Client(_server(prefix="big", ttl=None)) as c:
            names = [t.name for t in await c.list_tools()]
            assert "big_search" in names and "result_search" not in names
            notice = json.loads(text(await c.call_tool("rows", {"n": 500})))
            assert "Kept for" not in notice["how_to_explore"]
            assert "big_pipe" in notice["how_to_explore"]


class TestReviewRegressions:
    async def test_data_with_an_error_field_is_not_a_tool_error(self):
        proxy = create_offload_proxy([sys.executable, UPSTREAM])
        async with Client(proxy) as c:
            ws = json.loads(text(await c.call_tool("log_rows", {})))["stored_result"]
            item = await c.call_tool("result_get_item", {"workspace_id": ws, "index": 1})
            assert not item.is_error
            assert json.loads(text(item))["error"] == "timeout"

    async def test_large_text_with_small_structured_content_keeps_the_text(self):
        proxy = create_offload_proxy([sys.executable, UPSTREAM])
        async with Client(proxy) as c:
            notice = json.loads(text(await c.call_tool("text_with_summary", {})))
            assert notice["chars"] > 40_000
            hits = json.loads(text(await c.call_tool(
                "result_search_in_text",
                {"workspace_id": notice["stored_result"], "query": "NEEDLE-7"})))
            assert hits["match_count"] == 1

    async def test_clashing_upstream_tool_is_not_hidden(self, monkeypatch):
        monkeypatch.setenv("UPSTREAM_CLASH", "1")
        proxy = create_offload_proxy([sys.executable, UPSTREAM])
        async with Client(proxy) as c:
            names = [t.name for t in await c.list_tools()]
            assert "result_list" in names and "ctxtual_result_list" in names
            upstream = json.loads(text(await c.call_tool("result_list", {})))
            assert upstream == {"from": "upstream"}
            notice = json.loads(text(await c.call_tool("list_issues", {"n": 800})))
            assert "ctxtual_result_search" in notice["how_to_explore"]
            ws = notice["stored_result"]
            page = json.loads(text(await c.call_tool(
                "ctxtual_result_paginate", {"workspace_id": ws, "size": 2})))
            assert page["result"]["total"] == 800

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
                    sys.executable,
                    ["-m", "ctxtual", "wrap", f"http://127.0.0.1:{port}/sse"],
                )
                async with Client(transport) as c:
                    r = await c.call_tool("list_issues", {"n": 300})
                    return json.loads(text(r))["item_count"]

            assert asyncio.run(run()) == 300
        finally:
            upstream.terminate()
            upstream.wait(5)

    def test_store_file_is_private(self, tmp_path):
        db = tmp_path / "results.db"

        async def run():
            transport = StdioTransport(
                sys.executable,
                ["-m", "ctxtual", "wrap", "--store", str(db), "--", sys.executable, UPSTREAM],
            )
            async with Client(transport) as c:
                await c.call_tool("list_issues", {"n": 300})

        asyncio.run(run())
        assert db.stat().st_mode & 0o077 == 0


class TestPreview:
    def test_parse_and_unwrap(self):
        assert parse_text_payload('[1, 2]') == [1, 2]
        assert parse_text_payload("not json") == "not json"
        assert parse_text_payload("{broken") == "{broken"
        assert unwrap_structured({"result": [1]}) == [1]
        assert unwrap_structured({"result": 1, "x": 2}) == {"result": 1, "x": 2}

    def test_preview_is_bounded(self):
        huge = [{"blob": "q" * 50_000, "nested": list(range(1000))}] * 5
        preview = build_preview(huge)
        assert len(json.dumps(preview)) <= PREVIEW_MAX_CHARS + 100
        assert preview[0]["blob"].endswith("[50,000 chars]")
        wide = {f"k{i}": "v" * 100 for i in range(500)}
        assert len(json.dumps(build_preview(wide))) <= PREVIEW_MAX_CHARS + 100


# ── wrapping other servers ────────────────────────────────────────────────


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
            assert notice["item_count"] == 1000

    async def test_cli_wrap_over_stdio(self):
        transport = StdioTransport(
            sys.executable,
            ["-m", "ctxtual", "wrap", "--threshold", "5000", "--", sys.executable, UPSTREAM],
            env={**os.environ, "CTX_TEST_SECRET": "cli"},
        )
        async with Client(transport) as c:
            notice = json.loads(text(await c.call_tool("list_issues", {"n": 300})))
            assert notice["item_count"] == 300
            found = json.loads(text(await c.call_tool(
                "result_pipe",
                {"workspace_id": notice["stored_result"],
                 "steps": [{"filter": {"id": 42}}, {"select": ["id"]}]})))
            assert found["items"] == [{"id": 42}]
            env = json.loads(text(await c.call_tool("env_value", {"name": "CTX_TEST_SECRET"})))
            assert env["value"] == "cli"


class TestProxyHttp:
    async def test_results_over_one_mebibyte(self, http_upstream):
        # The MCP SDK rejects SSE events over 1 MiB unless told otherwise.
        proxy = create_offload_proxy(http_upstream)
        async with Client(proxy) as c:
            notice = json.loads(text(await c.call_tool("list_issues", {"n": 3000})))
            assert notice["item_count"] == 3000

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
                ws = json.loads(text(await a.call_tool("list_issues", {"n": 1000})))["stored_result"]
            async with client("tok-b") as b:
                r = await b.call_tool("result_paginate", {"workspace_id": ws}, raise_on_error=False)
                assert r.is_error and "not found" in text(r)
                assert ws not in json.dumps(json.loads(text(r)).get("available_workspaces", []))
                assert json.loads(text(await b.call_tool("result_list", {})))["count"] == 0
            async with client("tok-a") as a:
                page = json.loads(text(await a.call_tool("result_paginate", {"workspace_id": ws})))
                assert page["result"]["total"] == 1000
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
                assert asyncio.run(call())["item_count"] == 400
            except TimeoutError:
                pytest.fail(f"login did not complete: opened={opened} errors={browser_errors}")
            assert len(opened) == 1
            assert asyncio.run(call())["item_count"] == 400
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
