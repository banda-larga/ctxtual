"""
Upstream MCP server used by the MCP integration tests.

Run with no arguments for stdio, ``http PORT`` for streamable HTTP or
``sse PORT`` for the legacy SSE transport.  Set ``UPSTREAM_CLASH=1`` to also
expose a tool named ``result_list`` (clashes with ctxtual's).
"""

import os
import sys

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools.base import ToolResult
from mcp_types import ImageContent, TextContent

mcp = FastMCP("upstream")
_hits = {"n": 0}


def _issues(n: int) -> list[dict]:
    states = ["open", "closed"]
    labels = ["bug", "feature", "docs", "security"]
    items = [
        {
            "id": i,
            "title": f"Issue number {i} about {labels[i % 4]}",
            "state": states[i % 2],
            "labels": [labels[i % 4]],
            "comments": i % 37,
            "body": "lorem ipsum dolor sit amet " * 8,
        }
        for i in range(n)
    ]
    if n > 500:
        items[500]["title"] = "Data loss when exporting invoices"
    return items


@mcp.tool
def list_issues(n: int = 1000) -> list[dict]:
    """List issues."""
    return _issues(n)


@mcp.tool
def issues_page(n: int = 1000) -> dict:
    """Issues wrapped in an object with paging metadata."""
    return {"items": _issues(n), "total": n, "next_cursor": "abc"}


@mcp.tool
def long_text(chars: int = 50_000) -> str:
    """A long plain-text document."""
    base = "The quick brown fox jumps over the lazy dog. "
    text = (base * (chars // len(base) + 1))[:chars]
    middle = len(text) // 2
    return text[:middle] + " NEEDLE-42 " + text[middle:]


@mcp.tool
def small() -> dict:
    """A small result."""
    return {"ok": True, "value": 7}


@mcp.tool
def hit() -> dict:
    """Increment a counter held by this server process."""
    _hits["n"] += 1
    return {"pid": os.getpid(), "hits": _hits["n"]}


@mcp.tool
def env_value(name: str) -> dict:
    """Return an environment variable of the server process."""
    return {"name": name, "value": os.environ.get(name)}


@mcp.tool
def seen_headers() -> dict:
    """Return the HTTP headers this server received (HTTP transport only)."""
    return {"headers": get_http_headers(include_all=True)}


@mcp.tool
def picture() -> ToolResult:
    """A large result that includes an image."""
    return ToolResult(content=[
        TextContent(type="text", text="x" * 20_000),
        ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png"),
    ])


@mcp.tool
def fails() -> str:
    """Always fails."""
    raise ValueError("upstream exploded " + "y" * 20_000)


@mcp.tool
def text_with_summary() -> ToolResult:
    """Large text plus a small, different structured summary."""
    return ToolResult(
        content=[TextContent(type="text", text="# Report\n" + "line\n" * 9000 + "NEEDLE-7\n")],
        structured_content={"pages": 1, "ok": True},
    )


@mcp.tool
def log_rows(n: int = 400) -> list[dict]:
    """Rows that have an 'error' field (ordinary data)."""
    return [{"id": i, "error": "timeout" if i % 2 else None, "msg": "m" * 40} for i in range(n)]


if os.environ.get("UPSTREAM_CLASH"):
    @mcp.tool(name="result_list")
    def upstream_result_list() -> dict:
        """The upstream server's own tool that happens to be called result_list."""
        return {"from": "upstream"}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("http", "sse"):
        mcp.run(transport=sys.argv[1], host="127.0.0.1", port=int(sys.argv[2]), show_banner=False)
    else:
        mcp.run(show_banner=False)
