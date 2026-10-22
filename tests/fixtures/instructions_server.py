"""Upstream MCP server with instructions (stdio), for metadata tests."""

from fastmcp import FastMCP

mcp = FastMCP(
    "with-instructions", instructions="UPSTREAM-GUIDANCE: always pass org=acme."
)


@mcp.tool
def ping() -> str:
    """Reply pong."""
    return "pong"


if __name__ == "__main__":
    mcp.run(show_banner=False)
