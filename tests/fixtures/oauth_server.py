"""OAuth-protected upstream MCP server for tests: ``oauth_server.py PORT``."""

import sys

from fastmcp import FastMCP
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from mcp.server.auth.settings import ClientRegistrationOptions

port = int(sys.argv[1])
auth = InMemoryOAuthProvider(
    base_url=f"http://127.0.0.1:{port}",
    client_registration_options=ClientRegistrationOptions(enabled=True),
)
mcp = FastMCP("oauth-upstream", auth=auth)


@mcp.tool
def list_issues(n: int = 500) -> list[dict]:
    """List issues (requires login)."""
    return [{"id": i, "title": f"issue {i}", "body": "text " * 40} for i in range(n)]


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=port, show_banner=False)
