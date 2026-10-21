"""
MCP integration: offload large tool results from any MCP server.

Requires the ``mcp`` extra: ``pip install "ctxtual[mcp]"``.

- :class:`Offload` — FastMCP middleware for your own server.
- :func:`create_offload_proxy` — wrap a server you don't control
  (also available as the ``ctxtual wrap`` command).
"""

try:
    import fastmcp  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "ctxtual.mcp needs FastMCP. Install the extra: pip install 'ctxtual[mcp]'"
    ) from exc

from ctxtual.mcp.offload import Offload
from ctxtual.mcp.proxy import create_offload_proxy, expand_env, make_transport

__all__ = ["Offload", "create_offload_proxy", "expand_env", "make_transport"]
