"""
Wrap an existing MCP server (local command or remote URL) so its large
results are offloaded.  Used by ``ctxtual wrap``.
"""

import contextlib
import hashlib
import os
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import fastmcp.client.transports.http as _fastmcp_http
from fastmcp.client.transports import (
    SSETransport,
    StdioTransport,
    StreamableHttpTransport,
)
from fastmcp.server.providers.proxy import FastMCPProxy, ProxyClient

from ctxtual.mcp.offload import Offload

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Largest single upstream message (one tool result) accepted over HTTP.
#: The MCP SDK defaults to 1 MiB, which rejects exactly the results this
#: package exists to handle.
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024

_max_sse_event: ContextVar[int | None] = ContextVar("ctxtual_max_sse_event", default=None)


def _install_sse_limit_hook() -> None:
    """
    Let :class:`LargeResultHttpTransport` raise the SDK's per-event limit.

    FastMCP's transport calls the SDK's ``streamable_http_client`` without a
    ``max_sse_event_size``.  We wrap that function once; the wrapper changes
    nothing unless a ctxtual transport set a limit for the current context.
    """
    original = _fastmcp_http.streamable_http_client
    if getattr(original, "__ctxtual_hook__", False):
        return

    def streamable_http_client(url: Any, **kwargs: Any) -> Any:
        limit = _max_sse_event.get()
        if limit is not None and "max_sse_event_size" not in kwargs:
            kwargs["max_sse_event_size"] = limit
        return original(url, **kwargs)

    streamable_http_client.__ctxtual_hook__ = True  # type: ignore[attr-defined]
    _fastmcp_http.streamable_http_client = streamable_http_client  # type: ignore[assignment]


class LargeResultHttpTransport(StreamableHttpTransport):
    """Streamable HTTP transport that accepts results up to ``max_message_bytes``."""

    def __init__(
        self, *args: Any, max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_message_bytes = max_message_bytes
        _install_sse_limit_hook()

    @contextlib.asynccontextmanager
    async def connect_session(self, **kwargs: Any) -> AsyncIterator[Any]:
        _max_sse_event.set(self.max_message_bytes)
        async with super().connect_session(**kwargs) as session:
            yield session


def expand_env(value: str, env: Mapping[str, str] | None = None) -> str:
    """Replace ``${VAR}`` with the variable's value (unset → error)."""
    env = os.environ if env is None else env

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise ValueError(f"environment variable {name} is not set")
        return env[name]

    return _ENV_REF.sub(sub, value)


def make_transport(
    target: str | Sequence[str],
    *,
    headers: Mapping[str, str] | None = None,
    oauth: bool = False,
    oauth_scopes: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    sse: bool | None = None,
) -> Any:
    """
    Transport for *target*: an ``http(s)://`` URL (streamable HTTP, or the
    legacy HTTP+SSE transport when *sse* is true — by default when the path
    ends in ``/sse``) or a command line (stdio, as a list).
    """
    if isinstance(target, str):
        if not target.startswith(("http://", "https://")):
            raise ValueError(f"not an http(s) URL: {target!r}")
        auth: Any = None
        if oauth:
            auth = _oauth(target, oauth_scopes)
        hdrs = dict(headers or {})
        if sse is None:
            sse = target.rstrip("/").endswith("/sse")
        if sse:
            # Deprecated transport; the SDK caps its events at 1 MiB.
            return SSETransport(target, headers=hdrs, auth=auth)
        return LargeResultHttpTransport(target, headers=hdrs, auth=auth)

    command = list(target)
    if not command:
        raise ValueError("empty command")
    # The MCP SDK only passes a small allowlist of variables (PATH, HOME, …)
    # to stdio servers.  Forward everything: the client that launched us
    # (Claude Desktop, Cursor, …) put the server's secrets in our env.
    full_env = {**os.environ, **(env or {})}
    return StdioTransport(command[0], command[1:], env=full_env, cwd=cwd, keep_alive=True)


def create_offload_proxy(
    target: str | Sequence[str],
    *,
    headers: Mapping[str, str] | None = None,
    oauth: bool = False,
    oauth_scopes: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    sse: bool | None = None,
    name: str | None = None,
    shared_session: bool = True,
    auth: Any = None,
    offload: Offload | None = None,
    **offload_kwargs: Any,
) -> FastMCPProxy:
    """
    A FastMCP server that proxies *target* and offloads its large results.

    Args:
        target:         ``https://…/mcp`` URL or a command as a list
                        (``["npx", "-y", "@scope/server"]``).
        headers:        Extra HTTP headers for a URL target (e.g. auth).
        oauth:          Log in to the URL target with OAuth (browser flow) when
                        it answers 401; tokens are cached under
                        ``~/.ctxtual/oauth``.
        sse:            The URL uses the legacy HTTP+SSE transport (default:
                        guessed from a ``/sse`` path).
        env / cwd:      Extra environment / working directory for a command.
        shared_session: Reuse one upstream session for all requests (right
                        for a local, single-user wrapper and required for
                        stateful servers such as browser automation).  Set
                        ``False`` when serving several users over HTTP.
        auth:           FastMCP auth provider for the proxy's own HTTP endpoint
                        (hosted deployments).  Stored results are then
                        isolated per authenticated principal.
        offload:        A configured :class:`Offload`; otherwise one is built
                        from ``offload_kwargs`` (threshold, ttl, store, …).

    The proxy never forwards the caller's HTTP headers (including
    ``Authorization``) to the upstream server: upstream credentials come only
    from ``headers``/``oauth``/``env``.
    """
    transport = make_transport(
        target, headers=headers, oauth=oauth, oauth_scopes=oauth_scopes,
        env=env, cwd=cwd, sse=sse,
    )
    base = ProxyClient(transport, forward_incoming_headers=False)
    if shared_session:
        def factory() -> ProxyClient:
            return base
    else:
        def factory() -> ProxyClient:
            return base.new()

    async def upstream_tool_names() -> list[str]:
        async with factory() as client:
            return [tool.name for tool in await client.list_tools()]

    proxy = FastMCPProxy(
        client_factory=factory,
        name=name or "ctxtual",
        auth=auth,
        # Clients see the wrapped server's name and version unless a name
        # is given, and (because the proxy sets none) its instructions.
        identity="proxy" if name else "upstream",
        # Surface upstream failures (e.g. a wrong URL or transport) instead
        # of serving an empty tool list.
        provider_error_strategy="raise",
    )
    if offload is None:
        offload = Offload(upstream_tool_names=upstream_tool_names, **offload_kwargs)
    elif offload._upstream_tool_names is None:
        offload._upstream_tool_names = upstream_tool_names
        offload._clash_checked = False
    # Outermost, so it appends its explanation to the upstream instructions
    # the proxy's metadata middleware has already forwarded.
    proxy.middleware.insert(0, offload)
    return proxy


def _oauth(url: str, scopes: Sequence[str] | None) -> Any:
    from fastmcp.client.auth import OAuth
    from key_value.aio.stores.filetree import (
        FileTreeStore,
        FileTreeV1CollectionSanitizationStrategy,
        FileTreeV1KeySanitizationStrategy,
    )

    directory = oauth_cache_dir() / hashlib.sha256(url.encode()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return OAuth(
        mcp_url=url,
        scopes=list(scopes) if scopes else None,
        client_name="ctxtual",
        # Keys are URLs; sanitize them into safe file names.
        token_storage=FileTreeStore(
            data_directory=directory,
            key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(directory),
            collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(directory),
        ),
    )


def oauth_cache_dir() -> Path:
    base = Path(os.environ.get("CTXTUAL_HOME", Path.home() / ".ctxtual"))
    path = base / "oauth"
    path.mkdir(parents=True, exist_ok=True)
    # Tokens are credentials: keep the whole tree private to the user.
    os.chmod(base, 0o700)
    os.chmod(path, 0o700)
    return path
