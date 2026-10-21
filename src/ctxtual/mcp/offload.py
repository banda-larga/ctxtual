"""
``Offload``: FastMCP middleware that keeps large tool results out of the
model's context.

Results larger than ``threshold`` characters are stored in a ctxtual
workspace; the model receives a short notice with a preview, the inferred
fields and the id, plus a fixed set of ``result_*`` tools to search, filter,
aggregate and page through the stored data.  Smaller results pass through
untouched.

Usage on your own server::

    from fastmcp import FastMCP
    from ctxtual.mcp import Offload

    mcp = FastMCP("my-server")
    mcp.add_middleware(Offload())

To wrap a server you don't control, see :func:`ctxtual.mcp.create_offload_proxy`
or the ``ctxtual wrap`` command.
"""

import contextlib
import fnmatch
import functools
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware
from fastmcp.tools.base import InputRequiredToolResult, Tool, ToolResult
from mcp_types import TextContent, ToolAnnotations

from ctxtual.ctx import Ctx, _infer_item_schema
from ctxtual.mcp.preview import (
    build_preview,
    compact_schema,
    parse_text_payload,
    unwrap_structured,
)
from ctxtual.session import is_visible, session_scope, visible_workspaces
from ctxtual.store.base import BaseStore
from ctxtual.store.memory import MemoryStore
from ctxtual.utils import paginator, pipeline, text_content, text_search

logger = logging.getLogger("ctxtual.mcp")

#: Fraction of a dict result's size a single list field must hold to be
#: treated as "the data" (e.g. ``{"items": [...], "total": 900}``).
_DOMINANT_LIST_SHARE = 0.8

#: Exploration tools exposed to the model, per ctxtual factory tool suffix.
_EXPOSED = ("paginate", "get_item", "search", "pipe", "read_page", "search_in_text")


class Offload(ToolInjectionMiddleware):
    """
    Store large tool results in workspaces and give the model tools to
    explore them.

    Args:
        threshold:   Results whose text (or structured content) is longer
                     than this many characters are offloaded.
        store:       ctxtual store (default: an in-process ``MemoryStore``).
        ttl:         Seconds a stored result is kept (stated in the tool
                     descriptions so the model knows).  ``None`` = forever.
        max_workspaces: Cap for the default ``MemoryStore`` (LRU eviction).
        prefix:      Name prefix of the injected tools (``result_search``…).
        include / exclude: Glob patterns of tool names to offload / never
                     offload.  By default every tool is eligible.
        allow_regex: Allow ``$regex`` in ``result_pipe`` filters (patterns
                     run on this machine; off by default).
        scope:       Returns the isolation key for the current request.
                     Default: the authenticated principal (token subject or
                     client id) when the server uses auth, else ``None``
                     (one shared, single-user space).
        upstream_tool_names: Returns the names of the tools being wrapped
                     (set by :func:`create_offload_proxy`).  If one clashes
                     with an injected tool, the injected tools are renamed
                     to ``ctxtual_<prefix>_*`` instead of hiding it.  On your
                     own server, pick a ``prefix`` no tool of yours uses.
    """

    def __init__(
        self,
        *,
        threshold: int = 8_000,
        store: BaseStore | None = None,
        ttl: float | None = 3600.0,
        max_workspaces: int = 500,
        prefix: str = "result",
        include: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        preview_items: int = 3,
        allow_regex: bool = False,
        max_result_chars: int = 25_000,
        scope: Callable[[], str | None] | None = None,
        upstream_tool_names: Callable[[], Awaitable[Sequence[str]]] | None = None,
    ) -> None:
        self.threshold = threshold
        self.ttl = ttl
        self.prefix = prefix
        self.include = list(include) if include is not None else None
        self.exclude = list(exclude)
        self.preview_items = preview_items
        self._scope_fn = scope or _principal_scope
        self.ctx = Ctx(
            store=store or MemoryStore(max_workspaces=max_workspaces),
            default_ttl=ttl,
            max_result_chars=max_result_chars,
        )
        self._allow_regex = allow_regex
        self._upstream_tool_names = upstream_tool_names
        self._clash_checked = upstream_tool_names is None
        super().__init__(tools=self._build_tools(prefix))

    def _build_tools(self, prefix: str) -> list[Tool]:
        self.prefix = prefix
        self._toolsets = [
            paginator(self.ctx, prefix, max_page_size=50, enforce_type=False),
            text_search(self.ctx, prefix, max_results_cap=50, enforce_type=False),
            pipeline(self.ctx, prefix, max_items=50, allow_regex=self._allow_regex,
                     enforce_type=False),
            text_content(self.ctx, prefix, max_chars_per_page=10_000, enforce_type=False),
        ]
        tools = [self._exploration_tool(ts, f"{prefix}_{suffix}")
                 for suffix in _EXPOSED
                 for ts in self._toolsets
                 if f"{prefix}_{suffix}" in ts.tool_names]
        tools.append(self._list_tool())
        self._own_names = {t.name for t in tools}
        return tools

    async def _avoid_name_clashes(self) -> None:
        """Rename the injected tools if the wrapped server uses their names."""
        self._clash_checked = True
        try:
            upstream = set(await self._upstream_tool_names())  # type: ignore[misc]
        except Exception:  # the listing itself will report the problem
            self._clash_checked = False
            return
        clashes = sorted(self._own_names & upstream)
        if not clashes:
            return
        prefix = f"ctxtual_{self.prefix}"
        tools = self._build_tools(prefix)
        self._tools_to_inject = tools
        self._tools_to_inject_by_name = {t.name: t for t in tools}
        logger.warning(
            "The wrapped server has tools named %s; ctxtual's tools are renamed "
            "to %s_* so they don't hide them (or pass --prefix).",
            ", ".join(clashes), prefix,
        )

    # ── which tools are offloaded ─────────────────────────────────────────

    def applies_to(self, tool_name: str) -> bool:
        """Whether results of *tool_name* may be offloaded."""
        if tool_name in self._own_names:
            return False
        if any(fnmatch.fnmatchcase(tool_name, p) for p in self.exclude):
            return False
        if self.include is not None:
            return any(fnmatch.fnmatchcase(tool_name, p) for p in self.include)
        return True

    # ── middleware hooks ──────────────────────────────────────────────────

    async def on_list_tools(self, context: Any, call_next: Any) -> Any:
        if not self._clash_checked:
            await self._avoid_name_clashes()
        tools = await call_next(context)
        result = []
        for tool in tools:
            if tool.name not in self._own_names and self.applies_to(tool.name) and tool.output_schema:
                # A replaced result is a notice, not the declared output, and
                # clients are told to validate structured results.
                tool = tool.model_copy(update={"output_schema": None})
            result.append(tool)
        return result

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        name = context.message.name
        try:
            result = await call_next(context)
        except ToolError as exc:
            # Tool exceptions reach us as ToolError; trim huge messages too.
            message = str(exc)
            if self.applies_to(name) and len(message) > self.threshold:
                raise ToolError(self._trimmed(message)) from exc
            raise
        if (
            not isinstance(result, ToolResult)
            or isinstance(result, InputRequiredToolResult)
            or not self.applies_to(name)
        ):
            return result
        if result.is_error:
            return self._trim_error(result)
        try:
            return self._maybe_offload(name, dict(context.message.arguments or {}), result)
        except Exception:  # never lose a result because offloading failed
            logger.exception("ctxtual: could not offload result of %s", name)
            return result

    # ── offloading ────────────────────────────────────────────────────────

    def _trimmed(self, text: str) -> str:
        keep = max(self.threshold // 2 - 50, 200)
        return (
            f"{text[:keep]}\n… [{len(text) - 2 * keep:,} characters of the error "
            f"omitted] …\n{text[-keep:]}"
        )

    def _trim_error(self, result: ToolResult) -> ToolResult:
        """
        Keep the beginning and end of an oversized error.  Errors are for the
        model to correct its call, not data to explore, so they are not
        stored; the middle of a 20 KB traceback rarely helps.
        """
        blocks = list(result.content or [])
        if any(getattr(b, "type", None) != "text" for b in blocks):
            return result
        text = "".join(b.text for b in blocks)
        if len(text) <= self.threshold:
            return result
        return ToolResult(
            content=[TextContent(type="text", text=self._trimmed(text))],
            meta=result.meta,
            is_error=True,
        )

    def _maybe_offload(
        self, name: str, arguments: dict[str, Any], result: ToolResult
    ) -> ToolResult:
        content = list(result.content or [])
        if any(getattr(block, "type", None) != "text" for block in content):
            return result  # images, audio, resources: pass through
        text = "".join(block.text for block in content)
        structured = result.structured_content
        size = max(len(text), _json_len(structured) if structured is not None else 0)
        if size <= self.threshold:
            return result

        # Store the part that made the result large: structured content when
        # the text is (about) its serialization, else the text itself.
        if structured is not None and 2 * _json_len(structured) >= len(text):
            payload = unwrap_structured(structured)
        else:
            payload = parse_text_payload(text)
        payload, container = _focus(payload)

        # A new id per call: an earlier result the model still refers to
        # must not change under it (TTL and LRU clean up).
        with self._scope():
            ref = self.ctx.store_workspace(
                payload,
                workspace_type=name,
                workspace_id=None,
                toolsets=self._toolsets,
                producer_fn=name,
                producer_kwargs=arguments,
            )
        notice = self._notice(ref.workspace_id, name, payload, size, container)
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(notice, ensure_ascii=False, default=str))],
            structured_content=notice,
            meta=result.meta,
        )

    def _notice(
        self,
        workspace_id: str,
        tool: str,
        payload: Any,
        size: int,
        container: dict[str, Any] | None,
    ) -> dict[str, Any]:
        p = self.prefix
        notice: dict[str, Any] = {"stored_result": workspace_id, "tool": tool}
        if isinstance(payload, list):
            notice["item_count"] = len(payload)
            fields = compact_schema(_infer_item_schema(payload))
            if fields:
                notice["fields"] = fields
            notice["preview"] = build_preview(payload, max_items=self.preview_items)
            how = (
                f"Too large to show in full ({size:,} chars, {len(payload):,} items); "
                f"stored as workspace '{workspace_id}'. Explore it instead of calling "
                f"{tool} again: {p}_search(workspace_id, query) finds items by text; "
                f"{p}_pipe(workspace_id, steps) filters, sorts and aggregates in one "
                f'call, e.g. steps=[{{"filter": {{"field": {{"$gte": 1}}}}}}, '
                f'{{"sort": {{"field": "field", "order": "desc"}}}}, {{"limit": 10}}]; '
                f"{p}_paginate(workspace_id, page) and {p}_get_item(workspace_id, index) "
                f"read items directly."
            )
        else:
            text = payload if isinstance(payload, str) else str(payload)
            notice["chars"] = len(text)
            notice["preview"] = build_preview(text)
            how = (
                f"Too large to show in full ({len(text):,} chars); stored as workspace "
                f"'{workspace_id}'. Explore it instead of calling {tool} again: "
                f"{p}_search_in_text(workspace_id, query) finds passages; "
                f"{p}_read_page(workspace_id, page) reads it page by page."
            )
        if container:
            notice["other_fields"] = container
        if self.ttl is not None:
            how += f" Kept for {_duration(self.ttl)}."
        notice["how_to_explore"] = how
        return notice

    # ── injected tools ────────────────────────────────────────────────────

    def _exploration_tool(self, ts: Any, name: str) -> Tool:
        target = ts._tools[name]
        ctx = self.ctx

        @functools.wraps(target)
        async def call(**kwargs: Any) -> ToolResult:
            with self._scope():
                return _to_tool_result(ctx.dispatch_tool_call(name, kwargs))

        first, _, rest = inspect.getdoc(target).partition("\n")
        description = (
            f"{first} Works on results stored by ctxtual: pass the workspace_id "
            f"from a 'stored_result' notice.{_retention(self.ttl)}\n{rest}"
        )
        return Tool.from_function(
            call,
            name=name,
            description=description,
            output_schema=None,
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        )

    def _list_tool(self) -> Tool:
        ctx = self.ctx

        async def list_results() -> ToolResult:
            """List the large tool results stored in this conversation, newest first.

            Use it to find a workspace_id again, e.g. after the conversation
            was summarized.
            """
            with self._scope():
                rows = []
                for ws_id in visible_workspaces(ctx.store, limit=50):
                    meta = ctx.store.get_meta(ws_id)
                    if meta is None or not is_visible(meta):
                        continue
                    rows.append({
                        "workspace_id": ws_id,
                        "tool": meta.producer_fn,
                        "arguments": meta.producer_kwargs,
                        "items": meta.item_count if meta.data_shape == "list" else None,
                        "shape": meta.data_shape,
                    })
            return _to_tool_result({"stored_results": rows, "count": len(rows)})

        return Tool.from_function(
            list_results,
            name=f"{self.prefix}_list",
            description=inspect.getdoc(list_results) + _retention(self.ttl),
            output_schema=None,
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        )

    @contextlib.contextmanager
    def _scope(self) -> Iterator[None]:
        key = self._scope_fn()
        if key is None:
            yield
        else:
            with session_scope(key):
                yield


# ── helpers ───────────────────────────────────────────────────────────────


def _principal_scope() -> str | None:
    """Isolation key: the authenticated principal, if the server has auth."""
    try:
        token = get_access_token()
    except Exception:  # no request context (e.g. direct calls in tests)
        return None
    if token is None:
        return None
    principal = token.subject or token.client_id
    return f"principal:{principal}" if principal else None


def _focus(payload: Any) -> tuple[Any, dict[str, Any] | None]:
    """
    Pick the part of *payload* worth exploring.

    Returns ``(data, other_fields)``: a list stays as is; an object whose
    size is dominated by one list field (``{"items": [...], "total": 900}``)
    yields that list plus a preview of the remaining fields; any other value
    becomes readable text.
    """
    if isinstance(payload, (list, str)):
        return payload, None
    if isinstance(payload, dict) and len(payload) == 1:
        # {"content": "<text>"} / {"rows": [...]}: the value is the data.
        (value,) = payload.values()
        if isinstance(value, (list, str)):
            return value, None
    if isinstance(payload, dict) and payload:
        total = _json_len(payload)
        for key, value in payload.items():
            if isinstance(value, list) and _json_len(value) >= _DOMINANT_LIST_SHARE * total:
                rest = {k: v for k, v in payload.items() if k != key}
                return value, {"list_field": key, **build_preview(rest)}
    return json.dumps(payload, ensure_ascii=False, indent=1, default=str), None


def _to_tool_result(value: Any) -> ToolResult:
    # ctxtual's own errors carry both keys; stored data may have "error".
    is_error = isinstance(value, dict) and "error" in value and "suggested_action" in value
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=value if isinstance(value, dict) else None,
        is_error=is_error,
    )


def _json_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _duration(seconds: float) -> str:
    if seconds >= 86_400 and seconds % 86_400 == 0:
        n = int(seconds // 86_400)
        return f"{n} day{'s' if n > 1 else ''}"
    if seconds >= 3600 and seconds % 3600 == 0:
        n = int(seconds // 3600)
        return f"{n} hour{'s' if n > 1 else ''}"
    n = max(1, int(seconds // 60))
    return f"{n} minute{'s' if n > 1 else ''}"


def _retention(ttl: float | None) -> str:
    if ttl is None:
        return ""
    return f" Stored results are kept for {_duration(ttl)}."
