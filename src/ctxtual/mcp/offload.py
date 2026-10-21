"""
``Offload``: FastMCP middleware that keeps large tool results out of the
model's context.

Results longer than ``threshold`` characters are stored; the model receives
a short notice (``result_id``, size, fields, preview, how to explore) and
four tools to work with stored results:

- ``result_search`` — matching items of a list, or passages of a text
- ``result_read``   — a page, one item, or one long field page by page
- ``result_query``  — filter, sort, group and count a list in one call
- ``result_list``   — what is stored

Smaller results pass through untouched.  The general explanation is appended
to the server's instructions, so tool descriptions stay short.

Usage on your own server::

    from fastmcp import FastMCP
    from ctxtual.mcp import Offload

    mcp = FastMCP("my-server")
    mcp.add_middleware(Offload())

To wrap a server you don't control, see :func:`ctxtual.mcp.create_offload_proxy`
or the ``ctxtual wrap`` command.
"""

import asyncio
import contextlib
import fnmatch
import inspect
import itertools
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware
from fastmcp.tools.base import InputRequiredToolResult, Tool, ToolResult
from mcp_types import TextContent, ToolAnnotations
from pydantic import WithJsonSchema

from ctxtual.ctx import Ctx, _infer_item_schema
from ctxtual.exceptions import CtxtualError, WorkspaceConflictError
from ctxtual.mcp.preview import (
    MISSING,
    build_preview,
    compact_schema,
    cut_strings,
    parse_text_payload,
    resolve_path,
    shrink,
    unwrap_structured,
)
from ctxtual.pipeline import PipelineEngine, PipelineError
from ctxtual.session import current_session, is_visible, session_scope
from ctxtual.store.base import BaseStore, _score_item, query_terms
from ctxtual.store.memory import MemoryStore

logger = logging.getLogger("ctxtual.mcp")

#: Fraction of an object's size a single list field must hold to be treated
#: as "the data" (e.g. ``{"items": [...], "total": 900}``).
_DOMINANT_SHARE = 0.8
#: Characters per page when reading text or a long field.
PAGE_CHARS = 10_000
#: Characters a string keeps in pages of items and query rows.
LIST_STRING_CHARS = 300
#: Characters a string keeps in search hits (several hits per response).
HIT_STRING_CHARS = 160
#: Characters a string keeps when one item is read.
ITEM_STRING_CHARS = 4_000
#: Most items, rows or passages one call returns.
MAX_ITEMS = 50
#: The tools Offload can inject (``tools=`` picks a subset).
TOOLS = ("search", "read", "query", "list")
_SAVE_AS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")


class _FixableError(Exception):
    """A mistake the model can fix; becomes a tool error with a hint."""

    def __init__(self, error: str, action: str, **extra: Any) -> None:
        super().__init__(error)
        self.payload = {"error": error, "suggested_action": action, **extra}


class Offload(ToolInjectionMiddleware):
    """
    Store large tool results and give the model tools to explore them.

    Args:
        threshold:   Results whose text (or structured content) is longer
                     than this many characters are stored.
        thresholds:  Per-tool overrides, glob pattern → characters, e.g.
                     ``{"get_file": 40_000}``.
        include / exclude: Glob patterns of tools to store results of /
                     never store results of (default: all tools).
        ttl:         A stored result is dropped after this many seconds
                     without use (``None``: kept until evicted).
        max_results: Most stored results kept per user (per session scope);
                     the least recently used go first.
        max_total_chars: Upper bound on the size of one user's stored results.
                     The result just stored is always kept.
        store:       A ctxtual store (default: an in-process MemoryStore).
        prefix:      Name prefix of the injected tools.
        tools:       Which tools to inject: any of ``"search"``, ``"read"``,
                     ``"query"``, ``"list"`` (default: all four).
        preview_items: Items shown in the notice for list results.
        allow_regex: Allow ``$regex`` in ``result_query`` filters (patterns
                     are model-written and run on this machine).
        max_output_chars: Largest response an exploration tool returns.
        instructions: Append the explanation of stored results to the
                     server's instructions (on by default).
        scope:       Returns the isolation key for the current request.
                     Default: the authenticated principal (token subject or
                     client id) when the server uses auth, else ``None``
                     (one shared, single-user space).
        upstream_tool_names: Returns the names of the wrapped server's tools
                     (set by :func:`create_offload_proxy`).  On a clash the
                     injected tools are renamed ``ctxtual_<prefix>_*``
                     instead of hiding the server's tool.  On your own
                     server, pick a ``prefix`` none of your tools use.
    """

    def __init__(
        self,
        *,
        threshold: int = 20_000,
        thresholds: Mapping[str, int] | None = None,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
        ttl: float | None = 3600.0,
        max_results: int = 200,
        max_total_chars: int = 100_000_000,
        store: BaseStore | None = None,
        prefix: str = "result",
        tools: Sequence[str] = TOOLS,
        preview_items: int = 3,
        allow_regex: bool = False,
        max_output_chars: int = 25_000,
        instructions: bool = True,
        scope: Callable[[], str | None] | None = None,
        upstream_tool_names: Callable[[], Awaitable[Sequence[str]]] | None = None,
    ) -> None:
        wanted = [str(t).strip().lower() for t in tools]
        unknown = sorted(set(wanted) - set(TOOLS))
        if unknown:
            raise ValueError(f"unknown tools {unknown}; choose from {', '.join(TOOLS)}")
        if not wanted:
            raise ValueError(f"tools must name at least one of {', '.join(TOOLS)}")
        if max_results < 1 or max_total_chars < 1:
            raise ValueError("max_results and max_total_chars must be at least 1")
        if threshold < 1 or any(v < 1 for v in (thresholds or {}).values()):
            raise ValueError("thresholds must be at least 1")
        self.threshold = threshold
        self.thresholds = dict(thresholds or {})
        self.include = list(include) if include is not None else None
        self.exclude = list(exclude)
        self.ttl = ttl
        self.max_results = max_results
        self.max_total_chars = max_total_chars
        self.tools = tuple(t for t in TOOLS if t in wanted)
        self.preview_items = preview_items
        self.max_output_chars = max_output_chars
        self.add_instructions = instructions
        self._scope_fn = scope or _principal_scope
        self._engine = PipelineEngine(allow_regex=allow_regex)
        # Expiry and eviction are handled here (idle timeout, size cap).
        self.ctx = Ctx(store=store or MemoryStore(), default_ttl=None, max_result_chars=None)
        self._upstream_tool_names = upstream_tool_names
        self._clash_checked = upstream_tool_names is None
        self._clash_lock = asyncio.Lock()
        super().__init__(tools=self._build_tools(prefix))

    # ── which tools are offloaded ─────────────────────────────────────────

    def applies_to(self, tool_name: str) -> bool:
        """Whether results of *tool_name* may be stored."""
        if tool_name in self._own_names:
            return False
        if any(fnmatch.fnmatchcase(tool_name, p) for p in self.exclude):
            return False
        if self.include is not None:
            return any(fnmatch.fnmatchcase(tool_name, p) for p in self.include)
        return True

    def threshold_for(self, tool_name: str) -> int:
        for pattern, value in self.thresholds.items():
            if fnmatch.fnmatchcase(tool_name, pattern):
                return value
        return self.threshold

    # ── middleware hooks ──────────────────────────────────────────────────

    async def on_initialize(self, context: Any, call_next: Any) -> Any:
        await self._ensure_names()
        return self._with_instructions(await call_next(context))

    async def on_discover(self, context: Any, call_next: Any) -> Any:
        await self._ensure_names()
        return self._with_instructions(await call_next(context))

    def _with_instructions(self, result: Any) -> Any:
        """Append the stored-results explanation to the server's instructions."""
        if not self.add_instructions or not hasattr(result, "instructions"):
            return result
        ours = self.instructions_text()
        theirs = result.instructions
        merged = f"{theirs.rstrip()}\n\n{ours}" if theirs else ours
        return result.model_copy(update={"instructions": merged})

    def instructions_text(self) -> str:
        p = self.prefix
        named = {
            "search": f"{p}_search finds items or passages",
            "read": f"{p}_read reads pages, single items and long fields",
            "query": f"{p}_query filters, sorts, groups and counts in one call",
            "list": f"{p}_list shows what is stored",
        }
        lines = "; ".join(named[t] for t in self.tools)
        expiry = (
            f" Stored results are dropped after {_duration(self.ttl)} without use."
            if self.ttl is not None
            else ""
        )
        return (
            "Large tool results: when a result is too large to return in full, "
            "the tool returns a stored-result notice instead, with a result_id, "
            f"the size, the fields and a preview. Pass the result_id to: {lines}.{expiry}"
        )

    async def on_list_tools(self, context: Any, call_next: Any) -> Any:
        await self._ensure_names()
        tools = await call_next(context)
        result = []
        for tool in tools:
            if tool.name not in self._own_names and self.applies_to(tool.name) and tool.output_schema:
                # A stored-result notice is not the declared output, and
                # clients are told to validate structured results.
                tool = tool.model_copy(update={"output_schema": None})
            result.append(tool)
        return result

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        await self._ensure_names()
        name = context.message.name
        limit = self.threshold_for(name)
        try:
            result = await call_next(context)
        except ToolError as exc:
            message = str(exc)
            if self.applies_to(name) and len(message) > limit:
                raise ToolError(_trim(message, limit)) from exc
            raise
        if (
            not isinstance(result, ToolResult)
            or isinstance(result, InputRequiredToolResult)
            or not self.applies_to(name)
        ):
            return result
        if result.is_error:
            return self._trim_error(result, limit)
        try:
            return self._maybe_store(name, dict(context.message.arguments or {}), result, limit)
        except Exception:  # never lose a result because storing failed
            logger.exception("ctxtual: could not store result of %s", name)
            return result

    # ── storing ───────────────────────────────────────────────────────────

    def _trim_error(self, result: ToolResult, limit: int) -> ToolResult:
        """Errors are for correcting the call, not data: keep start and end."""
        blocks = list(result.content or [])
        if any(getattr(b, "type", None) != "text" for b in blocks):
            return result
        text = "".join(b.text for b in blocks)
        trimmed = _trim(text, limit)
        if trimmed is text:
            return result
        return ToolResult(
            content=[TextContent(type="text", text=trimmed)],
            meta=result.meta,
            is_error=True,
        )

    def _maybe_store(
        self, name: str, arguments: dict[str, Any], result: ToolResult, limit: int
    ) -> ToolResult:
        # Text and text resources can be stored; images, audio, binary
        # resources and links are kept in the response next to the notice.
        parts: list[tuple[str, str | None]] = []
        kept: list[Any] = []
        for block in result.content or []:
            kind = getattr(block, "type", None)
            resource_text = getattr(getattr(block, "resource", None), "text", None)
            if kind == "text":
                parts.append((block.text, None))
            elif kind == "resource" and isinstance(resource_text, str):
                parts.append((resource_text, str(block.resource.uri)))
            else:
                kept.append(block)
        text = _join_parts(parts)
        structured = result.structured_content
        size = max(len(text), _json_len(structured) if structured is not None else 0)
        if size <= limit:
            return result

        # Store the part that made the result large: structured content when
        # the text is (about) its serialization, else the text itself.
        extra: dict[str, Any] = {}
        if structured is not None and 2 * _json_len(structured) >= len(text):
            payload = unwrap_structured(structured)
            if isinstance(payload, str):  # e.g. a `-> str` tool: still JSON / JSON Lines?
                payload = parse_text_payload(payload)
            if text.strip() and 2 * len(text) < _json_len(structured):
                extra["message"] = _trim(text, 2_000, "the message")  # a summary next to the data
        else:
            payload = _parse_parts(parts)
            filled = [uri for t, uri in parts if t.strip()]
            if len(filled) == 1 and filled[0]:
                extra["uri"] = filled[0]
        payload, other_fields, shape = _focus(payload)
        if shape:
            extra["note"] = shape
        if not isinstance(payload, list):
            payload = _as_text(payload)
        chars = size if isinstance(payload, list) else len(payload)

        with self._scope():
            # A new id per call: a result the model still refers to must not
            # change under it.  Expiry and eviction clean up.
            ref = self.ctx.store_workspace(
                payload,
                workspace_type=name,
                producer_fn=name,
                producer_kwargs=arguments,
                meta={"chars": chars},
            )
            self._evict(keep=ref.workspace_id)
        notice = self._notice(
            ref.workspace_id, name, payload, size, other_fields,
            has_arguments=bool(arguments), extra=extra,
        )
        return ToolResult(
            content=[TextContent(type="text", text=_dumps(notice)), *kept],
            structured_content=notice,
            meta=result.meta,
        )

    def _evict(self, keep: str | None = None) -> None:
        """
        Drop results idle for longer than ``ttl``, then the least recently
        used ones beyond the count and size caps.  Caps apply per session
        scope (per user), so one user's results never push out another's;
        the result *keep* (just stored) is never dropped.
        """
        store = self.ctx.store
        now = time.time()
        groups: dict[Any, list[Any]] = {}
        for ws_id in store.list_workspaces():
            meta = store.get_meta(ws_id)
            if meta is None:
                continue
            if self.ttl is not None and now - meta.last_accessed_at > self.ttl and ws_id != keep:
                store.drop_workspace(ws_id)
            else:
                groups.setdefault(meta.session_id, []).append(meta)
        for metas in groups.values():
            metas.sort(key=lambda m: (m.workspace_id == keep, m.last_accessed_at))
            total = sum(int(m.extra.get("chars", 0)) for m in metas)
            while len(metas) > 1 and (len(metas) > self.max_results or total > self.max_total_chars):
                oldest = metas.pop(0)
                total -= int(oldest.extra.get("chars", 0))
                store.drop_workspace(oldest.workspace_id)

    def _notice(
        self,
        result_id: str,
        tool: str,
        payload: Any,
        size: int,
        other_fields: dict[str, Any] | None,
        *,
        saved: bool = False,
        has_arguments: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        p = self.prefix
        notice: dict[str, Any] = {"result_id": result_id, "tool": tool, **(extra or {})}
        steps = []
        if isinstance(payload, list):
            notice["items"] = len(payload)
            fields = compact_schema(_infer_item_schema(payload))
            if fields:
                notice["fields"] = fields
            notice["preview"] = build_preview(payload, max_items=self.preview_items)
            lead = (
                f"Stored {len(payload):,} items from {tool}"
                if saved
                else f"{tool} returned {len(payload):,} items ({size:,} chars), "
                "too many to show in full; they are stored"
            )
            if "search" in self.tools:
                steps.append(f"{p}_search(result_id, query) to find items by text")
            if "query" in self.tools and any(isinstance(i, dict) for i in payload[:200]):
                steps.append(
                    f"{p}_query(result_id, steps) to filter, sort and count, e.g. "
                    f"steps={_example_steps(payload, fields or {})}"
                )
            if "read" in self.tools:
                steps.append(f"{p}_read(result_id, page) or {p}_read(result_id, index=N) to read items")
        else:
            text = payload
            pages = max(1, math.ceil(len(text) / PAGE_CHARS))
            notice["chars"] = len(text)
            outline = _outline(text)
            if outline:
                # Headings say more about a long document than its first lines.
                notice["outline"] = outline
                notice["preview"] = text[:300]
            else:
                notice["preview"] = build_preview(text)
            lead = (
                f"Stored a {len(text):,}-char text from {tool}"
                if saved
                else f"{tool} returned {len(text):,} chars, too long to show in full; "
                "the text is stored"
            )
            if "search" in self.tools:
                steps.append(f"{p}_search(result_id, query) to find passages")
            if "read" in self.tools:
                steps.append(f"{p}_read(result_id, page) to read it ({pages:,} pages of {PAGE_CHARS:,} chars)")
        if other_fields:
            notice["other_fields"] = other_fields
        how = f"{lead}. " + ("Use " + "; ".join(steps) + "." if steps else "")
        if has_arguments:
            how += f" Calling {tool} again with narrower arguments also works."
        notice["how_to_explore"] = how.strip()
        return notice

    # ── the injected tools ────────────────────────────────────────────────

    def _build_tools(self, prefix: str) -> list[Tool]:
        self.prefix = prefix
        makers = {
            "search": self._search_tool,
            "read": self._read_tool,
            "query": self._query_tool,
            "list": self._list_tool,
        }
        tools = [makers[name]() for name in self.tools]
        self._own_names = {t.name for t in tools}
        return tools

    async def _ensure_names(self) -> None:
        """Before anything is listed, described or called: avoid name clashes."""
        if self._clash_checked:
            return
        async with self._clash_lock:
            if not self._clash_checked:
                await self._avoid_name_clashes()

    async def _avoid_name_clashes(self) -> None:
        """Rename the injected tools if the wrapped server uses their names."""
        try:
            upstream = set(await self._upstream_tool_names())  # type: ignore[misc]
        except Exception as exc:  # the listing itself will report the problem
            logger.debug("ctxtual: could not list the wrapped server's tools: %r", exc)
            return
        self._clash_checked = True
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

    def _tool(self, fn: Callable[..., Any], name: str, description: str, size_hint: str) -> Tool:
        async def call(**kwargs: Any) -> ToolResult:
            try:
                with self._scope():
                    value = fn(**kwargs)
                size = _json_len(value)
                if size > self.max_output_chars:
                    raise _FixableError(
                        f"The response would be {size:,} chars, more than "
                        f"{self.max_output_chars:,}.", size_hint,
                    )
                return _result(value)
            except _FixableError as problem:
                return _result(problem.payload, is_error=True)
            except CtxtualError as exc:
                return _result(exc.to_llm_dict(), is_error=True)

        call.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        call.__annotations__ = dict(fn.__annotations__)
        return Tool.from_function(
            call,
            name=name,
            description=description,
            output_schema=None,
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        )

    def _search_tool(self) -> Tool:
        p = self.prefix
        open_hint = (f" (open one with {p}_read(result_id, index=…))"
                     if "read" in self.tools else "")

        def search(
            result_id: Annotated[Any, _param(_STR, "result_id from a stored-result notice.")],
            query: Annotated[Any, _param(_STR, "Words to look for.")],
            max_results: Annotated[Any, _param(_INT, f"Most hits to return (up to {MAX_ITEMS}).", 10)] = 10,
            fields: Annotated[Any, _param(_FIELDS, "Only return these fields of each item "
                                          "(dot paths allowed).", None)] = None,
        ) -> Any:
            _, payload = self._load(result_id)
            limit = _clamp(_int_arg("max_results", max_results), 1, MAX_ITEMS)
            fields = _fields_arg(fields)
            query = _str_arg("query", query)
            if not query.strip():
                raise _FixableError("query is empty.", "Pass the words to look for.")
            if not isinstance(payload, list):
                if fields:
                    raise _FixableError("This stored result is a text; fields= applies to lists.",
                                        "Search it without fields.")
                return {"result_id": result_id, "max_results": limit,
                        **_passages(payload, query, limit)}
            terms = query_terms(query.lower())
            scored = []
            for index, item in enumerate(payload):
                score = _score_item(item, terms, None, False)
                if score > 0:
                    scored.append((score, index, item))
            scored.sort(key=lambda t: (-t[0], t[1]))
            all_terms = sum(
                1 for _, _, item in scored
                if all(t in json.dumps(item, ensure_ascii=False, default=str).lower() for t in terms)
            )
            hits = []
            for _, i, item in scored[:limit]:
                hit = {"index": i, "item": self._shape(i, item, fields, HIT_STRING_CHARS)}
                # Where the words are, when the cut item doesn't show it.
                match = _match_passage(item, query_terms(query), HIT_STRING_CHARS)
                if match is not None:
                    hit["match"] = match
                hits.append(hit)
            return {"result_id": result_id, "matching_any_word": len(scored),
                    "matching_all_words": all_terms, "returned": len(hits),
                    "max_results": limit, "hits": hits}

        return self._tool(search, f"{p}_search", (
            f"Search a stored result.{_STORED} For a list: the best-matching items, best "
            f"first, each with its index{open_hint} and, when the words are deep inside "
            "a long value, the passage that contains them. For a text: the passages "
            "that contain the query (the exact phrase if it occurs, else the most query "
            "words), each with its page."
        ), size_hint="Ask for less: a smaller max_results, or fields=[...] with only "
                     "the fields you need.")

    def _read_tool(self) -> Tool:
        p = self.prefix

        def read(
            result_id: Annotated[Any, _param(_STR, "result_id from a stored-result notice.")],
            page: Annotated[Any, _param(_INT, "Zero-based page.", 0)] = 0,
            page_size: Annotated[Any, _param(_INT, f"Items per page, for lists (up to {MAX_ITEMS}).", 10)] = 10,
            index: Annotated[Any, _param(_INT, "Read this one item of a list.", None)] = None,
            field: Annotated[Any, _param(
                _STR, "With index: read this value of the item in full, page by page (dot "
                "paths allowed, e.g. 'author.bio' or 'comments.0.body').", None)] = None,
            fields: Annotated[Any, _param(_FIELDS, "Only return these fields of each item "
                                          "(dot paths allowed).", None)] = None,
        ) -> Any:
            _, payload = self._load(result_id)
            page = _int_arg("page", page)
            page_size = _int_arg("page_size", page_size)
            index = None if index is None else _int_arg("index", index)
            field = None if field is None else _str_arg("field", field)
            fields = _fields_arg(fields)
            if page < 0:
                raise _FixableError(f"page {page} is negative.", "Pages start at 0.")
            if not isinstance(payload, list):
                if index is not None or field is not None or fields:
                    raise _FixableError(
                        "This stored result is a text; index, field and fields apply to lists.",
                        f"Read it by page, or find passages with {p}_search.")
                return {"result_id": result_id, **_text_page(payload, page)}
            if field is not None and index is None:
                raise _FixableError("field needs an index.", "Pass index=N with field=….")
            if index is None:
                size = _clamp(page_size, 1, MAX_ITEMS)
                start = page * size
                pages = max(1, math.ceil(len(payload) / size))
                if start >= len(payload) and payload:
                    raise _FixableError(f"Page {page} is past the end ({pages} pages).",
                                        f"Use a page from 0 to {pages - 1}.")
                items = [
                    {"index": i, "item": self._shape(i, payload[i], fields, LIST_STRING_CHARS)}
                    for i in range(start, min(start + size, len(payload)))
                ]
                return {"result_id": result_id, "page": page, "page_size": size, "pages": pages,
                        "total": len(payload), "has_more": start + size < len(payload),
                        "items": items}
            if not 0 <= index < len(payload):
                raise _FixableError(f"Index {index} is out of range.",
                                    f"Use an index from 0 to {len(payload) - 1}.")
            item = payload[index]
            if field is None:
                if isinstance(item, str) and len(item) > ITEM_STRING_CHARS:
                    return {"result_id": result_id, "index": index, **_text_page(item, page)}
                # Cut long values shorter until the item fits the response.
                limit, many = ITEM_STRING_CHARS, MAX_ITEMS
                while True:
                    response = {"result_id": result_id, "index": index,
                                "item": self._shape(index, item, fields, limit, many)}
                    if limit <= 200 or _json_len(response) <= self.max_output_chars:
                        return response
                    limit, many = limit // 2, max(5, many // 2)
            value = resolve_path(item, field)
            if value is MISSING:
                available = [str(k) for k in item][:50] if isinstance(item, dict) else []
                raise _FixableError(f"Item {index} has no field {field!r}.",
                                    "Use one of the item's fields.", fields=available)
            encoded = value if isinstance(value, str) else _dumps(value)
            if len(encoded) <= PAGE_CHARS:
                return {"result_id": result_id, "index": index, "field": field, "value": value}
            return {"result_id": result_id, "index": index, "field": field, **_text_page(encoded, page)}

        return self._tool(read, f"{p}_read", (
            f"Read a stored result.{_STORED} A text: page by page ({PAGE_CHARS:,} chars "
            "each). A list: a page of items, or one item with index=N; long values in an "
            "item are cut, read one in full with index=N and field=… (paged the same "
            "way). fields=[...] keeps only some fields.\n"
            "Arguments besides result_id, for example: none (the first page); page=3; "
            "index=7; index=7, field='body', page=1; page=2, page_size=20, "
            "fields=['title', 'state']."
        ), size_hint="Ask for less: a smaller page_size, or fields=[...] with only the "
                     "fields you need.")

    def _query_tool(self) -> Tool:
        p = self.prefix
        regex = " $regex" if self._engine.allow_regex else ""

        def query(
            result_id: Annotated[Any, _param(_STR, "result_id of a stored list.")],
            steps: Annotated[Any, _param(
                {"type": "array", "items": {"type": "object"}},
                "Operations applied in order; each is {operation: parameters}.")],
            save_as: Annotated[Any, _param(
                _STR, "Store the full output as a new result named saved_<name> instead of "
                "returning it; saving again under the same name replaces it.", None)] = None,
        ) -> Any:
            meta, payload = self._load(result_id)
            if not isinstance(payload, list):
                raise _FixableError("This stored result is a text, not a list.",
                                    f"Use {p}_search or {p}_read on it.")
            steps = _steps_arg(steps)
            save_as = None if save_as is None else _str_arg("save_as", save_as)
            try:
                output = self._engine.execute(payload, steps)
            except PipelineError as exc:
                extra = {k: v for k, v in exc.context.items() if k != "step"}
                raise _FixableError(str(exc), "Fix that step; the tool description shows the "
                                    "syntax.", **extra) from None
            if save_as is not None:
                return self._save(meta.workspace_type, result_id, output, save_as)
            if not isinstance(output, list):
                return {"result_id": result_id, "result": output}
            return self._rows(result_id, payload, steps, output)

        return self._tool(query, f"{p}_query", (
            f"Filter, sort, group or count a stored list in one call.{_STORED} Only the output "
            f"comes back (at most {MAX_ITEMS} rows, each with its index when it is an "
            "item of the list). Dot paths ('author.name') address nested fields.\n"
            "Steps:\n"
            '  {"filter": {"state": "open", "score": {"$gte": 3}}}  operators: $gt $gte $lt '
            f"$lte $ne $in $nin $contains $startswith $exists{regex}; $or/$and/$not\n"
            '  {"search": {"query": "words"}}   {"sort": {"field": "score", "order": "desc"}}\n'
            '  {"select": ["title", "score"]}   {"exclude": ["body"]}   {"limit": 10}   '
            '{"skip": 10}   {"slice": [10, 20]}   {"unique": "author"}   {"flatten": "tags"}   '
            '{"sample": 5}\n'
            '  {"group_by": {"field": "state", "metrics": {"n": "count", "avg": "mean:score"}}}'
            "  metrics: count sum: mean: min: max: median: stddev: values:\n"
            '  {"count": true}'
        ), size_hint="Ask for less: add a select step with fewer fields, or a limit step.")

    def _rows(
        self, result_id: str, payload: list[Any], steps: list[dict[str, Any]], output: list[Any]
    ) -> dict[str, Any]:
        """Query output: rows with their source index when known, within budget."""
        indices = _source_indices(self._engine, payload, steps, output)
        rows: list[Any] = []
        for n, row in enumerate(output[:MAX_ITEMS]):
            index = indices[n] if indices else None
            if index is None:
                rows.append(cut_strings(row, LIST_STRING_CHARS))
            else:
                rows.append({"index": index, "row": self._shape(index, row, None, LIST_STRING_CHARS)})
        response: dict[str, Any] = {"result_id": result_id, "total": len(output),
                                    "returned": len(rows), "rows": rows}
        # Fit the response into the output budget by showing fewer rows.
        while len(rows) > 1 and _json_len(response) > self.max_output_chars:
            del rows[max(1, len(rows) * 3 // 4):]
            response["returned"] = len(rows)
        if len(rows) < len(output):
            response["note"] = (
                f"{len(rows)} of {len(output)} rows shown. Add a select step with fewer "
                "fields, a limit or filter step, or save_as to keep them all."
            )
        return response

    def _list_tool(self) -> Tool:
        def list_results() -> Any:
            rows = []
            for meta in self._visible_metas():
                row: dict[str, Any] = {
                    "result_id": meta.workspace_id,
                    "tool": meta.producer_fn,
                    "arguments": shrink(meta.producer_kwargs, depth=1),
                }
                if meta.data_shape == "list":
                    row["items"] = meta.item_count
                else:
                    row["chars"] = meta.extra.get("chars")
                rows.append(row)
            response: dict[str, Any] = {"stored_results": rows}
            while len(rows) > 1 and _json_len(response) > self.max_output_chars:
                rows.pop()
                response["note"] = "Only the most recent stored results are listed."
            return response

        return self._tool(list_results, f"{self.prefix}_list", (
            "List the stored results of this conversation, newest first: large tool "
            "results kept out of the conversation, with the tool and arguments that "
            "produced them. Use it to find a result_id again, e.g. after the "
            "conversation was summarized."
        ), size_hint="Use a result_id from a recent notice.")

    # ── helpers ───────────────────────────────────────────────────────────

    def _idle(self, meta: Any) -> bool:
        return self.ttl is not None and time.time() - meta.last_accessed_at > self.ttl

    def _visible_metas(self, limit: int = MAX_ITEMS) -> list[Any]:
        """Stored results the current user may see, newest first, not idle."""
        store = self.ctx.store
        found = []
        for ws_id in reversed(store.list_workspaces()):
            meta = store.get_meta(ws_id)
            if meta is not None and is_visible(meta) and not self._idle(meta):
                found.append(meta)
                if len(found) >= limit:
                    break
        return found

    def _load(self, result_id: Any) -> tuple[Any, Any]:
        store = self.ctx.store
        meta = store.get_meta(result_id) if isinstance(result_id, str) and result_id else None
        if meta is None or not is_visible(meta) or self._idle(meta):
            if meta is not None and is_visible(meta):
                store.drop_workspace(result_id)
            known = [m.workspace_id for m in self._visible_metas(limit=10)]
            raise _FixableError(
                f"No stored result {result_id!r} (it may have expired).",
                "Use a result_id from a recent notice or from "
                f"{self.prefix}_list, or call the original tool again.",
                **({"stored_results": known} if known else {}),
            )
        store.touch(result_id)
        return meta, store.get_items(result_id)

    def _shape(
        self, index: int, item: Any, fields: Sequence[str] | None, limit: int,
        max_items: int = 20,
    ) -> Any:
        if fields and isinstance(item, dict):
            projected = {}
            for f in fields:
                value = resolve_path(item, f)
                if value is not MISSING:
                    projected[f] = value
            item = projected
        p = self.prefix

        def pointer(path: str, length: int, unit: str) -> str:
            where = f", field='{path}'" if path else ""
            return f"{length:,} {unit}; read it with {p}_read(result_id, index={index}{where})"

        return cut_strings(item, limit, pointer if "read" in self.tools else None, "", max_items)

    def _save(self, ws_type: str, source: str, output: Any, name: str) -> dict[str, Any]:
        if not isinstance(output, list):
            raise _FixableError("Only list outputs can be saved.",
                                "Drop save_as for count/group results.")
        if not _SAVE_AS_RE.fullmatch(str(name)):
            raise _FixableError(f"Invalid save_as name {name!r}.",
                                "Use 1-100 letters, digits, '_', '-' or '.'.")
        query_tool = f"{self.prefix}_query"
        try:
            ref = self.ctx.store_workspace(
                output,
                workspace_type=ws_type,
                workspace_id=f"saved_{name}",
                producer_fn=query_tool,
                producer_kwargs={"result_id": source, "save_as": name},
                meta={"chars": _json_len(output)},
                replace_if=lambda m: None if m.producer_fn == query_tool
                else "it is not a saved query result",
            )
        except WorkspaceConflictError as exc:
            raise _FixableError(str(exc), "Choose another save_as name.") from None
        self._evict(keep=ref.workspace_id)
        return self._notice(ref.workspace_id, query_tool, output, _json_len(output), None, saved=True)

    @contextlib.contextmanager
    def _scope(self) -> Iterator[None]:
        key = self._scope_fn()
        if key is None or key == current_session():
            yield
        else:
            with session_scope(key):
                yield


# ── module helpers ────────────────────────────────────────────────────────


#: Said in each tool description: some clients don't show server instructions.
_STORED = (" A tool result too large to return in full is stored, and the tool returns "
           "a notice with its result_id instead.")
_STR: dict[str, Any] = {"type": "string"}
_INT: dict[str, Any] = {"type": "integer"}
_FIELDS: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
_NO_DEFAULT = object()


def _param(schema: dict[str, Any], description: str, default: Any = _NO_DEFAULT) -> Any:
    """
    Parameter metadata: the schema clients see, while any value is accepted
    and checked by the tool itself, so a wrong argument gets the same fixable
    error as every other mistake instead of a raw validation message.
    """
    full = {**schema, "description": description}
    if default is not _NO_DEFAULT:
        full["default"] = default
    return WithJsonSchema(full)


def _str_arg(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _FixableError(f"{name} must be a non-empty string, not {value!r:.60}.",
                            f"Pass {name} as a string.")
    return value


def _int_arg(name: str, value: Any) -> int:
    if isinstance(value, bool):
        value = None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise _FixableError(f"{name} must be an integer, not {value!r:.60}.",
                            f"Pass {name} as a whole number.") from None


def _fields_arg(value: Any) -> list[str] | None:
    if value is None or value == []:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise _FixableError("fields must be a list of field names.",
                            'Pass fields like ["title", "author.name"].')
    return value


def _steps_arg(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise _FixableError("steps is not valid JSON.",
                                'Pass steps as a list like [{"limit": 10}].') from None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise _FixableError("steps must be a list of operations.",
                            'Pass steps as a list like [{"filter": {...}}, {"limit": 10}].')
    return value


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


def _focus(payload: Any) -> tuple[Any, dict[str, Any] | None, str | None]:
    """
    Pick the part of *payload* worth exploring, as ``(payload, other_fields,
    note)``: a list or a text stays as it is; an object made mostly of one
    list or one text (also nested, as in ``{"data": {"items": [...]}}`` or
    ``{"url": ..., "markdown": "..."}``) yields that value plus a preview of
    the other fields; an object with many keys of the same kind (``{"pkg-a":
    {...}, "pkg-b": {...}}``) becomes a list of its entries; anything else
    becomes readable text.
    """
    if isinstance(payload, (list, str)):
        return payload, None, None
    if isinstance(payload, dict) and payload:
        found = _dominant(payload)
        if found is not None:
            value, path, rest, note = found
            kind = "list_field" if isinstance(value, list) else "text_field"
            preview = build_preview(rest) if rest else {}
            if isinstance(value, str) and preview != rest:
                # Too much besides the text for a preview: keep it readable
                # and searchable at the end of the stored text.
                value = (f"{value}\n\n---\nOther fields of the result:\n"
                         + json.dumps(rest, ensure_ascii=False, indent=1, default=str))
                preview = {**preview, "…": "all other fields are at the end of the text"}
            # A lone top-level key needs no explanation.
            if "." in path or rest or note:
                return value, {kind: path, **preview}, note
            return value, None, None
        entries = _entries(payload)
        if entries is not None:
            return entries
    return json.dumps(payload, ensure_ascii=False, indent=1, default=str), None, None


def _dominant(
    node: dict[str, Any], depth: int = 0
) -> tuple[Any, str, dict[str, Any], str | None] | None:
    """The list or text holding most of *node*: (value, path, other fields, note)."""
    if depth >= 4:
        return None
    total = _json_len(node)
    for key, value in node.items():
        if not isinstance(value, (list, dict, str)) or _json_len(value) < _DOMINANT_SHARE * total:
            continue
        rest = {str(k): v for k, v in node.items() if k != key}
        if isinstance(value, (list, str)):
            return value, str(key), rest, None
        inner = _dominant(value, depth + 1)
        if inner is not None:
            value, path, inner_rest, note = inner
            return value, f"{key}.{path}", {**rest, **inner_rest}, note
        entries = _entries(value)
        if entries is not None:
            items, _, note = entries
            return items, str(key), rest, note
        return None
    return None


#: Keys an object needs before it is treated as a collection of entries.
_MIN_ENTRIES = 20


def _entries(obj: dict[str, Any]) -> tuple[list[Any], None, str] | None:
    """
    An object keyed by name or id (``{"a": {...}, "b": {...}}``, or a map of
    counts) as a list of entries, so it can be searched, sorted and counted.
    Each entry is the value with its key in ``key`` (or ``{"key", "value"}``
    for plain values).
    """
    if len(obj) < _MIN_ENTRIES:
        return None
    values = list(obj.values())
    # Entries of a collection are alike in size; one value holding half the
    # data is not an entry (it would be too big to read as one item).
    total = _json_len(obj)
    if any(_json_len(v) > total / 2 for v in values):
        return None
    objects = sum(isinstance(v, dict) for v in values)
    scalars = sum(not isinstance(v, (dict, list)) for v in values)
    if objects >= 0.9 * len(values):
        used = {f for v in values if isinstance(v, dict) for f in v}
        key = next((name for name in ("key", "_key", "__key", "entry_key")
                    if name not in used), "_entry_key_")
        items = [{**v, key: k} if isinstance(v, dict) else {key: k, "value": v}
                 for k, v in obj.items()]
        items = [{key: item.pop(key), **item} for item in items]  # key first
    elif scalars >= 0.9 * len(values):
        key = "key"
        items = [{"key": k, "value": v} for k, v in obj.items()]
    else:
        return None
    note = (f"The result is an object with {len(obj):,} keys; each item is one entry, "
            f"its key in {key!r}.")
    return items, None, note


def _source_indices(
    engine: PipelineEngine, payload: list[Any], steps: list[dict[str, Any]], output: list[Any]
) -> list[int | None] | None:
    """
    The position in *payload* of each output row, when rows are items of the
    list.  Most operations keep the stored objects; select/exclude make
    copies, so those steps are replayed without projection to find them.
    """
    positions = {id(item): i for i, item in enumerate(payload)}
    found = [positions.get(id(row)) for row in output[:MAX_ITEMS]]
    if all(i is not None for i in found):
        return found
    projections = {"select", "exclude"}
    if not any(next(iter(s), None) in projections for s in steps if isinstance(s, dict)):
        return None  # rows are new objects (e.g. group_by): no index
    bare = [s for s in steps if next(iter(s), None) not in projections]
    try:
        replay = engine.execute(payload, bare)
    except PipelineError:
        return None
    if not isinstance(replay, list) or len(replay) != len(output):
        return None
    found = [positions.get(id(row)) for row in replay[:MAX_ITEMS]]
    return found if all(i is not None for i in found) else None


def _example_steps(items: list[Any], fields: Mapping[str, Any]) -> str:
    """A query example built from the result's real fields and values."""
    rows = [i for i in items[:200] if isinstance(i, dict)]
    usable = {n: k for n, k in fields.items() if "." not in n and n != "…"}
    steps: list[dict[str, Any]] = []
    # Filter on a category-like field: a short string with few distinct values.
    best: tuple[int, str, str] | None = None
    for name, kind in usable.items():
        if kind != "string":
            continue
        values = [r[name] for r in rows if isinstance(r.get(name), str) and 0 < len(r[name]) <= 30]
        distinct = len(set(values))
        category_like = len(values) >= max(2, len(rows) // 2) and 1 < distinct <= max(2, len(rows) // 4)
        if category_like and (best is None or distinct < best[0]):
            best = (distinct, name, values[0])
    if best is not None:
        steps.append({"filter": {best[1]: best[2]}})
    numeric = [n for n, k in usable.items() if k in ("integer", "number") and not _id_like(n)]
    if numeric:
        steps.append({"sort": {"field": numeric[0], "order": "desc"}})
    steps.append({"limit": 10})
    return _dumps(steps)


def _id_like(name: str) -> bool:
    """Field names that identify rather than measure (poor sort examples)."""
    lower = name.lower()
    return lower in ("id", "index", "number", "no", "num", "key", "idx") or \
        lower.endswith(("_id", "_no", "_num", "_number")) or name.endswith(("Id", "ID"))


# ── text search ───────────────────────────────────────────────────────────

_PASSAGE_CONTEXT = 150
#: Occurrences considered per query word (keeps rare words findable).
_HITS_PER_WORD = 2_000


def _passages(text: str, query: str, limit: int) -> dict[str, Any]:
    """
    Passages of *text* for *query*: the exact phrase (case-insensitive),
    nearby occurrences merged into one passage; if the phrase doesn't occur,
    the passages with the most (and rarest) query words.
    """
    phrase = " ".join(query.split())
    pattern = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()), re.IGNORECASE)
    spans = [(m.start(), m.end()) for m in pattern.finditer(text)]
    if spans:
        groups = _group_spans(spans)
        passages = [
            {**_excerpt(text, start, end), **({"occurrences": n} if n > 1 else {})}
            for start, end, n in groups[:limit]
        ]
        return {"match": "phrase", "occurrences": len(spans), "returned": len(passages),
                "has_more": len(groups) > limit, "passages": passages}

    words = sorted({w for w in query_terms(phrase.lower()) if len(w) > 1})
    hits: list[tuple[int, int, str]] = []
    counts: dict[str, int] = {}
    for word in words:
        found = [(m.start(), m.end(), word)
                 for m in re.finditer(re.escape(word), text, re.IGNORECASE)]
        counts[word] = len(found)
        hits.extend(found[:_HITS_PER_WORD])
    if not hits:
        return {"match": "none", "returned": 0, "has_more": False, "passages": []}
    hits.sort()
    total = sum(counts.values())
    weight = {w: math.log(1 + total / counts[w]) for w in counts if counts[w]}
    window = 2 * _PASSAGE_CONTEXT
    candidates = []
    for i, (start, _, _) in enumerate(hits):
        inside = [h for h in hits[i:i + 64] if h[0] - start <= window]
        terms = {h[2] for h in inside}
        score = sum(weight[t] for t in terms)
        candidates.append((score, start, max(h[1] for h in inside), terms))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    chosen: list[tuple[float, int, int, set[str]]] = []
    for cand in candidates:
        if all(cand[2] < c[1] or cand[1] > c[2] for c in chosen):
            chosen.append(cand)
        if len(chosen) > limit:
            break
    passages = [{**_excerpt(text, start, end), "words": sorted(terms)}
                for _, start, end, terms in chosen[:limit]]
    return {"match": "words",
            "note": "The exact phrase does not occur; these passages contain the most "
                    "(and rarest) query words.",
            "returned": len(passages), "has_more": len(chosen) > limit, "passages": passages}


def _group_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Merge occurrences closer than one passage into (start, end, count)."""
    groups: list[list[int]] = []
    for start, end in spans:
        if groups and start - groups[-1][1] <= 2 * _PASSAGE_CONTEXT:
            groups[-1][1] = end
            groups[-1][2] += 1
        else:
            groups.append([start, end, 1])
    return [(s, e, n) for s, e, n in groups]


def _excerpt(text: str, start: int, end: int) -> dict[str, Any]:
    """The matched span with context (at most ~2 passages long)."""
    end = min(end, start + 4 * _PASSAGE_CONTEXT)
    lo, hi = max(0, start - _PASSAGE_CONTEXT), min(len(text), end + _PASSAGE_CONTEXT)
    return {"offset": start, "page": start // PAGE_CHARS, "text": text[lo:hi]}


def _match_passage(item: Any, terms: list[str], shown: int) -> dict[str, Any] | None:
    """
    The passage of *item* with the most query words, if it lies in a value
    longer than *shown* chars (cut in the hit) and past what the hit shows.
    """
    words = [t for t in terms if t]
    if not words:
        return None
    pattern = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)),
                         re.IGNORECASE)
    best: tuple[int, int, str, str] | None = None  # (-distinct words, start, path, value)
    for path, value in _string_leaves(item):
        if len(value) <= shown:
            continue
        found = [(m.start(), m.group().lower()) for m in itertools.islice(pattern.finditer(value), 200)]
        if not found:
            continue
        candidate = (-len({w for _, w in found}), found[0][0], path, value)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None or best[1] < shown - 40:
        return None
    _, start, path, value = best
    lo, hi = max(0, start - 120), min(len(value), start + 200)
    passage = ("…" if lo else "") + value[lo:hi] + ("…" if hi < len(value) else "")
    return {"field": path, "text": passage} if path else {"text": passage}


def _string_leaves(value: Any, path: str = "", budget: list[int] | None = None) -> Iterator[tuple[str, str]]:
    """``(path, string)`` for the strings inside *value* (paths as resolve_path reads them)."""
    budget = budget if budget is not None else [2_000]
    if budget[0] <= 0:
        return
    budget[0] -= 1
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _string_leaves(v, f"{path}.{k}" if path else str(k), budget)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _string_leaves(v, f"{path}.{i}" if path else str(i), budget)


# ── text outline and pages ────────────────────────────────────────────────

_HEADING = re.compile(r"^ {0,3}(#{1,3})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
#: A definition in source code: top level, or one indent in (methods).
_DEFINITION = re.compile(
    r"^( {0,4}|\t?)(?:export\s+)?(?:default\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(?:def|class|function|func|fn|struct|enum|trait|impl|interface)\s+[\w$(*&<]"
)
_MAX_OUTLINE = 50


def _outline(text: str) -> list[str] | None:
    """
    Where things are in a long text, as ``"page: title"`` lines indented by
    level: markdown headings (levels 1-3, not inside fenced code), or, for
    source code, its definitions (``def``, ``class``, ``function``, …).
    ``None`` if the text has fewer than three of either.
    """
    lines = text.splitlines(keepends=True)
    headings: list[tuple[int, str, int]] = []
    definitions: list[tuple[int, str, int]] = []
    offset, fence = 0, ""
    for n, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        opener = _FENCE.match(stripped)
        page = offset // PAGE_CHARS
        offset += len(line)
        if fence:
            if opener and opener.group(1)[0] == fence[0] and len(opener.group(1)) >= len(fence) \
                    and stripped.strip() == opener.group(1):
                fence = ""
            continue
        if opener:
            fence = opener.group(1)
            continue
        match = _HEADING.match(stripped)
        if match and not _in_comment_block(lines, n, match.group(1)):
            title = match.group(2)[:100]
            previous = headings[-1][1] if headings else ""
            # DeepWiki-style "Page: Overview" followed by "Overview".
            if not (previous and previous.endswith(f": {title}")):
                headings.append((len(match.group(1)), title, page))
            continue
        definition = _DEFINITION.match(stripped)
        if definition and re.search(r"[(:{]", stripped):
            level = 1 if not definition.group(1) else 2
            definitions.append((level, stripped.strip().rstrip(":{ ")[:100], page))
    entries = headings if len(headings) >= len(definitions) else definitions
    if len(entries) < 3:
        return None
    # Too many: drop the deepest level while more than one entry would remain.
    while len(entries) > _MAX_OUTLINE:
        deepest = max(level for level, _, _ in entries)
        shallower = [e for e in entries if e[0] < deepest]
        if len(shallower) <= 1:
            break
        entries = shallower
    more = len(entries) - _MAX_OUTLINE
    if more > 0:
        entries = entries[:_MAX_OUTLINE - 1]
    top = min(level for level, _, _ in entries)
    out = [f"{'  ' * (level - top)}{page}: {title}" for level, title, page in entries]
    if more > 0:
        out.append(f"… {more + 1} more")
    return out


def _in_comment_block(lines: list[str], n: int, hashes: str) -> bool:
    """
    A ``#`` line in a run of three or more like it is a comment block (YAML,
    shell, Python), not headings.
    """
    def like(m: int) -> bool:
        return 0 <= m < len(lines) and lines[m].lstrip(" ").startswith(hashes + " ")

    before = next(k for k in range(4) if not like(n - 1 - k))
    after = next(k for k in range(4) if not like(n + 1 + k))
    return before + after >= 2


def _text_page(text: str, page: int) -> dict[str, Any]:
    pages = max(1, math.ceil(len(text) / PAGE_CHARS))
    if page >= pages:
        raise _FixableError(f"Page {page} is past the end ({pages} pages).",
                            f"Use a page from 0 to {pages - 1}.")
    start = page * PAGE_CHARS
    return {"page": page, "pages": pages, "has_more": page + 1 < pages,
            "text": text[start:start + PAGE_CHARS]}


# ── small helpers ─────────────────────────────────────────────────────────


def _join_parts(parts: list[tuple[str, str | None]]) -> str:
    if len(parts) == 1:
        return parts[0][0]
    return "\n\n".join(f"<{uri}>\n{text}" if uri else text for text, uri in parts)


def _parse_parts(parts: list[tuple[str, str | None]]) -> Any:
    """
    The stored payload of a result's text blocks.  Several blocks that are
    each JSON become one list when they are alike (lists, or one object per
    item, as some servers send); one block holding most of the data is
    stored by itself; otherwise the blocks are parsed as one text.
    """
    if len(parts) > 1:
        if not any(uri for _, uri in parts):  # one document sent in pieces?
            whole = parse_text_payload("".join(text for text, _ in parts))
            if not isinstance(whole, str):
                return whole
        values = []
        for text, _ in parts:
            value = parse_text_payload(text)
            if isinstance(value, str):
                break
            values.append(value)
        else:
            if all(isinstance(v, list) for v in values):
                return [item for value in values for item in value]
            sizes = [_json_len(v) for v in values]
            if all(isinstance(v, dict) for v in values) and max(sizes) <= 0.5 * sum(sizes):
                return values
            if max(sizes) >= _DOMINANT_SHARE * sum(sizes):
                return values[sizes.index(max(sizes))]
    return parse_text_payload(_join_parts(parts))


def _as_text(payload: Any) -> str:
    return payload if isinstance(payload, str) else _dumps(payload)


def _result(value: Any, *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=_dumps(value))],
        structured_content=value if isinstance(value, dict) else None,
        is_error=is_error,
    )


def _trim(text: str, limit: int, what: str = "the error") -> str:
    """Start and end of *text* within about *limit* chars (unchanged if it fits)."""
    if len(text) <= limit:
        return text
    keep = max((limit - 80) // 2, 50)
    trimmed = (f"{text[:keep]}\n… [{len(text) - 2 * keep:,} characters of {what} "
               f"omitted] …\n{text[-keep:]}")
    return trimmed if len(trimmed) < len(text) else text


def _clamp(value: Any, low: int, high: int) -> int:
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return low


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_len(value: Any) -> int:
    return len(_dumps(value))


def _duration(seconds: float) -> str:
    for unit, size in (("day", 86_400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = round(seconds / size)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    n = max(1, round(seconds))
    return f"{n} second{'s' if n != 1 else ''}"
