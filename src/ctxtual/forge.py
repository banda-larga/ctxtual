"""
Forge — the central orchestrator for ctx.

Forge ties together:

- A storage backend (any :class:`~ctx.store.base.BaseStore`)
- One or more :class:`~ctx.toolset.ToolSet` instances (consumer tool groups)
- Producer decorators that intercept large results, store them, and return
  compact :class:`~ctx.types.WorkspaceRef` notifications to the agent.

Typical usage::

    from ctxtual import Forge, MemoryStore

    forge = Forge(store=MemoryStore())

    papers = forge.toolset("papers")

    @papers.tool
    def paginate(workspace_id: str, page: int = 0, size: int = 10):
        items = papers.store.get_items(workspace_id)
        return items[page * size : (page + 1) * size]

    @forge.producer(workspace_type="papers", toolsets=[papers])
    def search_papers(query: str, limit: int = 10_000):
        return db.search(query, limit)
"""

import asyncio
import functools
import inspect
import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any, Self

from ctxtual.exceptions import (
    PayloadTooLargeError,
    WorkspaceExpiredError,
    WorkspaceNotFoundError,
    WorkspaceTypeMismatchError,
)
from ctxtual.store.base import BaseStore
from ctxtual.store.memory import MemoryStore
from ctxtual.toolset import (
    ToolSet,
    _build_param_description,
    _extract_param_descriptions,
    _python_type_to_json_schema,
)
from ctxtual.types import (
    KeyFactory,
    ResultTransformer,
    WorkspaceMeta,
    WorkspaceRef,
)

logger = logging.getLogger("ctx")


# Sentinel for distinguishing "not passed" from None


class _SentinelType:
    """Unique sentinel to distinguish 'not passed' from None."""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"

    def __bool__(self) -> bool:
        return False


_SENTINEL = _SentinelType()


class Forge:
    """
    Central orchestrator for workspace-based context management.

    Args:
        store:          Storage backend.  Defaults to :class:`MemoryStore`.
        default_notify: If ``True``, producers return
                        :meth:`WorkspaceRef.to_dict` (a plain dict the LLM can
                        parse).  Set to ``False`` to receive the
                        :class:`WorkspaceRef` object directly.
        default_ttl:    Default time-to-live in seconds for workspaces.
                        ``None`` means no expiry.  Individual producers can
                        override.
        max_items:      Optional cap on the number of items a single producer
                        can store.  Raises :exc:`PayloadTooLargeError` if
                        exceeded.  ``None`` means no limit.
    """

    def __init__(
        self,
        store: BaseStore | None = None,
        *,
        default_notify: bool = True,
        default_ttl: float | None = None,
        max_items: int | None = None,
    ) -> None:
        self.store: BaseStore = store or MemoryStore()
        self.default_notify = default_notify
        self.default_ttl = default_ttl
        self.max_items = max_items
        self._lock = threading.RLock()
        self._toolsets: dict[str, ToolSet] = {}
        self._producers: dict[str, dict[str, Any]] = {}
        self._producer_wrappers: dict[str, Callable[..., Any]] = {}

    # ToolSet management

    def toolset(self, name: str, *, enforce_type: bool = True) -> ToolSet:
        """
        Create (or retrieve) a :class:`ToolSet` attached to this Forge.

        Always use this factory instead of constructing ``ToolSet()`` directly —
        it wires the store reference automatically.
        """
        with self._lock:
            if name in self._toolsets:
                return self._toolsets[name]
            ts = ToolSet(name, enforce_type=enforce_type)
            return self.register_toolset(ts)

    def register_toolset(self, toolset: ToolSet) -> ToolSet:
        """Attach an externally-created ToolSet to this Forge."""
        with self._lock:
            toolset._store = self.store
            self._toolsets[toolset.name] = toolset
            return toolset

    # Producer decorator

    def producer(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        workspace_type: str,
        toolsets: list[ToolSet] | ToolSet | None = None,
        key: str | KeyFactory | None = None,
        transform: ResultTransformer | None = None,
        meta: dict[str, Any] | None = None,
        notify: bool | None = None,
        ttl: float | None = _SENTINEL,  # type: ignore[assignment]
    ) -> Callable[..., Any]:
        """
        Decorator that turns a function into a context-aware producer.

        The wrapped function:

        1. Executes normally and captures the return value.
        2. Optionally transforms it via *transform*.
        3. Stores it under a new workspace in the store.
        4. Returns a :class:`WorkspaceRef` (or its dict form) instead of raw data.

        Args:
            workspace_type: Logical type of data being produced (e.g. ``"papers"``).
            toolsets:       ToolSet(s) whose tools the agent can use to consume
                            the stored data.
            key:            Controls workspace_id generation:

                            - ``None`` (default): auto UUID.
                            - str template: formatted with call kwargs.
                            - callable: receives kwargs dict, returns id string.
            transform:      Optional callable to pre-process the return value
                            before storage.
            meta:           Extra key/value pairs for :attr:`WorkspaceMeta.extra`.
            notify:         Override :attr:`Forge.default_notify` for this producer.
            ttl:            Time-to-live in seconds for this workspace.
                            Defaults to :attr:`Forge.default_ttl`.

        Can be used with or without parentheses::

            @forge.producer(workspace_type="papers", toolsets=[pager])
            def search(query: str): ...
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            _raw: list[ToolSet] = (
                [toolsets] if isinstance(toolsets, ToolSet) else (toolsets or [])
            )
            # Deduplicate by identity
            seen: set[int] = set()
            _toolsets: list[ToolSet] = []
            for ts in _raw:
                if id(ts) not in seen:
                    seen.add(id(ts))
                    _toolsets.append(ts)
            # Wire toolsets to this forge if they aren't already
            for ts in _toolsets:
                if ts._store is None:
                    self.register_toolset(ts)

            is_async = asyncio.iscoroutinefunction(func)

            def _build_result(
                func: Callable[..., Any],
                raw_result: Any,
                call_kwargs: dict[str, Any],
            ) -> Any:
                """Shared logic for sync and async paths."""
                payload = transform(raw_result) if transform else raw_result

                # Enforce max_items
                count = _count(payload)
                if self.max_items is not None and count > self.max_items:
                    raise PayloadTooLargeError(count, self.max_items)

                workspace_id = _resolve_key(key, workspace_type, call_kwargs)

                resolved_ttl = ttl if ttl is not _SENTINEL else self.default_ttl

                # Infer data shape from the payload
                shape = _infer_shape(payload)

                # Per-tool shape tracking means mixed-shape toolsets are
                # handled by the filter below (lines 277-288).  Only warn
                # if a toolset has **zero** compatible tools — that
                # indicates a likely configuration mistake, not just a
                # mixed-shape scenario that the filter handles silently.

                ws_meta = WorkspaceMeta(
                    workspace_id=workspace_id,
                    workspace_type=workspace_type,
                    producer_fn=func.__name__,
                    producer_kwargs=_safe_kwargs(call_kwargs),
                    item_count=count,
                    ttl=resolved_ttl,
                    data_shape=shape,
                    extra=meta or {},
                )
                with self.store.transaction():
                    self.store.init_workspace(ws_meta)
                    self.store.set_items(workspace_id, payload)

                # Build tool descriptions from raw tool docstrings
                tool_descriptions: dict[str, str] = {}
                for ts in _toolsets:
                    for t_name, t_fn in ts._raw_tools.items():
                        doc = inspect.getdoc(t_fn) or ""
                        # Take just the first sentence for brevity
                        first_line = doc.split("\n")[0].strip().rstrip(".")
                        if first_line:
                            tool_descriptions[t_name] = first_line

                # Infer JSON Schema for item structure so the LLM
                # knows field names and types without paginating first.
                item_schema = _infer_item_schema(payload)
                sample_fields: list[str] = []
                if item_schema and "properties" in item_schema:
                    sample_fields = list(item_schema["properties"].keys())
                elif isinstance(payload, dict):
                    sample_fields = list(payload.keys())

                # Filter tools to only include shape-compatible ones.
                # This prevents the LLM from seeing (and calling) tools
                # that will fail on the actual data.  Uses per-tool shape
                # (not per-ToolSet) so shared ToolSets work correctly.
                compatible_tools: list[str] = []
                compatible_descriptions: dict[str, str] = {}
                for ts in _toolsets:
                    for t_name in ts.tool_names:
                        tool_shape = ts._tool_shapes.get(t_name, "")
                        is_compatible = (
                            not tool_shape or not shape or tool_shape == shape
                        )
                        if is_compatible:
                            compatible_tools.append(t_name)
                            if t_name in tool_descriptions:
                                compatible_descriptions[t_name] = tool_descriptions[t_name]

                # Warn only for toolsets where ZERO tools survived filtering
                for ts in _toolsets:
                    ts_compatible = any(
                        t_name in compatible_tools for t_name in ts.tool_names
                    )
                    if not ts_compatible and ts.data_shape and shape and ts.data_shape != shape:
                        logger.warning(
                            "Producer '%s' returns %s data but toolset '%s' "
                            "expects %s data. None of its tools (%s) are "
                            "compatible with this workspace.",
                            func.__name__,
                            shape,
                            ts.name,
                            ts.data_shape,
                            ", ".join(ts.tool_names[:3]),
                        )

                ref = WorkspaceRef(
                    workspace_id=workspace_id,
                    workspace_type=workspace_type,
                    item_count=ws_meta.item_count,
                    data_shape=shape,
                    producer_fn=func.__name__,
                    available_tools=compatible_tools,
                    tool_descriptions=compatible_descriptions,
                    metadata=meta or {},
                    sample_fields=sample_fields,
                    item_schema=item_schema,
                )

                should_notify = notify if notify is not None else self.default_notify
                return ref.to_dict() if should_notify else ref

            if is_async:

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    call_kwargs = dict(bound.arguments)
                    raw_result = await func(*args, **kwargs)
                    return _build_result(func, raw_result, call_kwargs)

                async_wrapper.__workspace_type__ = workspace_type  # type: ignore[attr-defined]
                async_wrapper.__toolsets__ = [ts.name for ts in _toolsets]  # type: ignore[attr-defined]
                async_wrapper.__is_producer__ = True  # type: ignore[attr-defined]
                with self._lock:
                    self._producers[func.__name__] = {
                        "func": func,
                        "workspace_type": workspace_type,
                        "toolset_names": [t for ts in _toolsets for t in ts.tool_names],
                    }
                    self._producer_wrappers[func.__name__] = async_wrapper
                return async_wrapper

            else:

                @functools.wraps(func)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    call_kwargs = dict(bound.arguments)
                    raw_result = func(*args, **kwargs)
                    return _build_result(func, raw_result, call_kwargs)

                wrapper.__workspace_type__ = workspace_type  # type: ignore[attr-defined]
                wrapper.__toolsets__ = [ts.name for ts in _toolsets]  # type: ignore[attr-defined]
                wrapper.__is_producer__ = True  # type: ignore[attr-defined]
                with self._lock:
                    self._producers[func.__name__] = {
                        "func": func,
                        "workspace_type": workspace_type,
                        "toolset_names": [t for ts in _toolsets for t in ts.tool_names],
                    }
                    self._producer_wrappers[func.__name__] = wrapper
                return wrapper

        if fn is not None:
            raise TypeError(
                "@forge.producer requires keyword arguments. "
                "Use @forge.producer(workspace_type='...', ...) form."
            )
        return decorator

    # Consumer decorator

    def consumer(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        workspace_type: str | None = None,
        produces: str | None = None,
        produces_toolsets: list[ToolSet] | ToolSet | None = None,
    ) -> Callable[..., Any]:
        """
        Decorator for consumer tools that may also *produce* a new workspace.

        Use this when a tool reads from one workspace and writes a derived
        workspace (e.g. a "filter_papers" that stores filtered results so the
        agent can paginate them independently).

        Args:
            workspace_type:    Expected type of the input workspace_id.
                               If set, validates before executing.
            produces:          If this tool stores results, the workspace type
                               for the derived workspace.
            produces_toolsets: ToolSets available on the derived workspace.

        The wrapped function receives a :class:`ConsumerContext` via a
        parameter named ``forge_ctx`` (or any parameter annotated with
        ``ConsumerContext``).  No ``= None`` default is needed::

            @forge.consumer(workspace_type="data")
            def process(workspace_id: str, forge_ctx: ConsumerContext):
                return forge_ctx.get_items()

        Works with both sync and async functions.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            is_async = asyncio.iscoroutinefunction(func)

            # Detect the forge_ctx parameter by name or ConsumerContext
            # annotation so the user doesn't need ``= None``.
            _orig_sig = inspect.signature(func)
            _ctx_param: str | None = None
            for pname, param in _orig_sig.parameters.items():
                if pname == "forge_ctx" or param.annotation is ConsumerContext:
                    _ctx_param = pname
                    break

            # Build a binding signature that omits the context parameter
            # so callers don't need to supply it.
            if _ctx_param is not None:
                _bind_params = [
                    p
                    for n, p in _orig_sig.parameters.items()
                    if n != _ctx_param
                ]
                _bind_sig = _orig_sig.replace(parameters=_bind_params)
            else:
                _bind_sig = _orig_sig

            def _prepare(args: tuple, kwargs: dict) -> tuple[dict, ConsumerContext]:
                bound = _bind_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                call_kwargs = dict(bound.arguments)
                workspace_id = call_kwargs.get("workspace_id")

                if workspace_id and workspace_type:
                    ws_meta = self.store.get_meta(workspace_id)
                    if ws_meta is None:
                        raise WorkspaceNotFoundError(workspace_id)
                    if ws_meta.is_expired:
                        self.store.drop_workspace(workspace_id)
                        raise WorkspaceExpiredError(workspace_id)

                ctx = ConsumerContext(
                    forge=self,
                    input_workspace_id=workspace_id or "",
                    output_type=produces,
                    output_toolsets=(
                        [produces_toolsets]
                        if isinstance(produces_toolsets, ToolSet)
                        else (produces_toolsets or [])
                    ),
                )
                return call_kwargs, ctx

            if is_async:

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    call_kwargs, ctx = _prepare(args, kwargs)
                    if _ctx_param:
                        call_kwargs[_ctx_param] = ctx
                    return await func(**call_kwargs)

                async_wrapper.__is_consumer__ = True  # type: ignore[attr-defined]
                async_wrapper.__workspace_type__ = workspace_type  # type: ignore[attr-defined]
                return async_wrapper

            else:

                @functools.wraps(func)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    call_kwargs, ctx = _prepare(args, kwargs)
                    if _ctx_param:
                        call_kwargs[_ctx_param] = ctx
                    return func(**call_kwargs)

                wrapper.__is_consumer__ = True  # type: ignore[attr-defined]
                wrapper.__workspace_type__ = workspace_type  # type: ignore[attr-defined]
                return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator

    # Introspection & management

    def list_workspaces(self, workspace_type: str | None = None) -> list[str]:
        """Return workspace_ids, optionally filtered by type."""
        return self.store.list_workspaces(workspace_type)

    def workspace_meta(self, workspace_id: str) -> WorkspaceMeta | None:
        """Return metadata for a workspace."""
        return self.store.get_meta(workspace_id)

    def drop_workspace(self, workspace_id: str) -> None:
        """Remove a workspace and all its data."""
        self.store.drop_workspace(workspace_id)

    def sweep_expired(self) -> list[str]:
        """Remove all expired workspaces and return their ids."""
        return self.store.sweep_expired()

    def clear(self) -> None:
        """Remove **all** workspaces from the store."""
        self.store.clear()

    # Schema export for LLM integration

    def get_all_tool_schemas(
        self, workspace_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Export every tool from every registered ToolSet as OpenAI-compatible
        function-calling schemas.

        This is the main integration point: pass the output directly to
        ``openai.chat.completions.create(tools=forge.get_all_tool_schemas())``.

        Raises:
            ValueError: if duplicate tool names are found across ToolSets.
        """
        schemas: list[dict[str, Any]] = []
        seen_names: dict[str, str] = {}  # tool_name → toolset_name
        for ts in self._toolsets.values():
            for schema in ts.to_tool_schemas(workspace_id=workspace_id):
                name = schema["function"]["name"]
                if name in seen_names:
                    raise ValueError(
                        f"Duplicate tool name '{name}' in ToolSets "
                        f"'{seen_names[name]}' and '{ts.name}'. "
                        f"Tool names must be unique across all ToolSets. "
                        f"Use the built-in utilities (paginator, text_search, "
                        f"filter_set) which auto-prefix names, or manually "
                        f"assign unique names with @ts.tool(name='...')."
                    )
                seen_names[name] = ts.name
                schemas.append(schema)
        return schemas

    def get_producer_schemas(self) -> list[dict[str, Any]]:
        """
        Export OpenAI-compatible function-calling schemas for all registered
        producer functions.

        This eliminates the need for hand-written producer tool definitions.
        Combine with :meth:`get_all_tool_schemas` for a complete tool list::

            tools = forge.get_producer_schemas() + forge.get_all_tool_schemas()
        """
        schemas: list[dict[str, Any]] = []
        for name, info in self._producers.items():
            func = info["func"]
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or f"Producer '{name}'."

            # Append a hint about the workspace pattern
            ws_type = info.get("workspace_type", "")
            if ws_type:
                doc += (
                    "\n\nReturns a workspace notification (not raw data). "
                    "Use the exploration tools to read the stored results."
                )

            properties: dict[str, Any] = {}
            required: list[str] = []
            docstring_descs = _extract_param_descriptions(func)

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                p_schema = _python_type_to_json_schema(param.annotation)
                p_schema["description"] = _build_param_description(
                    param_name, param, docstring_descs
                )
                if (
                    param.default is not inspect.Parameter.empty
                    and param.default is not None
                ):
                    p_schema["default"] = param.default
                properties[param_name] = p_schema
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": doc,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schemas

    def get_tools(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """
        Export **all** tool schemas — both producers and consumers — in one call.

        This is the simplest integration point::

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=forge.get_tools(),
            )

        Returns OpenAI-compatible function-calling schemas.
        """
        return self.get_producer_schemas() + self.get_all_tool_schemas(
            workspace_id=workspace_id
        )

    def system_prompt(self, *, preamble: str = "") -> str:
        """
        Auto-generate a system prompt section that explains ctx tools.

        The generated prompt is derived from the registered producers and
        toolsets.  It covers the workspace pattern, available data sources,
        exploration tools, pipeline syntax (if registered), and error recovery.

        Args:
            preamble: Optional text prepended before the tool instructions.
                      Use this for your agent's role description.

        Usage::

            SYSTEM = forge.system_prompt(
                preamble="You are a research assistant."
            )
        """
        lines: list[str] = []

        if preamble:
            lines.append(preamble.rstrip())
            lines.append("")

        # ── Core workspace pattern ───────────────────────────────────────
        lines.append(
            "## Data tools\n"
            "Some of your tools produce large results that are stored in "
            "workspaces instead of returned directly. When this happens, "
            "the response includes a `workspace_id`, the data `fields`, "
            "a `sample_item`, and a list of exploration tools.\n"
            "Follow the instructions in each tool response."
        )

        # ── Producers ────────────────────────────────────────────────────
        if self._producers:
            lines.append("")
            lines.append("### Data sources")
            for name, info in self._producers.items():
                func = info["func"]
                doc = inspect.getdoc(func) or ""
                first_line = doc.split("\n")[0].strip() if doc else name
                ts_names = info.get("toolset_names", [])
                lines.append(f"- **{name}**: {first_line}")
                if ts_names:
                    tools_str = ", ".join(f"`{t}`" for t in ts_names[:6])
                    if len(ts_names) > 6:
                        tools_str += f" (+{len(ts_names) - 6} more)"
                    lines.append(f"  Exploration tools: {tools_str}")

        # ── Collect tool categories ──────────────────────────────────────
        has_pipeline = False
        has_search = False
        has_filter = False
        has_kv = False
        with self._lock:
            for ts in self._toolsets.values():
                for t_name in ts.tool_names:
                    if t_name.endswith("_pipe"):
                        has_pipeline = True
                    elif t_name.endswith("_search"):
                        has_search = True
                    elif t_name.endswith("_filter_by"):
                        has_filter = True
                    elif t_name.endswith("_get_keys"):
                        has_kv = True

        # ── Pipeline syntax ──────────────────────────────────────────────
        if has_pipeline:
            lines.append("")
            lines.append(
                "### Pipeline (compound operations in one call)\n"
                "Use `*_pipe(workspace_id, steps)` to chain operations:\n"
                "```\n"
                'steps: [{"filter": {"year": {"$gte": 2020}}}, '
                '{"sort": {"field": "score", "order": "desc"}}, '
                '{"limit": 5}]\n'
                "```\n"
                "Operations: filter, search, sort, select, exclude, "
                "limit, skip, slice, sample, unique, flatten, group_by, "
                "count.\n"
                "Filter operators: `$gt` `$gte` `$lt` `$lte` `$ne` `$in` "
                "`$nin` `$contains` `$startswith` `$regex` `$exists`.\n"
                "Logical: `$or` `$and` `$not`. Dot notation: `author.name`."
            )

        # ── Exploration tools summary ────────────────────────────────────
        tool_hints: list[str] = []
        if has_search:
            tool_hints.append("`*_search(query)` for text search")
        if has_filter:
            tool_hints.append("`*_filter_by(field, value, operator)` for filtering")
        if has_kv:
            tool_hints.append("`*_get_keys / *_get_value` for dict workspaces")

        if tool_hints:
            lines.append("")
            lines.append("### Other exploration tools")
            for h in tool_hints:
                lines.append(f"- {h}")

        # ── Error recovery ───────────────────────────────────────────────
        lines.append("")
        lines.append(
            "### Errors\n"
            "If a tool returns an `error` field, read the "
            "`suggested_action` for recovery guidance. "
            "Wrong workspace_id errors include a list of "
            "`available_workspaces`."
        )

        return "\n".join(lines)

    def dispatch_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Route a tool call from an LLM response to the correct function.

        Handles **both** producer and consumer tools.  This is the counterpart
        to :meth:`get_tools`: when the LLM returns a function call, pass it
        here to execute.

        Args:
            tool_name:  The ``name`` field from the LLM's tool call.
            arguments:  The ``arguments`` dict from the LLM's tool call.

        Raises:
            KeyError: if no tool with that name is registered, or if the name
                      is ambiguous across ToolSets and cannot be resolved.
        """
        all_tools = [t for ts in self._toolsets.values() for t in ts.tool_names]
        all_producers = list(self._producers.keys())

        def _execute(fn: Callable[..., Any]) -> Any:
            """Execute and convert ctx errors to LLM-friendly dicts."""
            try:
                return fn(**arguments)
            except (
                WorkspaceNotFoundError,
                WorkspaceExpiredError,
                WorkspaceTypeMismatchError,
            ) as exc:
                return exc.to_llm_dict()

        # Collect all ToolSets that have this tool
        matches = [
            ts for ts in self._toolsets.values() if tool_name in ts._tools
        ]

        if len(matches) == 1:
            return _execute(matches[0]._tools[tool_name])

        if len(matches) > 1:
            # Disambiguate via workspace_id → workspace_type → ToolSet.name
            workspace_id = arguments.get("workspace_id")
            if workspace_id:
                meta = self.store.get_meta(workspace_id)
                if meta:
                    for ts in matches:
                        if ts.name == meta.workspace_type:
                            return _execute(ts._tools[tool_name])
            ts_names = [ts.name for ts in matches]
            return {
                "error": (
                    f"Tool '{tool_name}' exists in multiple ToolSets: {ts_names}. "
                    f"Cannot determine which to use."
                ),
                "suggested_action": (
                    f"Use the workspace-type-prefixed name (e.g., "
                    f"'{ts_names[0]}_{tool_name}') or pass a valid workspace_id "
                    f"so the correct ToolSet can be inferred."
                ),
                "available_tools": all_tools,
            }

        # Check registered producer wrappers
        if tool_name in self._producer_wrappers:
            return self._producer_wrappers[tool_name](**arguments)

        return {
            "error": f"Tool '{tool_name}' not found.",
            "available_tools": all_tools,
            "available_producers": all_producers,
            "suggested_action": (
                "Check the tool name and try again. "
                "Use one of the available tools or producers listed above."
            ),
        }

    # Context manager

    def close(self) -> None:
        """Release resources held by the underlying store."""
        self.store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Forge(store={self.store!r}, " f"toolsets={list(self._toolsets.keys())})"
        )


# ConsumerContext — injected into @forge.consumer functions


class ConsumerContext:
    """
    Provides a consumer function with store access and the ability to
    produce a derived workspace from its results.

    Injected automatically by :meth:`Forge.consumer` into the parameter
    named ``forge_ctx`` (or annotated with ``ConsumerContext``).

    For direct use in tests, construct with just ``forge`` and
    ``input_workspace_id``::

        ctx = ConsumerContext(forge, "ws_1")
        result = my_consumer.__wrapped__("ws_1", forge_ctx=ctx)
    """

    def __init__(
        self,
        forge: Forge,
        input_workspace_id: str = "",
        output_type: str | None = None,
        output_toolsets: list[ToolSet] | None = None,
    ) -> None:
        self._forge = forge
        self.input_workspace_id = input_workspace_id
        self._output_type = output_type
        self._output_toolsets = output_toolsets or []

    @property
    def store(self) -> BaseStore:
        """The Forge's storage backend."""
        return self._forge.store

    def get_items(self, workspace_id: str | None = None) -> Any:
        """
        Shortcut: get items from the input workspace or a specified one.
        """
        return self.store.get_items(workspace_id or self.input_workspace_id)

    def emit(
        self,
        payload: Any,
        *,
        workspace_type: str | None = None,
        workspace_id: str | None = None,
        toolsets: list[ToolSet] | None = None,
        meta: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> dict[str, Any]:
        """
        Store *payload* as a new workspace and return a ``WorkspaceRef`` dict.

        Call this inside a ``@forge.consumer`` to produce a derived workspace
        from filtered / processed results.
        """
        out_type = workspace_type or self._output_type or "derived"
        out_toolsets = toolsets or self._output_toolsets
        ws_id = workspace_id or f"{out_type}_{uuid.uuid4().hex[:8]}"
        shape = _infer_shape(payload)

        ws_meta = WorkspaceMeta(
            workspace_id=ws_id,
            workspace_type=out_type,
            item_count=_count(payload),
            ttl=ttl or self._forge.default_ttl,
            data_shape=shape,
            extra=meta or {},
        )
        with self.store.transaction():
            self.store.init_workspace(ws_meta)
            self.store.set_items(ws_id, payload)

        ref = WorkspaceRef(
            workspace_id=ws_id,
            workspace_type=out_type,
            item_count=ws_meta.item_count,
            data_shape=shape,
            available_tools=[t for ts in out_toolsets for t in ts.tool_names],
        )
        return ref.to_dict()


# Internal helpers


def _resolve_key(
    key: str | KeyFactory | None,
    workspace_type: str,
    call_kwargs: dict[str, Any],
) -> str:
    """Generate a workspace_id from the key specification."""
    if key is None:
        return f"{workspace_type}_{uuid.uuid4().hex[:10]}"
    if callable(key):
        return key(call_kwargs)
    try:
        return key.format(**call_kwargs)
    except (KeyError, ValueError):
        return f"{workspace_type}_{uuid.uuid4().hex[:10]}"


def _count(value: Any) -> int:
    """Count items in a value.  For strings, returns 1 (one document)."""
    if isinstance(value, str):
        return 1  # A string is one item, not len(string) characters
    try:
        return len(value)
    except TypeError:
        return 1


def _infer_shape(payload: Any) -> str:
    """Infer the data shape of a workspace payload."""
    if isinstance(payload, list):
        return "list"
    if isinstance(payload, dict):
        return "dict"
    return "scalar"


_SCHEMA_SAMPLE_SIZE = 100


def _infer_item_schema(
    payload: Any,
    *,
    sample_size: int = _SCHEMA_SAMPLE_SIZE,
) -> dict[str, Any] | None:
    """Infer a JSON Schema from the workspace payload.

    For list payloads, scans up to *sample_size* items to discover all
    fields and their types.  Returns a JSON Schema ``object`` describing
    the item structure, or ``None`` for non-list payloads.

    For dict payloads, returns a schema of the top-level keys.
    """
    if isinstance(payload, list):
        if not payload:
            return None
        items_to_scan = payload[:sample_size]
        # Only infer schema for list-of-dicts
        if not isinstance(items_to_scan[0], dict):
            return None
        return _schema_from_dicts(items_to_scan, len(payload))

    if isinstance(payload, dict):
        return _schema_from_dicts([payload], 1)

    return None


def _schema_from_dicts(
    items: list[dict[str, Any]], total: int
) -> dict[str, Any]:
    """Build a JSON Schema object from a sample of dicts."""
    field_types: dict[str, set[str]] = {}
    field_counts: dict[str, int] = {}
    n = len(items)

    for item in items:
        if not isinstance(item, dict):
            continue
        for k, v in item.items():
            field_counts[k] = field_counts.get(k, 0) + 1
            field_types.setdefault(k, set()).add(_json_type(v))

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field_name, types in field_types.items():
        types.discard("null")
        prop: dict[str, Any] = {}

        if len(types) == 0:
            prop["type"] = "null"
        elif len(types) == 1:
            prop["type"] = next(iter(types))
        else:
            prop["type"] = sorted(types)

        properties[field_name] = prop

        # Field is required if present in all sampled items
        if field_counts.get(field_name, 0) >= n:
            required.append(field_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _json_type(value: Any) -> str:
    """Map a Python value to its JSON Schema type string."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Serialise kwargs for metadata — drop non-primitive values gracefully."""
    safe: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, str | int | float | bool | type(None)):
            safe[k] = v
        else:
            safe[k] = repr(v)
    return safe
