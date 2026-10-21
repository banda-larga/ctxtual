"""
Ctx — the central orchestrator for ctxtual.

Ctx ties together:

- A storage backend (any :class:`~ctxtual.store.base.BaseStore`)
- One or more :class:`~ctxtual.toolset.ToolSet` instances (consumer tool groups)
- Producer decorators that intercept large results, store them, and return
  compact :class:`~ctxtual.types.WorkspaceRef` notifications to the agent.
- Consumer decorators for tools that derive new workspaces from old ones.

Typical usage::

    from ctxtual import Ctx, MemoryStore

    ctx = Ctx(store=MemoryStore())

    papers = ctx.toolset("papers")

    @papers.tool
    def paginate(workspace_id: str, page: int = 0, size: int = 10):
        items = papers.store.get_items(workspace_id)
        return items[page * size : (page + 1) * size]

    @ctx.producer(workspace_type="papers", toolsets=[papers])
    def search_papers(query: str, limit: int = 10_000):
        return db.search(query, limit)
"""

import asyncio
import contextlib
import functools
import inspect
import json
import logging
import string
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Self

from ctxtual.exceptions import (
    CtxtualError,
    PayloadTooLargeError,
    WorkspaceConflictError,
    WorkspaceNotFoundError,
)
from ctxtual.session import current_session, is_visible, scoped_id, session_scope
from ctxtual.store.base import BaseStore
from ctxtual.store.memory import MemoryStore
from ctxtual.toolset import (
    ToolSet,
    ToolSpec,
    _build_param_description,
    _extract_param_descriptions,
    _python_type_to_json_schema,
    check_workspace,
    validate_workspace_id,
)
from ctxtual.types import KeyFactory, ResultTransformer, WorkspaceMeta, WorkspaceRef

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

#: Decides whether an existing workspace may be overwritten by a new write.
ReplacePolicy = Callable[[WorkspaceMeta], str | None]


@dataclass
class _Target:
    """A resolved dispatch target."""

    fn: Callable[..., Any]
    signature: inspect.Signature
    kind: str  # "tool" | "producer" | "consumer"


class Ctx:
    """
    Central orchestrator for workspace-based context management.

    Args:
        store:          Storage backend.  Defaults to :class:`MemoryStore`.
        default_notify: If ``True``, producers return
                        :meth:`WorkspaceRef.to_dict` (a plain dict the LLM can
                        parse).  Set to ``False`` to receive the
                        :class:`WorkspaceRef` object directly.
        default_ttl:    Default time-to-live in seconds for workspaces,
                        measured from creation.  ``None`` means no expiry.
                        Individual producers can override.
        max_items:      Optional cap on the number of items a single workspace
                        can hold.  Raises :exc:`PayloadTooLargeError` if
                        exceeded.  ``None`` means no limit.
        safe:           If ``True`` (default), exceptions raised by producer
                        and consumer functions during
                        :meth:`dispatch_tool_call` are returned to the LLM
                        as error dicts instead of propagating.  ToolSet tools
                        follow their own ``ToolSet.safe`` flag.
        max_result_chars: Upper bound on the JSON size of any single result
                        returned by :meth:`dispatch_tool_call`.  Larger
                        results are replaced by an error with a short
                        preview, so no tool — built-in or custom — can flood
                        the context.  ``None`` disables the check.
    """

    def __init__(
        self,
        store: BaseStore | None = None,
        *,
        default_notify: bool = True,
        default_ttl: float | None = None,
        max_items: int | None = None,
        safe: bool = True,
        max_result_chars: int | None = 60_000,
    ) -> None:
        self.store: BaseStore = store or MemoryStore()
        self.default_notify = default_notify
        self.default_ttl = default_ttl
        self.max_items = max_items
        self.safe = safe
        self.max_result_chars = max_result_chars
        self._lock = threading.RLock()
        self._toolsets: dict[str, ToolSet] = {}
        self._producers: dict[str, dict[str, Any]] = {}
        self._producer_wrappers: dict[str, Callable[..., Any]] = {}
        self._consumers: dict[str, dict[str, Any]] = {}
        self._type_toolsets: dict[str, list[ToolSet]] = {}

    # ToolSet management

    def toolset(self, name: str, *, enforce_type: bool = True) -> ToolSet:
        """
        Create (or retrieve) a :class:`ToolSet` attached to this Ctx.

        Always use this factory instead of constructing ``ToolSet()`` directly —
        it wires the store reference automatically.  ``enforce_type`` only
        applies when the ToolSet is created.
        """
        with self._lock:
            if name in self._toolsets:
                return self._toolsets[name]
            ts = ToolSet(name, enforce_type=enforce_type)
            return self.register_toolset(ts)

    def register_toolset(self, toolset: ToolSet) -> ToolSet:
        """Attach an externally-created ToolSet to this Ctx."""
        with self._lock:
            toolset._store = self.store
            self._toolsets[toolset.name] = toolset
            return toolset

    def toolsets_for_type(self, workspace_type: str) -> list[ToolSet]:
        """ToolSets registered (by producers/consumers) for *workspace_type*."""
        with self._lock:
            return list(self._type_toolsets.get(workspace_type, []))

    def _resolve_toolsets(
        self,
        toolsets: list[ToolSet | ToolSpec] | ToolSet | ToolSpec | None,
        workspace_type: str | None,
    ) -> list[ToolSet]:
        """Materialize ToolSpecs, de-duplicate, wire to this Ctx."""
        raw: list[ToolSet | ToolSpec] = (
            [toolsets]
            if isinstance(toolsets, (ToolSet, ToolSpec))
            else list(toolsets or [])
        )
        resolved: list[ToolSet] = []
        for ts in raw:
            if isinstance(ts, ToolSpec):
                if workspace_type is None:
                    raise ValueError(
                        f"{ts!r} has no name; pass a workspace type "
                        "(e.g. consumer(produces=...)) or name the toolset."
                    )
                ts = ts.materialize(workspace_type)
            if all(ts is not seen for seen in resolved):
                resolved.append(ts)
        with self._lock:
            for ts in resolved:
                if ts._store is None:
                    self.register_toolset(ts)
            if workspace_type is not None:
                known = self._type_toolsets.setdefault(workspace_type, [])
                for ts in resolved:
                    if all(ts is not k for k in known):
                        known.append(ts)
        return resolved

    # Sessions

    def session(self, session_id: str | None) -> contextlib.AbstractContextManager:
        """
        Scope everything in the ``with`` block to *session_id*.

        See :mod:`ctxtual.session` for the isolation rules.
        """
        return session_scope(session_id)

    # Storing workspaces (shared by producers, consumers and pipelines)

    def store_workspace(
        self,
        payload: Any,
        *,
        workspace_type: str,
        workspace_id: str | None = None,
        toolsets: list[ToolSet] | None = None,
        producer_fn: str = "",
        producer_kwargs: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        ttl: float | None | _SentinelType = _SENTINEL,
        replace_if: ReplacePolicy | None = None,
    ) -> WorkspaceRef:
        """
        Store *payload* as a workspace and return its :class:`WorkspaceRef`.

        Args:
            workspace_id: Explicit id.  It is namespaced to the active
                          session (see :func:`ctxtual.session.scoped_id`).
                          ``None`` generates a random id.
            toolsets:     Tools advertised in the ref.  Defaults to the
                          ToolSets registered for *workspace_type*.
            replace_if:   Called with the existing workspace's metadata when
                          *workspace_id* is taken; return ``None`` to allow
                          the overwrite or a reason string to refuse it
                          (raises :exc:`WorkspaceConflictError`).  Without a
                          policy, only workspaces of the same type that the
                          session can see may be replaced.
        """
        count = _count(payload)
        if self.max_items is not None and count > self.max_items:
            raise PayloadTooLargeError(count, self.max_items)

        if workspace_id is None:
            workspace_id = f"{workspace_type}_{uuid.uuid4().hex[:12]}"
        else:
            workspace_id = scoped_id(workspace_id)

        shape = _infer_shape(payload)
        ws_meta = WorkspaceMeta(
            workspace_id=workspace_id,
            workspace_type=workspace_type,
            producer_fn=producer_fn,
            producer_kwargs=_safe_kwargs(producer_kwargs or {}),
            item_count=count,
            ttl=self.default_ttl if ttl is _SENTINEL else ttl,  # type: ignore[arg-type]
            data_shape=shape,
            extra=dict(meta or {}),
            session_id=current_session(),
        )
        with self.store.transaction():
            existing = self.store.get_meta(workspace_id)
            if existing is not None:
                reason = _replace_reason(existing, workspace_type, replace_if)
                if reason is not None:
                    raise WorkspaceConflictError(workspace_id, reason)
            self.store.init_workspace(ws_meta)
            self.store.set_items(workspace_id, payload)

        if toolsets is None:
            toolsets = self.toolsets_for_type(workspace_type)
        return self._build_ref(ws_meta, payload, toolsets)

    def _build_ref(
        self, meta: WorkspaceMeta, payload: Any, toolsets: list[ToolSet]
    ) -> WorkspaceRef:
        shape = meta.data_shape
        compatible: list[str] = []
        descriptions: dict[str, str] = {}
        for ts in toolsets:
            ts_compatible = []
            for t_name in ts.tool_names:
                tool_shape = ts.tool_shape(t_name)
                if not tool_shape or not shape or tool_shape == shape:
                    ts_compatible.append(t_name)
                    doc = inspect.getdoc(ts._raw_tools[t_name]) or ""
                    first_line = doc.split("\n")[0].strip().rstrip(".")
                    if first_line:
                        descriptions[t_name] = first_line
            if not ts_compatible and ts.tool_names:
                expected = sorted(
                    {ts.tool_shape(t) or "any" for t in ts.tool_names}
                )
                logger.warning(
                    "Workspace '%s' holds %s data but toolset '%s' expects "
                    "%s data; none of its tools (%s) are offered.",
                    meta.workspace_id,
                    shape,
                    ts.name,
                    "/".join(expected),
                    ", ".join(ts.tool_names[:3]),
                )
            compatible.extend(t for t in ts_compatible if t not in compatible)

        item_schema = _infer_item_schema(payload)
        sample_fields: list[str] = []
        if item_schema and "properties" in item_schema:
            sample_fields = list(item_schema["properties"].keys())

        return WorkspaceRef(
            workspace_id=meta.workspace_id,
            workspace_type=meta.workspace_type,
            item_count=meta.item_count,
            data_shape=shape,
            producer_fn=meta.producer_fn,
            available_tools=compatible,
            tool_descriptions=descriptions,
            metadata=meta.extra,
            sample_fields=sample_fields,
            item_schema=item_schema,
        )

    # Producer decorator

    def producer(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        workspace_type: str,
        toolsets: list[ToolSet | ToolSpec] | ToolSet | ToolSpec | None = None,
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

                            - ``None`` (default): random id.
                            - str template: formatted with the call arguments,
                              e.g. ``"papers_{query}"``.  Every field must be
                              a parameter of the function.
                            - callable: receives the arguments dict, returns
                              the id string.

                            Deterministic ids are idempotent: calling again
                            with the same arguments replaces the workspace.
                            A key never overwrites a workspace created by a
                            different producer or of a different type.
            transform:      Optional callable to pre-process the return value
                            before storage.
            meta:           Extra key/value pairs for :attr:`WorkspaceMeta.extra`.
            notify:         Override :attr:`Ctx.default_notify` for this producer.
            ttl:            Time-to-live in seconds for this workspace.
                            Defaults to :attr:`Ctx.default_ttl`.

        Sync and ``async def`` functions are both supported::

            @ctx.producer(workspace_type="papers", toolsets=[pager])
            def search(query: str): ...
        """
        if fn is not None:
            raise TypeError(
                "@ctx.producer requires keyword arguments. "
                "Use @ctx.producer(workspace_type='...', ...) form."
            )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            sig = inspect.signature(func)
            _validate_key_template(key, sig, func.__name__)
            _toolsets = self._resolve_toolsets(toolsets, workspace_type)
            fn_name = func.__name__

            def _replace_policy(existing: WorkspaceMeta) -> str | None:
                # A producer refreshes its own workspaces and may displace a
                # saved pipeline result, never another producer's data.
                if existing.producer_fn not in (fn_name, "pipeline"):
                    return f"it was created by '{existing.producer_fn or 'another tool'}'"
                return None

            def _finish(raw_result: Any, call_kwargs: dict[str, Any]) -> Any:
                payload = transform(raw_result) if transform else raw_result
                ref = self.store_workspace(
                    payload,
                    workspace_type=workspace_type,
                    workspace_id=_resolve_key(key, call_kwargs),
                    toolsets=_toolsets,
                    producer_fn=fn_name,
                    producer_kwargs=call_kwargs,
                    meta=meta,
                    ttl=ttl,
                    replace_if=_replace_policy,
                )
                should_notify = notify if notify is not None else self.default_notify
                return ref.to_dict() if should_notify else ref

            def _bind(args: tuple, kwargs: dict) -> dict[str, Any]:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return dict(bound.arguments)

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    call_kwargs = _bind(args, kwargs)
                    return _finish(await func(*args, **kwargs), call_kwargs)

                wrapper: Callable[..., Any] = async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    call_kwargs = _bind(args, kwargs)
                    return _finish(func(*args, **kwargs), call_kwargs)

                wrapper = sync_wrapper

            wrapper.__workspace_type__ = workspace_type  # type: ignore[attr-defined]
            wrapper.__toolsets__ = [ts.name for ts in _toolsets]  # type: ignore[attr-defined]
            wrapper.__is_producer__ = True  # type: ignore[attr-defined]
            with self._lock:
                self._producers[fn_name] = {
                    "func": func,
                    "workspace_type": workspace_type,
                    "toolset_names": [t for ts in _toolsets for t in ts.tool_names],
                }
                self._producer_wrappers[fn_name] = wrapper
            return wrapper

        return decorator

    # Consumer decorator

    def consumer(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        workspace_type: str | None = None,
        produces: str | None = None,
        produces_toolsets: list[ToolSet | ToolSpec] | ToolSet | ToolSpec | None = None,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
        ttl: float | None = _SENTINEL,  # type: ignore[assignment]
        expose: bool = True,
    ) -> Callable[..., Any]:
        """
        Decorator for tools that read a workspace and may *produce* a new one.

        Use this when a tool reads from one workspace and writes a derived
        workspace (e.g. a "clean_papers" that stores cleaned results so the
        agent can paginate them independently).

        Consumers are registered with the Ctx: they appear in
        :meth:`get_tools` and can be called through :meth:`dispatch_tool_call`
        (pass ``expose=False`` for consumers only your own code calls).

        Args:
            workspace_type:    Expected type of the input ``workspace_id``.
                               Calls with a workspace of another type are
                               rejected with a type-mismatch error.
            produces:          Workspace type of the derived workspace.
            produces_toolsets: ToolSets (or nameless ToolSpecs, named after
                               *produces*) available on the derived workspace.
            name:              Tool name (default: the function name).
            meta:              Default metadata for workspaces this consumer
                               emits (merged with ``emit(meta=...)``).
            ttl:               Default TTL for emitted workspaces.
            expose:            Offer the consumer to the model (schema export
                               and dispatch).  ``False`` keeps it callable
                               from Python only.

        The wrapped function receives a :class:`ConsumerContext` via a
        parameter named ``ctx`` (or any parameter annotated with
        ``ConsumerContext``).  That parameter is hidden from the LLM's
        schema::

            @ctx.consumer(workspace_type="data", produces="clean")
            def clean(workspace_id: str, ctx: ConsumerContext):
                return ctx.emit([normalize(x) for x in ctx.get_items()])

        Works with both sync and async functions.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            orig_sig = inspect.signature(func)
            ctx_param: str | None = None
            for pname, param in orig_sig.parameters.items():
                if pname in ("ctx", "forge_ctx") or param.annotation in (
                    ConsumerContext,
                    "ConsumerContext",  # postponed annotations
                ):
                    ctx_param = pname
                    break
            bind_sig = (
                orig_sig.replace(
                    parameters=[
                        p for n, p in orig_sig.parameters.items() if n != ctx_param
                    ]
                )
                if ctx_param
                else orig_sig
            )
            out_toolsets = self._resolve_toolsets(produces_toolsets, produces)

            def _prepare(args: tuple, kwargs: dict) -> dict[str, Any]:
                bound = bind_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                call_kwargs = dict(bound.arguments)
                workspace_id = call_kwargs.get("workspace_id")
                if "workspace_id" in call_kwargs and not (
                    workspace_id is None
                    and bind_sig.parameters["workspace_id"].default is None
                ):
                    validate_workspace_id(workspace_id)
                    check_workspace(
                        self.store, workspace_id, expected_type=workspace_type
                    )
                    self.store.touch(workspace_id)
                if ctx_param:
                    call_kwargs[ctx_param] = ConsumerContext(
                        ctx=self,
                        input_workspace_id=workspace_id or "",
                        output_type=produces,
                        output_toolsets=out_toolsets,
                        producer_fn=tool_name,
                        default_meta=meta,
                        default_ttl=ttl,
                    )
                return call_kwargs

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await func(**_prepare(args, kwargs))

                wrapper: Callable[..., Any] = async_wrapper
            else:

                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return func(**_prepare(args, kwargs))

                wrapper = sync_wrapper

            wrapper.__is_consumer__ = True  # type: ignore[attr-defined]
            wrapper.__workspace_type__ = workspace_type  # type: ignore[attr-defined]
            if not expose:
                return wrapper
            with self._lock:
                self._consumers[tool_name] = {
                    "func": func,
                    "wrapper": wrapper,
                    "signature": bind_sig,
                    "workspace_type": workspace_type,
                    "produces": produces,
                }
            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator

    # Introspection & management

    def list_workspaces(self, workspace_type: str | None = None) -> list[str]:
        """Return workspace_ids, optionally filtered by type (all sessions)."""
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

        Raises:
            ValueError: if duplicate tool names are found across ToolSets.
        """
        schemas: list[dict[str, Any]] = []
        seen_names: dict[str, str] = {}  # tool_name → toolset_name
        with self._lock:
            toolsets = list(self._toolsets.values())
        for ts in toolsets:
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
        """
        with self._lock:
            producers = list(self._producers.items())
        return [
            _function_schema(
                name,
                info["func"],
                inspect.signature(info["func"]),
                (inspect.getdoc(info["func"]) or f"Producer '{name}'.")
                + "\n\nReturns a workspace notification (not raw data). "
                "Use the exploration tools to read the stored results.",
            )
            for name, info in producers
        ]

    def get_consumer_schemas(self) -> list[dict[str, Any]]:
        """Export function-calling schemas for all registered consumers."""
        with self._lock:
            consumers = list(self._consumers.items())
        schemas = []
        for name, info in consumers:
            doc = inspect.getdoc(info["func"]) or f"Consumer '{name}'."
            if info["produces"]:
                doc += (
                    f"\n\nStores its result as a new '{info['produces']}' "
                    "workspace and returns a workspace notification."
                )
            schemas.append(_function_schema(name, info["func"], info["signature"], doc))
        return schemas

    def get_tools(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """
        Export **all** tool schemas — producers, consumers and ToolSet
        tools — in one call (OpenAI function-calling format)::

            response = client.chat.completions.create(
                model=..., messages=messages, tools=ctx.get_tools(),
            )

        Raises:
            ValueError: if two tools share a name.
        """
        schemas = (
            self.get_producer_schemas()
            + self.get_consumer_schemas()
            + self.get_all_tool_schemas(workspace_id=workspace_id)
        )
        seen: set[str] = set()
        for schema in schemas:
            name = schema["function"]["name"]
            if name in seen:
                raise ValueError(
                    f"Duplicate tool name '{name}': producers, consumers and "
                    f"ToolSet tools must have unique names."
                )
            seen.add(name)
        return schemas

    def system_prompt(self, *, preamble: str = "") -> str:
        """
        Auto-generate a system prompt section that explains ctx tools.

        The generated prompt is derived from the registered producers,
        consumers and toolsets.  It covers the workspace pattern, available
        data sources, exploration tools, pipeline syntax (if registered),
        and error recovery.

        Args:
            preamble: Optional text prepended before the tool instructions.
                      Use this for your agent's role description.
        """
        lines: list[str] = []

        if preamble:
            lines.append(preamble.rstrip())
            lines.append("")

        lines.append(
            "## Data tools\n"
            "Some of your tools produce large results that are stored in "
            "workspaces instead of returned directly. When this happens, "
            "the response includes a `workspace_id`, an `item_schema` "
            "and exploration tools. Results are capped: on `has_more` or "
            "`truncated`, narrow the query or page further."
        )

        with self._lock:
            producers = list(self._producers.items())
            consumers = list(self._consumers.items())
            toolsets = list(self._toolsets.values())

        if producers:
            lines.append("")
            lines.append("### Data sources")
            for name, info in producers:
                doc = inspect.getdoc(info["func"]) or ""
                first_line = doc.split("\n")[0].strip() if doc else name
                ts_names = info.get("toolset_names", [])
                lines.append(f"- **{name}**: {first_line}")
                if ts_names:
                    tools_str = ", ".join(f"`{t}`" for t in ts_names[:6])
                    if len(ts_names) > 6:
                        tools_str += f" (+{len(ts_names) - 6} more)"
                    lines.append(f"  Exploration tools: {tools_str}")

        if consumers:
            lines.append("")
            lines.append("### Processing tools")
            for name, info in consumers:
                doc = inspect.getdoc(info["func"]) or ""
                first_line = doc.split("\n")[0].strip() if doc else name
                lines.append(f"- **{name}**: {first_line}")

        has_pipeline = has_search = has_filter = has_kv = False
        for ts in toolsets:
            for t_name in ts.tool_names:
                if t_name.endswith("_pipe"):
                    has_pipeline = True
                elif t_name.endswith("_search"):
                    has_search = True
                elif t_name.endswith("_filter_by"):
                    has_filter = True
                elif t_name.endswith("_get_keys"):
                    has_kv = True

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

        lines.append("")
        lines.append(
            "### Errors\n"
            "If a tool returns an `error` field, read the "
            "`suggested_action` for recovery guidance. "
            "Wrong workspace_id errors include a list of "
            "`available_workspaces`."
        )

        return "\n".join(lines)

    # Dispatch

    def dispatch_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        session_id: str | None = None,
    ) -> Any:
        """
        Route a tool call from an LLM response to the correct function.

        Handles producers, consumers and ToolSet tools.  Problems the LLM
        can fix — unknown tool, bad arguments, missing/expired/foreign
        workspace, failing producer — come back as a dict with an
        ``error`` and a ``suggested_action`` instead of raising.

        Args:
            tool_name:  The ``name`` field from the LLM's tool call.
            arguments:  The arguments dict (or its JSON string).
            session_id: Run the call inside this session
                        (see :mod:`ctxtual.session`).

        Async tools are run with :func:`asyncio.run` when no event loop is
        running.  Inside a running loop use :meth:`adispatch_tool_call`.

        Raises:
            RuntimeError: when an async tool is dispatched synchronously
                          from inside a running event loop.
        """
        with _maybe_session(session_id):
            prepared = self._prepare_call(tool_name, arguments)
            if isinstance(prepared, dict):
                return prepared
            target, args = prepared
            if inspect.iscoroutinefunction(target.fn):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:
                    raise RuntimeError(
                        f"Tool '{tool_name}' is async and an event loop is "
                        "running; use `await ctx.adispatch_tool_call(...)`."
                    )
            try:
                result = target.fn(**args)
                if inspect.isawaitable(result):
                    result = asyncio.run(_await(result))
            except Exception as exc:
                return self._handle_error(exc, tool_name, target)
            return self._limit_size(result, tool_name)

    async def adispatch_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        session_id: str | None = None,
    ) -> Any:
        """
        Async counterpart of :meth:`dispatch_tool_call`.

        Awaits async producers, consumers and tools; sync ones are called
        directly (wrap them in ``asyncio.to_thread`` yourself if they block).
        """
        with _maybe_session(session_id):
            prepared = self._prepare_call(tool_name, arguments)
            if isinstance(prepared, dict):
                return prepared
            target, args = prepared
            try:
                result = target.fn(**args)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                return self._handle_error(exc, tool_name, target)
            return self._limit_size(result, tool_name)

    def _prepare_call(
        self, tool_name: str, arguments: dict[str, Any] | str | None
    ) -> tuple[_Target, dict[str, Any]] | dict[str, Any]:
        """Resolve the target and validate arguments (or return an error)."""
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return {
                    "error": f"Arguments for '{tool_name}' are not valid JSON: {exc}",
                    "suggested_action": "Pass the arguments as a JSON object.",
                }
        if not isinstance(arguments, dict):
            return {
                "error": f"Arguments for '{tool_name}' must be an object.",
                "suggested_action": "Pass the arguments as a JSON object.",
            }

        target = self._resolve(tool_name, arguments)
        if isinstance(target, dict):
            return target
        try:
            target.signature.bind(**arguments)
        except TypeError as exc:
            params = {
                n: _annotation_name(p.annotation)
                for n, p in target.signature.parameters.items()
            }
            required = [
                n
                for n, p in target.signature.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            return {
                "error": f"Invalid arguments for '{tool_name}': {exc}",
                "expected_parameters": params,
                "required": required,
                "suggested_action": "Call the tool again with the expected parameters.",
            }
        return target, arguments

    def _resolve(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> _Target | dict[str, Any]:
        with self._lock:
            toolsets = list(self._toolsets.values())
            consumers = dict(self._consumers)
            producers = dict(self._producer_wrappers)
            producer_funcs = {n: i["func"] for n, i in self._producers.items()}

        matches = [ts for ts in toolsets if tool_name in ts._tools]
        if len(matches) > 1:
            # Disambiguate via workspace_id → workspace_type → ToolSet.name
            workspace_id = arguments.get("workspace_id")
            meta = (
                self.store.get_meta(workspace_id)
                if isinstance(workspace_id, str) and workspace_id
                else None
            )
            if meta is not None and not is_visible(meta):
                meta = None  # don't reveal other sessions' workspaces
            chosen = [ts for ts in matches if meta and ts.name == meta.workspace_type]
            if not chosen:
                ts_names = [ts.name for ts in matches]
                return {
                    "error": (
                        f"Tool '{tool_name}' exists in multiple ToolSets: "
                        f"{ts_names}. Cannot determine which to use."
                    ),
                    "suggested_action": (
                        "Pass a valid workspace_id so the correct ToolSet "
                        "can be inferred."
                    ),
                }
            matches = chosen[:1]
        if matches:
            ts = matches[0]
            return _Target(
                ts._tools[tool_name],
                inspect.signature(ts._raw_tools[tool_name]),
                "tool",
            )
        if tool_name in consumers:
            info = consumers[tool_name]
            return _Target(info["wrapper"], info["signature"], "consumer")
        if tool_name in producers:
            return _Target(
                producers[tool_name],
                inspect.signature(producer_funcs[tool_name]),
                "producer",
            )
        return {
            "error": f"Tool '{tool_name}' not found.",
            "available_tools": [t for ts in toolsets for t in ts.tool_names],
            "available_producers": list(producers),
            "available_consumers": list(consumers),
            "suggested_action": (
                "Check the tool name and try again. "
                "Use one of the available tools listed above."
            ),
        }

    def _limit_size(self, result: Any, tool_name: str) -> Any:
        if self.max_result_chars is None:
            return result
        try:
            encoded = (
                result
                if isinstance(result, str)
                else json.dumps(result, ensure_ascii=False, default=str)
            )
        except (TypeError, ValueError):
            return result  # not JSON-like; leave it to the caller
        if len(encoded) <= self.max_result_chars:
            return result
        return {
            "error": (
                f"The result of '{tool_name}' is too large to return "
                f"({len(encoded):,} characters, limit "
                f"{self.max_result_chars:,})."
            ),
            "truncated": True,
            "result_chars": len(encoded),
            "preview": encoded[: min(2_000, self.max_result_chars)],
            "suggested_action": (
                "Request less data: a smaller page or limit, fewer fields "
                "(select), or a more specific query."
            ),
        }

    def _handle_error(self, exc: Exception, tool_name: str, target: _Target) -> Any:
        if isinstance(exc, CtxtualError):
            return exc.to_llm_dict()
        # ToolSet tools already converted errors unless their ToolSet is
        # unsafe — in which case the caller asked for exceptions.
        if target.kind == "tool" or not self.safe:
            raise exc
        logger.exception("Tool '%s' raised", tool_name)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "tool": tool_name,
            "suggested_action": (
                "The tool failed. Check the arguments, or try a different "
                "approach if the problem persists."
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
        return f"Ctx(store={self.store!r}, toolsets={list(self._toolsets.keys())})"


# ConsumerContext — injected into @ctx.consumer functions


class ConsumerContext:
    """
    Provides a consumer function with store access and the ability to
    produce a derived workspace from its results.

    Injected automatically by :meth:`Ctx.consumer` into the parameter
    named ``ctx`` (or annotated with ``ConsumerContext``).

    For direct use in tests, construct with ``ctx_instance`` and
    ``input_workspace_id``::

        consumer_ctx = ConsumerContext(ctx_instance, "ws_1")
        result = my_consumer.__wrapped__("ws_1", ctx=consumer_ctx)
    """

    def __init__(
        self,
        ctx: Ctx,
        input_workspace_id: str = "",
        output_type: str | None = None,
        output_toolsets: list[ToolSet] | None = None,
        *,
        producer_fn: str = "",
        default_meta: dict[str, Any] | None = None,
        default_ttl: float | None | _SentinelType = _SENTINEL,
    ) -> None:
        self._ctx = ctx
        self.input_workspace_id = input_workspace_id
        self._output_type = output_type
        self._output_toolsets = output_toolsets or []
        self._producer_fn = producer_fn
        self._default_meta = default_meta or {}
        self._default_ttl = default_ttl

    @property
    def store(self) -> BaseStore:
        """The Ctx's storage backend."""
        return self._ctx.store

    def get_items(self, workspace_id: str | None = None) -> Any:
        """
        Get items from the input workspace, or from *workspace_id*.

        Other workspaces must be visible to the active session.
        """
        ws_id = workspace_id or self.input_workspace_id
        if workspace_id and workspace_id != self.input_workspace_id:
            meta = self.store.get_meta(workspace_id)
            if meta is None or not is_visible(meta):
                raise WorkspaceNotFoundError(workspace_id)
        return self.store.get_items(ws_id)

    def emit(
        self,
        payload: Any,
        *,
        workspace_type: str | None = None,
        workspace_id: str | None = None,
        toolsets: list[ToolSet] | None = None,
        meta: dict[str, Any] | None = None,
        ttl: float | None | _SentinelType = _SENTINEL,
    ) -> dict[str, Any]:
        """
        Store *payload* as a new workspace and return a ``WorkspaceRef`` dict.

        Call this inside a ``@ctx.consumer`` to produce a derived workspace
        from filtered / processed results.  ``ttl`` defaults to the
        consumer's ``ttl`` and then to ``Ctx.default_ttl``; pass ``None``
        for no expiry.
        """
        out_type = workspace_type or self._output_type or "derived"
        out_toolsets = toolsets or self._output_toolsets or None
        resolved_ttl = ttl if ttl is not _SENTINEL else self._default_ttl
        ref = self._ctx.store_workspace(
            payload,
            workspace_type=out_type,
            workspace_id=workspace_id,
            toolsets=out_toolsets,
            producer_fn=self._producer_fn,
            meta={**self._default_meta, **(meta or {})},
            ttl=resolved_ttl,
        )
        return ref.to_dict()


# Internal helpers for Ctx


@contextlib.contextmanager
def _maybe_session(session_id: str | None) -> Iterator[None]:
    if session_id is None:
        yield
    else:
        with session_scope(session_id):
            yield


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _replace_reason(
    existing: WorkspaceMeta,
    workspace_type: str,
    replace_if: ReplacePolicy | None,
) -> str | None:
    """Why *existing* may not be overwritten, or ``None`` if it may."""
    if not is_visible(existing):
        return "it belongs to another session"
    if existing.workspace_type != workspace_type:
        return f"it holds '{existing.workspace_type}' data"
    if replace_if is not None:
        return replace_if(existing)
    return None


def _validate_key_template(
    key: str | KeyFactory | None, sig: inspect.Signature, fn_name: str
) -> None:
    if not isinstance(key, str):
        return
    fields = {
        f.split(".")[0].split("[")[0]
        for _, f, _, _ in string.Formatter().parse(key)
        if f is not None
    }
    unknown = sorted(f for f in fields if f not in sig.parameters)
    if unknown or "" in fields:
        raise ValueError(
            f"key template {key!r} of producer '{fn_name}' references "
            f"{unknown or ['a positional field']}, which are not parameters "
            f"of the function."
        )


def _function_schema(
    name: str, func: Callable[..., Any], sig: inspect.Signature, doc: str
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    docstring_descs = _extract_param_descriptions(func)
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls") or param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue
        p_schema = _python_type_to_json_schema(param.annotation)
        p_schema["description"] = _build_param_description(
            param_name, param, docstring_descs
        )
        if param.default is not inspect.Parameter.empty and param.default is not None:
            p_schema["default"] = param.default
        properties[param_name] = p_schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return {
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


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    return getattr(annotation, "__name__", None) or str(annotation)


# Payload helpers


def _resolve_key(
    key: str | KeyFactory | None,
    call_kwargs: dict[str, Any],
) -> str | None:
    """Deterministic workspace_id from the key spec (``None`` = random)."""
    if key is None:
        return None
    if callable(key):
        return str(key(call_kwargs))
    return key.format(**call_kwargs)


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


def _schema_from_dicts(items: list[dict[str, Any]], total: int) -> dict[str, Any]:
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
