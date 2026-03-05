"""
ToolSet — a named collection of consumer tools scoped to a workspace type.

A ToolSet does three things:

1. Registers consumer functions and wraps them with workspace validation.
2. Exposes a list of tool names so producers can reference them in notifications.
3. Provides a :meth:`bind` method that returns pre-filled callables — useful
   when you want to give an agent a concrete handle without forcing it to pass
   ``workspace_id`` every time.
"""

import functools
import inspect
import types
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin

from ctxtual.exceptions import (
    WorkspaceExpiredError,
    WorkspaceNotFoundError,
    WorkspaceTypeMismatchError,
)
from ctxtual.types import WorkspaceMeta


class ToolSet:
    """
    A logical group of tools that operate on a specific workspace type.

    Args:
        name:            Identifier for this toolset (typically matches the
                         ``workspace_type`` it will operate on).
        enforce_type:    If ``True`` (default) consumer tools will raise
                         :exc:`WorkspaceTypeMismatchError` when called with a
                         workspace_id whose type doesn't match *name*.
        safe:            If ``True`` (default), tool exceptions are caught and
                         returned as ``{"error": ...}`` dicts instead of
                         propagating up and crashing the agent loop.

    Usage::

        papers = ToolSet("papers")

        @papers.tool
        def paginate(workspace_id: str, page: int = 0, size: int = 10) -> list:
            items = papers.store.get_items(workspace_id)
            return items[page * size : (page + 1) * size]
    """

    def __init__(
        self,
        name: str,
        *,
        enforce_type: bool = True,
        safe: bool = True,
        data_shape: str | None = None,
    ) -> None:
        self.name = name
        self.enforce_type = enforce_type
        self.safe = safe
        self.data_shape = data_shape
        self._tools: dict[str, Callable[..., Any]] = {}
        self._raw_tools: dict[str, Callable[..., Any]] = {}
        self._output_hints: dict[str, str | None] = {}
        self._schema_extras: dict[str, dict[str, dict[str, Any]]] = {}
        self._tool_shapes: dict[str, str] = {}
        # Injected by Forge.register_toolset()
        self._store: Any | None = None

    # Store access

    @property
    def store(self) -> Any:
        """Return the attached store, raising if the toolset isn't wired yet."""
        if self._store is None:
            raise RuntimeError(
                f"ToolSet '{self.name}' has not been attached to a Forge / store. "
                "Pass it to Forge() or call forge.register_toolset(toolset)."
            )
        return self._store

    # Decorator

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        validate_workspace: bool = True,
        output_hint: str | None = None,
        schema_extra: dict[str, dict[str, Any]] | None = None,
    ) -> Callable[..., Any]:
        """
        Register a consumer function as a tool in this ToolSet.

        The decorated function **must** accept ``workspace_id: str`` as its
        first parameter.  The wrapper will:

        - Verify the workspace exists in the store.
        - Optionally verify its type matches this toolset's *name*.
        - Check TTL expiry.
        - Forward the call to the original function.
        - If ``safe=True`` on the ToolSet, catch exceptions and return a
          structured error dict.
        - If ``output_hint`` is set, wrap the result with a contextual hint
          that tells the LLM what to do next — making tools self-describing.

        Args:
            name:               Override the tool name (default: function name).
            validate_workspace: Whether to validate workspace existence/type.
            output_hint:        A short instruction appended to the tool result
                                when returned to the LLM. This makes tools
                                self-describing without requiring system prompt
                                boilerplate. Supports ``{workspace_id}``
                                placeholder.
            schema_extra:       Per-parameter JSON Schema overrides.  Maps
                                parameter names to schema dicts that are merged
                                into the auto-generated schema.  Use to add
                                ``examples``, ``items`` details, ``enum``, etc.
                                Example: ``{"steps": {"examples": [[{"limit": 5}]]}}``

        Can be used as ``@toolset.tool`` or ``@toolset.tool(name="x")``.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            self._raw_tools[tool_name] = func
            self._output_hints[tool_name] = output_hint
            self._schema_extras[tool_name] = schema_extra or {}
            self._tool_shapes[tool_name] = self.data_shape

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if args and params and params[0] == "workspace_id":
                    workspace_id = args[0]
                else:
                    workspace_id = kwargs.get("workspace_id")

                if workspace_id is not None and validate_workspace:
                    meta: WorkspaceMeta | None = self._store.get_meta(workspace_id)
                    if meta is None:
                        raise WorkspaceNotFoundError(
                            workspace_id,
                            available=self._store.list_workspaces(self.name)
                            if self._store
                            else [],
                        )
                    if meta.is_expired:
                        self._store.drop_workspace(workspace_id)
                        raise WorkspaceExpiredError(
                            workspace_id,
                            producer_fn=meta.producer_fn,
                        )
                    if self.enforce_type and meta.workspace_type != self.name:
                        raise WorkspaceTypeMismatchError(
                            workspace_id,
                            self.name,
                            meta.workspace_type,
                            matching_workspaces=self._store.list_workspaces(
                                self.name
                            )
                            if self._store
                            else [],
                        )
                    meta.touch()

                    # Validate data shape matches toolset expectation
                    if (
                        self.data_shape
                        and meta.data_shape
                        and meta.data_shape != self.data_shape
                    ):
                        expected = self.data_shape
                        actual = meta.data_shape
                        _shape_tools = {
                            "list": "paginate, search, filter_by",
                            "dict": "get_keys, get_value",
                        }
                        return {
                            "error": (
                                f"Data shape mismatch: this tool expects "
                                f"'{expected}' data but workspace "
                                f"'{workspace_id}' contains '{actual}' data."
                            ),
                            "expected_shape": expected,
                            "actual_shape": actual,
                            "workspace_id": workspace_id,
                            "suggested_action": (
                                f"This workspace stores {actual} data. "
                                f"Use tools designed for {actual} data "
                                f"(e.g. {_shape_tools.get(actual, actual + '-compatible tools')}) "
                                f"instead."
                            ),
                        }
                try:
                    result = func(*args, **kwargs)
                except (
                    WorkspaceNotFoundError,
                    WorkspaceExpiredError,
                    WorkspaceTypeMismatchError,
                ):
                    raise  # always propagate ctx validation errors
                except Exception as exc:
                    if not self.safe:
                        raise
                    return {
                        "error": f"{type(exc).__name__}: {exc}",
                        "tool": tool_name,
                        "workspace_id": workspace_id,
                        "suggested_action": (
                            "Check the parameters and try again. "
                            "Use workspace_id from a previous producer call."
                        ),
                    }

                # Apply output hint if configured — always use a
                # consistent envelope so the result shape never mutates.
                hint = output_hint
                if hint and workspace_id:
                    hint = hint.replace("{workspace_id}", str(workspace_id))
                if hint:
                    return {"result": result, "_hint": hint}

                return result

            wrapper.__toolset__ = self.name  # type: ignore[attr-defined]
            wrapper.__tool_name__ = tool_name  # type: ignore[attr-defined]
            self._tools[tool_name] = wrapper
            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator

    # Introspection

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return list(self._tools.keys())

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        """Snapshot of all registered tools (name → wrapped callable)."""
        return dict(self._tools)

    def get_tool(self, name: str) -> Callable[..., Any]:
        """Retrieve a single tool by name, or raise ``KeyError``."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in ToolSet '{self.name}'.")
        return self._tools[name]

    # Schema export — for LLM function-calling integration

    def to_tool_schemas(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """
        Export all tools as OpenAI-compatible function-calling schemas.

        Each entry is a ``{"type": "function", "function": {...}}`` dict
        that can be passed directly to ``openai.chat.completions.create(tools=...)``,
        Anthropic's tool_use, or any framework that accepts the same format.

        Args:
            workspace_id: If given, the schema description mentions this
                          workspace_id to help the LLM route calls correctly.
        """
        schemas: list[dict[str, Any]] = []
        for tool_name, raw_fn in self._raw_tools.items():
            sig = inspect.signature(raw_fn)
            doc = inspect.getdoc(raw_fn) or f"Tool '{tool_name}'."
            if workspace_id:
                doc += f"\n\nUse with workspace_id='{workspace_id}'."

            properties: dict[str, Any] = {}
            required: list[str] = []
            docstring_descs = _extract_param_descriptions(raw_fn)
            extras = self._schema_extras.get(tool_name, {})

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                p_schema: dict[str, Any] = _python_type_to_json_schema(param.annotation)
                p_schema["description"] = _build_param_description(
                    param_name, param, docstring_descs
                )
                if (
                    param.default is not inspect.Parameter.empty
                    and param.default is not None
                ):
                    p_schema["default"] = param.default
                # Merge schema_extra overrides (examples, items, enum, etc.)
                if param_name in extras:
                    p_schema.update(extras[param_name])
                properties[param_name] = p_schema

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
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

    # Binding

    def bind(self, workspace_id: str) -> "BoundToolSet":
        """
        Return a view where ``workspace_id`` is pre-filled on every tool.

        Useful for providing an agent a concrete handle without requiring
        it to remember the workspace_id on every call.
        """
        return BoundToolSet(self, workspace_id)

    def __repr__(self) -> str:
        return f"ToolSet(name={self.name!r}, tools={self.tool_names})"


class ToolSpec:
    """
    Deferred toolset — created when a factory is called without a name.

    Materialized into a real :class:`ToolSet` when bound to a
    ``workspace_type`` by :meth:`Forge.producer`.

    Usage::

        from ctxtual.utils import paginator, text_search

        @forge.producer("papers", toolsets=[
            paginator(forge),
            text_search(forge, fields=["title"]),
        ])
        def fetch(query: str): ...
    """

    def __init__(
        self,
        factory: Callable[..., "ToolSet"],
        forge: Any,
        **kwargs: Any,
    ) -> None:
        self._factory = factory
        self._forge = forge
        self._kwargs = kwargs

    def materialize(self, name: str) -> "ToolSet":
        """Create the real ToolSet by calling the factory with the given name."""
        return self._factory(self._forge, name, **self._kwargs)

    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"ToolSpec({self._factory.__name__}{', ' + kw if kw else ''})"


class BoundToolSet:
    """
    A ToolSet with a fixed ``workspace_id`` — every tool is partially applied.

    Created via :meth:`ToolSet.bind`.
    """

    def __init__(self, toolset: ToolSet, workspace_id: str) -> None:
        self._toolset = toolset
        self.workspace_id = workspace_id
        for tool_name, fn in toolset.tools.items():
            bound = functools.partial(fn, workspace_id=workspace_id)
            bound.__doc__ = fn.__doc__  # type: ignore[attr-defined]
            setattr(self, tool_name, bound)

    @property
    def name(self) -> str:
        return self._toolset.name

    @property
    def tool_names(self) -> list[str]:
        return self._toolset.tool_names

    def to_tool_schemas(self) -> list[dict[str, Any]]:
        """Export schemas for tools with ``workspace_id`` already documented."""
        return self._toolset.to_tool_schemas(workspace_id=self.workspace_id)

    def __repr__(self) -> str:
        return (
            f"BoundToolSet(name={self.name!r}, " f"workspace_id={self.workspace_id!r})"
        )


# Schema helpers


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """
    Map a Python type annotation to a JSON Schema dict.

    Handles basic types, Optional, Union, Literal, list[X], dict, tuple, etc.
    Falls back to ``{"type": "string"}`` for unrecognised annotations.
    """
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    # typing.Any → no type constraint
    if annotation is Any:
        return {}

    # NoneType
    if annotation is type(None):
        return {"type": "null"}

    # Basic types
    basic: dict[type, dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    if annotation in basic:
        return dict(basic[annotation])

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional  (typing.Union or  X | Y)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] → schema of X
            return _python_type_to_json_schema(non_none[0])
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}

    # Literal["a", "b", ...]
    if origin is Literal:
        values = list(args)
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        return {"enum": values}

    # list[X]
    if origin is list:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    # dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # tuple[X, Y, ...]
    if origin is tuple:
        schema = {"type": "array"}
        if args:
            schema["prefixItems"] = [
                _python_type_to_json_schema(a) for a in args
            ]
        return schema

    # set[X] / frozenset[X]
    if origin in (set, frozenset):
        schema = {"type": "array", "uniqueItems": True}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    # Fallback
    return {"type": "string"}


# ── Parameter description extraction ──────────────────────────────────────

# Well-known parameter names → meaningful descriptions
_WELL_KNOWN_PARAMS: dict[str, str] = {
    "workspace_id": "Identifier of the workspace to operate on.",
    "page": "Zero-based page number.",
    "size": "Number of items per page.",
    "query": "Search query string.",
    "field": "Name of the field to operate on.",
    "value": "Value to compare or filter against.",
    "index": "Zero-based item index.",
    "key": "Dictionary key to look up.",
    "limit": "Maximum number of results to return.",
    "offset": "Number of items to skip from the start.",
    "max_results": "Maximum number of matching items to return.",
    "case_sensitive": "Whether the search should be case-sensitive.",
    "descending": "If true, sort in descending order.",
    "operator": "Comparison operator (eq, ne, lt, lte, gt, gte, contains, startswith).",
    "start": "Start index (inclusive).",
    "end": "End index (exclusive).",
    "max_values": "Maximum number of distinct values to return.",
    "fields": "List of field names to include.",
    "top_k": "Number of top results to retrieve.",
}


def _extract_param_descriptions(func: Callable[..., Any]) -> dict[str, str]:
    """
    Extract parameter descriptions from the function's docstring.

    Supports Google style (``Args:`` section) and Sphinx style
    (``:param name: desc``).  Returns a dict mapping parameter names
    to their description strings.
    """
    doc = inspect.getdoc(func)
    if not doc:
        return {}

    descriptions: dict[str, str] = {}
    lines = doc.split("\n")

    # ── Try Sphinx style first: ":param name: desc" ──
    for line in lines:
        s = line.strip()
        if s.startswith(":param ") and ":" in s[7:]:
            rest = s[7:]
            name, _, desc = rest.partition(":")
            descriptions[name.strip()] = desc.strip()
    if descriptions:
        return descriptions

    # ── Google style: "Args:\n    name: desc" ──
    in_args = False
    current_param: str | None = None
    current_lines: list[str] = []
    param_indent = 0

    for line in lines:
        s = line.strip()

        # Detect "Args:" header
        if not in_args:
            if s.lower().rstrip(":") in ("args", "arguments", "parameters"):
                in_args = True
            continue

        # End of args section
        if s == "":
            if current_param:
                descriptions[current_param] = " ".join(current_lines).strip()
                current_param = None
                current_lines = []
            continue

        # Non-indented line → new section header → end of args
        leading = len(line) - len(line.lstrip())
        if leading == 0 and s.endswith(":"):
            if current_param:
                descriptions[current_param] = " ".join(current_lines).strip()
            break

        # New param line: "    name: desc" or "    name (type): desc"
        if ":" in s:
            before, _, after = s.partition(":")
            maybe_name = before.split("(")[0].strip()
            if maybe_name.isidentifier() and leading <= param_indent + 4:
                if current_param:
                    descriptions[current_param] = " ".join(current_lines).strip()
                current_param = maybe_name
                current_lines = [after.strip()] if after.strip() else []
                param_indent = leading
                continue

        # Continuation line
        if current_param and leading > param_indent:
            current_lines.append(s)

    if current_param:
        descriptions[current_param] = " ".join(current_lines).strip()

    return descriptions


def _build_param_description(
    param_name: str,
    param: inspect.Parameter,
    docstring_descs: dict[str, str],
) -> str:
    """
    Build a parameter description using (in priority order):
    1. Docstring-extracted description
    2. Well-known parameter name
    3. Generated fallback from name + type + default
    """
    # 1. Docstring
    if param_name in docstring_descs:
        desc = docstring_descs[param_name]
        # Append default info if not already mentioned
        if (
            param.default is not inspect.Parameter.empty
            and "default" not in desc.lower()
        ):
            desc += f" Defaults to {param.default!r}."
        return desc

    # 2. Well-known name
    if param_name in _WELL_KNOWN_PARAMS:
        desc = _WELL_KNOWN_PARAMS[param_name]
        if param.default is not inspect.Parameter.empty:
            desc += f" Defaults to {param.default!r}."
        return desc

    # 3. Fallback: humanise the parameter name
    human = param_name.replace("_", " ")
    annotation = param.annotation
    if annotation is not inspect.Parameter.empty:
        type_hint = getattr(annotation, "__name__", str(annotation))
        desc = f"The {human} ({type_hint})."
    else:
        desc = f"The {human}."

    if param.default is not inspect.Parameter.empty:
        desc += f" Defaults to {param.default!r}."
    return desc
