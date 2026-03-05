"""
LangChain / LangGraph integration for ctx.

Wraps all forge tools (producers + consumers) as LangChain
:class:`~langchain_core.tools.StructuredTool` objects that can be passed
directly to agents, chains, or graphs.

Requires ``langchain-core`` (``pip install langchain-core``).

Typical usage::

    from ctxtual import Forge, MemoryStore
    from ctxtual.integrations.langchain import to_langchain_tools

    forge = Forge(store=MemoryStore())
    # ... register producers and toolsets ...

    tools = to_langchain_tools(forge)
    # Pass `tools` to any LangChain agent/graph
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctxtual.forge import Forge


def to_langchain_tools(
    forge: Forge, *, workspace_id: str | None = None
) -> list[Any]:
    """
    Wrap all forge tools as LangChain ``StructuredTool`` objects.

    Each tool dispatches through ``forge.dispatch_tool_call()`` so workspace
    validation, type checking, and safe error handling all work as expected.

    Args:
        forge:        The :class:`~ctx.Forge` instance.
        workspace_id: Optional workspace ID to scope consumer tools to.

    Returns:
        A list of ``StructuredTool`` instances ready for LangChain agents.

    Raises:
        ImportError: If ``langchain-core`` is not installed.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise ImportError(
            "langchain-core is required for the LangChain integration. "
            "Install it with: pip install langchain-core"
        ) from None

    try:
        from pydantic import Field, create_model
    except ImportError:
        raise ImportError(
            "pydantic is required for the LangChain integration. "
            "Install it with: pip install pydantic"
        ) from None

    schemas = forge.get_tools(workspace_id=workspace_id)
    tools: list[StructuredTool] = []

    for schema in schemas:
        func_schema = schema["function"]
        tool_name = func_schema["name"]
        description = func_schema["description"]
        parameters = func_schema.get("parameters", {})

        args_model = _json_schema_to_pydantic(
            tool_name, parameters, create_model, Field
        )

        # Closure captures tool_name correctly
        def _make_fn(name: str, desc: str):  # noqa: E301
            def fn(**kwargs: Any) -> Any:
                return forge.dispatch_tool_call(name, kwargs)

            fn.__name__ = name
            fn.__doc__ = desc
            return fn

        tool = StructuredTool(
            name=tool_name,
            description=description,
            func=_make_fn(tool_name, description),
            args_schema=args_model,
        )
        tools.append(tool)

    return tools


# ── internal helpers ──────────────────────────────────────────────────────

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_schema_to_pydantic(
    name: str,
    schema: dict[str, Any],
    create_model: Any,
    field_factory: Any,
) -> type:
    """Build a Pydantic model from a JSON Schema ``properties`` dict."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        python_type = _JSON_TYPE_MAP.get(prop_schema.get("type", "string"), Any)
        description = prop_schema.get("description", prop_name)

        if prop_name in required:
            fields[prop_name] = (python_type, field_factory(description=description))
        else:
            default = prop_schema.get("default")
            fields[prop_name] = (
                python_type,
                field_factory(default=default, description=description),
            )

    # Pydantic model name: PascalCase from the tool name
    model_name = name.replace("_", " ").title().replace(" ", "") + "Input"
    return create_model(model_name, **fields)
