"""
Anthropic integration for ctx.

Converts tool schemas from OpenAI format to Anthropic's ``tool_use`` format
and dispatches tool results back.  Works with both the ``anthropic`` SDK
response objects **and** raw dicts — no ``anthropic`` package import required.

Typical usage::

    from anthropic import Anthropic
    from ctxtual import Ctx, MemoryStore
    from ctxtual.integrations.anthropic import (
        to_anthropic_tools,
        handle_tool_use,
        has_tool_use,
    )

    ctx = Ctx(store=MemoryStore())
    client = Anthropic()

    tools = to_anthropic_tools(ctx)
    messages = [{"role": "user", "content": "Search for papers on AI"}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514", messages=messages, tools=tools, max_tokens=4096
        )
        if not has_tool_use(response):
            break
        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {"role": "user", "content": handle_tool_use(ctx, response)}
        )
"""

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctxtual.ctx import Ctx


def to_anthropic_tools(
    ctx: "Ctx", *, workspace_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Export all ctx tools in Anthropic tool definition format.

    Anthropic uses a flat structure with ``input_schema`` instead of
    OpenAI's nested ``function.parameters``.
    """
    openai_tools = ctx.get_tools(workspace_id=workspace_id)
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in openai_tools
    ]


def has_tool_use(response: Any) -> bool:
    """
    Return ``True`` if the response contains ``tool_use`` content blocks.

    Accepts an ``anthropic`` SDK response object or a raw dict.
    """
    for block in _get_content_blocks(response):
        btype = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if btype == "tool_use":
            return True
    return False


def handle_tool_use(
    ctx: "Ctx",
    response: Any,
) -> list[dict[str, Any]]:
    """
    Dispatch every ``tool_use`` block in *response* through the ctx and
    return ``tool_result`` content blocks for the next user message.

    Args:
        ctx:    The :class:`~ctx.Ctx` instance.
        response: An Anthropic ``Message`` (SDK object or dict).

    Returns:
        A list of ``{"type": "tool_result", "tool_use_id": ..., "content": ...}``
        dicts.  Pass these as the ``content`` of the next ``role: "user"``
        message.
    """
    tool_uses = _extract_tool_use(response)
    results: list[dict[str, Any]] = []

    for tu in tool_uses:
        input_data = tu["input"]
        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        try:
            result = ctx.dispatch_tool_call(tu["name"], input_data)
            content = (
                json.dumps(result, ensure_ascii=False, default=str)
                if not isinstance(result, str)
                else result
            )
            # Detect error dicts returned by dispatch_tool_call
            is_error = isinstance(result, dict) and "error" in result
        except Exception as exc:
            content = json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "suggested_action": "Check the tool name and parameters.",
                }
            )
            is_error = True

        results.append(
            {
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": content,
                **({"is_error": True} if is_error else {}),
            }
        )

    return results


# ── internal helpers ──────────────────────────────────────────────────────


def _get_content_blocks(response: Any) -> list[Any]:
    """Get content blocks from an SDK object or raw dict."""
    if hasattr(response, "content") and not isinstance(response, dict):
        return response.content or []
    if isinstance(response, dict):
        return response.get("content", [])
    return []


def _extract_tool_use(response: Any) -> list[dict[str, Any]]:
    """Extract tool_use blocks from content."""
    tool_uses: list[dict[str, Any]] = []

    for block in _get_content_blocks(response):
        # SDK object
        if hasattr(block, "type") and block.type == "tool_use":
            tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
        # Raw dict
        elif isinstance(block, dict) and block.get("type") == "tool_use":
            tool_uses.append(
                {
                    "id": block["id"],
                    "name": block["name"],
                    "input": block["input"],
                }
            )

    return tool_uses
