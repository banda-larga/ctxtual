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
            model="claude-sonnet-5-5", messages=messages, tools=tools, max_tokens=4096
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
    *,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Dispatch every ``tool_use`` block in *response* through the ctx and
    return ``tool_result`` content blocks for the next user message.

    Args:
        ctx:        The :class:`~ctxtual.Ctx` instance.
        response:   An Anthropic ``Message`` (SDK object or dict).
        session_id: Run the calls inside this session
                    (see :mod:`ctxtual.session`).

    Returns:
        A list of ``{"type": "tool_result", "tool_use_id": ..., "content": ...}``
        dicts.  Pass these as the ``content`` of the next ``role: "user"``
        message.
    """
    results: list[dict[str, Any]] = []
    for tu in _extract_tool_use(response):
        try:
            result = ctx.dispatch_tool_call(
                tu["name"], tu["input"], session_id=session_id
            )
        except Exception as exc:
            result = _exception_result(exc)
        results.append(_tool_result(tu, result))
    return results


async def ahandle_tool_use(
    ctx: "Ctx",
    response: Any,
    *,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Async version of :func:`handle_tool_use` (awaits async tools)."""
    results: list[dict[str, Any]] = []
    for tu in _extract_tool_use(response):
        try:
            result = await ctx.adispatch_tool_call(
                tu["name"], tu["input"], session_id=session_id
            )
        except Exception as exc:
            result = _exception_result(exc)
        results.append(_tool_result(tu, result))
    return results


# ── internal helpers ──────────────────────────────────────────────────────


def _exception_result(exc: Exception) -> dict[str, Any]:
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "suggested_action": "Check the tool name and parameters.",
    }


def _tool_result(tu: dict[str, Any], result: Any) -> dict[str, Any]:
    content = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, default=str)
    )
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tu["id"],
        "content": content,
    }
    if isinstance(result, dict) and "error" in result:
        block["is_error"] = True
    return block


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
