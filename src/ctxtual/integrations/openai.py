"""
OpenAI integration for ctx.

Provides helpers that bridge ``forge.get_tools()`` / ``forge.dispatch_tool_call()``
with the OpenAI Chat Completions API.  Works with both the ``openai`` SDK
response objects **and** raw dicts — no ``openai`` package import required.

Typical usage::

    from openai import OpenAI
    from ctxtual import Forge, MemoryStore
    from ctxtual.integrations.openai import (
        to_openai_tools,
        handle_tool_calls,
        has_tool_calls,
    )

    forge = Forge(store=MemoryStore())
    client = OpenAI()

    tools = to_openai_tools(forge)
    messages = [{"role": "user", "content": "Search for papers on AI"}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools
        )
        if not has_tool_calls(response):
            break
        messages.append(response.choices[0].message.model_dump())
        messages.extend(handle_tool_calls(forge, response))
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctxtual.forge import Forge


def to_openai_tools(
    forge: Forge, *, workspace_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Export all forge tools in OpenAI function-calling format.

    This is an alias for ``forge.get_tools()`` — provided for symmetry
    with the other integration modules.
    """
    return forge.get_tools(workspace_id=workspace_id)


def has_tool_calls(response: Any) -> bool:
    """
    Return ``True`` if the response contains tool calls.

    Accepts an ``openai`` SDK ``ChatCompletion`` object or a raw dict.
    """
    # SDK object
    if hasattr(response, "choices") and not isinstance(response, dict):
        choices = response.choices
        if choices and hasattr(choices[0], "message"):
            tc = getattr(choices[0].message, "tool_calls", None)
            return bool(tc)
        return False

    # Raw dict
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            tc = choices[0].get("message", {}).get("tool_calls")
            return bool(tc)

    return False


def handle_tool_calls(
    forge: Forge,
    response: Any,
    *,
    max_content_length: int | None = None,
) -> list[dict[str, Any]]:
    """
    Dispatch every tool call in *response* through the forge and return
    ``role: "tool"`` messages ready to append to the conversation.

    Args:
        forge:              The :class:`~ctx.Forge` instance.
        response:           An OpenAI ``ChatCompletion`` (SDK object or dict).
        max_content_length: If set, truncate each tool result string to this
                            many characters (useful for staying within token
                            budgets).

    Returns:
        A list of ``{"role": "tool", "tool_call_id": ..., "content": ...}``
        dicts — one per tool call.
    """
    tool_calls = _extract_tool_calls(response)
    results: list[dict[str, Any]] = []

    for tc in tool_calls:
        args = tc["arguments"]
        if isinstance(args, str):
            args = json.loads(args)

        try:
            result = forge.dispatch_tool_call(tc["name"], args)
        except Exception as exc:
            result = {
                "error": f"{type(exc).__name__}: {exc}",
                "suggested_action": "Check the tool name and parameters.",
            }

        content = (
            json.dumps(result, ensure_ascii=False, default=str)
            if not isinstance(result, str)
            else result
        )
        if max_content_length and len(content) > max_content_length:
            content = content[: max_content_length - 20] + '… [truncated]"}'

        results.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": content,
            }
        )

    return results


# ── internal helpers ──────────────────────────────────────────────────────


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract tool calls from an SDK object or raw dict."""
    # SDK object
    if hasattr(response, "choices") and not isinstance(response, dict):
        choices = response.choices
        if choices and hasattr(choices[0], "message"):
            msg = choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in msg.tool_calls
                ]
        return []

    # Raw dict
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            tool_calls = choices[0].get("message", {}).get("tool_calls", [])
            return [
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                for tc in tool_calls
            ]

    return []
