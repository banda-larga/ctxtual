"""
OpenAI integration for ctx.

Provides helpers that bridge ``ctx.get_tools()`` / ``ctx.dispatch_tool_call()``
with the OpenAI Chat Completions API.  Works with both the ``openai`` SDK
response objects **and** raw dicts — no ``openai`` package import required.

Typical usage::

    from openai import OpenAI
    from ctxtual import Ctx, MemoryStore
    from ctxtual.integrations.openai import (
        to_openai_tools,
        handle_tool_calls,
        has_tool_calls,
    )

    ctx = Ctx(store=MemoryStore())
    client = OpenAI()

    tools = to_openai_tools(ctx)
    messages = [{"role": "user", "content": "Search for papers on AI"}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools
        )
        if not has_tool_calls(response):
            break
        messages.append(response.choices[0].message.model_dump())
        messages.extend(handle_tool_calls(ctx, response))
"""

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctxtual.ctx import Ctx


def to_openai_tools(
    ctx: "Ctx", *, workspace_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Export all ctx tools in OpenAI function-calling format.

    This is an alias for ``ctx.get_tools()`` — provided for symmetry
    with the other integration modules.
    """
    return ctx.get_tools(workspace_id=workspace_id)


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
    ctx: "Ctx",
    response: Any,
    *,
    max_content_length: int | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Dispatch every tool call in *response* through the ctx and return
    ``role: "tool"`` messages ready to append to the conversation.

    Args:
        ctx:                The :class:`~ctxtual.Ctx` instance.
        response:           An OpenAI ``ChatCompletion`` (SDK object or dict).
        max_content_length: If set, truncate each tool result string to this
                            many characters (useful for staying within token
                            budgets).
        session_id:         Run the calls inside this session
                            (see :mod:`ctxtual.session`).

    Returns:
        A list of ``{"role": "tool", "tool_call_id": ..., "content": ...}``
        dicts — one per tool call.
    """
    return [
        _tool_message(
            tc,
            _safe_call(lambda tc=tc: ctx.dispatch_tool_call(
                tc["name"], tc["arguments"], session_id=session_id
            )),
            max_content_length,
        )
        for tc in _extract_tool_calls(response)
    ]


async def ahandle_tool_calls(
    ctx: "Ctx",
    response: Any,
    *,
    max_content_length: int | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Async version of :func:`handle_tool_calls` (awaits async tools)."""
    messages = []
    for tc in _extract_tool_calls(response):
        try:
            result = await ctx.adispatch_tool_call(
                tc["name"], tc["arguments"], session_id=session_id
            )
        except Exception as exc:
            result = _exception_result(exc)
        messages.append(_tool_message(tc, result, max_content_length))
    return messages


# ── internal helpers ──────────────────────────────────────────────────────


def _exception_result(exc: Exception) -> dict[str, Any]:
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "suggested_action": "Check the tool name and parameters.",
    }


def _safe_call(call: Any) -> Any:
    try:
        return call()
    except Exception as exc:
        return _exception_result(exc)


def _tool_message(
    tc: dict[str, Any], result: Any, max_content_length: int | None
) -> dict[str, Any]:
    content = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, default=str)
    )
    if max_content_length and len(content) > max_content_length:
        marker = f"\n…[truncated {len(content) - max_content_length:,} characters]"
        if len(marker) < max_content_length:
            content = content[: max_content_length - len(marker)] + marker
        else:
            content = content[:max_content_length]
    return {"role": "tool", "tool_call_id": tc["id"], "content": content}


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
