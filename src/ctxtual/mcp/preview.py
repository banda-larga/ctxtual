"""
Framework-independent helpers for offloading MCP tool results.

Nothing here imports FastMCP, so the logic that decides *what* the model sees
is unit-testable on its own.
"""

import json
from typing import Any

#: Characters a single string value may keep inside a preview.
PREVIEW_STRING_CHARS = 200
#: Elements a nested list may keep inside a preview.
PREVIEW_LIST_ITEMS = 5
#: Upper bound on the JSON size of the whole preview.
PREVIEW_MAX_CHARS = 2_000
#: Most fields described in the reference's schema.
SCHEMA_MAX_FIELDS = 50


def parse_text_payload(text: str) -> Any:
    """Return parsed JSON if *text* is a JSON document, else the text itself."""
    stripped = text.strip()
    if stripped[:1] in ("[", "{"):
        try:
            return json.loads(stripped)
        except ValueError:
            pass
    return text


def unwrap_structured(value: Any) -> Any:
    """Undo the ``{"result": ...}`` wrapper servers put around non-object output."""
    if isinstance(value, dict) and len(value) == 1 and "result" in value:
        return value["result"]
    return value


def shrink(value: Any, depth: int = 0) -> Any:
    """A size-bounded copy of *value* for previews (long strings/lists cut)."""
    if isinstance(value, str):
        if len(value) > PREVIEW_STRING_CHARS:
            return value[:PREVIEW_STRING_CHARS] + f"… [{len(value):,} chars]"
        return value
    if isinstance(value, list):
        if depth >= 3:
            return f"[list of {len(value)}]"
        items = [shrink(v, depth + 1) for v in value[:PREVIEW_LIST_ITEMS]]
        if len(value) > PREVIEW_LIST_ITEMS:
            items.append(f"… {len(value) - PREVIEW_LIST_ITEMS} more")
        return items
    if isinstance(value, dict):
        if depth >= 3:
            return f"{{object with {len(value)} keys}}"
        return {str(k): shrink(v, depth + 1) for k, v in value.items()}
    return value


def build_preview(payload: Any, *, max_items: int = 3) -> Any:
    """
    A small, representative slice of *payload* that fits in
    :data:`PREVIEW_MAX_CHARS`: the first items of a list, the top-level keys
    of an object, or the beginning of a text.
    """
    if isinstance(payload, str):
        return payload[: PREVIEW_MAX_CHARS // 2]
    if isinstance(payload, list):
        for n in range(min(max_items, len(payload)), 0, -1):
            preview = [shrink(item) for item in payload[:n]]
            if _json_len(preview) <= PREVIEW_MAX_CHARS:
                return preview
        return [_truncate_json(shrink(payload[0]))] if payload else []
    if isinstance(payload, dict):
        preview: dict[str, Any] = {}
        for key, value in payload.items():
            preview[str(key)] = shrink(value, depth=1)
            if _json_len(preview) > PREVIEW_MAX_CHARS:
                preview.pop(str(key))
                preview["…"] = f"{len(payload) - len(preview)} more keys"
                break
        return preview
    return payload


def compact_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Field name → type map (at most :data:`SCHEMA_MAX_FIELDS` fields)."""
    if not schema or "properties" not in schema:
        return None
    fields: dict[str, Any] = {}
    for name, prop in list(schema["properties"].items())[:SCHEMA_MAX_FIELDS]:
        fields[name] = prop.get("type", "any")
    if len(schema["properties"]) > SCHEMA_MAX_FIELDS:
        fields["…"] = f"{len(schema['properties']) - SCHEMA_MAX_FIELDS} more fields"
    return fields


def _json_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _truncate_json(value: Any) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= PREVIEW_MAX_CHARS:
        return value
    return encoded[:PREVIEW_MAX_CHARS] + "…"
