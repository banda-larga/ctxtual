"""
Framework-independent helpers for offloading MCP tool results.

Nothing here imports FastMCP, so the logic that decides *what* the model sees
is unit-testable on its own.
"""

import json
from typing import Any

#: Characters a single string value may keep inside a preview.
PREVIEW_STRING_CHARS = 120
#: Elements a nested list may keep inside a preview.
PREVIEW_LIST_ITEMS = 5
#: Upper bound on the JSON size of the whole preview.
PREVIEW_MAX_CHARS = 2_000
#: Most fields described in the reference's schema.
SCHEMA_MAX_FIELDS = 50


def parse_text_payload(text: str) -> Any:
    """
    Parse *text* into data where it is unambiguous: a JSON document, or
    JSON Lines (one JSON object per line, as many logs and exports are).
    Anything else is returned as text.
    """
    stripped = text.strip()
    if stripped[:1] in ("[", "{"):
        try:
            return json.loads(stripped)
        except ValueError:
            pass
        lines = [line for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1 and all(line.lstrip().startswith("{") for line in lines):
            try:
                rows = [json.loads(line) for line in lines]
            except ValueError:
                return text
            if all(isinstance(row, dict) for row in rows):
                return rows
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


def cut_strings(
    value: Any, limit: int, pointer: Any = None, path: str = ""
) -> Any:
    """
    Copy of *value* with strings longer than *limit* cut.

    ``pointer(path, length)`` returns the note appended to a cut string (how
    to read the rest); *path* is the value's path inside an item, as
    :func:`resolve_path` reads it (``""`` for the item itself).
    """
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        note = pointer(path, len(value)) if pointer else f"{len(value):,} chars in total"
        return f"{value[:limit]}… [{note}]"
    if isinstance(value, dict):
        return {
            k: cut_strings(v, limit, pointer, _join(path, str(k)))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [cut_strings(v, limit, pointer, _join(path, str(i))) for i, v in enumerate(value)]
    return value


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


#: Returned by :func:`resolve_path` when the path does not exist.
MISSING = object()


def resolve_path(value: Any, path: str) -> Any:
    """
    Follow a dotted *path* (``"author.name"``, ``"comments.0.body"``) into
    *value*.  Keys that themselves contain dots (``"a.b"``) are matched
    greedily against the real keys.  Returns :data:`MISSING` if the path
    doesn't exist (a key that holds ``None`` returns ``None``).
    """
    if path == "":
        return value
    parts = path.split(".")

    def walk(node: Any, i: int) -> Any:
        if i == len(parts):
            return node
        if isinstance(node, dict):
            for j in range(len(parts), i, -1):  # longest key first
                key = ".".join(parts[i:j])
                if key in node:
                    found = walk(node[key], j)
                    if found is not MISSING:
                        return found
            return MISSING
        if isinstance(node, list) and parts[i].isdigit():
            index = int(parts[i])
            return walk(node[index], i + 1) if index < len(node) else MISSING
        return MISSING

    return walk(value, 0)
