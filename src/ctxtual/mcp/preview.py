"""
Framework-independent helpers for offloading MCP tool results.

Nothing here imports FastMCP, so the logic that decides *what* the model sees
is unit-testable on its own.
"""

import ast
import json
import re
from typing import Any

#: Characters a single string value may keep inside a preview.
PREVIEW_STRING_CHARS = 120
#: Elements a nested list may keep inside a preview.
PREVIEW_LIST_ITEMS = 5
#: Upper bound on the JSON size of the whole preview.
PREVIEW_MAX_CHARS = 2_000
#: Most fields described in the reference's schema.
SCHEMA_MAX_FIELDS = 50
#: Largest text parsed as a Python literal (``str(rows)`` output); parsing
#: costs about 100 bytes of memory per character.
LITERAL_MAX_CHARS = 2_000_000
_PYTHON_ONLY = re.compile(r"\b(?:None|True|False)\b|'")
_JSON_ONLY = re.compile(r"\b(?:null|true|false)\b(?=\s*[,\]}])")


def parse_text_payload(text: str) -> Any:
    """
    Parse *text* into data where it is unambiguous: a JSON document, or
    JSON Lines (one JSON object per line, as many logs and exports are), or
    a Python list/dict literal (servers that return ``str(rows)``).  Anything
    else is returned as text.
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
                rows = []
            if rows and all(isinstance(row, dict) for row in rows):
                return rows
        # Only text that looks like Python, not like (cut-off) JSON.
        if (
            len(stripped) <= LITERAL_MAX_CHARS
            and _PYTHON_ONLY.search(stripped)
            and not _JSON_ONLY.search(stripped)
        ):
            try:  # literal_eval only builds literals, it runs no code
                value = ast.literal_eval(stripped)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                return text
            if isinstance(value, (list, dict)):
                return _json_safe(value)
    return text


def _json_safe(value: Any) -> Any:
    """Python values as JSON would carry them: str keys, lists for tuples and sets."""
    if isinstance(value, dict):
        return {
            k
            if isinstance(k, str)
            else json.dumps(k)
            if isinstance(k, (int, float, bool)) or k is None
            else str(k): _json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, complex):
        return str(value)
    return value


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
    value: Any,
    limit: int,
    pointer: Any = None,
    path: str = "",
    max_items: int | None = None,
) -> Any:
    """
    Copy of *value* with strings longer than *limit* cut, and nested lists
    longer than *max_items* (if given) cut to their first elements.

    ``pointer(path, length, unit)`` returns the note appended to a cut value
    (how to read the rest; *unit* is ``"chars"`` or ``"items"``); *path* is
    the value's path inside an item, as :func:`resolve_path` reads it
    (``""`` for the item itself).
    """
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        note = (
            pointer(path, len(value), "chars")
            if pointer
            else f"{len(value):,} chars in total"
        )
        return f"{value[:limit]}… [{note}]"
    if isinstance(value, dict):
        return {
            k: cut_strings(v, limit, pointer, _join(path, str(k)), max_items)
            for k, v in value.items()
        }
    if isinstance(value, list):
        shown = (
            value if max_items is None or len(value) <= max_items else value[:max_items]
        )
        out = [
            cut_strings(v, limit, pointer, _join(path, str(i)), max_items)
            for i, v in enumerate(shown)
        ]
        if len(shown) < len(value):
            note = (
                pointer(path, len(value), "items")
                if pointer
                else f"{len(value):,} items in total"
            )
            out.append(f"… [{note}]")
        return out
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
