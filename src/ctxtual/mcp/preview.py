"""
Framework-independent helpers for offloading MCP tool results.

Nothing here imports FastMCP, so the logic that decides *what* the model sees
is unit-testable on its own.
"""

import ast
import json
import random
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


# ── field summaries ───────────────────────────────────────────────────────

#: Items scanned to describe fields; larger lists are sampled at random
#: (values are then given as examples, without counts).
SUMMARY_SCAN = 5_000
#: Distinct values listed for a field (more: "N distinct").
SUMMARY_VALUES = 8
#: Distinct values tracked per field before it counts as open-ended.
_TRACK = 64
#: Most fields described (nested paths included).
_MAX_FIELDS = 50
#: Nested objects with at most this many keys get one line per key.
_EXPAND_KEYS = 8


class _FieldStats:
    __slots__ = (
        "chars",
        "count",
        "distinct",
        "elements",
        "example",
        "high",
        "keys",
        "low",
        "overflow",
        "strings",
        "types",
        "values",
    )

    def __init__(self) -> None:
        self.count = 0
        self.types: dict[str, int] = {}
        self.values: dict[Any, int] = {}
        self.overflow = False
        self.low: Any = None
        self.high: Any = None
        self.chars = 0
        self.strings = 0
        self.elements: _FieldStats | None = None
        self.keys: dict[str, int] = {}
        self.distinct: set[Any] = set()  # for "unique" (bounded by SUMMARY_SCAN)
        self.example: Any = None

    def add(self, value: Any) -> None:
        self.count += 1
        kind = _kind(value)
        self.types[kind] = self.types.get(kind, 0) + 1
        if kind in ("integer", "number"):
            self.low = value if self.low is None else min(self.low, value)
            self.high = value if self.high is None else max(self.high, value)
        if kind == "string":
            self.chars += len(value)
            self.strings += 1
        if kind in ("string", "boolean", "integer", "number"):
            self.distinct.add(value)
            if self.example is None:
                self.example = value
            if value in self.values:
                self.values[value] += 1
            elif len(self.values) < _TRACK:
                self.values[value] = 1
            else:
                self.overflow = True
        elif kind == "array":
            if self.elements is None:
                self.elements = _FieldStats()
            for element in value[:100]:
                self.elements.add(element)
        elif kind == "object":
            for key in value:
                if key in self.keys or len(self.keys) < _TRACK:
                    self.keys[str(key)] = self.keys.get(str(key), 0) + 1


_KINDS = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


def _kind(value: Any) -> str:
    return _KINDS.get(type(value), "string")


def summarize_fields(items: list[Any]) -> tuple[dict[str, str], dict[str, str]]:
    """
    What a model needs to query a list of objects without guessing: each
    field's type and, depending on the field, its values with counts
    (``"state": "string · open 134, closed 66"``), its range, what a list
    holds, or that it is free text.  Small nested objects are described per
    key (``author.name``).  Returns ``(descriptions, types)``; types feed the
    query example.
    """
    if len(items) > SUMMARY_SCAN:
        # Random, not every n-th: data often repeats with a period.
        picks = sorted(random.Random(0).sample(range(len(items)), SUMMARY_SCAN))
        rows, exact = [items[i] for i in picks], False
    else:
        rows, exact = items, True
    objects = [row for row in rows if isinstance(row, dict)]
    if not objects:
        return {}, {}
    stats: dict[str, _FieldStats] = {}
    for row in objects:
        for key, value in row.items():
            name = str(key)
            if name not in stats and len(stats) >= _MAX_FIELDS:
                continue
            stats.setdefault(name, _FieldStats()).add(value)
    # Nested paths for small objects (author.name), in order after the parent.
    nested: dict[str, dict[str, _FieldStats]] = {}
    for name, st in stats.items():
        if st.keys and len(st.keys) <= _EXPAND_KEYS and not st.types.get("array"):
            children: dict[str, _FieldStats] = {}
            for row in objects:
                inner = row.get(name)
                if isinstance(inner, dict):
                    for key, value in inner.items():
                        children.setdefault(f"{name}.{key}", _FieldStats()).add(value)
            nested[name] = children
    descriptions: dict[str, str] = {}
    types: dict[str, str] = {}
    total = len(objects)
    for name, st in stats.items():
        for path, s in [(name, st), *nested.get(name, {}).items()]:
            if len(descriptions) >= _MAX_FIELDS:
                break
            descriptions[path] = _describe(s, total, exact, expanded=path in nested)
            types[path] = _main_type(s)
    return descriptions, types


def _main_type(st: _FieldStats) -> str:
    kinds = [k for k in st.types if k != "null"]
    if not kinds:
        return "null"
    if set(kinds) == {"integer", "number"}:
        return "number"
    return max(kinds, key=lambda k: st.types[k])


def _describe(st: _FieldStats, total: int, exact: bool, expanded: bool = False) -> str:
    kind = _main_type(st)
    others = [
        k
        for k in st.types
        if k not in ("null", kind) and not (kind == "number" and k == "integer")
    ]
    label = {"array": "list", "number": "number"}.get(kind, kind)
    if kind == "string" and st.strings and st.chars / st.strings > 60:
        label = "text"
    if others:
        label += " or " + " or ".join(others)
    if st.types.get("null"):
        label += " (or null)"
    detail = _values_detail(st, kind, exact)
    if kind == "array" and st.elements is not None and st.elements.count:
        inner = _main_type(st.elements)
        label = f"list of {inner}"
        detail = _values_detail(st.elements, inner, exact)
        if inner == "object" and st.elements.keys:
            detail = "keys: " + ", ".join(list(st.elements.keys)[:SUMMARY_VALUES])
    elif kind == "array":
        detail = "always empty"
    elif kind == "object" and st.keys:
        detail = (
            "keys below"
            if expanded
            else f"{len(st.keys)} keys: " + ", ".join(list(st.keys)[:SUMMARY_VALUES])
        )
    present = st.count / total if total else 1
    if present < 0.95:
        detail = (
            f"{detail} · in {present:.0%} of items"
            if detail
            else f"in {present:.0%} of items"
        )
    return f"{label} · {detail}" if detail else label


def _values_detail(st: _FieldStats, kind: str, exact: bool) -> str:
    unique = exact and st.count > SUMMARY_VALUES and len(st.distinct) == st.count
    if kind in ("integer", "number") and st.low is not None:
        if st.low == st.high:
            return f"always {_num(st.low)}"
        span = f"{_num(st.low)} to {_num(st.high)}"
        if not exact:
            span = f"about {span}"
        return f"{span}, unique" if unique else span
    if kind in ("string", "boolean"):
        if kind == "string" and st.strings and st.chars / st.strings > 60:
            return f"~{st.chars // st.strings:,} chars"
        if st.overflow or len(st.values) > SUMMARY_VALUES:
            if unique:
                many = "unique"
            elif exact:
                many = f"{len(st.distinct):,} distinct"
            else:
                many = (
                    "many distinct"
                    if st.overflow
                    else f"at least {len(st.distinct)} distinct"
                )
            return f"{many}, e.g. {_short(st.example)}"
        if exact and len(st.values) == 1 and st.count == sum(st.values.values()):
            return f"always {_short(next(iter(st.values)))}"
        ordered = sorted(st.values.items(), key=lambda kv: (-kv[1], str(kv[0])))
        shown = [f"{_short(v)} {n}" if exact else _short(v) for v, n in ordered]
        return ", ".join(shown) if exact else "includes " + ", ".join(shown)
    return ""


def _short(value: Any) -> str:
    text = (
        json.dumps(value, ensure_ascii=False) if isinstance(value, bool) else str(value)
    )
    return text if len(text) <= 32 else text[:31] + "…"


def _num(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)
