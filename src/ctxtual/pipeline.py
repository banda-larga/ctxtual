"""
Pipeline engine — declarative data transformation for workspace items.

The pipeline engine executes an ordered list of operations on workspace data.
Each step transforms the output of the previous step.  Only the final result
is returned to the LLM, so intermediate data never enters the context window.

This is ctx's answer to "programmatic tool calling" — compound data
operations in a single tool call, framework-agnostic, works with any LLM.

Operations
----------

**Filtering:**

- ``{"filter": {"field": "value"}}`` — equality
- ``{"filter": {"field": {"$gt": 5}}}`` — comparisons: $gt, $gte, $lt, $lte,
  $ne, $in, $nin, $contains, $startswith, $regex
- ``{"filter": {"$or": [...]}}`` — logical: $or, $and, $not
- Dot notation for nested fields: ``"author.name"``

**Search:**

- ``{"search": {"query": "text"}}`` — case-insensitive across all fields
- ``{"search": {"query": "text", "fields": ["title"]}}`` — targeted

**Ordering:**

- ``{"sort": {"field": "name"}}`` — ascending by default
- ``{"sort": {"field": "score", "order": "desc"}}``
- ``{"sort": [{"field": "score", "order": "desc"}, {"field": "name"}]}``

**Projection:**

- ``{"select": ["field1", "field2"]}`` — keep only these fields
- ``{"exclude": ["big_blob"]}`` — remove these fields

**Slicing:**

- ``{"limit": 10}`` — first N items
- ``{"skip": 5}`` — skip first N items
- ``{"slice": [5, 15]}`` — items[5:15]
- ``{"sample": 10}`` or ``{"sample": {"n": 10, "seed": 42}}`` — random sample

**Deduplication:**

- ``{"unique": "field"}`` — deduplicate by field, keep first

**Expansion:**

- ``{"flatten": "tags"}`` — expand items where *tags* is a list

**Aggregation (terminal):**

- ``{"group_by": {"field": "cat", "metrics": {"n": "count", "avg": "mean:score"}}}``
- ``{"count": true}`` — return ``{"count": N}``

Metric specs: ``"count"``, ``"sum:field"``, ``"mean:field"``, ``"min:field"``,
``"max:field"``, ``"values:field"`` (unique values list).
"""

from __future__ import annotations

import math
import random
import re
from typing import Any


class PipelineError(ValueError):
    """Error raised during pipeline execution with LLM-friendly context."""

    def __init__(self, message: str, **context: Any) -> None:
        self.context = context
        super().__init__(message)


# Public API


class PipelineEngine:
    """Execute declarative data pipelines on lists of dicts."""

    OPERATORS: dict[str, Any] = {}  # populated after class definition

    def execute(self, items: list[Any], steps: list[dict[str, Any]]) -> Any:
        """
        Run *steps* sequentially on *items*, returning the final result.

        Each step must be a dict with exactly **one** key (the operation name)
        whose value is the operation's parameters.

        Raises :class:`PipelineError` on invalid steps or operator failures.
        """
        if not isinstance(steps, list):
            raise PipelineError(
                "steps must be a list of operation dicts.",
                example=[{"filter": {"status": "active"}}, {"limit": 10}],
            )

        data: Any = items
        for idx, step in enumerate(steps):
            op_name, params = _parse_step(step, idx)
            op_fn = self.OPERATORS.get(op_name)
            if op_fn is None:
                raise PipelineError(
                    f"Unknown operation '{op_name}' at step {idx}.",
                    available_operations=sorted(self.OPERATORS.keys()),
                )
            try:
                data = op_fn(data, params)
            except PipelineError:
                raise
            except Exception as exc:
                raise PipelineError(
                    f"Step {idx} ({op_name}) failed: {exc}",
                    step=step,
                ) from exc
        return data


# Operators


def _op_filter(data: list[Any], params: Any) -> list[Any]:
    """MongoDB-style filtering."""
    if not isinstance(data, list):
        raise PipelineError("filter requires list input.")
    if not isinstance(params, dict):
        raise PipelineError(
            "filter params must be a dict of conditions.",
            example={"status": "active", "score": {"$gte": 50}},
        )
    return [item for item in data if _matches(item, params)]


def _op_search(data: list[Any], params: Any) -> list[Any]:
    """Case-insensitive text search across fields."""
    if not isinstance(data, list):
        raise PipelineError("search requires list input.")
    if isinstance(params, str):
        params = {"query": params}
    if not isinstance(params, dict) or "query" not in params:
        raise PipelineError(
            'search requires {"query": "text"} or just a string.',
            example={"query": "machine learning", "fields": ["title"]},
        )
    query = str(params["query"]).lower()
    fields = params.get("fields")

    results = []
    for item in data:
        if not isinstance(item, dict):
            if query in str(item).lower():
                results.append(item)
            continue
        values = (
            [str(item.get(f, "")) for f in fields]
            if fields
            else [str(v) for v in item.values()]
        )
        if any(query in v.lower() for v in values):
            results.append(item)
    return results


def _op_sort(data: list[Any], params: Any) -> list[Any]:
    """Sort by one or more fields."""
    if not isinstance(data, list):
        raise PipelineError("sort requires list input.")

    # Normalise to list of sort specs
    if isinstance(params, dict):
        specs = [params]
    elif isinstance(params, list):
        specs = params
    elif isinstance(params, str):
        specs = [{"field": params}]
    else:
        raise PipelineError(
            'sort params must be {"field": ..., "order": "asc"|"desc"} '
            "or a list of such dicts.",
        )

    # Multi-field sort: apply in reverse order (stable sort composes)
    result = list(data)
    for spec in reversed(specs):
        field = spec if isinstance(spec, str) else spec.get("field", "")
        order = "asc" if isinstance(spec, str) else spec.get("order", "asc")
        reverse = order == "desc"
        result.sort(key=lambda x, f=field: _sort_key(_get_field(x, f)), reverse=reverse)
    return result


def _op_select(data: list[Any], params: Any) -> list[Any]:
    """Project: keep only the specified fields."""
    if not isinstance(params, list):
        raise PipelineError(
            "select requires a list of field names.",
            example=["title", "score", "author"],
        )
    fields = set(params)
    return [
        (
            {k: v for k, v in item.items() if k in fields}
            if isinstance(item, dict)
            else item
        )
        for item in (data if isinstance(data, list) else [data])
    ]


def _op_exclude(data: list[Any], params: Any) -> list[Any]:
    """Remove the specified fields."""
    if not isinstance(params, list):
        raise PipelineError("exclude requires a list of field names.")
    fields = set(params)
    return [
        (
            {k: v for k, v in item.items() if k not in fields}
            if isinstance(item, dict)
            else item
        )
        for item in (data if isinstance(data, list) else [data])
    ]


def _op_limit(data: Any, params: Any) -> list[Any]:
    """Take first N items."""
    n = params if isinstance(params, int) else int(params)
    if isinstance(data, list):
        return data[:n]
    return data


def _op_skip(data: Any, params: Any) -> list[Any]:
    """Skip first N items."""
    n = params if isinstance(params, int) else int(params)
    if isinstance(data, list):
        return data[n:]
    return data


def _op_slice(data: Any, params: Any) -> list[Any]:
    """Slice [start, end)."""
    if not isinstance(params, list) or len(params) != 2:
        raise PipelineError(
            "slice requires [start, end] (two integers).",
            example=[5, 15],
        )
    start, end = int(params[0]), int(params[1])
    if isinstance(data, list):
        return data[start:end]
    return data


def _op_sample(data: list[Any], params: Any) -> list[Any]:
    """Random sample of N items."""
    if not isinstance(data, list):
        raise PipelineError("sample requires list input.")
    if isinstance(params, (int, float)):
        n, seed = int(params), None
    elif isinstance(params, dict):
        n = int(params.get("n", 10))
        seed = params.get("seed")
    else:
        n, seed = int(params), None

    rng = random.Random(seed)
    n = min(n, len(data))
    return rng.sample(data, n)


def _op_unique(data: list[Any], params: Any) -> list[Any]:
    """Deduplicate by field value, keeping first occurrence."""
    if not isinstance(data, list):
        raise PipelineError("unique requires list input.")
    field = params if isinstance(params, str) else str(params)
    seen: set[Any] = set()
    result = []
    for item in data:
        val = _get_field(item, field)
        # Convert unhashable to string repr for dedup
        key = val if isinstance(val, (str, int, float, bool, type(None))) else str(val)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _op_flatten(data: list[Any], params: Any) -> list[Any]:
    """Expand items where the named field is a list into multiple items."""
    if not isinstance(data, list):
        raise PipelineError("flatten requires list input.")
    field = params if isinstance(params, str) else str(params)
    result = []
    for item in data:
        if not isinstance(item, dict):
            result.append(item)
            continue
        val = item.get(field)
        if isinstance(val, list):
            for v in val:
                expanded = dict(item)
                expanded[field] = v
                result.append(expanded)
        else:
            result.append(item)
    return result


def _op_group_by(data: list[Any], params: Any) -> list[dict[str, Any]]:
    """Group items and compute aggregate metrics."""
    if not isinstance(data, list):
        raise PipelineError("group_by requires list input.")
    if not isinstance(params, dict):
        raise PipelineError(
            'group_by requires {"field": ..., "metrics": {...}}.',
            example={
                "field": "category",
                "metrics": {"n": "count", "avg": "mean:score"},
            },
        )

    field = params.get("field") or params.get("by", "")
    metrics = params.get("metrics", {"count": "count"})

    groups: dict[Any, list[dict[str, Any]]] = {}
    for item in data:
        key = _get_field(item, field)
        # Make key serialisable
        group_key = (
            key if isinstance(key, (str, int, float, bool, type(None))) else str(key)
        )
        groups.setdefault(group_key, []).append(item)

    results: list[dict[str, Any]] = []
    for key, group_items in groups.items():
        row: dict[str, Any] = {field: key}
        for metric_name, metric_spec in metrics.items():
            row[metric_name] = _compute_metric(metric_spec, group_items)
        results.append(row)
    return results


def _op_count(data: Any, params: Any) -> dict[str, int]:
    """Return item count. Terminal operation."""
    if isinstance(data, list):
        return {"count": len(data)}
    if isinstance(data, dict):
        return {"count": len(data)}
    return {"count": 1}


# Register operators

PipelineEngine.OPERATORS = {
    "filter": _op_filter,
    "search": _op_search,
    "sort": _op_sort,
    "select": _op_select,
    "exclude": _op_exclude,
    "limit": _op_limit,
    "skip": _op_skip,
    "slice": _op_slice,
    "sample": _op_sample,
    "unique": _op_unique,
    "flatten": _op_flatten,
    "group_by": _op_group_by,
    "count": _op_count,
}


# Aggregate helper


def compute_aggregates(
    items: list[dict[str, Any]],
    metrics: dict[str, str],
    group_by: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Compute aggregate statistics over a list of dicts.

    Args:
        items:    The data to aggregate.
        metrics:  Dict mapping output names to metric specs.
                  Specs: ``"count"``, ``"sum:field"``, ``"mean:field"``,
                  ``"min:field"``, ``"max:field"``, ``"values:field"``.
        group_by: Optional field to group by before aggregating.

    Returns:
        Without *group_by*: a single metrics dict.
        With *group_by*: a list of dicts, each containing the group key
        and its computed metrics.
    """
    if group_by:
        return _op_group_by(items, {"field": group_by, "metrics": metrics})

    result: dict[str, Any] = {}
    for metric_name, metric_spec in metrics.items():
        result[metric_name] = _compute_metric(metric_spec, items)
    return result


# Internal helpers


def _parse_step(step: Any, idx: int) -> tuple[str, Any]:
    """Parse a pipeline step dict into (op_name, params)."""
    if not isinstance(step, dict) or len(step) != 1:
        raise PipelineError(
            f"Step {idx}: each step must be a dict with exactly one key "
            f"(the operation name). Got: {step!r}",
            example={"filter": {"status": "active"}},
            available_operations=sorted(PipelineEngine.OPERATORS.keys()),
        )
    op_name = next(iter(step))
    return op_name, step[op_name]


def _get_field(item: Any, key: str) -> Any:
    """Get a field value, supporting dot notation for nested access."""
    if not isinstance(item, dict):
        return None
    parts = key.split(".")
    val = item
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        elif isinstance(val, list) and p.isdigit():
            idx = int(p)
            val = val[idx] if 0 <= idx < len(val) else None
        else:
            return None
    return val


def _sort_key(val: Any) -> tuple[int, Any]:
    """
    Return a sort key that handles mixed types and None.

    Strategy: (type_rank, value) where None sorts first, then numbers,
    then strings, then everything else.
    """
    if val is None:
        return (0, "")
    if isinstance(val, (int, float)):
        return (1, val)
    if isinstance(val, str):
        return (2, val)
    if isinstance(val, bool):
        return (1, int(val))
    return (3, str(val))


def _matches(item: Any, conditions: dict[str, Any]) -> bool:
    """Test if *item* matches MongoDB-style *conditions*."""
    for key, value in conditions.items():
        if key == "$or":
            if not isinstance(value, list) or not any(_matches(item, c) for c in value):
                return False
        elif key == "$and":
            if not isinstance(value, list) or not all(_matches(item, c) for c in value):
                return False
        elif key == "$not":
            if _matches(item, value):
                return False
        elif isinstance(value, dict) and any(k.startswith("$") for k in value):
            # Comparison operators on a field
            field_val = _get_field(item, key)
            if not _compare(field_val, value):
                return False
        else:
            # Simple equality
            if _get_field(item, key) != value:
                return False
    return True


def _compare(field_val: Any, ops: dict[str, Any]) -> bool:
    """Evaluate comparison operators against a field value."""
    for op, threshold in ops.items():
        if op == "$gt":
            if field_val is None or not (field_val > threshold):
                return False
        elif op == "$gte":
            if field_val is None or not (field_val >= threshold):
                return False
        elif op == "$lt":
            if field_val is None or not (field_val < threshold):
                return False
        elif op == "$lte":
            if field_val is None or not (field_val <= threshold):
                return False
        elif op == "$ne":
            if field_val == threshold:
                return False
        elif op == "$in":
            if field_val not in threshold:
                return False
        elif op == "$nin":
            if field_val in threshold:
                return False
        elif op == "$contains":
            if threshold not in str(field_val or ""):
                return False
        elif op == "$startswith":
            if not str(field_val or "").startswith(str(threshold)):
                return False
        elif op == "$regex":
            if not re.search(str(threshold), str(field_val or "")):
                return False
        elif op == "$exists":
            if threshold and field_val is None:
                return False
            if not threshold and field_val is not None:
                return False
        else:
            raise PipelineError(
                f"Unknown comparison operator '{op}'.",
                available_operators=[
                    "$gt",
                    "$gte",
                    "$lt",
                    "$lte",
                    "$ne",
                    "$in",
                    "$nin",
                    "$contains",
                    "$startswith",
                    "$regex",
                    "$exists",
                ],
            )
    return True


def _compute_metric(spec: str, items: list[dict[str, Any]]) -> Any:
    """Compute a single metric from a spec string."""
    if spec == "count":
        return len(items)
    if ":" not in spec:
        raise PipelineError(
            f"Unknown metric spec '{spec}'. "
            "Use 'count', 'sum:field', 'mean:field', 'min:field', "
            "'max:field', or 'values:field'.",
        )
    agg_fn, agg_field = spec.split(":", 1)
    values = [
        _get_field(i, agg_field) for i in items if _get_field(i, agg_field) is not None
    ]
    numeric = [v for v in values if isinstance(v, (int, float))]

    if agg_fn == "sum":
        return sum(numeric)
    if agg_fn == "mean":
        return sum(numeric) / len(numeric) if numeric else 0
    if agg_fn == "min":
        return min(numeric) if numeric else None
    if agg_fn == "max":
        return max(numeric) if numeric else None
    if agg_fn == "values":
        seen: set[str] = set()
        unique: list[Any] = []
        for v in values:
            k = str(v)
            if k not in seen:
                seen.add(k)
                unique.append(v)
        return unique
    if agg_fn == "median":
        if not numeric:
            return None
        s = sorted(numeric)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
    if agg_fn == "stddev":
        if len(numeric) < 2:
            return 0
        mean = sum(numeric) / len(numeric)
        variance = sum((x - mean) ** 2 for x in numeric) / (len(numeric) - 1)
        return math.sqrt(variance)

    raise PipelineError(
        f"Unknown aggregation function '{agg_fn}'.",
        available=["sum", "mean", "min", "max", "values", "median", "stddev"],
    )
