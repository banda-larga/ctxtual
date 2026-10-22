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
``"max:field"``, ``"values:field"`` (unique values list), ``"median:field"``,
``"stddev:field"``.

Regular expressions
-------------------

``$regex`` patterns are written by the model and run on your server with
Python's backtracking :mod:`re` engine, which has no timeout, so a crafted
pattern can pin a CPU.  ``$regex`` is therefore **disabled by default**;
enable it with ``PipelineEngine(allow_regex=True)`` /
``pipeline(ctx, allow_regex=True)``.  When enabled, patterns are checked on
Python's own parse tree and rejected if they contain a repeat nested inside
an unbounded repeat (``(a+)+``, ``((a)+)+``), alternation inside an unbounded
repeat (``(a|a)+``), backreferences, or more than
:data:`MAX_UNBOUNDED_REPEATS` unbounded repeats; values are matched only up
to :data:`MAX_REGEX_INPUT` characters.  This removes the known exponential
shapes but is not a proof of safety — only enable it for trusted prompts.
"""

import functools
import math
import random
import re
from contextvars import ContextVar
from re import _constants as _sre_c  # type: ignore[attr-defined]
from re import _parser as _sre_parser  # type: ignore[attr-defined]
from typing import Any, ClassVar

#: Longest ``$regex`` pattern accepted.
MAX_REGEX_LENGTH = 200
#: Characters of each value a ``$regex`` is matched against.
MAX_REGEX_INPUT = 1_000
#: Most distinct values a ``values:field`` metric returns.  Larger sets come
#: back as ``{"values": [...], "total": N, "truncated": True}``.
MAX_METRIC_VALUES = 100
#: Most unbounded repeats (``*``, ``+``, ``{n,}``) a pattern may contain.
MAX_UNBOUNDED_REPEATS = 4

_REPEAT_OPS = {
    _sre_c.MAX_REPEAT,
    _sre_c.MIN_REPEAT,
    *([_sre_c.POSSESSIVE_REPEAT] if hasattr(_sre_c, "POSSESSIVE_REPEAT") else []),
}

_allow_regex: ContextVar[bool] = ContextVar("ctxtual_allow_regex", default=False)


class PipelineError(ValueError):
    """Error raised during pipeline execution with LLM-friendly context."""

    def __init__(self, message: str, **context: Any) -> None:
        self.context = context
        super().__init__(message)


# Public API


class PipelineEngine:
    """
    Execute declarative data pipelines on lists of dicts.

    Args:
        allow_regex: Whether ``$regex`` filters are permitted (default
                     ``False``; see the module docs).
    """

    OPERATORS: ClassVar[dict[str, Any]] = {}  # populated after class definition

    def __init__(self, *, allow_regex: bool = False) -> None:
        self.allow_regex = allow_regex

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

        token = _allow_regex.set(self.allow_regex)
        try:
            return self._run(items, steps)
        finally:
            _allow_regex.reset(token)

    def _run(self, items: list[Any], steps: list[dict[str, Any]]) -> Any:
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
            if isinstance(field_val, (list, tuple)):
                if threshold not in field_val:
                    return False
            elif str(threshold) not in str(field_val or ""):
                return False
        elif op == "$startswith":
            if not str(field_val or "").startswith(str(threshold)):
                return False
        elif op == "$regex":
            pattern = _compile_regex(str(threshold))
            if not pattern.search(str(field_val or "")[:MAX_REGEX_INPUT]):
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


def _compile_regex(pattern: str) -> "re.Pattern[str]":
    if not _allow_regex.get():
        raise PipelineError(
            "$regex is disabled for this pipeline. Use $contains or "
            "$startswith instead."
        )
    return _compile_checked(pattern)


@functools.lru_cache(maxsize=256)
def _compile_checked(pattern: str) -> "re.Pattern[str]":
    if len(pattern) > MAX_REGEX_LENGTH:
        raise PipelineError(
            f"$regex pattern is longer than {MAX_REGEX_LENGTH} characters."
        )
    try:
        tree = _sre_parser.parse(pattern)
        compiled = re.compile(pattern)
    except re.error as exc:
        raise PipelineError(f"Invalid $regex pattern: {exc}") from exc
    problem = _regex_problem(tree)
    if problem:
        raise PipelineError(
            f"$regex pattern rejected: {problem}, which can take "
            "exponential time. Simplify it."
        )
    return compiled


def _regex_problem(tree: Any) -> str | None:
    """Return why a parsed pattern is unsafe to run, or ``None``."""
    unbounded = 0

    # in_unbounded: inside a repeat with no upper bound (any inner repeat
    #   multiplies backtracking paths exponentially);
    # in_multi: inside a repeat with upper bound > 1 (an inner unbounded
    #   repeat makes the cost polynomial with that bound as the degree).
    def walk(items: Any, in_unbounded: bool, in_multi: bool) -> str | None:
        nonlocal unbounded
        for op, av in items:
            if op in _REPEAT_OPS:
                _low, high, sub = av
                is_unbounded = high == _sre_c.MAXREPEAT
                if in_unbounded or (in_multi and is_unbounded):
                    return "it repeats a group that itself repeats"
                if is_unbounded:
                    unbounded += 1
                    if unbounded > MAX_UNBOUNDED_REPEATS:
                        return f"it has more than {MAX_UNBOUNDED_REPEATS} unbounded repeats"
                    if _contains(sub, {_sre_c.BRANCH}):
                        return "it repeats an alternation"
                found = walk(sub, in_unbounded or is_unbounded, in_multi or high > 1)
            elif op == _sre_c.SUBPATTERN:
                found = walk(av[-1], in_unbounded, in_multi)
            elif op == _sre_c.BRANCH:
                found = None
                for branch in av[1]:
                    found = found or walk(branch, in_unbounded, in_multi)
            elif op in (_sre_c.ASSERT, _sre_c.ASSERT_NOT):
                found = walk(av[1], in_unbounded, in_multi)
            elif op in (_sre_c.GROUPREF, _sre_c.GROUPREF_EXISTS):
                found = "it uses a backreference"
            else:
                found = None
            if found:
                return found
        return None

    return walk(tree, False, False)


def _contains(items: Any, ops: set[Any]) -> bool:
    for op, av in items:
        if op in ops:
            return True
        if op in _REPEAT_OPS and _contains(av[2], ops):
            return True
        if op == _sre_c.SUBPATTERN and _contains(av[-1], ops):
            return True
        if op in (_sre_c.ASSERT, _sre_c.ASSERT_NOT) and _contains(av[1], ops):
            return True
    return False


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
        if len(unique) > MAX_METRIC_VALUES:
            return {
                "values": unique[:MAX_METRIC_VALUES],
                "total": len(unique),
                "truncated": True,
            }
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
