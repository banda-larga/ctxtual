"""
Built-in utility ToolSets.

These cover the most common patterns when an agent needs to consume a
workspace.  Import them and pass to ``@ctx.producer`` instead of writing
boilerplate every time.

Available factories
-------------------

- :func:`paginator`     — paginate / count / get-by-index / slice
- :func:`text_search`   — ranked text search / field value discovery
- :func:`filter_set`    — structured filtering and sorting
- :func:`kv_reader`     — for dict-type (single-document) workspaces
- :func:`pipeline`      — declarative data pipelines (filter→sort→group in one call)
- :func:`text_content`  — for string/text workspaces (documents, webpages, PDFs)

Every tool caps how much it returns, so a single call can't flood the
context window.  When a result is cut, the response says so (``has_more``,
``truncated``, ``total``) and suggests how to narrow it.

Usage::

    from ctxtual import Ctx, MemoryStore
    from ctxtual.utils import paginator, text_search, filter_set, pipeline

    ctx = Ctx(store=MemoryStore())

    # Simple — name is inferred from workspace_type:
    @ctx.producer(workspace_type="papers", toolsets=[
        paginator(ctx),
        text_search(ctx, fields=["title", "abstract"]),
        filter_set(ctx),
        pipeline(ctx),
    ])
    def fetch_papers(query: str):
        ...

    # Explicit — name is passed directly (still works):
    pager = paginator(ctx, "papers")
    search = text_search(ctx, "papers", fields=["title"])

    # Generic — one set of tools shared by every workspace type, which
    # keeps the tool list short:
    data = paginator(ctx, "data", enforce_type=False)
"""

import json
import math
import re
from typing import Any

from ctxtual.ctx import Ctx
from ctxtual.pipeline import PipelineEngine, PipelineError, compute_aggregates
from ctxtual.session import is_visible
from ctxtual.toolset import ToolSet, ToolSpec
from ctxtual.types import WorkspaceMeta


def _make_toolset(ctx: Ctx, name: str, enforce_type: bool, shape: str) -> ToolSet:
    ts = ctx.toolset(name, enforce_type=enforce_type)
    ts.claim_shape(shape)
    return ts


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


# paginator: page through list-type workspaces


def paginator(
    ctx: Ctx,
    name: str | None = None,
    *,
    max_page_size: int = 100,
    enforce_type: bool = True,
) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with pagination tools for list-type workspace data.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@ctx.producer``.

    Args:
        max_page_size: Upper bound for ``size`` and slice length.
        enforce_type:  Set ``False`` to share these tools across workspace
                       types.

    Tools:

    - ``paginate(workspace_id, page, size)`` — list slice + metadata
    - ``count(workspace_id)`` — total item count
    - ``get_item(workspace_id, index)`` — single item by index
    - ``get_slice(workspace_id, start, end)`` — arbitrary slice
    """
    if name is None:
        return ToolSpec(
            paginator, ctx, max_page_size=max_page_size, enforce_type=enforce_type
        )
    ts = _make_toolset(ctx, name, enforce_type, "list")

    @ts.tool(
        name=f"{name}_paginate",
        data_shape="list",
        output_hint=f"Navigate with {name}_paginate(workspace_id='{{workspace_id}}', page=NEXT_PAGE). Use {name}_search() to find specific items.",
    )
    def paginate(workspace_id: str, page: int = 0, size: int = 10) -> dict[str, Any]:
        """Return a page of items from the workspace with pagination metadata.

        Args:
            workspace_id: Identifier of the workspace to paginate.
            page: Zero-based page number to retrieve.
            size: Number of items per page.
        """
        size = _clamp(size, 1, max_page_size)
        page = max(0, int(page))
        total = ts.store.count_items(workspace_id)
        chunk = ts.store.get_page(workspace_id, page * size, size)
        return {
            "items": chunk,
            "page": page,
            "size": size,
            "total": total,
            "total_pages": max(1, math.ceil(total / size)),
            "has_next": (page + 1) * size < total,
            "has_prev": page > 0,
        }

    @ts.tool(name=f"{name}_count", data_shape="list")
    def count(workspace_id: str) -> dict[str, Any]:
        """Return total number of items in the workspace.

        Args:
            workspace_id: Identifier of the workspace to count.
        """
        total = ts.store.count_items(workspace_id)
        return {"workspace_id": workspace_id, "count": total}

    @ts.tool(name=f"{name}_get_item", data_shape="list")
    def get_item(workspace_id: str, index: int) -> Any:
        """Return a single item by zero-based index.

        Args:
            workspace_id: Identifier of the workspace to read from.
            index: Zero-based position of the item to retrieve.
        """
        total = ts.store.count_items(workspace_id)
        page = ts.store.get_page(workspace_id, index, 1) if 0 <= index < total else []
        if not page:
            return {
                "error": f"Index {index} out of range.",
                "valid_range": f"0–{max(0, total - 1)}",
                "total_items": total,
                "suggested_action": (
                    f"Use an index between 0 and {max(0, total - 1)}, "
                    f"or call {name}_paginate to browse items."
                ),
            }
        return page[0]

    @ts.tool(name=f"{name}_get_slice", data_shape="list")
    def get_slice(workspace_id: str, start: int = 0, end: int = 20) -> dict[str, Any]:
        """Return an arbitrary ``[start:end]`` slice of workspace items.

        Args:
            workspace_id: Identifier of the workspace to slice.
            start: Start index (inclusive).
            end: End index (exclusive).
        """
        start = max(0, int(start))
        requested = max(0, int(end) - start)
        length = min(requested, max_page_size)
        total = ts.store.count_items(workspace_id)
        result: dict[str, Any] = {
            "items": ts.store.get_page(workspace_id, start, length),
            "start": start,
            "end": start + length,
            "total": total,
        }
        if length < requested:
            result["truncated"] = True
            result["suggested_action"] = (
                f"Slices are limited to {max_page_size} items; request "
                f"start={start + length} next."
            )
        return result

    return ts


# text_search — ranked search over list workspaces


def text_search(
    ctx: Ctx,
    name: str | None = None,
    *,
    fields: list[str] | None = None,
    max_results_cap: int = 100,
    enforce_type: bool = True,
) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with text-search tools for list-type workspace data.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@ctx.producer``.

    Args:
        fields:          If provided, search is limited to these dict keys.
                         If ``None``, searches all values in each item.
        max_results_cap: Upper bound for ``max_results``.
        enforce_type:    Set ``False`` to share these tools across types.

    Tools:

    - ``search(workspace_id, query, max_results, case_sensitive)`` — ranked matches
    - ``field_values(workspace_id, field, max_values)`` — distinct field values
    """
    if name is None:
        return ToolSpec(
            text_search,
            ctx,
            fields=fields,
            max_results_cap=max_results_cap,
            enforce_type=enforce_type,
        )
    ts = _make_toolset(ctx, name, enforce_type, "list")
    _fields = fields

    @ts.tool(name=f"{name}_search", data_shape="list")
    def search(
        workspace_id: str,
        query: str,
        max_results: int = 20,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """Search workspace items for words in the query, best matches first.

        Args:
            workspace_id: Identifier of the workspace to search.
            query: Words to search for in item fields.
            max_results: Maximum number of matching items to return.
            case_sensitive: Whether the search should be case-sensitive.
        """
        max_results = _clamp(max_results, 1, max_results_cap)
        # Ask for one extra item to learn whether more matches exist.
        matches = ts.store.search_items(
            workspace_id,
            query,
            fields=_fields,
            max_results=max_results + 1,
            case_sensitive=case_sensitive,
        )
        has_more = len(matches) > max_results
        result: dict[str, Any] = {
            "query": query,
            "returned": min(len(matches), max_results),
            "has_more": has_more,
            "matches": matches[:max_results],
        }
        if has_more:
            result["suggested_action"] = (
                "More matches exist. Use a more specific query, or a pipeline "
                "with search + filter to narrow them down."
            )
        return result

    @ts.tool(name=f"{name}_field_values", data_shape="list")
    def field_values(
        workspace_id: str,
        field: str,
        max_values: int = 50,
    ) -> dict[str, Any]:
        """Return distinct values for a given field across all workspace items.

        Useful for discovering facets (e.g. all unique authors, all years).

        Args:
            workspace_id: Identifier of the workspace to inspect.
            field: Name of the field whose distinct values to collect.
            max_values: Maximum number of distinct values to return.
        """
        max_values = _clamp(max_values, 1, max_results_cap)
        values = ts.store.distinct_field_values(
            workspace_id, field, max_values=max_values + 1
        )
        return {
            "field": field,
            "distinct_values": values[:max_values],
            "count": min(len(values), max_values),
            "has_more": len(values) > max_values,
        }

    return ts


# filter_set — structured filtering on dict-list workspaces


def filter_set(
    ctx: Ctx,
    name: str | None = None,
    *,
    max_results_cap: int = 200,
    enforce_type: bool = True,
) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with structured filtering and sorting tools.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@ctx.producer``.

    Args:
        max_results_cap: Upper bound for the ``limit`` of both tools.
        enforce_type:    Set ``False`` to share these tools across types.

    Tools:

    - ``filter_by(workspace_id, field, value, operator, limit)`` — filtered list
    - ``sort_by(workspace_id, field, descending, limit)`` — sorted list

    Supported operators: ``eq``, ``ne``, ``lt``, ``lte``, ``gt``, ``gte``,
    ``contains``, ``startswith``.
    """
    if name is None:
        return ToolSpec(
            filter_set, ctx, max_results_cap=max_results_cap, enforce_type=enforce_type
        )
    ts = _make_toolset(ctx, name, enforce_type, "list")

    @ts.tool(name=f"{name}_filter_by", data_shape="list")
    def filter_by(
        workspace_id: str,
        field: str,
        value: Any,
        operator: str = "eq",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Filter workspace items by a field value using a comparison operator.

        Args:
            workspace_id: Identifier of the workspace to filter.
            field: Name of the field to compare.
            value: Value to compare against.
            operator: Comparison operator (eq, ne, lt, lte, gt, gte, contains, startswith).
            limit: Maximum number of matching items to return.
        """
        limit = _clamp(limit, 1, max_results_cap)
        results = ts.store.filter_items(workspace_id, field, value, operator)
        response: dict[str, Any] = {
            "field": field,
            "operator": operator,
            "value": value,
            "total_matches": len(results),
            "returned": min(len(results), limit),
            "results": results[:limit],
        }
        if len(results) > limit:
            response["truncated"] = True
            response["suggested_action"] = (
                "Only the first matches are shown. Add conditions with a "
                "pipeline, or use save_as to browse all matches."
            )
        return response

    @ts.tool(name=f"{name}_sort_by", data_shape="list")
    def sort_by(
        workspace_id: str,
        field: str,
        descending: bool = False,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Sort workspace items by a field and return the top results.

        Items without the field are left out.

        Args:
            workspace_id: Identifier of the workspace to sort.
            field: Name of the field to sort by.
            descending: If true, sort in descending (high-to-low) order.
            limit: Maximum number of sorted results to return.
        """
        limit = _clamp(limit, 1, max_results_cap)
        items = ts.store.sort_items(
            workspace_id, field, descending=descending, limit=limit + 1
        )
        return {
            "field": field,
            "descending": descending,
            "items": items[:limit],
            "returned": min(len(items), limit),
            "has_more": len(items) > limit,
        }

    return ts


# kv_reader — for single-item / dict workspaces (non-list payloads)


def kv_reader(
    ctx: Ctx,
    name: str | None = None,
    *,
    max_value_chars: int = 20_000,
    max_keys: int = 200,
    enforce_type: bool = True,
) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` for dict-type workspaces (single JSON document).

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@ctx.producer``.

    Args:
        max_value_chars: Values whose JSON form is longer than this are
                         returned as a truncated preview.
        max_keys:        Most keys ``get_keys`` lists.
        enforce_type:    Set ``False`` to share these tools across types.

    Tools:

    - ``get_keys(workspace_id)`` — list of top-level keys
    - ``get_value(workspace_id, key)`` — value at key
    """
    if name is None:
        return ToolSpec(
            kv_reader,
            ctx,
            max_value_chars=max_value_chars,
            max_keys=max_keys,
            enforce_type=enforce_type,
        )
    ts = _make_toolset(ctx, name, enforce_type, "dict")

    @ts.tool(
        name=f"{name}_get_keys",
        data_shape="dict",
        output_hint=f"Use {name}_get_value(workspace_id='{{workspace_id}}', key=KEY) to read each section.",
    )
    def get_keys(workspace_id: str) -> dict[str, Any]:
        """Return the top-level keys of the stored dict.

        Args:
            workspace_id: Identifier of the workspace to inspect.
        """
        data = ts.store.get_items(workspace_id)
        keys = [str(k) for k in data] if isinstance(data, dict) else []
        return {
            "keys": keys[:max_keys],
            "count": len(keys),
            "has_more": len(keys) > max_keys,
        }

    @ts.tool(name=f"{name}_get_value", data_shape="dict")
    def get_value(workspace_id: str, key: str) -> Any:
        """Return the value at a specific top-level key.

        Args:
            workspace_id: Identifier of the workspace to read from.
            key: Top-level dictionary key to retrieve.
        """
        data = ts.store.get_items(workspace_id)
        if not isinstance(data, dict):
            return {
                "error": "Workspace payload is not a dict.",
                "suggested_action": "Use the tools listed for this workspace instead.",
            }
        if key not in data:
            return {
                "error": f"Key '{key}' not found.",
                "available_keys": list(data.keys())[:50],
                "suggested_action": "Use one of the available_keys listed above.",
            }
        value = data[key]
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) <= max_value_chars:
            return value
        summary: dict[str, Any] = {
            "key": key,
            "type": type(value).__name__,
            "truncated": True,
            "total_chars": len(encoded),
            "preview": encoded[:max_value_chars],
        }
        if isinstance(value, (list, dict)):
            summary["length"] = len(value)
        summary["suggested_action"] = (
            "The value is too large to return whole. Store it as its own "
            "workspace (a producer returning it) to explore it piece by piece."
        )
        return summary

    return ts


# pipeline: declarative data pipelines (compound operations in one call)

_SAVE_AS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
#: Prefix of workspace ids created by ``pipe(save_as=...)``.
SAVED_PREFIX = "saved_"


def pipeline(
    ctx: Ctx,
    name: str | None = None,
    *,
    max_items: int = 50,
    allow_regex: bool = False,
    enforce_type: bool = True,
) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with pipeline tools for compound data operations.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@ctx.producer``.

    This eliminates multi-round-trip patterns.  Instead of the LLM calling
    search → filter → sort → limit (4 tool calls, 4 round-trips), it
    describes the entire chain in one ``pipe`` call.  Intermediate data
    never enters the context window.

    Args:
        max_items:    Maximum number of items (or groups) returned inline.
                      Larger results are truncated; ``save_as`` stores the
                      full result as a new workspace instead.
        allow_regex:  Whether filters may use ``$regex`` (default ``False``).
                      Patterns come from the model and run on your server
                      with no timeout; only enable this for trusted prompts
                      (see :mod:`ctxtual.pipeline`).
        enforce_type: Set ``False`` to share these tools across types.

    Tools:

    - ``pipe(workspace_id, steps, save_as)`` — declarative pipeline
    - ``aggregate(workspace_id, group_by, metrics)`` — group-by aggregation

    Note:
        Pipelines run in Python over the whole workspace (also on
        SQLiteStore), so cost grows with workspace size.
    """
    if name is None:
        return ToolSpec(
            pipeline,
            ctx,
            max_items=max_items,
            allow_regex=allow_regex,
            enforce_type=enforce_type,
        )
    ts = _make_toolset(ctx, name, enforce_type, "list")
    engine = PipelineEngine(allow_regex=allow_regex)

    def _capped(items: list[Any], key: str) -> dict[str, Any]:
        response: dict[str, Any] = {
            key: items[:max_items],
            "count": min(len(items), max_items),
        }
        if len(items) > max_items:
            response["total"] = len(items)
            response["truncated"] = True
            response["suggested_action"] = (
                f"Only the first {max_items} of {len(items)} are shown. Add a "
                "limit/filter step, or pass save_as to store the full result "
                "as a new workspace."
            )
        return response

    _pipe_schema_extra: dict[str, dict[str, Any]] = {
        "steps": {
            "items": {
                "type": "object",
                "description": (
                    "Each step is a single-key dict: the key is the operation "
                    "name, the value is its parameters."
                ),
            },
            "examples": [
                [
                    {"filter": {"year": {"$gte": 2020}}},
                    {"sort": {"field": "citations", "order": "desc"}},
                    {"limit": 5},
                ],
                [
                    {"search": {"query": "transformer"}},
                    {"select": ["title", "author", "year"]},
                ],
                [
                    {
                        "group_by": {
                            "field": "category",
                            "metrics": {"count": "count", "avg_score": "mean:score"},
                        }
                    },
                ],
            ],
        },
    }

    @ts.tool(name=f"{name}_pipe", data_shape="list", schema_extra=_pipe_schema_extra)
    def pipe(
        workspace_id: str,
        steps: list[dict[str, Any]],
        save_as: str | None = None,
    ) -> dict[str, Any]:
        """Execute a chain of data operations on workspace items in one call.

        Each step transforms the output of the previous step.  Only the final
        result is returned — intermediate data stays server-side, saving
        round-trips and context tokens.  Large results are truncated.

        Args:
            workspace_id: The workspace to operate on.
            steps: Ordered list of operations. Each is a dict with one key
                (the operation name) and its parameters as the value.
                Operations: filter, search, sort, select, exclude, limit,
                skip, slice, sample, unique, flatten, group_by, count.
            save_as: If provided, store the full result as a new workspace
                named after this value (the id gets a "saved_" prefix) and
                return a summary instead of raw data.  Use when the result
                is large and you want to paginate it.  The response
                contains the new workspace_id; saving again under the same
                name replaces it.

        Filter syntax (MongoDB-style):
            {"filter": {"field": "value"}}  — equality
            {"filter": {"field": {"$gt": 5}}}  — comparison ($gt $gte $lt $lte $ne $in $nin $contains $startswith $regex $exists)
            {"filter": {"$or": [{"a": 1}, {"b": 2}]}}  — logical ($or $and $not)
            Dot notation for nested fields: "author.name"

        Sort syntax:
            {"sort": {"field": "score", "order": "desc"}}
            {"sort": [{"field": "score", "order": "desc"}, {"field": "name"}]}

        Aggregation syntax:
            {"group_by": {"field": "cat", "metrics": {"n": "count", "avg": "mean:score"}}}
            {"count": true}
            Metric specs: "count", "sum:field", "mean:field", "min:field", "max:field", "values:field", "median:field", "stddev:field"

        Examples:
            Search + filter: [{"search": {"query": "ML"}}, {"filter": {"year": {"$gte": 2023}}}]
            Top 5 by score: [{"sort": {"field": "score", "order": "desc"}}, {"limit": 5}]
            Unique authors: [{"select": ["author"]}, {"unique": "author"}]
        """
        items = ts.store.get_items(workspace_id)
        if not isinstance(items, list):
            return {
                "error": "Pipeline requires list-type workspace data.",
                "actual_type": type(items).__name__,
                "suggested_action": "Use the tools listed for this workspace instead.",
            }

        # Parse steps if they came as a JSON string (some LLMs do this)
        if isinstance(steps, str):
            try:
                steps = json.loads(steps)
            except (json.JSONDecodeError, TypeError):
                return {
                    "error": "Could not parse steps. Must be a JSON list.",
                    "suggested_action": "Pass steps as a list of operation dicts.",
                }

        if save_as is not None and not _SAVE_AS_RE.fullmatch(str(save_as)):
            return {
                "error": f"Invalid save_as name {save_as!r}.",
                "suggested_action": (
                    "Use 1-100 letters, digits, '_', '-' or '.', starting "
                    "with a letter or digit."
                ),
            }

        try:
            result = engine.execute(items, steps)
        except PipelineError as exc:
            return {
                "error": str(exc),
                "available_operations": sorted(PipelineEngine.OPERATORS.keys()),
                "suggested_action": "Check operation names and parameter formats.",
                **exc.context,
            }

        if save_as and isinstance(result, list):
            source = ts.store.get_meta(workspace_id)
            ws_type = source.workspace_type if source else name
            ref = ctx.store_workspace(
                result,
                workspace_type=ws_type,
                # Own namespace, so a saved result can never occupy an id
                # a producer's ``key=`` would generate.
                workspace_id=f"{SAVED_PREFIX}{save_as}",
                toolsets=ctx.toolsets_for_type(ws_type) or [ts],
                producer_fn="pipeline",
                meta={"derived_from": workspace_id},
                replace_if=_only_pipeline_results,
            )
            ref_dict = ref.to_dict()
            ref_dict["status"] = "pipeline_result_saved"
            ref_dict["message"] = (
                f"Pipeline result saved as workspace '{ref.workspace_id}' "
                f"with {len(result):,} item(s). Use the tools below to explore."
            )
            return ref_dict

        if isinstance(result, list):
            return _capped(result, "items")
        # Terminal operations (count) return dicts/non-lists
        if isinstance(result, dict):
            return result
        return {"result": result}

    _agg_schema_extra: dict[str, dict[str, Any]] = {
        "metrics": {
            "examples": [
                {"count": "count", "total": "sum:revenue", "avg": "mean:score"},
                {"min_price": "min:price", "max_price": "max:price"},
            ],
        },
    }

    @ts.tool(
        name=f"{name}_aggregate", data_shape="list", schema_extra=_agg_schema_extra
    )
    def aggregate(
        workspace_id: str,
        group_by: str | None = None,
        metrics: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate statistics over workspace data in a single call.

        A convenience tool for common aggregation patterns.  For complex
        multi-step pipelines, use the pipe tool with group_by steps.

        Args:
            workspace_id: The workspace to aggregate.
            group_by: Optional field name to group items by before computing
                metrics.  Without this, metrics are computed over all items.
            metrics: Dict mapping output names to metric specs.
                Specs: "count", "sum:field", "mean:field", "min:field",
                "max:field", "values:field", "median:field", "stddev:field".
                If omitted, returns a count.

        Examples:
            Total count: aggregate(workspace_id="ws_1")
            By category: aggregate(workspace_id="ws_1", group_by="category",
                metrics={"n": "count", "total": "sum:revenue", "avg": "mean:revenue"})
        """
        items = ts.store.get_items(workspace_id)
        if not isinstance(items, list):
            return {
                "error": "Aggregate requires list-type workspace data.",
                "suggested_action": "Use this tool with list workspaces.",
            }

        _metrics: Any = metrics or {"count": "count"}

        # Parse metrics if they came as a JSON string
        if isinstance(_metrics, str):
            try:
                _metrics = json.loads(_metrics)
            except (json.JSONDecodeError, TypeError):
                return {
                    "error": "Could not parse metrics. Must be a JSON dict.",
                    "suggested_action": 'Pass metrics as a dict, e.g. {"n": "count", "avg": "mean:score"}.',
                }

        try:
            result = compute_aggregates(items, _metrics, group_by=group_by)
        except PipelineError as exc:
            return {
                "error": str(exc),
                "suggested_action": "Check metric specs: 'count', 'sum:field', 'mean:field', etc.",
                **exc.context,
            }

        if isinstance(result, list):
            capped = _capped(result, "groups")
            capped["group_count"] = len(result)
            capped.pop("count")
            return capped
        return {"result": result}

    return ts


def _only_pipeline_results(existing: WorkspaceMeta) -> str | None:
    """``save_as`` may only replace an earlier pipeline result."""
    if existing.producer_fn != "pipeline" or not is_visible(existing):
        return "it is not a saved pipeline result"
    return None


# text_content — tools for navigating raw string/text data


def text_content(
    ctx: Ctx,
    name: str | None = None,
    *,
    chars_per_page: int = 3000,
    max_chars_per_page: int = 20_000,
    enforce_type: bool = True,
) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with tools for navigating string/text workspaces.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@ctx.producer``.

    Use this for producers that return raw text — documents, webpages, PDFs,
    logs, source code, etc.  Unlike ``paginator`` (which expects ``list``),
    this toolset works with scalar string data.

    Alternatively, use ``chunk_text()`` or ``split_sections()`` transforms
    to convert text into ``list[dict]`` and use the regular toolsets.

    Tools:

    - ``read_page(workspace_id, page, chars_per_page)`` — character-based pagination
    - ``search_in_text(workspace_id, query, context_chars)`` — find occurrences with surrounding context
    - ``get_length(workspace_id)`` — character and word count

    Args:
        ctx: The Ctx instance.
        name:  Workspace type name (e.g. ``"doc"``).  Optional.
        chars_per_page: Default characters per page for ``read_page``.
        max_chars_per_page: Upper bound the LLM may request per page.
        enforce_type: Set ``False`` to share these tools across types.
    """
    if name is None:
        return ToolSpec(
            text_content,
            ctx,
            chars_per_page=chars_per_page,
            max_chars_per_page=max_chars_per_page,
            enforce_type=enforce_type,
        )
    ts = _make_toolset(ctx, name, enforce_type, "scalar")
    default_page = chars_per_page

    def _not_text(text: Any) -> dict[str, Any]:
        return {
            "error": "This tool requires string data.",
            "actual_type": type(text).__name__,
            "suggested_action": "Use the tools listed for this workspace instead.",
        }

    @ts.tool(
        name=f"{name}_read_page",
        data_shape="scalar",
        output_hint=(
            f"Navigate with {name}_read_page(workspace_id='{{workspace_id}}', "
            f"page=NEXT_PAGE). Use {name}_search_in_text() to find specific content."
        ),
    )
    def read_page(
        workspace_id: str,
        page: int = 0,
        chars_per_page: int = default_page,
    ) -> dict[str, Any]:
        """Read a page of text from a document workspace.

        Returns a slice of the text by character offset, with metadata
        about position and total pages.

        Args:
            workspace_id: The workspace containing the text.
            page: Zero-based page number.
            chars_per_page: Characters per page (default from toolset config).
        """
        text = ts.store.get_items(workspace_id)
        if not isinstance(text, str):
            return _not_text(text)
        chars_per_page = _clamp(chars_per_page, 1, max_chars_per_page)
        page = int(page)
        total_chars = len(text)
        total_pages = max(1, math.ceil(total_chars / chars_per_page))
        start = page * chars_per_page
        end = min(start + chars_per_page, total_chars)

        if page < 0 or (start >= total_chars and total_chars > 0):
            return {
                "error": f"Page {page} out of range.",
                "total_pages": total_pages,
                "suggested_action": f"Use page 0–{total_pages - 1}.",
            }

        return {
            "text": text[start:end],
            "page": page,
            "total_pages": total_pages,
            "char_offset": start,
            "chars_in_page": end - start,
            "total_chars": total_chars,
            "has_next": end < total_chars,
            "has_prev": page > 0,
        }

    @ts.tool(name=f"{name}_search_in_text", data_shape="scalar")
    def search_in_text(
        workspace_id: str,
        query: str,
        context_chars: int = 150,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """Search for text within a document and return matches with surrounding context.

        Case-insensitive search.  Each match includes the surrounding text
        so the agent can understand the context without reading the full document.

        Args:
            workspace_id: The workspace containing the text.
            query: Text to search for (case-insensitive).
            context_chars: Characters of context to include before and after each match.
            max_results: Maximum number of matches to return.
        """
        text = ts.store.get_items(workspace_id)
        if not isinstance(text, str):
            return _not_text(text)
        if not query:
            return {
                "error": "Query must not be empty.",
                "suggested_action": "Pass the text to look for.",
            }
        context_chars = _clamp(context_chars, 0, 1000)
        max_results = _clamp(max_results, 1, 50)

        text_lower = text.lower()
        query_lower = query.lower()
        matches: list[dict[str, Any]] = []
        start = 0

        while True:
            pos = text_lower.find(query_lower, start)
            if pos == -1:
                break
            if len(matches) == max_results:
                break
            ctx_start = max(0, pos - context_chars)
            ctx_end = min(len(text), pos + len(query) + context_chars)
            matches.append(
                {
                    "match_index": len(matches),
                    "char_offset": pos,
                    "context": text[ctx_start:ctx_end],
                    "page": pos // default_page,
                }
            )
            start = pos + len(query)

        return {
            "query": query,
            "match_count": len(matches),
            "has_more": len(matches) == max_results
            and text_lower.find(query_lower, start) != -1,
            "matches": matches,
            "total_chars": len(text),
        }

    @ts.tool(name=f"{name}_get_length", data_shape="scalar")
    def get_length(workspace_id: str) -> dict[str, Any]:
        """Return character count, word count, and line count of the document.

        Args:
            workspace_id: The workspace containing the text.
        """
        text = ts.store.get_items(workspace_id)
        if not isinstance(text, str):
            return _not_text(text)
        return {
            "workspace_id": workspace_id,
            "chars": len(text),
            "words": len(text.split()),
            "lines": text.count("\n") + 1,
            "pages": max(1, math.ceil(len(text) / default_page)),
        }

    return ts
