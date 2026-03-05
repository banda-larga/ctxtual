"""
Built-in utility ToolSets.

These cover the most common patterns when an agent needs to consume a
workspace of list-type data.  Import them and pass to ``@forge.producer``
instead of writing boilerplate every time.

Available factories
-------------------

- :func:`paginator`     — paginate / count / get-by-index / slice
- :func:`text_search`   — substring search / field value discovery
- :func:`filter_set`    — structured filtering and sorting
- :func:`kv_reader`     — for dict-type (single-document) workspaces
- :func:`pipeline`      — declarative data pipelines (filter→sort→group in one call)
- :func:`text_content`  — for string/text workspaces (documents, webpages, PDFs)

Usage::

    from ctxtual import Forge, MemoryStore
    from ctxtual.utils import paginator, text_search, filter_set, pipeline

    forge = Forge(store=MemoryStore())

    # Simple — name is inferred from workspace_type:
    @forge.producer(workspace_type="papers", toolsets=[
        paginator(forge),
        text_search(forge, fields=["title", "abstract"]),
        filter_set(forge),
        pipeline(forge),
    ])
    def fetch_papers(query: str):
        ...

    # Explicit — name is passed directly (still works):
    pager = paginator(forge, "papers")
    search = text_search(forge, "papers", fields=["title"])

    @forge.producer(workspace_type="papers", toolsets=[pager, search])
    def fetch_papers_v2(query: str):
        ...
"""

import inspect
import json
import math
from typing import Any

from ctxtual.forge import Forge, _infer_item_schema
from ctxtual.pipeline import PipelineEngine, PipelineError, compute_aggregates
from ctxtual.toolset import ToolSet, ToolSpec
from ctxtual.types import WorkspaceMeta, WorkspaceRef

# paginator: page through list-type workspaces


def paginator(forge: Forge, name: str | None = None) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with pagination tools for list-type workspace data.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@forge.producer``.

    Tools:

    - ``paginate(workspace_id, page, size)`` — list slice + metadata
    - ``count(workspace_id)`` — total item count
    - ``get_item(workspace_id, index)`` — single item by index
    - ``get_slice(workspace_id, start, end)`` — arbitrary slice
    """
    if name is None:
        return ToolSpec(paginator, forge)
    ts = forge.toolset(name)
    ts.data_shape = "list"

    @ts.tool(
        name=f"{name}_paginate",
        output_hint=f"Navigate with {name}_paginate(workspace_id='{{workspace_id}}', page=NEXT_PAGE). Use {name}_search() to find specific items.",
    )
    def paginate(workspace_id: str, page: int = 0, size: int = 10) -> dict[str, Any]:
        """Return a page of items from the workspace with pagination metadata.

        Args:
            workspace_id: Identifier of the workspace to paginate.
            page: Zero-based page number to retrieve.
            size: Number of items per page.
        """
        total = ts.store.count_items(workspace_id)
        chunk = ts.store.get_page(workspace_id, page * size, size)
        total_pages = max(1, math.ceil(total / size))
        return {
            "items": chunk,
            "page": page,
            "size": size,
            "total": total,
            "total_pages": total_pages,
            "has_next": (page + 1) * size < total,
            "has_prev": page > 0,
        }

    @ts.tool(name=f"{name}_count")
    def count(workspace_id: str) -> dict[str, Any]:
        """Return total number of items in the workspace.

        Args:
            workspace_id: Identifier of the workspace to count.
        """
        total = ts.store.count_items(workspace_id)
        return {"workspace_id": workspace_id, "count": total}

    @ts.tool(name=f"{name}_get_item")
    def get_item(workspace_id: str, index: int) -> Any:
        """Return a single item by zero-based index.

        Args:
            workspace_id: Identifier of the workspace to read from.
            index: Zero-based position of the item to retrieve.
        """
        total = ts.store.count_items(workspace_id)
        if not (0 <= index < total):
            return {
                "error": f"Index {index} out of range.",
                "valid_range": f"0–{max(0, total - 1)}",
                "total_items": total,
                "suggested_action": (
                    f"Use an index between 0 and {max(0, total - 1)}, "
                    f"or call {name}_paginate to browse items."
                ),
            }
        page = ts.store.get_page(workspace_id, index, 1)
        return (
            page[0]
            if page
            else {
                "error": f"Index {index} out of range.",
                "suggested_action": f"Call {name}_count to check the total.",
            }
        )

    @ts.tool(name=f"{name}_get_slice")
    def get_slice(workspace_id: str, start: int = 0, end: int = 20) -> list[Any]:
        """Return an arbitrary ``[start:end]`` slice of workspace items.

        Args:
            workspace_id: Identifier of the workspace to slice.
            start: Start index (inclusive).
            end: End index (exclusive).
        """
        return ts.store.get_page(workspace_id, start, end - start)

    return ts


# text_search — substring / field search over list workspaces


def text_search(
    forge: Forge,
    name: str | None = None,
    *,
    fields: list[str] | None = None,
) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with text-search tools for list-type workspace data.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@forge.producer``.

    Args:
        fields: If provided, search is limited to these dict keys.
                If ``None``, searches all string values in each item.

    Tools:

    - ``search(workspace_id, query, max_results, case_sensitive)`` — matched items
    - ``field_values(workspace_id, field, max_values)`` — distinct field values
    """
    if name is None:
        return ToolSpec(text_search, forge, fields=fields)
    ts = forge.toolset(name)
    ts.data_shape = "list"
    _fields = fields

    @ts.tool(name=f"{name}_search")
    def search(
        workspace_id: str,
        query: str,
        max_results: int = 20,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """Search workspace items for a substring query.

        Args:
            workspace_id: Identifier of the workspace to search.
            query: Substring to search for in item fields.
            max_results: Maximum number of matching items to return.
            case_sensitive: Whether the search should be case-sensitive.
        """
        matches = ts.store.search_items(
            workspace_id,
            query,
            fields=_fields,
            max_results=max_results,
            case_sensitive=case_sensitive,
        )

        return {
            "query": query,
            "total_matches": len(matches),
            "max_results": max_results,
            "matches": matches,
        }

    @ts.tool(name=f"{name}_field_values")
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
        values = ts.store.distinct_field_values(
            workspace_id, field, max_values=max_values
        )
        return {"field": field, "distinct_values": values, "count": len(values)}

    return ts


# filter_set — structured filtering on dict-list workspaces


def filter_set(forge: Forge, name: str | None = None) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with structured filtering and sorting tools.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@forge.producer``.

    Tools:

    - ``filter_by(workspace_id, field, value, operator)`` — filtered list
    - ``sort_by(workspace_id, field, descending, limit)`` — sorted list

    Supported operators: ``eq``, ``ne``, ``lt``, ``lte``, ``gt``, ``gte``,
    ``contains``, ``startswith``.
    """
    if name is None:
        return ToolSpec(filter_set, forge)
    ts = forge.toolset(name)
    ts.data_shape = "list"

    @ts.tool(name=f"{name}_filter_by")
    def filter_by(
        workspace_id: str,
        field: str,
        value: Any,
        operator: str = "eq",
    ) -> dict[str, Any]:
        """Filter workspace items by a field value using a comparison operator.

        Args:
            workspace_id: Identifier of the workspace to filter.
            field: Name of the field to compare.
            value: Value to compare against.
            operator: Comparison operator (eq, ne, lt, lte, gt, gte, contains, startswith).
        """
        results = ts.store.filter_items(workspace_id, field, value, operator)
        return {
            "field": field,
            "operator": operator,
            "value": value,
            "count": len(results),
            "results": results,
        }

    @ts.tool(name=f"{name}_sort_by")
    def sort_by(
        workspace_id: str,
        field: str,
        descending: bool = False,
        limit: int = 100,
    ) -> list[Any]:
        """Sort workspace items by a field and return the top results.

        Args:
            workspace_id: Identifier of the workspace to sort.
            field: Name of the field to sort by.
            descending: If true, sort in descending (high-to-low) order.
            limit: Maximum number of sorted results to return.
        """
        return ts.store.sort_items(
            workspace_id, field, descending=descending, limit=limit
        )

    return ts


# kv_reader — for single-item / dict workspaces (non-list payloads)


def kv_reader(forge: Forge, name: str | None = None) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` for dict-type workspaces (single JSON document).

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@forge.producer``.

    Tools:

    - ``get_keys(workspace_id)`` — list of top-level keys
    - ``get_value(workspace_id, key)`` — value at key
    """
    if name is None:
        return ToolSpec(kv_reader, forge)
    ts = forge.toolset(name)
    ts.data_shape = "dict"

    @ts.tool(
        name=f"{name}_get_keys",
        output_hint=f"Use {name}_get_value(workspace_id='{{workspace_id}}', key=KEY) to read each section.",
    )
    def get_keys(workspace_id: str) -> list[str]:
        """Return the top-level keys of the stored dict.

        Args:
            workspace_id: Identifier of the workspace to inspect.
        """
        data = ts.store.get_items(workspace_id)
        if isinstance(data, dict):
            return list(data.keys())
        return []

    @ts.tool(name=f"{name}_get_value")
    def get_value(workspace_id: str, key: str) -> Any:
        """Return the value at a specific top-level key.

        Args:
            workspace_id: Identifier of the workspace to read from.
            key: Top-level dictionary key to retrieve.
        """
        data = ts.store.get_items(workspace_id)
        if isinstance(data, dict):
            if key in data:
                return data[key]
            return {
                "error": f"Key '{key}' not found.",
                "available_keys": list(data.keys())[:50],
                "suggested_action": "Use one of the available_keys listed above.",
            }
        return {
            "error": "Workspace payload is not a dict.",
            "suggested_action": (
                (
                    f"This workspace contains a list. Use {name}_paginate "
                    f"or {name}_get_item instead."
                )
                if name
                else "This workspace contains a list, not a dict."
            ),
        }

    return ts


# pipeline: declarative data pipelines (compound operations in one call)


def pipeline(forge: Forge, name: str | None = None) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with pipeline tools for compound data operations.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@forge.producer``.

    This eliminates multi-round-trip patterns.  Instead of the LLM calling
    search → filter → sort → limit (4 tool calls, 4 round-trips), it
    describes the entire chain in one ``pipe`` call.  Intermediate data
    never enters the context window.

    Tools:

    - ``pipe(workspace_id, steps, save_as)`` — declarative pipeline
    - ``aggregate(workspace_id, group_by, metrics)`` — group-by aggregation
    """
    if name is None:
        return ToolSpec(pipeline, forge)
    ts = forge.toolset(name)
    ts.data_shape = "list"
    engine = PipelineEngine()

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

    @ts.tool(name=f"{name}_pipe", schema_extra=_pipe_schema_extra)
    def pipe(
        workspace_id: str,
        steps: list[dict[str, Any]],
        save_as: str | None = None,
    ) -> dict[str, Any]:
        """Execute a chain of data operations on workspace items in one call.

        Each step transforms the output of the previous step.  Only the final
        result is returned — intermediate data stays server-side, saving
        round-trips and context tokens.

        Args:
            workspace_id: The workspace to operate on.
            steps: Ordered list of operations. Each is a dict with one key
                (the operation name) and its parameters as the value.
                Operations: filter, search, sort, select, exclude, limit,
                skip, slice, sample, unique, flatten, group_by, count.
            save_as: If provided, save the result as a new workspace with
                this ID and return a summary instead of raw data.  Use when
                the pipeline result is large and you want to paginate it.

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
                "suggested_action": (
                    f"This workspace contains {type(items).__name__} data. "
                    f"Use {name}_get_keys / {name}_get_value for dict workspaces."
                ),
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

        try:
            result = engine.execute(items, steps)
        except PipelineError as exc:
            return {
                "error": str(exc),
                "available_operations": sorted(PipelineEngine.OPERATORS.keys()),
                "suggested_action": "Check operation names and parameter formats.",
                **exc.context,
            }

        # Save result as a new workspace if requested
        if save_as and isinstance(result, list):
            meta = WorkspaceMeta(
                workspace_id=save_as,
                workspace_type=name,
                item_count=len(result),
                data_shape="list",
            )
            with ts.store.transaction():
                ts.store.init_workspace(meta)
                ts.store.set_items(save_as, result)

            # Build a proper WorkspaceRef so the LLM knows exactly
            # what tools are available for the new workspace.
            tool_descriptions: dict[str, str] = {}
            for t_name, t_fn in ts._raw_tools.items():
                doc = inspect.getdoc(t_fn) or ""
                first_line = doc.split("\n")[0].strip().rstrip(".")
                if first_line:
                    tool_descriptions[t_name] = first_line

            sample_fields: list[str] = []
            item_schema = _infer_item_schema(result)
            if item_schema and "properties" in item_schema:
                sample_fields = list(item_schema["properties"].keys())

            ref = WorkspaceRef(
                workspace_id=save_as,
                workspace_type=name,
                item_count=len(result),
                data_shape="list",
                producer_fn="pipeline",
                available_tools=ts.tool_names,
                tool_descriptions=tool_descriptions,
                sample_fields=sample_fields,
                item_schema=item_schema,
            )
            ref_dict = ref.to_dict()
            ref_dict["status"] = "pipeline_result_saved"
            ref_dict["message"] = (
                f"✓ Pipeline result saved as workspace '{save_as}' "
                f"with {len(result):,} item(s). "
                f"Use the tools below to explore."
            )
            return ref_dict

        # Return result directly
        if isinstance(result, list):
            return {
                "items": result,
                "count": len(result),
            }
        # Terminal operations (count, group_by) return dicts/non-lists
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

    @ts.tool(name=f"{name}_aggregate", schema_extra=_agg_schema_extra)
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

        _metrics = metrics or {"count": "count"}

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
            return {"groups": result, "group_count": len(result)}
        return {"result": result}

    return ts


# text_content — tools for navigating raw string/text data


def text_content(forge: Forge, name: str | None = None, *, chars_per_page: int = 3000) -> ToolSet | ToolSpec:
    """
    Create a :class:`ToolSet` with tools for navigating string/text workspaces.

    If *name* is omitted, returns a :class:`ToolSpec` that is materialized
    automatically when passed to ``@forge.producer``.

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
        forge: The Forge instance.
        name:  Workspace type name (e.g. ``"doc"``).  Optional.
        chars_per_page: Default characters per page for ``read_page``.
    """
    if name is None:
        return ToolSpec(text_content, forge, chars_per_page=chars_per_page)
    ts = forge.toolset(name)
    ts.data_shape = "scalar"

    @ts.tool(
        name=f"{name}_read_page",
        output_hint=(
            f"Navigate with {name}_read_page(workspace_id='{{workspace_id}}', "
            f"page=NEXT_PAGE). Use {name}_search_in_text() to find specific content."
        ),
    )
    def read_page(
        workspace_id: str,
        page: int = 0,
        chars_per_page: int = chars_per_page,
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
            return {
                "error": "This tool requires string data.",
                "actual_type": type(text).__name__,
                "suggested_action": (
                    f"Use {name}_paginate for list data, or "
                    f"{name}_get_keys for dict data."
                ),
            }
        total_chars = len(text)
        total_pages = max(1, math.ceil(total_chars / chars_per_page))
        start = page * chars_per_page
        end = min(start + chars_per_page, total_chars)

        if start >= total_chars:
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

    @ts.tool(name=f"{name}_search_in_text")
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
            return {
                "error": "This tool requires string data.",
                "suggested_action": f"Use {name}_search for list data.",
            }

        text_lower = text.lower()
        query_lower = query.lower()
        matches: list[dict[str, Any]] = []
        start = 0

        while len(matches) < max_results:
            pos = text_lower.find(query_lower, start)
            if pos == -1:
                break
            ctx_start = max(0, pos - context_chars)
            ctx_end = min(len(text), pos + len(query) + context_chars)
            matches.append(
                {
                    "match_index": len(matches),
                    "char_offset": pos,
                    "context": text[ctx_start:ctx_end],
                    "page": pos // chars_per_page,
                }
            )
            start = pos + len(query)

        return {
            "query": query,
            "match_count": len(matches),
            "matches": matches,
            "total_chars": len(text),
        }

    @ts.tool(name=f"{name}_get_length")
    def get_length(workspace_id: str) -> dict[str, Any]:
        """Return character count, word count, and line count of the document.

        Args:
            workspace_id: The workspace containing the text.
        """
        text = ts.store.get_items(workspace_id)
        if not isinstance(text, str):
            return {
                "error": "This tool requires string data.",
                "suggested_action": "Check the workspace data_shape.",
            }
        return {
            "workspace_id": workspace_id,
            "chars": len(text),
            "words": len(text.split()),
            "lines": text.count("\n") + 1,
            "pages": max(1, math.ceil(len(text) / chars_per_page)),
        }

    return ts
