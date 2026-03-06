# ctxtual

[![PyPI version](https://img.shields.io/pypi/v/ctxtual.svg)](https://pypi.org/project/ctxtual/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/ctxtual.svg)](https://pypi.org/project/ctxtual/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/banda-larga/ctxtual/actions/workflows/ci.yml/badge.svg)](https://github.com/banda-larga/ctxtual/actions)

**The context engineering library for AI agents.**

Stop truncating tool results. Start engineering context.

```
pip install ctxtual
```

---

## Table of Contents

- [ctxtual](#ctxtual)
  - [Table of Contents](#table-of-contents)
  - [The Problem You Already Have](#the-problem-you-already-have)
  - [Why ctxtual](#why-ctxtual)
  - [Quick Start](#quick-start)
  - [Install](#install)
  - [Architecture](#architecture)
  - [Core API](#core-api)
    - [Ctx](#ctx)
    - [`@ctx.producer` — Store Data, Return a Map](#ctxproducer--store-data-return-a-map)
    - [`@ctx.consumer` — Transform and Derive](#ctxconsumer--transform-and-derive)
    - [`ctx.dispatch_tool_call()` — Single Entry Point](#ctxdispatch_tool_call--single-entry-point)
    - [Schema Export](#schema-export)
  - [Built-in ToolSets](#built-in-toolsets)
    - [`paginator(ctx, name)` — List Navigation](#paginatorctx-name--list-navigation)
    - [`text_search(ctx, name, *, fields=None)` — Full-Text Search](#text_searchctx-name--fieldsnone--full-text-search)
    - [`filter_set(ctx, name)` — Structured Filtering](#filter_setctx-name--structured-filtering)
    - [`kv_reader(ctx, name)` — Dict Workspaces](#kv_readerctx-name--dict-workspaces)
    - [`text_content(ctx, name)` — Raw Text / Scalar Navigation](#text_contentctx-name--raw-text--scalar-navigation)
    - [Text Transforms — Convert Strings to Structured Data](#text-transforms--convert-strings-to-structured-data)
    - [`pipeline(ctx, name)` — Declarative Data Pipelines](#pipelinectx-name--declarative-data-pipelines)
  - [Custom ToolSets](#custom-toolsets)
  - [Storage Backends](#storage-backends)
    - [MemoryStore](#memorystore)
    - [SQLiteStore](#sqlitestore)
    - [Custom Backends](#custom-backends)
  - [Workspace Mutations](#workspace-mutations)
  - [Framework Integrations](#framework-integrations)
    - [OpenAI](#openai)
    - [Anthropic](#anthropic)
    - [LangChain](#langchain)
  - [Concurrency \& Thread Safety](#concurrency--thread-safety)
  - [Error Recovery](#error-recovery)
  - [Advanced Patterns](#advanced-patterns)
    - [Deterministic Workspace IDs (Idempotency)](#deterministic-workspace-ids-idempotency)
    - [Multi-Hop Pipelines](#multi-hop-pipelines)
    - [Multi-Agent Collaboration](#multi-agent-collaboration)
    - [BoundToolSet (Fixed Workspace)](#boundtoolset-fixed-workspace)
    - [Result Transformation](#result-transformation)
    - [TTL \& Automatic Cleanup](#ttl--automatic-cleanup)
    - [System Prompt Generation](#system-prompt-generation)
    - [Item Schema in Notifications](#item-schema-in-notifications)
  - [Workspace Introspection](#workspace-introspection)
  - [Examples](#examples)
  - [Design Principles](#design-principles)
  - [Testing](#testing)
  - [API Reference](#api-reference)
    - [Ctx](#ctx-1)
    - [ToolSet](#toolset)
    - [Store](#store)
  - [Contributing](#contributing)
  - [License](#license)

---

## The Problem You Already Have

Your agent calls a tool. The tool returns 10,000 results. The LLM context window fits 200. You truncate to 200, and the agent misses the answer buried in result #8,432.

Every production agent team builds the same workaround: store the data somewhere, give the agent tools to explore it. **ctxtual is that workaround, extracted into a library.**

```python
# Before: raw data floods the context window
def search_papers(query: str) -> list[dict]:
    return database.search(query)  # 10,000 results → LLM chokes

# After: data is stored, agent gets a map
@ctx.producer(workspace_type="papers", toolsets=[paginator(ctx), search, filters])
def search_papers(query: str) -> list[dict]:
    return database.search(query)  # 10,000 results → stored in workspace
    # Agent receives:
    # {
    #   "workspace_id": "papers_f3a8bc12",
    #   "item_count": 10000,
    #   "data_shape": "list",
    #   "available_tools": ["papers_paginate(...)", "papers_search(...)"],
    #   "next_steps": ["• papers_paginate: Return a page of items...", ...]
    # }
```

The agent then explores with surgical precision — paginating, searching, filtering — pulling only what it needs into the context window. **The data stays server-side. The agent stays smart.**

---

## Why ctxtual

| Problem | Without ctxtual | With ctxtual |
|---------|---------------------|-------------------|
| Tool returns 10K items | Truncate and hope | Store, paginate, search on demand |
| Agent needs to filter by field | Load everything, pray it fits | `filter_by(field="year", value=2024, operator="gte")` |
| Agent needs one specific item | Return entire list for index lookup | `get_item(index=42)` |
| Tool returns large text (HTML, PDF) | Dump entire string into context | `text_content` — char-based pagination, in-text search |
| Multiple tools return overlapping data | Duplicate data across messages | Workspaces with deterministic IDs, deduplication built-in |
| Agent crashes mid-conversation | All data lost | `SQLiteStore` persists across restarts |
| LLM calls wrong tool name | Unhandled `KeyError`, agent loop crashes | Structured error dict: `{"error": ..., "available_tools": [...], "suggested_action": ...}` |
| Multi-agent sharing data | Custom IPC | Shared store, workspace-level isolation |
| Framework lock-in | Rewrite for every framework | Adapters for OpenAI, Anthropic, LangChain |

**ctxtual is not a framework.** It's a library. It doesn't own your agent loop, your LLM calls, or your data. It's the plumbing between "tool returns data" and "agent explores data."

---

## Quick Start

```python
from ctxtual import Ctx, MemoryStore
from ctxtual.utils import paginator, text_search, filter_set, pipeline

ctx = Ctx(store=MemoryStore())

# Wrap your data-fetching function — toolsets get their name from workspace_type
@ctx.producer(workspace_type="papers", toolsets=[
    paginator(ctx),
    text_search(ctx, fields=["title", "abstract"]),
    filter_set(ctx),
    pipeline(ctx),
])
def search_papers(query: str, limit: int = 10_000) -> list[dict]:
    return database.search(query, limit)

# Agent calls the producer → gets a notification, not 10K items
ref = search_papers("machine learning")
ws_id = ref["workspace_id"]

# Agent explores using consumer tools:
ctx.dispatch_tool_call("papers_paginate",  {"workspace_id": ws_id, "page": 0, "size": 10})
ctx.dispatch_tool_call("papers_search",    {"workspace_id": ws_id, "query": "transformer"})
ctx.dispatch_tool_call("papers_filter_by", {"workspace_id": ws_id, "field": "year", "value": 2024, "operator": "gte"})

# Or: compound operations in ONE call (no round-trips, no intermediate context)
ctx.dispatch_tool_call("papers_pipe", {
    "workspace_id": ws_id,
    "steps": [
        {"search": {"query": "transformer"}},
        {"filter": {"year": {"$gte": 2024}}},
        {"sort": {"field": "citations", "order": "desc"}},
        {"limit": 5},
    ],
})
```

---

## Install

```bash
pip install ctxtual     # or: uv add ctxtual
```

**Zero required dependencies.** The core library uses only the Python standard library.

Optional extras:

```bash
pip install ctxtual[dev]   # pytest, ruff, coverage
```

Integration adapters (OpenAI, Anthropic, LangChain) are zero-dependency by default — they duck-type against SDK objects. Install the SDK you need separately:

```bash
pip install openai              # for ctxtual.integrations.openai
pip install anthropic           # for ctxtual.integrations.anthropic
pip install langchain-core      # for ctxtual.integrations.langchain
```

---

## Architecture

```
Your Agent Loop
  LLM ←→ tool_calls ←→ ctx.dispatch_tool_call()

Ctx (orchestrator)
  @producer / @consumer decorators
  Schema export (OpenAI, Anthropic, LangChain format)
  Dispatch routing — one method handles all tool calls
  Thread-safe (RLock)

ToolSets (consumer tools)
  paginator · text_search · filter_set
  kv_reader · text_content · pipeline
  + your custom domain tools

Store (pluggable backend)
  MemoryStore — fast, in-process, LRU eviction
  SQLiteStore — persistent, FTS5 search, WAL mode
  BaseStore   — subclass for Redis, Postgres, S3, etc.
```

**Data flow:**
1. **Producer** runs your function, stores the result in a workspace, returns a `WorkspaceRef` notification to the LLM.
2. **LLM** reads the notification, sees available tools, calls them.
3. **Consumer tools** (paginate, search, filter, etc.) read from the store and return only the requested slice.
4. **The LLM never sees the full dataset.** It sees pages, search results, filtered subsets — exactly what it needs.

---

## Core API

### Ctx

The central orchestrator. One per application (or per agent session).

```python
from ctxtual import Ctx, MemoryStore, SQLiteStore

# In-memory (default) — fast, test-friendly, process-scoped
ctx = Ctx(store=MemoryStore())

# Persistent — survives process restarts
ctx = Ctx(store=SQLiteStore("agent.db"))

# With configuration
ctx = Ctx(
    store=MemoryStore(max_workspaces=500),  # LRU eviction when exceeded
    default_ttl=3600,     # Workspaces expire after 1 hour
    max_items=100_000,    # Reject payloads larger than 100K items
    default_notify=True,  # Producers return dicts (not WorkspaceRef objects)
)
```

### `@ctx.producer` — Store Data, Return a Map

Wraps any function so its return value is stored in a workspace. The agent gets a self-describing notification instead of raw data.

```python
@ctx.producer(
    workspace_type="papers",           # Logical data category
    toolsets=[pager, search, filters], # Consumer tools for this data
    key="papers_{query}",              # Deterministic workspace ID (optional)
    transform=lambda r: r[:5000],      # Pre-store transform (optional)
    meta={"source": "arxiv"},          # Metadata attached to workspace
    ttl=1800,                          # Time-to-live in seconds
    notify=True,                       # True=dict, False=WorkspaceRef object
)
def search_papers(query: str) -> list[dict]:
    return external_api.search(query)
```

**`key` parameter** controls workspace identity:

| Key | Behavior | Use case |
|-----|----------|----------|
| `None` (default) | Auto UUID: `papers_f3a8bc12a0` | Every call creates a new workspace |
| `"papers_{query}"` | Templated from kwargs | Idempotent — same args = same workspace (overwritten) |
| `lambda kw: f"ws_{kw['user_id']}"` | Custom callable | Full control over deduplication logic |

### `@ctx.consumer` — Transform and Derive

Consumers read from one workspace and optionally produce a new one. This enables multi-hop agent pipelines.

```python
@ctx.consumer(
    workspace_type="raw_data",               # Input workspace type
    produces="cleaned_data",                 # Output workspace type
    produces_toolsets=[clean_pager],          # Tools for the output
)
def clean_and_filter(workspace_id: str, forge_ctx: ConsumerContext):
    raw = forge_ctx.get_items()
    cleaned = [normalize(item) for item in raw if item["quality"] > 0.8]
    return forge_ctx.emit(cleaned, meta={"derived_from": workspace_id})
```

`ConsumerContext` is injected automatically. It provides:

| Method | Description |
|--------|-------------|
| `get_items(workspace_id=None)` | Read items from the input workspace (or any workspace) |
| `emit(payload, *, workspace_type, meta, ttl)` | Store derived data as a new workspace, return `WorkspaceRef` dict |
| `store` | Direct access to the store backend |

### `ctx.dispatch_tool_call()` — Single Entry Point

Route any tool call — producers and consumers — through one method. This is what your agent loop calls.

```python
result = ctx.dispatch_tool_call("papers_search", {"workspace_id": ws_id, "query": "attention"})
```

**Always returns a value, never raises on tool errors:**

```python
# Success → tool result (any type)
{"matches": [...], "total_matches": 42}

# Unknown tool → structured error dict
{"error": "Tool 'bad_name' not found.", "available_tools": [...], "suggested_action": "..."}

# Workspace not found → structured error with available workspaces
{"error": "Workspace 'ws_bad' not found.", "available_workspaces": ["ws_real"], "suggested_action": "..."}
```

This means your agent loop needs **zero try/except for tool calls**. The LLM reads the error and self-corrects.

### Schema Export

Export tool schemas in OpenAI function-calling format. Pass them to any LLM that supports tool use.

```python
# All tools (producers + consumers)
tools = ctx.get_tools()

# Only producer tools
producers = ctx.get_producer_schemas()

# All consumer tools (optionally scoped to a workspace)
consumers = ctx.get_all_tool_schemas(workspace_id="papers_abc")
```

Schemas include:
- **Rich parameter descriptions** extracted from docstrings (Google-style and Sphinx-style)
- **Proper JSON Schema types** for `Optional`, `Union`, `Literal`, `list[X]`, `dict[K,V]`, enums
- **Well-known parameter descriptions** for common names like `workspace_id`, `page`, `query`
- **Examples on complex parameters** — pipeline `steps` and `metrics` include concrete JSON examples that help LLMs construct valid queries
- **`schema_extra` support** — pass `schema_extra={"param": {"minimum": 0}}` to `@ts.tool()` to enrich any parameter's schema

---

## Built-in ToolSets

These cover 90% of what agents need. Import them, pass to `@producer`, done. The `name` parameter is **optional** — when omitted, the toolset inherits its name from the producer's `workspace_type`:

```python
# Simple — name inferred from workspace_type:
@ctx.producer(workspace_type="papers", toolsets=[paginator(ctx), text_search(ctx)])
def fetch(query): ...

# Explicit — still works for advanced use:
pager = paginator(ctx, "papers")
```

### `paginator(ctx, name)` — List Navigation

```python
from ctxtual.utils import paginator
pager = paginator(ctx, "papers")  # data_shape="list"
```

| Tool | Description |
|------|-------------|
| `{name}_paginate(workspace_id, page=0, size=10)` | Page of items + metadata (total, has_next, has_prev) |
| `{name}_count(workspace_id)` | Total item count |
| `{name}_get_item(workspace_id, index)` | Single item by zero-based index |
| `{name}_get_slice(workspace_id, start=0, end=20)` | Arbitrary slice |

### `text_search(ctx, name, *, fields=None)` — Full-Text Search

```python
from ctxtual.utils import text_search
search = text_search(ctx, "papers", fields=["title", "abstract"])  # data_shape="list"
```

| Tool | Description |
|------|-------------|
| `{name}_search(workspace_id, query, max_results=20, case_sensitive=False)` | BM25-ranked full-text search (FTS5 on SQLiteStore, TF scoring on MemoryStore) |
| `{name}_field_values(workspace_id, field, max_values=50)` | Distinct values for a field (facet discovery) |

### `filter_set(ctx, name)` — Structured Filtering

```python
from ctxtual.utils import filter_set
filters = filter_set(ctx, "papers")  # data_shape="list"
```

| Tool | Description |
|------|-------------|
| `{name}_filter_by(workspace_id, field, value, operator="eq")` | Filter by field. Operators: `eq`, `ne`, `lt`, `lte`, `gt`, `gte`, `contains`, `startswith` |
| `{name}_sort_by(workspace_id, field, descending=False, limit=100)` | Sort by field |

### `kv_reader(ctx, name)` — Dict Workspaces

For single-document workspaces (config, metadata, API responses that are dicts, not lists).

```python
from ctxtual.utils import kv_reader
kv = kv_reader(ctx, "config")  # data_shape="dict"
```

| Tool | Description |
|------|-------------|
| `{name}_get_keys(workspace_id)` | List top-level keys |
| `{name}_get_value(workspace_id, key)` | Read value at key |

### `text_content(ctx, name)` — Raw Text / Scalar Navigation

For producers that return a **string** (HTML page, PDF text, log file, API response body). Instead of stuffing the entire string into context, the agent navigates it with character-based pagination and search.

```python
from ctxtual.utils import text_content
reader = text_content(ctx, "page")  # data_shape="scalar"
```

| Tool | Description |
|------|-------------|
| `{name}_read_page(workspace_id, page=0, chars_per_page=4000)` | Character-based page with `has_next`/`has_prev`, word count, navigation hints |
| `{name}_search_in_text(workspace_id, query, context_chars=100, max_results=20)` | Find occurrences with surrounding context, character offsets |
| `{name}_get_length(workspace_id)` | Character, word, and line counts |

**Example — webpage producer:**

```python
@ctx.producer(workspace_type="page", toolsets=[reader])
def read_webpage(url: str) -> str:
    return requests.get(url).text  # Could be 100KB of HTML

# Agent receives WorkspaceRef with item_count=1, data_shape="scalar"
# and tools: page_read_page, page_search_in_text, page_get_length
```

### Text Transforms — Convert Strings to Structured Data

When you want to use **list-based** tools (paginator, filter, pipeline) with text content, transform the string before storing:

```python
from ctxtual import chunk_text, split_sections, split_markdown_sections

# Fixed-size overlapping chunks (good for embeddings, RAG)
@ctx.producer(workspace_type="chunks", toolsets=[pager, search])
def ingest_document(path: str) -> list[dict]:
    text = open(path).read()
    return chunk_text(text, chunk_size=2000, overlap=200)
    # → [{"chunk_index": 0, "text": "...", "char_offset": 0}, ...]

# Split by blank lines (paragraphs)
@ctx.producer(workspace_type="paragraphs", toolsets=[pager])
def split_doc(text: str) -> list[dict]:
    return split_sections(text, separator="\n\n")
    # → [{"section_index": 0, "text": "First paragraph..."}, ...]

# Split by Markdown headers
@ctx.producer(workspace_type="sections", toolsets=[pager, search])
def parse_markdown(content: str) -> list[dict]:
    return split_markdown_sections(content)
    # → [{"section_index": 0, "heading": "Introduction", "level": 1, "text": "..."}, ...]
```

All transforms pass non-strings through unchanged, so they're safe in pipelines that might receive mixed data.

### `pipeline(ctx, name)` — Declarative Data Pipelines

The most powerful built-in. Instead of the LLM making 4+ round-trips
(search → filter → sort → limit), it describes the entire chain in **one tool call**.
Intermediate data stays server-side — only the final result enters context.

This is ctxtual's answer to [programmatic tool calling](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/programmatic-tool-calling):
compound operations in a single step, framework-agnostic, works with any LLM.

```python
from ctxtual.utils import pipeline
pipe = pipeline(ctx, "papers")  # data_shape="list"
```

| Tool | Description |
|------|-------------|
| `{name}_pipe(workspace_id, steps, save_as=None)` | Chain operations declaratively. `save_as` stores the result as a new browsable workspace. |
| `{name}_aggregate(workspace_id, group_by=None, metrics=None)` | Group-by aggregation with computed statistics |

**Pipeline operations** (each step is `{op_name: params}`):

| Operation | Example | Description |
|-----------|---------|-------------|
| `filter` | `{"filter": {"year": {"$gte": 2023}}}` | MongoDB-style conditions: `$gt`, `$gte`, `$lt`, `$lte`, `$ne`, `$in`, `$nin`, `$contains`, `$startswith`, `$regex`, `$exists`. Logical: `$or`, `$and`, `$not`. Dot notation: `"author.name"` |
| `search` | `{"search": {"query": "ML", "fields": ["title"]}}` | Case-insensitive text search across all or specified fields |
| `sort` | `{"sort": {"field": "score", "order": "desc"}}` | Single or multi-field sort. Multi: `[{"field": "year"}, {"field": "score", "order": "desc"}]` |
| `select` | `{"select": ["title", "author"]}` | Keep only these fields |
| `exclude` | `{"exclude": ["raw_blob"]}` | Remove these fields |
| `limit` | `{"limit": 10}` | First N items |
| `skip` | `{"skip": 5}` | Skip first N items |
| `slice` | `{"slice": [5, 15]}` | Items `[5:15]` |
| `sample` | `{"sample": {"n": 10, "seed": 42}}` | Random sample (reproducible with seed) |
| `unique` | `{"unique": "author"}` | Deduplicate by field, keep first |
| `flatten` | `{"flatten": "tags"}` | Expand list fields into multiple items |
| `group_by` | `{"group_by": {"field": "category", "metrics": {"n": "count", "avg": "mean:score"}}}` | Aggregation. Metrics: `count`, `sum:f`, `mean:f`, `min:f`, `max:f`, `median:f`, `stddev:f`, `values:f` |
| `count` | `{"count": true}` | Terminal: return `{"count": N}` |

**Example — what would take 4 tool calls in 1:**

```python
result = ctx.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [
        {"search": {"query": "neural networks"}},
        {"filter": {"year": {"$gte": 2023}}},
        {"sort": {"field": "citations", "order": "desc"}},
        {"limit": 5},
        {"select": ["title", "author", "citations"]},
    ],
})
# → {"items": [...], "count": 5}  — one call, zero intermediate context
```

**Example — tag frequency analysis (flatten → group → sort):**

```python
result = ctx.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [
        {"flatten": "tags"},
        {"group_by": {"field": "tags", "metrics": {"count": "count"}}},
        {"sort": {"field": "count", "order": "desc"}},
        {"limit": 10},
    ],
})
```

**Example — save pipeline result as a new workspace:**

```python
result = ctx.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [{"filter": {"author": "Alice"}}, {"sort": {"field": "year"}}],
    "save_as": "alice_papers",  # stored as a new workspace
})
# LLM can then paginate "alice_papers" with papers_paginate
```

---

## Custom ToolSets

When built-ins aren't enough, create domain-specific tools:

```python
analytics = ctx.toolset("transactions")
analytics.data_shape = "list"  # Validates payload shape at both produce-time and tool-time

@analytics.tool(
    name="transactions_anomalies",
    output_hint="Investigate flagged transactions with transactions_get_item(workspace_id='{workspace_id}', index=INDEX).",
)
def detect_anomalies(workspace_id: str, std_threshold: float = 2.0) -> dict:
    """Flag transactions with amounts that deviate significantly from the mean.

    Args:
        workspace_id: The transactions workspace to analyze.
        std_threshold: Number of standard deviations to flag as anomaly.
    """
    items = analytics.store.get_items(workspace_id)
    # ... your domain logic ...
    return {"anomalies": flagged, "count": len(flagged)}
```

**`output_hint`** is appended to the tool result as a `_hint` field, making every tool self-describing. The `{workspace_id}` placeholder is replaced at runtime. The result envelope is always `{"result": <original>, "_hint": "<hint>"}` — the original return shape is never mutated.

**`data_shape`** validation:
- Set `data_shape="list"` on toolsets that expect list data (paginator, search, filter, pipeline)
- Set `data_shape="dict"` on toolsets that expect dict data (kv_reader)
- Set `data_shape="scalar"` on toolsets that expect raw strings (text_content)
- At producer time: logs a warning if the payload shape doesn't match
- At tool-call time: returns a structured error dict with `expected_shape`, `actual_shape`, `suggested_action`
- **Shape-aware WorkspaceRef**: incompatible tools are automatically filtered from `available_tools` — if a producer returns a string, the LLM only sees `text_content` tools, not paginator tools

---

## Storage Backends

### MemoryStore

```python
from ctxtual import MemoryStore

store = MemoryStore(max_workspaces=500)  # Optional LRU eviction
```

Fast, thread-safe (RLock), zero dependencies. Data lives in process memory.

### SQLiteStore

```python
from ctxtual import SQLiteStore

store = SQLiteStore("agent.db")          # Persistent file
store = SQLiteStore(":memory:")          # In-memory SQLite (for testing)
```

Per-row storage in `cf_items` table enables SQL-pushed queries — `LIMIT`/`OFFSET`, `json_extract` `WHERE`, `ORDER BY` all happen at the SQL level, not in Python. Full-text search uses **FTS5 with Porter stemming** and `bm25()` ranking.

**Schema:**
- `cf_meta` — workspace metadata (one row per workspace)
- `cf_items` — per-item storage (one row per item per workspace)
- `cf_fts` — FTS5 virtual table for full-text search
- Automatic migration on open (adds new columns to old databases)

### Custom Backends

Implement `BaseStore` for Redis, Postgres, S3, or any storage:

```python
from ctxtual import BaseStore
from ctxtual.types import WorkspaceMeta

class RedisStore(BaseStore):
    # Required — implement these 7 methods:
    def init_workspace(self, meta: WorkspaceMeta) -> None: ...
    def drop_workspace(self, workspace_id: str) -> None: ...
    def get_meta(self, workspace_id: str) -> WorkspaceMeta | None: ...
    def list_workspaces(self, workspace_type: str | None = None) -> list[str]: ...
    def set_items(self, workspace_id: str, payload: Any) -> None: ...
    def get_items(self, workspace_id: str) -> Any: ...
    def set(self, workspace_id: str, key: str, value: Any) -> None: ...
    def get(self, workspace_id: str, key: str, default: Any = None) -> Any: ...

    # Optional — override for performance (defaults iterate over get_items):
    def count_items(self, workspace_id: str) -> int: ...
    def get_page(self, workspace_id: str, offset: int, limit: int) -> list: ...
    def search_items(self, workspace_id, query, *, fields=None, max_results=20, case_sensitive=False) -> list: ...
    def filter_items(self, workspace_id, field, value, operator="eq") -> list: ...
    def sort_items(self, workspace_id, field, *, descending=False, limit=100) -> list: ...

    # Optional — override for mutation support:
    def append_items(self, workspace_id: str, new_items: list) -> int: ...
    def update_item(self, workspace_id: str, index: int, item: Any) -> None: ...
    def patch_item(self, workspace_id: str, index: int, fields: dict) -> None: ...
    def delete_items(self, workspace_id: str, indices: list[int]) -> int: ...
```

All query and mutation methods have default implementations that work on any store — override them only for performance.

---

## Workspace Mutations

Workspaces are not read-only. Agents can modify data in place:

```python
store = ctx.store

# Append new items to an existing workspace
store.append_items("tasks_sprint_1", [{"title": "New task", "status": "todo"}])

# Update an item by index (full replacement)
store.update_item("tasks_sprint_1", 0, {"title": "Updated", "status": "done"})

# Patch specific fields on an item (merge, not replace)
store.patch_item("tasks_sprint_1", 2, {"status": "in_progress"})

# Delete items by indices
store.delete_items("tasks_sprint_1", [5, 6, 7])
```

All mutations maintain FTS index consistency on SQLiteStore.

---

## Framework Integrations

### OpenAI

```python
from ctxtual.integrations.openai import to_openai_tools, has_tool_calls, handle_tool_calls

# Get tool schemas
tools = to_openai_tools(ctx)

# In your agent loop
response = client.chat.completions.create(model="gpt-5-mini", messages=messages, tools=tools)

if has_tool_calls(response):
    # Dispatch all tool calls, get tool-result messages
    tool_messages = handle_tool_calls(ctx, response)
    messages.append(response.choices[0].message)
    messages.extend(tool_messages)
```

### Anthropic

```python
from ctxtual.integrations.anthropic import to_anthropic_tools, has_tool_use, handle_tool_use

tools = to_anthropic_tools(ctx)  # Anthropic's flat schema format

response = client.messages.create(model="claude-sonnet-4.6", tools=tools, messages=messages)

if has_tool_use(response):
    tool_results = handle_tool_use(ctx, response)  # Returns tool_result content blocks
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
```

### LangChain

```python
from ctxtual.integrations.langchain import to_langchain_tools

# Returns list of StructuredTool instances — plug into any LangChain agent
tools = to_langchain_tools(ctx)
agent = create_react_agent(llm, tools)
```

All adapters are **zero-hard-dependency** — they duck-type against SDK objects and raw dicts. Install the SDK you need; the adapter works with whatever version you have.

---

## Concurrency & Thread Safety

ctxtual is designed for production web servers (FastAPI, Django, Flask) serving concurrent agent sessions:

- **`Ctx`** — `threading.RLock` protects all registration dicts
- **`MemoryStore`** — `threading.RLock` protects all data access; `get_meta()` returns deep copies
- **`SQLiteStore`** — per-thread connections, `threading.RLock`, `WAL` journal mode
- **Transactions** — `store.transaction()` context manager for atomic multi-step operations (nesting supported in SQLiteStore)

```python
# Safe: one Ctx instance, many concurrent requests
ctx = Ctx(store=MemoryStore(max_workspaces=1000), default_ttl=1800)

@app.post("/search")
async def search(query: str):
    return search_papers(query=query)  # Thread-safe

@app.get("/explore/{workspace_id}")
async def explore(workspace_id: str, page: int = 0):
    return ctx.dispatch_tool_call("papers_paginate", {"workspace_id": workspace_id, "page": page})
```

---

## Error Recovery

LLMs make mistakes. ctxtual never crashes — it teaches the LLM to self-correct.

| LLM Mistake | Response |
|-------------|----------|
| Calls unknown tool | `{"error": "Tool 'bad_name' not found.", "available_tools": [...], "suggested_action": "Use one of the available tools."}` |
| Wrong workspace_id | `{"error": "Workspace 'ws_bad' not found.", "available_workspaces": ["ws_real"], "suggested_action": "Use one of the available workspaces."}` |
| Workspace expired | `{"error": "Workspace 'ws_old' has expired.", "suggested_action": "Call load_papers() to create a fresh workspace."}` |
| Type mismatch | `{"error": "...", "expected_type": "papers", "actual_type": "employees", "workspaces_of_correct_type": [...]}` |
| Shape mismatch | `{"error": "This tool expects 'list' data but workspace contains 'dict' data.", "suggested_action": "Use get_keys/get_value instead."}` |
| Index out of range | `{"error": "Index 999 out of range.", "valid_range": "0–42", "total_items": 43, "suggested_action": "..."}` |
| Missing dict key | `{"error": "Key 'bad' not found.", "available_keys": ["host", "port"], "suggested_action": "..."}` |

Every error includes `suggested_action` — the LLM reads it and knows exactly what to do next.

---

## Advanced Patterns

### Deterministic Workspace IDs (Idempotency)

```python
@ctx.producer(workspace_type="inventory", toolsets=[pager], key="inv_{warehouse}")
def sync_inventory(warehouse: str) -> list:
    return fetch_from_wms(warehouse)

sync_inventory(warehouse="us-east")  # Creates workspace "inv_us-east"
sync_inventory(warehouse="us-east")  # Overwrites same workspace — no duplicates
```

### Multi-Hop Pipelines

```python
# Step 1: Load raw data
@ctx.producer(workspace_type="raw", toolsets=[raw_pager])
def ingest(source: str) -> list: ...

# Step 2: Filter and enrich
@ctx.consumer(workspace_type="raw", produces="clean", produces_toolsets=[clean_pager])
def clean(workspace_id: str, forge_ctx: ConsumerContext):
    data = forge_ctx.get_items()
    return forge_ctx.emit([normalize(d) for d in data if d["quality"] > 0.5])

# Step 3: Aggregate into a report
@ctx.consumer(workspace_type="clean", produces="report", produces_toolsets=[report_kv])
def summarize(workspace_id: str, forge_ctx: ConsumerContext):
    items = forge_ctx.get_items()
    return forge_ctx.emit({"total": len(items), "summary": aggregate(items)})
```

Each step creates a new workspace. The agent can explore any level.

### Multi-Agent Collaboration

Multiple agents share one store. Each agent reads/writes workspaces. Workspace metadata tracks lineage.

```python
shared_store = SQLiteStore("shared.db")
ctx = Ctx(store=shared_store)

# Agent A: collect data
@ctx.producer(workspace_type="raw", toolsets=[...], meta={"agent": "collector"})
def collect(topic: str) -> list: ...

# Agent B: analyze
@ctx.consumer(workspace_type="raw", produces="analysis", meta={"agent": "analyst"})
def analyze(workspace_id: str, forge_ctx: ConsumerContext): ...

# Agent C: write report from analysis
@ctx.consumer(workspace_type="analysis", produces="report", meta={"agent": "writer"})
def report(workspace_id: str, forge_ctx: ConsumerContext): ...
```

### BoundToolSet (Fixed Workspace)

When an agent is focused on one workspace, bind it once:

```python
bound = pager.bind("papers_abc123")
bound.papers_paginate(page=0)    # No workspace_id needed
bound.papers_get_item(index=5)   # Cleaner API for sub-agents
```

### Result Transformation

Pre-process data before storage — normalize, deduplicate, trim:

```python
@ctx.producer(
    workspace_type="papers",
    toolsets=[pager],
    transform=lambda papers: [
        {k: v for k, v in p.items() if k in ("title", "abstract", "year")}
        for p in papers
    ],
)
def fetch_papers(query: str) -> list: ...
```

### TTL & Automatic Cleanup

```python
ctx = Ctx(store=MemoryStore(), default_ttl=3600)  # 1 hour default

# Override per-producer
@ctx.producer(workspace_type="cache", toolsets=[pager], ttl=300)  # 5 min
def fetch_live_prices() -> list: ...

# Sweep expired workspaces (call periodically in production)
expired_ids = ctx.sweep_expired()
```

### System Prompt Generation

```python
# Auto-generate a system prompt from registered producers and tools.
# Includes: workspace pattern, producer descriptions, pipeline syntax
# (if registered), exploration tools, and error recovery guidance.
system = ctx.system_prompt(preamble="You are a research assistant.")
```

The generated prompt adapts to your ctx configuration — if you register
pipeline tools, it includes pipeline syntax; if you register search, it
mentions search. Each producer is listed by name with its docstring summary.

### Item Schema in Notifications

Producer tool calls return a self-describing `WorkspaceRef` that includes a
JSON Schema for the item structure, so the LLM can construct filter/sort/pipeline
queries **immediately** without paginating first to discover the data:

```python
ref = search_papers("transformers")
# ref["item_schema"]     → {"type": "object", "properties": {"title": {"type": "string"},
#                            "year": {"type": "integer"}, "citations": {"type": "integer"}, ...},
#                            "required": ["title", "year", ...]}
# ref["available_tools"] → ["papers_paginate(...)", "papers_pipe(...)", ...]
# ref["next_steps"]      → ["• papers_paginate: Return a page of items...", ...]
```

The schema tells the LLM **types** — it knows `year` is an integer (so `$gte`/`$lte` work),
`tags` is an array (so `$contains` works), `name` is a string (so `$startswith` works).
Fields not present in every item are omitted from `required`.

---

## Workspace Introspection

```python
ctx.list_workspaces()                    # All workspace IDs
ctx.list_workspaces("papers")            # Filtered by type

meta = ctx.workspace_meta("papers_abc")
meta.workspace_id          # "papers_abc"
meta.workspace_type        # "papers"
meta.data_shape            # "list", "dict", or "scalar"
meta.item_count            # 8234
meta.producer_fn           # "search_papers"
meta.producer_kwargs       # {"query": "ml", "limit": 10000}
meta.created_at            # Unix timestamp
meta.last_accessed_at      # Unix timestamp
meta.ttl                   # 3600.0 or None
meta.is_expired            # True/False
meta.extra                 # {"source": "arxiv"}

ctx.drop_workspace("papers_abc")
ctx.clear()                              # Drop all workspaces
```

---

## Examples

The `examples/` directory contains production-pattern examples organized from beginner to advanced:

| # | Example | What It Shows |
|---|---------|---------------|
| 01 | **quickstart.py** | The 20-line pattern — copy and customize |
| 02 | **rag_support_agent.py** | Search → filter → read knowledge base articles |
| 03 | **data_pipeline.py** | Producer → Consumer → Consumer with workspace lineage |
| 04 | **custom_tools.py** | Financial analytics: anomaly detection, aggregation, date ranges |
| 05 | **pipelines.py** | Compound operations in one call: filter→sort→group→save |
| 06 | **persistence.py** | SQLite + mutations: `patch_item`, `append_items`, process restart |
| 07 | **error_handling.py** | Every failure mode and how the LLM self-corrects |
| 08 | **multi_agent.py** | Collector → Analyst → Writer sharing one store |
| 09 | **openai_agent.py** | Full OpenAI agent loop with schema export and tool dispatch |
| 10 | **anthropic_agent.py** | Anthropic Claude code review agent with dict + list workspaces |
| 11 | **concurrent_server.py** | FastAPI server with 20 concurrent sessions, thread safety, TTL |

Run any example:

```bash
uv run python examples/01_quickstart.py
```

---

## Design Principles

**1. The LLM never sees bulk data.** Producers store data and return a map. The agent pulls what it needs.

**2. Tools are self-describing.** Every `WorkspaceRef` includes tool names, descriptions, and call examples. Every error includes `suggested_action`. The agent doesn't need system prompt instructions to use the tools.

**3. No framework lock-in.** ctxtual is a library, not a framework. It doesn't own your agent loop. Use it with OpenAI, Anthropic, LangChain, or raw HTTP — the adapters are thin and optional.

**4. Errors are data, not exceptions.** `dispatch_tool_call()` returns structured error dicts. Your agent loop doesn't need try/except. The LLM reads the error and self-corrects.

**5. Zero dependencies.** The core library uses only the Python standard library. You don't inherit someone else's dependency tree.

**6. Thread-safe by default.** One Ctx instance handles concurrent requests. No external locking required.

---

## Testing

```bash
uv run pytest tests/ -v          # 456 tests
uv run ruff check src/ tests/    # Lint
```

Test coverage includes:
- Core ctx operations (producers, consumers, dispatch)
- Both storage backends (MemoryStore, SQLiteStore)
- All built-in toolsets (paginator, search, filter, kv_reader, pipeline)
- All integration adapters (OpenAI, Anthropic, LangChain)
- Concurrency (11 threading tests with parallel sessions)
- Schema quality (43 tests for type mapping and docstring extraction)
- Search quality (17 tests for relevance ranking and FTS5)
- Error recovery (12 tests for LLM-friendly error dicts)
- Data shape validation (13 tests for producer-consumer contracts)
- Pipeline engine (73 tests for all 13 operators and compound pipelines)
- **LLM interface quality** (36 tests for data preview, system prompt, schema examples, end-to-end workflows)

---

## API Reference

### Ctx

| Method | Description |
|--------|-------------|
| `Ctx(store, *, default_notify, default_ttl, max_items)` | Create orchestrator |
| `@ctx.producer(workspace_type, toolsets, key, transform, meta, notify, ttl)` | Producer decorator |
| `@ctx.consumer(workspace_type, produces, produces_toolsets)` | Consumer decorator |
| `ctx.toolset(name, *, enforce_type)` | Create/get a ToolSet |
| `ctx.dispatch_tool_call(tool_name, arguments)` | Route a tool call |
| `ctx.get_tools(workspace_id=None)` | All tool schemas (OpenAI format) |
| `ctx.get_producer_schemas()` | Producer-only schemas |
| `ctx.get_all_tool_schemas(workspace_id=None)` | Consumer-only schemas |
| `ctx.system_prompt(preamble="")` | Auto-generated system prompt |
| `ctx.list_workspaces(workspace_type=None)` | List workspace IDs |
| `ctx.workspace_meta(workspace_id)` | Get workspace metadata |
| `ctx.drop_workspace(workspace_id)` | Delete a workspace |
| `ctx.sweep_expired()` | Delete expired workspaces |
| `ctx.clear()` | Delete all workspaces |

### ToolSet

| Method | Description |
|--------|-------------|
| `ToolSet(name, *, enforce_type, safe, data_shape)` | Create tool group |
| `@toolset.tool(name, validate_workspace, output_hint)` | Register a tool |
| `toolset.bind(workspace_id)` | Create BoundToolSet with fixed workspace |
| `toolset.tools` | Dict of `{name: callable}` |
| `toolset.tool_names` | List of tool names |
| `toolset.to_tool_schemas(workspace_id=None)` | OpenAI schemas for this toolset |

### Store

| Method | Description |
|--------|-------------|
| `init_workspace(meta)` | Create/update workspace |
| `drop_workspace(workspace_id)` | Delete workspace and all data |
| `get_meta(workspace_id)` | Get metadata (or None) |
| `list_workspaces(workspace_type=None)` | List workspace IDs |
| `set_items(workspace_id, payload)` | Store payload (list, dict, or scalar) |
| `get_items(workspace_id)` | Read full payload |
| `count_items(workspace_id)` | Item count |
| `get_page(workspace_id, offset, limit)` | Paginated read |
| `search_items(workspace_id, query, *, fields, max_results, case_sensitive)` | Full-text search |
| `filter_items(workspace_id, field, value, operator)` | Structured filter |
| `sort_items(workspace_id, field, *, descending, limit)` | Sorted read |
| `distinct_field_values(workspace_id, field, *, max_values)` | Facet values |
| `append_items(workspace_id, new_items)` | Append to list workspace |
| `update_item(workspace_id, index, item)` | Replace item at index |
| `patch_item(workspace_id, index, fields)` | Merge fields into item |
| `delete_items(workspace_id, indices)` | Remove items by index |
| `transaction()` | Atomic operation context manager |
| `sweep_expired()` | Delete expired workspaces |

---

## Contributing

Contributions are welcome. Please open an issue to discuss before submitting large changes.

```bash
git clone https://github.com/banda-larga/ctxtual.git
cd ctxtual
uv sync --dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

---

## License

MIT
