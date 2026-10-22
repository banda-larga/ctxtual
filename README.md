# ctxtual

[![PyPI version](https://img.shields.io/pypi/v/ctxtual.svg)](https://pypi.org/project/ctxtual/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/ctxtual.svg)](https://pypi.org/project/ctxtual/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/banda-larga/ctxtual/actions/workflows/ci.yml/badge.svg)](https://github.com/banda-larga/ctxtual/actions)

**Keep large tool results out of the context window.**

When a tool returns 10,000 rows or a 500 KB page, clients truncate it and hope
the answer wasn't in the part they cut. ctxtual stores the result instead and
gives the model a short notice — a preview, the fields, the size — plus tools
to search, filter, aggregate and page through the stored data. Small results
pass through untouched.

[![Benchmark results](benchmarks/results.svg)](benchmarks/RESULTS.md)

Across five models, questions about large tool results were answered
correctly every time through ctxtual, at about 8,000 prompt tokens each
whether the result held 30k or 300k tokens. With the whole 300k-token result
in context, models were right 39% of the time; with it truncated, 17%.
[Full results and method →](benchmarks/RESULTS.md)

It works two ways:

- **With any MCP server, no code:** `ctxtual install <client>` wraps the
  servers your AI client already uses.
- **As a Python library** for your own agent tools (zero dependencies).

## Use it with your MCP servers

Requires [uv](https://docs.astral.sh/uv/). Wrap every server a client
already has (a backup is made; `uninstall` puts the original entries back):

```bash
uvx --from 'ctxtual[mcp]' ctxtual install claude-desktop   # or: cursor, vscode, windsurf,
                                                           #     codex, claude-code, gemini
```

Restart the client. That's it. `ctxtual status` shows what is wrapped,
`--dry-run` shows the changes first, and you can pass server names to wrap
only some of them. (To type plain `ctxtual …` instead of the `uvx` prefix,
install it once with `uv tool install 'ctxtual[mcp]'`.)

Adding a new server, already wrapped:

```bash
uvx --from 'ctxtual[mcp]' ctxtual add cursor github https://api.githubcopilot.com/mcp/ \
    -H 'Authorization: Bearer ${GITHUB_TOKEN}'
uvx --from 'ctxtual[mcp]' ctxtual add claude-desktop fs -- npx -y @modelcontextprotocol/server-filesystem ~/docs
```

`add --print …` prints the config snippet instead of writing it, plus a
one-click install link for Cursor and VS Code (ctxtual's own options go
before `--`; everything after it is the server's command). Or write the entry
yourself — it is a normal stdio server that runs `ctxtual wrap`:

```json
{
  "mcpServers": {
    "deepwiki": {
      "command": "uvx",
      "args": ["--from", "ctxtual[mcp]", "ctxtual", "wrap", "https://mcp.deepwiki.com/mcp"]
    }
  }
}
```

**What the model sees.** DeepWiki's page for the MCP Python SDK is 519,810
characters (roughly 130k tokens). Through ctxtual the model gets a
2,671-character notice, including an outline that says where each section
starts:

```json
{
  "result_id": "read_wiki_contents_4ee142f7bab5",
  "tool": "read_wiki_contents",
  "chars": 519810,
  "outline": ["0: Page: Overview", "0: Page: Installation & Dependencies",
              "1: Page: Key Concepts & Architecture", "…",
              "19: Page: StreamableHTTP Transport", "…"],
  "preview": "# Page: Overview …",
  "how_to_explore": "read_wiki_contents returned 519,810 chars, too long to show in full; the text is stored. Use result_search(result_id, query) to find passages; result_read(result_id, page) to read it (52 pages of 10,000 chars). Calling read_wiki_contents again with narrower arguments also works."
}
```

It then reads page 19, or searches, instead of taking in all 52 pages.
Lists get their field types, a preview, and a query example built from
their real fields, e.g. `[{"filter": {"state": "open"}}, {"sort": {"field":
"comments", "order": "desc"}}, {"limit": 10}]`.

Four tools are added, about 4,300 characters of definitions in total:
`result_search` (items or passages), `result_read` (pages, single items, long
fields), `result_query` (filter, sort, group, count in one call) and
`result_list`. The wrapped server's own instructions are kept; a short
explanation of stored results is appended to them.

**Where it helps most.** Most MCP clients put a tool result into the
model's context as it is, or cut it off at a limit (Codex, for example, has a
per-tool token budget). That is where ctxtual makes the difference, and in
agent loops you build yourself. Claude Code already saves results over 50,000
characters to files; ctxtual adds a lower threshold and query tools there. Web-only clients (claude.ai in the browser, ChatGPT) can't run
local commands, so they need a hosted wrapper: see the docs.

Details — every option, OAuth, remote/hosted use, security:
**[docs/mcp.md](docs/mcp.md)**.

### On your own FastMCP server

```python
from fastmcp import FastMCP
from ctxtual.mcp import Offload

mcp = FastMCP("my-server")
mcp.add_middleware(Offload())  # results > 20,000 chars are stored
```

Options: `threshold`, per-tool `thresholds`, `include`/`exclude`, `ttl`
(idle time before a stored result is dropped), `max_results`,
`max_total_chars`, `tools` (which of the four to add), `prefix`, `store`
and more — see [docs/mcp.md](docs/mcp.md#on-your-own-fastmcp-server).

---

## The Python library

For tools you call from your own agent loop, without MCP:

```
pip install ctxtual
```

Zero dependencies (standard library only), Python 3.11+.

### Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Producers](#producers)
- [Built-in tools](#built-in-tools)
- [Pipelines](#pipelines)
- [Consumers: derived workspaces](#consumers-derived-workspaces)
- [Dispatching tool calls](#dispatching-tool-calls)
- [Sessions: isolating conversations](#sessions-isolating-conversations)
- [Keeping the tool list short](#keeping-the-tool-list-short)
- [Custom tools](#custom-tools)
- [Storage backends](#storage-backends)
- [Framework integrations](#framework-integrations)
- [Limitations](#limitations)
- [API reference](#api-reference)
- [Development](#development)

## How it works

```python
@ctx.producer(workspace_type="papers", toolsets=[paginator(ctx), text_search(ctx)])
def search_papers(query: str) -> list[dict]:
    return database.search(query)  # 10,000 rows → stored, not returned
```

Calling the producer (directly or as a tool call from the model) returns:

```python
{
  "status": "workspace_ready",
  "workspace_id": "papers_3f8a0c12b9d4",
  "item_count": 10000,
  "data_shape": "list",
  "item_schema": {"type": "object", "properties": {"title": {"type": "string"},
                  "year": {"type": "integer"}, ...}},
  "available_tools": ["papers_paginate(workspace_id='papers_3f8a0c12b9d4')", ...],
  "next_steps": ["papers_paginate: Return a page of items from the workspace...", ...]
}
```

The model then calls `papers_search`, `papers_paginate`, … with that
`workspace_id`. Every exploration tool caps its output and says so when it
does (`has_more`, `truncated`, `total`), and `dispatch_tool_call` rejects any
result larger than `Ctx(max_result_chars=60_000)` — custom tools included —
so no single call can flood the context.

ctxtual is a library, not a framework: it doesn't own your agent loop or your
LLM calls. You export tool schemas with `ctx.get_tools()` and route the
model's tool calls through `ctx.dispatch_tool_call()`.

## Quick start

```python
from ctxtual import Ctx, MemoryStore
from ctxtual.utils import filter_set, paginator, pipeline, text_search

ctx = Ctx(store=MemoryStore())


@ctx.producer(
    workspace_type="papers",
    toolsets=[
        paginator(ctx),
        text_search(ctx, fields=["title", "abstract"]),
        filter_set(ctx),
        pipeline(ctx),
    ],
)
def search_papers(query: str, limit: int = 10_000) -> list[dict]:
    """Search the paper database."""
    return database.search(query, limit)


ref = search_papers("machine learning")
ws = ref["workspace_id"]

ctx.dispatch_tool_call("papers_paginate", {"workspace_id": ws, "page": 0, "size": 10})
ctx.dispatch_tool_call("papers_search", {"workspace_id": ws, "query": "transformer"})
ctx.dispatch_tool_call(
    "papers_filter_by",
    {"workspace_id": ws, "field": "year", "value": 2024, "operator": "gte"},
)

# Or several operations in one call:
ctx.dispatch_tool_call(
    "papers_pipe",
    {
        "workspace_id": ws,
        "steps": [
            {"search": {"query": "transformer"}},
            {"filter": {"year": {"$gte": 2024}}},
            {"sort": {"field": "citations", "order": "desc"}},
            {"limit": 5},
        ],
    },
)
```

In an agent loop you pass `ctx.get_tools()` to the model and send each tool
call to `ctx.dispatch_tool_call(name, arguments)`; see
[Framework integrations](#framework-integrations).
`ctx.system_prompt(preamble=...)` generates a short system-prompt section
describing the registered tools.

## Producers

```python
@ctx.producer(
    workspace_type="papers",  # logical data type
    toolsets=[pager, search],  # tools offered for the stored data
    key="papers_{query}",  # deterministic id (optional)
    transform=lambda rows: rows[:5000],  # pre-store transform (optional)
    meta={"source": "arxiv"},  # metadata shown to the model
    ttl=1800,  # seconds until expiry (optional)
    notify=True,  # True: dict, False: WorkspaceRef
)
def search_papers(query: str) -> list[dict]: ...
```

- **Workspace ids.** By default each call gets a random id. With `key` (a
  template over the function's parameters, or a callable receiving them), the
  same arguments map to the same workspace, which is replaced on each call.
  A key never overwrites a workspace created by a different producer or of a
  different type — that returns a conflict error instead. Template fields are
  checked against the function signature when the decorator runs.
- **TTL** is an absolute lifetime measured from when the data was stored;
  re-running the producer resets it. Call `ctx.sweep_expired()` periodically
  to delete expired workspaces (expired ones are also dropped when accessed).
- **Shapes.** The payload's shape (`list`, `dict` or `scalar`) is recorded,
  and the returned description only lists tools that accept that shape — a
  producer that returns a string gets text tools, not pagination tools.
- **`item_schema`** is inferred from up to 100 items, so the model knows field
  names and types before it reads any data.
- **Size limit.** `Ctx(max_items=...)` rejects larger payloads.
- Producers can be `async def`.

## Built-in tools

Factories in `ctxtual.utils`. Each takes the `Ctx` and an optional name; when
the name is omitted it is taken from the producer's `workspace_type`, so
`paginator(ctx)` on a `"papers"` producer creates `papers_paginate`, etc.

| Factory | Tools | Data shape |
|---|---|---|
| `paginator` | `_paginate(page, size)`, `_count()`, `_get_item(index)`, `_get_slice(start, end)` | list |
| `text_search` | `_search(query, max_results, case_sensitive)`, `_field_values(field, max_values)` | list |
| `filter_set` | `_filter_by(field, value, operator, limit)`, `_sort_by(field, descending, limit)` | list |
| `pipeline` | `_pipe(steps, save_as)`, `_aggregate(group_by, metrics)` | list |
| `kv_reader` | `_get_keys()`, `_get_value(key)` | dict |
| `text_content` | `_read_page(page, chars_per_page)`, `_search_in_text(query, ...)`, `_get_length()` | scalar (string) |

All tools take `workspace_id` as their first argument.

**Output caps** (factory keyword arguments, with defaults):

| Factory | Cap |
|---|---|
| `paginator(max_page_size=100)` | page size and slice length |
| `text_search(max_results_cap=100)` | results; responses include `has_more` |
| `filter_set(max_results_cap=200)` | `limit` (default 50 for `filter_by`, 20 for `sort_by`); `filter_by` reports the true `total_matches`, `sort_by` reports `has_more` |
| `pipeline(max_items=50)` | items/groups returned inline; use `save_as` for the full result. `values:f` metrics list at most 100 values |
| `kv_reader(max_value_chars=20_000, max_keys=200)` | larger values come back as a truncated preview; `get_keys` reports `count`/`has_more` |
| `text_content(max_chars_per_page=20_000)` | page size |

**Search.** Terms are matched individually (any term matches; items matching
more terms rank higher). On `SQLiteStore`, plain-word queries use FTS5 with
Porter stemming, prefix matching and bm25 ranking — also when `fields=`
restricts the search. It falls back to substring scoring (what `MemoryStore`
always uses) when FTS finds nothing, e.g. for a fragment from the middle of a
word, and for terms with symbols like `C++` or `e-mail`, which the tokenizer
can't distinguish from `C` or `e`.

**Text transforms.** To use the list tools on text, split it first:
`chunk_text(text, chunk_size, overlap)`, `split_sections(text, separator)`,
`split_markdown_sections(text)` (all in `ctxtual`).

## Pipelines

`{name}_pipe` runs a list of steps in one tool call; only the final result is
returned.

```python
ctx.dispatch_tool_call(
    "papers_pipe",
    {
        "workspace_id": ws,
        "steps": [
            {"flatten": "tags"},
            {
                "group_by": {
                    "field": "tags",
                    "metrics": {"n": "count", "avg": "mean:citations"},
                }
            },
            {"sort": {"field": "n", "order": "desc"}},
            {"limit": 10},
        ],
    },
)
```

| Step | Example |
|---|---|
| `filter` | `{"filter": {"year": {"$gte": 2023}, "author.name": "Ada"}}` — `$gt $gte $lt $lte $ne $in $nin $contains $startswith $regex $exists`, `$or $and $not`, dot notation |
| `search` | `{"search": {"query": "ML", "fields": ["title"]}}` |
| `sort` | `{"sort": {"field": "score", "order": "desc"}}` or a list of such specs |
| `select` / `exclude` | `{"select": ["title", "year"]}` |
| `limit` / `skip` / `slice` | `{"limit": 10}`, `{"skip": 5}`, `{"slice": [5, 15]}` |
| `sample` | `{"sample": {"n": 10, "seed": 42}}` |
| `unique` | `{"unique": "author"}` |
| `flatten` | `{"flatten": "tags"}` |
| `group_by` | `{"group_by": {"field": "cat", "metrics": {"n": "count", "avg": "mean:score"}}}` — `count`, `sum:f`, `mean:f`, `min:f`, `max:f`, `median:f`, `stddev:f`, `values:f` |
| `count` | `{"count": true}` |

`save_as="name"` stores the full result as a new workspace `saved_name` (of
the source's type) and returns its description. Saved results live in their
own `saved_` namespace and can only replace earlier saved results, never a
producer's workspace.

**`$regex` is disabled by default.** Patterns come from the model and run on
your server with Python's backtracking regex engine, which has no timeout.
`pipeline(ctx, allow_regex=True)` enables it with guards: patterns are
checked on Python's parse tree and rejected if they nest a repeat inside a
repeat (`(a+)+`, `(.*a){12}`), repeat an alternation (`(a|a)+`), use
backreferences, or have more than 4 unbounded repeats; values are matched
only up to 1,000 characters. That removes the known exponential shapes but
is not a proof of safety — enable it only when you trust the prompt.

## Consumers: derived workspaces

A consumer reads one workspace and can store a new one. Consumers are
registered with the `Ctx`: they are exported by `get_tools()` and callable
through `dispatch_tool_call()`.

```python
from ctxtual import ConsumerContext


@ctx.consumer(
    workspace_type="raw",  # input must be a "raw" workspace
    produces="clean",  # output workspace type
    produces_toolsets=[paginator(ctx)],
    meta={"stage": "cleaning"},
)
def clean(workspace_id: str, min_quality: float = 0.5, *, cc: ConsumerContext):
    """Drop low-quality rows and normalize the rest."""
    rows = cc.get_items()
    return cc.emit([normalize(r) for r in rows if r["quality"] >= min_quality])
```

The `ConsumerContext` parameter (named `ctx` or annotated `ConsumerContext`)
is injected and hidden from the model's schema. Calls with a missing,
expired, or wrong-type workspace return an error before your function runs.
Pass `expose=False` for consumers that only your own code should call.

## Dispatching tool calls

```python
result = ctx.dispatch_tool_call(
    "papers_search", {"workspace_id": ws, "query": "attention"}
)
result = await ctx.adispatch_tool_call("papers_search", {...})  # async tools
```

`arguments` may be a dict or a JSON string. Mistakes the model can fix come
back as a dict with `error` and `suggested_action` instead of an exception:

| Situation | Response includes |
|---|---|
| Unknown tool | `available_tools`, `available_producers`, `available_consumers` |
| Wrong/missing arguments (incl. a non-string `workspace_id`) | `expected_parameters`, `required` / `argument` |
| Unknown (or another session's) workspace | `available_workspaces` (visible ones only) |
| Expired workspace | which producer to re-run |
| Wrong workspace type | `expected_type`, `actual_type`, `workspaces_of_correct_type` |
| Wrong data shape | `expected_shape`, `actual_shape` |
| Producer/consumer raised | the exception message (logged with traceback) |
| Payload too large / id conflict | limits, or a hint to pick another name |
| Tool result over `max_result_chars` | `truncated`, `result_chars`, a short `preview` |

Set `Ctx(safe=False)` (producers/consumers) or `ToolSet(safe=False)` to let
exceptions propagate instead — useful in tests.

The sync `dispatch_tool_call` runs async tools with `asyncio.run` when no
event loop is running, and raises `RuntimeError` inside a running loop — use
`adispatch_tool_call` there.

## Sessions: isolating conversations

Workspace ids are the only handle the model has on stored data. If one `Ctx`
and store serve several users or conversations, scope each to a session:

```python
ctx.dispatch_tool_call(name, args, session_id=conversation_id)

# or for a block of code (works across async tasks, uses contextvars):
with ctx.session(conversation_id):
    search_papers("ml")
```

While a session is active:

- new workspaces belong to it;
- workspaces of other sessions behave exactly like missing ones — they can't
  be read, listed in error messages, or overwritten;
- deterministic ids (`key=`, `save_as`, `emit(workspace_id=...)`) get a
  session-derived suffix (`notes_ml@3fa2c1d0e9b7`), so two sessions using
  the same key never collide;
- workspaces created outside any session are shared and readable by all.

Without a session everything is visible, which is what single-user scripts
want. Sessions are isolation, not authentication: deriving `session_id` from
an authenticated user or conversation is up to your application.

## Keeping the tool list short

Each factory adds 2–4 tools per workspace type, and models choose tools less
reliably as the list grows. If you have several data sources, share one
generic set of tools across all of them:

```python
data_tools = [
    paginator(ctx, "data", enforce_type=False),
    pipeline(ctx, "data", enforce_type=False),
]


@ctx.producer(workspace_type="papers", toolsets=data_tools)
def search_papers(query: str) -> list[dict]: ...


@ctx.producer(workspace_type="people", toolsets=data_tools)
def search_people(name: str) -> list[dict]: ...
```

That is 6 exploration tools in total (`data_paginate`, `data_pipe`, …) no
matter how many producers you add. `pipeline` alone covers search, filter,
sort and aggregation.

## Custom tools

```python
analytics = ctx.toolset("transactions")
analytics.data_shape = "list"


@analytics.tool(
    name="transactions_anomalies",
    output_hint="Inspect one with transactions_get_item(workspace_id='{workspace_id}', index=I).",
)
def detect_anomalies(workspace_id: str, std_threshold: float = 2.0) -> dict:
    """Flag transactions whose amount deviates from the mean.

    Args:
        workspace_id: The transactions workspace to analyze.
        std_threshold: Standard deviations from the mean to flag.
    """
    items = analytics.store.get_items(workspace_id)
    ...
```

- The wrapper validates the workspace (existence, session, expiry, type,
  shape), records the access, and converts exceptions to error dicts.
- Schemas are generated from type hints and Google- or Sphinx-style
  docstrings; `schema_extra={"param": {...}}` merges extra JSON Schema.
- `output_hint` wraps results as `{"result": ..., "_hint": "..."}`.
- `data_shape=` on `@tool` overrides the toolset's shape per tool.
- `async def` tools are supported.
- `toolset.bind(workspace_id)` returns callables with the id pre-filled.

## Storage backends

**`MemoryStore(max_workspaces=None)`** — in-process, thread-safe. With
`max_workspaces`, the least recently used workspace is evicted.

**`SQLiteStore(path)`** — persistent, standard-library `sqlite3`, WAL mode,
one connection per thread (connections of exited threads are closed). Every
write runs in a transaction that is rolled back on error. List items are
stored one row each, so paging, `filter_by` and `sort_by` run as SQL. Search
uses an FTS5 index. Databases created by older versions are migrated on
open. `":memory:"` gives the store a private in-memory database shared by
its threads.

**Custom backends** subclass `BaseStore` and implement `init_workspace`,
`drop_workspace`, `get_meta`, `list_workspaces`, `set`, `get` and
`delete_key`. Everything else (paging, search, filter, sort, mutations,
`touch`, `sweep_expired`) has a default implementation you can override for
speed. `init_workspace` must fully replace existing metadata.

**Mutations** — `store.append_items`, `update_item`, `patch_item`,
`delete_items` — keep counts and the search index consistent.
`store.transaction()` groups writes atomically on SQLite.

## Framework integrations

The adapters don't import any SDK; they work with SDK objects or plain dicts.

```python
# OpenAI Chat Completions
from ctxtual.integrations.openai import (
    handle_tool_calls,
    has_tool_calls,
    to_openai_tools,
)

response = client.chat.completions.create(
    model=MODEL, messages=messages, tools=to_openai_tools(ctx)
)
if has_tool_calls(response):
    messages.append(response.choices[0].message)
    messages.extend(handle_tool_calls(ctx, response, session_id=conversation_id))
```

```python
# Anthropic Messages
from ctxtual.integrations.anthropic import (
    handle_tool_use,
    has_tool_use,
    to_anthropic_tools,
)

response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=4096,
    tools=to_anthropic_tools(ctx),
    messages=messages,
)
if has_tool_use(response):
    messages.append({"role": "assistant", "content": response.content})
    messages.append(
        {
            "role": "user",
            "content": handle_tool_use(ctx, response, session_id=conversation_id),
        }
    )
```

```python
# LangChain (requires langchain-core)
from ctxtual.integrations.langchain import to_langchain_tools

tools = to_langchain_tools(
    ctx, session_id=conversation_id
)  # supports invoke and ainvoke
```

Async variants: `ahandle_tool_calls`, `ahandle_tool_use`.

## Limitations

- **Pipelines and aggregation run in Python** over the whole workspace, on
  both backends. Fine for thousands to low hundreds of thousands of rows;
  for more, push work into your producer or a custom tool.
- **`MemoryStore.get_items` returns stored objects, not copies**; don't
  mutate them in custom tools.
- **`SQLiteStore` serialises access with one lock**, so it suits a single
  process. Several processes can share a database file, but there is no
  cross-process locking beyond SQLite's own.
- **`$regex`** is off by default — see [Pipelines](#pipelines).
- **`MemoryStore(max_workspaces=...)` eviction is global**: one busy session
  can evict another session's workspaces. Size the limit for all sessions,
  or use TTLs.
- **Sessions are not authentication** — see [Sessions](#sessions-isolating-conversations).

## API reference

### `Ctx`

| | |
|---|---|
| `Ctx(store=None, *, default_notify=True, default_ttl=None, max_items=None, safe=True, max_result_chars=60_000)` | |
| `@ctx.producer(workspace_type, toolsets, key, transform, meta, notify, ttl)` | store results, return a description |
| `@ctx.consumer(workspace_type, produces, produces_toolsets, name, meta, ttl, expose)` | derived workspaces |
| `ctx.toolset(name, *, enforce_type=True)` | create/get a ToolSet |
| `ctx.dispatch_tool_call(name, arguments, *, session_id=None)` | route a tool call |
| `await ctx.adispatch_tool_call(...)` | async version |
| `ctx.session(session_id)` | context manager scoping a session |
| `ctx.get_tools()` | producer + consumer + tool schemas (OpenAI format) |
| `ctx.get_producer_schemas()`, `get_consumer_schemas()`, `get_all_tool_schemas()` | subsets |
| `ctx.system_prompt(preamble="")` | prompt section describing the tools |
| `ctx.store_workspace(payload, *, workspace_type, ...)` | store data from your own code |
| `ctx.list_workspaces(workspace_type=None)`, `workspace_meta(id)`, `drop_workspace(id)`, `sweep_expired()`, `clear()` | management (not session-filtered) |

### `ToolSet`

| | |
|---|---|
| `ToolSet(name, *, enforce_type=True, safe=True, data_shape=None)` | |
| `@ts.tool(name, validate_workspace, output_hint, schema_extra, data_shape)` | register a tool |
| `ts.bind(workspace_id)` | `BoundToolSet` with the id pre-filled |
| `ts.tools`, `ts.tool_names`, `ts.to_tool_schemas()` | introspection |

### `WorkspaceMeta`

`workspace_id`, `workspace_type`, `created_at`, `last_accessed_at`,
`producer_fn`, `producer_kwargs`, `item_count`, `ttl`, `data_shape`,
`extra`, `session_id`, `is_expired`.

## Examples

`examples/` has runnable scripts (`uv run python examples/01_quickstart.py`)
and notebooks: quickstart, a RAG support agent, multi-step data pipelines,
custom tools, pipelines, SQLite persistence and mutations, error handling,
multi-agent collaboration, OpenAI and Anthropic agent loops (simulated when
no API key is set), a concurrent server, and the `Offload` middleware on a
FastMCP server (`uv run --extra mcp python examples/12_mcp_offload.py`).

## Development

```bash
git clone https://github.com/banda-larga/ctxtual.git
cd ctxtual
uv sync --extra dev --extra mcp
uv run pytest
uv run ruff check .
uv run ruff format .
```

[`benchmarks/`](benchmarks/README.md) measures ctxtual with real models
through OpenRouter: six questions about large tool results, each answered
with the result inline, truncated, and through ctxtual.

Contributions are welcome; please open an issue before large changes.
See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT
