# Changelog

## Unreleased

### Fixed

- When a list's other fields (`{"items": [...], "facets": {...}}`) were too
  long for the notice, the cut part could not be reached; it is now stored as
  a second result, whose `result_id` the notice gives.
- With a SQLite store, every exploration call read the whole result from
  disk (0.2 s for 100,000 rows); results read are now kept in memory, by id
  and creation time, up to 100 million characters.
- Outlines include underlined headings (`Title` over `=====` or `-----`).
- CI and release workflows use actions on the Node 24 runtime.

## 0.5.0

Result shapes real servers produce, and defaults tuned for clients that cache
the prompt: there, a round trip costs more than the tokens it saves.

### Changed

- Default threshold 20,000 characters (was 8,000): a result of a few
  thousand tokens is cheaper inline than through an extra call. Text pages
  are 10,000 characters (was 4,000), so a long document takes fewer reads;
  one item's values are cut at 4,000 characters (was 2,000); outlines show up
  to 50 headings (was 30).
- Each tool description says what a stored result is, since some clients
  don't show server instructions to the model (definitions: about 4,300
  characters, cached by clients with the rest of the prompt).

### Fixed

- Results with an image or audio next to a large text were passed through
  whole; the text is now stored and the image kept in the response.
- Embedded text resources (GitHub's `get_file_contents` returns files this
  way) were passed through whole; they are stored, with their `uri` in the
  notice.
- An object made of one long text (`{"url": …, "markdown": "…"}`) was stored
  as escaped JSON, without an outline and with broken phrase search; the text
  itself is stored now, the other fields shown in the notice.
- Objects keyed by name or id (`{"pkg-a": {...}, …}`, maps of counts), output
  sent as one JSON block per item, and Python-repr output (`str(rows)`) were
  stored as text; they are lists now (searchable, sortable, countable).
- A short summary sent next to large structured content was dropped; it is
  in the notice as `message`.
- Search hits on items with long values showed the start of the value, not
  where the words are; hits now include that passage (`match`).
- Comment blocks in YAML, shell or Python files were read as markdown
  headings, giving a misleading outline. Source code now gets an outline of
  its definitions (`def`, `class`, `function`, …) with the page of each.
- Long lists inside items are cut like long strings (first 20 elements, with
  the call that reads them all); reading one item shortens its long values
  until it fits the response instead of failing.
- Notices for lists of plain values no longer suggest `result_query`, and
  query examples no longer sort by id-like fields (`number`, `*_id`).

## 0.4.0

The MCP interface, redesigned from the model's side: fewer tools, one
vocabulary, and the missing pieces an agent gets stuck on.

### Changed (breaking for the MCP tools)

- Four tools instead of seven: `result_search` (items or passages),
  `result_read` (pages, one item, one long field page by page),
  `result_query` (the former `result_pipe`) and `result_list`. The model no
  longer has to pick a tool by the result's shape.
- One name for the handle: the notice and every tool use `result_id`
  (was `stored_result` / `workspace_id`). Notice keys: `items`/`chars`
  (was `item_count`).
- `ttl` is now an idle timeout: a stored result is dropped after `ttl`
  seconds *without use*. Eviction also caps the total size
  (`max_total_chars`) and count (`max_results`, default 200).

### Added

- Search hits carry the item's `index`; `fields=[...]` keeps only some fields
  in search, read and listings.
- Long values in listings are cut with the exact call to read them in full;
  `result_read(result_id, index=N, field="body", page=…)` pages through them
  (dot paths allowed).
- Notices include a query example built from the result's real fields, and,
  for markdown texts, an `outline` of headings with the page each starts on.
- Text search ranks passages by how many query words they contain when the
  exact phrase doesn't occur.
- JSON Lines output is stored as a list (filterable, countable).
- `Offload(thresholds={glob: chars})` / `--tool-threshold`, `tools=` /
  `--tools`, `max_results` / `--max-results`, `max_total_chars`,
  `max_output_chars`, `instructions=`.

### Fixed

- Wrapping a server replaced its instructions with ctxtual's; they are now
  kept, with a short explanation of stored results appended. Clients see the
  wrapped server's name instead of "ctxtual".
- Tool definitions are about 55% smaller (3,600 characters instead of
  8,000): the explanation lives once in the instructions, and parameter
  descriptions are no longer repeated in the tool description.
- Error messages no longer use library jargon ("producer tool").
- Wrong arguments to the result tools return a fixable error instead of a
  raw validation message; `steps` may be a single operation, `fields` a
  single name, numbers may arrive as strings.
- Query rows that are items of the stored list carry their `index`; large
  rows are shown fewer at a time instead of failing.
- Eviction limits apply per authenticated user; the result just stored is
  never evicted.
- `--name` is respected again for wrapped servers.

## 0.3.1

### Fixed

- `result_pipe(..., save_as=...)` over MCP advertised four tools the wrapper
  doesn't expose (`result_count`, `result_get_slice`, `result_field_values`,
  `result_aggregate`). A saved pipeline result now comes back as the same
  compact notice as other stored results, naming only tools that exist.

## 0.3.0

ctxtual now works with any MCP server, without code. Install with the new
`mcp` extra: `pip install "ctxtual[mcp]"` (the core stays dependency-free).

### Added

- **`ctxtual wrap`** — runs any MCP server (a local command or a remote
  streamable-HTTP/SSE URL) through ctxtual. Results above `--threshold`
  characters (default 8,000) are stored and replaced by a notice with a
  preview, the inferred fields and the size; seven read-only `result_*`
  tools search, filter, aggregate and page through the stored data. Small
  results, results with images, and errors pass through (very long errors are
  trimmed to their start and end).
- **`ctxtual install <client>` / `uninstall` / `status`** — wraps the servers
  already configured in Claude Desktop, Claude Code, Cursor, VS Code,
  Windsurf, Codex and Gemini CLI. Config files are backed up, original
  entries are saved for an exact `uninstall`, files with comments are left
  alone, and `--dry-run` shows the changes.
- **`ctxtual add <client> <name> …`** — adds a new server already wrapped;
  `--print` prints the snippet and, for Cursor and VS Code, a one-click
  install link.
- **`ctxtual.mcp.Offload`** — FastMCP middleware for your own server
  (`mcp.add_middleware(Offload())`), and `create_offload_proxy()` to wrap a
  server from Python.
- Remote servers: headers with `${VAR}` expansion, OAuth login with tokens
  cached under `~/.ctxtual/oauth` (log in once), legacy SSE.
- Hosted use: with a FastMCP auth provider, stored results are isolated per
  authenticated principal.

### Security

- The proxy never forwards its callers' HTTP headers (e.g. `Authorization`)
  upstream; FastMCP's proxy does by default.
- `ctxtual wrap --http` validates `Host`/`Origin` (FastMCP's HTTP server
  doesn't by default) and refuses non-loopback addresses.
- Local servers receive the wrapper's full environment (the MCP SDK passes
  only a short allowlist), so secrets in a client's `env` block keep working.
- `install`/`add` never write header values into process arguments: they go
  into the entry's `env`. Backups, saved originals, OAuth tokens and
  `--store` files are readable only by the user.

### Fixed (works around upstream limits)

- Results over 1 MiB from streamable-HTTP servers failed in the MCP SDK
  client ("Server-sent event exceeded the 1048576 byte limit"), which FastMCP
  doesn't let callers raise; ctxtual's transport accepts up to 64 MiB.

## 0.2.0

A correctness and safety release. Several tool outputs changed shape, so
check the **Breaking changes** section if you parse tool results in code.

### Fixed

- **Search matched internal columns (SQLite).** The FTS index made workspace
  ids and item indices searchable, so e.g. searching `"papers"` in a
  `papers_*` workspace matched every item. The index now only covers item
  values; existing databases are rebuilt automatically on open.
- **Search queries containing `"` raised an SQL error (SQLite).** Quotes are
  now escaped; `%` and `_` are no longer treated as wildcards.
- **`text_search(fields=[...])` never used FTS (SQLite).** Field-restricted
  search now uses the FTS index with bm25 ranking. Multi-word queries match
  any term (ranked by how many match) on both backends, instead of requiring
  every term on SQLite only.
- **Mixing toolsets on one workspace type broke shape checks.** Because the
  built-in factories share one `ToolSet` per name, the last factory's
  `data_shape` was applied to every tool, so e.g.
  `toolsets=[paginator(ctx), text_content(ctx)]` advertised `paginate` and
  then rejected it. Shapes are now tracked per tool.
- **Consumers could not be called by the model.** `@ctx.consumer` functions
  are now registered: exported by `get_tools()`, dispatchable, and listed in
  `system_prompt()`.
- **Consumers ignored `workspace_type`.** Calls with a workspace of another
  type are now rejected with a type-mismatch error.
- **`dispatch_tool_call` returned an un-awaited coroutine for async
  producers.** Async tools now run (via `asyncio.run` when no loop is
  running); `adispatch_tool_call` awaits them inside a loop. `ToolSet` tools
  can be `async def`.
- **`dispatch_tool_call` raised on producer errors and bad arguments.**
  Producer/consumer exceptions, `PayloadTooLargeError`, invalid or
  non-JSON arguments now return error dicts with recovery hints
  (`Ctx(safe=False)` restores raising).
- **"LRU" eviction evicted the oldest workspace, and accesses were never
  recorded.** `last_accessed_at` updates were applied to a copy; they are now
  persisted (`BaseStore.touch`), and `MemoryStore` evicts the least recently
  used workspace.
- **Re-running a keyed producer on SQLite kept the old `created_at`**, so a
  refreshed workspace could be expired immediately. Re-storing now replaces
  all metadata, as `MemoryStore` already did.
- **`pipe(save_as=...)` could overwrite any workspace**, including other
  types' data. Saved results now live in their own namespace
  (`save_as="top"` creates `saved_top`), may only replace earlier saved
  results, and names are validated. Producer `key=` templates can no longer
  overwrite a workspace created by another producer or of another type
  (`WorkspaceConflictError`); `emit(workspace_id=...)` refuses to replace a
  workspace of another type.
- **`SQLiteStore(":memory:")` gave each thread its own empty database.** It
  is now one private in-memory database shared by the store's threads.
  `SQLiteStore.close()` closes every thread's connection, and connections of
  exited threads are closed instead of piling up.
- **A failed SQLite write (e.g. `update_item` with a bad index) left a
  transaction open**, locking the database for other threads. Every write
  now runs in a transaction that rolls back on error, and `transaction()`
  starts with `BEGIN IMMEDIATE`, so check-then-write sequences and schema
  migrations are atomic across processes too.
- **A non-string `workspace_id` (list, number, …) crashed dispatch**; it now
  returns an `InvalidArgumentError` dict.
- **OpenAI adapter:** malformed JSON arguments crashed the whole batch, and
  `max_content_length` produced invalid pseudo-JSON. Both fixed.
- `emit(ttl=None)` now means "no expiry" (it used to fall back to the
  default TTL); `contains`/`startswith` filters on SQLite are case-sensitive
  like `MemoryStore`'s; `$contains` in pipelines works on list fields.
- Key templates referencing unknown parameters now fail when the decorator
  runs instead of silently falling back to a random id.

### Added

- **Sessions** (`ctx.session(id)`, `dispatch_tool_call(..., session_id=)`,
  `session_id=` on all integrations): isolate conversations that share a
  store. Other sessions' workspaces can't be read, listed or overwritten, and
  deterministic ids are namespaced per session.
- **Output caps on every built-in tool** (configurable on each factory), with
  honest `has_more` / `truncated` / `total` fields, plus
  `Ctx(max_result_chars=60_000)`: any dispatched result larger than that —
  from built-in or custom tools — is replaced by an error with a short
  preview.
- **`enforce_type=False`** on the built-in factories: share one set of tools
  across all workspace types to keep the tool list short.
- `@ctx.consumer(expose=False)` for consumers the model shouldn't see.
- `InvalidArgumentError`.
- `ctx.adispatch_tool_call`, `ahandle_tool_calls`, `ahandle_tool_use`;
  LangChain tools support `ainvoke`.
- `ctx.store_workspace(...)` for storing data from your own code,
  `ctx.get_consumer_schemas()`, `ctx.toolsets_for_type(...)`,
  `@consumer(name=, meta=, ttl=)`, `@tool(data_shape=)`.

### Breaking changes

- `*_search` returns `returned` and `has_more` instead of `total_matches`
  (which was only ever the length of the capped list).
- `*_filter_by` returns `total_matches` (true total) and `returned` instead
  of `count`, and returns at most `limit` (default 50) results.
- `*_get_slice` returns `{"items", "start", "end", "total"}` instead of a
  bare list.
- `*_sort_by` defaults to `limit=20` (was 100).
- `*_pipe` and `*_aggregate` return at most 50 items/groups inline (see
  `truncated`/`total`); use `save_as` for full results.
- `*_paginate` caps `size` at 100; `*_read_page` caps `chars_per_page` at
  20,000.
- **`$regex` in pipelines is disabled by default.** Model-written patterns
  run with no timeout, so a crafted pattern could pin a CPU. Opt in with
  `pipeline(ctx, allow_regex=True)`; enabled patterns are checked on Python's
  parse tree (nested repeats, repeated alternation, backreferences and more
  than 4 unbounded repeats are rejected) and matched against at most 1,000
  characters per value.
- `get_tools()` now includes `@ctx.consumer` functions (they were never
  callable before). Use `@ctx.consumer(expose=False)` to keep one internal.
- `pipe(save_as="x")` creates workspace `saved_x` (was `x`).
- `*_get_keys` returns `{"keys", "count", "has_more"}` (was a bare list,
  at most 200 keys now); `*_sort_by` returns
  `{"field", "descending", "items", "returned", "has_more"}` (was a bare
  list).
- `values:field` metrics return at most 100 values; larger sets come back as
  `{"values", "total", "truncated"}`.
- Search queries containing symbol-bearing terms (`C++`, `e-mail`) use
  substring scoring on SQLite, since FTS can't distinguish them.
- Random workspace ids are 12 hex characters (were 10).
- Removed the `redis` and `all` extras, which installed `redis` although no
  Redis backend exists.
