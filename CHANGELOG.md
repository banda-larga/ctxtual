# Changelog

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
