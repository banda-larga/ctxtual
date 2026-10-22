# ctxtual for MCP

ctxtual sits between an AI client and an MCP server. Results up to a
threshold (20,000 characters, about 5,000 tokens, by default) pass through
unchanged. Larger ones are stored, and the model receives a **stored-result
notice** instead:

- `result_id`, the tool, and the size (`items` for a list, `chars` for a text)
- for lists: `fields` (name → type), a 3-item `preview`, and a query example
  built from the result's real fields
- for markdown texts: an `outline` of the headings with the page each starts
  on (`#` and underlined `===`/`---` headings); for source code, of its
  definitions (`def`, `class`, `function`, …)
- `how_to_explore`: which tool to call next

and four read-only tools that work on stored results:

| Tool | Use |
|---|---|
| `result_search(result_id, query, max_results, fields)` | A list: the best-matching items, each with its `index`, plus how many items match any / all query words. A text: the passages with the exact phrase (nearby occurrences merged), or else the passages with the most and rarest query words, with their page. |
| `result_read(result_id, page, page_size, index, field, fields)` | A text: one page (10,000 chars). A list: a page of items, or one item (`index`); one long value of an item in full, page by page (`index` + `field`, dot paths allowed). |
| `result_query(result_id, steps, save_as)` | Filter, sort, select, group and count a list in one call. Rows that are items of the list come with their `index`. At most 50 rows (fewer if they are large) come back; `save_as` keeps the full output as a new stored result. |
| `result_list()` | The stored results of this conversation, e.g. after it was summarized. |

Long string values in listings are cut, with the exact `result_read` call
that returns them in full. `fields=[...]` keeps only some fields. Responses
are capped at 25,000 characters; above that, the tool asks for fewer fields
or a smaller page instead of answering. Wrong arguments (a missing id, a
string where a number belongs) get the same kind of fixable error as any
other mistake, never a raw validation message.

What gets stored is the part of the result worth exploring:

- Objects that wrap the data, like `{"data": {"items": [...]}, "meta": {...}}`,
  are unpacked to the list; objects made of one long text, like
  `{"url": …, "markdown": "…"}`, to the text (with its outline). The other
  fields are shown in the notice under `other_fields`. When they are too
  long for the notice, they are stored as a second result, whose `result_id`
  the notice gives.
- Objects keyed by name or id (`{"pkg-a": {...}, "pkg-b": {...}}`, or a map
  of counts) become a list of entries with the key in `key`, so they can be
  searched, sorted and counted.
- JSON Lines (one object per line, as many logs are), Python-repr output
  (`str(rows)`), and results sent as one JSON block per item become lists.
- Text resources embedded in the result (as GitHub's `get_file_contents`
  returns files) are stored like text, with their `uri` in the notice.
  Images, audio and binary resources stay in the response next to the notice.
- A short summary sent next to large structured content is kept in the
  notice as `message`.

Search hits on items with long values also show the passage where the words
are, when the cut item doesn't.

The explanation of stored results is appended to the server's instructions
(the wrapped server's own instructions are kept), and each tool description
also says it, since some clients don't show server instructions to the
model. The four definitions take about 4,300 characters; clients cache them
with the rest of the prompt. The upstream
tool is never called again to explore a result, which matters for tools with
side effects. JSON Lines output (one JSON object per line, as many logs are)
is stored as a list, so it can be filtered and counted.

- [Install into a client](#install-into-a-client)
- [`ctxtual wrap`](#ctxtual-wrap)
- [Remote servers, headers and OAuth](#remote-servers-headers-and-oauth)
- [Choosing the threshold](#choosing-the-threshold)
- [On your own FastMCP server](#on-your-own-fastmcp-server)
- [Serving over HTTP and multi-user deployments](#serving-over-http-and-multi-user-deployments)
- [Security](#security)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## Install into a client

```bash
uvx --from 'ctxtual[mcp]' ctxtual install <client> [server ...] [--dry-run]
uvx --from 'ctxtual[mcp]' ctxtual uninstall <client> [server ...]
uvx --from 'ctxtual[mcp]' ctxtual status
```

| Client | `<client>` | File that is changed |
|---|---|---|
| Claude Desktop | `claude-desktop` | `claude_desktop_config.json` (Settings → Developer → Edit Config) |
| Claude Code | `claude-code` | through `claude mcp add-json` (the config is backed up first); `--scope user\|local\|project` |
| Cursor | `cursor` | `~/.cursor/mcp.json`, or `.cursor/mcp.json` with `--scope project` |
| VS Code | `vscode` | user `mcp.json`, or `.vscode/mcp.json` with `--scope project` |
| Windsurf | `windsurf` | `~/.codeium/windsurf/mcp_config.json` |
| Codex | `codex` | `~/.codex/config.toml` (comments and other settings are kept) |
| Gemini CLI | `gemini` | `~/.gemini/settings.json` |

What `install` does to each server entry:

- **Local servers** (`command` + `args`) become
  `uvx --from ctxtual[mcp] ctxtual wrap -- <original command>`. `env` and
  `cwd` stay where they were; the wrapper passes its whole environment to the
  original server.
- **Remote servers** (`url`, `serverUrl`, `httpUrl`) become
  `ctxtual wrap <url>` with `--header` flags. Header values are moved into
  the entry's `env` (`CTXTUAL_HEADER_1`, …) and referenced as `${…}`, so
  secrets never appear in the process arguments that `ps` shows. Codex
  `bearer_token_env_var`/`env_http_headers` become `${VAR}` headers and the
  variables are added to `env_vars`, so Codex passes them to the wrapper.
  URLs ending in `/sse` or declared as `sse` keep the legacy transport.
  Entries without an `Authorization` header get `--oauth`, since the client
  used to handle the login itself.
- Disabled servers (`disabled: true`, Codex `enabled = false`), servers that
  are already wrapped and entries ctxtual does not understand are skipped.

Before writing, the config file is copied to
`<file>.ctxtual-backup-<timestamp>` (readable only by you; never
overwritten), and the original entries are saved under `~/.ctxtual/installs/`.
`uninstall` restores them exactly; if you edited a wrapped entry in the
meantime (rotated a token, changed an argument), those edits are kept in the
restored entry. Symlinked configs are written through the link. Files that
are not plain JSON (for example with comments) are left alone, and Codex's
TOML is edited in place, keeping comments. Claude Desktop's
*Connectors* (remote servers added in Settings) are not in its config file;
add them with `ctxtual add claude-desktop <name> <url>`.

`install` and `add` accept `--threshold` and `--ttl`, which are written into
the entries.

The entry uses the absolute path of `uvx`, because GUI apps start servers
with a minimal `PATH`. `--launcher python` uses the current interpreter
instead (it must have `ctxtual[mcp]` installed); `--package` changes what
uvx installs (a pinned version, a local checkout).

### Adding new servers

```bash
ctxtual add [--print] <client> <name> <url> [-H 'Name: value' ...]
ctxtual add [--print] [--env KEY=VALUE ...] <client> <name> -- <command> [args ...]
```

Everything after `--` belongs to the server's command, so put ctxtual's own
options (`--print`, `--env`, `--scope`, …) before it.

`--print` prints the snippet for the client's config, and for Cursor and
VS Code also a one-click install link (`cursor://…`, `vscode:mcp/install?…`)
you can put in your own README. Printed snippets use a plain `uvx` so they
work on other machines.

## `ctxtual wrap`

```text
ctxtual wrap [options] <url>
ctxtual wrap [options] -- <command> [args ...]
```

| Option | |
|---|---|
| `--threshold N` | Store results longer than N characters (default 20000) |
| `--tool-threshold GLOB=N` | A different threshold for some tools, e.g. `get_file=40000` (repeatable) |
| `--include GLOB`, `--exclude GLOB` | Only / never store results of these tools (repeatable) |
| `--ttl SECONDS` | Drop a stored result after this long without use (default 3600; `0` = never). The model is told. |
| `--max-results N` | Most stored results kept; the least recently used go first (default 200) |
| `--store PATH` | Keep stored results in a SQLite file across restarts (default: memory) |
| `--tools LIST` | Which tools to add, e.g. `search,read` (default `search,read,query,list`) |
| `--prefix NAME` | Prefix of the added tools (default `result`) |
| `--allow-regex` | Allow `$regex` in `result_query` filters (off: patterns are model-written) |
| `-H 'Name: value'` | Header for a remote server; `${VAR}` is read from the environment |
| `--oauth`, `--oauth-scope S` | Log in with OAuth if the server answers 401 |
| `--sse` | The URL uses the legacy HTTP+SSE transport (default: when the path ends in `/sse`) |
| `--env KEY=VALUE`, `--cwd DIR` | Environment / directory for a local command |
| `--http [--port 8766]` | Serve streamable HTTP on 127.0.0.1 instead of stdio |
| `--name`, `--log-level` | Server name; log level (stderr) |

Stored results live in the wrapper process. A client normally starts one
wrapper per server and keeps it running for the session. Every stored call
gets its own `result_id`, so a result the model is still using never changes
under it. Results are dropped after `--ttl` seconds without use, and beyond
`--max-results` (or 100 million characters in total) the least recently used
go first. These limits apply per user when the server uses authentication,
and the result just stored is always kept. With `--store`, the SQLite file is
created readable only by you. Results read from it are kept in memory (up to
100 million characters, least recently used first), so only the first call
on a result reads it from disk.

If the wrapped server has a tool with the same name as one of ctxtual's
(say `result_list`), ctxtual's tools are renamed to `ctxtual_result_*`
instead of hiding it. If the server can't be reached, listing tools fails
with the error rather than showing an empty server.

## Remote servers, headers and OAuth

```bash
ctxtual wrap https://mcp.example.com/mcp -H 'Authorization: Bearer ${API_TOKEN}'
ctxtual wrap https://mcp.notion.com/mcp --oauth
```

Put secrets in the client's `env` block for the server and reference them as
`${VAR}`, rather than writing them into `args`.

With `--oauth`, the first call that gets a 401 opens the browser for login.
Tokens are stored under `~/.ctxtual/oauth/` (readable only by you) and reused
on the next start, so you log in once. Servers that don't require auth are
unaffected.

## Choosing the threshold

Storing a result is not free: the model needs at least one more call to see
what it needs, and each call is a round trip that re-reads the conversation
(cheaply, when the client caches the prompt, but not instantly). A result of
a few thousand tokens is usually better left inline. The default, 20,000
characters, stores what would crowd the context (long pages, big listings,
logs) and leaves ordinary results alone. Raise it per tool for results the
model should read whole, e.g. `--tool-threshold read_file=60000`; lower it
for clients with small context windows.

## On your own FastMCP server

```python
from fastmcp import FastMCP
from ctxtual.mcp import Offload

mcp = FastMCP("my-server", instructions="…")  # yours are kept
mcp.add_middleware(
    Offload(
        threshold=20_000,  # characters
        thresholds={"get_file": 40_000},  # per tool (glob patterns)
        include=None,
        exclude=(),  # glob patterns of tools to store / never store
        ttl=3600,  # seconds without use before a result is dropped
        max_results=200,  # least recently used go first
        max_total_chars=100_000_000,
        tools=("search", "read", "query", "list"),
        prefix="result",  # tool names: result_search, …
        store=None,  # any ctxtual store, e.g. SQLiteStore("results.db")
        preview_items=3,
        max_output_chars=25_000,  # largest response of the added tools
        instructions=True,  # append the explanation to the instructions
    )
)
```

`Offload` lists its tools next to yours and removes the `outputSchema` of
tools whose results it may replace (a stored-result notice doesn't match the
declared schema); tools you `exclude` keep theirs. Very long error messages
are trimmed to their beginning and end instead of being stored. Images,
audio and binary resources stay in the response; only text is stored. If
one of your tools is called like an added one (`result_search`, …), pass
another `prefix`.

To wrap a server you don't control from Python:

```python
from ctxtual.mcp import create_offload_proxy

proxy = create_offload_proxy(
    "https://mcp.example.com/mcp", headers={"Authorization": "Bearer …"}
)
proxy.run()  # stdio
proxy = create_offload_proxy(["npx", "-y", "@scope/server"], threshold=5_000)
```

## Serving over HTTP and multi-user deployments

`ctxtual wrap --http` serves on `127.0.0.1` only and rejects requests whose
`Host` or `Origin` is not local, which blocks DNS-rebinding attacks from web
pages. It refuses other bind addresses: the wrapper holds your upstream
credentials and has no authentication of its own.

To host a wrapper for several users, use the Python API with a FastMCP auth
provider. Stored results are then isolated per authenticated principal (token
subject or client id): one user can't read, list or overwrite another's.

```python
from fastmcp.server.auth.providers.jwt import JWTVerifier
from ctxtual.mcp import create_offload_proxy

proxy = create_offload_proxy(
    "https://upstream.example.com/mcp",
    headers={"Authorization": "Bearer <service credential>"},
    auth=JWTVerifier(
        jwks_uri="https://auth.example.com/.well-known/jwks.json",
        issuer="https://auth.example.com",
        audience="https://ctxtual.example.com/mcp",
    ),
    shared_session=False,
)
proxy.run(transport="http", host="0.0.0.0", port=8000)
```

The proxy never forwards its callers' HTTP headers (including
`Authorization`) to the upstream server; upstream credentials come only from
what you configure. That is what the MCP spec requires (no token passthrough).

## Security

- Upstream credentials are passed only from your configuration (`--header`,
  `--oauth`, the environment). Callers' headers are not forwarded.
- OAuth tokens are stored with user-only permissions.
- The local HTTP mode checks `Host` and `Origin`; non-loopback binding is
  refused.
- Exploration tools are read-only and never call the upstream server.
- `$regex` is disabled unless you pass `--allow-regex` (Python's regex engine
  has no timeout; enabled patterns are checked for known catastrophic shapes).
- Stored results stay in the wrapper's memory (or the `--store` file you
  choose) and expire after `--ttl`.

## Limitations

- **Tasks that need every row cost more**, not less: reading a large result
  through pages takes more calls and tokens than receiving it whole. ctxtual
  pays off when the model needs a part of the result — a search, a filter, a
  count — which is the common case.
- Each wrapped server adds four tools (about 4,300 characters of
  definitions) to the client's tool list; `--tools` can add fewer.
- Images, audio and binary resources are never stored: they stay in the
  response next to the notice.
- Over the legacy HTTP+SSE transport (`--sse`), the MCP SDK rejects single
  results larger than 1 MiB. Streamable HTTP accepts up to 64 MiB.
- Web-only clients (claude.ai in the browser, ChatGPT) can't start local
  commands; they need a hosted wrapper (see above).
- OAuth tokens are cached per server URL: two entries for the same URL share
  one login. Use different `CTXTUAL_HOME` values in their `env` to keep two
  accounts apart.
- `install` for Claude Code goes through the `claude` CLI, which receives
  the entry (including `env` values) as a command-line argument for the
  moment it runs.

## Troubleshooting

- **The client says the server failed to start:** run the entry's command in
  a terminal (copy `command` and `args`). Add `--log-level DEBUG` to the
  `args` to see what the wrapper does; logs go to stderr, which clients keep
  in their MCP logs.
- **`uvx was not found`:** install [uv](https://docs.astral.sh/uv/), or use
  `--launcher python` from an environment with `ctxtual[mcp]`.
- **Changes don't show up:** restart the client; most read their MCP config
  only at startup.
- **Undo everything:** `ctxtual uninstall <client>`, or restore the
  `.ctxtual-backup-*` file next to the config.
