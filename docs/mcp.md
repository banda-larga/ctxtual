# ctxtual for MCP

ctxtual sits between an AI client and an MCP server. Results up to a
threshold (8,000 characters by default) pass through unchanged. Larger ones
are stored, and the model receives a notice with a preview, the inferred
fields and a `workspace_id`, plus seven read-only tools to explore the stored
data:

| Tool | Use |
|---|---|
| `result_search` | Ranked search over a stored list |
| `result_pipe` | Filter, sort, select, group and count in one call |
| `result_paginate`, `result_get_item` | Read a list directly |
| `result_search_in_text`, `result_read_page` | Search and read a stored text |
| `result_list` | Stored results of this conversation (e.g. after it was summarized) |

The upstream tool is never called again to explore a result, which matters
for tools with side effects.

- [Install into a client](#install-into-a-client)
- [`ctxtual wrap`](#ctxtual-wrap)
- [Remote servers, headers and OAuth](#remote-servers-headers-and-oauth)
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
| `--threshold N` | Offload results longer than N characters (default 8000) |
| `--ttl SECONDS` | Keep stored results this long (default 3600; `0` = until exit). The model is told. |
| `--store PATH` | Keep stored results in a SQLite file across restarts (default: memory) |
| `--include GLOB`, `--exclude GLOB` | Only / never offload these tools (repeatable) |
| `--prefix NAME` | Prefix of the injected tools (default `result`) |
| `--allow-regex` | Allow `$regex` in `result_pipe` filters (off: patterns are model-written) |
| `-H 'Name: value'` | Header for a remote server; `${VAR}` is read from the environment |
| `--oauth`, `--oauth-scope S` | Log in with OAuth if the server answers 401 |
| `--sse` | The URL uses the legacy HTTP+SSE transport (default: when the path ends in `/sse`) |
| `--env KEY=VALUE`, `--cwd DIR` | Environment / directory for a local command |
| `--http [--port 8766]` | Serve streamable HTTP on 127.0.0.1 instead of stdio |
| `--name`, `--log-level` | Server name; log level (stderr) |

Stored results live in the wrapper process. A client normally starts one
wrapper per server and keeps it running for the session, so results last as
long as the conversation (or `--ttl`). Every offloaded call gets its own
stored result, so an id the model is still using never changes under it; old
ones expire after `--ttl`, and at most 500 are kept (least recently used go
first). With `--store`, the SQLite file is created readable only by you.

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

## On your own FastMCP server

```python
from fastmcp import FastMCP
from ctxtual.mcp import Offload

mcp = FastMCP("my-server")
mcp.add_middleware(Offload(
    threshold=8_000,      # characters
    ttl=3600,             # seconds, or None
    store=None,           # any ctxtual store, e.g. SQLiteStore("results.db")
    include=None,         # glob patterns of tools to offload
    exclude=(),           # glob patterns never to offload
))
```

`Offload` lists its tools next to yours and removes the `outputSchema` of
tools whose results it may replace (a stored-result notice doesn't match the
declared schema). Very long error messages are trimmed to their beginning and
end instead of being stored. Results with images or other non-text content
pass through.

To wrap a server you don't control from Python:

```python
from ctxtual.mcp import create_offload_proxy

proxy = create_offload_proxy("https://mcp.example.com/mcp", headers={"Authorization": "Bearer …"})
proxy.run()                                  # stdio
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
    auth=JWTVerifier(jwks_uri="https://auth.example.com/.well-known/jwks.json",
                     issuer="https://auth.example.com", audience="https://ctxtual.example.com/mcp"),
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
- Each wrapped server adds seven tools to the client's tool list.
- Results that contain images or audio are passed through, not stored.
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
