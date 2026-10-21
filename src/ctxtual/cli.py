"""
``ctxtual`` command line.

    ctxtual wrap https://mcp.example.com/mcp          # wrap a remote server (stdio front)
    ctxtual wrap -- npx -y @scope/some-mcp-server     # wrap a local command
    ctxtual install cursor                             # wrap every server Cursor has
    ctxtual uninstall cursor                           # restore them
    ctxtual status                                     # what is configured where
    ctxtual add claude-desktop github https://…/mcp    # add a new, wrapped server
"""

import argparse
import ipaddress
import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

from ctxtual import __version__
from ctxtual.installers import (
    CLIENTS,
    ConfigFile,
    InstallError,
    WrapOptions,
    claude_code_command,
    cursor_link,
    entry_kind,
    get_client,
    install,
    is_wrapped,
    launcher,
    redact,
    referenced_env_vars,
    uninstall,
    vscode_link,
    wrap_entry,
)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after "--" is the upstream command, never ctxtual options.
    upstream: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, upstream = argv[:i], argv[i + 1:]
    parser = _parser()
    args = parser.parse_args(argv)
    args.upstream = upstream
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except InstallError as exc:
        print(f"ctxtual: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ctxtual",
        description="Keep large MCP tool results out of your model's context.",
    )
    p.add_argument("--version", action="version", version=f"ctxtual {__version__}")
    sub = p.add_subparsers(metavar="command")

    w = sub.add_parser(
        "wrap",
        help="run an MCP server through ctxtual",
        description="Proxy an MCP server and offload its large results. "
        "Give a URL, or a command after `--`.",
        epilog="examples:\n"
        "  ctxtual wrap https://mcp.example.com/mcp --oauth\n"
        "  ctxtual wrap https://api.example.com/mcp -H 'Authorization: Bearer ${API_TOKEN}'\n"
        "  ctxtual wrap -- npx -y @modelcontextprotocol/server-filesystem ~/docs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    w.add_argument("url", nargs="?", help="streamable HTTP endpoint of the server")
    _wrap_options(w)
    w.add_argument("-H", "--header", action="append", default=[], metavar="'NAME: VALUE'",
                   help="HTTP header for the upstream server; ${VAR} is read from the environment")
    w.add_argument("--sse", action="store_const", const=True, default=None,
                   help="the URL uses the legacy HTTP+SSE transport "
                   "(default: when the path ends in /sse)")
    w.add_argument("--oauth", action="store_true",
                   help="log in to the upstream server with OAuth if it asks (browser flow)")
    w.add_argument("--oauth-scope", action="append", default=[], metavar="SCOPE")
    w.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra environment variable for a command upstream")
    w.add_argument("--cwd", help="working directory for a command upstream")
    w.add_argument("--http", action="store_true",
                   help="serve streamable HTTP instead of stdio")
    w.add_argument("--host", default="127.0.0.1", help="HTTP bind address (loopback only)")
    w.add_argument("--port", type=int, default=8766, help="HTTP port (default 8766)")
    w.add_argument("--name", help="server name shown to the client")
    w.add_argument("--log-level", default="WARNING", help="log level (stderr)")
    w.set_defaults(func=_cmd_wrap)

    for name, helptext, fn in (
        ("install", "wrap the MCP servers an AI client already has", _cmd_install),
        ("uninstall", "restore servers wrapped by `install`", _cmd_uninstall),
    ):
        c = sub.add_parser(name, help=helptext, description=helptext)
        c.add_argument("client", choices=list(CLIENTS))
        c.add_argument("servers", nargs="*", help="only these servers (default: all)")
        _client_options(c)
        c.add_argument("--dry-run", action="store_true", help="show the changes only")
        if name == "install":
            _wrap_options(c, for_install=True)
        c.set_defaults(func=fn)

    s = sub.add_parser("status", help="list configured servers and whether they are wrapped")
    s.add_argument("clients", nargs="*", metavar="client",
                   help=f"one of {', '.join(CLIENTS)} (default: all found)")
    s.set_defaults(func=_cmd_status)

    a = sub.add_parser(
        "add",
        help="add a new MCP server to a client, wrapped",
        description="Add a server to a client config, already wrapped. "
        "Give a URL, or a command after `--`.",
    )
    a.add_argument("client", choices=list(CLIENTS))
    a.add_argument("name", help="name for the server")
    a.add_argument("url", nargs="?", help="streamable HTTP endpoint (or a command after --)")
    a.add_argument("-H", "--header", action="append", default=[], metavar="'NAME: VALUE'")
    a.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    a.add_argument("--print", dest="print_only", action="store_true",
                   help="print the config snippet / install link instead of writing")
    _client_options(a)
    _wrap_options(a, for_install=True)
    a.set_defaults(func=_cmd_add)
    return p


def _wrap_options(p: argparse.ArgumentParser, *, for_install: bool = False) -> None:
    g = p.add_argument_group("offloading")
    g.add_argument("--threshold", type=int, default=None if for_install else 8_000,
                   help="offload results longer than this many characters (default 8000)")
    g.add_argument("--ttl", type=float, default=None if for_install else 3600.0,
                   help="seconds to keep stored results; 0 = until exit (default 3600)")
    if for_install:
        return
    g.add_argument("--store", metavar="PATH",
                   help="SQLite file to keep stored results across restarts (default: memory)")
    g.add_argument("--include", action="append", default=None, metavar="GLOB",
                   help="only offload these tools (repeatable)")
    g.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="never offload these tools (repeatable)")
    g.add_argument("--prefix", default="result", help="prefix of the injected tools")
    g.add_argument("--allow-regex", action="store_true",
                   help="allow $regex in result_pipe filters (runs model-written patterns)")


def _client_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scope", default="user",
                   help="config scope: user (default), project, or local (claude-code)")
    p.add_argument("--config", help="path of the client's config file (overrides detection)")
    p.add_argument("--launcher", choices=["uvx", "python"], default="uvx",
                   help="how the client starts ctxtual: uvx (default) or this Python")
    p.add_argument("--package", default="ctxtual[mcp]",
                   help="package spec uvx installs (e.g. a pinned version or a path)")


# ── wrap ──────────────────────────────────────────────────────────────────


def _cmd_wrap(args: argparse.Namespace) -> int:
    from pathlib import Path

    target: str | list[str]
    if args.upstream and args.url:
        raise InstallError("give either a URL or a command after --, not both")
    if args.upstream:
        target = args.upstream
    elif args.url:
        target = args.url
    else:
        raise InstallError("nothing to wrap: give a URL, or a command after --")

    level = args.log_level.upper()
    # FastMCP configures its own logger from this variable when imported.
    os.environ.setdefault("FASTMCP_LOG_LEVEL", level)
    logging.basicConfig(level=level, stream=sys.stderr,
                        format="ctxtual %(levelname)s %(name)s: %(message)s")
    try:
        from ctxtual.mcp import create_offload_proxy, expand_env
    except ImportError as exc:
        raise InstallError(str(exc)) from None

    headers = {}
    for name, value in _parse_headers(args.header).items():
        try:
            headers[name] = expand_env(value)
        except ValueError as exc:
            raise InstallError(f"--header {name}: {exc}") from None
    env = dict(_kv(e, "--env") for e in args.env)

    store = None
    if args.store:
        from ctxtual import SQLiteStore

        db = Path(args.store).expanduser()
        if not db.exists():
            # Stored results can be private data; SQLite gives its -wal/-shm
            # files the database file's permissions.
            db.parent.mkdir(parents=True, exist_ok=True)
            os.close(os.open(db, os.O_CREAT | os.O_WRONLY, 0o600))
        store = SQLiteStore(db)

    if args.sse and not isinstance(target, str):
        raise InstallError("--sse applies to URLs only")
    proxy = create_offload_proxy(
        target,
        sse=args.sse,
        headers=headers,
        oauth=args.oauth,
        oauth_scopes=args.oauth_scope or None,
        env=env,
        cwd=args.cwd,
        name=args.name,
        threshold=args.threshold,
        ttl=args.ttl or None,
        store=store,
        include=args.include,
        exclude=args.exclude,
        prefix=args.prefix,
        allow_regex=args.allow_regex,
    )
    if not args.http:
        proxy.run(show_banner=False)
        return 0
    if not _is_loopback(args.host):
        raise InstallError(
            f"refusing to listen on {args.host}: the wrapper carries your upstream "
            "credentials and has no authentication. Bind to 127.0.0.1, or use the "
            "Python API (ctxtual.mcp.create_offload_proxy(..., auth=...)) for a "
            "hosted deployment."
        )
    names = {"127.0.0.1", "localhost", "[::1]",
             f"[{args.host.strip('[]')}]" if ":" in args.host else args.host}
    hosts = [f"{h}:{args.port}" for h in sorted(names)]
    print(f"ctxtual: serving on http://{args.host}:{args.port}/mcp", file=sys.stderr)
    proxy.run(
        transport="http",
        host=args.host,
        port=args.port,
        show_banner=False,
        # Reject other Hosts/Origins: blocks DNS rebinding from web pages.
        host_origin_protection=True,
        allowed_hosts=hosts,
        allowed_origins=[f"http://{h}" for h in hosts],
    )
    return 0


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _parse_headers(raw_headers: Sequence[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw in raw_headers:
        name, sep, value = raw.partition(":")
        name = name.strip()
        if not sep or not name or any(c.isspace() or c in "=\"" for c in name):
            raise InstallError(f"bad --header {raw!r}; use 'Name: value'")
        headers[name] = value.strip()
    return headers


def _kv(raw: str, flag: str) -> tuple[str, str]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise InstallError(f"bad {flag} {raw!r}; use KEY=VALUE")
    return key, value


# ── install / uninstall / status ──────────────────────────────────────────


def _config(args: argparse.Namespace) -> ConfigFile:
    from pathlib import Path

    client = get_client(args.client)
    return ConfigFile(client, args.scope, Path(args.config).expanduser() if args.config else None)


def _cmd_install(args: argparse.Namespace) -> int:
    config = _config(args)
    if not config.path.exists() and config.client.fmt != "claude-code":
        raise InstallError(f"{config.client.title} config not found at {config.path}")
    launch = launcher(args.launcher, args.package)
    options = WrapOptions(threshold=args.threshold, ttl=args.ttl)
    changes, backup = install(config, launch=launch, servers=args.servers or None,
                              options=options, dry_run=args.dry_run)
    _report(changes, config, backup, dry_run=args.dry_run, verb="wrapped")
    if config.client.notes:
        print(f"note: {config.client.notes}")
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    config = _config(args)
    changes, backup = uninstall(config, servers=args.servers or None, dry_run=args.dry_run)
    _report(changes, config, backup, dry_run=args.dry_run, verb="restored")
    return 0


def _report(changes: list[Any], config: ConfigFile, backup: Any, *, dry_run: bool, verb: str) -> None:
    title = config.client.title
    if not changes:
        print(f"{title}: nothing to do ({config.path})")
        return
    for c in changes:
        if c.action == "skip":
            print(f"  - {c.server}: skipped ({c.reason})")
        else:
            arrow = "would be " if dry_run else ""
            note = f" ({c.reason})" if c.reason else ""
            print(f"  ✓ {c.server}: {arrow}{verb}{note}")
            if dry_run:
                print(f"      {json.dumps(redact(c.after))}")
    done = [c for c in changes if c.action in ("wrap", "restore")]
    if done and not dry_run:
        where = config.path if config.client.fmt != "claude-code" else "via `claude mcp`"
        print(f"{title}: {len(done)} server(s) {verb} in {where}")
        if backup:
            print(f"backup: {backup}")
        print(f"Restart {title} to apply.")


def _cmd_status(args: argparse.Namespace) -> int:
    names = args.clients or list(CLIENTS)
    for name in names:
        get_client(name)
    found = False
    for name in names:
        client = CLIENTS[name]
        for scope in client.scopes:
            config = ConfigFile(client, scope)
            if not config.path.exists():
                continue
            try:
                servers = config.servers()
            except InstallError as exc:
                print(f"{client.title} ({scope}): {exc}")
                continue
            if not servers:
                continue
            found = True
            print(f"{client.title} ({scope}) — {config.path}")
            for sname, entry in servers.items():
                state = "wrapped" if is_wrapped(entry) else entry_kind(entry)
                print(f"  {sname:<28} {state}")
    if not found:
        print("No MCP servers found in the supported clients.")
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    if args.upstream and args.url:
        raise InstallError("give either a URL or a command after --, not both")
    if args.upstream:
        entry: dict[str, Any] = {"command": args.upstream[0], "args": args.upstream[1:]}
        if args.env:
            entry["env"] = dict(_kv(e, "--env") for e in args.env)
    elif args.url:
        entry = {"url": args.url}
        if args.header:
            entry["headers"] = _parse_headers(args.header)
    else:
        raise InstallError("nothing to add: give a URL, or a command after --")
    launch = launcher(args.launcher, args.package, portable=args.print_only)
    wrapped, _ = wrap_entry(entry, launch, WrapOptions(threshold=args.threshold, ttl=args.ttl),
                            client=args.client)
    needed = [v for v in referenced_env_vars(wrapped) if v not in (wrapped.get("env") or {})]
    if args.print_only:
        _print_snippet(args.client, args.name, wrapped, args.scope)
        if needed:
            print(f"\nnote: the wrapper reads {', '.join(needed)} from its environment; "
                  "set it in the server's env (or export it before starting the client).")
        return 0
    config = _config(args)
    if args.name in config.servers():
        raise InstallError(f"{config.client.title} already has a server named {args.name!r}")
    backup = config.write({args.name: wrapped})
    where = config.path if config.client.fmt != "claude-code" else "via `claude mcp`"
    print(f"✓ added {args.name} to {config.client.title} ({where})")
    if needed:
        print(f"note: set {', '.join(needed)} in this server's env block; the wrapper reads it from there.")
    if backup:
        print(f"backup: {backup}")
    print(f"Restart {config.client.title} to apply.")
    return 0


def _print_snippet(client: str, name: str, entry: dict[str, Any], scope: str) -> None:
    if client == "cursor":
        print(f"Install link:\n{cursor_link(name, entry)}\n\nOr add to {CLIENTS[client].path('user')}:")
    elif client == "vscode":
        print(f"Install link:\n{vscode_link(name, entry)}\n\nOr add to {CLIENTS[client].path('user')}:")
    elif client == "claude-code":
        print(claude_code_command(name, entry, scope))
        return
    elif client == "codex":
        # JSON strings and arrays of strings are valid TOML.
        key = name if name.replace("_", "").replace("-", "").isalnum() else json.dumps(name)
        lines = [f"[mcp_servers.{key}]",
                 f"command = {json.dumps(entry['command'])}",
                 f"args = {json.dumps(entry['args'])}"]
        env_vars = [v for v in referenced_env_vars(entry) if v not in (entry.get("env") or {})]
        if env_vars:
            # Listed variables are passed by Codex to the server.
            lines.append(f"env_vars = {json.dumps(env_vars)}")
        if entry.get("env"):
            lines.append(f"\n[mcp_servers.{key}.env]")
            lines += [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in entry["env"].items()]
        print(f"Add to {CLIENTS[client].path('user')}:\n")
        print("\n".join(lines))
        return
    key = CLIENTS[client].servers_key
    if client not in ("cursor", "vscode"):
        print(f"Add to {CLIENTS[client].path('user')}:")
    print(json.dumps({key: {name: entry}}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
