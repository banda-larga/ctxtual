"""
Wrap the MCP servers already configured in an AI client with ``ctxtual wrap``.

Supported clients and the files they read::

    claude-desktop  claude_desktop_config.json        mcpServers
    claude-code     ~/.claude.json / .mcp.json        mcpServers (via the `claude` CLI)
    cursor          ~/.cursor/mcp.json                mcpServers
    windsurf        ~/.codeium/windsurf/mcp_config.json  mcpServers
    vscode          <user dir>/Code/User/mcp.json     servers
    gemini          ~/.gemini/settings.json           mcpServers
    codex           ~/.codex/config.toml              [mcp_servers.*]

Every change is reversible: the original entry of each wrapped server is
saved under ``~/.ctxtual/installs`` (``uninstall`` restores it exactly) and
the config file is backed up before it is written.
"""

import copy
import datetime
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

#: Keys that hold a remote server's URL, per client dialect.
URL_KEYS = ("url", "serverUrl", "httpUrl")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
#: A header value whose ${...} parts are all plain ${VAR} references.
_ENV_ONLY = re.compile(r"(?:[^$]|\$\{[A-Za-z_][A-Za-z0-9_]*\})*")
#: Package spec uvx installs the wrapper from.
DEFAULT_PACKAGE = "ctxtual[mcp]"


class InstallError(Exception):
    """A problem the user has to fix (missing file, unsupported content, …)."""


# ── where each client keeps its config ────────────────────────────────────


def _home() -> Path:
    return Path(os.environ.get("CTXTUAL_TEST_HOME") or Path.home())


def _appdata() -> Path:
    return Path(os.environ.get("APPDATA") or _home() / "AppData" / "Roaming")


def _app_support(name: str) -> Path:
    system = platform.system()
    if system == "Darwin":
        return _home() / "Library" / "Application Support" / name
    if system == "Windows":
        return _appdata() / name
    return Path(os.environ.get("XDG_CONFIG_HOME") or _home() / ".config") / name


@dataclass
class Client:
    name: str
    title: str
    servers_key: str
    path: Callable[[str], Path]
    fmt: str = "json"  # "json" | "toml" | "claude-code"
    scopes: tuple[str, ...] = ("user",)
    notes: str = ""


CLIENTS: dict[str, Client] = {
    c.name: c
    for c in [
        Client(
            "claude-desktop", "Claude Desktop", "mcpServers",
            lambda scope: _app_support("Claude") / "claude_desktop_config.json",
            notes="Remote connectors added in Settings → Connectors are not in "
                  "this file; add them wrapped with `ctxtual add claude-desktop`.",
        ),
        Client(
            "claude-code", "Claude Code", "mcpServers",
            lambda scope: (Path.cwd() / ".mcp.json") if scope == "project"
            else _home() / ".claude.json",
            fmt="claude-code", scopes=("user", "local", "project"),
            notes="Claude Code already saves results over 50,000 characters to "
                  "files; ctxtual adds a lower threshold and query tools.",
        ),
        Client(
            "cursor", "Cursor", "mcpServers",
            lambda scope: (Path.cwd() / ".cursor" / "mcp.json") if scope == "project"
            else _home() / ".cursor" / "mcp.json",
            scopes=("user", "project"),
        ),
        Client(
            "windsurf", "Windsurf", "mcpServers",
            lambda scope: _home() / ".codeium" / "windsurf" / "mcp_config.json",
        ),
        Client(
            "vscode", "VS Code", "servers",
            lambda scope: (Path.cwd() / ".vscode" / "mcp.json") if scope == "project"
            else _app_support("Code") / "User" / "mcp.json",
            scopes=("user", "project"),
        ),
        Client(
            "gemini", "Gemini CLI", "mcpServers",
            lambda scope: (Path.cwd() / ".gemini" / "settings.json") if scope == "project"
            else _home() / ".gemini" / "settings.json",
            scopes=("user", "project"),
        ),
        Client(
            "codex", "Codex", "mcp_servers",
            lambda scope: (Path.cwd() / ".codex" / "config.toml") if scope == "project"
            else Path(os.environ.get("CODEX_HOME") or _home() / ".codex") / "config.toml",
            fmt="toml", scopes=("user", "project"),
        ),
    ]
}


def get_client(name: str) -> Client:
    try:
        return CLIENTS[name]
    except KeyError:
        raise InstallError(
            f"unknown client {name!r}; choose one of: {', '.join(CLIENTS)}"
        ) from None


# ── launcher: the command a client runs to start the wrapper ─────────────


def launcher(kind: str = "uvx", package: str = DEFAULT_PACKAGE, *, portable: bool = False) -> list[str]:
    """
    Command prefix that starts ``ctxtual``.

    ``uvx`` (default) runs the published package in an isolated environment
    and is written with its absolute path, because GUI apps such as Claude
    Desktop start servers with a minimal ``PATH``.  ``python`` uses the
    current interpreter, which must have ``ctxtual[mcp]`` installed.
    ``portable`` writes plain ``uvx`` (for snippets and links meant to be
    shared).
    """
    if kind == "python":
        return [sys.executable, "-m", "ctxtual"]
    if kind != "uvx":
        raise InstallError(f"unknown launcher {kind!r} (use 'uvx' or 'python')")
    if portable:
        return ["uvx", "--from", package, "ctxtual"]
    uvx = shutil.which("uvx")
    if uvx is None:
        raise InstallError(
            "uvx was not found. Install uv (https://docs.astral.sh/uv/) or pass "
            "--launcher python from an environment where ctxtual[mcp] is installed."
        )
    return [uvx, "--from", package, "ctxtual"]


# ── entry transformation ──────────────────────────────────────────────────


def referenced_env_vars(entry: dict[str, Any]) -> list[str]:
    """Names of ``${VAR}`` references in a wrapped entry's arguments."""
    found: list[str] = []
    for arg in entry.get("args") or []:
        for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(arg)):
            if var not in found:
                found.append(var)
    return found


def entry_kind(entry: dict[str, Any]) -> str:
    """``stdio``, ``remote`` or ``unknown``."""
    if entry.get("command"):
        return "stdio"
    if any(entry.get(k) for k in URL_KEYS):
        return "remote"
    return "unknown"


def is_wrapped(entry: dict[str, Any]) -> bool:
    args = [str(a) for a in entry.get("args") or []]
    cmd = Path(str(entry.get("command") or "")).name
    mentions = cmd.startswith("ctxtual") or any(
        a == "ctxtual" or a.startswith(("ctxtual[", "ctxtual==", "ctxtual>")) for a in args
    )
    return mentions and "wrap" in args


@dataclass
class WrapOptions:
    """``ctxtual wrap`` flags written into wrapped entries."""

    threshold: int | None = None
    ttl: float | None = None
    extra: list[str] = field(default_factory=list)

    def args(self) -> list[str]:
        out: list[str] = []
        if self.threshold is not None:
            out += ["--threshold", str(self.threshold)]
        if self.ttl is not None:
            out += ["--ttl", _number(self.ttl)]
        return out + list(self.extra)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _arg(value: Any) -> str:
    """A config value as a command-line argument (JSON spelling for non-strings)."""
    return value if isinstance(value, str) else json.dumps(value)


#: Environment variable names used to pass header values to the wrapper.
HEADER_VAR = "CTXTUAL_HEADER_{}"


def wrap_entry(
    entry: dict[str, Any],
    launch: Sequence[str],
    options: WrapOptions | None = None,
    *,
    client: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """
    Return ``(wrapped_entry, env_vars)``: the stdio entry that runs *entry*
    through ``ctxtual wrap``, and the environment variables it reads.

    Header values are never written into ``args`` (visible to every local
    user in ``ps``): literal values go into the entry's ``env`` and are
    referenced as ``${CTXTUAL_HEADER_n}``; values that are already
    ``${VAR}`` references stay as they are.  For Codex, the variables are
    also added to ``env_vars`` so Codex passes them on.
    """
    options = options or WrapOptions()
    kind = entry_kind(entry)
    if kind == "unknown":
        raise InstallError("entry has neither a command nor a URL")
    new = copy.deepcopy(entry)
    wrap_args = ["wrap", *options.args()]
    env_vars: list[str] = []

    if kind == "stdio":
        upstream = [_arg(entry["command"]), *[_arg(a) for a in entry.get("args") or []]]
        wrap_args += ["--", *upstream]
    else:
        url = next(str(entry[k]) for k in URL_KEYS if entry.get(k))
        headers: dict[str, str] = {}
        for key in ("headers", "http_headers"):
            headers.update({str(k): _arg(v) for k, v in (entry.get(key) or {}).items()})
        # Codex: header values that come from environment variables.
        for header, var in (entry.get("env_http_headers") or {}).items():
            headers[str(header)] = f"${{{var}}}"
        if entry.get("bearer_token_env_var"):
            headers["Authorization"] = f"Bearer ${{{entry['bearer_token_env_var']}}}"
        declared = str(entry.get("type") or entry.get("transport") or "").lower()
        legacy_sse = declared == "sse" or (
            not declared
            and (
                urlsplit(url).path.rstrip("/").endswith("/sse")
                or (client == "gemini" and "url" in entry and "httpUrl" not in entry)
            )
        )
        for k in (*URL_KEYS, "headers", "http_headers", "env_http_headers",
                  "bearer_token_env_var", "transport"):
            new.pop(k, None)
        wrap_args.append(url)
        if legacy_sse:
            wrap_args.append("--sse")
        env = new.setdefault("env", {})
        for i, (name, value) in enumerate(headers.items(), 1):
            if not ("${" in value and _ENV_ONLY.fullmatch(value)):
                var = HEADER_VAR.format(i)
                env[var] = value
                value = f"${{{var}}}"
            wrap_args += ["--header", f"{name}: {value}"]
        if not env:
            new.pop("env")
        if not any(h.lower() == "authorization" for h in headers):
            # The client used to run OAuth itself; now the wrapper does.
            # It only starts when the server answers 401.
            wrap_args.append("--oauth")
        for arg in wrap_args:
            for var in _ENV_REF.findall(arg):
                if var not in env and var not in env_vars:
                    env_vars.append(var)

    new["command"] = launch[0]
    new["args"] = [*launch[1:], *wrap_args]
    if client in ("vscode", "claude-code") or "type" in new:
        new["type"] = "stdio"
    if client == "codex" and env_vars:
        existing = [v for v in new.get("env_vars") or [] if v not in env_vars]
        new["env_vars"] = [*existing, *env_vars]
    return new, env_vars


def redact(entry: dict[str, Any]) -> dict[str, Any]:
    """Copy of *entry* with environment values hidden, for display."""
    shown = copy.deepcopy(entry)
    if isinstance(shown.get("env"), dict):
        shown["env"] = {k: "***" for k in shown["env"]}
    return shown


# ── config files ──────────────────────────────────────────────────────────


@dataclass
class Change:
    server: str
    action: str  # "wrap" | "skip" | "restore"
    reason: str = ""
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


def state_path(client: Client, config: Path, scope: str) -> Path:
    base = Path(os.environ.get("CTXTUAL_HOME") or _home() / ".ctxtual") / "installs"
    digest = hashlib.sha256(f"{config.resolve()}|{scope}".encode()).hexdigest()[:12]
    return base / f"{client.name}-{scope}-{digest}.json"


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if state:
        _atomic_write(path, json.dumps(state, indent=2) + "\n", mode=0o600)
    elif path.exists():
        path.unlink()


def _target(path: Path) -> Path:
    """The file behind *path* (configs are often symlinks into dotfiles)."""
    return path.resolve() if path.is_symlink() else path


def _atomic_write(path: Path, text: str, *, mode: int | None = None) -> None:
    import tempfile

    path = _target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".ctxtual-tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        elif path.exists():
            shutil.copymode(path, tmp)
        else:
            os.chmod(tmp, 0o600)  # new config files may hold credentials
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _backup(path: Path) -> Path | None:
    path = _target(path)
    if not path.exists():
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = path.with_name(f"{path.name}.ctxtual-backup-{stamp}")
    n = 1
    while backup.exists():  # never overwrite an earlier backup
        backup = path.with_name(f"{path.name}.ctxtual-backup-{stamp}-{n}")
        n += 1
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)
    return backup


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InstallError(
            f"{path} is not plain JSON ({exc.msg} at line {exc.lineno}). It may "
            "contain comments, which ctxtual will not rewrite; wrap servers by "
            "hand with the snippet from `ctxtual add --print`."
        ) from None
    if not isinstance(data, dict):
        raise InstallError(f"{path} does not contain a JSON object")
    return data


class ConfigFile:
    """Read/modify/write the servers section of one client config."""

    def __init__(self, client: Client, scope: str = "user", path: Path | None = None) -> None:
        if scope not in client.scopes:
            raise InstallError(
                f"{client.title} has no {scope!r} scope (choose: {', '.join(client.scopes)})"
            )
        self.client = client
        self.scope = scope
        self.path = path or client.path(scope)
        self._doc: Any = None

    # reading

    def servers(self) -> dict[str, Any]:
        """Server entries by name (values are dicts, or whatever the file has)."""
        if self.client.fmt == "toml":
            section = self._toml().get("mcp_servers") or {}
            return {str(name): _plain(entry) for name, entry in section.items()}
        data = _read_json(self.path)
        if self.client.fmt == "claude-code" and self.scope == "local":
            data = (data.get("projects") or {}).get(str(Path.cwd())) or {}
        section = data.get(self.client.servers_key) or {}
        if not isinstance(section, dict):
            raise InstallError(f"{self.client.servers_key} in {self.path} is not an object")
        return {name: (dict(e) if isinstance(e, dict) else e) for name, e in section.items()}

    def _toml(self) -> Any:
        if self._doc is None:
            tomlkit = _tomlkit()
            self._doc = tomlkit.parse(self.path.read_text()) if self.path.exists() else tomlkit.document()
        return self._doc

    # writing

    def write(
        self,
        updates: dict[str, dict[str, Any]],
        *,
        previous: dict[str, Any] | None = None,
    ) -> Path | None:
        """Replace the given server entries; returns the backup path."""
        if self.client.fmt == "claude-code":
            backup = _backup(self.path)
            for name, entry in updates.items():
                _claude_cli_replace(name, entry, (previous or {}).get(name), self.scope)
            return backup
        if self.client.fmt == "toml":
            tomlkit = _tomlkit()
            from tomlkit.items import InlineTable

            doc = self._toml()
            servers = doc.get("mcp_servers")
            if isinstance(servers, InlineTable):
                raise InstallError(
                    f"mcp_servers in {self.path} is an inline table; write each server "
                    "as a [mcp_servers.NAME] table and run again."
                )
            if servers is None:
                servers = tomlkit.table(is_super_table=True)
                doc["mcp_servers"] = servers
            backup = _backup(self.path)
            for name, entry in updates.items():
                current = servers.get(name)
                if hasattr(current, "keys"):
                    _update_toml_table(current, entry)  # keeps unrelated comments
                else:
                    servers[name] = entry
            _atomic_write(self.path, tomlkit.dumps(doc))
            return backup
        data = _read_json(self.path)
        if not isinstance(data.get(self.client.servers_key), dict):
            data[self.client.servers_key] = {}
        backup = _backup(self.path)
        section = data[self.client.servers_key]
        for name, entry in updates.items():
            section[name] = entry
        _atomic_write(self.path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return backup


def _update_toml_table(table: Any, entry: dict[str, Any]) -> None:
    """Make *table* equal *entry*, touching only the keys that differ."""
    for key in list(table.keys()):
        if key not in entry:
            del table[key]
    for key, value in entry.items():
        current = table.get(key)
        if isinstance(value, dict) and hasattr(current, "keys"):
            _update_toml_table(current, value)
        elif _plain(current) != value:
            table[key] = value


def _plain(value: Any) -> Any:
    """tomlkit containers → plain Python values."""
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return value


def _tomlkit() -> Any:
    try:
        import tomlkit
    except ImportError:
        raise InstallError("editing Codex config needs tomlkit: pip install 'ctxtual[mcp]'") from None
    return tomlkit


def _claude_payload(entry: dict[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    if "type" not in payload:
        payload["type"] = "stdio" if payload.get("command") else "http"
    return payload


def _claude_cli_replace(
    name: str, entry: dict[str, Any], previous: Any, scope: str
) -> None:
    claude = shutil.which("claude")
    if claude is None:
        raise InstallError(
            "the `claude` CLI was not found; install Claude Code, or add the server "
            "yourself with `ctxtual add claude-code NAME … --print`."
        )
    subprocess.run([claude, "mcp", "remove", name, "--scope", scope],
                   check=False, capture_output=True)
    added = subprocess.run(
        [claude, "mcp", "add-json", name, json.dumps(_claude_payload(entry)), "--scope", scope],
        check=False, capture_output=True, text=True,
    )
    if added.returncode == 0:
        return
    restored = ""
    if isinstance(previous, dict):
        back = subprocess.run(
            [claude, "mcp", "add-json", name, json.dumps(_claude_payload(previous)), "--scope", scope],
            check=False, capture_output=True, text=True,
        )
        restored = " The previous entry was put back." if back.returncode == 0 else (
            " Putting the previous entry back failed too; restore it from the backup."
        )
    detail = (added.stderr or added.stdout or "").strip()
    raise InstallError(f"`claude mcp add-json {name}` failed: {detail}.{restored}")


def claude_code_command(name: str, entry: dict[str, Any], scope: str = "user") -> str:
    return shlex.join(
        ["claude", "mcp", "add-json", name, json.dumps(_claude_payload(entry)), "--scope", scope]
    )


# ── install / uninstall ───────────────────────────────────────────────────


def _skip_reason(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return "not a server entry"
    if is_wrapped(entry):
        return "already wrapped"
    if entry.get("disabled") is True or entry.get("enabled") is False:
        return "disabled"
    return None


def plan_install(
    config: ConfigFile,
    *,
    launch: Sequence[str],
    servers: Iterable[str] | None = None,
    options: WrapOptions | None = None,
) -> list[Change]:
    existing = config.servers()
    wanted = set(servers) if servers else None
    if wanted:
        missing = sorted(wanted - set(existing))
        if missing:
            raise InstallError(f"no server named {', '.join(missing)} in {config.path}")
    changes: list[Change] = []
    for name, entry in existing.items():
        if wanted is not None and name not in wanted:
            continue
        reason = _skip_reason(entry)
        if reason:
            changes.append(Change(name, "skip", reason))
            continue
        try:
            new, _ = wrap_entry(entry, launch, options, client=config.client.name)
        except InstallError as exc:
            changes.append(Change(name, "skip", str(exc)))
            continue
        changes.append(Change(name, "wrap", before=entry, after=new))
    return changes


def install(
    config: ConfigFile,
    *,
    launch: Sequence[str],
    servers: Iterable[str] | None = None,
    options: WrapOptions | None = None,
    dry_run: bool = False,
) -> tuple[list[Change], Path | None]:
    changes = plan_install(config, launch=launch, servers=servers, options=options)
    todo = {c.server: c.after for c in changes if c.action == "wrap"}
    if dry_run or not todo:
        return changes, None
    previous = {c.server: c.before for c in changes if c.action == "wrap"}
    backup = config.write(todo, previous=previous)  # type: ignore[arg-type]
    # Record what was replaced only once the write succeeded.  A fresh
    # install always overwrites: the entry may have changed since last time.
    spath = state_path(config.client, config.path, config.scope)
    state = _load_state(spath)
    for c in changes:
        if c.action == "wrap":
            state[c.server] = {"original": c.before, "wrapped": c.after}
    _save_state(spath, state)
    return changes, backup


def _restore_entry(saved: Any, current: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """
    The entry to restore for *current* (a wrapped entry), and a note.

    With saved state: the original entry, plus any edits made to the wrapped
    entry since install (other keys, env values, the upstream command).
    Without state: reconstructed from ``args`` for local servers.
    """
    if isinstance(saved, dict) and "original" in saved and "wrapped" in saved:
        original, wrapped = saved["original"], saved["wrapped"]
    elif isinstance(saved, dict):  # state written by ctxtual 0.3.0 pre-release
        original, wrapped = saved, None
    else:
        unwrapped = unwrap_entry(current)
        if unwrapped is None:
            return None, "not installed by `ctxtual install`; edit or remove it in the client"
        return unwrapped, ""
    if wrapped is None or current == wrapped:
        return copy.deepcopy(original), ""
    restored = copy.deepcopy(original)
    added_env = set(wrapped.get("env") or {}) - set(original.get("env") or {})
    added_vars = set(wrapped.get("env_vars") or []) - set(original.get("env_vars") or [])
    wrapper_keys = {"command", "args", "type"}
    for key, value in current.items():
        if key in wrapper_keys:
            continue
        if key == "env" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in added_env}
            if not value and "env" not in original:
                continue
        if key == "env_vars" and isinstance(value, list):
            value = [v for v in value if v not in added_vars]
            if not value and "env_vars" not in original:
                continue
        restored[key] = value
    for key in wrapped:
        if key not in current and key not in wrapper_keys:
            restored.pop(key, None)
    if entry_kind(original) == "stdio":
        upstream = unwrap_entry(current)
        if upstream is not None:
            restored["command"], restored["args"] = upstream["command"], upstream["args"]
    return restored, "kept your edits made after install"


def uninstall(
    config: ConfigFile,
    *,
    servers: Iterable[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[Change], Path | None]:
    existing = config.servers()
    spath = state_path(config.client, config.path, config.scope)
    state = _load_state(spath)
    wanted = set(servers) if servers else None
    changes: list[Change] = []
    restore: dict[str, dict[str, Any]] = {}
    for name, entry in existing.items():
        if wanted is not None and name not in wanted:
            continue
        if not isinstance(entry, dict) or not is_wrapped(entry):
            continue
        original, note = _restore_entry(state.get(name), entry)
        if original is None:
            changes.append(Change(name, "skip", note))
            continue
        restore[name] = original
        changes.append(Change(name, "restore", reason=note, before=entry, after=original))
    if dry_run or not restore:
        return changes, None
    backup = config.write(restore, previous=existing)
    for name in restore:
        state.pop(name, None)
    _save_state(spath, state)
    return changes, backup


def unwrap_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort reconstruction of a wrapped stdio entry (no saved state)."""
    args = [str(a) for a in entry.get("args") or []]
    if "--" not in args:
        return None
    upstream = args[args.index("--") + 1:]
    if not upstream:
        return None
    original = {k: v for k, v in entry.items() if k not in ("command", "args")}
    original["command"] = upstream[0]
    original["args"] = upstream[1:]
    return original


# ── links and snippets for adding new servers ────────────────────────────


def cursor_link(name: str, entry: dict[str, Any]) -> str:
    import base64
    from urllib.parse import quote

    config = base64.b64encode(json.dumps(entry, separators=(",", ":")).encode()).decode()
    # base64 may contain "+" and "/", which query parsers would mangle.
    return (
        "cursor://anysphere.cursor-deeplink/mcp/install"
        f"?name={quote(name, safe='')}&config={quote(config, safe='')}"
    )


def vscode_link(name: str, entry: dict[str, Any]) -> str:
    from urllib.parse import quote

    payload = {"name": name, "type": "stdio", **{k: v for k, v in entry.items() if k != "type"}}
    return "vscode:mcp/install?" + quote(json.dumps(payload, separators=(",", ":")))
