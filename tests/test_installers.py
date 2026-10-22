"""Tests for ``ctxtual install/uninstall/add`` (client config rewriting)."""

import base64
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from ctxtual import installers
from ctxtual.cli import main
from ctxtual.installers import (
    ConfigFile,
    InstallError,
    WrapOptions,
    cursor_link,
    get_client,
    install,
    is_wrapped,
    uninstall,
    unwrap_entry,
    vscode_link,
    wrap_entry,
)

LAUNCH = ["/usr/bin/python3", "-m", "ctxtual"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CTXTUAL_TEST_HOME", str(tmp_path))
    monkeypatch.setenv("CTXTUAL_HOME", str(tmp_path / ".ctxtual"))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(installers.platform, "system", lambda: "Darwin")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


# ── entry transformation ──────────────────────────────────────────────────


class TestWrapEntry:
    def test_stdio(self):
        entry = {
            "command": "npx",
            "args": ["-y", "pkg"],
            "env": {"TOKEN": "t"},
            "cwd": "/x",
        }
        new, env_vars = wrap_entry(entry, LAUNCH)
        assert new == {
            "command": "/usr/bin/python3",
            "args": ["-m", "ctxtual", "wrap", "--", "npx", "-y", "pkg"],
            "env": {"TOKEN": "t"},
            "cwd": "/x",
        }
        assert env_vars == []
        assert is_wrapped(new) and not is_wrapped(entry)
        assert unwrap_entry(new) == entry

    def test_options_are_written(self):
        new, _ = wrap_entry(
            {"command": "x"}, LAUNCH, WrapOptions(threshold=5000, ttl=600)
        )
        assert new["args"][2:7] == ["wrap", "--threshold", "5000", "--ttl", "600"]

    def test_remote_without_auth_uses_oauth(self):
        new, _ = wrap_entry({"url": "https://m.example/mcp"}, LAUNCH)
        assert new["args"][-2:] == ["https://m.example/mcp", "--oauth"]
        assert "url" not in new

    def test_remote_with_auth_header_keeps_secrets_out_of_args(self):
        entry = {
            "url": "https://m.example/mcp",
            "headers": {"Authorization": "Bearer abc", "X-A": "1"},
        }
        new, env_vars = wrap_entry(entry, LAUNCH)
        assert new["args"][2:] == [
            "wrap",
            "https://m.example/mcp",
            "--header",
            "Authorization: ${CTXTUAL_HEADER_1}",
            "--header",
            "X-A: ${CTXTUAL_HEADER_2}",
        ]
        # argv is visible to every local user (ps); env is not.
        assert new["env"] == {"CTXTUAL_HEADER_1": "Bearer abc", "CTXTUAL_HEADER_2": "1"}
        assert not any("abc" in a for a in new["args"])
        assert "headers" not in new and env_vars == []

    def test_codex_env_based_credentials(self):
        entry = {
            "url": "https://m.example/mcp",
            "bearer_token_env_var": "API_TOKEN",
            "env_http_headers": {"X-Key": "KEY_VAR"},
            "http_headers": {"X-R": "eu"},
        }
        new, env_vars = wrap_entry(entry, LAUNCH, client="codex")
        args = new["args"]
        assert "Authorization: Bearer ${API_TOKEN}" in args
        assert "X-Key: ${KEY_VAR}" in args
        assert "X-R: ${CTXTUAL_HEADER_1}" in args and new["env"] == {
            "CTXTUAL_HEADER_1": "eu"
        }
        assert sorted(env_vars) == ["API_TOKEN", "KEY_VAR"]
        assert sorted(new["env_vars"]) == ["API_TOKEN", "KEY_VAR"]
        assert "--oauth" not in args

    def test_legacy_sse_is_preserved(self):
        gem, _ = wrap_entry(
            {"url": "https://g.example/events"}, LAUNCH, client="gemini"
        )
        assert "--sse" in gem["args"]
        http, _ = wrap_entry(
            {"httpUrl": "https://g.example/mcp"}, LAUNCH, client="gemini"
        )
        assert "--sse" not in http["args"]
        typed, _ = wrap_entry(
            {"type": "sse", "url": "https://x/events"}, LAUNCH, client="cursor"
        )
        assert "--sse" in typed["args"] and typed["type"] == "stdio"
        by_path, _ = wrap_entry({"url": "https://x/sse"}, LAUNCH, client="cursor")
        assert "--sse" in by_path["args"]
        declared_http, _ = wrap_entry({"type": "http", "url": "https://x/sse"}, LAUNCH)
        assert "--sse" not in declared_http["args"]

    def test_non_string_args_use_json_spelling(self):
        new, _ = wrap_entry({"command": "srv", "args": [1, True, None, "x"]}, LAUNCH)
        assert new["args"][-4:] == ["1", "true", "null", "x"]

    def test_vscode_input_variables_go_through_env(self):
        entry = {
            "type": "http",
            "url": "https://c7/mcp",
            "headers": {"Authorization": "Bearer ${input:tok}"},
        }
        new, _ = wrap_entry(entry, LAUNCH, client="vscode")
        assert new["type"] == "stdio"
        assert new["env"] == {"CTXTUAL_HEADER_1": "Bearer ${input:tok}"}
        assert "Authorization: ${CTXTUAL_HEADER_1}" in new["args"]

    def test_unknown_entry(self):
        with pytest.raises(InstallError):
            wrap_entry({"something": 1}, LAUNCH)


# ── install / uninstall round trips ───────────────────────────────────────

JSON_CASES = {
    "claude-desktop": (
        "Library/Application Support/Claude/claude_desktop_config.json",
        {
            "mcpServers": {
                "fs": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/d"],
                },
                "gh": {
                    "command": "docker",
                    "args": ["run", "-i", "ghcr.io/github/github-mcp-server"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp"},
                },
            },
            "globalShortcut": "Ctrl+Space",
        },
    ),
    "cursor": (
        ".cursor/mcp.json",
        {
            "mcpServers": {
                "linear": {"url": "https://mcp.linear.app/mcp"},
                "api": {
                    "url": "https://a/mcp",
                    "headers": {"Authorization": "Bearer x"},
                },
                "off": {"command": "x", "disabled": True},
            }
        },
    ),
    "windsurf": (
        ".codeium/windsurf/mcp_config.json",
        {"mcpServers": {"w": {"serverUrl": "https://w/mcp"}}},
    ),
    "vscode": (
        "Library/Application Support/Code/User/mcp.json",
        {
            "servers": {
                "pw": {"type": "stdio", "command": "npx", "args": ["@playwright/mcp"]},
                "c7": {"type": "http", "url": "https://c7/mcp"},
            },
            "inputs": [{"type": "promptString", "id": "t", "description": "t"}],
        },
    ),
    "gemini": (
        ".gemini/settings.json",
        {
            "theme": "dark",
            "mcpServers": {
                "r": {"httpUrl": "https://g/mcp", "headers": {"X": "1"}},
                "l": {"command": "python", "args": ["s.py"], "cwd": "/t"},
            },
        },
    ),
}


class TestInstallRoundTrip:
    @pytest.mark.parametrize("client", sorted(JSON_CASES))
    def test_json_clients(self, home, client):
        rel, original = JSON_CASES[client]
        path = write_json(home / rel, original)
        config = ConfigFile(get_client(client))
        assert config.path == path

        changes, backup = install(config, launch=LAUNCH)
        wrapped = [c.server for c in changes if c.action == "wrap"]
        assert wrapped
        assert backup is not None and json.loads(backup.read_text()) == original
        after = json.loads(path.read_text())
        key = get_client(client).servers_key
        for name in wrapped:
            assert is_wrapped(after[key][name])
        # untouched settings survive
        for k in original:
            if k != key:
                assert after[k] == original[k]
        if client == "cursor":
            assert after[key]["off"] == original[key]["off"]  # disabled: skipped

        again, _ = install(config, launch=LAUNCH)
        assert {c.action for c in again} == {"skip"}

        restored, _ = uninstall(config)
        assert {c.server for c in restored} == set(wrapped)
        assert json.loads(path.read_text()) == original
        assert not list((home / ".ctxtual" / "installs").glob("*.json"))

    def test_subset_and_dry_run(self, home):
        rel, original = JSON_CASES["claude-desktop"]
        path = write_json(home / rel, original)
        config = ConfigFile(get_client("claude-desktop"))
        changes, backup = install(config, launch=LAUNCH, dry_run=True)
        assert backup is None and json.loads(path.read_text()) == original
        assert {c.action for c in changes} == {"wrap"}
        install(config, launch=LAUNCH, servers=["gh"])
        data = json.loads(path.read_text())["mcpServers"]
        assert is_wrapped(data["gh"]) and not is_wrapped(data["fs"])
        with pytest.raises(InstallError, match="no server named nope"):
            install(config, launch=LAUNCH, servers=["nope"])

    def test_refuses_json_with_comments(self, home):
        path = home / ".cursor" / "mcp.json"
        path.parent.mkdir(parents=True)
        text = '{\n  // my servers\n  "mcpServers": {"a": {"command": "x"}}\n}\n'
        path.write_text(text)
        with pytest.raises(InstallError, match="comments"):
            install(ConfigFile(get_client("cursor")), launch=LAUNCH)
        assert path.read_text() == text

    def test_uninstall_without_state_reconstructs_stdio(self, home):
        rel, original = JSON_CASES["claude-desktop"]
        path = write_json(home / rel, original)
        config = ConfigFile(get_client("claude-desktop"))
        install(config, launch=LAUNCH)
        for f in (home / ".ctxtual" / "installs").glob("*.json"):
            f.unlink()
        uninstall(config)
        assert json.loads(path.read_text()) == original

    def test_project_scope(self, home):
        path = write_json(
            home / ".cursor" / "mcp.json", {"mcpServers": {"p": {"command": "x"}}}
        )
        config = ConfigFile(get_client("cursor"), "project")
        assert config.path == path  # cwd is home in this fixture
        with pytest.raises(InstallError, match="no 'local' scope"):
            ConfigFile(get_client("cursor"), "local")


class TestEditsAndFiles:
    def test_reinstall_after_manual_edit_restores_the_new_entry(self, home):
        path = write_json(
            home / ".cursor" / "mcp.json",
            {"mcpServers": {"s": {"command": "old", "args": ["v1"]}}},
        )
        config = ConfigFile(get_client("cursor"))
        install(config, launch=LAUNCH)
        new_entry = {"command": "new", "args": ["v2"], "env": {"K": "new"}}
        write_json(path, {"mcpServers": {"s": new_entry}})
        install(config, launch=LAUNCH)
        uninstall(config)
        assert json.loads(path.read_text())["mcpServers"]["s"] == new_entry

    def test_edits_to_the_wrapped_entry_survive_uninstall(self, home):
        original = {"command": "srv", "args": ["a"], "env": {"TOKEN": "old"}}
        path = write_json(
            home / ".cursor" / "mcp.json", {"mcpServers": {"s": original}}
        )
        config = ConfigFile(get_client("cursor"))
        install(config, launch=LAUNCH)
        data = json.loads(path.read_text())
        data["mcpServers"]["s"]["env"]["TOKEN"] = "rotated"
        data["mcpServers"]["s"]["args"].append("b")  # new upstream argument
        write_json(path, data)
        changes, _ = uninstall(config)
        assert "kept your edits" in changes[0].reason
        assert json.loads(path.read_text())["mcpServers"]["s"] == {
            "command": "srv",
            "args": ["a", "b"],
            "env": {"TOKEN": "rotated"},
        }

    def test_header_env_is_removed_again_on_uninstall_after_edit(self, home):
        original = {"url": "https://a/mcp", "headers": {"Authorization": "Bearer x"}}
        path = write_json(
            home / ".cursor" / "mcp.json", {"mcpServers": {"a": original}}
        )
        config = ConfigFile(get_client("cursor"))
        install(config, launch=LAUNCH)
        data = json.loads(path.read_text())
        data["mcpServers"]["a"]["env"]["EXTRA"] = "1"
        write_json(path, data)
        uninstall(config)
        assert json.loads(path.read_text())["mcpServers"]["a"] == {
            **original,
            "env": {"EXTRA": "1"},
        }

    def test_symlinked_config_stays_a_symlink(self, home):
        real = write_json(
            home / "dotfiles" / "mcp.json", {"mcpServers": {"s": {"command": "x"}}}
        )
        link = home / ".cursor" / "mcp.json"
        link.parent.mkdir()
        link.symlink_to(real)
        config = ConfigFile(get_client("cursor"))
        install(config, launch=LAUNCH)
        assert link.is_symlink()
        assert is_wrapped(json.loads(real.read_text())["mcpServers"]["s"])
        uninstall(config)
        assert link.is_symlink() and json.loads(real.read_text())["mcpServers"][
            "s"
        ] == {"command": "x"}

    def test_backups_are_never_overwritten(self, home):
        pristine = {"mcpServers": {"a": {"command": "x"}, "b": {"command": "y"}}}
        path = write_json(home / ".cursor" / "mcp.json", pristine)
        config = ConfigFile(get_client("cursor"))
        _, first = install(config, launch=LAUNCH, servers=["a"])
        _, second = install(config, launch=LAUNCH, servers=["b"])
        assert first != second
        assert json.loads(first.read_text()) == pristine
        assert first.stat().st_mode & 0o077 == 0  # backups may hold secrets
        assert path.exists()

    def test_unusual_shapes(self, home):
        path = write_json(home / ".cursor" / "mcp.json", {"mcpServers": None})
        config = ConfigFile(get_client("cursor"))
        assert config.servers() == {}
        config.write({"n": {"command": "x"}})
        assert json.loads(path.read_text())["mcpServers"] == {"n": {"command": "x"}}
        write_json(
            path, {"mcpServers": {"_comment": "my servers", "s": {"command": "x"}}}
        )
        changes, _ = install(config, launch=LAUNCH)
        assert {c.server: c.action for c in changes} == {
            "_comment": "skip",
            "s": "wrap",
        }
        uninstall(config)
        assert json.loads(path.read_text())["mcpServers"]["_comment"] == "my servers"

    def test_added_remote_server_is_left_alone_by_uninstall(self, home, capsys):
        write_json(home / ".cursor" / "mcp.json", {"mcpServers": {}})
        assert (
            main(["add", "cursor", "r", "https://r/mcp", "--launcher", "python"]) == 0
        )
        changes, _ = uninstall(ConfigFile(get_client("cursor")))
        assert changes[0].action == "skip" and "ctxtual install" in changes[0].reason


class TestCodex:
    TOML = """model = "gpt-5.5"  # keep me

# docs server
[mcp_servers.docs]
command = "npx"
args = ["-y", "some-mcp-server"]
startup_timeout_sec = 20

[mcp_servers.docs.env]
MY_VAR = "value"

[mcp_servers.api]
url = "https://mcp.example.com/mcp"
bearer_token_env_var = "API_TOKEN"

[profiles.fast]
model = "gpt-5.5-mini"
"""

    def test_round_trip(self, home):
        import tomllib

        pytest.importorskip("tomlkit")
        path = home / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text(self.TOML)
        config = ConfigFile(get_client("codex"))
        install(config, launch=LAUNCH)
        text = path.read_text()
        assert "# keep me" in text and "# docs server" in text
        data = tomllib.loads(text)
        docs, api = data["mcp_servers"]["docs"], data["mcp_servers"]["api"]
        assert docs["args"][-3:] == ["npx", "-y", "some-mcp-server"]
        assert docs["env"] == {"MY_VAR": "value"} and docs["startup_timeout_sec"] == 20
        assert "Authorization: Bearer ${API_TOKEN}" in api["args"]
        assert api["env_vars"] == ["API_TOKEN"]  # Codex forwards only listed vars
        assert "url" not in api
        assert data["profiles"] == {"fast": {"model": "gpt-5.5-mini"}}
        uninstall(config)
        assert tomllib.loads(path.read_text()) == tomllib.loads(self.TOML)

    def test_comments_inside_server_tables_survive(self, home):
        pytest.importorskip("tomlkit")
        text = (
            "[mcp_servers.docs]\n"
            "# why this server exists\n"
            'command = "npx"\n'
            'args = ["-y", "pkg"]\n'
            "startup_timeout_sec = 20 # slow machine\n"
        )
        path = home / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text(text)
        config = ConfigFile(get_client("codex"))
        install(config, launch=LAUNCH)
        wrapped = path.read_text()
        assert "# why this server exists" in wrapped and "# slow machine" in wrapped
        uninstall(config)
        assert path.read_text() == text

    def test_disabled_and_inline_tables(self, home):
        pytest.importorskip("tomlkit")
        path = home / ".codex" / "config.toml"
        path.parent.mkdir()
        path.write_text('[mcp_servers.off]\ncommand = "x"\nenabled = false\n')
        changes, _ = install(ConfigFile(get_client("codex")), launch=LAUNCH)
        assert changes[0].action == "skip" and changes[0].reason == "disabled"
        path.write_text('mcp_servers = { foo = { command = "npx", args = ["a"] } }\n')
        with pytest.raises(InstallError, match="inline table"):
            install(ConfigFile(get_client("codex")), launch=LAUNCH)
        assert (
            path.read_text()
            == 'mcp_servers = { foo = { command = "npx", args = ["a"] } }\n'
        )
        assert not list((home / ".ctxtual" / "installs").glob("*.json"))


class TestClaudeCode:
    def test_uses_the_claude_cli(self, home, monkeypatch):
        write_json(
            home / ".claude.json",
            {
                "mcpServers": {
                    "gh": {"type": "stdio", "command": "npx", "args": ["gh-mcp"]}
                },
                "numStartups": 3,
            },
        )
        calls = []

        class Done:
            returncode, stdout, stderr = 0, "", ""

        def run(cmd, **kw):
            calls.append(cmd)
            return Done()

        monkeypatch.setattr(installers.shutil, "which", lambda name: f"/bin/{name}")
        monkeypatch.setattr(installers.subprocess, "run", run)
        install(ConfigFile(get_client("claude-code")), launch=LAUNCH)
        assert calls[0] == ["/bin/claude", "mcp", "remove", "gh", "--scope", "user"]
        assert calls[1][:4] == ["/bin/claude", "mcp", "add-json", "gh"]
        payload = json.loads(calls[1][4])
        assert payload["type"] == "stdio" and is_wrapped(payload)
        # ~/.claude.json itself is never rewritten by ctxtual, but backed up
        assert json.loads((home / ".claude.json").read_text())["numStartups"] == 3
        assert list(home.glob(".claude.json.ctxtual-backup-*"))

    def test_uninstall_restores_remote_types_and_rolls_back_failures(
        self, home, monkeypatch
    ):
        remote = {
            "type": "http",
            "url": "https://r/mcp",
            "headers": {"Authorization": "Bearer t"},
        }
        config_path = write_json(
            home / ".claude.json", {"mcpServers": {"remote": remote}}
        )
        added = []

        class Result:
            def __init__(self, code=0, err=""):
                self.returncode, self.stdout, self.stderr = code, "", err

        fail_next = {"on": False}

        def run(cmd, **kw):
            if cmd[2] == "add-json":
                if fail_next["on"]:
                    fail_next["on"] = False
                    return Result(1, "invalid config")
                added.append(json.loads(cmd[4]))
                # emulate the CLI writing the file
                data = json.loads(config_path.read_text())
                data["mcpServers"][cmd[3]] = json.loads(cmd[4])
                config_path.write_text(json.dumps(data))
            return Result()

        monkeypatch.setattr(installers.shutil, "which", lambda name: f"/bin/{name}")
        monkeypatch.setattr(installers.subprocess, "run", run)
        config = ConfigFile(get_client("claude-code"))
        install(config, launch=LAUNCH)
        assert added[-1]["type"] == "stdio" and is_wrapped(added[-1])
        uninstall(config)
        assert added[-1] == remote  # type http again, not stdio
        # a failing add puts the previous entry back and reports it
        fail_next["on"] = True
        with pytest.raises(InstallError, match=r"invalid config.*put back"):
            install(config, launch=LAUNCH)
        assert added[-1] == remote


# ── links, launcher, CLI ──────────────────────────────────────────────────


class TestLinks:
    def test_cursor_link_round_trips(self):
        entry = {
            "command": "uvx",
            "args": ["--from", "ctxtual[mcp]", "ctxtual", "wrap", "x?y=+/"],
        }
        url = cursor_link("my server", entry)
        query = parse_qs(urlparse(url).query)
        assert query["name"] == ["my server"]
        assert json.loads(base64.b64decode(query["config"][0])) == entry

    def test_vscode_link(self):
        url = vscode_link("s", {"command": "uvx", "args": ["a"]})
        payload = json.loads(unquote(url.split("?", 1)[1]))
        assert payload == {
            "name": "s",
            "type": "stdio",
            "command": "uvx",
            "args": ["a"],
        }


class TestLauncher:
    def test_python_and_portable(self):
        assert installers.launcher("python") == [sys.executable, "-m", "ctxtual"]
        assert installers.launcher("uvx", portable=True) == [
            "uvx",
            "--from",
            "ctxtual[mcp]",
            "ctxtual",
        ]

    def test_missing_uvx(self, monkeypatch):
        monkeypatch.setattr(installers.shutil, "which", lambda name: None)
        with pytest.raises(InstallError, match="uvx was not found"):
            installers.launcher("uvx")

    def test_absolute_uvx(self, monkeypatch):
        monkeypatch.setattr(installers.shutil, "which", lambda name: "/opt/bin/uvx")
        assert installers.launcher("uvx")[0] == "/opt/bin/uvx"


class TestCli:
    def test_install_status_uninstall(self, home, capsys):
        rel, original = JSON_CASES["cursor"]
        path = write_json(home / rel, original)
        assert (
            main(["install", "cursor", "--launcher", "python", "--threshold", "4000"])
            == 0
        )
        out = capsys.readouterr().out
        assert "2 server(s) wrapped" in out and "Restart Cursor" in out
        data = json.loads(path.read_text())["mcpServers"]
        assert "--threshold" in data["linear"]["args"]
        assert main(["status", "cursor"]) == 0
        assert "wrapped" in capsys.readouterr().out
        assert main(["uninstall", "cursor"]) == 0
        assert json.loads(path.read_text()) == original

    def test_missing_config(self, home, capsys):
        assert main(["install", "windsurf", "--launcher", "python"]) == 2
        assert "config not found" in capsys.readouterr().err

    def test_add_writes_and_refuses_duplicates(self, home, capsys):
        rel, original = JSON_CASES["claude-desktop"]
        path = write_json(home / rel, original)
        assert (
            main(
                [
                    "add",
                    "claude-desktop",
                    "notion",
                    "https://mcp.notion.com/mcp",
                    "--launcher",
                    "python",
                ]
            )
            == 0
        )
        entry = json.loads(path.read_text())["mcpServers"]["notion"]
        assert entry["args"][-2:] == ["https://mcp.notion.com/mcp", "--oauth"]
        assert (
            main(
                [
                    "add",
                    "claude-desktop",
                    "notion",
                    "https://x/mcp",
                    "--launcher",
                    "python",
                ]
            )
            == 2
        )

    def test_add_print_snippets(self, home, capsys):
        assert (
            main(
                [
                    "add",
                    "codex",
                    "gh",
                    "https://api.example/mcp",
                    "-H",
                    "Authorization: Bearer ${GH_TOKEN}",
                    "--print",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert (
            'env_vars = ["GH_TOKEN"]' in out and "GH_TOKEN from its environment" in out
        )
        assert (
            main(["add", "cursor", "fs", "--print", "--", "npx", "-y", "fs-mcp"]) == 0
        )
        out = capsys.readouterr().out
        assert out.startswith(
            "Install link:\ncursor://anysphere.cursor-deeplink/mcp/install?"
        )
        assert '"command": "uvx"' in out  # portable, not a local absolute path

    def test_dry_run_hides_secrets(self, home, capsys):
        write_json(
            home / ".cursor" / "mcp.json",
            {
                "mcpServers": {
                    "a": {
                        "url": "https://a/mcp",
                        "headers": {"Authorization": "Bearer s3cret"},
                    }
                }
            },
        )
        assert main(["install", "cursor", "--dry-run", "--launcher", "python"]) == 0
        out = capsys.readouterr().out
        assert "s3cret" not in out and "CTXTUAL_HEADER_1" in out

    def test_add_validates_headers(self, home, capsys):
        write_json(home / ".cursor" / "mcp.json", {"mcpServers": {}})
        assert (
            main(
                [
                    "add",
                    "cursor",
                    "x",
                    "https://x/mcp",
                    "-H",
                    "Authorization=Bearer t",
                    "--launcher",
                    "python",
                ]
            )
            == 2
        )
        assert "bad --header" in capsys.readouterr().err

    def test_wrap_needs_a_target(self, capsys):
        assert main(["wrap"]) == 2
        assert "nothing to wrap" in capsys.readouterr().err
        assert main(["wrap", "https://x/mcp", "--", "cmd"]) == 2
