"""
Benchmark tasks: MCP tools that return large, realistic results, and
questions with one exact answer computed from the same data.

Every task is deterministic for a given size and seed, so runs are
comparable across models, conditions and days.
"""

import json
import random
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP

#: Approximate characters of the main tool result, per size.  "medium" fits
#: in most context windows (~30k tokens); "large" doesn't (~300k tokens).
SIZES = {"medium": 120_000, "large": 1_200_000}

_WORDS = [
    "request",
    "response",
    "handler",
    "cache",
    "token",
    "parser",
    "buffer",
    "queue",
    "worker",
    "thread",
    "config",
    "session",
    "client",
    "server",
    "schema",
    "index",
    "payload",
    "stream",
    "socket",
    "retry",
    "timeout",
    "latency",
    "module",
    "plugin",
    "adapter",
    "router",
    "metric",
    "trace",
    "event",
    "batch",
    "commit",
    "branch",
    "release",
    "build",
    "deploy",
    "rollback",
    "migration",
    "snapshot",
    "replica",
    "shard",
    "cluster",
    "node",
    "pool",
    "lease",
    "lock",
    "signal",
    "backoff",
    "quota",
    "limit",
    "window",
]
_LABELS = ["bug", "feature", "docs", "performance", "question", "security", "ci"]
_AUTHORS = [f"dev{i:02d}" for i in range(40)]
_SERVICES = ["auth", "billing", "search", "gateway", "notifier", "scheduler", "storage"]
_LICENSES = ["MIT", "Apache-2.0", "BSD-3-Clause", "ISC", "MPL-2.0", "GPL-3.0"]


def _sentence(rng: random.Random, n: int = 12) -> str:
    words = [rng.choice(_WORDS) for _ in range(n)]
    return " ".join(words).capitalize() + "."


def _paragraph(rng: random.Random, sentences: int = 5) -> str:
    return " ".join(_sentence(rng, rng.randint(8, 16)) for _ in range(sentences))


# ── data ──────────────────────────────────────────────────────────────────


def issues(size: str, seed: int = 7) -> list[dict[str, Any]]:
    """GitHub-like issues; one body mentions a tokenizer segfault."""
    rng = random.Random(seed)
    count = max(50, SIZES[size] // 600)
    rows = []
    for number in range(1, count + 1):
        rows.append(
            {
                "number": number,
                "title": f"{rng.choice(_WORDS).capitalize()} {rng.choice(_WORDS)} "
                f"{rng.choice(['fails', 'is slow', 'leaks', 'hangs', 'breaks'])} "
                f"with {rng.choice(_WORDS)} #{number}",
                "state": rng.choice(["open", "open", "closed"]),
                "labels": rng.sample(_LABELS, rng.randint(0, 2)),
                "comments": rng.randint(0, 25),
                "author": rng.choice(_AUTHORS),
                "body": _paragraph(rng, 4),
            }
        )
    needle = rng.randrange(count // 3, count)
    rows[needle]["body"] += (
        " Reproduced a segfault in the tokenizer when the input ends with a backslash."
    )
    # One unambiguous "most commented open issue".
    top = rng.randrange(count)
    rows[top].update(state="open", comments=99)
    return rows


def docs(size: str, seed: int = 11) -> str:
    """A long markdown manual; one section states the default pool size."""
    rng = random.Random(seed)
    sections, total = [], 0
    target = SIZES[size]
    titles = [
        f"{a.capitalize()} {b}"
        for a in _WORDS
        for b in ("basics", "tuning", "internals")
    ]
    rng.shuffle(titles)
    i = 0
    while total < target:
        title = titles[i % len(titles)] + (
            f" ({i // len(titles) + 1})" if i >= len(titles) else ""
        )
        body = "\n\n".join(
            _paragraph(rng, rng.randint(3, 6)) for _ in range(rng.randint(2, 5))
        )
        sections.append(f"## {title}\n\n{body}\n")
        total += len(sections[-1])
        i += 1
    fact = (
        "## Connection pooling\n\nConnections are reused across requests. "
        "The default pool size is 37 connections; raise it with POOL_SIZE when "
        "workers wait for a free connection.\n"
    )
    sections.insert(rng.randrange(len(sections) // 2, len(sections)), fact)
    return "# Widget Server Manual\n\n" + "\n".join(sections)


def logs(size: str, seed: int = 13) -> str:
    """JSON Lines logs; one service has clearly the most ERROR entries."""
    rng = random.Random(seed)
    count = max(100, SIZES[size] // 140)
    worst = rng.choice(_SERVICES)
    lines = []
    for i in range(count):
        service = rng.choice(_SERVICES)
        error_rate = 0.12 if service == worst else 0.05
        level = (
            "ERROR"
            if rng.random() < error_rate
            else rng.choice(["INFO", "INFO", "WARN"])
        )
        lines.append(
            json.dumps(
                {
                    "ts": f"2026-10-01T{i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d}Z",
                    "service": service,
                    "level": level,
                    "latency_ms": rng.randint(1, 900),
                    "msg": _sentence(rng, 6),
                }
            )
        )
    return "\n".join(lines)


def dependencies(size: str, seed: int = 17) -> dict[str, dict[str, Any]]:
    """A lockfile-like map of package name → details."""
    rng = random.Random(seed)
    count = max(40, SIZES[size] // 160)
    return {
        f"{rng.choice(_WORDS)}-{rng.choice(_WORDS)}-{i}": {
            "version": f"{rng.randint(0, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}",
            "license": rng.choice(_LICENSES),
            "size_kb": rng.randint(5, 4000),
            "description": _sentence(rng, 10),
        }
        for i in range(count)
    }


# ── tasks ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Task:
    name: str
    question: str
    answer: str
    kind: str  # "int" or "text": how the answer is compared
    # Wrong answers of the same kind: a text answer naming one of these too
    # is not correct (it hedges between candidates).
    others: tuple[str, ...] = ()


def tasks(size: str) -> list[Task]:
    rows = issues(size)
    log_rows = [json.loads(line) for line in logs(size).splitlines()]
    errors: dict[str, int] = {}
    for row in log_rows:
        if row["level"] == "ERROR":
            errors[row["service"]] = errors.get(row["service"], 0) + 1
    deps = dependencies(size)
    needle = next(r["number"] for r in rows if "segfault in the tokenizer" in r["body"])
    top = max((r for r in rows if r["state"] == "open"), key=lambda r: r["comments"])
    return [
        Task(
            "issues_count",
            "In the repository acme/widgets, how many open issues have the label 'bug' "
            "and more than 10 comments?",
            str(
                sum(
                    1
                    for r in rows
                    if r["state"] == "open"
                    and "bug" in r["labels"]
                    and r["comments"] > 10
                )
            ),
            "int",
        ),
        Task(
            "issues_needle",
            "In the repository acme/widgets, which issue reports a segfault in the "
            "tokenizer? Answer with the issue number.",
            str(needle),
            "int",
        ),
        Task(
            "issues_top",
            "In the repository acme/widgets, what is the exact title of the open issue "
            "with the most comments?",
            top["title"],
            "text",
        ),
        Task(
            "docs_fact",
            "According to the Widget Server manual, what is the default connection pool "
            "size? Answer with the number.",
            "37",
            "int",
        ),
        Task(
            "logs_errors",
            "Looking at today's logs of all services, which service logged the most "
            "ERROR entries? Answer with the service name.",
            max(errors, key=errors.get),
            "text",
            tuple(s for s in _SERVICES if s != max(errors, key=errors.get)),
        ),
        Task(
            "deps_mit",
            "How many of the project's dependencies are MIT licensed?",
            str(sum(1 for d in deps.values() if d["license"] == "MIT")),
            "int",
        ),
    ]


def server(size: str) -> FastMCP:
    """The MCP server every condition talks to (the same data, the same tools)."""
    mcp = FastMCP("acme")
    rows, manual, log_text, deps = (
        issues(size),
        docs(size),
        logs(size),
        dependencies(size),
    )

    @mcp.tool
    def list_issues(repo: str, state: str = "all") -> list[dict]:
        """List the issues of a repository (state: open, closed or all)."""
        if repo != "acme/widgets":
            raise ValueError(f"unknown repository {repo!r}; try acme/widgets")
        return [r for r in rows if state in ("all", r["state"])]

    @mcp.tool
    def read_manual() -> str:
        """The Widget Server manual, as markdown."""
        return manual

    @mcp.tool
    def get_logs(date: str = "today") -> str:
        """Today's logs of all services, as JSON Lines."""
        return log_text

    @mcp.tool
    def list_dependencies() -> dict:
        """The project's dependencies: name → version, license, size."""
        return deps

    return mcp
