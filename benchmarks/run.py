"""
Benchmark ctxtual with real models through OpenRouter.

Every task runs through a real MCP server in three conditions:

- ``inline``     the tool result goes into the context whole
- ``truncated``  the result is cut at --truncate characters, as clients do
- ``ctxtual``    large results are stored (``Offload`` defaults) and the
                 model gets a notice plus the result_* tools

and is scored on the exact answer, tokens, cost, tool calls and time.

    export OPENROUTER_API_KEY=...
    uv run --extra mcp python benchmarks/run.py \\
        --models anthropic/claude-sonnet-4.5 openai/gpt-5-mini --sizes medium large

    uv run --extra mcp python benchmarks/run.py --self-test   # no API calls
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import Client, FastMCP
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult
from mcp_types import TextContent

sys.path.insert(0, str(Path(__file__).parent))
from tasks import SIZES, Task, server, tasks  # noqa: E402

from ctxtual.mcp import Offload  # noqa: E402

CONDITIONS = ("inline", "truncated", "ctxtual")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = (
    "You are an agent with tools. Use them to answer the user's question. "
    "When you are done, end your reply with a line 'ANSWER: <answer>' and nothing after it."
)


# ── conditions ────────────────────────────────────────────────────────────


class Truncate(Middleware):
    """Cut text results at *limit* characters, the way many clients do."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    async def on_call_tool(self, context: Any, call_next: Any) -> Any:
        result = await call_next(context)
        if not isinstance(result, ToolResult):
            return result
        text = "".join(getattr(b, "text", "") for b in result.content or [])
        if len(text) <= self.limit:
            return result
        cut = f"{text[:self.limit]}\n… [output truncated: {len(text) - self.limit:,} more characters]"
        return ToolResult(content=[TextContent(type="text", text=cut)], is_error=result.is_error)


def build(condition: str, size: str, truncate: int) -> FastMCP:
    mcp = server(size)
    if condition == "truncated":
        mcp.add_middleware(Truncate(truncate))
    elif condition == "ctxtual":
        mcp.add_middleware(Offload())
    return mcp


# ── models ────────────────────────────────────────────────────────────────


class ModelError(Exception):
    pass


class OpenRouter:
    def __init__(self, model: str, key: str, timeout: float = 300.0) -> None:
        self.model, self.key, self.timeout = model, key, timeout

    async def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "tools": tools,
                "usage": {"include": True}}
        return await asyncio.to_thread(self._post, body)

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(
            OPENROUTER_URL, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "X-Title": "ctxtual benchmark"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                if exc.code in (429, 502, 503) and attempt < 3:
                    time.sleep(5 * 2 ** attempt)
                    continue
                raise ModelError(f"HTTP {exc.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < 3:
                    time.sleep(5 * 2 ** attempt)
                    continue
                raise ModelError(str(exc)) from None
            if "error" in data:
                raise ModelError(json.dumps(data["error"])[:500])
            return data
        raise ModelError("no response")


class ScriptedModel:
    """For --self-test: exercises the loop and the tools without an API."""

    def __init__(self, task: Task) -> None:
        self.task, self.step = task, 0

    async def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        names = {t["function"]["name"] for t in tools}
        source = {"issues": ("list_issues", {"repo": "acme/widgets"}),
                  "docs": ("read_manual", {}), "logs": ("get_logs", {}),
                  "deps": ("list_dependencies", {})}[self.task.name.split("_")[0]]
        self.step += 1
        calls = []
        if self.step == 1:
            calls = [source]
        elif self.step == 2 and "result_list" in names:
            calls = [("result_list", {})]
        if calls:
            return _reply(None, [(f"call_{self.step}_{i}", n, a) for i, (n, a) in enumerate(calls)])
        return _reply(f"Done.\nANSWER: {self.task.answer}", [])


def _reply(content: str | None, calls: list[tuple[str, str, dict]]) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for cid, n, a in calls]
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


# ── one run ───────────────────────────────────────────────────────────────


@dataclass
class Run:
    model: str
    condition: str
    size: str
    task: str
    correct: bool = False
    answer: str | None = None
    expected: str = ""
    turns: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    largest_tool_result: int = 0
    seconds: float = 0.0
    error: str | None = None
    tools_used: list[str] = field(default_factory=list)


async def run_one(model: Any, model_name: str, condition: str, size: str, task: Task,
                  truncate: int, max_turns: int) -> Run:
    run = Run(model_name, condition, size, task.name, expected=task.answer)
    started = time.perf_counter()
    async with Client(build(condition, size, truncate)) as client:
        tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description or "",
            "parameters": t.input_schema}} for t in await client.list_tools()]
        instructions = getattr(client.initialize_result, "instructions", None) if \
            client.initialize_result else None
        system = SYSTEM + (f"\n\nServer instructions:\n{instructions}" if instructions else "")
        messages: list[dict] = [{"role": "system", "content": system},
                                {"role": "user", "content": task.question}]
        try:
            for _ in range(max_turns):
                data = await model.complete(messages, tools)
                run.turns += 1
                usage = data.get("usage") or {}
                run.prompt_tokens += usage.get("prompt_tokens") or 0
                run.completion_tokens += usage.get("completion_tokens") or 0
                run.cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                run.cost += float(usage.get("cost") or 0)
                message = data["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                messages.append({k: v for k, v in message.items()
                                 if k in ("role", "content", "tool_calls", "reasoning_details")})
                if not calls:
                    run.answer = _extract_answer(message.get("content") or "")
                    run.correct = _grade(run.answer, task)
                    break
                for call in calls:
                    name = call["function"]["name"]
                    run.tool_calls += 1
                    run.tools_used.append(name)
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    try:
                        result = await client.call_tool(name, args, raise_on_error=False)
                        text = "".join(getattr(b, "text", "") for b in result.content)
                    except Exception as exc:  # unknown tool, bad arguments
                        text = f"Error: {exc}"
                    run.largest_tool_result = max(run.largest_tool_result, len(text))
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            else:
                run.error = f"no answer after {max_turns} turns"
        except ModelError as exc:
            run.error = str(exc)
    run.seconds = round(time.perf_counter() - started, 1)
    return run


def _extract_answer(content: str) -> str | None:
    found = re.findall(r"ANSWER:\s*(.+)", content)
    return found[-1].strip() if found else None


def _grade(answer: str | None, task: Task) -> bool:
    if answer is None:
        return False
    clean = answer.strip().strip("*`\"'. ").lower()
    if task.kind == "int":
        numbers = re.findall(r"-?\d+", clean.replace(",", ""))
        return bool(numbers) and numbers[0].lstrip("#") == task.answer
    return clean == task.answer.lower() or clean.strip("#") == task.answer.lower()


# ── the whole benchmark ───────────────────────────────────────────────────


async def benchmark(args: argparse.Namespace) -> list[Run]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not args.self_test and not key:
        sys.exit("Set OPENROUTER_API_KEY (or use --self-test).")
    limit = asyncio.Semaphore(args.concurrency)
    jobs = []
    for model_name in (["scripted"] if args.self_test else args.models):
        for size in args.sizes:
            for task in tasks(size):
                if args.tasks and task.name not in args.tasks:
                    continue
                for condition in args.conditions:
                    for repeat in range(args.repeats):
                        jobs.append((model_name, size, task, condition, repeat))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    async def one(model_name: str, size: str, task: Task, condition: str, repeat: int) -> Run:
        async with limit:
            model = ScriptedModel(task) if args.self_test else OpenRouter(model_name, key)
            run = await run_one(model, model_name, condition, size, task, args.truncate,
                                args.max_turns)
            with out.open("a") as f:
                f.write(json.dumps(asdict(run)) + "\n")
            mark = "ok " if run.correct else ("ERR" if run.error else "no ")
            print(f"{mark} {model_name:40} {condition:9} {size:6} {task.name:14} "
                  f"answer={run.answer!r} expected={task.answer!r} "
                  f"tokens={run.prompt_tokens:,} calls={run.tool_calls}"
                  + (f" error={run.error[:80]}" if run.error else ""), flush=True)
            return run

    return await asyncio.gather(*(one(*job) for job in jobs))


def report(runs: list[Run]) -> str:
    groups: dict[tuple[str, str, str], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.model, run.size, run.condition), []).append(run)
    lines = ["| model | size | condition | correct | errors | prompt tokens (mean) "
             "| cost (total $) | tool calls (mean) | seconds (mean) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (model, size, condition), rs in sorted(groups.items()):
        n = len(rs)
        lines.append(
            f"| {model} | {size} | {condition} | {sum(r.correct for r in rs)}/{n} "
            f"| {sum(bool(r.error) for r in rs)} | {sum(r.prompt_tokens for r in rs) // n:,} "
            f"| {sum(r.cost for r in rs):.4f} | {sum(r.tool_calls for r in rs) / n:.1f} "
            f"| {sum(r.seconds for r in rs) / n:.1f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=[], help="OpenRouter model ids")
    parser.add_argument("--sizes", nargs="+", choices=sorted(SIZES), default=["medium"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--tasks", nargs="+", help="only these tasks (default: all)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--truncate", type=int, default=40_000,
                        help="characters kept in the truncated condition (default 40000)")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default=f"benchmarks/results/{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    parser.add_argument("--self-test", action="store_true",
                        help="run with a scripted model, no API calls")
    args = parser.parse_args(argv)
    if not args.self_test and not args.models:
        parser.error("--models is required (or --self-test)")
    runs = asyncio.run(benchmark(args))
    table = report(runs)
    print("\n" + table)
    Path(args.out).with_suffix(".md").write_text(table + "\n")
    print(f"\nRuns: {args.out}\nTable: {Path(args.out).with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
