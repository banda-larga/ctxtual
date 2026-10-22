# Benchmark

Does ctxtual help real models answer questions about large tool results,
and at what cost? This harness measures it through
[OpenRouter](https://openrouter.ai), so any model can be compared.

## What it runs

An MCP server with four tools that return large, realistic results:

| Tool | Result |
|---|---|
| `list_issues(repo, state)` | GitHub-like issues (JSON list) |
| `read_manual()` | a long markdown manual |
| `get_logs()` | service logs (JSON Lines) |
| `list_dependencies()` | a lockfile-like map of name → details |

and six questions with one exact answer, computed from the same data:

| Task | Needs |
|---|---|
| `issues_count` | filtering and counting (open, label `bug`, > 10 comments) |
| `issues_needle` | finding one sentence in one issue body |
| `issues_top` | sorting (the open issue with the most comments) |
| `docs_fact` | finding a fact in the middle of the manual |
| `logs_errors` | grouping and counting log entries |
| `deps_mit` | counting entries of a keyed object |

Each task runs in three conditions, against the same server:

- **inline**: the result goes into the context whole, as most clients do.
- **truncated**: the result is cut at 40,000 characters (`--truncate`), as
  clients with an output limit do.
- **ctxtual**: `Offload()` with its defaults; the model gets a notice and the
  `result_*` tools.

Two sizes: **medium** (about 120,000 characters, ~30k tokens: fits in most
context windows) and **large** (about 1.2 million characters, ~300k tokens:
doesn't). The data is generated from fixed seeds, so every run sees the same
results.

The model gets a short system prompt (plus the server instructions, as
clients pass them), the question, and the tools, and must end with
`ANSWER: …`. A run is correct when that answer matches exactly (numbers
compared as numbers).

## Running it

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run --extra mcp python benchmarks/run.py \
    --models anthropic/claude-sonnet-4.5 openai/gpt-5-mini \
    --sizes medium large --repeats 3
```

Options: `--conditions`, `--tasks`, `--repeats`, `--truncate`,
`--max-turns` (15), `--concurrency` (4), `--out`. Each run is appended to a
JSON Lines file under `benchmarks/results/` as it finishes (answer, turns,
tool calls, prompt and completion tokens, cached tokens, cost in dollars as
OpenRouter reports it, the largest tool result, seconds, error), and a
summary table is written next to it.

Cost: one medium run of all tasks and conditions sends roughly 0.5–1 million
prompt tokens per model (most of it in the inline condition, which re-sends
the ~30k-token result on every turn); prompt caching lowers what that costs. The large
inline condition usually fails with a context-length error, which is
recorded as an error, not retried.

`--self-test` runs everything with a scripted model and no API calls; the
test suite runs it too.
