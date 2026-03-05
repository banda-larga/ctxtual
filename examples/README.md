# ctxtual Examples

Runnable examples organized from beginner to advanced. Each script is self-contained — no API keys or external services needed (LLM integrations fall back to simulation mode).

```bash
# Run any example
uv run python examples/01_quickstart.py
```

## Core Concepts

| # | Example | What it demonstrates |
|---|---------|---------------------|
| 01 | [Quickstart](01_quickstart.py) | Minimal 20-line pattern: producer → dispatch → explore |
| 02 | [RAG Support Agent](02_rag_support_agent.py) | Knowledge base ingestion with search, filter, and pagination |
| 03 | [Data Pipeline](03_data_pipeline.py) | Derived workspaces: orders → VIP filter → region stats |
| 04 | [Custom Tools](04_custom_tools.py) | Domain-specific toolsets (financial analytics, anomaly detection) |
| 05 | [Pipelines](05_pipelines.py) | Declarative compound operations in a single tool call |

## Production Patterns

| # | Example | What it demonstrates |
|---|---------|---------------------|
| 06 | [Persistence](06_persistence.py) | SQLite store, workspace mutations, survive process restarts |
| 07 | [Error Handling](07_error_handling.py) | Structured LLM-friendly errors for every failure mode |
| 08 | [Multi-Agent](08_multi_agent.py) | Shared store, workspace lineage across specialized agents |

## LLM Integrations

| # | Example | What it demonstrates |
|---|---------|---------------------|
| 09 | [OpenAI Agent](09_openai_agent.py) | Full agent loop with `openai` SDK (simulated without API key) |
| 10 | [Anthropic Agent](10_anthropic_agent.py) | Full agent loop with `anthropic` SDK (simulated without API key) |
| 11 | [Concurrent Server](11_concurrent_server.py) | FastAPI + thread-safe concurrent sessions |

## Notebooks

Interactive Jupyter notebooks in [`notebooks/`](notebooks/) cover the same concepts with inline visualizations.
