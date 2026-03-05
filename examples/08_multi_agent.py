"""
Real-world example: Multi-agent collaboration with shared workspaces.

Scenario: A research team of specialized agents working together:
  1. **Collector agent** — gathers data from multiple sources
  2. **Analyst agent** — processes and enriches the data
  3. **Writer agent** — produces a final report

Each agent has its own Forge consumer but they share the same store,
reading each other's workspaces. This is the pattern for CrewAI-style
multi-agent systems where agents have different roles.

Run:
    uv run python examples/08_multi_agent.py
"""

from ctxtual import Forge, MemoryStore, ConsumerContext
from ctxtual.utils import paginator, text_search, kv_reader

# ── Shared infrastructure ────────────────────────────────────────────────

# All agents share one store — this is the collaboration backbone.
# In production, use SQLiteStore for persistence across processes.
store = MemoryStore()
forge = Forge(store=store)

# ── Agent 1: Data Collector ──────────────────────────────────────────────

collector_pager  = paginator(forge, "raw_data")
collector_search = text_search(forge, "raw_data", fields=["source", "title", "content"])


@forge.producer(
    workspace_type="raw_data",
    toolsets=[collector_pager, collector_search],
    key="research_{topic}",
    meta={"agent": "collector"},
)
def collect_research(topic: str) -> list[dict]:
    """Collector agent: gather data from multiple sources.

    Args:
        topic: Research topic to collect data about.
    """
    # In production: call APIs, scrape websites, query databases
    return [
        {
            "source": "arxiv",
            "title": "Scaling Laws for Neural Language Models",
            "content": "We study empirical scaling laws for language model performance. "
                       "Cross-entropy loss scales as a power-law with model size, "
                       "dataset size, and compute budget.",
            "year": 2020,
            "relevance": 0.95,
        },
        {
            "source": "blog",
            "title": "The Bitter Lesson by Rich Sutton",
            "content": "The biggest lesson from 70 years of AI research is that "
                       "general methods that leverage computation are ultimately "
                       "the most effective by a large margin.",
            "year": 2019,
            "relevance": 0.88,
        },
        {
            "source": "arxiv",
            "title": "Chinchilla: Training Compute-Optimal LLMs",
            "content": "Current large language models are significantly undertrained. "
                       "Given a compute budget, the model size and training tokens "
                       "should be scaled equally.",
            "year": 2022,
            "relevance": 0.92,
        },
        {
            "source": "industry",
            "title": "GPT-4 Technical Report",
            "content": "GPT-4 is a large multimodal model that exhibits human-level "
                       "performance on various professional benchmarks.",
            "year": 2023,
            "relevance": 0.85,
        },
        {
            "source": "arxiv",
            "title": "Emergent Abilities of Large Language Models",
            "content": "We discuss emergent abilities — abilities that are not present "
                       "in smaller models but are present in larger models. This is "
                       "a fundamental property of scaling.",
            "year": 2022,
            "relevance": 0.90,
        },
    ]


# ── Agent 2: Analyst ─────────────────────────────────────────────────────

analysis_pager = paginator(forge, "analysis")


@forge.consumer(
    workspace_type="raw_data",
    produces="analysis",
    produces_toolsets=[analysis_pager],
)
def analyze_sources(
    workspace_id: str,
    min_relevance: float = 0.85,
    forge_ctx: ConsumerContext = None,
) -> dict:
    """Analyst agent: filter, enrich, and score the collected data.

    Args:
        workspace_id: Raw data workspace to analyze.
        min_relevance: Minimum relevance score to include.
    """
    raw_items = forge_ctx.get_items()

    # Filter by relevance
    relevant = [item for item in raw_items if item.get("relevance", 0) >= min_relevance]

    # Enrich: add analysis metadata
    for i, item in enumerate(relevant):
        item["analysis_rank"] = i + 1
        item["key_finding"] = item["content"].split(".")[0] + "."
        item["word_count"] = len(item["content"].split())

    # Sort by relevance
    relevant.sort(key=lambda x: x["relevance"], reverse=True)

    return forge_ctx.emit(
        relevant,
        meta={
            "agent": "analyst",
            "derived_from": workspace_id,
            "original_count": len(raw_items),
            "filtered_count": len(relevant),
            "min_relevance": min_relevance,
        },
    )


# ── Agent 3: Report Writer ──────────────────────────────────────────────

report_kv = kv_reader(forge, "report")


@forge.consumer(
    workspace_type="analysis",
    produces="report",
    produces_toolsets=[report_kv],
)
def write_report(
    workspace_id: str,
    forge_ctx: ConsumerContext = None,
) -> dict:
    """Writer agent: produce a structured report from analyzed data.

    Args:
        workspace_id: Analysis workspace to summarize.
    """
    items = forge_ctx.get_items()

    # Build report sections
    report = {
        "title": "Research Report: AI Scaling Laws",
        "summary": (
            f"Analysis of {len(items)} key sources on AI scaling. "
            f"Sources span {min(i['year'] for i in items)}-{max(i['year'] for i in items)}."
        ),
        "key_findings": [item["key_finding"] for item in items],
        "sources_by_origin": {},
        "timeline": [],
        "methodology": (
            f"Collected from multiple sources, filtered by relevance "
            f"(threshold: {items[0].get('relevance', 'N/A')}+), "
            f"ranked and enriched by the analyst agent."
        ),
    }

    # Group by source
    for item in items:
        src = item["source"]
        if src not in report["sources_by_origin"]:
            report["sources_by_origin"][src] = []
        report["sources_by_origin"][src].append(item["title"])

    # Build timeline
    for item in sorted(items, key=lambda x: x["year"]):
        report["timeline"].append({
            "year": item["year"],
            "title": item["title"],
            "key_finding": item["key_finding"],
        })

    return forge_ctx.emit(
        report,
        meta={"agent": "writer", "derived_from": workspace_id},
    )


# ── Orchestration ────────────────────────────────────────────────────────

def run_multi_agent():
    print("=" * 70)
    print("MULTI-AGENT PIPELINE")
    print("  Collector → Analyst → Writer")
    print("=" * 70)

    # Agent 1: Collect
    print("\n🔍 [Collector] Gathering research data...")
    raw_ref = collect_research(topic="scaling_laws")
    print(f"  → {raw_ref['item_count']} sources collected")
    print(f"  → Workspace: {raw_ref['workspace_id']}")

    # Agent 2: Analyze
    print("\n📊 [Analyst] Analyzing and filtering sources...")
    analysis_ref = analyze_sources(
        workspace_id=raw_ref["workspace_id"],
        min_relevance=0.88,
    )
    print(f"  → {analysis_ref['item_count']} sources passed analysis filter")
    print(f"  → Workspace: {analysis_ref['workspace_id']}")

    # Browse the analysis workspace
    ws_analysis = analysis_ref["workspace_id"]
    result = forge.dispatch_tool_call(
        "analysis_paginate", {"workspace_id": ws_analysis, "page": 0, "size": 10}
    )
    data = result["result"]
    print(f"\n  Analyzed sources (ranked):")
    for item in data["items"]:
        print(f"    #{item['analysis_rank']} [{item['source']}] "
              f"{item['title']} (relevance: {item['relevance']})")

    # Agent 3: Write report
    print("\n📝 [Writer] Generating report...")
    report_ref = write_report(workspace_id=ws_analysis)
    print(f"  → Report stored in: {report_ref['workspace_id']}")
    ws_report = report_ref["workspace_id"]

    # Read the report (dict workspace → kv_reader)
    keys = forge.dispatch_tool_call(
        "report_get_keys", {"workspace_id": ws_report}
    )
    print(f"\n  Report sections: {keys['result']}")

    title = forge.dispatch_tool_call(
        "report_get_value", {"workspace_id": ws_report, "key": "title"}
    )
    print(f"\n  📄 {title}")

    summary = forge.dispatch_tool_call(
        "report_get_value", {"workspace_id": ws_report, "key": "summary"}
    )
    print(f"  {summary}")

    findings = forge.dispatch_tool_call(
        "report_get_value", {"workspace_id": ws_report, "key": "key_findings"}
    )
    print(f"\n  Key Findings:")
    for f in findings:
        print(f"    • {f}")

    timeline = forge.dispatch_tool_call(
        "report_get_value", {"workspace_id": ws_report, "key": "timeline"}
    )
    print(f"\n  Timeline:")
    for entry in timeline:
        print(f"    {entry['year']}: {entry['title']}")

    # Show the workspace lineage
    print(f"\n{'=' * 70}")
    print("WORKSPACE LINEAGE (all agents share one store):")
    for ws_id in forge.list_workspaces():
        meta = forge.workspace_meta(ws_id)
        agent = meta.extra.get("agent", "?")
        derived = meta.extra.get("derived_from", "—")
        print(f"  [{agent:>10}] {ws_id}")
        print(f"             type={meta.workspace_type}, shape={meta.data_shape}, "
              f"items={meta.item_count}, from={derived}")
    print("=" * 70)


if __name__ == "__main__":
    run_multi_agent()
