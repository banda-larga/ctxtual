"""
Example 11 — Declarative Data Pipelines
========================================

Demonstrates ctx's pipeline utility: compound data operations
in a single tool call.  This is the ctx equivalent of
"programmatic tool calling" — instead of the LLM making 4+ round-trips
(search → filter → sort → limit), it describes the entire chain once.

Intermediate data stays server-side.  Only the final result enters context.

Run:
    uv run python examples/05_pipelines.py
"""

from ctxtual import Forge, MemoryStore
from ctxtual.utils import paginator, pipeline

# ── Setup ─────────────────────────────────────────────────────────────────

forge = Forge(store=MemoryStore())
pager = paginator(forge, "papers")
pipe = pipeline(forge, "papers")   # adds pipe + aggregate tools to same ToolSet

# Sample academic paper dataset
PAPERS = [
    {"id": i, "title": t, "author": a, "year": y, "citations": c, "field": f, "tags": tags}
    for i, (t, a, y, c, f, tags) in enumerate([
        ("Deep Learning Basics", "Alice Chen", 2021, 150, "ML", ["deep-learning", "tutorial"]),
        ("Transformer Architecture", "Bob Smith", 2022, 300, "NLP", ["transformers", "attention"]),
        ("Quantum Computing 101", "Carol Davis", 2023, 50, "Quantum", ["intro", "quantum"]),
        ("ML in Production", "Alice Chen", 2023, 200, "ML", ["mlops", "deployment"]),
        ("Reinforcement Learning", "Bob Smith", 2021, 120, "ML", ["rl", "policy-gradient"]),
        ("Graph Neural Networks", "Dave Wilson", 2022, 80, "ML", ["gnn", "graph-theory"]),
        ("NLP with Transformers", "Eve Brown", 2023, 250, "NLP", ["transformers", "nlp"]),
        ("Computer Vision Survey", "Alice Chen", 2022, 180, "CV", ["survey", "cnn"]),
        ("Quantum ML Intersection", "Carol Davis", 2023, 30, "Quantum", ["quantum", "ml-theory"]),
        ("Ethics in AI", "Frank Lee", 2021, 90, "Ethics", ["fairness", "bias"]),
        ("Large Language Models", "Grace Kim", 2023, 400, "NLP", ["llm", "scaling"]),
        ("Federated Learning", "Henry Park", 2022, 110, "ML", ["privacy", "distributed"]),
    ], start=1)
]


@forge.producer(workspace_type="papers", toolsets=[pager, pipe])
def load_papers():
    """Load the full paper database."""
    return PAPERS


# ── Load data ─────────────────────────────────────────────────────────────

ref = load_papers()
wid = ref["workspace_id"]
print(f"Loaded {ref['item_count']} papers → workspace '{wid}'")
print(f"Available tools: {ref['available_tools']}")
print()


# ── Pattern 1: Top-N (replaces sort → paginate — 2 calls → 1) ────────────

print("═══ Pattern 1: Top 5 most-cited papers ═══")
result = forge.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [
        {"sort": {"field": "citations", "order": "desc"}},
        {"limit": 5},
        {"select": ["title", "author", "citations"]},
    ],
})
for item in result["items"]:
    print(f"  {item['citations']:>4} citations — {item['title']} ({item['author']})")
print()


# ── Pattern 2: Filtered search (replaces search → filter — 2+ calls → 1) ─

print("═══ Pattern 2: ML papers from 2022+ sorted by citations ═══")
result = forge.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [
        {"filter": {"field": "ML", "year": {"$gte": 2022}}},
        {"sort": {"field": "citations", "order": "desc"}},
        {"select": ["title", "year", "citations"]},
    ],
})
print(f"  Found {result['count']} papers:")
for item in result["items"]:
    print(f"  [{item['year']}] {item['title']} — {item['citations']} cites")
print()


# ── Pattern 3: Aggregation (replaces manual iteration — N calls → 1) ─────

print("═══ Pattern 3: Statistics by research field ═══")
result = forge.dispatch_tool_call("papers_aggregate", {
    "workspace_id": wid,
    "group_by": "field",
    "metrics": {
        "papers": "count",
        "total_citations": "sum:citations",
        "avg_citations": "mean:citations",
        "top_cited": "max:citations",
    },
})
for group in sorted(result["groups"], key=lambda g: g["total_citations"], reverse=True):
    print(f"  {group['field']:>8}: {group['papers']} papers, "
          f"{group['total_citations']} total cites, "
          f"avg {group['avg_citations']:.0f}, "
          f"top {group['top_cited']}")
print()


# ── Pattern 4: Tag analysis with flatten (replaces manual expansion) ──────

print("═══ Pattern 4: Most common tags (flatten → group → sort) ═══")
result = forge.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [
        {"flatten": "tags"},
        {"group_by": {"field": "tags", "metrics": {"count": "count"}}},
        {"sort": {"field": "count", "order": "desc"}},
        {"limit": 8},
    ],
})
for item in result["items"]:
    bar = "█" * item["count"]
    print(f"  {item['tags']:>16}: {bar} ({item['count']})")
print()


# ── Pattern 5: Save pipeline result as new workspace ─────────────────────

print("═══ Pattern 5: Save filtered results as new workspace ═══")
result = forge.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [
        {"filter": {"year": 2023}},
        {"sort": {"field": "citations", "order": "desc"}},
    ],
    "save_as": "papers_2023",
})
print(f"  Saved: {result['item_count']} papers → workspace '{result['workspace_id']}'")

# Now paginate the derived workspace
page = forge.dispatch_tool_call("papers_paginate", {
    "workspace_id": "papers_2023",
    "page": 0,
    "size": 3,
})
print(f"  First page: {[p['title'] for p in page['result']['items']]}")
print()


# ── Pattern 6: Complex compound query ────────────────────────────────────

print("═══ Pattern 6: Authors with 2+ papers and avg citations > 100 ═══")
result = forge.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [
        {"group_by": {
            "field": "author",
            "metrics": {
                "papers": "count",
                "avg_cites": "mean:citations",
                "fields": "values:field",
            },
        }},
        {"filter": {"papers": {"$gte": 2}, "avg_cites": {"$gt": 100}}},
        {"sort": {"field": "avg_cites", "order": "desc"}},
    ],
})
for item in result["items"]:
    print(f"  {item['author']}: {item['papers']} papers, "
          f"avg {item['avg_cites']:.0f} cites, "
          f"fields: {item['fields']}")
print()


# ── Pattern 7: Error handling ─────────────────────────────────────────────

print("═══ Pattern 7: LLM-friendly error on bad pipeline ═══")
result = forge.dispatch_tool_call("papers_pipe", {
    "workspace_id": wid,
    "steps": [{"nonexistent_op": True}],
})
print(f"  Error: {result['error']}")
print(f"  Available ops: {result['available_operations']}")
print()


# ── Summary ───────────────────────────────────────────────────────────────

print("═══ Summary ═══")
print("Without pipeline: Patterns 1-6 would each require 2-5 tool calls")
print("With pipeline:    Each pattern is ONE tool call, zero intermediate context")
schemas = forge.get_all_tool_schemas()
names = [s["function"]["name"] for s in schemas]
print(f"Registered tools: {names}")
