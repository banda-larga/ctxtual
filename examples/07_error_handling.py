"""
Real-world example: Error handling and graceful degradation.

Scenario: Demonstrates how ctx handles common failure modes
that LLM agents encounter — wrong workspace IDs, expired data, type
mismatches, out-of-range indices — and returns structured, LLM-friendly
error messages instead of crashing.

This is critical for production: LLMs WILL make mistakes, and your
agent loop must not crash when they do.

Run:
    uv run python examples/07_error_handling.py
"""

from ctxtual import Ctx, MemoryStore
from ctxtual.utils import kv_reader, paginator, text_search

# Setup

ctx = Ctx(
    store=MemoryStore(),
    default_ttl=2,  # Short TTL to demonstrate expiration
)

# List-type workspace
list_pager = paginator(ctx, "articles")
list_search = text_search(ctx, "articles")

# Dict-type workspace
dict_kv = kv_reader(ctx, "config")


@ctx.producer(
    workspace_type="articles",
    toolsets=[list_pager, list_search],
    key="articles_demo",
)
def load_articles() -> list[dict]:
    return [
        {"title": "First Article", "body": "Content of first"},
        {"title": "Second Article", "body": "Content of second"},
        {"title": "Third Article", "body": "Content of third"},
    ]


@ctx.producer(
    workspace_type="config",
    toolsets=[dict_kv],
    key="config_demo",
    ttl=None,  # Override: config never expires
)
def load_config() -> dict:
    return {"database_url": "postgres://...", "debug": False, "max_retries": 3}


# Demonstrate error scenarios


def demo_errors():
    print("=" * 70)
    print("ERROR HANDLING: How ctx helps LLMs recover from mistakes")
    print("=" * 70)

    # Load data
    articles_ref = load_articles()
    config_ref = load_config()
    print(f"\nLoaded articles: {articles_ref['workspace_id']}")
    print(f"Loaded config: {config_ref['workspace_id']}")

    # Error 1: Unknown tool name
    print(f"\n{'─' * 50}")
    print("ERROR 1: LLM calls a tool that doesn't exist")
    print("─" * 50)

    result = ctx.dispatch_tool_call("search_papers", {"query": "AI"})
    print(f"  error: {result['error']}")
    print(f"  available_tools: {result['available_tools'][:3]}...")
    print(f"  suggested_action: {result['suggested_action']}")
    print(f"\n  → The LLM sees available tools and can self-correct.")

    # Error 2: Wrong workspace ID
    print(f"\n{'─' * 50}")
    print("ERROR 2: LLM uses a workspace_id that doesn't exist")
    print("─" * 50)

    result = ctx.dispatch_tool_call(
        "articles_paginate",
        {"workspace_id": "articles_WRONG_ID", "page": 0},
    )
    print(f"  error: {result['error']}")
    if "available_workspaces" in result:
        print(f"  available_workspaces: {result['available_workspaces']}")
    print(f"  suggested_action: {result['suggested_action']}")
    print(f"\n  → The LLM sees valid workspace IDs and can retry.")

    # Error 3: Index out of range
    print(f"\n{'─' * 50}")
    print("ERROR 3: LLM requests an item at an invalid index")
    print("─" * 50)

    result = ctx.dispatch_tool_call(
        "articles_get_item",
        {"workspace_id": articles_ref["workspace_id"], "index": 999},
    )
    print(f"  error: {result['error']}")
    print(f"  valid_range: {result['valid_range']}")
    print(f"  total_items: {result['total_items']}")
    print(f"  suggested_action: {result['suggested_action']}")
    print(f"\n  → The LLM knows the valid range and can pick a valid index.")

    # Error 4: Wrong key in dict workspace
    print(f"\n{'─' * 50}")
    print("ERROR 4: LLM asks for a key that doesn't exist in dict workspace")
    print("─" * 50)

    result = ctx.dispatch_tool_call(
        "config_get_value",
        {"workspace_id": config_ref["workspace_id"], "key": "api_key"},
    )
    print(f"  error: {result['error']}")
    print(f"  available_keys: {result['available_keys']}")
    print(f"  suggested_action: {result['suggested_action']}")
    print(f"\n  → The LLM sees all valid keys and can pick the right one.")

    # Error 5: Data shape mismatch
    print(f"\n{'─' * 50}")
    print("ERROR 5: LLM uses list-tools on a dict workspace (shape mismatch)")
    print("─" * 50)

    result = ctx.dispatch_tool_call(
        "articles_paginate",
        {"workspace_id": config_ref["workspace_id"], "page": 0},
    )
    # This triggers workspace type mismatch (config != articles)
    print(f"  error: {result['error']}")
    print(f"  suggested_action: {result['suggested_action']}")
    print(f"\n  → The LLM learns it's using the wrong tool type.")

    # Error 6: Expired workspace
    print(f"\n{'─' * 50}")
    print("ERROR 6: LLM accesses a workspace after its TTL expired")
    print("─" * 50)

    import time

    print(f"  Waiting for TTL to expire (2 seconds)...")
    time.sleep(2.5)

    result = ctx.dispatch_tool_call(
        "articles_paginate",
        {"workspace_id": articles_ref["workspace_id"], "page": 0},
    )
    print(f"  error: {result['error']}")
    print(f"  suggested_action: {result['suggested_action']}")
    print(f"\n  → The LLM knows to call the producer again to refresh data.")

    # Summary
    print(f"\n{'=' * 70}")
    print("KEY PRINCIPLE: Every error returns a structured dict with:")
    print("  1. 'error' — what went wrong (human + LLM readable)")
    print("  2. 'suggested_action' — what the LLM should do next")
    print("  3. Context — available tools/workspaces/keys/ranges")
    print()
    print("This means your agent loop NEVER needs try/except for tool calls.")
    print("dispatch_tool_call() always returns a dict — success or error.")
    print("=" * 70)


if __name__ == "__main__":
    demo_errors()
