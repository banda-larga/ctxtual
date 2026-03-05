"""
Real-world example: Minimal integration — the 20-line pattern.

Most agent implementations need just this: store data, let the LLM
explore it via tools, dispatch tool calls. No decorators, no consumers,
no complex setup.

This is the starting point. Copy this pattern and customize.

Run:
    uv run python examples/01_quickstart.py
"""

from ctxtual import Forge, MemoryStore
from ctxtual.utils import paginator, text_search

# 1. Create forge
forge = Forge(store=MemoryStore())

# 2. Register a producer (toolsets get their name from workspace_type automatically)
@forge.producer(workspace_type="items", toolsets=[paginator(forge), text_search(forge)])
def load_data(source: str) -> list[dict]:
    """Load data from a source.

    Args:
        source: Data source identifier.
    """
    # Replace with your actual data loading logic
    return [
        {"name": "Alice", "role": "engineer", "team": "platform"},
        {"name": "Bob", "role": "designer", "team": "product"},
        {"name": "Carol", "role": "engineer", "team": "infra"},
        {"name": "Dave", "role": "manager", "team": "platform"},
    ]


# 3. Your agent loop
def agent_loop(user_message: str):
    """
    Minimal agent loop pattern.

    In production, replace the hardcoded steps with actual LLM calls.
    The structure stays the same:
      response = llm.chat(messages, tools=forge.get_tools())
      for tool_call in response.tool_calls:
          result = forge.dispatch_tool_call(tool_call.name, tool_call.args)
    """
    # LLM decides to call the producer
    ref = load_data(source="hr_database")
    print(f"Loaded {ref['item_count']} items (ws: {ref['workspace_id']})")

    # LLM gets tool schemas (OpenAI function-calling format)
    tools = forge.get_tools()
    print(f"Tools available: {[t['function']['name'] for t in tools]}")

    # LLM calls a tool — dispatch returns the result (or error dict)
    ws_id = ref["workspace_id"]
    result = forge.dispatch_tool_call(
        "items_search",
        {"workspace_id": ws_id, "query": "engineer"},
    )
    print(f"Search 'engineer': {result['total_matches']} matches")

    # LLM calls another tool
    result = forge.dispatch_tool_call(
        "items_paginate",
        {"workspace_id": ws_id, "page": 0, "size": 2},
    )
    print(f"Page 0: {len(result['result']['items'])} items")

    # If LLM makes a mistake, it gets a helpful error — not an exception
    result = forge.dispatch_tool_call("nonexistent_tool", {})
    print(f"Bad call → error: {result['error'][:50]}...")


if __name__ == "__main__":
    agent_loop("Find all engineers on the team")
