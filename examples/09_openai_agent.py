"""
Real-world example: OpenAI agent with tool calling.

Scenario: A research assistant that searches academic papers, explores
results via tools, and synthesizes findings. Uses the OpenAI integration
adapter for zero-glue tool dispatch.

This demonstrates the RECOMMENDED pattern for building an agent loop
with ctx + OpenAI's chat completions API.

Prerequisites:
    pip install openai
    export OPENAI_API_KEY=sk-...

Run:
    uv run python examples/09_openai_agent.py

Note: If no API key is set, the example runs in simulation mode.
"""

import json
import os

from ctxtual import Forge, MemoryStore
from ctxtual.utils import paginator, text_search

# ── Setup ────────────────────────────────────────────────────────────────

forge = Forge(store=MemoryStore())

pager  = paginator(forge, "papers")
search = text_search(forge, "papers", fields=["title", "abstract"])


@forge.producer(
    workspace_type="papers",
    toolsets=[pager, search],
    key="arxiv_{query}",
)
def search_arxiv(query: str) -> list[dict]:
    """Search for academic papers. Returns structured results.

    Args:
        query: Search query for academic papers.
    """
    # In production: call arxiv API, Semantic Scholar, etc.
    return [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "abstract": "We propose a new simple network architecture, the Transformer, "
                        "based solely on attention mechanisms, dispensing with recurrence "
                        "and convolutions entirely.",
            "year": 2017,
            "citations": 95000,
            "url": "https://arxiv.org/abs/1706.03762",
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin et al."],
            "abstract": "We introduce BERT, a new language representation model which "
                        "stands for Bidirectional Encoder Representations from Transformers. "
                        "BERT is designed to pre-train deep bidirectional representations.",
            "year": 2018,
            "citations": 78000,
            "url": "https://arxiv.org/abs/1810.04805",
        },
        {
            "title": "GPT-4 Technical Report",
            "authors": ["OpenAI"],
            "abstract": "We report the development of GPT-4, a large-scale, multimodal "
                        "model which can accept image and text inputs and produce text "
                        "outputs. GPT-4 exhibits human-level performance on various "
                        "professional and academic benchmarks.",
            "year": 2023,
            "citations": 12000,
            "url": "https://arxiv.org/abs/2303.08774",
        },
        {
            "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs",
            "authors": ["Wei et al."],
            "abstract": "We explore how generating a chain of thought — a series of "
                        "intermediate reasoning steps — significantly improves the ability "
                        "of large language models to perform complex reasoning.",
            "year": 2022,
            "citations": 8500,
            "url": "https://arxiv.org/abs/2201.11903",
        },
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "authors": ["Bai et al."],
            "abstract": "We experiment with methods for training a harmless AI assistant "
                        "through self-improvement, without any human labels identifying "
                        "harmful outputs. We refer to this approach as Constitutional AI.",
            "year": 2022,
            "citations": 3200,
            "url": "https://arxiv.org/abs/2212.08073",
        },
    ]


# ── Agent loop (real OpenAI or simulation) ───────────────────────────────

def run_with_openai():
    """Full agent loop using the OpenAI SDK."""
    from openai import OpenAI
    from ctxtual.integrations.openai import (
        to_openai_tools,
        has_tool_calls,
        handle_tool_calls,
    )

    client = OpenAI()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Use the provided tools to "
                "search for papers and explore results. Always search before "
                "answering. Cite specific papers with titles and years."
            ),
        },
        {
            "role": "user",
            "content": "What are the key papers on transformer architectures and reasoning in LLMs?",
        },
    ]

    # The agent loop: call LLM → execute tools → feed results back → repeat
    for turn in range(10):
        # Get tool schemas (includes both producer AND consumer tools)
        tools = to_openai_tools(forge)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools if tools else None,
        )

        choice = response.choices[0]

        # If the LLM wants to call tools
        if has_tool_calls(response):
            messages.append(choice.message)

            # Dispatch all tool calls through ctx
            tool_results = handle_tool_calls(forge, response)
            messages.extend(tool_results)

            print(f"[Turn {turn + 1}] {len(tool_results)} tool call(s) dispatched")
            for tr in tool_results:
                content = json.loads(tr["content"]) if isinstance(tr["content"], str) else tr["content"]
                if isinstance(content, dict) and "workspace_id" in content:
                    print(f"  → Workspace created: {content['workspace_id']} ({content['item_count']} items)")
                elif isinstance(content, dict) and "total_matches" in content:
                    print(f"  → Search returned {content['total_matches']} matches")
                else:
                    print(f"  → Tool result: {str(content)[:100]}...")
        else:
            # LLM is done — print final answer
            print(f"\n{'=' * 70}")
            print("AGENT RESPONSE:")
            print("=" * 70)
            print(choice.message.content)
            break


def run_simulation():
    """Simulate what the OpenAI loop does, without an API key."""
    from ctxtual.integrations.openai import to_openai_tools

    print("=" * 70)
    print("SIMULATION MODE (no OPENAI_API_KEY set)")
    print("=" * 70)

    # Show what schemas the LLM would see
    tools = to_openai_tools(forge)
    print(f"\n[Setup] {len(tools)} tools available to the LLM:")
    for t in tools:
        fn = t["function"]
        print(f"  • {fn['name']}: {fn['description'][:60]}...")

    # Step 1: LLM calls the producer
    print("\n[Turn 1] LLM calls: search_arxiv(query='transformers')")
    ref = forge.dispatch_tool_call("search_arxiv", {"query": "transformers"})
    print(f"  → Workspace: {ref['workspace_id']}, {ref['item_count']} papers")
    print(f"  → data_shape: {ref['data_shape']}")

    # Now more tools are available (consumer tools for this workspace)
    tools = to_openai_tools(forge)
    print(f"\n  Tools now available: {len(tools)}")

    ws_id = ref["workspace_id"]

    # Step 2: LLM searches within results
    print(f"\n[Turn 2] LLM calls: papers_search(query='reasoning')")
    result = forge.dispatch_tool_call(
        "papers_search",
        {"workspace_id": ws_id, "query": "reasoning", "max_results": 3},
    )
    print(f"  → {result['total_matches']} matches")
    for m in result["matches"]:
        print(f"    • {m['title']} ({m['year']})")

    # Step 3: LLM reads a specific paper
    print(f"\n[Turn 3] LLM calls: papers_get_item(index=0)")
    result = forge.dispatch_tool_call(
        "papers_get_item", {"workspace_id": ws_id, "index": 0}
    )
    print(f"  → {result['title']}")
    print(f"    {result['abstract'][:80]}...")
    print(f"    Citations: {result['citations']:,}")

    # Step 4: LLM paginates
    print(f"\n[Turn 4] LLM calls: papers_paginate(page=0, size=2)")
    result = forge.dispatch_tool_call(
        "papers_paginate", {"workspace_id": ws_id, "page": 0, "size": 2}
    )
    # Result is wrapped in hint envelope since paginator has output_hint
    data = result["result"]
    print(f"  → Page 0: {len(data['items'])} items, {data['total_pages']} pages total")
    print(f"  → Hint: {result['_hint'][:80]}...")

    print(f"\n{'=' * 70}")
    print("In production, the LLM reads these results and composes an answer.")
    print("The key: 5 papers never entered the context window simultaneously.")
    print("=" * 70)


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        run_with_openai()
    else:
        run_simulation()
