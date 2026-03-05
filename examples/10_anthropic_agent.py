"""
Real-world example: Anthropic Claude agent with tool_use.

Scenario: A code review agent that loads pull request data, explores
files and comments, and generates review summaries. Uses the Anthropic
integration adapter.

This demonstrates:
  - Anthropic tool_use format (different from OpenAI)
  - The ctx Anthropic adapter for zero-glue dispatch
  - Multiple workspace types in one agent
  - kv_reader for structured metadata + paginator for file lists

Prerequisites:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    uv run python examples/10_anthropic_agent.py

Note: If no API key is set, runs in simulation mode.
"""

import json
import os

from ctxtual import Forge, MemoryStore
from ctxtual.utils import paginator, text_search, kv_reader

# ── Setup ────────────────────────────────────────────────────────────────

forge = Forge(store=MemoryStore())

# For file changes (list of dicts)
files_pager  = paginator(forge, "pr_files")
files_search = text_search(forge, "pr_files", fields=["filename", "patch"])

# For PR metadata (single dict)
meta_kv = kv_reader(forge, "pr_meta")


# ── Producers ────────────────────────────────────────────────────────────

@forge.producer(
    workspace_type="pr_files",
    toolsets=[files_pager, files_search],
    key="pr_{repo}_{number}_files",
)
def load_pr_files(repo: str, number: int) -> list[dict]:
    """Load changed files for a pull request.

    Args:
        repo: Repository in owner/name format.
        number: Pull request number.
    """
    # In production: call GitHub API
    return [
        {
            "filename": "src/auth/jwt.py",
            "status": "modified",
            "additions": 45,
            "deletions": 12,
            "patch": "@@ -10,6 +10,12 @@\n+def verify_token(token: str) -> dict:\n"
                     "+    \"\"\"Verify JWT token and return claims.\"\"\"\n"
                     "+    try:\n+        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])\n"
                     "+    except jwt.ExpiredSignatureError:\n+        raise AuthError('Token expired')\n",
        },
        {
            "filename": "src/auth/__init__.py",
            "status": "modified",
            "additions": 2,
            "deletions": 0,
            "patch": "@@ -1,3 +1,5 @@\n+from .jwt import verify_token\n+from .jwt import create_token\n",
        },
        {
            "filename": "tests/test_auth.py",
            "status": "added",
            "additions": 85,
            "deletions": 0,
            "patch": "@@ -0,0 +1,85 @@\n+import pytest\n+from src.auth.jwt import verify_token, create_token\n"
                     "+\n+def test_create_and_verify():\n+    token = create_token({'user_id': 1})\n"
                     "+    claims = verify_token(token)\n+    assert claims['user_id'] == 1\n",
        },
        {
            "filename": "src/middleware/auth_middleware.py",
            "status": "modified",
            "additions": 15,
            "deletions": 8,
            "patch": "@@ -5,8 +5,15 @@\n-def auth_required(f):\n-    # Old decorator\n"
                     "+def auth_required(f):\n+    \"\"\"Decorator that validates JWT in Authorization header.\"\"\"\n"
                     "+    @wraps(f)\n+    def wrapper(*args, **kwargs):\n"
                     "+        token = request.headers.get('Authorization', '').replace('Bearer ', '')\n"
                     "+        if not token:\n+            return jsonify({'error': 'Missing token'}), 401\n",
        },
        {
            "filename": "requirements.txt",
            "status": "modified",
            "additions": 1,
            "deletions": 0,
            "patch": "@@ -5,0 +6 @@\n+PyJWT==2.8.0\n",
        },
    ]


@forge.producer(
    workspace_type="pr_meta",
    toolsets=[meta_kv],
    key="pr_{repo}_{number}_meta",
)
def load_pr_metadata(repo: str, number: int) -> dict:
    """Load pull request metadata (title, description, author, etc).

    Args:
        repo: Repository in owner/name format.
        number: Pull request number.
    """
    # In production: call GitHub API
    return {
        "title": "feat: Add JWT authentication",
        "description": "Implements JWT-based auth with token creation, verification, "
                       "and middleware integration. Adds PyJWT dependency.\n\n"
                       "## Changes\n- New `verify_token` and `create_token` in auth/jwt.py\n"
                       "- Auth middleware now validates JWT from Authorization header\n"
                       "- Full test coverage for token lifecycle",
        "author": "alice",
        "base_branch": "main",
        "head_branch": "feat/jwt-auth",
        "state": "open",
        "created_at": "2024-12-15T10:30:00Z",
        "labels": ["feature", "security", "needs-review"],
        "reviewers": ["bob", "carol"],
        "stats": {
            "total_additions": 148,
            "total_deletions": 20,
            "files_changed": 5,
        },
    }


# ── Simulation ───────────────────────────────────────────────────────────

def run_simulation():
    from ctxtual.integrations.anthropic import to_anthropic_tools

    print("=" * 70)
    print("CODE REVIEW AGENT (Anthropic Claude simulation)")
    print("=" * 70)

    # Show available tools in Anthropic format
    tools = to_anthropic_tools(forge)
    print(f"\n[Setup] {len(tools)} tools registered (Anthropic format):")
    for t in tools:
        schema = t["input_schema"]
        params = list(schema.get("properties", {}).keys())
        print(f"  • {t['name']}({', '.join(params)})")

    # Step 1: Load PR metadata
    print(f"\n[Turn 1] Load PR metadata")
    meta_ref = forge.dispatch_tool_call(
        "load_pr_metadata", {"repo": "acme/backend", "number": 42}
    )
    print(f"  → Workspace: {meta_ref['workspace_id']} (shape: {meta_ref['data_shape']})")
    ws_meta = meta_ref["workspace_id"]

    # Step 2: Load PR files
    print(f"\n[Turn 2] Load PR files")
    files_ref = forge.dispatch_tool_call(
        "load_pr_files", {"repo": "acme/backend", "number": 42}
    )
    print(f"  → Workspace: {files_ref['workspace_id']} ({files_ref['item_count']} files)")
    ws_files = files_ref["workspace_id"]

    # Step 3: Read PR title and description
    print(f"\n[Turn 3] Read PR description")
    result = forge.dispatch_tool_call(
        "pr_meta_get_keys", {"workspace_id": ws_meta}
    )
    keys = result["result"]  # unwrap hint envelope
    print(f"  Keys: {keys}")

    title = forge.dispatch_tool_call(
        "pr_meta_get_value", {"workspace_id": ws_meta, "key": "title"}
    )
    print(f"  Title: {title}")

    labels = forge.dispatch_tool_call(
        "pr_meta_get_value", {"workspace_id": ws_meta, "key": "labels"}
    )
    print(f"  Labels: {labels}")

    stats = forge.dispatch_tool_call(
        "pr_meta_get_value", {"workspace_id": ws_meta, "key": "stats"}
    )
    print(f"  Stats: +{stats['total_additions']}/-{stats['total_deletions']} "
          f"across {stats['files_changed']} files")

    # Step 4: Browse changed files
    print(f"\n[Turn 4] Browse files")
    result = forge.dispatch_tool_call(
        "pr_files_paginate", {"workspace_id": ws_files, "page": 0, "size": 10}
    )
    data = result["result"]
    for f in data["items"]:
        print(f"  {f['status']:>10} | +{f['additions']:<3} -{f['deletions']:<3} | {f['filename']}")

    # Step 5: Search for security-relevant patterns
    print(f"\n[Turn 5] Search for 'secret' in patches")
    result = forge.dispatch_tool_call(
        "pr_files_search",
        {"workspace_id": ws_files, "query": "SECRET_KEY", "max_results": 5},
    )
    print(f"  Found {result['total_matches']} file(s) referencing SECRET_KEY:")
    for m in result["matches"]:
        print(f"  ⚠ {m['filename']} — hardcoded secret reference in patch")

    # Step 6: Read specific file detail
    print(f"\n[Turn 6] Read test file details")
    result = forge.dispatch_tool_call(
        "pr_files_get_item", {"workspace_id": ws_files, "index": 2}
    )
    print(f"  {result['filename']} ({result['status']}, +{result['additions']} lines)")
    print(f"  Patch preview: {result['patch'][:100]}...")

    print(f"\n{'=' * 70}")
    print("Agent now has enough context to write a thorough code review:")
    print("  • PR overview from metadata workspace")
    print("  • All file changes from files workspace")
    print("  • Security concern: SECRET_KEY reference found")
    print("  • Test coverage verified (test_auth.py added)")
    print("=" * 70)


def run_with_anthropic():
    """Full agent loop using the Anthropic SDK."""
    import anthropic
    from ctxtual.integrations.anthropic import (
        to_anthropic_tools,
        has_tool_use,
        handle_tool_use,
    )

    client = anthropic.Anthropic()
    messages = [
        {
            "role": "user",
            "content": "Review PR #42 in acme/backend. Load the metadata and files, "
                       "check for security issues, and provide a thorough review.",
        },
    ]

    system = (
        "You are a senior code reviewer. Use the tools to load PR data, "
        "explore files, search for patterns, and then provide a detailed review. "
        "Flag security issues, missing tests, and code quality concerns."
    )

    for turn in range(15):
        tools = to_anthropic_tools(forge)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        if has_tool_use(response):
            messages.append({"role": "assistant", "content": response.content})
            tool_results = handle_tool_use(forge, response)
            messages.append({"role": "user", "content": tool_results})
            print(f"[Turn {turn + 1}] {len(tool_results)} tool call(s)")
        else:
            text = "".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            print(f"\n{'=' * 70}")
            print("CODE REVIEW:")
            print("=" * 70)
            print(text)
            break


if __name__ == "__main__":
    if os.environ.get("ANTHROPIC_API_KEY"):
        run_with_anthropic()
    else:
        run_simulation()
