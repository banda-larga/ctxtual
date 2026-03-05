"""
Real-world example: FastAPI server with concurrent agent sessions.

Scenario: A web API that serves multiple users simultaneously, each
running their own agent session against a shared ctx store.
Demonstrates thread safety, session isolation, and proper concurrent
access patterns.

This is what production deployment looks like: a stateless HTTP API
where each request may trigger tool calls against the shared forge.

Prerequisites:
    pip install fastapi uvicorn

Run:
    uv run python examples/11_concurrent_server.py

Then test:
    curl -X POST http://localhost:8000/search -d '{"query": "machine learning"}'
    curl http://localhost:8000/workspace/{workspace_id}/page/0
"""

import threading
import time
import uuid

from ctxtual import Forge, MemoryStore
from ctxtual.utils import paginator, text_search

# ── Setup: one Forge instance shared across all requests ─────────────────

# MemoryStore is thread-safe (uses RLock internally).
# For persistence across restarts, swap to SQLiteStore.
forge = Forge(
    store=MemoryStore(max_workspaces=1000),  # LRU eviction
    default_ttl=1800,  # 30 min TTL — auto-cleanup stale sessions
)

pager  = paginator(forge, "results")
search = text_search(forge, "results", fields=["title", "description"])


@forge.producer(
    workspace_type="results",
    toolsets=[pager, search],
    # Each search gets a unique workspace (no key template = auto UUID)
)
def search_catalog(query: str, category: str = "all") -> list[dict]:
    """Search the product catalog. Thread-safe — called concurrently.

    Args:
        query: Search query string.
        category: Product category filter.
    """
    # Simulate DB query
    time.sleep(0.01)  # Simulate latency
    return [
        {"id": f"prod-{i}", "title": f"Result {i} for '{query}'",
         "description": f"A product matching '{query}' in {category}",
         "price": 10.0 + i * 5, "category": category}
        for i in range(50)
    ]


# ── FastAPI app ──────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="ctx Demo API")

    class SearchRequest(BaseModel):
        query: str
        category: str = "all"

    @app.post("/search")
    async def api_search(req: SearchRequest):
        """Producer endpoint: search and store results."""
        ref = search_catalog(query=req.query, category=req.category)
        return ref  # WorkspaceRef dict — self-describing

    @app.get("/workspace/{workspace_id}/page/{page}")
    async def api_paginate(workspace_id: str, page: int = 0, size: int = 10):
        """Consumer endpoint: paginate stored results."""
        result = forge.dispatch_tool_call(
            "results_paginate",
            {"workspace_id": workspace_id, "page": page, "size": size},
        )
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/workspace/{workspace_id}/search")
    async def api_search_within(workspace_id: str, q: str, max_results: int = 10):
        """Consumer endpoint: search within stored results."""
        result = forge.dispatch_tool_call(
            "results_search",
            {"workspace_id": workspace_id, "query": q, "max_results": max_results},
        )
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.get("/workspaces")
    async def api_list_workspaces():
        """List all active workspaces with metadata."""
        workspaces = []
        for ws_id in forge.list_workspaces():
            meta = forge.workspace_meta(ws_id)
            if meta:
                workspaces.append({
                    "workspace_id": ws_id,
                    "type": meta.workspace_type,
                    "items": meta.item_count,
                    "shape": meta.data_shape,
                    "age_seconds": round(time.time() - meta.created_at),
                })
        return {"workspaces": workspaces, "total": len(workspaces)}

    @app.post("/sweep")
    async def api_sweep():
        """Manually sweep expired workspaces."""
        expired = forge.sweep_expired()
        return {"swept": len(expired), "workspace_ids": expired}

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ── Simulation (no FastAPI needed) ───────────────────────────────────────

def simulate_concurrent():
    """Simulate concurrent requests to demonstrate thread safety."""
    print("=" * 70)
    print("CONCURRENT AGENT SESSIONS (thread safety demo)")
    print("=" * 70)

    results = {}
    errors = []

    def user_session(user_id: str, query: str):
        """Simulate one user's full session: search → paginate → search."""
        try:
            # Producer call (thread-safe)
            ref = search_catalog(query=query)
            ws_id = ref["workspace_id"]

            # Consumer calls (thread-safe)
            page_result = forge.dispatch_tool_call(
                "results_paginate",
                {"workspace_id": ws_id, "page": 0, "size": 5},
            )

            search_result = forge.dispatch_tool_call(
                "results_search",
                {"workspace_id": ws_id, "query": query.split()[0], "max_results": 3},
            )

            results[user_id] = {
                "workspace_id": ws_id,
                "items": ref["item_count"],
                "page_items": len(page_result["result"]["items"]),
                "search_matches": search_result["total_matches"],
            }
        except Exception as e:
            errors.append((user_id, str(e)))

    # Launch 20 concurrent user sessions
    threads = []
    queries = [
        "machine learning", "web development", "data science",
        "cloud computing", "mobile apps", "security tools",
        "database optimization", "API design", "testing frameworks",
        "deployment automation", "monitoring", "CI/CD pipelines",
        "containerization", "microservices", "serverless",
        "GraphQL", "REST API", "WebSocket", "gRPC", "message queues",
    ]

    print(f"\nLaunching {len(queries)} concurrent sessions...")
    start = time.time()

    for i, query in enumerate(queries):
        t = threading.Thread(
            target=user_session,
            args=(f"user_{i:03d}", query),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start
    print(f"All {len(queries)} sessions completed in {elapsed:.2f}s")
    print(f"Errors: {len(errors)}")

    if errors:
        for user_id, err in errors:
            print(f"  ✗ {user_id}: {err}")

    # Verify isolation: each session has its own workspace
    all_workspaces = forge.list_workspaces()
    print(f"\nWorkspaces created: {len(all_workspaces)}")
    assert len(all_workspaces) == len(queries), \
        f"Expected {len(queries)} workspaces, got {len(all_workspaces)}"

    # Show a few results
    print(f"\nSample results:")
    for user_id in sorted(results.keys())[:5]:
        r = results[user_id]
        print(f"  {user_id}: ws={r['workspace_id'][:20]}... "
              f"items={r['items']}, page={r['page_items']}, "
              f"search={r['search_matches']}")

    # Cleanup: sweep all (they're not expired yet, but demonstrate the API)
    print(f"\nActive workspaces before sweep: {len(forge.list_workspaces())}")
    forge.clear()
    print(f"Active workspaces after clear: {len(forge.list_workspaces())}")

    print(f"\n{'=' * 70}")
    print("KEY POINTS:")
    print("  • One Forge instance safely handles 20+ concurrent sessions")
    print("  • Each session gets an isolated workspace (auto UUID keys)")
    print("  • Store operations are thread-safe (RLock on MemoryStore)")
    print("  • TTL + sweep_expired() prevents memory leaks in long-running servers")
    if HAS_FASTAPI:
        print("  • FastAPI app defined — run with: uvicorn examples.11_concurrent_server:app")
    else:
        print("  • Install fastapi for the HTTP server: pip install fastapi uvicorn")
    print("=" * 70)


if __name__ == "__main__":
    simulate_concurrent()
