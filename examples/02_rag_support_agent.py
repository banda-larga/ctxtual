"""
Real-world example: RAG pipeline with ctx.

Scenario: A support agent ingests a company's knowledge base (FAQ + docs),
then answers user questions by searching, filtering, and reading relevant
articles — all through tool calls, never loading the full corpus into the
LLM context window.

This is the bread-and-butter use case for ctx: you have too much
data for the context window, so you let the agent explore it via tools.

Run:
    uv run python examples/02_rag_support_agent.py
"""

from ctxtual import Forge, MemoryStore
from ctxtual.utils import paginator, text_search, filter_set

# ── Setup ────────────────────────────────────────────────────────────────

forge = Forge(store=MemoryStore(), default_ttl=3600)

# Create toolsets for "articles" workspace type
pager   = paginator(forge, "articles")
search  = text_search(forge, "articles", fields=["title", "body", "tags"])
filters = filter_set(forge, "articles")


# ── Producer: load knowledge base ────────────────────────────────────────

@forge.producer(
    workspace_type="articles",
    toolsets=[pager, search, filters],
    # Deterministic key so repeated calls reuse the same workspace
    key="kb_{category}",
    meta={"source": "internal_kb"},
)
def load_knowledge_base(category: str = "all") -> list[dict]:
    """
    In production, this would query your vector DB, CMS, or file system.
    The agent never sees this raw data — it uses tools to explore it.
    """
    return [
        {
            "id": "art-001",
            "title": "How to reset your password",
            "body": "Go to Settings > Security > Reset Password. You'll receive "
                    "a confirmation email within 5 minutes.",
            "category": "account",
            "tags": ["password", "security", "account"],
            "views": 15420,
        },
        {
            "id": "art-002",
            "title": "Billing cycle explained",
            "body": "We bill on the 1st of each month. Pro-rated charges apply "
                    "when you upgrade mid-cycle. Downgrade takes effect next cycle.",
            "category": "billing",
            "tags": ["billing", "subscription", "payment"],
            "views": 8930,
        },
        {
            "id": "art-003",
            "title": "API rate limits",
            "body": "Free tier: 100 req/min. Pro: 1000 req/min. Enterprise: "
                    "unlimited. Rate limit headers are included in every response. "
                    "Use exponential backoff when you hit 429 errors.",
            "category": "api",
            "tags": ["api", "rate-limit", "developer"],
            "views": 22100,
        },
        {
            "id": "art-004",
            "title": "Two-factor authentication setup",
            "body": "Enable 2FA in Settings > Security > Two-Factor. We support "
                    "TOTP apps (Google Authenticator, Authy) and hardware keys "
                    "(YubiKey). SMS is not supported for security reasons.",
            "category": "account",
            "tags": ["2fa", "security", "account", "authentication"],
            "views": 11200,
        },
        {
            "id": "art-005",
            "title": "Webhook configuration",
            "body": "Configure webhooks in Settings > Integrations > Webhooks. "
                    "We send POST requests with JSON payloads. Events: "
                    "user.created, order.completed, subscription.changed. "
                    "Retry policy: 3 attempts with exponential backoff.",
            "category": "api",
            "tags": ["webhook", "api", "integration", "developer"],
            "views": 6800,
        },
        {
            "id": "art-006",
            "title": "Cancellation and refund policy",
            "body": "Cancel anytime from Settings > Subscription > Cancel. "
                    "Refunds are processed within 5-10 business days. "
                    "Annual plans get pro-rated refund for unused months.",
            "category": "billing",
            "tags": ["cancellation", "refund", "billing"],
            "views": 19500,
        },
        {
            "id": "art-007",
            "title": "Data export and GDPR compliance",
            "body": "Request a full data export from Settings > Privacy > Export. "
                    "We process requests within 48 hours. Format: JSON archive. "
                    "For deletion requests (right to be forgotten), contact "
                    "privacy@example.com.",
            "category": "privacy",
            "tags": ["gdpr", "privacy", "data-export", "compliance"],
            "views": 4200,
        },
    ]


# ── Simulate an agent answering a support question ───────────────────────

def simulate_agent():
    """
    Simulate what an LLM agent would do, step by step.
    In production, each step is a tool_call from the LLM.
    """
    print("=" * 70)
    print("SCENARIO: User asks 'How do I set up 2FA?'")
    print("=" * 70)

    # Step 1: Agent calls the producer to load the knowledge base
    ref = load_knowledge_base(category="all")
    print(f"\n[Step 1] Producer loaded KB → {ref['item_count']} articles")
    print(f"  Workspace: {ref['workspace_id']}")
    print(f"  Shape: {ref['data_shape']}")
    print(f"  Tools: {[t.split('(')[0] for t in ref['available_tools']]}")

    ws_id = ref["workspace_id"]

    # Step 2: Agent searches for "2FA" using text_search
    result = forge.dispatch_tool_call(
        "articles_search",
        {"workspace_id": ws_id, "query": "2FA authentication", "max_results": 5},
    )
    print(f"\n[Step 2] Search '2FA authentication' → {result['total_matches']} matches")
    for match in result["matches"]:
        print(f"  • {match['title']}")

    # Step 3: Agent reads the most relevant article
    result = forge.dispatch_tool_call(
        "articles_get_item",
        {"workspace_id": ws_id, "index": 3},  # 2FA article
    )
    print(f"\n[Step 3] Read article → '{result['title']}'")
    print(f"  Body: {result['body'][:80]}...")

    # Step 4: Agent also checks related security articles
    result = forge.dispatch_tool_call(
        "articles_filter_by",
        {"workspace_id": ws_id, "field": "category", "value": "account"},
    )
    print(f"\n[Step 4] Filter category=account → {result['count']} articles")
    for item in result["results"]:
        print(f"  • {item['title']} ({item['views']:,} views)")

    # Step 5: Agent checks most viewed articles (popular = likely helpful)
    result = forge.dispatch_tool_call(
        "articles_sort_by",
        {"workspace_id": ws_id, "field": "views", "descending": True, "limit": 3},
    )
    print(f"\n[Step 5] Top 3 by views:")
    for item in result:
        print(f"  • {item['title']} — {item['views']:,} views")

    # Step 6: Discover what categories exist
    result = forge.dispatch_tool_call(
        "articles_field_values",
        {"workspace_id": ws_id, "field": "category"},
    )
    print(f"\n[Step 6] Available categories: {result['distinct_values']}")

    print("\n" + "=" * 70)
    print("Agent would now compose an answer using the retrieved articles.")
    print("Key point: the LLM never saw all 7 articles at once — it")
    print("searched, filtered, and read only what it needed.")
    print("=" * 70)


if __name__ == "__main__":
    simulate_agent()
