"""
Real-world example: Custom domain toolset with data shape validation.

Scenario: A financial analysis agent with domain-specific tools that go
beyond the built-in paginator/search. Shows how to create custom toolsets
that are type-safe, self-describing, and validate data shape.

This is the pattern when your domain needs tools that don't map to
"paginate" or "search" — aggregation, pivoting, time-series ops, etc.

Run:
    uv run python examples/04_custom_tools.py
"""

from typing import Any

from ctxtual import Forge, MemoryStore
from ctxtual.toolset import ToolSet
from ctxtual.utils import paginator

# ── Setup ────────────────────────────────────────────────────────────────

forge = Forge(store=MemoryStore())


# ── Custom ToolSet: Financial analytics ──────────────────────────────────

def financial_analytics(forge: Forge) -> ToolSet:
    """
    Domain-specific tools for financial transaction data.

    By setting data_shape="list", ctx will:
    1. Warn at producer-time if the payload isn't a list
    2. Return a clear error at tool-time if the workspace has wrong shape
    """
    ts = forge.toolset("transactions")
    ts.data_shape = "list"  # These tools expect list data

    @ts.tool(
        name="transactions_summary",
        output_hint=(
            "For details, use transactions_by_category or transactions_paginate "
            "with workspace_id='{workspace_id}'."
        ),
    )
    def summary(workspace_id: str) -> dict[str, Any]:
        """Compute aggregate statistics for all transactions.

        Returns total revenue, average transaction value, count by status,
        and date range.

        Args:
            workspace_id: The transactions workspace to analyze.
        """
        items = ts.store.get_items(workspace_id)
        if not items:
            return {"total": 0, "count": 0}

        amounts = [t["amount"] for t in items]
        statuses: dict[str, int] = {}
        for t in items:
            statuses[t["status"]] = statuses.get(t["status"], 0) + 1

        return {
            "count": len(items),
            "total_revenue": sum(amounts),
            "average": round(sum(amounts) / len(amounts), 2),
            "min": min(amounts),
            "max": max(amounts),
            "by_status": statuses,
        }

    @ts.tool(name="transactions_by_category")
    def by_category(workspace_id: str, top_n: int = 5) -> dict[str, Any]:
        """Group transactions by category and rank by total revenue.

        Args:
            workspace_id: The transactions workspace to analyze.
            top_n: Number of top categories to return.
        """
        items = ts.store.get_items(workspace_id)
        categories: dict[str, dict] = {}
        for t in items:
            cat = t.get("category", "uncategorized")
            if cat not in categories:
                categories[cat] = {"count": 0, "total": 0}
            categories[cat]["count"] += 1
            categories[cat]["total"] += t["amount"]

        ranked = sorted(categories.items(), key=lambda x: x[1]["total"], reverse=True)
        return {
            "categories": [
                {"name": name, **stats}
                for name, stats in ranked[:top_n]
            ],
            "total_categories": len(categories),
        }

    @ts.tool(name="transactions_anomalies")
    def anomalies(workspace_id: str, std_threshold: float = 2.0) -> dict[str, Any]:
        """Detect transactions with amounts that are statistical outliers.

        Uses simple z-score detection: flags transactions where the amount
        deviates more than std_threshold standard deviations from the mean.

        Args:
            workspace_id: The transactions workspace to analyze.
            std_threshold: Number of standard deviations to flag as anomaly.
        """
        items = ts.store.get_items(workspace_id)
        amounts = [t["amount"] for t in items]
        mean = sum(amounts) / len(amounts)
        variance = sum((a - mean) ** 2 for a in amounts) / len(amounts)
        std = variance ** 0.5

        if std == 0:
            return {"anomalies": [], "threshold": std_threshold}

        flagged = []
        for t in items:
            z_score = abs(t["amount"] - mean) / std
            if z_score > std_threshold:
                flagged.append({
                    **t,
                    "z_score": round(z_score, 2),
                    "deviation": "high" if t["amount"] > mean else "low",
                })

        return {
            "anomalies": flagged,
            "count": len(flagged),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "threshold": std_threshold,
        }

    @ts.tool(name="transactions_date_range")
    def date_range(
        workspace_id: str, start: str = "", end: str = ""
    ) -> dict[str, Any]:
        """Filter transactions to a specific date range.

        Args:
            workspace_id: The transactions workspace to filter.
            start: Start date (inclusive, ISO format YYYY-MM-DD). Empty = no lower bound.
            end: End date (inclusive, ISO format YYYY-MM-DD). Empty = no upper bound.
        """
        items = ts.store.get_items(workspace_id)
        filtered = []
        for t in items:
            date = t.get("date", "")
            if start and date < start:
                continue
            if end and date > end:
                continue
            filtered.append(t)

        return {
            "transactions": filtered,
            "count": len(filtered),
            "date_range": {"start": start or "earliest", "end": end or "latest"},
        }

    return ts


# ── Producer ─────────────────────────────────────────────────────────────

analytics = financial_analytics(forge)
pager = paginator(forge, "transactions")


@forge.producer(
    workspace_type="transactions",
    toolsets=[analytics, pager],
    key="txn_{account_id}_{period}",
)
def load_transactions(account_id: str, period: str = "2024-Q4") -> list[dict]:
    """Load financial transactions for an account.

    Args:
        account_id: Customer account identifier.
        period: Time period to load.
    """
    return [
        {"id": "TXN-001", "date": "2024-10-05", "amount": 150.00,  "category": "software",  "status": "completed", "vendor": "GitHub"},
        {"id": "TXN-002", "date": "2024-10-12", "amount": 2500.00, "category": "cloud",     "status": "completed", "vendor": "AWS"},
        {"id": "TXN-003", "date": "2024-10-15", "amount": 49.99,   "category": "software",  "status": "completed", "vendor": "Figma"},
        {"id": "TXN-004", "date": "2024-11-01", "amount": 2500.00, "category": "cloud",     "status": "completed", "vendor": "AWS"},
        {"id": "TXN-005", "date": "2024-11-03", "amount": 15000.00,"category": "consulting","status": "pending",   "vendor": "Acme Corp"},
        {"id": "TXN-006", "date": "2024-11-10", "amount": 89.00,   "category": "tools",     "status": "completed", "vendor": "Linear"},
        {"id": "TXN-007", "date": "2024-11-15", "amount": 350.00,  "category": "software",  "status": "completed", "vendor": "Datadog"},
        {"id": "TXN-008", "date": "2024-12-01", "amount": 2500.00, "category": "cloud",     "status": "completed", "vendor": "AWS"},
        {"id": "TXN-009", "date": "2024-12-05", "amount": 75000.00,"category": "hardware",  "status": "completed", "vendor": "Dell"},
        {"id": "TXN-010", "date": "2024-12-10", "amount": 199.00,  "category": "software",  "status": "refunded",  "vendor": "JetBrains"},
    ]


# ── Run ──────────────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("CUSTOM DOMAIN TOOLS: Financial Transaction Analysis")
    print("=" * 70)

    # Load data
    ref = load_transactions(account_id="ACCT-42", period="2024-Q4")
    ws_id = ref["workspace_id"]
    print(f"\nLoaded {ref['item_count']} transactions → {ws_id}")
    print(f"Data shape: {ref['data_shape']}")
    print(f"Tools: {[t.split('(')[0] for t in ref['available_tools']]}")

    # Summary stats
    print(f"\n{'─' * 50}")
    print("📊 Transaction Summary")
    result = forge.dispatch_tool_call(
        "transactions_summary", {"workspace_id": ws_id}
    )
    data = result["result"]  # unwrap hint envelope
    print(f"  Total: ${data['total_revenue']:,.2f} across {data['count']} transactions")
    print(f"  Average: ${data['average']:,.2f} (min: ${data['min']:,.2f}, max: ${data['max']:,.2f})")
    print(f"  By status: {data['by_status']}")

    # By category
    print(f"\n{'─' * 50}")
    print("📁 Revenue by Category")
    result = forge.dispatch_tool_call(
        "transactions_by_category", {"workspace_id": ws_id, "top_n": 5}
    )
    for cat in result["categories"]:
        bar = "█" * max(1, int(cat["total"] / 2000))
        print(f"  {cat['name']:>12}: ${cat['total']:>10,.2f} ({cat['count']} txns) {bar}")

    # Anomaly detection
    print(f"\n{'─' * 50}")
    print("⚠️  Anomaly Detection (>2σ)")
    result = forge.dispatch_tool_call(
        "transactions_anomalies", {"workspace_id": ws_id, "std_threshold": 2.0}
    )
    print(f"  Mean: ${result['mean']:,.2f}, Std: ${result['std']:,.2f}")
    print(f"  Flagged: {result['count']} anomalies")
    for a in result["anomalies"]:
        print(f"    {a['id']} | ${a['amount']:>10,.2f} | z={a['z_score']} | {a['vendor']}")

    # Date range filter
    print(f"\n{'─' * 50}")
    print("📅 November transactions only")
    result = forge.dispatch_tool_call(
        "transactions_date_range",
        {"workspace_id": ws_id, "start": "2024-11-01", "end": "2024-11-30"},
    )
    print(f"  {result['count']} transactions in November:")
    for t in result["transactions"]:
        print(f"    {t['date']} | ${t['amount']:>10,.2f} | {t['vendor']}")

    # Also works with built-in paginator
    print(f"\n{'─' * 50}")
    print("📃 Paginate (built-in + custom tools coexist)")
    result = forge.dispatch_tool_call(
        "transactions_paginate", {"workspace_id": ws_id, "page": 0, "size": 3}
    )
    data = result["result"]
    print(f"  Page 0 of {data['total_pages']}: {len(data['items'])} items")
    for item in data["items"]:
        print(f"    {item['id']} | {item['vendor']} | ${item['amount']}")

    # Show all registered tool schemas (what the LLM sees)
    print(f"\n{'─' * 50}")
    print("🔧 Tool schemas for LLM:")
    schemas = forge.get_all_tool_schemas()
    for s in schemas:
        fn = s["function"]
        params = list(fn["parameters"]["properties"].keys())
        print(f"  {fn['name']}({', '.join(params)})")
        print(f"    → {fn['description'][:70]}...")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    run()
