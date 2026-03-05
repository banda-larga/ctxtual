"""
Real-world example: Multi-step data pipeline with derived workspaces.

Scenario: An analytics agent ingests raw e-commerce order data, then
progressively narrows it through a pipeline:

  1. Load all orders (producer)
  2. Filter to high-value orders (consumer → derived workspace)
  3. Group by region and compute stats (consumer → derived workspace)

Each step creates a new workspace. The agent can explore any workspace
at any level — raw orders, filtered orders, or aggregated stats.

This is the pattern for any "ETL in the agent loop" use case: data
enrichment, report generation, multi-step analysis.

Run:
    uv run python examples/03_data_pipeline.py
"""

from ctxtual import Forge, MemoryStore, ConsumerContext
from ctxtual.utils import paginator, text_search, filter_set, kv_reader

# ── Setup ────────────────────────────────────────────────────────────────

forge = Forge(store=MemoryStore())

# Toolsets for raw orders (list of dicts)
orders_pager   = paginator(forge, "orders")
orders_search  = text_search(forge, "orders", fields=["customer", "product", "region"])
orders_filter  = filter_set(forge, "orders")

# Toolsets for the filtered subset (also a list)
vip_pager  = paginator(forge, "vip_orders")
vip_filter = filter_set(forge, "vip_orders")

# Toolset for aggregated stats (a dict, not a list)
stats_kv = kv_reader(forge, "region_stats")


# ── Step 1: Producer — load raw orders ───────────────────────────────────

@forge.producer(
    workspace_type="orders",
    toolsets=[orders_pager, orders_search, orders_filter],
    key="orders_{period}",
)
def load_orders(period: str = "2024-Q4") -> list[dict]:
    """Load orders for a given period. In production: query your DB/API.

    Args:
        period: Time period to load (e.g., '2024-Q4').
    """
    return [
        {"order_id": "ORD-001", "customer": "Alice Chen",     "product": "Enterprise License", "amount": 15000, "region": "APAC",   "status": "completed"},
        {"order_id": "ORD-002", "customer": "Bob Martinez",   "product": "Pro Plan",           "amount": 2400,  "region": "LATAM",  "status": "completed"},
        {"order_id": "ORD-003", "customer": "Carol Schmidt",  "product": "Enterprise License", "amount": 24000, "region": "EMEA",   "status": "completed"},
        {"order_id": "ORD-004", "customer": "David Park",     "product": "Starter Plan",       "amount": 600,   "region": "APAC",   "status": "pending"},
        {"order_id": "ORD-005", "customer": "Eve Johnson",    "product": "Enterprise License", "amount": 18000, "region": "NA",     "status": "completed"},
        {"order_id": "ORD-006", "customer": "Frank Liu",      "product": "Pro Plan",           "amount": 2400,  "region": "APAC",   "status": "completed"},
        {"order_id": "ORD-007", "customer": "Grace Okafor",   "product": "Enterprise License", "amount": 30000, "region": "EMEA",   "status": "completed"},
        {"order_id": "ORD-008", "customer": "Henry Singh",    "product": "Starter Plan",       "amount": 600,   "region": "APAC",   "status": "cancelled"},
        {"order_id": "ORD-009", "customer": "Irene Volkov",   "product": "Pro Plan",           "amount": 4800,  "region": "EMEA",   "status": "completed"},
        {"order_id": "ORD-010", "customer": "James Wright",   "product": "Enterprise License", "amount": 22000, "region": "NA",     "status": "pending"},
    ]


# ── Step 2: Consumer — filter to high-value orders ──────────────────────

@forge.consumer(
    workspace_type="orders",
    produces="vip_orders",
    produces_toolsets=[vip_pager, vip_filter],
)
def extract_vip_orders(
    workspace_id: str,
    min_amount: int = 10000,
    forge_ctx: ConsumerContext = None,
) -> dict:
    """Filter orders above a minimum amount threshold.

    Args:
        workspace_id: Source workspace with all orders.
        min_amount: Minimum order amount to qualify as VIP.
    """
    all_orders = forge_ctx.get_items()
    vip = [o for o in all_orders if o["amount"] >= min_amount]

    return forge_ctx.emit(
        vip,
        meta={
            "derived_from": workspace_id,
            "filter": f"amount >= {min_amount}",
            "original_count": len(all_orders),
        },
    )


# ── Step 3: Consumer — aggregate stats by region ────────────────────────

@forge.consumer(
    workspace_type="vip_orders",
    produces="region_stats",
    produces_toolsets=[stats_kv],
)
def compute_region_stats(
    workspace_id: str,
    forge_ctx: ConsumerContext = None,
) -> dict:
    """Compute revenue and count statistics grouped by region.

    Args:
        workspace_id: Source workspace with VIP orders.
    """
    orders = forge_ctx.get_items()

    stats: dict[str, dict] = {}
    for order in orders:
        region = order["region"]
        if region not in stats:
            stats[region] = {"count": 0, "total_revenue": 0, "orders": []}
        stats[region]["count"] += 1
        stats[region]["total_revenue"] += order["amount"]
        stats[region]["orders"].append(order["order_id"])

    # Add summary
    stats["_summary"] = {
        "total_regions": len(stats) - 1,  # exclude _summary itself
        "total_revenue": sum(s["total_revenue"] for k, s in stats.items() if k != "_summary"),
        "total_orders": len(orders),
    }

    # Emit as a dict workspace (kv_reader will handle it)
    return forge_ctx.emit(
        stats,
        meta={"derived_from": workspace_id},
    )


# ── Run the pipeline ─────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 70)
    print("DATA PIPELINE: Orders → VIP Filter → Region Stats")
    print("=" * 70)

    # Step 1: Load raw data
    orders_ref = load_orders(period="2024-Q4")
    print(f"\n[Step 1] Loaded {orders_ref['item_count']} orders")
    print(f"  Workspace: {orders_ref['workspace_id']} (shape: {orders_ref['data_shape']})")
    ws_orders = orders_ref["workspace_id"]

    # Agent explores: what regions exist?
    result = forge.dispatch_tool_call(
        "orders_field_values",
        {"workspace_id": ws_orders, "field": "region"},
    )
    print(f"  Regions: {result['distinct_values']}")

    # Agent explores: sort by amount
    result = forge.dispatch_tool_call(
        "orders_sort_by",
        {"workspace_id": ws_orders, "field": "amount", "descending": True, "limit": 3},
    )
    top3 = [f"{o['customer']} (${o['amount']:,})" for o in result]
    print(f"  Top 3 orders: {top3}")

    # Step 2: Agent decides to filter for VIP orders
    vip_ref = extract_vip_orders(workspace_id=ws_orders, min_amount=10000)
    print(f"\n[Step 2] Filtered to {vip_ref['item_count']} VIP orders (≥$10,000)")
    print(f"  Workspace: {vip_ref['workspace_id']} (shape: {vip_ref['data_shape']})")
    ws_vip = vip_ref["workspace_id"]

    # Agent explores the VIP subset
    result = forge.dispatch_tool_call(
        "vip_orders_paginate",
        {"workspace_id": ws_vip, "page": 0, "size": 5},
    )
    data = result["result"]  # unwrap hint envelope
    print(f"  VIP customers: {[o['customer'] for o in data['items']]}")

    # Step 3: Compute region stats
    stats_ref = compute_region_stats(workspace_id=ws_vip)
    print(f"\n[Step 3] Region stats computed")
    print(f"  Workspace: {stats_ref['workspace_id']} (shape: {stats_ref['data_shape']})")
    ws_stats = stats_ref["workspace_id"]

    # Agent reads the stats (dict workspace → kv_reader)
    result = forge.dispatch_tool_call(
        "region_stats_get_keys",
        {"workspace_id": ws_stats},
    )
    data = result["result"]  # unwrap hint envelope
    print(f"  Stat keys: {data}")

    for region in [k for k in data if not k.startswith("_")]:
        val = forge.dispatch_tool_call(
            "region_stats_get_value",
            {"workspace_id": ws_stats, "key": region},
        )
        print(f"  {region}: {val['count']} orders, ${val['total_revenue']:,} revenue")

    # Read summary
    summary = forge.dispatch_tool_call(
        "region_stats_get_value",
        {"workspace_id": ws_stats, "key": "_summary"},
    )
    print(f"\n  TOTAL: {summary['total_orders']} VIP orders across "
          f"{summary['total_regions']} regions = ${summary['total_revenue']:,}")

    # Show all workspaces in the system
    print(f"\n{'=' * 70}")
    print("WORKSPACE LINEAGE:")
    for ws_id in forge.list_workspaces():
        meta = forge.workspace_meta(ws_id)
        print(f"  {ws_id} ({meta.workspace_type}, {meta.data_shape}, "
              f"{meta.item_count} items)")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
