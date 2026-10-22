"""
12 — Offload large results from an MCP server (FastMCP middleware).

One line on your own server: results above 20,000 characters are stored and
the model gets a notice plus four result_* tools (search, read, query, list)
to explore them.

Requires the extra:  pip install "ctxtual[mcp]"
Run:                 uv run --extra mcp python examples/12_mcp_offload.py
"""

import asyncio
import json
import random

from fastmcp import Client, FastMCP

from ctxtual.mcp import Offload

mcp = FastMCP("orders")


def make_orders(limit: int) -> list[dict]:
    rng = random.Random(7)
    return [
        {
            "id": i,
            "customer": f"customer_{rng.randint(1, 400)}",
            "region": rng.choice(["EU", "US", "APAC"]),
            "status": rng.choice(["paid", "refunded", "pending"]),
            "amount": round(rng.uniform(5, 900), 2),
            "note": "delivered to front desk, signed by recipient",
        }
        for i in range(limit)
    ]


@mcp.tool
def list_orders(limit: int = 5000) -> list[dict]:
    """List recent orders."""
    return make_orders(limit)


@mcp.tool
def order_count() -> dict:
    """Total number of orders (a small result: passes through)."""
    return {"count": 5000}


mcp.add_middleware(Offload())


def text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


async def main() -> None:
    async with Client(mcp) as client:
        print("tools:", [t.name for t in await client.list_tools()])

        small = await client.call_tool("order_count", {})
        print(f"\norder_count → {text(small)}  (passed through)")

        big = await client.call_tool("list_orders", {})
        notice = json.loads(text(big))
        raw = len(json.dumps(make_orders(5000)))
        print(
            f"\nlist_orders → {len(text(big)):,}-char notice instead of ~{raw:,} chars"
        )
        print(
            json.dumps(
                {k: notice[k] for k in ("result_id", "items", "fields")},
                indent=2,
                ensure_ascii=False,
            )
        )
        print("how_to_explore:", notice["how_to_explore"])
        result_id = notice["result_id"]

        # What a model would do next: one compound query instead of reading 5,000 rows.
        refunds = await client.call_tool(
            "result_query",
            {
                "result_id": result_id,
                "where": {"status": "refunded"},
                "steps": [
                    {
                        "group_by": {
                            "field": "region",
                            "metrics": {"orders": "count", "total": "sum:amount"},
                        }
                    }
                ],
                "sort": "-total",
            },
        )
        print(f"\nrefunds by region ({len(text(refunds))} chars):")
        for row in json.loads(text(refunds))["rows"]:
            print(
                f"  {row['region']:<5} {row['orders']:>4} orders  {row['total']:>10,.2f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
