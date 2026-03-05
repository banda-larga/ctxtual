"""
Core types for ctx.

All public-facing data structures live here so they can be imported
cleanly without pulling in any heavy dependencies.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Workspace identity


@dataclass
class WorkspaceMeta:
    """
    Everything the store knows about a workspace, excluding the payload.

    Attributes:
        workspace_id:     Unique identifier for this workspace.
        workspace_type:   Logical type (e.g. ``"papers"``, ``"employees"``).
        created_at:       Unix timestamp of creation.
        last_accessed_at: Unix timestamp of most recent read/write.
        producer_fn:      Name of the function that created this workspace.
        producer_kwargs:  Serialisable copy of the arguments passed to the producer.
        item_count:       Number of items in the primary payload.
        ttl:              Time-to-live in seconds.  ``None`` = never expires.
        data_shape:       Shape of the stored payload: ``"list"``, ``"dict"``,
                          ``"scalar"``, or ``""`` (unknown/legacy).
        extra:            Arbitrary user metadata attached at produce-time.
    """

    workspace_id: str
    workspace_type: str
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    producer_fn: str = ""
    producer_kwargs: dict[str, Any] = field(default_factory=dict)
    item_count: int = 0
    ttl: float | None = None
    data_shape: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Return ``True`` if the workspace has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Update ``last_accessed_at`` to the current time."""
        self.last_accessed_at = time.time()


# What gets returned to the agent after a produce() call


@dataclass
class WorkspaceRef:
    """
    Compact notification returned to the LLM agent instead of raw bulk data.

    This is what appears in the tool result — it tells the agent *where* the
    data lives, *which* tools it can use, and *what to do next*.

    The design principle is **self-describing output**: the agent should be
    able to understand how to explore the data purely from this notification,
    without relying on system prompt instructions.
    """

    workspace_id: str
    workspace_type: str
    item_count: int
    data_shape: str = ""
    producer_fn: str = ""
    available_tools: list[str] = field(default_factory=list)
    tool_descriptions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_fields: list[str] = field(default_factory=list)
    item_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return a plain dict suitable for JSON serialisation and LLM consumption.

        The output is self-describing: it includes tool call examples,
        step-by-step instructions, and a JSON Schema so the agent can
        construct queries (filter, sort, pipeline) without first paginating
        to discover the data structure.
        """
        # Build actionable next-step instructions
        next_steps: list[str] = []
        tool_examples: list[str] = []

        for tool_name in self.available_tools:
            call_example = f"{tool_name}(workspace_id='{self.workspace_id}')"
            tool_examples.append(call_example)
            desc = self.tool_descriptions.get(tool_name, "")
            if desc:
                next_steps.append(f"• {tool_name}: {desc}")

        # Fallback if no descriptions were provided (backward compat)
        if not next_steps and self.available_tools:
            next_steps = [
                f"• Call {t}(workspace_id='{self.workspace_id}') to explore."
                for t in self.available_tools
            ]

        result: dict[str, Any] = {
            "status": "workspace_ready",
            "workspace_id": self.workspace_id,
            "workspace_type": self.workspace_type,
            "item_count": self.item_count,
            "message": (
                f"✓ {self.item_count:,} item(s) stored in workspace "
                f"'{self.workspace_id}' (type: {self.workspace_type}). "
                f"Use the tools below to explore the data."
            ),
            "available_tools": tool_examples,
        }

        if self.data_shape:
            result["data_shape"] = self.data_shape

        # JSON Schema — tells the LLM field names AND types so it can
        # construct filter/sort/pipeline queries immediately.
        if self.item_schema is not None:
            result["item_schema"] = self.item_schema

        if next_steps:
            result["next_steps"] = next_steps

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_compact(self) -> str:
        """
        Return a single-line string for tight token budgets.

        Example:
            ``workspace_ready | id=papers_abc | 100 items | tools: paginate, search``
        """
        tools = ", ".join(self.available_tools) if self.available_tools else "none"
        return (
            f"workspace_ready | id={self.workspace_id} | "
            f"{self.item_count:,} items | tools: {tools}"
        )

    def __repr__(self) -> str:
        return (
            f"WorkspaceRef(id={self.workspace_id!r}, "
            f"type={self.workspace_type!r}, "
            f"count={self.item_count})"
        )


# Callable type aliases

#: Receives the raw kwargs dict of the producer call, returns the workspace_id.
KeyFactory = Callable[[dict[str, Any]], str]

#: Receives the raw return value from the producer; returns the value to store.
ResultTransformer = Callable[[Any], Any]
