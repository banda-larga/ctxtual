"""
ctx — context engineering library for AI agents.

Provides a decorator-based system for managing large tool results:
instead of returning bulk data directly to an LLM, producers store
results in a workspace and return a compact notification.  Consumer
tools then let the agent explore the data incrementally.

Quick start::

    from ctxtual import Forge, MemoryStore
    from ctxtual.utils import paginator, text_search

    forge = Forge(store=MemoryStore())

    @forge.producer(workspace_type="papers", toolsets=[
        paginator(forge),
        text_search(forge, fields=["title", "abstract"]),
    ])
    def fetch_papers(query: str, limit: int = 10_000):
        return database.search(query, limit)

    result = fetch_papers("machine learning")
    # → { "status": "workspace_ready", "workspace_id": "papers_...", ... }
"""

from ctxtual.exceptions import (
    ContextForgeError,
    PayloadTooLargeError,
    ToolExecutionError,
    ToolSetNotRegisteredError,
    WorkspaceExpiredError,
    WorkspaceNotFoundError,
    WorkspaceTypeMismatchError,
)
from ctxtual.forge import ConsumerContext, Forge
from ctxtual.store import BaseStore, MemoryStore, SQLiteStore
from ctxtual.toolset import BoundToolSet, ToolSet, ToolSpec
from ctxtual.transforms import chunk_text, split_markdown_sections, split_sections
from ctxtual.types import WorkspaceMeta, WorkspaceRef

__all__ = [
    # Core
    "Forge",
    "ConsumerContext",
    # ToolSet
    "ToolSet",
    "ToolSpec",
    "BoundToolSet",
    # Stores
    "BaseStore",
    "MemoryStore",
    "SQLiteStore",
    # Types
    "WorkspaceMeta",
    "WorkspaceRef",
    # Transforms
    "chunk_text",
    "split_sections",
    "split_markdown_sections",
    # Exceptions
    "ContextForgeError",
    "WorkspaceNotFoundError",
    "WorkspaceExpiredError",
    "WorkspaceTypeMismatchError",
    "ToolSetNotRegisteredError",
    "PayloadTooLargeError",
    "ToolExecutionError",
]

__version__ = "0.1.2"
