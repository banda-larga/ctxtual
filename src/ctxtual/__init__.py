"""
ctxtual — context engineering library for AI agents.

Provides a decorator-based system for managing large tool results:
instead of returning bulk data directly to an LLM, producers store
results in a workspace and return a compact notification.  Consumer
tools then let the agent explore the data incrementally.

Quick start::

    from ctxtual import Ctx, MemoryStore
    from ctxtual.utils import paginator, text_search

    ctx = Ctx(store=MemoryStore())

    @ctx.producer(workspace_type="papers", toolsets=[
        paginator(ctx),
        text_search(ctx, fields=["title", "abstract"]),
    ])
    def fetch_papers(query: str, limit: int = 10_000):
        return database.search(query, limit)

    result = fetch_papers("machine learning")
    # → { "status": "workspace_ready", "workspace_id": "papers_...", ... }
"""

from ctxtual.ctx import ConsumerContext, Ctx
from ctxtual.exceptions import (
    CtxtualError,
    InvalidArgumentError,
    PayloadTooLargeError,
    ToolExecutionError,
    ToolSetNotRegisteredError,
    WorkspaceConflictError,
    WorkspaceExpiredError,
    WorkspaceNotFoundError,
    WorkspaceTypeMismatchError,
)
from ctxtual.session import current_session, session_scope
from ctxtual.store import BaseStore, MemoryStore, SQLiteStore
from ctxtual.toolset import BoundToolSet, ToolSet, ToolSpec
from ctxtual.transforms import chunk_text, split_markdown_sections, split_sections
from ctxtual.types import WorkspaceMeta, WorkspaceRef

__all__ = [
    # Stores
    "BaseStore",
    "BoundToolSet",
    "ConsumerContext",
    # Core
    "Ctx",
    # Exceptions
    "CtxtualError",
    "InvalidArgumentError",
    "MemoryStore",
    "PayloadTooLargeError",
    "SQLiteStore",
    "ToolExecutionError",
    # ToolSet
    "ToolSet",
    "ToolSetNotRegisteredError",
    "ToolSpec",
    "WorkspaceConflictError",
    "WorkspaceExpiredError",
    # Types
    "WorkspaceMeta",
    "WorkspaceNotFoundError",
    "WorkspaceRef",
    "WorkspaceTypeMismatchError",
    # Transforms
    "chunk_text",
    # Sessions
    "current_session",
    "session_scope",
    "split_markdown_sections",
    "split_sections",
]

__version__ = "0.5.1"
