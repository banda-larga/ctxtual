"""
Session scoping: isolate workspaces between concurrent agent conversations.

Workspace IDs are the only handle an LLM has on stored data, so a server
that shares one store across many conversations needs a way to stop one
conversation from reading, listing, or overwriting another's workspaces.

A *session* is an opaque string (a conversation id, a user id, ...) held in
a :class:`contextvars.ContextVar`, so it follows the current thread or
asyncio task without being threaded through every call::

    with ctx.session("conversation-42"):
        ctx.dispatch_tool_call("search_papers", {"query": "ml"})

    # or, per call:
    ctx.dispatch_tool_call("search_papers", {"query": "ml"},
                           session_id="conversation-42")

Rules while a session is active:

- New workspaces are stamped with the session id.
- Deterministic workspace IDs (producer ``key=``, ``emit(workspace_id=)``,
  pipeline ``save_as``) get a short session-derived suffix, so two sessions
  using the same key never collide.
- A workspace is *visible* if it belongs to the current session or to no
  session at all (shared, global data).  Anything else behaves exactly like
  a missing workspace — its existence is not revealed.

With no active session everything is visible, which keeps single-user
scripts and trusted back-office code working unchanged.
"""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctxtual.types import WorkspaceMeta

_current_session: ContextVar[str | None] = ContextVar(
    "ctxtual_session", default=None
)


def current_session() -> str | None:
    """Return the active session id, or ``None`` when unscoped."""
    return _current_session.get()


@contextmanager
def session_scope(session_id: str | None) -> Iterator[None]:
    """Run the enclosed block inside *session_id* (``None`` = unscoped)."""
    token = _current_session.set(session_id)
    try:
        yield
    finally:
        _current_session.reset(token)


def session_suffix(session_id: str) -> str:
    """Short, stable, non-reversible tag derived from a session id."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:12]


def scoped_id(base: str) -> str:
    """Namespace a deterministic workspace id to the active session."""
    session = current_session()
    if session is None:
        return base
    return f"{base}@{session_suffix(session)}"


def is_visible(meta: "WorkspaceMeta") -> bool:
    """Whether the active session may see the workspace described by *meta*."""
    session = current_session()
    return session is None or meta.session_id is None or meta.session_id == session


def visible_workspaces(
    store: Any,
    workspace_type: str | None = None,
    *,
    limit: int = 20,
    scan_limit: int = 500,
) -> list[str]:
    """
    Workspace ids the active session may see, newest first, at most *limit*.

    Used to build "did you mean" lists in error messages without leaking
    other sessions' workspaces.  Only the *scan_limit* newest workspaces are
    inspected, so error responses stay cheap in large shared stores.
    """
    result: list[str] = []
    for ws_id in reversed(store.list_workspaces(workspace_type)[-scan_limit:]):
        meta = store.get_meta(ws_id)
        if meta is not None and is_visible(meta) and not meta.is_expired:
            result.append(ws_id)
            if len(result) >= limit:
                break
    return result
