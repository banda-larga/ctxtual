"""ctx exceptions."""

from typing import Any


class CtxtualError(Exception):
    """Base exception for all ctx errors."""

    def to_llm_dict(self) -> dict[str, Any]:
        """Return a structured dict designed for LLM consumption.

        Subclasses override to add contextual recovery hints.
        """
        return {
            "error": str(self),
            "suggested_action": "Check the arguments and try again.",
        }


class WorkspaceNotFoundError(CtxtualError):
    """Raised when a workspace_id doesn't exist in the store."""

    def __init__(
        self,
        workspace_id: str,
        *,
        available: list[str] | None = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.available = available or []
        super().__init__(f"Workspace '{workspace_id}' not found in store.")

    def to_llm_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "error": f"Workspace '{self.workspace_id}' not found.",
            "workspace_id": self.workspace_id,
        }
        if self.available:
            d["available_workspaces"] = self.available[:20]
        d["suggested_action"] = (
            "Call the producer tool to create the workspace first, "
            "or use one of the available workspace IDs listed above."
            if self.available
            else "Call the producer tool to create a workspace first."
        )
        return d


class WorkspaceExpiredError(CtxtualError):
    """Raised when a workspace exists but has exceeded its TTL."""

    def __init__(self, workspace_id: str, *, producer_fn: str = "") -> None:
        self.workspace_id = workspace_id
        self.producer_fn = producer_fn
        super().__init__(
            f"Workspace '{workspace_id}' has expired. "
            f"Re-run the producer to refresh the data."
        )

    def to_llm_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "error": f"Workspace '{self.workspace_id}' has expired (TTL exceeded).",
            "workspace_id": self.workspace_id,
        }
        if self.producer_fn:
            d["suggested_action"] = f"Re-run '{self.producer_fn}' to refresh the data."
        else:
            d["suggested_action"] = (
                "Re-run the producer tool that created this workspace to refresh it."
            )
        return d


class WorkspaceTypeMismatchError(CtxtualError):
    """Raised when a consumer tool is called with a workspace of the wrong type."""

    def __init__(
        self,
        workspace_id: str,
        expected: str,
        actual: str,
        *,
        matching_workspaces: list[str] | None = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.expected = expected
        self.actual = actual
        self.matching_workspaces = matching_workspaces or []
        super().__init__(
            f"Workspace '{workspace_id}' is type '{actual}', "
            f"but this tool expects type '{expected}'."
        )

    def to_llm_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "error": (
                f"Wrong workspace type: '{self.workspace_id}' is "
                f"'{self.actual}', but this tool requires '{self.expected}'."
            ),
            "workspace_id": self.workspace_id,
            "expected_type": self.expected,
            "actual_type": self.actual,
        }
        if self.matching_workspaces:
            d["workspaces_of_correct_type"] = self.matching_workspaces[:20]
            d["suggested_action"] = (
                f"Use one of the '{self.expected}' workspaces listed above, "
                f"or call the appropriate producer to create one."
            )
        else:
            d["suggested_action"] = (
                f"This tool works with '{self.expected}' workspaces. "
                f"Call the producer that creates '{self.expected}' workspaces first."
            )
        return d


class ToolSetNotRegisteredError(CtxtualError):
    """Raised when a producer references an unregistered toolset."""


class PayloadTooLargeError(CtxtualError):
    """Raised when a producer result exceeds the configured max_items limit."""

    def __init__(self, count: int, limit: int) -> None:
        self.count = count
        self.limit = limit
        super().__init__(
            f"Producer returned {count:,} items, which exceeds the limit "
            f"of {limit:,}. Increase max_items on your Ctx instance or filter upstream."
        )

    def to_llm_dict(self) -> dict[str, Any]:
        return {
            "error": (
                f"The result has {self.count:,} items, more than the "
                f"{self.limit:,} allowed."
            ),
            "item_count": self.count,
            "max_items": self.limit,
            "suggested_action": "Call the tool again with a narrower query.",
        }


class InvalidArgumentError(CtxtualError):
    """Raised when a tool argument has the wrong type or format."""

    def __init__(self, argument: str, message: str) -> None:
        self.argument = argument
        super().__init__(f"Invalid '{argument}': {message}")

    def to_llm_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "argument": self.argument,
            "suggested_action": "Call the tool again with a valid value.",
        }


class WorkspaceConflictError(CtxtualError):
    """
    Raised when storing would overwrite a workspace that the write is not
    allowed to replace (different type, different producer, or another
    session's data).
    """

    def __init__(self, workspace_id: str, reason: str) -> None:
        self.workspace_id = workspace_id
        self.reason = reason
        super().__init__(
            f"Refusing to overwrite workspace '{workspace_id}': {reason}"
        )

    def to_llm_dict(self) -> dict[str, Any]:
        return {
            "error": f"Workspace '{self.workspace_id}' already exists: {self.reason}",
            "workspace_id": self.workspace_id,
            "suggested_action": "Choose a different name and try again.",
        }


class ToolExecutionError(CtxtualError):
    """
    Wraps an exception raised inside a consumer tool.

    Returned as a structured error dict to the LLM instead of crashing
    the entire agent loop.
    """

    def __init__(self, tool_name: str, original: Exception) -> None:
        self.tool_name = tool_name
        self.original = original
        super().__init__(
            f"Tool '{tool_name}' failed: {type(original).__name__}: {original}"
        )
