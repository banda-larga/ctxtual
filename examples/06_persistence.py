"""
Real-world example: Persistent agent with SQLite and workspace mutations.

Scenario: A task management agent that persists across process restarts.
The agent can:
  - Load tasks from an external source (producer)
  - Mark tasks as done, add notes, delete tasks (mutations)
  - Resume after a restart without losing state

This demonstrates:
  - SQLiteStore for crash-resilient persistence
  - Workspace mutations (update_item, patch_item, append_items, delete_items)
  - Deterministic keys for idempotent reloads
  - Custom toolsets for domain-specific operations

Run:
    uv run python examples/06_persistence.py
"""

import tempfile
from pathlib import Path

from ctxtual import Forge, SQLiteStore
from ctxtual.toolset import ToolSet
from ctxtual.utils import paginator, text_search

# Use a temp file so the example is self-contained
DB_PATH = Path(tempfile.mkdtemp()) / "tasks.db"


def create_forge(db_path: Path = DB_PATH) -> Forge:
    """Create a Forge instance with SQLite persistence."""
    return Forge(
        store=SQLiteStore(path=db_path),
        default_ttl=None,  # Tasks never expire
    )


# ── Custom toolset: task operations (mutations) ─────────────────────────

def task_ops(forge: Forge) -> ToolSet:
    """
    Create a ToolSet with task-specific mutation tools.

    These go beyond read-only exploration — they let the agent
    modify task state in place.
    """
    ts = forge.toolset("tasks")
    ts.data_shape = "list"

    @ts.tool(name="tasks_complete")
    def complete_task(workspace_id: str, index: int) -> dict:
        """Mark a task as completed by index.

        Args:
            workspace_id: The tasks workspace to modify.
            index: Zero-based index of the task to complete.
        """
        page = ts.store.get_page(workspace_id, index, 1)
        if not page:
            return {"error": f"No task at index {index}"}

        ts.store.patch_item(workspace_id, index, {"status": "done"})
        task = ts.store.get_page(workspace_id, index, 1)[0]
        return {"message": f"Task '{task['title']}' marked as done", "task": task}

    @ts.tool(name="tasks_add_note")
    def add_note(workspace_id: str, index: int, note: str) -> dict:
        """Add a note to a task.

        Args:
            workspace_id: The tasks workspace to modify.
            index: Zero-based index of the task.
            note: Note text to append.
        """
        page = ts.store.get_page(workspace_id, index, 1)
        if not page:
            return {"error": f"No task at index {index}"}

        task = page[0]
        existing_notes = task.get("notes", [])
        existing_notes.append(note)
        ts.store.patch_item(workspace_id, index, {"notes": existing_notes})
        return {"message": f"Note added to '{task['title']}'", "note": note}

    @ts.tool(name="tasks_add")
    def add_task(workspace_id: str, title: str, priority: str = "medium") -> dict:
        """Add a new task to the workspace.

        Args:
            workspace_id: The tasks workspace to add to.
            title: Title of the new task.
            priority: Priority level (low, medium, high, critical).
        """
        new_task = {
            "title": title,
            "priority": priority,
            "status": "todo",
            "notes": [],
        }
        ts.store.append_items(workspace_id, [new_task])
        count = ts.store.count_items(workspace_id)
        return {"message": f"Task '{title}' added", "total_tasks": count}

    @ts.tool(name="tasks_remove")
    def remove_tasks(workspace_id: str, indices: str) -> dict:
        """Remove tasks by their indices (comma-separated).

        Args:
            workspace_id: The tasks workspace to modify.
            indices: Comma-separated list of zero-based indices to remove.
        """
        idx_list = [int(i.strip()) for i in indices.split(",")]
        ts.store.delete_items(workspace_id, idx_list)
        count = ts.store.count_items(workspace_id)
        return {"message": f"Removed {len(idx_list)} task(s)", "remaining": count}

    return ts


# ── Producer: load tasks ─────────────────────────────────────────────────

def setup_producer(forge: Forge, ops: ToolSet):
    pager  = paginator(forge, "tasks")
    search = text_search(forge, "tasks", fields=["title", "priority", "status"])

    @forge.producer(
        workspace_type="tasks",
        toolsets=[pager, search, ops],
        key="tasks_sprint_{sprint}",  # Deterministic: same sprint = same workspace
    )
    def load_tasks(sprint: str = "current") -> list[dict]:
        """Load sprint tasks. Idempotent: re-calling with same sprint reuses workspace.

        Args:
            sprint: Sprint identifier (e.g., 'current', '2024-w48').
        """
        return [
            {"title": "Implement user authentication",  "priority": "critical", "status": "in_progress", "notes": []},
            {"title": "Write API documentation",        "priority": "high",     "status": "todo",        "notes": []},
            {"title": "Fix pagination bug on /search",  "priority": "high",     "status": "todo",        "notes": []},
            {"title": "Add dark mode toggle",           "priority": "medium",   "status": "todo",        "notes": []},
            {"title": "Refactor database migrations",   "priority": "medium",   "status": "todo",        "notes": []},
            {"title": "Update dependency versions",     "priority": "low",      "status": "todo",        "notes": []},
        ]

    return load_tasks


# ── Simulation ───────────────────────────────────────────────────────────

def simulate():
    print("=" * 70)
    print("PHASE 1: Initial session — load tasks and work on them")
    print("=" * 70)

    forge = create_forge()
    ops = task_ops(forge)
    load_tasks = setup_producer(forge, ops)

    # Load tasks
    ref = load_tasks(sprint="current")
    ws_id = ref["workspace_id"]
    print(f"\nLoaded {ref['item_count']} tasks → {ws_id}")

    # Agent browses tasks
    result = forge.dispatch_tool_call(
        "tasks_paginate", {"workspace_id": ws_id, "page": 0, "size": 10}
    )
    for i, task in enumerate(result["result"]["items"]):
        print(f"  [{i}] [{task['status']:>11}] {task['priority']:>8} | {task['title']}")

    # Agent completes a task
    print("\n[Action] Complete task 0 (user auth)")
    result = forge.dispatch_tool_call(
        "tasks_complete", {"workspace_id": ws_id, "index": 0}
    )
    print(f"  → {result['message']}")

    # Agent adds a note
    print("\n[Action] Add note to task 2 (pagination bug)")
    result = forge.dispatch_tool_call(
        "tasks_add_note",
        {"workspace_id": ws_id, "index": 2, "note": "Root cause: off-by-one in OFFSET calc"},
    )
    print(f"  → {result['message']}")

    # Agent adds a new task
    print("\n[Action] Add new urgent task")
    result = forge.dispatch_tool_call(
        "tasks_add",
        {"workspace_id": ws_id, "title": "Hotfix: login 500 error on Safari", "priority": "critical"},
    )
    print(f"  → {result['message']} (total: {result['total_tasks']})")

    # Agent removes a low-priority task
    print("\n[Action] Remove task 5 (update deps — deprioritized)")
    result = forge.dispatch_tool_call(
        "tasks_remove", {"workspace_id": ws_id, "indices": "5"}
    )
    print(f"  → {result['message']} ({result['remaining']} remaining)")

    forge.close()
    print(f"\n  Session closed. DB saved at: {DB_PATH}")

    # ── Phase 2: Restart ─────────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("PHASE 2: New process — resume from SQLite")
    print("=" * 70)

    forge2 = create_forge()
    ops2 = task_ops(forge2)
    setup_producer(forge2, ops2)  # Re-register (schemas only, won't reload data)

    # Workspaces survived the restart
    workspaces = forge2.list_workspaces()
    print(f"\nWorkspaces found: {workspaces}")

    for ws_id in workspaces:
        meta = forge2.workspace_meta(ws_id)
        print(f"  {ws_id}: {meta.item_count} items, shape={meta.data_shape}")

    # Read back the mutated state
    ws_id = workspaces[0]
    print(f"\nCurrent tasks in {ws_id}:")
    result = forge2.dispatch_tool_call(
        "tasks_paginate", {"workspace_id": ws_id, "page": 0, "size": 10}
    )
    for i, task in enumerate(result["result"]["items"]):
        notes = f" — notes: {task['notes']}" if task.get("notes") else ""
        print(f"  [{i}] [{task['status']:>11}] {task['priority']:>8} | {task['title']}{notes}")

    # Search still works
    result = forge2.dispatch_tool_call(
        "tasks_search", {"workspace_id": ws_id, "query": "critical"}
    )
    print(f"\nSearch 'critical': {result['total_matches']} matches")
    for m in result["matches"]:
        print(f"  • {m['title']} ({m['status']})")

    forge2.close()
    print(f"\n{'=' * 70}")
    print("All mutations persisted across process restarts.")
    print("=" * 70)

    # Cleanup
    DB_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    simulate()
