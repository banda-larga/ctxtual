"""Tests for MemoryStore and the BaseStore contract."""

from ctxtual import MemoryStore
from ctxtual.types import WorkspaceMeta


class TestMemoryStoreLifecycle:
    def test_init_and_get_meta(self, memory_store: MemoryStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="papers")
        memory_store.init_workspace(meta)
        retrieved = memory_store.get_meta("ws_1")
        assert retrieved is not None
        assert retrieved.workspace_id == "ws_1"
        assert retrieved.workspace_type == "papers"

    def test_get_meta_missing(self, memory_store: MemoryStore) -> None:
        assert memory_store.get_meta("nonexistent") is None

    def test_workspace_exists(self, memory_store: MemoryStore) -> None:
        assert memory_store.workspace_exists("ws_1") is False
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        memory_store.init_workspace(meta)
        assert memory_store.workspace_exists("ws_1") is True

    def test_drop_workspace(self, memory_store: MemoryStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        memory_store.init_workspace(meta)
        memory_store.set("ws_1", "items", [1, 2, 3])
        memory_store.drop_workspace("ws_1")
        assert memory_store.get_meta("ws_1") is None
        assert memory_store.get("ws_1", "items") is None

    def test_drop_nonexistent_no_error(self, memory_store: MemoryStore) -> None:
        memory_store.drop_workspace("ghost")  # should not raise

    def test_list_workspaces_all(self, memory_store: MemoryStore) -> None:
        for i, t in enumerate(["a", "b", "a"]):
            meta = WorkspaceMeta(
                workspace_id=f"ws_{i}", workspace_type=t, created_at=float(i)
            )
            memory_store.init_workspace(meta)
        assert memory_store.list_workspaces() == ["ws_0", "ws_1", "ws_2"]

    def test_list_workspaces_filtered(self, memory_store: MemoryStore) -> None:
        for i, t in enumerate(["papers", "employees", "papers"]):
            meta = WorkspaceMeta(
                workspace_id=f"ws_{i}", workspace_type=t, created_at=float(i)
            )
            memory_store.init_workspace(meta)
        assert memory_store.list_workspaces("papers") == ["ws_0", "ws_2"]
        assert memory_store.list_workspaces("employees") == ["ws_1"]
        assert memory_store.list_workspaces("unknown") == []


class TestMemoryStoreDataIO:
    def test_set_get(self, memory_store: MemoryStore) -> None:
        memory_store.set("ws_1", "key1", {"hello": "world"})
        assert memory_store.get("ws_1", "key1") == {"hello": "world"}

    def test_get_default(self, memory_store: MemoryStore) -> None:
        assert memory_store.get("ws_1", "missing") is None
        assert memory_store.get("ws_1", "missing", default=42) == 42

    def test_overwrite(self, memory_store: MemoryStore) -> None:
        memory_store.set("ws_1", "k", "v1")
        memory_store.set("ws_1", "k", "v2")
        assert memory_store.get("ws_1", "k") == "v2"

    def test_delete_key(self, memory_store: MemoryStore) -> None:
        memory_store.set("ws_1", "k", "v")
        memory_store.delete_key("ws_1", "k")
        assert memory_store.get("ws_1", "k") is None

    def test_delete_key_nonexistent(self, memory_store: MemoryStore) -> None:
        memory_store.delete_key("ghost_ws", "ghost_key")  # no error

    def test_set_items_and_get_items(self, memory_store: MemoryStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        memory_store.init_workspace(meta)
        items = [1, 2, 3, 4, 5]
        memory_store.set_items("ws_1", items)
        assert memory_store.get_items("ws_1") == items
        # Check item_count was updated
        updated_meta = memory_store.get_meta("ws_1")
        assert updated_meta is not None
        assert updated_meta.item_count == 5

    def test_get_items_empty(self, memory_store: MemoryStore) -> None:
        assert memory_store.get_items("nonexistent") == []


class TestMemoryStoreRepr:
    def test_repr(self, memory_store: MemoryStore) -> None:
        r = repr(memory_store)
        assert "MemoryStore" in r
        assert "workspaces=0" in r


class TestMemoryStoreMutations:
    """Tests for append_items, update_item, patch_item, delete_items."""

    def _setup_ws(self, store: MemoryStore) -> str:
        ws_id = "ws_mut"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="data")
        store.init_workspace(meta)
        store.set_items(ws_id, [{"a": 1}, {"a": 2}, {"a": 3}])
        return ws_id

    def test_append_items(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        new_count = memory_store.append_items(ws_id, [{"a": 4}, {"a": 5}])
        assert new_count == 5
        assert memory_store.get_items(ws_id) == [
            {"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5},
        ]
        assert memory_store.count_items(ws_id) == 5

    def test_append_items_empty_list(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        new_count = memory_store.append_items(ws_id, [])
        assert new_count == 3

    def test_append_items_non_list_raises(self, memory_store: MemoryStore) -> None:
        ws_id = "ws_dict"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="config")
        memory_store.init_workspace(meta)
        memory_store.set_items(ws_id, {"key": "value"})
        import pytest

        with pytest.raises(TypeError, match="not list"):
            memory_store.append_items(ws_id, [1])

    def test_update_item(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        memory_store.update_item(ws_id, 1, {"a": 99})
        assert memory_store.get_items(ws_id)[1] == {"a": 99}
        assert memory_store.count_items(ws_id) == 3

    def test_update_item_out_of_range(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        import pytest

        with pytest.raises(IndexError, match="out of range"):
            memory_store.update_item(ws_id, 99, {"a": 0})

    def test_update_item_non_list_raises(self, memory_store: MemoryStore) -> None:
        ws_id = "ws_dict"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="config")
        memory_store.init_workspace(meta)
        memory_store.set_items(ws_id, {"key": "value"})
        import pytest

        with pytest.raises(TypeError, match="not list"):
            memory_store.update_item(ws_id, 0, "x")

    def test_patch_item(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        memory_store.patch_item(ws_id, 0, {"status": "done", "a": 10})
        item = memory_store.get_items(ws_id)[0]
        assert item == {"a": 10, "status": "done"}

    def test_patch_item_out_of_range(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        import pytest

        with pytest.raises(IndexError, match="out of range"):
            memory_store.patch_item(ws_id, 99, {"x": 1})

    def test_patch_item_non_dict_raises(self, memory_store: MemoryStore) -> None:
        ws_id = "ws_ints"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="data")
        memory_store.init_workspace(meta)
        memory_store.set_items(ws_id, [1, 2, 3])
        import pytest

        with pytest.raises(TypeError, match="not dict"):
            memory_store.patch_item(ws_id, 0, {"x": 1})

    def test_delete_items(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        new_count = memory_store.delete_items(ws_id, [0, 2])
        assert new_count == 1
        assert memory_store.get_items(ws_id) == [{"a": 2}]
        assert memory_store.count_items(ws_id) == 1

    def test_delete_items_empty_indices(self, memory_store: MemoryStore) -> None:
        ws_id = self._setup_ws(memory_store)
        new_count = memory_store.delete_items(ws_id, [])
        assert new_count == 3

    def test_delete_items_non_list_raises(self, memory_store: MemoryStore) -> None:
        ws_id = "ws_dict"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="config")
        memory_store.init_workspace(meta)
        memory_store.set_items(ws_id, {"key": "value"})
        import pytest

        with pytest.raises(TypeError, match="not list"):
            memory_store.delete_items(ws_id, [0])
