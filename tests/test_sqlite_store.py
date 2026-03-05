"""Tests for SQLiteStore."""

from ctxtual.store.sqlite import SQLiteStore
from ctxtual.types import WorkspaceMeta


class TestSQLiteStoreLifecycle:
    def test_init_and_get_meta(self, sqlite_memory_store: SQLiteStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="papers")
        sqlite_memory_store.init_workspace(meta)
        retrieved = sqlite_memory_store.get_meta("ws_1")
        assert retrieved is not None
        assert retrieved.workspace_id == "ws_1"
        assert retrieved.workspace_type == "papers"

    def test_get_meta_missing(self, sqlite_memory_store: SQLiteStore) -> None:
        assert sqlite_memory_store.get_meta("ghost") is None

    def test_workspace_exists(self, sqlite_memory_store: SQLiteStore) -> None:
        assert sqlite_memory_store.workspace_exists("ws_1") is False
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        sqlite_memory_store.init_workspace(meta)
        assert sqlite_memory_store.workspace_exists("ws_1") is True

    def test_drop_workspace(self, sqlite_memory_store: SQLiteStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        sqlite_memory_store.init_workspace(meta)
        sqlite_memory_store.set("ws_1", "items", [1, 2, 3])
        sqlite_memory_store.drop_workspace("ws_1")
        assert sqlite_memory_store.get_meta("ws_1") is None
        assert sqlite_memory_store.get("ws_1", "items") is None

    def test_list_workspaces(self, sqlite_memory_store: SQLiteStore) -> None:
        for i, t in enumerate(["a", "b", "a"]):
            meta = WorkspaceMeta(
                workspace_id=f"ws_{i}", workspace_type=t, created_at=float(i)
            )
            sqlite_memory_store.init_workspace(meta)
        assert sqlite_memory_store.list_workspaces() == ["ws_0", "ws_1", "ws_2"]
        assert sqlite_memory_store.list_workspaces("a") == ["ws_0", "ws_2"]

    def test_upsert_meta(self, sqlite_memory_store: SQLiteStore) -> None:
        """init_workspace should update item_count on conflict."""
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data", item_count=0)
        sqlite_memory_store.init_workspace(meta)
        meta.item_count = 99
        sqlite_memory_store.init_workspace(meta)
        retrieved = sqlite_memory_store.get_meta("ws_1")
        assert retrieved is not None
        assert retrieved.item_count == 99


class TestSQLiteStoreDataIO:
    def test_set_get(self, sqlite_memory_store: SQLiteStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        sqlite_memory_store.init_workspace(meta)
        sqlite_memory_store.set("ws_1", "key1", {"hello": "world"})
        assert sqlite_memory_store.get("ws_1", "key1") == {"hello": "world"}

    def test_get_default(self, sqlite_memory_store: SQLiteStore) -> None:
        assert sqlite_memory_store.get("ws_1", "missing") is None
        assert sqlite_memory_store.get("ws_1", "missing", default=42) == 42

    def test_overwrite(self, sqlite_memory_store: SQLiteStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        sqlite_memory_store.init_workspace(meta)
        sqlite_memory_store.set("ws_1", "k", "v1")
        sqlite_memory_store.set("ws_1", "k", "v2")
        assert sqlite_memory_store.get("ws_1", "k") == "v2"

    def test_delete_key(self, sqlite_memory_store: SQLiteStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        sqlite_memory_store.init_workspace(meta)
        sqlite_memory_store.set("ws_1", "k", "v")
        sqlite_memory_store.delete_key("ws_1", "k")
        assert sqlite_memory_store.get("ws_1", "k") is None

    def test_set_items_and_get_items(self, sqlite_memory_store: SQLiteStore) -> None:
        meta = WorkspaceMeta(workspace_id="ws_1", workspace_type="data")
        sqlite_memory_store.init_workspace(meta)
        items = [{"a": 1}, {"a": 2}]
        sqlite_memory_store.set_items("ws_1", items)
        assert sqlite_memory_store.get_items("ws_1") == items
        updated = sqlite_memory_store.get_meta("ws_1")
        assert updated is not None
        assert updated.item_count == 2

    def test_get_items_empty(self, sqlite_memory_store: SQLiteStore) -> None:
        assert sqlite_memory_store.get_items("nonexistent") == []


class TestSQLiteStorePersistence:
    def test_data_survives_new_instance(self, tmp_path) -> None:
        db_path = tmp_path / "persist.db"
        store1 = SQLiteStore(db_path)
        meta = WorkspaceMeta(workspace_id="ws_p", workspace_type="data")
        store1.init_workspace(meta)
        store1.set("ws_p", "items", [1, 2, 3])
        store1.close()

        store2 = SQLiteStore(db_path)
        assert store2.get_meta("ws_p") is not None
        assert store2.get("ws_p", "items") == [1, 2, 3]
        store2.close()


class TestSQLiteStoreRepr:
    def test_repr(self, sqlite_memory_store: SQLiteStore) -> None:
        r = repr(sqlite_memory_store)
        assert "SQLiteStore" in r
        assert ":memory:" in r


class TestSQLiteStoreMutations:
    """Tests for append_items, update_item, patch_item, delete_items on SQLiteStore."""

    def _setup_ws(self, store: SQLiteStore) -> str:
        ws_id = "ws_mut"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="data")
        store.init_workspace(meta)
        store.set_items(ws_id, [{"a": 1}, {"a": 2}, {"a": 3}])
        return ws_id

    def test_append_items(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        new_count = sqlite_memory_store.append_items(ws_id, [{"a": 4}, {"a": 5}])
        assert new_count == 5
        items = sqlite_memory_store.get_items(ws_id)
        assert len(items) == 5
        assert items[3] == {"a": 4}
        assert items[4] == {"a": 5}
        assert sqlite_memory_store.count_items(ws_id) == 5

    def test_append_items_empty(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        new_count = sqlite_memory_store.append_items(ws_id, [])
        assert new_count == 3

    def test_append_non_list_raises(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = "ws_dict"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="config")
        sqlite_memory_store.init_workspace(meta)
        sqlite_memory_store.set_items(ws_id, {"key": "value"})
        import pytest

        with pytest.raises(TypeError, match="not list"):
            sqlite_memory_store.append_items(ws_id, [1])

    def test_update_item(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        sqlite_memory_store.update_item(ws_id, 1, {"a": 99})
        assert sqlite_memory_store.get_items(ws_id)[1] == {"a": 99}
        assert sqlite_memory_store.count_items(ws_id) == 3

    def test_update_item_out_of_range(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        import pytest

        with pytest.raises(IndexError, match="out of range"):
            sqlite_memory_store.update_item(ws_id, 99, {"a": 0})

    def test_patch_item(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        sqlite_memory_store.patch_item(ws_id, 0, {"status": "done", "a": 10})
        item = sqlite_memory_store.get_items(ws_id)[0]
        assert item == {"a": 10, "status": "done"}

    def test_patch_item_out_of_range(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        import pytest

        with pytest.raises(IndexError, match="out of range"):
            sqlite_memory_store.patch_item(ws_id, 99, {"x": 1})

    def test_patch_item_non_dict_raises(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = "ws_ints"
        meta = WorkspaceMeta(workspace_id=ws_id, workspace_type="data")
        sqlite_memory_store.init_workspace(meta)
        sqlite_memory_store.set_items(ws_id, [1, 2, 3])
        import pytest

        with pytest.raises(TypeError, match="not dict"):
            sqlite_memory_store.patch_item(ws_id, 0, {"x": 1})

    def test_delete_items(self, sqlite_memory_store: SQLiteStore) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        new_count = sqlite_memory_store.delete_items(ws_id, [0, 2])
        assert new_count == 1
        assert sqlite_memory_store.get_items(ws_id) == [{"a": 2}]
        assert sqlite_memory_store.count_items(ws_id) == 1

    def test_delete_items_empty_indices(
        self, sqlite_memory_store: SQLiteStore
    ) -> None:
        ws_id = self._setup_ws(sqlite_memory_store)
        new_count = sqlite_memory_store.delete_items(ws_id, [])
        assert new_count == 3

    def test_delete_then_append(self, sqlite_memory_store: SQLiteStore) -> None:
        """Ensure append works correctly after delete re-indexes."""
        ws_id = self._setup_ws(sqlite_memory_store)
        sqlite_memory_store.delete_items(ws_id, [1])  # remove {"a": 2}
        sqlite_memory_store.append_items(ws_id, [{"a": 99}])
        items = sqlite_memory_store.get_items(ws_id)
        assert len(items) == 3
        assert items == [{"a": 1}, {"a": 3}, {"a": 99}]

    def test_pagination_after_mutations(
        self, sqlite_memory_store: SQLiteStore
    ) -> None:
        """Ensure get_page works correctly after mutations."""
        ws_id = self._setup_ws(sqlite_memory_store)
        sqlite_memory_store.append_items(ws_id, [{"a": 4}, {"a": 5}])
        page = sqlite_memory_store.get_page(ws_id, offset=2, limit=2)
        assert page == [{"a": 3}, {"a": 4}]
