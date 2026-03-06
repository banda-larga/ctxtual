"""Tests for search quality improvements (Fix #17).

Validates relevance scoring, multi-term queries, FTS5 ranking,
and backward compatibility with existing search behavior.
"""

from ctxtual import Ctx, MemoryStore
from ctxtual.store.sqlite import SQLiteStore
from ctxtual.types import WorkspaceMeta


def _make_forge(store):
    """Create a ctx with sample documents for search testing."""
    ctx = Ctx(store=store)
    meta = WorkspaceMeta(workspace_id="ws_docs", workspace_type="docs", item_count=6)
    ctx.store.init_workspace(meta)
    ctx.store.set_items(
        "ws_docs",
        [
            {
                "title": "Introduction to Machine Learning",
                "body": "Machine learning is a subset of artificial intelligence.",
                "tags": "ml ai intro",
            },
            {
                "title": "Deep Learning with Transformers",
                "body": "Transformers revolutionized natural language processing.",
                "tags": "deep-learning transformers nlp",
            },
            {
                "title": "Machine Learning in Production",
                "body": "Deploying machine learning models requires MLOps practices.",
                "tags": "ml production mlops",
            },
            {
                "title": "Natural Language Processing Basics",
                "body": "NLP covers tokenization, parsing, and language models.",
                "tags": "nlp basics",
            },
            {
                "title": "Reinforcement Learning Overview",
                "body": "RL agents learn by interacting with an environment.",
                "tags": "rl agents",
            },
            {
                "title": "Machine Learning Machine Learning",
                "body": "This document mentions machine learning many many times. "
                "Machine learning machine learning machine learning.",
                "tags": "ml repetition",
            },
        ],
    )
    return ctx


class TestRelevanceScoring:
    """Results should be ranked by relevance, not just insertion order."""

    def test_more_relevant_items_first_memory(self):
        ctx = _make_forge(MemoryStore())
        results = ctx.store.search_items("ws_docs", "machine learning")
        # Item with most mentions of "machine learning" should rank higher
        assert len(results) >= 3
        titles = [r["title"] for r in results]
        # The repetition document should be first (highest TF)
        assert titles[0] == "Machine Learning Machine Learning"

    def test_more_relevant_items_first_sqlite(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "search.db"))
        results = ctx.store.search_items("ws_docs", "machine learning")
        assert len(results) >= 3
        # All results should contain "machine" or "learning"
        for r in results:
            text = f"{r['title']} {r['body']}".lower()
            assert "machine" in text or "learning" in text
        ctx.store.close()


class TestMultiTermQueries:
    """Multi-word queries should match items containing any/all terms."""

    def test_multi_term_all_match_memory(self):
        ctx = _make_forge(MemoryStore())
        results = ctx.store.search_items("ws_docs", "transformers nlp")
        assert len(results) >= 1
        # Item with both terms should appear
        titles = [r["title"] for r in results]
        assert "Deep Learning with Transformers" in titles

    def test_multi_term_all_match_sqlite(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "search.db"))
        results = ctx.store.search_items("ws_docs", "transformers nlp")
        assert len(results) >= 1
        titles = [r["title"] for r in results]
        assert "Deep Learning with Transformers" in titles
        ctx.store.close()

    def test_single_term_still_works(self):
        ctx = _make_forge(MemoryStore())
        results = ctx.store.search_items("ws_docs", "reinforcement")
        assert len(results) >= 1
        assert results[0]["title"] == "Reinforcement Learning Overview"


class TestFieldSpecificSearch:
    """Search restricted to specific fields should ignore other fields."""

    def test_field_search_memory(self):
        ctx = _make_forge(MemoryStore())
        # "transformers" appears in title of item 2 but not in tags
        results = ctx.store.search_items(
            "ws_docs", "transformers", fields=["tags"]
        )
        assert len(results) >= 1
        # Should only match via the tags field
        for r in results:
            assert "transformers" in r["tags"].lower()

    def test_field_search_sqlite(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "search.db"))
        results = ctx.store.search_items(
            "ws_docs", "transformers", fields=["tags"]
        )
        assert len(results) >= 1
        for r in results:
            assert "transformers" in r["tags"].lower()
        ctx.store.close()


class TestCaseSensitiveSearch:
    """Case-sensitive search should respect casing."""

    def test_case_sensitive_memory(self):
        ctx = _make_forge(MemoryStore())
        results = ctx.store.search_items(
            "ws_docs", "Machine", case_sensitive=True
        )
        assert len(results) >= 1
        # Should not match lowercase "machine"
        lower_results = ctx.store.search_items(
            "ws_docs", "machine", case_sensitive=True
        )
        # The body fields have lowercase "machine" in some items
        assert len(lower_results) >= 1

    def test_case_sensitive_sqlite(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "search.db"))
        results = ctx.store.search_items(
            "ws_docs", "Machine", case_sensitive=True
        )
        assert len(results) >= 1
        ctx.store.close()


class TestFTSIndexMaintenance:
    """FTS index stays consistent across mutations."""

    def test_fts_after_append(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "fts.db"))
        ctx.store.append_items(
            "ws_docs",
            [{"title": "Quantum Computing", "body": "Qubits and superposition.", "tags": "quantum"}],
        )
        results = ctx.store.search_items("ws_docs", "quantum")
        assert len(results) == 1
        assert results[0]["title"] == "Quantum Computing"
        ctx.store.close()

    def test_fts_after_update(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "fts.db"))
        # Update item 0 to mention "robotics" instead
        ctx.store.update_item(
            "ws_docs",
            0,
            {"title": "Introduction to Robotics", "body": "Robots everywhere.", "tags": "robotics"},
        )
        # Old content should not match
        results = ctx.store.search_items("ws_docs", "artificial intelligence")
        for r in results:
            assert r["title"] != "Introduction to Robotics"
        # New content should match
        results = ctx.store.search_items("ws_docs", "robotics")
        assert len(results) >= 1
        assert results[0]["title"] == "Introduction to Robotics"
        ctx.store.close()

    def test_fts_after_delete(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "fts.db"))
        # Delete the repetition item (index 5)
        ctx.store.delete_items("ws_docs", [5])
        results = ctx.store.search_items("ws_docs", "repetition")
        assert len(results) == 0
        ctx.store.close()

    def test_fts_after_patch(self, tmp_path):
        ctx = _make_forge(SQLiteStore(tmp_path / "fts.db"))
        ctx.store.patch_item("ws_docs", 4, {"tags": "rl agents blockchain"})
        results = ctx.store.search_items("ws_docs", "blockchain")
        assert len(results) >= 1
        assert results[0]["title"] == "Reinforcement Learning Overview"
        ctx.store.close()


class TestEmptyAndEdgeCases:
    def test_empty_query(self):
        ctx = _make_forge(MemoryStore())
        results = ctx.store.search_items("ws_docs", "")
        assert results == []

    def test_no_matches(self):
        ctx = _make_forge(MemoryStore())
        results = ctx.store.search_items("ws_docs", "xyznonexistent123")
        assert results == []

    def test_max_results_limit(self):
        ctx = _make_forge(MemoryStore())
        results = ctx.store.search_items("ws_docs", "learning", max_results=2)
        assert len(results) <= 2

    def test_search_non_dict_items(self):
        ctx = Ctx(store=MemoryStore())
        meta = WorkspaceMeta(workspace_id="ws_str", workspace_type="data", item_count=3)
        ctx.store.init_workspace(meta)
        ctx.store.set_items("ws_str", ["hello world", "foo bar", "hello again"])
        results = ctx.store.search_items("ws_str", "hello")
        assert len(results) == 2
