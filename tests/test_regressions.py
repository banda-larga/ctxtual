"""
Regression tests for issues found in the 0.1.x code review.

Each class pins one bug (or one guarantee the README makes) so it can't
silently come back.
"""

import asyncio
import json
import sqlite3
import threading
import time

import pytest

from ctxtual import (
    ConsumerContext,
    Ctx,
    MemoryStore,
    SQLiteStore,
    WorkspaceConflictError,
)
from ctxtual.integrations.anthropic import handle_tool_use
from ctxtual.integrations.openai import handle_tool_calls
from ctxtual.utils import (
    filter_set,
    kv_reader,
    paginator,
    pipeline,
    text_content,
    text_search,
)

PAPERS = [
    {"title": "Attention is all you need", "abstract": "transformers", "year": 2017},
    {"title": "BERT", "abstract": "bidirectional transformers", "year": 2018},
    {"title": "ResNet", "abstract": "residual networks", "year": 2015},
]


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        s = MemoryStore()
    else:
        s = SQLiteStore(tmp_path / "test.db")
    yield s
    s.close()


def _papers_ctx(store, **toolset_kwargs):
    ctx = Ctx(store=store)

    @ctx.producer(
        workspace_type="papers",
        toolsets=[
            paginator(ctx),
            text_search(ctx, **toolset_kwargs),
            filter_set(ctx),
            pipeline(ctx),
        ],
    )
    def load_papers(query: str = "") -> list[dict]:
        """Load papers."""
        return [dict(p) for p in PAPERS]

    return ctx, load_papers


# ── Search ───────────────────────────────────────────────────────────────


class TestSearchDoesNotMatchInternals:
    def test_query_matching_workspace_type_returns_nothing(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]  # "papers_..."
        r = ctx.dispatch_tool_call("papers_search", {"workspace_id": ws, "query": "papers"})
        assert r["returned"] == 0

    def test_query_matching_item_index_returns_nothing(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="words", toolsets=[text_search(ctx)])
        def words() -> list[dict]:
            return [{"w": "alpha"}, {"w": "beta"}, {"w": "gamma"}]

        ws = words()["workspace_id"]
        for q in ("1", "2", "w"):  # an index, an index, a field name
            r = ctx.dispatch_tool_call("words_search", {"workspace_id": ws, "query": q})
            assert r["matches"] == [], q

    def test_query_with_quotes_does_not_error(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]
        r = ctx.dispatch_tool_call(
            "papers_search", {"workspace_id": ws, "query": 'he said "BERT'}
        )
        assert "error" not in r
        assert r["matches"][0]["title"] == "BERT"

    def test_like_wildcards_are_literal(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]
        r = ctx.dispatch_tool_call("papers_search", {"workspace_id": ws, "query": "%"})
        assert r["returned"] == 0

    def test_field_restricted_search_ranks_and_restricts(self, store):
        ctx, load = _papers_ctx(store, fields=["title"])
        ws = load()["workspace_id"]
        # "transformers" only appears in abstracts — must not match.
        r = ctx.dispatch_tool_call(
            "papers_search", {"workspace_id": ws, "query": "transformers"}
        )
        assert r["returned"] == 0
        r = ctx.dispatch_tool_call("papers_search", {"workspace_id": ws, "query": "bert"})
        assert [m["title"] for m in r["matches"]] == ["BERT"]

    def test_multi_term_query_matches_any_term(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]
        r = ctx.dispatch_tool_call(
            "papers_search", {"workspace_id": ws, "query": "residual quantum"}
        )
        assert [m["title"] for m in r["matches"]] == ["ResNet"]

    def test_mid_word_fragment_still_found(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]
        r = ctx.dispatch_tool_call("papers_search", {"workspace_id": ws, "query": "esNe"})
        assert [m["title"] for m in r["matches"]] == ["ResNet"]

    def test_search_is_scoped_to_workspace(self, store):
        ctx, load = _papers_ctx(store)
        ws1 = load()["workspace_id"]
        store_ctx = ctx

        @store_ctx.producer(workspace_type="papers", toolsets=[])
        def other() -> list[dict]:
            return [{"title": "Unrelated BERT clone"}]

        other()
        r = ctx.dispatch_tool_call("papers_search", {"workspace_id": ws1, "query": "clone"})
        assert r["returned"] == 0

    def test_symbol_terms_are_not_prefix_matched(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="langs", toolsets=[text_search(ctx)])
        def langs() -> list[dict]:
            return [{"t": "Intro to C++"}, {"t": "cats and dogs"}, {"t": "C# basics"}]

        ws = langs()["workspace_id"]
        for q, want in (("C++", "Intro to C++"), ("C#", "C# basics")):
            r = ctx.dispatch_tool_call("langs_search", {"workspace_id": ws, "query": q})
            assert [m["t"] for m in r["matches"]][0] == want
            assert "cats and dogs" not in [m["t"] for m in r["matches"]]

    def test_nul_in_query_is_harmless(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]
        r = ctx.dispatch_tool_call(
            "papers_search", {"workspace_id": ws, "query": "BE\x00RT"}
        )
        assert "error" not in r

    def test_has_more_reports_truncation(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]
        r = ctx.dispatch_tool_call(
            "papers_search",
            {"workspace_id": ws, "query": "transformers", "max_results": 1},
        )
        assert r["returned"] == 1
        assert r["has_more"] is True


class TestSQLiteFtsMigration:
    def test_v1_database_is_reindexed(self, tmp_path):
        path = tmp_path / "old.db"
        # Build a v1-layout database by hand.
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE cf_meta (workspace_id TEXT PRIMARY KEY, workspace_type TEXT NOT NULL,
                created_at REAL NOT NULL, producer_fn TEXT NOT NULL DEFAULT '',
                producer_kwargs TEXT NOT NULL DEFAULT '{}', item_count INTEGER NOT NULL DEFAULT 0,
                extra TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE cf_items (workspace_id TEXT NOT NULL, idx INTEGER NOT NULL,
                value TEXT NOT NULL, PRIMARY KEY (workspace_id, idx));
            CREATE VIRTUAL TABLE cf_fts USING fts5(workspace_id, idx, content,
                tokenize='porter unicode61');
            """
        )
        conn.execute(
            "INSERT INTO cf_meta (workspace_id, workspace_type, created_at, item_count) "
            "VALUES ('papers_old', 'papers', ?, 1)",
            (time.time(),),
        )
        conn.execute(
            "INSERT INTO cf_items VALUES ('papers_old', 0, ?)",
            (json.dumps({"title": "Legacy document"}),),
        )
        conn.execute("INSERT INTO cf_fts VALUES ('papers_old', 0, 'Legacy document')")
        conn.commit()
        conn.close()

        store = SQLiteStore(path)
        assert store.search_items("papers_old", "legacy") == [{"title": "Legacy document"}]
        assert store.search_items("papers_old", "papers") == []
        meta = store.get_meta("papers_old")
        assert meta.session_id is None and meta.data_shape == ""
        store.close()


# ── Shapes ───────────────────────────────────────────────────────────────


class TestMixedShapeToolsets:
    def test_list_tools_work_when_text_tools_share_the_name(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="doc", toolsets=[paginator(ctx), text_content(ctx)])
        def fetch(as_text: bool = False):
            return "some text" if as_text else [{"a": 1}, {"a": 2}]

        ref = fetch()
        tools = [t.split("(")[0] for t in ref["available_tools"]]
        assert "doc_paginate" in tools and "doc_read_page" not in tools
        page = ctx.dispatch_tool_call("doc_paginate", {"workspace_id": ref["workspace_id"]})
        assert page["result"]["total"] == 2

        ref = fetch(as_text=True)
        tools = [t.split("(")[0] for t in ref["available_tools"]]
        assert "doc_read_page" in tools and "doc_paginate" not in tools
        page = ctx.dispatch_tool_call("doc_read_page", {"workspace_id": ref["workspace_id"]})
        assert page["result"]["text"] == "some text"


# ── Consumers ────────────────────────────────────────────────────────────


def _pipeline_ctx(store=None):
    ctx = Ctx(store=store or MemoryStore())

    @ctx.producer(workspace_type="raw", toolsets=[paginator(ctx)])
    def ingest() -> list[int]:
        """Ingest raw numbers."""
        return [1, 2, 3]

    @ctx.producer(workspace_type="other", toolsets=[])
    def other() -> list[int]:
        return [9]

    @ctx.consumer(
        workspace_type="raw",
        produces="clean",
        produces_toolsets=[paginator(ctx)],
        meta={"agent": "cleaner"},
    )
    def clean(workspace_id: str, factor: int, ctx: ConsumerContext):
        """Multiply every item by factor."""
        return ctx.emit([x * factor for x in ctx.get_items()])

    return ctx, ingest, other


class TestConsumers:
    def test_consumer_is_dispatchable(self, store):
        ctx, ingest, _ = _pipeline_ctx(store)
        ws = ingest()["workspace_id"]
        ref = ctx.dispatch_tool_call("clean", {"workspace_id": ws, "factor": 2})
        assert ref["workspace_type"] == "clean"
        assert ctx.store.get_items(ref["workspace_id"]) == [2, 4, 6]
        assert ref["metadata"] == {"agent": "cleaner"}
        assert any(t.startswith("clean_paginate") for t in ref["available_tools"])

    def test_consumer_schema_exported_without_ctx_param(self):
        ctx, _, _ = _pipeline_ctx()
        schema = next(t for t in ctx.get_tools() if t["function"]["name"] == "clean")
        params = schema["function"]["parameters"]
        assert set(params["properties"]) == {"workspace_id", "factor"}
        assert params["required"] == ["workspace_id", "factor"]
        assert "clean" in ctx.system_prompt()

    def test_consumer_rejects_wrong_workspace_type(self):
        ctx, _, other = _pipeline_ctx()
        ws = other()["workspace_id"]
        r = ctx.dispatch_tool_call("clean", {"workspace_id": ws, "factor": 2})
        assert r["expected_type"] == "raw" and r["actual_type"] == "other"

    def test_consumer_rejects_missing_workspace(self):
        ctx, _, _ = _pipeline_ctx()
        r = ctx.dispatch_tool_call("clean", {"workspace_id": "nope", "factor": 2})
        assert "not found" in r["error"]

    def test_emit_ttl_none_means_no_expiry(self):
        ctx = Ctx(default_ttl=10)

        @ctx.producer(workspace_type="raw", toolsets=[])
        def ingest() -> list[int]:
            return [1]

        @ctx.consumer(workspace_type="raw", produces="d")
        def derive(workspace_id: str, ctx: ConsumerContext):
            return ctx.emit([1], ttl=None)

        @ctx.consumer(workspace_type="raw", produces="d", name="derive_default")
        def derive2(workspace_id: str, ctx: ConsumerContext):
            return ctx.emit([1])

        ws = ingest()["workspace_id"]
        assert ctx.workspace_meta(derive(workspace_id=ws)["workspace_id"]).ttl is None
        assert ctx.workspace_meta(derive2(workspace_id=ws)["workspace_id"]).ttl == 10


# ── Dispatch robustness ──────────────────────────────────────────────────


class TestDispatchNeverRaisesOnLlmMistakes:
    @pytest.mark.parametrize("bad", [["x"], {"a": 1}, 42, "", None])
    def test_non_string_workspace_id(self, store, bad):
        ctx, load = _papers_ctx(store)
        load()
        r = ctx.dispatch_tool_call("papers_paginate", {"workspace_id": bad})
        assert "workspace_id" in r["error"]

    def test_non_string_workspace_id_for_consumer(self):
        ctx, ingest, _ = _pipeline_ctx()
        ingest()
        r = ctx.dispatch_tool_call("clean", {"workspace_id": ["x"], "factor": 2})
        assert "workspace_id" in r["error"]


    def test_producer_exception_becomes_error_dict(self):
        ctx = Ctx()

        @ctx.producer(workspace_type="x", toolsets=[])
        def boom() -> list:
            raise ConnectionError("upstream down")

        r = ctx.dispatch_tool_call("boom", {})
        assert r["error"] == "ConnectionError: upstream down"

    def test_payload_too_large_becomes_error_dict(self):
        ctx = Ctx(max_items=2)

        @ctx.producer(workspace_type="x", toolsets=[])
        def big() -> list:
            return [1, 2, 3]

        r = ctx.dispatch_tool_call("big", {})
        assert r["item_count"] == 3 and r["max_items"] == 2

    def test_bad_arguments_report_expected_parameters(self):
        ctx, load = _papers_ctx(MemoryStore())
        r = ctx.dispatch_tool_call("load_papers", {"bogus": 1})
        assert "Invalid arguments" in r["error"]
        assert "query" in r["expected_parameters"]
        r = ctx.dispatch_tool_call("papers_paginate", {"page": 0})
        assert r["required"] == ["workspace_id"]

    def test_json_string_arguments_are_parsed(self):
        ctx, load = _papers_ctx(MemoryStore())
        ws = load()["workspace_id"]
        r = ctx.dispatch_tool_call("papers_count", json.dumps({"workspace_id": ws}))
        assert r["count"] == 3
        r = ctx.dispatch_tool_call("papers_count", "{not json")
        assert "not valid JSON" in r["error"]

    def test_unsafe_ctx_propagates_producer_errors(self):
        ctx = Ctx(safe=False)

        @ctx.producer(workspace_type="x", toolsets=[])
        def boom() -> list:
            raise ValueError("bad")

        with pytest.raises(ValueError):
            ctx.dispatch_tool_call("boom", {})

    def test_key_template_must_reference_parameters(self):
        ctx = Ctx()
        with pytest.raises(ValueError, match="not parameters"):

            @ctx.producer(workspace_type="x", toolsets=[], key="x_{missing}")
            def f(query: str) -> list:
                return []


class TestAsync:
    def _ctx(self):
        ctx = Ctx()
        ts = ctx.toolset("a")

        @ts.tool(name="a_first", output_hint="ok")
        async def first(workspace_id: str) -> int:
            await asyncio.sleep(0)
            return ts.store.get_items(workspace_id)[0]

        @ctx.producer(workspace_type="a", toolsets=[ts])
        async def aload(n: int = 3) -> list[int]:
            await asyncio.sleep(0)
            return list(range(n))

        return ctx

    def test_sync_dispatch_runs_async_producer(self):
        ctx = self._ctx()
        ref = ctx.dispatch_tool_call("aload", {"n": 2})
        assert ref["item_count"] == 2
        r = ctx.dispatch_tool_call("a_first", {"workspace_id": ref["workspace_id"]})
        assert r == {"result": 0, "_hint": "ok"}

    def test_adispatch(self):
        ctx = self._ctx()

        async def main():
            ref = await ctx.adispatch_tool_call("aload", {"n": 2})
            missing = await ctx.adispatch_tool_call("a_first", {"workspace_id": "nope"})
            return ref, missing

        ref, missing = asyncio.run(main())
        assert ref["item_count"] == 2
        assert "not found" in missing["error"]

    def test_sync_dispatch_inside_running_loop_raises_clearly(self):
        ctx = self._ctx()

        async def main():
            ctx.dispatch_tool_call("aload", {})

        with pytest.raises(RuntimeError, match="adispatch_tool_call"):
            asyncio.run(main())

    def test_session_propagates_into_async_producer(self):
        ctx = self._ctx()
        ref = ctx.dispatch_tool_call("aload", {}, session_id="s1")
        assert ctx.workspace_meta(ref["workspace_id"]).session_id == "s1"


# ── Access tracking, LRU, TTL ────────────────────────────────────────────


class TestAccessTracking:
    def test_tool_access_updates_last_accessed(self, store):
        ctx, load = _papers_ctx(store)
        ws = load()["workspace_id"]
        before = store.get_meta(ws).last_accessed_at
        time.sleep(0.01)
        ctx.dispatch_tool_call("papers_count", {"workspace_id": ws})
        assert store.get_meta(ws).last_accessed_at > before

    def test_memory_eviction_is_lru(self):
        store = MemoryStore(max_workspaces=2)
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="t", toolsets=[paginator(ctx)])
        def mk(i: int) -> list[int]:
            return [i]

        a = mk(1)["workspace_id"]
        time.sleep(0.01)
        b = mk(2)["workspace_id"]
        time.sleep(0.01)
        ctx.dispatch_tool_call("t_count", {"workspace_id": a})  # a is now recent
        time.sleep(0.01)
        mk(3)
        assert a in store.list_workspaces()
        assert b not in store.list_workspaces()

    def test_reproducing_a_keyed_workspace_resets_ttl(self, store):
        ctx = Ctx(store=store)

        # A TTL with a wide margin: slow CI runners took >50 ms per call.
        @ctx.producer(workspace_type="inv", toolsets=[], key="inv_{wh}", ttl=0.5)
        def sync(wh: str) -> list[int]:
            return [1]

        sync(wh="eu")
        first = store.get_meta("inv_eu").created_at
        time.sleep(0.6)
        assert store.get_meta("inv_eu").is_expired
        sync(wh="eu")
        meta = store.get_meta("inv_eu")
        assert meta.created_at > first and not meta.is_expired


# ── Overwrite protection & sessions ──────────────────────────────────────


class TestOverwriteProtection:
    def _ctx(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="secret", toolsets=[paginator(ctx)])
        def sec() -> list[dict]:
            return [{"pw": "hunter2"}]

        @ctx.producer(workspace_type="pub", toolsets=[paginator(ctx), pipeline(ctx)])
        def pub() -> list[dict]:
            return [{"x": 1}, {"x": 2}]

        @ctx.producer(workspace_type="pub", toolsets=[], key="{name}")
        def named(name: str) -> list[dict]:
            return [{"x": 0}]

        return ctx, sec, pub, named

    def test_save_as_cannot_overwrite_other_workspaces(self, store):
        ctx, sec, pub, _ = self._ctx(store)
        sid, pid = sec()["workspace_id"], pub()["workspace_id"]
        for target in (sid, pid):
            r = ctx.dispatch_tool_call(
                "pub_pipe",
                {"workspace_id": pid, "steps": [{"limit": 1}], "save_as": target},
            )
            assert r["workspace_id"] == f"saved_{target}"
        assert store.get_items(sid) == [{"pw": "hunter2"}]
        assert store.get_meta(sid).workspace_type == "secret"
        assert len(store.get_items(pid)) == 2

    def test_save_as_cannot_squat_a_producer_key(self, store):
        ctx, _, pub, named = self._ctx(store)
        pid = pub()["workspace_id"]
        ctx.dispatch_tool_call(
            "pub_pipe", {"workspace_id": pid, "steps": [], "save_as": "report"}
        )
        ref = ctx.dispatch_tool_call("named", {"name": "report"})
        assert ref["workspace_id"] == "report"
        assert store.get_items("report") == [{"x": 0}]
        # Even a key template that produces the saved_ prefix wins over a
        # pipeline result.
        ref = ctx.dispatch_tool_call("named", {"name": "saved_report"})
        assert store.get_items(ref["workspace_id"]) == [{"x": 0}]

    def test_save_as_can_replace_its_own_result(self, store):
        ctx, _, pub, _ = self._ctx(store)
        pid = pub()["workspace_id"]
        for n in (1, 2):
            r = ctx.dispatch_tool_call(
                "pub_pipe",
                {"workspace_id": pid, "steps": [{"limit": n}], "save_as": "top"},
            )
            assert r["status"] == "pipeline_result_saved"
        assert store.get_items("saved_top") == [{"x": 1}, {"x": 2}]
        assert store.get_meta("saved_top").workspace_type == "pub"

    def test_save_as_name_is_validated(self, store):
        ctx, _, pub, _ = self._ctx(store)
        pid = pub()["workspace_id"]
        for bad in ("../../etc", "foo\n", "a@b", ""):
            r = ctx.dispatch_tool_call(
                "pub_pipe",
                {"workspace_id": pid, "steps": [], "save_as": bad},
            )
            assert "error" in r, bad

    def test_producer_key_cannot_overwrite_other_producer(self, store):
        ctx, sec, pub, named = self._ctx(store)
        pid = pub()["workspace_id"]
        sid = sec()["workspace_id"]
        for target in (pid, sid):
            r = ctx.dispatch_tool_call("named", {"name": target})
            assert "already exists" in r["error"]
        with pytest.raises(WorkspaceConflictError):
            named(name=sid)


class TestSessions:
    def _ctx(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(
            workspace_type="notes",
            toolsets=[paginator(ctx), pipeline(ctx)],
            key="notes_{topic}",
        )
        def notes(topic: str) -> list[dict]:
            return [{"topic": topic, "owner": "x"}]

        @ctx.producer(workspace_type="notes", toolsets=[])
        def shared() -> list[dict]:
            return [{"topic": "public"}]

        return ctx, notes, shared

    def test_other_sessions_workspaces_are_invisible(self, store):
        ctx, notes, _ = self._ctx(store)
        a = ctx.dispatch_tool_call("notes", {"topic": "ml"}, session_id="alice")
        ws_a = a["workspace_id"]
        r = ctx.dispatch_tool_call(
            "notes_paginate", {"workspace_id": ws_a}, session_id="bob"
        )
        assert "not found" in r["error"]
        assert ws_a not in r.get("available_workspaces", [])
        ok = ctx.dispatch_tool_call(
            "notes_paginate", {"workspace_id": ws_a}, session_id="alice"
        )
        assert ok["result"]["total"] == 1

    def test_error_lists_only_own_workspaces(self, store):
        ctx, notes, _ = self._ctx(store)
        ws_a = ctx.dispatch_tool_call("notes", {"topic": "a"}, session_id="alice")
        ws_b = ctx.dispatch_tool_call("notes", {"topic": "b"}, session_id="bob")
        r = ctx.dispatch_tool_call(
            "notes_paginate", {"workspace_id": "nope"}, session_id="bob"
        )
        assert r["available_workspaces"] == [ws_b["workspace_id"]]
        assert ws_a["workspace_id"] not in json.dumps(r)

    def test_same_key_in_two_sessions_does_not_collide(self, store):
        ctx, notes, _ = self._ctx(store)
        a = ctx.dispatch_tool_call("notes", {"topic": "ml"}, session_id="alice")
        b = ctx.dispatch_tool_call("notes", {"topic": "ml"}, session_id="bob")
        assert a["workspace_id"] != b["workspace_id"]
        assert a["workspace_id"].startswith("notes_ml@")
        assert "alice" not in a["workspace_id"]
        # Same session, same key → idempotent.
        a2 = ctx.dispatch_tool_call("notes", {"topic": "ml"}, session_id="alice")
        assert a2["workspace_id"] == a["workspace_id"]

    def test_global_workspaces_are_shared_read_only(self, store):
        ctx, _, shared = self._ctx(store)
        ws = shared()["workspace_id"]  # no session → global
        r = ctx.dispatch_tool_call(
            "notes_paginate", {"workspace_id": ws}, session_id="alice"
        )
        assert r["result"]["total"] == 1
        # save_as inside a session is namespaced, so it lands next to the
        # shared workspace instead of replacing it.
        r = ctx.dispatch_tool_call(
            "notes_pipe",
            {"workspace_id": ws, "steps": [{"limit": 0}], "save_as": ws},
            session_id="alice",
        )
        assert r["workspace_id"] != ws
        assert store.get_items(ws) == [{"topic": "public"}]
        # Unscoped code can't clobber it through save_as either.
        ctx.dispatch_tool_call(
            "notes_pipe", {"workspace_id": ws, "steps": [{"limit": 0}], "save_as": ws}
        )
        assert store.get_items(ws) == [{"topic": "public"}]

    def test_save_as_is_namespaced(self, store):
        ctx, notes, _ = self._ctx(store)
        a = ctx.dispatch_tool_call("notes", {"topic": "x"}, session_id="alice")
        b = ctx.dispatch_tool_call("notes", {"topic": "x"}, session_id="bob")
        ra = ctx.dispatch_tool_call(
            "notes_pipe",
            {"workspace_id": a["workspace_id"], "steps": [], "save_as": "mine"},
            session_id="alice",
        )
        rb = ctx.dispatch_tool_call(
            "notes_pipe",
            {"workspace_id": b["workspace_id"], "steps": [], "save_as": "mine"},
            session_id="bob",
        )
        assert ra["workspace_id"] != rb["workspace_id"]

    def test_context_manager_scope(self, store):
        ctx, notes, _ = self._ctx(store)
        with ctx.session("carol"):
            ref = notes(topic="z")
        assert store.get_meta(ref["workspace_id"]).session_id == "carol"

    def test_consumer_get_items_respects_session(self, store):
        ctx, notes, _ = self._ctx(store)

        @ctx.consumer(workspace_type="notes", produces="notes")
        def peek(workspace_id: str, other_id: str, ctx: ConsumerContext):
            return ctx.get_items(other_id)

        a = ctx.dispatch_tool_call("notes", {"topic": "a"}, session_id="alice")
        b = ctx.dispatch_tool_call("notes", {"topic": "b"}, session_id="bob")
        r = ctx.dispatch_tool_call(
            "peek",
            {"workspace_id": b["workspace_id"], "other_id": a["workspace_id"]},
            session_id="bob",
        )
        assert "not found" in r["error"]


# ── Result caps ──────────────────────────────────────────────────────────


class TestResultCaps:
    def _ctx(self, store, n=500):
        ctx = Ctx(store=store)

        @ctx.producer(
            workspace_type="rows",
            toolsets=[paginator(ctx), filter_set(ctx), pipeline(ctx)],
        )
        def rows() -> list[dict]:
            return [{"i": i, "even": i % 2 == 0, "g": i % 120} for i in range(n)]

        return ctx, rows()["workspace_id"]

    def test_paginate_size_is_clamped(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call("rows_paginate", {"workspace_id": ws, "size": 10_000})
        assert len(r["result"]["items"]) == 100

    def test_get_slice_is_clamped_and_says_so(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call(
            "rows_get_slice", {"workspace_id": ws, "start": 0, "end": 10_000}
        )
        assert len(r["items"]) == 100 and r["truncated"] is True

    def test_filter_by_is_capped_with_true_total(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call(
            "rows_filter_by", {"workspace_id": ws, "field": "even", "value": True}
        )
        assert r["total_matches"] == 250
        assert r["returned"] == 50 and len(r["results"]) == 50
        assert r["truncated"] is True

    def test_pipe_without_limit_is_capped(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call("rows_pipe", {"workspace_id": ws, "steps": []})
        assert len(r["items"]) == 50 and r["total"] == 500 and r["truncated"]

    def test_group_by_results_are_capped(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call(
            "rows_aggregate", {"workspace_id": ws, "group_by": "g"}
        )
        assert len(r["groups"]) == 50 and r["group_count"] == 120

    def test_save_as_keeps_full_result(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call(
            "rows_pipe", {"workspace_id": ws, "steps": [], "save_as": "all_rows"}
        )
        assert r["item_count"] == 500

    def test_kv_value_is_previewed_when_huge(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="cfg", toolsets=[kv_reader(ctx, max_value_chars=100)])
        def cfg() -> dict:
            return {"small": 1, "big": list(range(1000))}

        ws = cfg()["workspace_id"]
        assert ctx.dispatch_tool_call("cfg_get_value", {"workspace_id": ws, "key": "small"}) == 1
        r = ctx.dispatch_tool_call("cfg_get_value", {"workspace_id": ws, "key": "big"})
        assert r["truncated"] and len(r["preview"]) == 100 and r["length"] == 1000

    def test_sort_by_reports_more(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call("rows_sort_by", {"workspace_id": ws, "field": "i"})
        assert r["returned"] == 20 and r["has_more"] is True

    def test_values_metric_is_capped(self, store):
        ctx, ws = self._ctx(store)
        r = ctx.dispatch_tool_call(
            "rows_aggregate", {"workspace_id": ws, "metrics": {"v": "values:i"}}
        )
        v = r["result"]["v"]
        assert v["truncated"] and v["total"] == 500 and len(v["values"]) == 100

    def test_get_keys_is_capped(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="cfg", toolsets=[kv_reader(ctx)])
        def cfg() -> dict:
            return {f"k{i}": i for i in range(5000)}

        ws = cfg()["workspace_id"]
        r = ctx.dispatch_tool_call("cfg_get_keys", {"workspace_id": ws})["result"]
        assert len(r["keys"]) == 200 and r["count"] == 5000 and r["has_more"]

    def test_dispatch_backstop_bounds_custom_tools(self, store):
        ctx = Ctx(store=store, max_result_chars=1_000)
        ts = ctx.toolset("blob")

        @ts.tool(name="blob_all")
        def dump(workspace_id: str) -> list:
            return ts.store.get_items(workspace_id)

        @ctx.producer(workspace_type="blob", toolsets=[ts])
        def blob() -> list[str]:
            return ["x" * 100] * 100

        ws = blob()["workspace_id"]
        r = ctx.dispatch_tool_call("blob_all", {"workspace_id": ws})
        assert r["truncated"] and r["result_chars"] > 1_000
        assert len(json.dumps(r)) < 3_000
        unlimited = Ctx(store=store, max_result_chars=None)
        assert len(dump(ws)) == 100 and unlimited.max_result_chars is None

    def test_text_page_size_is_clamped(self, store):
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="page", toolsets=[text_content(ctx)])
        def page() -> str:
            return "x" * 50_000

        ws = page()["workspace_id"]
        r = ctx.dispatch_tool_call(
            "page_read_page", {"workspace_id": ws, "chars_per_page": 10**9}
        )
        assert len(r["result"]["text"]) == 20_000


class TestRegexGuard:
    def _ctx(self, **kw):
        ctx = Ctx()

        @ctx.producer(workspace_type="r", toolsets=[pipeline(ctx, **kw)])
        def rows() -> list[dict]:
            return [{"s": "a" * 26 + "!"}, {"s": "abc"}]

        return ctx, rows()["workspace_id"]

    def _run(self, ctx, ws, pattern):
        return ctx.dispatch_tool_call(
            "r_pipe",
            {"workspace_id": ws, "steps": [{"filter": {"s": {"$regex": pattern}}}]},
        )

    @pytest.mark.parametrize(
        "pattern",
        [
            "^(a+)+$",
            "((a+))+$",  # extra nesting
            "((a)+)+$",
            "(a|a)+$",  # overlapping alternation
            "(?:a|a)*$",
            "(a?)+$",
            "(.*a){12}",  # unbounded inside a bounded repeat
            "(a)\\1",  # backreference
            "a*a*a*a*a*b",  # too many unbounded repeats
        ],
    )
    def test_dangerous_patterns_rejected_quickly(self, pattern):
        ctx, ws = self._ctx(allow_regex=True)
        start = time.perf_counter()
        r = self._run(ctx, ws, pattern)
        assert "rejected" in r["error"]
        assert time.perf_counter() - start < 0.5

    def test_normal_pattern_works(self):
        ctx, ws = self._ctx(allow_regex=True)
        assert self._run(ctx, ws, "^ab")["items"] == [{"s": "abc"}]

    def test_regex_disabled_by_default(self):
        ctx, ws = self._ctx()
        assert "disabled" in self._run(ctx, ws, "a")["error"]

    def test_invalid_pattern_is_reported(self):
        ctx, ws = self._ctx(allow_regex=True)
        assert "Invalid $regex" in self._run(ctx, ws, "(")["error"]


# ── Generic toolsets (short tool lists) ──────────────────────────────────


class TestGenericToolsets:
    def test_one_toolset_serves_many_types(self, store):
        ctx = Ctx(store=store)
        data = [paginator(ctx, "data", enforce_type=False), pipeline(ctx, "data", enforce_type=False)]

        @ctx.producer(workspace_type="papers", toolsets=data)
        def papers() -> list[dict]:
            return [{"t": "a"}]

        @ctx.producer(workspace_type="people", toolsets=data)
        def people() -> list[dict]:
            return [{"n": "b"}, {"n": "c"}]

        p1, p2 = papers()["workspace_id"], people()["workspace_id"]
        assert ctx.dispatch_tool_call("data_count", {"workspace_id": p1})["count"] == 1
        assert ctx.dispatch_tool_call("data_count", {"workspace_id": p2})["count"] == 2
        names = [t["function"]["name"] for t in ctx.get_tools()]
        assert len([n for n in names if n.startswith("data_")]) == 6
        saved = ctx.dispatch_tool_call(
            "data_pipe", {"workspace_id": p2, "steps": [{"limit": 1}], "save_as": "one"}
        )
        assert saved["workspace_type"] == "people"


# ── SQLite threading / lifecycle ─────────────────────────────────────────


class TestSQLiteThreads:
    def test_dead_threads_connections_are_closed(self, tmp_path):
        store = SQLiteStore(tmp_path / "x.db")
        for _ in range(50):
            t = threading.Thread(target=lambda: store.list_workspaces())
            t.start()
            t.join()
        store.list_workspaces()
        # A new connection prunes the dead threads' ones.
        threading.Thread(target=lambda: store.list_workspaces()).start()
        time.sleep(0.05)
        assert len(store._connections) <= 3
        store.close()

    def test_failed_write_does_not_lock_other_threads(self, tmp_path):
        for store in (SQLiteStore(":memory:"), SQLiteStore(tmp_path / "y.db", busy_timeout=200)):
            ctx = Ctx(store=store)

            @ctx.producer(workspace_type="t", toolsets=[], key="ws")
            def mk() -> list[int]:
                return [1]

            mk()
            with pytest.raises(IndexError):
                store.update_item("ws", 5, 99)
            errors = []

            def other(store=store, errors=errors):
                try:
                    store.get_items("ws")
                    store.set_items("ws", [1, 2])
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            t = threading.Thread(target=other)
            t.start()
            t.join()
            assert errors == []
            assert store.get_items("ws") == [1, 2]
            store.close()

    def test_memory_database_is_shared_across_threads(self):
        store = SQLiteStore(":memory:")
        ctx = Ctx(store=store)

        @ctx.producer(workspace_type="t", toolsets=[paginator(ctx)])
        def mk() -> list[int]:
            return [1, 2, 3]

        ws = mk()["workspace_id"]
        seen = []
        t = threading.Thread(
            target=lambda: seen.append(
                ctx.dispatch_tool_call("t_count", {"workspace_id": ws})
            )
        )
        t.start()
        t.join()
        assert seen[0]["count"] == 3
        store.close()

    def test_close_closes_every_thread_connection(self, tmp_path):
        store = SQLiteStore(tmp_path / "x.db")
        t = threading.Thread(target=lambda: store.list_workspaces())
        t.start()
        t.join()
        assert len(store._connections) == 2
        store.close()
        assert not store._connections
        # Usable again after close (reconnects lazily).
        assert store.list_workspaces() == []
        store.close()


# ── Integrations ─────────────────────────────────────────────────────────


class TestIntegrationRobustness:
    def test_openai_bad_json_arguments_do_not_crash(self):
        ctx, _ = _papers_ctx(MemoryStore())
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "1", "function": {"name": "load_papers", "arguments": "{oops"}},
                            {"id": "2", "function": {"name": "load_papers", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        }
        msgs = handle_tool_calls(ctx, response)
        assert "not valid JSON" in json.loads(msgs[0]["content"])["error"]
        assert json.loads(msgs[1]["content"])["status"] == "workspace_ready"

    def test_openai_truncation_marker(self):
        ctx, _ = _papers_ctx(MemoryStore())
        response = {
            "choices": [
                {"message": {"tool_calls": [
                    {"id": "1", "function": {"name": "load_papers", "arguments": "{}"}}
                ]}}
            ]
        }
        content = handle_tool_calls(ctx, response, max_content_length=120)[0]["content"]
        assert len(content) <= 120 and "[truncated" in content

    def test_openai_tiny_limit_is_respected(self):
        ctx, _ = _papers_ctx(MemoryStore())
        response = {"choices": [{"message": {"tool_calls": [
            {"id": "1", "function": {"name": "load_papers", "arguments": "{}"}}
        ]}}]}
        assert len(handle_tool_calls(ctx, response, max_content_length=10)[0]["content"]) == 10

    def test_anthropic_session_id(self):
        ctx, _ = _papers_ctx(MemoryStore())
        response = {
            "content": [{"type": "tool_use", "id": "t1", "name": "load_papers", "input": {}}]
        }
        block = handle_tool_use(ctx, response, session_id="s9")[0]
        ws = json.loads(block["content"])["workspace_id"]
        assert ctx.workspace_meta(ws).session_id == "s9"
        assert "is_error" not in block


class TestConsumerExpose:
    def test_hidden_consumer(self):
        ctx = Ctx()

        @ctx.consumer(workspace_type="raw", expose=False)
        def internal(workspace_id: str, ctx: ConsumerContext):
            return len(ctx.get_items())

        assert all(t["function"]["name"] != "internal" for t in ctx.get_tools())
        assert "not found" in ctx.dispatch_tool_call("internal", {"workspace_id": "x"})["error"]


class TestConsumerContextDetection:
    def test_keyword_only_annotated_parameter(self):
        ctx = Ctx()

        @ctx.producer(workspace_type="raw", toolsets=[])
        def ingest() -> list[dict]:
            return [{"quality": 0.9}, {"quality": 0.1}]

        @ctx.consumer(workspace_type="raw", produces="clean")
        def clean(workspace_id: str, min_quality: float = 0.5, *, cc: ConsumerContext):
            return cc.emit([r for r in cc.get_items() if r["quality"] >= min_quality])

        ws = ingest()["workspace_id"]
        ref = ctx.dispatch_tool_call("clean", {"workspace_id": ws})
        assert ref["item_count"] == 1
        schema = next(t for t in ctx.get_tools() if t["function"]["name"] == "clean")
        assert "cc" not in schema["function"]["parameters"]["properties"]

    def test_postponed_annotation(self):
        ctx = Ctx()
        ns: dict = {}
        exec(
            "from __future__ import annotations\n"
            "from ctxtual import ConsumerContext\n"
            "def count_rows(workspace_id: str, data: ConsumerContext):\n"
            "    return len(data.get_items())\n",
            ns,
        )
        consumer = ctx.consumer(workspace_type="raw")(ns["count_rows"])

        @ctx.producer(workspace_type="raw", toolsets=[])
        def ingest() -> list[int]:
            return [1, 2]

        assert consumer(workspace_id=ingest()["workspace_id"]) == 2
