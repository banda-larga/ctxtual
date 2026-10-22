"""The benchmark harness runs end to end (scripted model, no API calls)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

import run
import tasks


def test_self_test_runs_every_task_and_condition(tmp_path, capsys):
    out = tmp_path / "runs.jsonl"
    assert run.main(["--self-test", "--out", str(out)]) == 0
    runs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(runs) == len(tasks.tasks("medium")) * len(run.CONDITIONS)
    assert all(r["correct"] and not r["error"] for r in runs)
    # every condition delivers the tool's data (a broken baseline would be unfair)
    assert all(r["tool_errors"] == 0 for r in runs)
    stored = [r for r in runs if r["condition"] == "ctxtual"]
    assert all(r["largest_tool_result"] < 20_000 for r in stored)  # notices, not data
    assert "| scripted | medium | ctxtual | 6/6 |" in out.with_suffix(".md").read_text()


def test_answers_are_unambiguous():
    for size in tasks.SIZES:
        log_rows = [json.loads(line) for line in tasks.logs(size).splitlines()]
        errors = {}
        for row in log_rows:
            if row["level"] == "ERROR":
                errors[row["service"]] = errors.get(row["service"], 0) + 1
        first, second = sorted(errors.values(), reverse=True)[:2]
        assert first > second
        rows = tasks.issues(size)
        assert sum("segfault in the tokenizer" in r["body"] for r in rows) == 1


def test_grading():
    task = tasks.Task("t", "q", "37", "int")
    assert run._grade("37 connections", task) and run._grade("**37**", task)
    assert not run._grade("370", task) and not run._grade(None, task)
    text = tasks.Task("t", "q", "search", "text", ("billing", "auth"))
    assert run._grade("`search`", text) and not run._grade("search, billing", text)
    title = tasks.Task("t", "q", "Batch shard is slow with quota #22", "text")
    assert run._grade('**"Batch shard is slow with quota #22"** (99 comments)', title)
    assert not run._grade("Buffer token is slow with module #26", title)
    assert not run._grade("at least 46", tasks.Task("t", "q", "136", "int"))
