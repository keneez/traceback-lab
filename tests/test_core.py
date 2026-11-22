from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import difflib
import json
import subprocess
import time

from fastapi.testclient import TestClient
import pytest

from traceback_lab.api import create_app
from traceback_lab.config import SearchConfig
from traceback_lab.corpus import temporal_split
from traceback_lab.evaluation import ranking_metrics, compare_reports
from traceback_lab.indexing import build_snapshot, read_repository
from traceback_lab.labels import map_patch
from traceback_lab.parser import parse_file, build_graph
from traceback_lab.query import bind_trace, lexical_search
from traceback_lab.retrieval import SearchEngine, candidate_ids


def test_nested_unicode_and_decorators_have_exact_source():
    source = 'title = "Привет"\n@decorator\nasync def outer(x):\n    def inner():\n        return "ёж"\n    return inner()\n'
    symbols, _, _ = parse_file("pkg/file.py", source, "snapshot")
    assert [s.qualname for s in symbols] == ["outer", "outer.inner"]
    for symbol in symbols:
        assert source.encode()[symbol.start_byte:symbol.end_byte].decode() == symbol.code
    assert symbols[0].start_line == 2
    assert 'return "ёж"' not in symbols[0].search_code


def test_import_alias_resolves_and_parameter_shadowing_does_not(corpus):
    _, store, _, snapshot = corpus
    edges = store.edges(snapshot["id"])
    assert len(edges) == 1
    assert edges[0]["expression"] == "reset_pool"
    assert edges[0]["resolution"] == "import"
    assert edges[0]["line"] == 5


def test_nested_helper_resolves_without_treating_class_as_lexical_scope():
    code = (
        "class Client:\n"
        "    def reset(self):\n"
        "        return 1\n"
        "    def request(self):\n"
        "        def inner():\n"
        "            return reset()\n"
        "        return inner()\n"
    )
    symbols, imports, _ = parse_file("pkg/client.py", code, "snapshot")
    edges = build_graph(symbols, {"pkg/client.py": imports})
    assert len(edges) == 1
    assert edges[0]["expression"] == "inner"
    assert edges[0]["target"] == next(s.id for s in symbols if s.qualname.endswith("request.inner"))


def test_bm25_handles_fts_operators_as_literals(corpus):
    _, store, _, snapshot = corpus
    rows = lexical_search(store, snapshot["id"], 'request OR "timeout" NEAR(1) reset_pool')
    assert rows
    assert store.symbol(snapshot["id"], rows[0]["id"])["qualname"] == "request"


def test_innermost_label_and_pure_insertion():
    before = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    symbols, _, _ = parse_file("module.py", before, "snapshot")
    after = before.replace("        return 1", "        result = 1\n        return 1")
    patch = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                       fromfile="a/module.py", tofile="b/module.py"))
    result = map_patch(patch, [s.public() for s in symbols])
    assert result["positive_ids"] == [symbols[1].id]


def test_new_function_is_recorded_outside_old_candidate_universe():
    before = "def old():\n    return 1\n"
    after = before + "\ndef added():\n    return 2\n"
    symbols, _, _ = parse_file("module.py", before, "snapshot")
    patch = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                       fromfile="a/module.py", tofile="b/module.py"))
    result = map_patch(patch, [s.public() for s in symbols])
    assert result["positive_ids"] == []
    assert result["changes"][0]["category"] == "added_symbol"


def test_docstring_only_change_is_auxiliary():
    before = 'def old():\n    """Old description."""\n    return 1\n'
    after = before.replace("Old description.", "Improved description.")
    symbols, _, _ = parse_file("module.py", before, "snapshot")
    patch = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                       fromfile="a/module.py", tofile="b/module.py"))
    result = map_patch(patch, [s.public() for s in symbols])
    assert result["positive_ids"] == []
    assert result["changes"][0]["category"] == "auxiliary"


def test_metric_hit_and_recall_differ_for_multiple_functions():
    metrics = ranking_metrics(["a", "a", "b"], ["a", "c"])
    assert metrics["hit@1"] == 1
    assert metrics["recall@1"] == .5
    assert metrics["recall@5"] == .5
    assert metrics["mrr@5"] == 1


def test_paired_comparison_preserves_case_alignment():
    records = [{"case_id": f"case-{i}", "group_id": f"group-{i}", "mode": "hybrid",
                "config": {"budget": 50}, "metrics": {"hit@5": 0}} for i in range(3)]
    improved = [{**r, "metrics": {"hit@5": 1}} for r in records]
    result = compare_reports({"records": records}, {"records": improved})
    assert result["difference_second_minus_first"] == 1
    assert result["paired_group_bootstrap_ci95"] == [1, 1]
    with pytest.raises(ValueError, match="identical case"):
        compare_reports({"records": records}, {"records": improved[1:]})


def test_candidate_budget_is_unique_and_trace_is_preserved():
    direct = [f"function-{i}" for i in range(70)]
    trace = [f"trace-{i}" for i in range(10)]
    graph = [{"id": f"neighbor-{i}"} for i in range(20)]
    chosen, origins = candidate_ids(direct, trace, graph, SearchConfig())
    assert len(chosen) == len(set(chosen)) == 50
    assert chosen[:10] == trace
    assert sum(o == "graph" for o in origins.values()) == 10


def test_trace_binding_needs_matching_revision_for_confirmation(corpus):
    _, store, _, snapshot = corpus
    text = 'Traceback (most recent call last):\n  File "/app/pkg/client.py", line 5, in request\nRuntimeError: timeout'
    symbols = store.symbols(snapshot["id"])
    approximate = bind_trace(text, symbols, snapshot["revision"])
    confirmed = bind_trace(text, symbols, snapshot["revision"], snapshot["revision"])
    assert approximate[0]["status"] == "approximate"
    assert confirmed[0]["status"] == "confirmed"


def test_snapshots_are_stable_under_concurrent_build(tmp_path):
    from traceback_lab.storage import SnapshotStore
    store = SnapshotStore(tmp_path / "snapshots")
    files = {"module.py": b"def go():\n    return 1\n"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(build_snapshot, store, files, "repo", "rev") for _ in range(2)]
        values = [f.result() for f in futures]
    assert values[0]["id"] == values[1]["id"]
    assert len(store.list()) == 1
    assert len(store.symbols(values[0]["id"])) == 1


def test_unpublished_index_is_not_listed(tmp_path):
    from traceback_lab.storage import SnapshotStore
    store = SnapshotStore(tmp_path)
    hidden = tmp_path / ".index-temporary"
    hidden.mkdir()
    (hidden / "manifest.json").write_text('{"id":"not-published"}')
    assert store.list() == []


def test_git_revision_is_used_instead_of_modified_worktree(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "fixture@example.org"], check=True)
    file = repository / "file.py"
    file.write_text("def go():\n    return 'before'\n")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "base"], check=True)
    file.write_text("def go():\n    return 'after'\n")
    files, _ = read_repository(repository, "HEAD")
    assert b"before" in files["file.py"]
    assert b"after" not in files["file.py"]


def test_api_keeps_gold_out_of_initial_responses_and_rejects_extra_fields(corpus, tmp_path):
    data, store, _, snapshot = corpus
    with TestClient(create_app(data, tmp_path / "state", SearchEngine(store))) as client:
        initial = client.get("/v1/cases")
        assert initial.status_code == 200
        serialized = initial.text
        assert '"patch"' not in serialized
        assert "positive_ids" not in serialized
        payload = {"snapshot_id": snapshot["id"], "description": "request timeout",
                   "mode": "bm25", "graph": False}
        assert client.post("/v1/investigations", json={**payload, "patch": "future answer"}).status_code == 422
        job = client.post("/v1/investigations", json=payload).json()
        for _ in range(100):
            state = client.get("/v1/investigations/" + job["id"]).json()
            if state["status"] in {"completed", "failed"}:
                break
            time.sleep(.01)
        assert state["status"] == "completed", state
        assert "positive_ids" not in json.dumps(state)
        assert state["result"]["candidates"]
        assert state["result"]["confidence"]["probability"] is None
        answer = client.post("/v1/cases/example-0/reveal").json()
        assert answer["patch"]
        assert answer["positive_ids"]
        cached = client.post("/v1/investigations", json=payload).json()
        assert cached["cached"]


def test_grouped_temporal_split_does_not_train_on_holdout_or_demo(corpus):
    _, _, cases, _ = corpus
    result = temporal_split(cases, "2021-01-01", "2022-01-01", "owner/example")
    assert result["counts"] == {"holdout": 3}
    assert all(c["split"] == "holdout" for c in cases.list())


def test_dense_mode_requires_actual_encoder(corpus):
    _, store, _, snapshot = corpus
    with pytest.raises(ValueError, match="not configured"):
        SearchEngine(store).search(snapshot["id"], description="timeout", config=SearchConfig(mode="dense"))

