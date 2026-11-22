import io
from pathlib import Path
import stat
import sys
import subprocess
import time
import warnings
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from traceback_lab.api import create_app
from traceback_lab.config import SearchConfig
from traceback_lab.diagnostics import scan_source
from traceback_lab.ml.compact import CompactEncoder
from traceback_lab.retrieval import SearchEngine
from traceback_lab.storage import SnapshotStore
from traceback_lab.uploads import import_source, MAX_FILE, UploadLimit


def archive(entries):
    data = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as handle:
        warnings.simplefilter("ignore", UserWarning)
        for path, content in entries:
            handle.writestr(path, content)
    return data.getvalue()


def test_zip_is_scoped_idempotent_and_never_executes(tmp_path):
    store = SnapshotStore(tmp_path)
    sentinel = tmp_path / "executed"
    data = f'open({str(sentinel)!r}, "w").close()\ndef connect():\n    return 1\n'.encode()
    link = zipfile.ZipInfo("demo/link.py")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    zipped = archive([("demo/src/pkg/client.py", data), ("demo/tests/other.py", data),
                      ("demo/node_modules/script.py", data), (link, "/etc/passwd"), ("demo/key.txt", "private")])
    first = import_source(store, zipped, "demo.zip")
    second = import_source(store, zipped, "demo.zip")
    assert first["id"] == second["id"]
    assert first["file_count"] == first["symbol_count"] == 1
    assert store.symbols(first["id"])[0]["path"] == "src/pkg/client.py"
    assert not sentinel.exists()


@pytest.mark.parametrize("name", ["../danger.py", "/danger.py", r"C:\other\danger.py", r"x\..\danger.py"])
def test_unsafe_paths_reject_whole_archive(tmp_path, name):
    store = SnapshotStore(tmp_path)
    with pytest.raises(ValueError, match="путь"):
        import_source(store, archive([("ok.py", "pass"), (name, "pass")]), "sources.zip")
    assert not store.list()


def test_malformed_duplicate_oversize_and_out_of_scope_inputs(tmp_path):
    store = SnapshotStore(tmp_path)
    for payload, name in [(b"garbage", "s.zip"), (archive([("test_only.py", "pass")]), "s.zip"),
                          (archive([("a.py", "pass"), ("a.py", "pass")]), "s.zip"), (b"pass", "a.js")]:
        with pytest.raises(ValueError):
            import_source(store, payload, name)
    with pytest.raises(UploadLimit):
        import_source(store, archive([("big.py", b"x" * (MAX_FILE + 1))]), "big.zip")
    assert not store.list()


def test_archive_preserves_importable_package_and_non_utf8_python(tmp_path):
    store = SnapshotStore(tmp_path)
    raw = '# coding: latin-1\ndef café():\n    return 1\n'.encode("latin-1")
    snap = import_source(store, archive([("pkg/__init__.py", ""), ("pkg/main.py", raw)]), "pkg.zip")
    assert store.symbols(snap["id"])[0]["qualname"] == "café"
    assert "café" in store.source(snap["id"], "pkg/main.py")["source"]


def test_checks_return_lines_and_avoid_near_misses():
    src = ("def fn(cache=[], *, names={}):\n    d = {True: 0, 1: 2}\n    if (False, False):\n"
           "        try:\n            return d\n        except:\n            pass\n")
    results = list(scan_source("a.py", src))
    assert sorted((r["rule"], r["line"]) for r in results) == [
        ("PY002", 1), ("PY002", 1), ("PY003", 2), ("PY004", 3), ("PY005", 6)]
    safe = ("def fn(cache=None):\n    d = {'x': 1, 'y': 2}\n    if (*cache,):\n"
            "        try:\n            return d\n        except KeyError:\n            pass\n")
    assert list(scan_source("a.py", safe)) == []


def wait_job(client, job):
    for _ in range(120):
        row = client.get("/v1/investigations/" + job["id"]).json()
        if row["status"] in {"completed", "failed", "cancelled"}:
            return row
        time.sleep(.01)
    raise AssertionError("Job did not finish")


def test_web_import_search_baseline_history_and_reopen(tmp_path):
    data, state = tmp_path / "data", tmp_path / "state"
    with TestClient(create_app(data, state)) as client:
        assert client.get("/v1/models").json()["encoder_name"] == "Compact-64"
        src = b"def collect_packet(packet, seen=[]):\n    seen.append(packet)\n    return seen\n"
        headers = {"Content-Type": "application/octet-stream"}
        snap = client.post("/v1/snapshots/import?filename=router.py", content=src, headers=headers).json()
        sid = snap["id"]
        assert (state / "projects" / "snapshots" / sid / "index.sqlite").is_file()
        assert not (data / "snapshots" / sid).exists()
        assert client.get("/v1/snapshots?compact=true").json()[0] == snap
        checks = client.get(f"/v1/snapshots/{sid}/checks").json()
        assert checks["findings"][0]["rule"] == "PY002"
        assert client.get(f"/v1/snapshots/{sid}/source", params={"path": "router.py"}).json()["source"] == src.decode()
        assert client.get(f"/v1/snapshots/{sid}/source", params={"path": "../outside.py"}).status_code == 404
        payload = {"snapshot_id": sid, "title": "collect_packet duplicates", "mode": "hybrid"}
        job = wait_job(client, client.post("/v1/investigations", json=payload).json())
        assert job["result"]["methods"] == ["bm25", "dense"]
        assert job["result"]["candidates"][0]["qualname"] == "collect_packet"
        assert job["result"]["models"]["encoder"] == CompactEncoder().identity
        baseline = client.get(f"/v1/investigations/{job['id']}/baseline").json()
        assert baseline["models"]["encoder"] is None and baseline["methods"] == ["bm25"]
        assert baseline["snapshot_id"] == sid and baseline["query"] == job["result"]["query"]
        assert baseline["config"]["budget"] == job["result"]["config"]["budget"]
        rows = client.get("/v1/investigations").json()
        assert [r["id"] for r in rows] == [job["id"]]
        assert "result" not in rows[0] and "patch" not in str(rows)
    with TestClient(create_app(data, state)) as client:
        assert client.get(f"/v1/investigations/{job['id']}").json()["result"] == job["result"]


def test_script_without_functions_and_invalid_source_still_open(tmp_path):
    with TestClient(create_app(tmp_path / "data", tmp_path / "state")) as client:
        url = "/v1/snapshots/import?filename=a.py"
        assert client.post(url, content=b"pass", headers={"Content-Type":"text/plain"}).status_code == 415
        result = client.post(url, content=b"def broken(:\n", headers={"Content-Type":"application/octet-stream"})
        assert result.status_code == 201, result.text
        sid = result.json()["id"]
        found = client.get(f"/v1/snapshots/{sid}/checks").json()
        assert found["total"] == 1 and found["findings"][0]["level"] == "error"
        assert client.post("/v1/investigations", json={"snapshot_id":sid,"title":"broken"}).status_code == 422


def test_context_reserves_trace_without_running_dense(corpus):
    _, store, _, snap = corpus
    text = '  File "/app/pkg/pool.py", line 2, in reset_pool'
    result = SearchEngine(store, CompactEncoder()).search(
        snap["id"], "unrelated", traceback=text, config=SearchConfig.profile("quality", "context"))
    assert result["methods"] == ["bm25"]
    assert result["candidates"][0]["qualname"] == "reset_pool"
    assert result["candidates"][0]["origin"] == "traceback"


def test_portable_search_has_no_torch_or_transformers_import(corpus):
    _, store, _, snap = corpus
    code = """
import sys
sys.modules["torch"] = sys.modules["transformers"] = None
from pathlib import Path
from traceback_lab.config import SearchConfig
from traceback_lab.storage import SnapshotStore
from traceback_lab.retrieval import SearchEngine
from traceback_lab.ml.compact import CompactEncoder
engine = SearchEngine(SnapshotStore(Path(sys.argv[1])), CompactEncoder())
r = engine.search(sys.argv[2], "connection timeout", config=SearchConfig(mode="dense"))
assert r["methods"] == ["dense"] and r["candidates"]
"""
    subprocess.run([sys.executable, "-c", code, str(store.root), snap["id"]], check=True)


def test_portable_embeddings_normalized_and_content_sensitive():
    encoder = CompactEncoder()
    a, b = encoder.encode_queries(["stream bytes", "timeout connection pool"])
    assert np.linalg.norm(a) == pytest.approx(1, abs=1e-5)
    assert np.dot(a, b) < .99
    assert np.isfinite(encoder.encode_queries([""])).all()


def test_training_output_can_be_selected_at_the_normal_entry_point(tmp_path):
    from argparse import Namespace
    from traceback_lab.cli import make_engine
    from traceback_lab.ml.compact import BUNDLED
    args = Namespace(encoder=BUNDLED, command="serve", compact=False)
    engine = make_engine(args, SnapshotStore(tmp_path))
    assert isinstance(engine.encoder, CompactEncoder)
