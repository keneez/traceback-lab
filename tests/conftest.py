from pathlib import Path
import difflib

import pytest

from traceback_lab.cases import CaseStore, atomic_json
from traceback_lab.indexing import build_snapshot
from traceback_lab.labels import map_patch
from traceback_lab.storage import SnapshotStore


@pytest.fixture
def corpus(tmp_path):
    data = tmp_path / "data"
    store = SnapshotStore(data / "snapshots")
    source = (
        "from .pool import reset_pool\n\n"
        "def request(connection):\n"
        '    """Send a request after a timeout."""\n'
        "    return reset_pool(connection)\n\n"
        "def format_date(value):\n"
        '    """Format a date."""\n'
        "    return str(value)\n"
    )
    files = {"pkg/client.py": source.encode(), "pkg/pool.py": (
        "def reset_pool(connection):\n"
        '    """Release a closed connection from the pool."""\n'
        "    connection.closed = False\n"
        "    return connection\n"
    ).encode()}
    manifest = build_snapshot(store, files, "owner/example", "a" * 40)
    modified = source.replace("return reset_pool(connection)", "return connection")
    patch = "".join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True),
                                       fromfile="a/pkg/client.py", tofile="b/pkg/client.py"))
    labels = map_patch(patch, store.symbols(manifest["id"]))
    for i, split in enumerate(["train", "train", "validation"]):
        case_id = f"example-{i}"
        case = {
            "id": case_id, "snapshot_id": manifest["id"], "repository": "owner/example",
            "base_commit": "a" * 40, "title": "Request cannot reuse a timed out connection",
            "description": "Sending a request after timeout fails. Reset the connection pool.",
            "traceback": "", "trace_revision": None, "split": split,
            "group_id": f"group-{i}", "historical_text": "verified_before_fix",
            "trace_origin": "absent", "audit": {}, "fix_available_at": f"202{i}-01-01",
        }
        atomic_json(data / "cases" / f"{case_id}.json", case)
        atomic_json(data / "gold" / f"{case_id}.json",
                    {"case_id": case_id, "patch": patch, **labels,
                     "positive_symbols": [], "fix_url": "https://example.org/fix"})
    return data, store, CaseStore(data), manifest

