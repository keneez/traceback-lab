from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3

import numpy as np


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        self.roots = [self.root]

    def include(self, store):
        if store.root not in self.roots:
            self.roots.append(store.root)

    def list(self):
        snapshots = {}
        for root in self.roots:
            for path in root.glob("*/manifest.json"):
                key = path.parent.name
                if len(key) == 24 and all(c in "0123456789abcdef" for c in key) and key not in snapshots:
                    snapshots[key] = json.loads(path.read_text(encoding="utf-8"))
        return [snapshots[key] for key in sorted(snapshots)]

    def directory(self, snapshot_id):
        if not snapshot_id or len(snapshot_id) != 24 or any(c not in "0123456789abcdef" for c in snapshot_id):
            raise KeyError(snapshot_id)
        for root in self.roots:
            path = root / snapshot_id
            if (path / "manifest.json").is_file():
                return path
        raise KeyError(snapshot_id)

    def manifest(self, snapshot_id):
        return json.loads((self.directory(snapshot_id) / "manifest.json").read_text(encoding="utf-8"))

    @contextmanager
    def connect(self, snapshot_id):
        path = self.directory(snapshot_id) / "index.sqlite"
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def symbols(self, snapshot_id):
        with self.connect(snapshot_id) as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM symbols ORDER BY id")]

    def symbol(self, snapshot_id, symbol_id):
        with self.connect(snapshot_id) as db:
            row = db.execute("SELECT data FROM symbols WHERE id=?", (symbol_id,)).fetchone()
            if row is None:
                raise KeyError(symbol_id)
            symbol = json.loads(row[0])
            source = db.execute("SELECT content FROM files WHERE path=?", (symbol["path"],)).fetchone()
            symbol["file_source"] = source[0]
            return symbol

    def sources(self, snapshot_id):
        with self.connect(snapshot_id) as db:
            for row in db.execute("SELECT path, content FROM files ORDER BY path"):
                yield row["path"], row["content"]

    def source(self, snapshot_id, path):
        with self.connect(snapshot_id) as db:
            row = db.execute("SELECT content FROM files WHERE path=?", (path,)).fetchone()
            if row is None:
                raise KeyError(path)
            return {"path": path, "source": row["content"]}

    def edges(self, snapshot_id):
        with self.connect(snapshot_id) as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM edges ORDER BY source,line,target")]

    def vectors(self, snapshot_id, model_id):
        directory = self.directory(snapshot_id) / "vectors" / model_id
        if not (directory / "manifest.json").exists():
            return None
        meta = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return meta, np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)

