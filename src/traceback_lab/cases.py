"""Historical answers are isolated from the serving query contract."""
from pathlib import Path
import json
import re
from .config import fingerprint


def atomic_json(path: Path, value):
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


class CaseStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, kind, case_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
            raise KeyError(case_id)
        path = self.root / kind / f"{case_id}.json"
        if not path.is_file():
            raise KeyError(case_id)
        return path

    def raw(self):
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((self.root / "cases").glob("*.json"))]

    def list(self):
        rows = self.raw()
        manifest = self.root / "split_manifest.json"
        roles = json.loads(manifest.read_text(encoding="utf-8")).get("roles", {}) if manifest.exists() else {}
        for row in rows:
            entry = roles.get(row["id"])
            if entry and fingerprint(row) == entry["case_fingerprint"]:
                row["split"] = entry["split"]
                row["group_id"] = entry["group_id"]
        return rows

    def public(self, case_id):
        return json.loads(self._path("cases", case_id).read_text(encoding="utf-8"))

    def reveal(self, case_id):
        return json.loads(self._path("gold", case_id).read_text(encoding="utf-8"))

    def labeled(self, splits=None):
        output = []
        for case in self.list():
            if splits is not None and case.get("split") not in splits:
                continue
            gold = self.reveal(case["id"])
            if gold["positive_ids"]:
                output.append((case, gold))
        return output

