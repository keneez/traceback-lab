from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os

_source_root = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("TRACEBACK_PROJECT", _source_root if
                         (_source_root / "pyproject.toml").exists() else Path.cwd())).resolve()
DATA_ROOT = Path(os.environ.get("TRACEBACK_DATA", ROOT / "data")).resolve()
STATE_ROOT = Path(os.environ.get("TRACEBACK_STATE", ROOT / ".state")).resolve()
SCHEMA_VERSION = "5"
PARSER_VERSION = "tree-sitter-0.25.2-python-0.25.0-v4"


def fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SearchConfig:
    mode: str = "hybrid"
    budget: int = 50
    raw_top_k: int = 60
    trace_slots: int = 10
    direct_slots: int = 30
    graph_slots: int = 10
    graph_seeds: int = 10
    graph: bool = True
    rerank: bool = True
    max_windows: int = 3
    rrf_constant: int = 60

    @classmethod
    def profile(cls, name: str, mode: str = "hybrid", graph: bool = True):
        if name == "fast":
            return cls(mode=mode, budget=20, trace_slots=4, direct_slots=12,
                       graph_slots=4, graph_seeds=5, max_windows=1, graph=graph)
        if name != "quality":
            raise ValueError("Unknown profile")
        return cls(mode=mode, graph=graph)

    def identity(self):
        return asdict(self)

