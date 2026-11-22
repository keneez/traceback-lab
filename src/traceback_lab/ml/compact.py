"""Portable inference for the shipped contrastive deep averaging encoder."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import keyword

import numpy as np

from ..config import fingerprint
from ..text import STOPWORDS, terms

FORMAT = "compact-dan-v1"
BUCKETS = 8192
DIM = 64
BUNDLED = Path(__file__).resolve().parents[1] / "assets" / "compact"
IGNORE = STOPWORDS | set(keyword.kwlist) | {"self", "cls", "none", "true", "false"}


@lru_cache(maxsize=32768)
def bucket(term):
    digest = hashlib.blake2s(term.encode(), digest_size=4).digest()
    return 1 + int.from_bytes(digest, "little") % (BUCKETS - 1)


def tokens(text, limit):
    counts = Counter(t for t in terms(text[:32000]) if len(t) > 1 and t not in IGNORE)
    return [bucket(term) for term, _ in counts.most_common(limit)]


def code_ids(path, name, body, doc=""):
    named = tokens(name, 16)
    ids = named * 3 + tokens(path, 12) + tokens(doc, 26) + tokens(body, 70)
    return ids or [0]


class CompactEncoder:
    name = "Compact-64"
    device = "cpu"
    dimensions = DIM
    serialization = FORMAT

    def __init__(self, path=BUNDLED):
        self.path = Path(path)
        metadata = json.loads((self.path / "training.json").read_text(encoding="utf-8"))
        weights = self.path / "weights.npz"
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if metadata["format"] != FORMAT or metadata["weights_sha256"] != digest:
            raise ValueError("Несовместимые или повреждённые веса Compact-64")
        self.identity = fingerprint([FORMAT, digest])[:24]
        with np.load(weights, allow_pickle=False) as arrays:
            self.embedding, self.query_head, self.code_head = (
                arrays[key].astype(np.float32) for key in ("embedding", "query", "code"))
        if (self.embedding.shape != (BUCKETS, DIM) or
                self.query_head.shape != (DIM, DIM) or self.code_head.shape != (DIM, DIM)):
            raise ValueError("Неверный размер весов Compact-64")

    @staticmethod
    def symbol_ids(symbol):
        return code_ids(symbol["path"], symbol["qualname"], symbol["search_code"],
                        symbol.get("docstring", ""))

    def _forward(self, sequences, head):
        if not sequences:
            return np.empty((0, DIM), dtype=np.float32)
        pooled = np.stack([self.embedding[ids].mean(axis=0) for ids in sequences])
        output = pooled + np.tanh(pooled @ head)
        return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)

    def _encode(self, sequences, batch_size=8):
        return self._forward(sequences, self.code_head)

    def encode_symbols(self, symbols, batch_size=8):
        return self._encode([self.symbol_ids(s) for s in symbols], batch_size)

    def encode_queries(self, texts, batch_size=8):
        return self._forward([tokens(text, 80) or [0] for text in texts], self.query_head)
