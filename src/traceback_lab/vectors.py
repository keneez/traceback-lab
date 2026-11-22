"""Atomic, reusable embedding indexes for NumPy and transformer encoders."""
from pathlib import Path
import json
import os
import shutil
import tempfile

import numpy as np

from .config import STATE_ROOT, fingerprint

def index_vectors(store, snapshot_id, encoder, batch_size=8, checkpoint=None):
    final = store.directory(snapshot_id) / "vectors" / encoder.identity
    if (final / "manifest.json").exists():
        return
    final.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".vectors-", dir=final.parent))
    try:
        symbols = store.symbols(snapshot_id)
        sequences = [encoder.symbol_ids(s) for s in symbols]
        cache = STATE_ROOT / "embedding-cache" / encoder.identity
        cache.mkdir(parents=True, exist_ok=True)
        keys = [fingerprint(ids) for ids in sequences]
        cached = {}
        missing = {}
        for key, ids in zip(keys, sequences):
            path = cache / f"{key}.npy"
            if path.exists():
                try:
                    vector = np.load(path, allow_pickle=False)
                    if vector.shape == (encoder.dimensions,) and np.isfinite(vector).all():
                        cached[key] = vector
                        continue
                except (OSError, ValueError):
                    pass
            missing[key] = ids
        missing_keys = list(missing)
        for start in range(0, len(missing_keys), batch_size):
            if checkpoint:
                checkpoint()
            batch = missing_keys[start:start + batch_size]
            values = encoder._encode([missing[key] for key in batch], batch_size)
            for key, vector in zip(batch, values):
                cached[key] = vector
                descriptor, name = tempfile.mkstemp(prefix=".vector-", dir=cache)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        np.save(stream, vector, allow_pickle=False)
                    os.replace(name, cache / f"{key}.npy")
                finally:
                    if Path(name).exists():
                        Path(name).unlink()
        vectors = np.stack([cached[key] for key in keys])
        if len(vectors) != len(symbols) or not np.isfinite(vectors).all():
            raise RuntimeError("Encoder produced invalid vectors")
        np.save(temporary / "vectors.npy", vectors, allow_pickle=False)
        (temporary / "manifest.json").write_text(json.dumps({
            "model_id": encoder.identity, "snapshot_id": snapshot_id,
            "serialization": encoder.serialization, "ids": [s["id"] for s in symbols],
            "shape": list(vectors.shape), "dtype": "float32",
        }), encoding="utf-8")
        try:
            os.rename(temporary, final)
        except OSError:
            if not (final / "manifest.json").exists():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

