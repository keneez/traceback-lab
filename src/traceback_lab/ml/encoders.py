from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..config import fingerprint
from ..vectors import index_vectors  # public import retained for existing callers
from ..cases import atomic_json

SERIALIZATION_VERSION = "overview-160-head-tail-512-v1"


def resolve_checkpoint(path):
    path = Path(path).resolve()
    pointer = path / "current.json"
    if pointer.exists():
        relative = json.loads(pointer.read_text(encoding="utf-8"))["path"]
        target = (path / relative).resolve()
        if not target.is_relative_to(path):
            raise ValueError("Checkpoint pointer leaves its model directory")
        return target
    return path


def publish_checkpoint(model, tokenizer, output, metadata):
    output = Path(output).resolve()
    versions = output / "checkpoints"
    versions.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=versions))
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        atomic_json(temporary / "traceback_model.json", metadata)
        identity = checkpoint_identity(temporary, metadata["kind"])
        final = versions / identity
        try:
            os.rename(temporary, final)
        except OSError:
            if not (final / "traceback_model.json").exists():
                raise
        atomic_json(output / "current.json", {"path": f"checkpoints/{identity}"})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def checkpoint_identity(path: Path, kind: str):
    digest = hashlib.sha256()
    for file in sorted(path.glob("*")):
        if file.suffix not in {".json", ".safetensors", ".bin", ".txt"}:
            continue
        if file.name in {"source.json", "traceback_model.json", "training_log.json", "current.json"}:
            continue
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return fingerprint([kind, SERIALIZATION_VERSION, digest.hexdigest()])[:24]


def download_checkpoint(model_id: str, output: Path, revision: str | None = None):
    from huggingface_hub import snapshot_download
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = Path(snapshot_download(
        model_id, revision=revision,
        allow_patterns=["config.json", "vocab.json", "merges.txt", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json",
                        "model.safetensors", "pytorch_model.bin"]))
    versions = output / "checkpoints"
    versions.mkdir(exist_ok=True)
    final = versions / downloaded.name
    temporary = Path(tempfile.mkdtemp(prefix=".download-", dir=versions))
    try:
        for item in downloaded.iterdir():
            if item.is_file() and item.name != "README.md":
                shutil.copy2(item, temporary / item.name)
        atomic_json(temporary / "source.json", {"model_id": model_id, "revision": downloaded.name})
        try:
            os.rename(temporary, final)
        except OSError:
            if not (final / "config.json").exists():
                raise
        atomic_json(output / "current.json", {"path": f"checkpoints/{downloaded.name}"})
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


class BiEncoder:
    name = "UniXcoder"
    serialization = SERIALIZATION_VERSION

    def __init__(self, path, device="cpu", trainable=False):
        self.path = resolve_checkpoint(path)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        config = AutoConfig.from_pretrained(self.path, local_files_only=True)
        config.is_decoder = True
        config._attn_implementation = "eager"
        self.model = AutoModel.from_pretrained(self.path, config=config,
                                               local_files_only=True).to(self.device)
        self.dimensions = self.model.config.hidden_size
        self.model.train(trainable)
        self.identity = checkpoint_identity(self.path, "unixcoder")
        mode_id = self.tokenizer.convert_tokens_to_ids("<encoder-only>")
        if mode_id == self.tokenizer.unk_token_id:
            raise ValueError("Checkpoint does not have UniXcoder's encoder-only token")
        self.prefix = [self.tokenizer.cls_token_id, mode_id, self.tokenizer.sep_token_id]

    def query_ids(self, text):
        payload = self.tokenizer.encode(text, add_special_tokens=False)[:252]
        return [*self.prefix, *payload, self.tokenizer.sep_token_id]

    def symbol_ids(self, symbol):
        header = "\n".join([symbol["path"], symbol["qualname"], symbol["signature"],
                            symbol.get("docstring", "")])
        head = self.tokenizer.encode(header, add_special_tokens=False)[:160]
        body = self.tokenizer.encode(symbol["search_code"], add_special_tokens=False)
        capacity = 508 - len(head)
        if len(body) > capacity:
            gap = self.tokenizer.encode("\n# ... omitted ...\n", add_special_tokens=False)
            available = max(0, capacity - len(gap))
            first = math.ceil(available * 2 / 3)
            last = available - first
            body = [*body[:first], *gap, *body[-last:]] if last else [*body[:first], *gap]
        return [*self.prefix, *head, *body, self.tokenizer.sep_token_id]

    def forward(self, sequences):
        batch = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        ids = batch["input_ids"].to(self.device)
        mask = ids.ne(self.tokenizer.pad_token_id)
        # Full bidirectional mask reproduces the official encoder-only forward pass.
        hidden = self.model(ids, attention_mask=mask.unsqueeze(1) & mask.unsqueeze(2)).last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
        return F.normalize(pooled, p=2, dim=-1)

    def _encode(self, sequences, batch_size=8):
        self.model.eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                rows.append(self.forward(sequences[start:start + batch_size]).cpu().numpy())
        if not rows:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(rows).astype(np.float32)

    def encode_queries(self, texts, batch_size=8):
        return self._encode([self.query_ids(t) for t in texts], batch_size)

    def encode_symbols(self, symbols, batch_size=8):
        return self._encode([self.symbol_ids(s) for s in symbols], batch_size)

    def save(self, output, metadata):
        publish_checkpoint(self.model, self.tokenizer, output,
                           {"kind": "bi_encoder", "trained": True, **metadata})

