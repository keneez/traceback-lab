from pathlib import Path
import json
import math

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..text import query_terms, terms
from .encoders import checkpoint_identity, publish_checkpoint, resolve_checkpoint


def select_windows(tokenizer, query, symbol, frames=(), maximum=3):
    encoded = tokenizer(symbol["code"], add_special_tokens=False, return_offsets_mapping=True)
    tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
    query_words = set(query_terms(query))
    relevant_lines = {f["line"] for f in frames
                      if f["symbol_id"] == symbol["id"] and f["status"] == "confirmed"}
    windows = []
    for start in range(0, max(1, len(tokens)), 236):
        end = min(start + 332, len(tokens))
        begin_char = offsets[start][0] if offsets and start < len(offsets) else 0
        end_char = offsets[end - 1][1] if end else 0
        text = symbol["code"][begin_char:end_char]
        start_line = symbol["start_line"] + symbol["code"][:begin_char].count("\n")
        end_line = symbol["start_line"] + symbol["code"][:end_char].count("\n")
        windows.append({
            "tokens": tokens[start:end], "start_line": start_line, "end_line": end_line,
            "lexical": len(query_words & set(terms(text))),
            "trace": bool(relevant_lines & set(range(start_line, end_line + 1))),
            "index": len(windows),
        })
        if end == len(tokens):
            break
    ranked = sorted(windows, key=lambda w: (-w["trace"], -w["lexical"], w["index"]))
    selected = ranked[:maximum]
    if maximum > 1 and windows and windows[0] not in selected:
        selected[-1] = windows[0]
    return selected


class CrossEncoder:
    def __init__(self, path, device="cpu", trainable=False, initialize=False):
        self.path = resolve_checkpoint(path)
        metadata_path = self.path / "traceback_model.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if not initialize and not (metadata.get("trained") and metadata.get("kind") == "cross_encoder"):
            raise ValueError("Reranker needs a trained Traceback checkpoint. Use train-cross.")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, use_fast=True,
                                                       local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path, num_labels=1, local_files_only=True).to(self.device)
        self.model.train(trainable)
        self.identity = checkpoint_identity(self.path, "cross-encoder")

    def sequences(self, query, symbol, frames=(), maximum=3):
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)[:128]
        header = "\n".join([symbol["path"], symbol["qualname"], symbol["signature"]])
        head_ids = self.tokenizer.encode(header, add_special_tokens=False)[:48]
        windows = select_windows(self.tokenizer, query, symbol, frames, maximum)
        sequences = [self.tokenizer.build_inputs_with_special_tokens(
            query_ids, head_ids + window["tokens"]) for window in windows]
        if any(len(ids) > 512 for ids in sequences):
            raise RuntimeError("Pair token budget exceeded")
        return sequences, windows

    def forward(self, sequences):
        batch = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        return self.model(**batch).logits.squeeze(-1)

    def function_score(self, query, symbol, frames=(), maximum=3):
        sequences, _ = self.sequences(query, symbol, frames, maximum)
        logits = self.forward(sequences)
        return torch.logsumexp(logits, dim=0) - math.log(len(sequences))

    def score(self, query, symbols, frames=(), maximum=3, batch_size=8):
        groups, sequences = [], []
        for symbol in symbols:
            inputs, windows = self.sequences(query, symbol, frames, maximum)
            groups.append((len(sequences), len(inputs), windows))
            sequences.extend(inputs)
        self.model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                values.extend(self.forward(sequences[start:start + batch_size]).cpu().tolist())
        result = []
        for start, count, windows in groups:
            logits = np.asarray(values[start:start + count], dtype=np.float64)
            best = int(logits.argmax())
            maximum_logit = float(logits.max())
            score = maximum_logit + math.log(float(np.exp(logits - maximum_logit).mean()))
            result.append({
                "rerank_score": score,
                "window": {k: windows[best][k] for k in ["start_line", "end_line"]},
            })
        return result

    def save(self, output, metadata):
        publish_checkpoint(self.model, self.tokenizer, output,
                           {"kind": "cross_encoder", "trained": True, **metadata})

