"""Train the portable encoder using docstrings from the installed Python standard library."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import tokenize
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..cases import atomic_json
from ..config import fingerprint
from .compact import BUCKETS, DIM, FORMAT, code_ids, tokens


def corpus(root):
    pairs, seen_text, seen_code = [], set(), set()
    excluded = {"site-packages", "dist-packages", "test", "tests", "idle_test", "__pycache__",
                "ensurepip", "turtledemo", "lib2to3", "venv"}
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(n for n in names if n not in excluded and not n.startswith("."))
        for filename in sorted(filenames):
            if not filename.endswith(".py") or filename.startswith("test_"):
                continue
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or relative.count("/") > 4:
                continue
            with tokenize.open(path) as stream:
                source = stream.read()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue
            lines = source.splitlines()
            # Every function belongs to the split of its top-level stdlib module.
            group = relative.split("/")[0].split(".")[0]
            validation = int(fingerprint(group)[:4], 16) % 8 == 0
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node)
                if not doc or len(doc.split()) < 5:
                    continue
                start = node.body[0].end_lineno
                # Keep implementation identifiers and signatures, withholding the query docstring.
                body = "\n".join(lines[start:node.end_lineno])
                ids = code_ids(relative, node.name, body)
                query = tokens(doc, 80)
                if len(query) < 3 or len(ids) < 5:
                    continue
                text_key, code_key = tuple(query), tuple(ids)
                if text_key in seen_text or code_key in seen_code:
                    continue
                seen_text.add(text_key)
                seen_code.add(code_key)
                pairs.append((query, ids, validation))
    return pairs


class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(BUCKETS, DIM, padding_idx=0)
        self.query = nn.Parameter(torch.eye(DIM))
        self.code = nn.Parameter(torch.eye(DIM))

    def encode(self, rows, head, dropout=0.):
        ids = nn.utils.rnn.pad_sequence([torch.tensor(row) for row in rows], batch_first=True)
        if dropout:
            ids = ids.masked_fill(torch.rand(ids.shape) < dropout, 0)
        mask = ids.ne(0)
        pooled = self.embedding(ids).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(pooled + torch.tanh(pooled @ head), dim=-1)


@torch.no_grad()
def validation_mrr(model, pairs):
    query = torch.cat([model.encode([p[0] for p in pairs[i:i+96]], model.query)
                       for i in range(0, len(pairs), 96)])
    code = torch.cat([model.encode([p[1] for p in pairs[i:i+96]], model.code)
                      for i in range(0, len(pairs), 96)])
    scores = query @ code.T
    ranks = 1 + (scores > scores.diagonal()[:, None]).sum(1)
    return ranks.float().reciprocal().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdlib", type=Path, default=Path(sysconfig.get_path("stdlib")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    torch.manual_seed(31)
    torch.set_num_threads(3)
    pairs = corpus(args.stdlib)
    train = [p for p in pairs if not p[2]]
    validation = [p for p in pairs if p[2]]
    if not train or not validation or args.epochs < 1:
        raise ValueError("Provide a Python standard library with training and validation pairs")
    args.output.mkdir(parents=True, exist_ok=True)
    model = DualEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    baseline = validation_mrr(model, validation)
    history, best = [], -1
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train)).tolist()
        losses = []
        for start in range(0, len(train), 64):
            batch = [train[i] for i in permutation[start:start + 64]]
            q = model.encode([p[0] for p in batch], model.query, dropout=.15)
            c = model.encode([p[1] for p in batch], model.code, dropout=.2)
            logits = q @ c.T / .09
            y = torch.arange(len(batch))
            loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
            loss = loss + .15 * (model.query - model.code).square().sum() / DIM
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mrr = validation_mrr(model, validation)
        history.append({"epoch": epoch, "mean_loss": round(float(np.mean(losses)), 6), "mrr": mrr})
        print(json.dumps(history[-1]), flush=True)
        if mrr > best:
            best, best_epoch = mrr, epoch
            np.savez_compressed(args.output / "weights.npz", **{
                key: value.detach().numpy().astype(np.float16) for key, value in
                (("embedding", model.embedding.weight), ("query", model.query), ("code", model.code))})
    weights = args.output / "weights.npz"
    atomic_json(args.output / "training.json", {
        "format": FORMAT, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "corpus": "installed CPython standard library; docstring to implementation",
        "python": sys.version.split()[0], "torch": torch.__version__, "seed": 31,
        "pairs_fingerprint": fingerprint(pairs), "training_pairs": len(train),
        "validation_pairs": len(validation), "metric": "docstring retrieval MRR; stdlib module holdout",
        "initial_random_mrr": baseline, "query_token_dropout": .15, "code_token_dropout": .2, "selected_epoch": best_epoch, "history": history})


if __name__ == "__main__":
    main()
