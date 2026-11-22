"""CPU/GPU training with multi-positive labels and snapshot-aware negative masks."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from ..cases import atomic_json
from ..config import SearchConfig, fingerprint
from ..evaluation import ranking_metrics
from ..query import lexical_search
from ..text import query_text
from .encoders import BiEncoder
from .reranker import CrossEncoder


@dataclass
class TrainConfig:
    epochs: int = 2
    micro_batch: int = 1
    accumulation: int = 4
    negatives: int = 12
    learning_rate: float = 2e-5
    weight_decay: float = .01
    temperature: float = .05
    seed: int = 31
    max_steps: int | None = None
    gradient_checkpointing: bool = True


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def multi_positive_loss(queries, codes, positive_mask, valid_mask, temperature=.05):
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("Every query needs a positive candidate")
    if torch.any(positive_mask & ~valid_mask):
        raise ValueError("A positive was masked out")
    logits = (queries @ codes.T) / temperature
    logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    targets = positive_mask.to(logits.dtype)
    targets = targets / targets.sum(dim=1, keepdim=True)
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def labeled_examples(cases, store, split):
    examples = []
    for case, gold in cases.labeled({split}):
        symbols = store.symbols(case["snapshot_id"])
        by_id = {s["id"]: s for s in symbols}
        positives = set(gold["positive_ids"])
        if not positives <= by_id.keys():
            raise ValueError(f"Labels do not belong to snapshot for {case['id']}")
        examples.append({"case": case, "query": query_text(
            case["title"], case["description"], case["traceback"]),
            "symbols": symbols, "by_id": by_id, "positives": positives})
    return examples


def negative_ids(example, count, rng, mined=None):
    positives = example["positives"]
    positive_hashes = {example["by_id"][i]["source_hash"] for i in positives}
    eligible = [s["id"] for s in example["symbols"]
                if s["id"] not in positives and s["source_hash"] not in positive_hashes]
    eligible_set = set(eligible)
    selected = []
    if mined:
        sources = mined.get("rows", {}).get(example["case"]["id"], {})
        quota = count // 3
        for source in ["bm25", "dense"]:
            selected.extend([i for i in sources.get(source, [])
                             if i in eligible_set and i not in selected][:quota])
    remaining = [i for i in eligible if i not in selected]
    rng.shuffle(remaining)
    return [*selected, *remaining[:max(0, count - len(selected))]][:count]


def make_batch(examples, tokenizer_model, rng, config, mined=None):
    code_pool = {}
    for example in examples:
        positive_ids = sorted(example["positives"])
        rng.shuffle(positive_ids)
        ids = [*positive_ids[:4], *negative_ids(example, config.negatives, rng, mined)]
        for symbol_id in ids:
            code_pool[symbol_id] = example["by_id"][symbol_id]
    ids = sorted(code_pool)
    positives = torch.zeros((len(examples), len(ids)), dtype=torch.bool)
    valid = torch.ones_like(positives)
    for row, example in enumerate(examples):
        positive_hashes = {example["by_id"][i]["source_hash"] for i in example["positives"]}
        related_positives = set()
        for other in examples:
            if other["case"]["group_id"] == example["case"]["group_id"]:
                related_positives.update(other["positives"])
        for col, symbol_id in enumerate(ids):
            symbol = code_pool[symbol_id]
            if symbol_id in example["positives"]:
                positives[row, col] = True
            elif symbol["source_hash"] in positive_hashes or symbol_id in related_positives:
                valid[row, col] = False
    query_ids = [tokenizer_model.query_ids(e["query"]) for e in examples]
    code_ids = [tokenizer_model.symbol_ids(code_pool[i]) for i in ids]
    return query_ids, code_ids, positives, valid


def validation_bi(encoder, examples):
    if not examples:
        return None
    scores = []
    # One snapshot at a time bounds the validation matrix memory.
    by_snapshot = {}
    for example in examples:
        by_snapshot.setdefault(example["case"]["snapshot_id"], []).append(example)
    for snapshot_examples in by_snapshot.values():
        symbols = snapshot_examples[0]["symbols"]
        matrix = encoder.encode_symbols(symbols)
        queries = encoder.encode_queries([e["query"] for e in snapshot_examples])
        for example, vector in zip(snapshot_examples, queries):
            values = matrix @ vector
            order = sorted(range(len(symbols)), key=lambda i: (-float(values[i]), symbols[i]["id"]))
            ids = [symbols[i]["id"] for i in order[:50]]
            scores.append(ranking_metrics(ids, example["positives"]))
    return {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}


def optimizer_for(model, config, updates):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(updates * .1), num_training_steps=max(1, updates))
    return optimizer, scheduler


def check_training_groups(training, validation):
    train_groups = {e["case"]["group_id"] for e in training}
    validation_groups = {e["case"]["group_id"] for e in validation}
    if train_groups & validation_groups:
        raise ValueError("Train and validation share an issue/patch group. Rebuild the split.")


def train_bi(base, output, cases, store, config, device="cpu", mined_path=None):
    seed_everything(config.seed)
    examples = labeled_examples(cases, store, "train")
    validation = labeled_examples(cases, store, "validation")
    check_training_groups(examples, validation)
    if not examples:
        raise ValueError("No training cases. Import data and assign custom train/validation splits.")
    mined = json.loads(Path(mined_path).read_text(encoding="utf-8")) if mined_path else None
    if mined and mined["case_manifest"] != fingerprint(cases.list()):
        raise ValueError("Negative mining manifest differs from this corpus")
    encoder = BiEncoder(base, device, trainable=True)
    if config.gradient_checkpointing:
        encoder.model.gradient_checkpointing_enable()
    batches = math.ceil(len(examples) / config.micro_batch)
    updates = math.ceil(batches / config.accumulation) * config.epochs
    if config.max_steps:
        updates = min(updates, config.max_steps)
    optimizer, scheduler = optimizer_for(encoder.model, config, updates)
    rng = random.Random(config.seed)
    log, best, step = [], (-1.0, -1.0), 0
    for epoch in range(config.epochs):
        order = list(examples)
        rng.shuffle(order)
        minibatches = [order[i:i + config.micro_batch]
                       for i in range(0, len(order), config.micro_batch)]
        losses = []
        encoder.model.train()
        for start in range(0, len(minibatches), config.accumulation):
            group = minibatches[start:start + config.accumulation]
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                query_ids, code_ids, positive, valid = make_batch(batch, encoder, rng, config, mined)
                q, c = encoder.forward(query_ids), encoder.forward(code_ids)
                loss = multi_positive_loss(q, c, positive.to(encoder.device),
                                          valid.to(encoder.device), config.temperature)
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            step += 1
            print(json.dumps({"epoch": epoch + 1, "step": step, "loss": losses[-1]}), flush=True)
            if step >= updates:
                break
        metrics = validation_bi(encoder, validation)
        score = (metrics["hit@50"], metrics["recall@50"]) if metrics else (-np.mean(losses), 0)
        record = {"epoch": epoch + 1, "steps": step, "loss": float(np.mean(losses)),
                  "validation": metrics}
        log.append(record)
        metadata = {"config": asdict(config), "case_manifest": fingerprint(cases.list()),
                    "train_case_ids": [e["case"]["id"] for e in examples],
                    "validation_case_ids": [e["case"]["id"] for e in validation],
                    "mined": bool(mined), "log": log, "steps": step,
                    "selection": "validation_hit50_recall50" if validation else "training_loss_smoke"}
        if score > best:
            best = score
            encoder.save(output, metadata)
        atomic_json(Path(output) / "training_log.json", metadata)
        if step >= updates:
            break
    return metadata


def mine_negatives(encoder, cases, store, output, count=30):
    examples = labeled_examples(cases, store, "train")
    rows = {}
    for example in examples:
        symbols = example["symbols"]
        matrix = encoder.encode_symbols(symbols)
        query = encoder.encode_queries([example["query"]])[0]
        similarities = matrix @ query
        order = sorted(range(len(symbols)), key=lambda i: (-float(similarities[i]), symbols[i]["id"]))
        positives = example["positives"]
        dense = [symbols[i]["id"] for i in order if symbols[i]["id"] not in positives][:count]
        sparse = lexical_search(store, example["case"]["snapshot_id"], example["query"], count + 10)
        rows[example["case"]["id"]] = {"dense": dense, "bm25": [
            r["id"] for r in sparse if r["id"] not in positives][:count]}
    report = {"case_manifest": fingerprint(cases.list()), "encoder": encoder.identity,
              "split": "train", "rows": rows}
    atomic_json(Path(output), report)
    return report


def train_cross(base, output, cases, store, config, device="cpu", mined_path=None,
                retrieval_encoder=None):
    seed_everything(config.seed)
    examples = labeled_examples(cases, store, "train")
    validation = cases.labeled({"validation"})
    check_training_groups(examples, [{"case": case} for case, _ in validation])
    if not examples:
        raise ValueError("No training cases")
    mined = json.loads(Path(mined_path).read_text(encoding="utf-8")) if mined_path else None
    if mined and mined["case_manifest"] != fingerprint(cases.list()):
        raise ValueError("Negative mining manifest differs from this corpus")
    reranker = CrossEncoder(base, device, trainable=True, initialize=True)
    if config.gradient_checkpointing:
        reranker.model.gradient_checkpointing_enable()
    queries_per_update = config.micro_batch * config.accumulation
    updates = math.ceil(len(examples) / queries_per_update) * config.epochs
    if config.max_steps:
        updates = min(updates, config.max_steps)
    optimizer, scheduler = optimizer_for(reranker.model, config, updates)
    rng = random.Random(config.seed)
    best, step, log = -1.0, 0, []
    for epoch in range(config.epochs):
        order = list(examples)
        rng.shuffle(order)
        losses = []
        reranker.model.train()
        for start in range(0, len(order), queries_per_update):
            group = order[start:start + queries_per_update]
            optimizer.zero_grad(set_to_none=True)
            for example in group:
                positive = sorted(example["positives"])
                rng.shuffle(positive)
                negative = negative_ids(example, config.negatives, rng, mined)
                ids = [*positive[:4], *negative]
                rng.shuffle(ids)
                logits = torch.stack([reranker.function_score(
                    example["query"], example["by_id"][i]) for i in ids])
                target = torch.tensor([float(i in example["positives"]) for i in ids],
                                      device=reranker.device)
                target /= target.sum()
                loss = -(target * F.log_softmax(logits, dim=0)).sum()
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(reranker.model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            step += 1
            print(json.dumps({"epoch": epoch + 1, "step": step, "loss": losses[-1]}), flush=True)
            if step >= updates:
                break
        validation_metrics = []
        if validation:
            from ..retrieval import SearchEngine
            engine = SearchEngine(store, retrieval_encoder, reranker)
            for case, gold in validation:
                result = engine.search(case["snapshot_id"], case["title"],
                                       case["description"], case["traceback"], SearchConfig())
                validation_metrics.append(ranking_metrics(
                    [r["id"] for r in result["candidates"]], gold["positive_ids"]))
        metrics = ({key: float(np.mean([s[key] for s in validation_metrics]))
                    for key in validation_metrics[0]} if validation_metrics else None)
        score = metrics["hit@5"] if metrics else -float(np.mean(losses))
        log.append({"epoch": epoch + 1, "steps": step, "loss": float(np.mean(losses)),
                    "validation": metrics})
        metadata = {"config": asdict(config), "case_manifest": fingerprint(cases.list()),
                    "train_case_ids": [e["case"]["id"] for e in examples],
                    "validation_case_ids": [c["id"] for c, _ in validation],
                    "steps": step, "log": log,
                    "selection": "validation_hit5" if validation else "training_loss_smoke"}
        if score > best:
            best = score
            reranker.save(output, metadata)
        atomic_json(Path(output) / "training_log.json", metadata)
        if step >= updates:
            break
    return metadata

