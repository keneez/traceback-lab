"""Candidate generation with explicit budgets and auditable provenance."""
from collections import Counter, defaultdict
from dataclasses import asdict
import math
import time

import numpy as np

from .config import SearchConfig, fingerprint
from .query import bind_trace, lexical_search
from .text import query_terms, query_text, terms


def reciprocal_rank_fusion(lists, constant=60):
    scores = defaultdict(float)
    ranks = defaultdict(dict)
    raw = defaultdict(dict)
    for source, rows in lists.items():
        seen = set()
        for rank, row in enumerate(rows, 1):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            scores[row["id"]] += 1 / (constant + rank)
            ranks[row["id"]][source] = rank
            raw[row["id"]][source] = row["score"]
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return ordered, scores, ranks, raw


def expand_graph(seeds, edges, direct_scores):
    adjacency = defaultdict(list)
    for edge in edges:
        if edge["approximate"]:
            continue
        adjacency[edge["source"]].append((edge["target"], edge, "callee"))
        adjacency[edge["target"]].append((edge["source"], edge, "caller"))
    result = {}
    for seed in seeds:
        for target, edge, direction in adjacency.get(seed, []):
            score = direct_scores.get(seed, 1 / 61) / (1 + math.log1p(len(adjacency[seed])))
            item = {"id": target, "score": score, "seed": seed, "edge": edge,
                    "direction": direction}
            if target not in result or score > result[target]["score"]:
                result[target] = item
    return sorted(result.values(), key=lambda r: (-r["score"], r["id"]))


def candidate_ids(direct, trace, graph, config):
    chosen, seen, origin = [], set(), {}

    def append(ids, count, source):
        added = 0
        for symbol_id in ids:
            if len(chosen) >= config.budget or added >= count:
                break
            if symbol_id in seen:
                continue
            chosen.append(symbol_id)
            seen.add(symbol_id)
            origin[symbol_id] = source
            added += 1

    append(trace, config.trace_slots, "traceback")
    append(direct, config.direct_slots, "retrieval")
    append([g["id"] for g in graph], config.graph_slots, "graph")
    append(direct, config.budget, "retrieval")
    append([g["id"] for g in graph], config.budget, "graph")
    return chosen, origin


class SearchEngine:
    def __init__(self, store, encoder=None, reranker=None, calibrator=None):
        self.store = store
        self.encoder = encoder
        self.reranker = reranker
        self.calibrator = calibrator

    def identity(self):
        return {
            "encoder": self.encoder.identity if self.encoder else None,
            "reranker": self.reranker.identity if self.reranker else None,
            "calibrator": self.calibrator.identity if self.calibrator else None,
        }

    def search(self, snapshot_id, title="", description="", traceback="",
               config=None, trace_revision=None, progress=None):
        config = config or SearchConfig()
        progress = progress or (lambda stage: None)
        started = time.perf_counter()
        progress("parsing")
        manifest = self.store.manifest(snapshot_id)
        symbols = self.store.symbols(snapshot_id)
        by_id = {s["id"]: s for s in symbols}
        text = query_text(title, description, traceback)
        if not text.strip():
            raise ValueError("Enter a description or traceback")
        reference = manifest["revision"]
        if reference in {"working-tree", "upload"}:
            reference = snapshot_id
        frames = bind_trace(traceback, symbols, reference, trace_revision)
        trace_ids = list(dict.fromkeys(f["symbol_id"] for f in reversed(frames)
                                      if f["symbol_id"]))
        if config.mode not in {"bm25", "dense", "hybrid", "traceback", "context"}:
            raise ValueError("Unknown retrieval mode")
        progress("retrieving")
        lists, warnings = {}, []
        if config.mode in {"bm25", "hybrid", "context"}:
            lists["bm25"] = lexical_search(self.store, snapshot_id, text, config.raw_top_k)
        if symbols and config.mode in {"dense", "hybrid"}:
            if self.encoder is None:
                if config.mode == "dense":
                    raise ValueError("Neural encoder is not configured. Run setup-models first.")
                warnings.append("Нейросетевой энкодер не загружен. Активен BM25 и граф.")
            else:
                vector_index = self.store.vectors(snapshot_id, self.encoder.identity)
                if vector_index is None:
                    from .vectors import index_vectors
                    progress("indexing")
                    index_vectors(self.store, snapshot_id, self.encoder,
                                  checkpoint=lambda: progress("indexing"))
                    progress("retrieving")
                    vector_index = self.store.vectors(snapshot_id, self.encoder.identity)
                metadata, matrix = vector_index
                query_vector = self.encoder.encode_queries([text])[0]
                scores = np.asarray(matrix @ query_vector)
                ordered = sorted(range(len(scores)),
                                 key=lambda i: (-float(scores[i]), metadata["ids"][i]))
                lists["dense"] = [{"id": metadata["ids"][i], "score": float(scores[i]),
                                   "rank": rank + 1}
                                  for rank, i in enumerate(ordered[:config.raw_top_k])]
        retrieved_at = time.perf_counter()
        direct, rrf, ranks, raw = reciprocal_rank_fusion(lists, config.rrf_constant)
        if config.mode == "traceback":
            direct = trace_ids
        trace_for_budget = trace_ids if config.mode in {"hybrid", "traceback", "context"} else []
        seeds = list(dict.fromkeys([*trace_for_budget, *direct[:config.graph_seeds]]))
        graph_rows = []
        if config.graph and config.mode in {"hybrid", "traceback", "context"}:
            graph_rows = expand_graph(seeds, self.store.edges(snapshot_id), rrf)
        ids, origin = candidate_ids(direct, trace_for_budget, graph_rows, config)
        before_graph = candidate_ids(direct, trace_for_budget, [], config)[0]
        graph_by_id = {r["id"]: r for r in graph_rows}
        query_words = set(query_terms(text))
        candidates = []
        for symbol_id in ids:
            symbol = by_id[symbol_id]
            matching = sorted(query_words & set(terms(
                symbol["qualname"] + " " + symbol["docstring"] + " " + symbol["search_code"])))
            row = {key: symbol[key] for key in [
                "id", "path", "qualname", "name", "start_line", "end_line", "signature"]}
            row.update(rrf_score=rrf.get(symbol_id, 0), ranks=ranks.get(symbol_id, {}),
                       raw_scores=raw.get(symbol_id, {}), origin=origin[symbol_id],
                       terms=matching[:15], trace=[f for f in frames if f["symbol_id"] == symbol_id],
                       graph=graph_by_id.get(symbol_id), rerank_score=None, window=None)
            candidates.append(row)
        # Preserve the trace/direct/graph ordering unless the learned reranker is used.
        progress("reranking")
        before_rerank = [r["id"] for r in candidates]
        reranked = False
        if config.rerank and self.reranker and candidates:
            predictions = self.reranker.score(
                text, [by_id[r["id"]] for r in candidates], frames, config.max_windows)
            for row, prediction in zip(candidates, predictions):
                row.update(prediction)
            candidates.sort(key=lambda r: (-r["rerank_score"], -r["rrf_score"], r["id"]))
            reranked = True
        for rank, row in enumerate(candidates, 1):
            row["rank"] = rank
        result = {
            "snapshot_id": snapshot_id, "repository": manifest["repository"],
            "revision": manifest["revision"], "query": text, "candidates": candidates,
            "before_graph_ids": before_graph, "candidate_ids": ids, "frames": frames,
            "before_rerank_ids": before_rerank,
            "graph_enabled": config.graph and config.mode in {"hybrid", "traceback", "context"},
            "warnings": warnings, "reranked": reranked,
            "methods": list(lists), "config": asdict(config), "models": self.identity(),
            "timings": {"retrieval_ms": round(1000 * (retrieved_at - started), 2),
                        "total_ms": round(1000 * (time.perf_counter() - started), 2)},
            "confidence": {"status": "uncalibrated", "probability": None, "accepted": False,
                           "event": "top1_matches_historical_changed_function"},
        }
        if self.calibrator:
            result["confidence"] = self.calibrator.predict(result)
        return result

