from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import platform
import os
import time

import numpy as np

from .cases import atomic_json
from .config import SearchConfig, fingerprint


def ranking_metrics(ids, positives):
    ids = list(dict.fromkeys(ids))
    positives = set(positives)
    if not positives:
        raise ValueError("Ranking metrics require at least one supported positive")
    result = {}
    for k in (1, 5, 20, 50):
        hits = len(set(ids[:k]) & positives)
        result[f"hit@{k}"] = float(hits > 0)
        result[f"recall@{k}"] = hits / len(positives)
    rank = next((i for i, symbol_id in enumerate(ids, 1) if symbol_id in positives), None)
    result["mrr@50"] = 1 / rank if rank and rank <= 50 else 0
    result["mrr@5"] = 1 / rank if rank and rank <= 5 else 0
    return result


def grouped_bootstrap(records, metric, iterations=1000, seed=31):
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record["metrics"][metric])
    values = list(groups.values())
    if not values:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        means.append(float(np.mean([value for i in sample for value in values[i]])))
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def evaluate(engine, cases, splits, modes, profile="quality", graph=True, output=None, budget=None,
             rerank=True, exclude_cases=None):
    excluded = exclude_cases or []
    excluded_ids = {case["id"] for case in excluded}
    excluded_groups = {case["group_id"] for case in excluded if case.get("group_id")}
    pairs = [(case, gold) for case, gold in cases.labeled(set(splits))
             if case["id"] not in excluded_ids and case.get("group_id") not in excluded_groups]
    if not pairs:
        raise ValueError("No labeled cases in the requested splits")
    records = []
    for case, gold in pairs:
        for mode in modes:
            config = SearchConfig.profile(profile, mode, graph=graph)
            config = replace(config, rerank=rerank)
            if budget:
                config = replace(config, budget=budget, raw_top_k=max(60, budget),
                                 trace_slots=max(1, budget // 5), direct_slots=budget * 3 // 5,
                                 graph_slots=max(1, budget // 5))
            started = time.perf_counter()
            result = engine.search(case["snapshot_id"], case["title"], case["description"],
                                   case["traceback"], config, case.get("trace_revision"))
            ids = [r["id"] for r in result["candidates"]]
            metrics = ranking_metrics(ids, gold["positive_ids"])
            confidence = result["confidence"]
            stack_ids = {f["symbol_id"] for f in result["frames"] if f["symbol_id"]}
            has_target = bool(set(result["candidate_ids"]) & set(gold["positive_ids"]))
            records.append({
                "case_id": case["id"], "group_id": case["group_id"], "split": case["split"],
                "repository": case["repository"], "mode": mode,
                "methods": result["methods"], "config": result["config"],
                "models": result["models"], "metrics": metrics,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "candidate_ids": ids, "positive_ids": gold["positive_ids"],
                "confidence": confidence, "historical_text": case["historical_text"],
                "trace_origin": case["trace_origin"], "mapping_scope": gold["scope"],
                "has_target_in_candidates": has_target,
                "before_rerank_metrics": ranking_metrics(result["before_rerank_ids"],
                                                         gold["positive_ids"]),
                "slices": {
                    "with_traceback": bool(case["traceback"]),
                    "without_traceback": not case["traceback"],
                    "target_outside_stack": bool(case["traceback"]) and not bool(
                        stack_ids & set(gold["positive_ids"])),
                    "multi_positive": len(gold["positive_ids"]) > 1,
                    "verified_historical_text": case["historical_text"] == "verified_before_fix",
                },
            })
    summaries = {}
    for mode in modes:
        subset = [r for r in records if r["mode"] == mode]
        scores = {name: float(np.mean([r["metrics"][name] for r in subset]))
                  for name in subset[0]["metrics"]}
        durations = [r["elapsed_ms"] for r in subset]
        accepted = [r for r in subset if r["confidence"]["accepted"]]
        eligible_rerank = [r for r in subset if r["has_target_in_candidates"]]
        repository_scores = defaultdict(list)
        for row in subset:
            repository_scores[row["repository"]].append(row["metrics"]["hit@5"])
        scores.update(count=len(subset), independent_groups=len({r["group_id"] for r in subset}),
                      latency_p50_ms=float(np.median(durations)),
                      latency_p95_ms=float(np.quantile(durations, .95)),
                      coverage=len(accepted) / len(subset),
                      selective_accuracy=float(np.mean([r["metrics"]["hit@1"] for r in accepted]))
                      if accepted else None,
                      hit5_ci=grouped_bootstrap(subset, "hit@5"),
                      macro_repository_hit5=float(np.mean(
                          [np.mean(values) for values in repository_scores.values()])),
                      conditional_hit5=float(np.mean(
                          [r["metrics"]["hit@5"] for r in eligible_rerank]))
                          if eligible_rerank else None)
        calibrated = [r for r in subset if r["confidence"]["probability"] is not None]
        if calibrated:
            probabilities = np.asarray([r["confidence"]["probability"] for r in calibrated])
            truth = np.asarray([r["metrics"]["hit@1"] for r in calibrated])
            scores["brier_score"] = float(np.mean((probabilities - truth) ** 2))
            reliability = []
            for bin_index in range(10):
                left, right = bin_index / 10, (bin_index + 1) / 10
                selected = ((probabilities >= left) &
                            ((probabilities < right) if bin_index < 9 else (probabilities <= right)))
                reliability.append({"left": left, "count": int(selected.sum()),
                                    "probability": float(probabilities[selected].mean())
                                    if selected.any() else None,
                                    "accuracy": float(truth[selected].mean())
                                    if selected.any() else None})
            scores["reliability"] = reliability
        summaries[mode] = scores
    processor = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not processor and cpuinfo.exists():
        processor = next((line.split(":", 1)[1].strip()
                          for line in cpuinfo.read_text(encoding="utf-8").splitlines()
                          if line.startswith("model name")), platform.machine())
    report = {"interpretation": "custom historical function localization",
              "splits": splits, "profile": profile, "graph": graph,
              "case_manifest": fingerprint(cases.list()), "models": engine.identity(),
              "encoder_name": getattr(engine.encoder, "name", None),
              "excluded_case_ids": sorted(excluded_ids),
              "hardware": {"platform": platform.platform(), "processor": processor,
                           "logical_cpus": os.cpu_count(),
                           "device": str(engine.encoder.device) if engine.encoder else "cpu",
                           "torch_threads": __import__("torch").get_num_threads()
                           if getattr(engine.encoder, "model", None) is not None or engine.reranker else None},
              "timing_scope": "search plus any missing vector index construction; model load excluded",
              "summaries": summaries, "records": records}
    if output:
        atomic_json(Path(output), report)
    return report


def compare_reports(first, second, mode="hybrid", metric="hit@5", seed=31, iterations=2000):
    left = {r["case_id"]: r for r in first["records"] if r["mode"] == mode}
    right = {r["case_id"]: r for r in second["records"] if r["mode"] == mode}
    if not left or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical case IDs and a shared mode")
    groups = defaultdict(list)
    for case_id, row in left.items():
        other = right[case_id]
        if row["config"]["budget"] != other["config"]["budget"]:
            raise ValueError("Candidate budgets differ")
        groups[row["group_id"]].append(other["metrics"][metric] - row["metrics"][metric])
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sample = rng.integers(0, len(values), len(values))
        differences.append(float(np.mean([v for i in sample for v in values[i]])))
    return {"metric": metric, "mode": mode, "cases": len(left), "groups": len(values),
            "difference_second_minus_first": float(np.mean([v for group in values for v in group])),
            "paired_group_bootstrap_ci95": np.quantile(differences, [.025, .975]).tolist(),
            "seed": seed, "iterations": iterations}

