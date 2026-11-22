"""Portable logistic calibration for a fixed retrieval/reranking pipeline."""
from pathlib import Path
import json

import numpy as np

from .config import fingerprint

FEATURES = ["top_score", "margin", "bm25_rank", "dense_rank", "agreement",
            "trace_confirmed", "trace_approximate", "function_lines", "query_chars"]


def pipeline_identity(result):
    return fingerprint({"encoder": result["models"]["encoder"],
                        "reranker": result["models"]["reranker"], "config": result["config"]})


def feature_vector(result):
    rows = result["candidates"]
    if not rows:
        return np.zeros(len(FEATURES))
    first = rows[0]
    score = lambda row: row["rerank_score"] if row["rerank_score"] is not None else row["rrf_score"]
    frames = first["trace"]
    return np.asarray([
        score(first), score(first) - score(rows[1]) if len(rows) > 1 else 0,
        1 / first["ranks"].get("bm25", 1000), 1 / first["ranks"].get("dense", 1000),
        float(len(first["ranks"]) > 1),
        float(any(f["status"] == "confirmed" for f in frames)),
        float(any(f["status"] == "approximate" for f in frames)),
        np.log1p(first["end_line"] - first["start_line"] + 1),
        np.log1p(len(result["query"])),
    ], dtype=np.float64)


class Calibrator:
    def __init__(self, path):
        self.metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        self.identity = fingerprint(self.metadata)[:24]

    def predict(self, result):
        outcome = {"status": "incompatible", "probability": None, "accepted": False,
                   "event": "top1_matches_historical_changed_function"}
        if self.metadata["pipeline"] != pipeline_identity(result):
            return outcome
        if not result["candidates"]:
            return dict(outcome, status="empty")
        m = self.metadata
        x = (feature_vector(result) - np.asarray(m["mean"])) / np.asarray(m["scale"])
        logit = float(x @ np.asarray(m["coefficients"]) + m["intercept"])
        probability = float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))
        return dict(outcome, status="calibrated", probability=probability,
                    accepted=probability >= m["threshold"])


def fit_calibrator(fit_records, threshold_records, output, target_accuracy=.8, min_coverage=.3):
    fit_ids = {r["case_id"] for r in fit_records}
    threshold_ids = {r["case_id"] for r in threshold_records}
    if fit_ids & threshold_ids:
        raise ValueError("Calibration fit and threshold cases overlap")
    fit_groups = {r.get("group_id", r["case_id"]) for r in fit_records}
    threshold_groups = {r.get("group_id", r["case_id"]) for r in threshold_records}
    if fit_groups & threshold_groups:
        raise ValueError("Calibration fit and threshold groups overlap")
    identities = {pipeline_identity(r["result"]) for r in [*fit_records, *threshold_records]}
    if len(identities) != 1:
        raise ValueError("All calibration records must use the same pipeline")
    if not fit_records or not threshold_records:
        raise ValueError("Provide both calibration-fit and threshold datasets")
    x = np.stack([feature_vector(r["result"]) for r in fit_records])
    y = np.asarray([int(r["correct"]) for r in fit_records])
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration requires both correct and incorrect predictions")
    from sklearn.linear_model import LogisticRegression
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    model = LogisticRegression(C=1, max_iter=1000, random_state=31).fit((x - mean) / scale, y)
    tx = np.stack([feature_vector(r["result"]) for r in threshold_records])
    probabilities = model.predict_proba((tx - mean) / scale)[:, 1]
    truth = np.asarray([int(r["correct"]) for r in threshold_records])
    threshold, selection = 1.000001, None
    for candidate in sorted(set(probabilities)):
        accepted = probabilities >= candidate
        coverage, accuracy = float(accepted.mean()), float(truth[accepted].mean())
        if coverage >= min_coverage and accuracy >= target_accuracy:
            threshold = float(candidate)
            selection = {"coverage": coverage, "accuracy": accuracy, "count": int(accepted.sum())}
            break
    metadata = {"pipeline": next(iter(identities)), "features": FEATURES,
                "mean": mean.tolist(), "scale": scale.tolist(),
                "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                "threshold": threshold, "threshold_selection": selection,
                "fit_case_ids": sorted(fit_ids), "threshold_case_ids": sorted(threshold_ids),
                "event": "top1_matches_historical_changed_function"}
    from .cases import atomic_json
    atomic_json(Path(output), metadata)
    return metadata

