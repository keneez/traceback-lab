"""Import, audit, and temporally split real issue–patch pairs."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile

import httpx

from .cases import CaseStore, atomic_json
from .config import fingerprint
from .indexing import build_snapshot, read_repository
from .labels import map_patch


def fetch_swe_rows(dataset, split, offset=0, limit=50, revision=None):
    if revision:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("Install traceback-lab[data] for revision-pinned dataset imports") from error
        loaded = load_dataset(dataset, split=split, revision=revision)
        return [dict(loaded[i]) for i in range(offset, min(offset + limit, len(loaded)))]
    rows = []
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for start in range(offset, offset + limit, 100):
            response = client.get("https://datasets-server.huggingface.co/rows", params={
                "dataset": dataset, "config": "default", "split": split, "offset": start,
                "length": min(100, offset + limit - start),
            })
            response.raise_for_status()
            chunk = response.json()["rows"]
            rows.extend(item["row"] for item in chunk)
            if not chunk:
                break
    return rows


def ensure_repository(repository, cache):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must have owner/name format")
    directory = Path(cache) / (repository.replace("/", "__") + ".git")
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".clone-", dir=directory.parent))
        try:
            subprocess.run(["git", "clone", "--mirror", f"https://github.com/{repository}.git",
                            str(temporary / "repo.git")], check=True)
            os.rename(temporary / "repo.git", directory)
        finally:
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)
    return directory


def patch_applies(repository, commit, patch):
    with tempfile.TemporaryDirectory() as directory:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        subprocess.run(["git", "-C", str(repository), "read-tree", commit],
                       env=environment, check=True, capture_output=True)
        result = subprocess.run(["git", "-C", str(repository), "apply", "--check", "--cached", "-"],
                                input=patch, text=True, capture_output=True, env=environment)
    return result.returncode == 0, result.stderr.strip()


def audit_issue(text, instance_id):
    number = instance_id.rsplit("-", 1)[-1]
    return {
        "contains_diff": bool(re.search(r"(?m)^(?:diff --git |@@ -\d|--- a/|\+\+\+ b/)", text)),
        "links_own_fix": bool(re.search(rf"/(?:pull|pulls)/{re.escape(number)}\b", text)),
        "contains_solution_language": bool(re.search(
            r"\b(?:fixed by|resolved by|here is the fix)\b", text, re.I)),
    }


def patch_group(repository, patch):
    # Ignore hunk offsets so backports with the same changes share a group.
    canonical = []
    for line in patch.splitlines():
        if line.startswith(("--- ", "+++ ")):
            canonical.append(line)
        elif line.startswith(("+", "-")):
            canonical.append(line[0] + re.sub(r"\s+", "", line[1:]))
    return fingerprint([repository, canonical])[:20]


def prepare_case(row, store, data_root, cache, split="demo", dataset="local",
                 source_revision="unversioned"):
    repository = ensure_repository(row["repo"], cache)
    files, commit = read_repository(repository, row["base_commit"])
    snapshot = build_snapshot(store, files, row["repo"], commit,
                              f"https://github.com/{row['repo']}")
    patch = row["patch"]
    applicable, check_error = patch_applies(repository, commit, patch)
    if not applicable:
        raise ValueError(f"Patch is incompatible with base_commit: {check_error[:250]}")
    labels = map_patch(patch, store.symbols(snapshot["id"]))
    text = row["problem_statement"]
    title, _, body = text.partition("\n")
    trace_start = text.find("Traceback (most recent call last)")
    traceback = text[trace_start:] if trace_start >= 0 else ""
    case_id = row["instance_id"]
    source_id = fingerprint(row)
    case = {
        "id": case_id, "snapshot_id": snapshot["id"], "repository": row["repo"],
        "base_commit": commit, "title": title[:240], "description": body.strip(),
        "traceback": traceback, "trace_revision": None, "split": split,
        "source_dataset": dataset, "source_revision": source_revision,
        "source_instance_id": case_id, "source_record_hash": source_id,
        "official_split": row.get("official_split", "unknown"),
        "issue_url": row.get("issue_url", ""), "created_at": row.get("created_at"),
        "fix_available_at": row.get("fix_available_at", row.get("created_at")),
        "timestamp_certainty": row.get("timestamp_certainty", "source_timestamp_unverified"),
        "historical_text": "unverified", "trace_origin": "issue" if traceback else "absent",
        "task_kind": "unreviewed", "audit": audit_issue(text, case_id),
        "group_id": patch_group(row["repo"], patch),
    }
    symbols = {s["id"]: s for s in store.symbols(snapshot["id"])}
    gold = {
        "case_id": case_id, **labels, "patch": patch,
        "patch_sha256": fingerprint(patch), "patch_applies": True,
        "dynamic_reproduction": "not_run",
        "fix_url": f"https://github.com/{row['repo']}/pull/{case_id.rsplit('-', 1)[-1]}",
        "positive_symbols": [
            {k: symbols[s][k] for k in ["id", "path", "qualname", "start_line", "end_line"]}
            for s in labels["positive_ids"]],
    }
    # Keep the complete source record in a research-only area.
    atomic_json(Path(data_root) / "research" / f"{case_id}.json", row)
    atomic_json(Path(data_root) / "gold" / f"{case_id}.json", gold)
    atomic_json(Path(data_root) / "cases" / f"{case_id}.json", case)
    licenses = Path(data_root) / "licenses" / row["repo"].replace("/", "__")
    licenses.mkdir(parents=True, exist_ok=True)
    for name in ["LICENSE", "LICENSE.txt", "LICENSE.rst", "COPYING", "NOTICE"]:
        response = subprocess.run(["git", "-C", str(repository), "show", f"{commit}:{name}"],
                                  capture_output=True)
        if response.returncode == 0:
            (licenses / name).write_bytes(response.stdout)
    return case, gold


def temporal_split(cases: CaseStore, train_before, validation_before, holdout_repo,
                   calibration_before=None):
    def timestamp(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    train_boundary, validation_boundary = timestamp(train_before), timestamp(validation_before)
    calibration_boundary = timestamp(calibration_before) if calibration_before else None
    if train_boundary >= validation_boundary:
        raise ValueError("Training boundary must precede validation boundary")
    if calibration_boundary and calibration_boundary <= validation_boundary:
        raise ValueError("Calibration boundary must follow validation boundary")
    records = [c for c in cases.raw() if c.get("split") != "demo"]
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    texts = {}
    for case in records:
        key = case["group_id"]
        text_id = fingerprint([case["repository"], case["title"], case["description"]])
        if text_id in texts:
            left, right = find(key), find(texts[text_id])
            parents[max(left, right)] = min(left, right)
        texts[text_id] = key
    groups = defaultdict(list)
    for case in records:
        groups[find(case["group_id"])].append(case)
    counts = defaultdict(int)
    roles = {}
    group_times = {}
    for group_id, group in groups.items():
        observed = [timestamp(c["fix_available_at"]) for c in group if c.get("fix_available_at")]
        latest = max(observed) if len(observed) == len(group) else None
        group_times[group_id] = latest
        if any(c["repository"] == holdout_repo for c in group):
            role = "holdout"
        elif latest is None:
            role = "unassigned"
        elif latest < train_boundary:
            role = "train"
        elif latest < validation_boundary:
            role = "validation"
        elif calibration_boundary and latest < calibration_boundary:
            role = "calibration"
        else:
            role = "test"
        for case in group:
            roles[case["id"]] = {"split": role, "group_id": group_id,
                                 "case_fingerprint": fingerprint(case)}
            counts[role] += 1
    calibration_groups = sorted(
        {entry["group_id"] for entry in roles.values() if entry["split"] == "calibration"},
        key=lambda key: (group_times[key], key))
    fit_groups = set(calibration_groups[:(len(calibration_groups) + 1) // 2])
    for entry in roles.values():
        if entry["split"] == "calibration":
            entry["split"] = ("calibration_fit" if entry["group_id"] in fit_groups
                              else "calibration_threshold")
    counts = defaultdict(int)
    for entry in roles.values():
        counts[entry["split"]] += 1
    protocol = {"train_before": train_before, "validation_before": validation_before,
                "calibration_before": calibration_before, "holdout_repo": holdout_repo,
                "counts": dict(counts), "case_manifest": fingerprint(records), "roles": roles,
                "interpretation": "custom historical function-localization protocol"}
    atomic_json(cases.root / "split_manifest.json", protocol)
    return protocol


def collect_github(repository, numbers, output, token=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must have owner/name format")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Traceback-Lab"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    records = []
    with httpx.Client(timeout=90, follow_redirects=True, headers=headers) as client:
        for number in numbers:
            response = client.get(f"https://api.github.com/repos/{repository}/pulls/{int(number)}")
            response.raise_for_status()
            pull = response.json()
            body = pull.get("body") or ""
            linked = re.search(r"(?i)\b(?:close[sd]?|fix(?:es|ed)?|resolve[sd]?)\s+#(\d+)", body)
            issue = None
            if linked:
                response = client.get(f"https://api.github.com/repos/{repository}/issues/{linked[1]}")
                response.raise_for_status()
                issue = response.json()
            diff = client.get(pull["url"], headers={"Accept": "application/vnd.github.diff"})
            diff.raise_for_status()
            problem = issue or pull
            records.append({
                "repo": repository, "instance_id": repository.replace("/", "__") + f"-{number}",
                "base_commit": pull["base"]["sha"], "fix_commit": pull.get("merge_commit_sha"),
                "problem_statement": problem["title"] + "\n" + (problem.get("body") or ""),
                "patch": diff.text, "issue_url": problem["html_url"],
                "created_at": problem["created_at"], "fix_available_at": pull.get("merged_at"),
                "timestamp_certainty": "github_merge_time" if pull.get("merged_at") else "unknown",
                "issue_text_observed_at": datetime.now(timezone.utc).isoformat(),
                "historical_text": "unverified", "official_split": "custom",
            })
    atomic_json(Path(output), records)
    return len(records)

