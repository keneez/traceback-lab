import argparse
from pathlib import Path
import json
import os
import sys

from .cases import CaseStore, atomic_json
from .config import DATA_ROOT, ROOT, STATE_ROOT, SearchConfig
from .storage import SnapshotStore


def make_engine(args, store):
    from .retrieval import SearchEngine
    from .ml.compact import BUNDLED, CompactEncoder

    selected = getattr(args, "encoder", None) or os.environ.get("TRACEBACK_ENCODER")
    default = ROOT / "models" / "unixcoder"
    if getattr(args, "compact", False):
        path = BUNDLED
    elif selected:
        path = Path(selected)
    elif (default / "config.json").exists() or (default / "current.json").exists():
        path = default
    elif getattr(args, "command", None) == "serve":
        path = BUNDLED
    else:
        path = None
    compact = path is not None and (path / "weights.npz").is_file()
    encoder = reranker = calibrator = None
    if (path and not compact) or getattr(args, "reranker", None):
        import torch
        torch.set_num_threads(getattr(args, "threads", 4))
    if compact:
        encoder = CompactEncoder(path)
    elif path:
        from .ml.encoders import BiEncoder
        encoder = BiEncoder(path, getattr(args, "device", "cpu"))
    if getattr(args, "reranker", None):
        from .ml.reranker import CrossEncoder
        reranker = CrossEncoder(args.reranker, args.device)
    if getattr(args, "calibrator", None):
        from .calibration import Calibrator
        calibrator = Calibrator(args.calibrator)
    return SearchEngine(store, encoder, reranker, calibrator)


def add_models(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--encoder", type=Path)
    group.add_argument("--compact", action="store_true", help="Use the bundled NumPy encoder")
    parser.add_argument("--reranker", type=Path)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="traceback-lab")
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--state", type=Path, default=STATE_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the local API and bundled web interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    add_models(serve)
    index = commands.add_parser("index", help="Index a local git revision or source directory")
    index.add_argument("repository", type=Path)
    index.add_argument("--revision")
    index.add_argument("--name")
    add_models(index)
    setup = commands.add_parser("setup-models", help="Download the pretrained encoders")
    setup.add_argument("--include-codebert", action="store_true")
    setup.add_argument("--output", type=Path, default=ROOT / "models")
    setup.add_argument("--unixcoder-revision", default="5604afdc964f6c53782a6813140ade5216b99006")
    importer = commands.add_parser("import-swe", help="Prepare historical issue–patch cases")
    importer.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    importer.add_argument("--revision")
    importer.add_argument("--split", default="test")
    importer.add_argument("--offset", type=int, default=0)
    importer.add_argument("--limit", type=int, default=50)
    importer.add_argument("--repos", default="")
    importer.add_argument("--role", default="unassigned")
    importer.add_argument("--cache", type=Path, default=STATE_ROOT / "repositories")
    local_import = commands.add_parser("import-json", help="Import source rows from a JSON file")
    local_import.add_argument("file", type=Path)
    local_import.add_argument("--role", default="unassigned")
    local_import.add_argument("--cache", type=Path, default=STATE_ROOT / "repositories")
    collector = commands.add_parser("collect-github", help="Collect public issue–PR rows")
    collector.add_argument("repository")
    collector.add_argument("--pulls", required=True)
    collector.add_argument("--output", required=True, type=Path)
    collector.add_argument("--token-env", help="Optional environment variable with an API token")
    split = commands.add_parser("split", help="Create a grouped temporal localization protocol")
    split.add_argument("--train-before", required=True)
    split.add_argument("--validation-before", required=True)
    split.add_argument("--calibration-before")
    split.add_argument("--holdout-repo", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--splits", default="test")
    evaluate.add_argument("--exclude-cases", type=Path, help="Exclude cases and fix groups from this cases directory")
    evaluate.add_argument("--modes", default="bm25,hybrid")
    evaluate.add_argument("--profile", choices=["quality", "fast"], default="quality")
    evaluate.add_argument("--no-graph", action="store_true")
    evaluate.add_argument("--no-rerank", action="store_true")
    evaluate.add_argument("--budget", choices=[20, 50, 100], type=int)
    evaluate.add_argument("--output", type=Path, default=ROOT / "reports" / "evaluation.json")
    add_models(evaluate)
    compare = commands.add_parser("compare")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--mode", default="hybrid")
    compare.add_argument("--metric", default="hit@5")
    for name in ["train-bi", "train-cross"]:
        trainer = commands.add_parser(name)
        trainer.add_argument("--base", required=True, type=Path)
        trainer.add_argument("--output", required=True, type=Path)
        trainer.add_argument("--negatives", type=Path)
        trainer.add_argument("--epochs", type=int, default=2)
        trainer.add_argument("--micro-batch", type=int, default=1)
        trainer.add_argument("--accumulation", type=int, default=4)
        trainer.add_argument("--negative-count", type=int, default=12)
        trainer.add_argument("--learning-rate", type=float, default=2e-5)
        trainer.add_argument("--seed", type=int, default=31)
        trainer.add_argument("--max-steps", type=int)
        trainer.add_argument("--no-gradient-checkpointing", action="store_true")
        add_models(trainer)
    miner = commands.add_parser("mine")
    miner.add_argument("--output", required=True, type=Path)
    add_models(miner)
    calibration_records = commands.add_parser("calibration-records")
    calibration_records.add_argument("--split", required=True)
    calibration_records.add_argument("--output", required=True, type=Path)
    add_models(calibration_records)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--fit", required=True, type=Path)
    calibrate.add_argument("--threshold", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)
    commands.add_parser("audit")
    commands.add_parser("doctor")
    args = parser.parse_args(argv)
    store = SnapshotStore(args.data / "snapshots")
    cases = CaseStore(args.data)
    try:
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            engine = make_engine(args, store)
            app = create_app(args.data, args.state, engine)
            uvicorn.run(app, host=args.host, port=args.port)
        elif args.command == "index":
            from .indexing import build_snapshot, read_repository
            files, revision = read_repository(args.repository, args.revision)
            manifest = build_snapshot(store, files, args.name or args.repository.name, revision)
            engine = make_engine(args, store)
            if engine.encoder:
                from .vectors import index_vectors
                index_vectors(store, manifest["id"], engine.encoder)
            print(json.dumps(manifest, indent=2))
        elif args.command == "setup-models":
            from .ml.encoders import download_checkpoint
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            print(download_checkpoint("microsoft/unixcoder-base", args.output / "unixcoder",
                                      args.unixcoder_revision))
            if args.include_codebert:
                print(download_checkpoint("microsoft/codebert-base", args.output / "codebert"))
        elif args.command in {"import-swe", "import-json"}:
            from .corpus import fetch_swe_rows, prepare_case
            if args.command == "import-swe":
                rows = fetch_swe_rows(args.dataset, args.split, args.offset, args.limit, args.revision)
                repos = set(args.repos.split(",")) - {""}
                if repos:
                    rows = [r for r in rows if r["repo"] in repos]
                dataset, revision = args.dataset, args.revision or "unversioned-content-hashed"
                for row in rows:
                    row["official_split"] = args.split
            else:
                rows = json.loads(args.file.read_text(encoding="utf-8"))
                dataset, revision = "local-json", "content-hashed"
            if not rows:
                raise ValueError("No source records matched the requested selection")
            report = []
            for row in rows:
                try:
                    case, gold = prepare_case(row, store, args.data, args.cache, args.role,
                                              dataset, revision)
                    item = {"id": case["id"], "status": "prepared",
                            "positives": len(gold["positive_ids"]),
                            "mapping_coverage": gold["mapping_coverage"]}
                except Exception as error:
                    item = {"id": row.get("instance_id"), "status": "failed", "error": str(error)}
                report.append(item)
                print(json.dumps(item), flush=True)
            atomic_json(args.data / "import_report.json", report)
            failed = sum(item["status"] == "failed" for item in report)
            if failed:
                parser.exit(1, f"{failed} of {len(report)} imports failed. See import_report.json.\n")
        elif args.command == "split":
            from .corpus import temporal_split
            result = temporal_split(cases, args.train_before, args.validation_before,
                                    args.holdout_repo, args.calibration_before)
            print(json.dumps(result, indent=2))
        elif args.command == "collect-github":
            from .corpus import collect_github
            count = collect_github(args.repository, args.pulls.split(","), args.output,
                                   os.environ.get(args.token_env) if args.token_env else None)
            print(f"Collected {count} source records")
        elif args.command == "evaluate":
            from .evaluation import evaluate
            excluded = None
            if args.exclude_cases:
                paths = sorted(args.exclude_cases.glob("*.json"))
                if not paths:
                    raise ValueError("No case manifests found for exclusion")
                excluded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            result = evaluate(make_engine(args, store), cases, args.splits.split(","),
                              args.modes.split(","), args.profile, not args.no_graph, args.output,
                              budget=args.budget, rerank=not args.no_rerank, exclude_cases=excluded)
            print(json.dumps(result["summaries"], indent=2))
        elif args.command == "compare":
            from .evaluation import compare_reports
            result = compare_reports(json.loads(args.first.read_text(encoding="utf-8")),
                                     json.loads(args.second.read_text(encoding="utf-8")), args.mode, args.metric)
            print(json.dumps(result, indent=2))
        elif args.command in {"train-bi", "train-cross"}:
            import torch
            from .ml.training import TrainConfig, train_bi, train_cross
            torch.set_num_threads(args.threads)
            config = TrainConfig(
                epochs=args.epochs, micro_batch=args.micro_batch, accumulation=args.accumulation,
                negatives=args.negative_count, learning_rate=args.learning_rate, seed=args.seed,
                max_steps=args.max_steps, gradient_checkpointing=not args.no_gradient_checkpointing)
            if min(config.epochs, config.micro_batch, config.accumulation, config.negatives) < 1:
                raise ValueError("Training counts must be positive")
            if args.command == "train-bi":
                train_bi(args.base, args.output, cases, store, config, args.device, args.negatives)
            else:
                encoder = make_engine(args, store).encoder if (args.encoder or args.compact) else None
                train_cross(args.base, args.output, cases, store, config, args.device,
                            args.negatives, encoder)
        elif args.command == "mine":
            from .ml.training import mine_negatives
            engine = make_engine(args, store)
            if not engine.encoder:
                raise ValueError("Provide --encoder")
            mine_negatives(engine.encoder, cases, store, args.output)
        elif args.command == "calibration-records":
            engine = make_engine(args, store)
            records = []
            for case, gold in cases.labeled({args.split}):
                result = engine.search(case["snapshot_id"], case["title"], case["description"],
                                       case["traceback"])
                first = result["candidates"][0]["id"] if result["candidates"] else None
                records.append({"case_id": case["id"], "group_id": case["group_id"],
                                "correct": first in gold["positive_ids"], "result": result})
            atomic_json(args.output, records)
        elif args.command == "calibrate":
            from .calibration import fit_calibrator
            result = fit_calibrator(json.loads(args.fit.read_text(encoding="utf-8")),
                                    json.loads(args.threshold.read_text(encoding="utf-8")), args.output)
            print(json.dumps(result["threshold_selection"], indent=2))
        elif args.command == "audit":
            from collections import Counter
            result = {"case_count": len(cases.list()),
                      "repositories": dict(Counter(c["repository"] for c in cases.list())),
                      "splits": dict(Counter(c["split"] for c in cases.list())),
                      "historical_text": dict(Counter(c["historical_text"] for c in cases.list())),
                      "cases": [{"id": c["id"], "audit": c["audit"],
                                 "mapping": cases.reveal(c["id"])["mapping_coverage"]}
                                for c in cases.list()]}
            print(json.dumps(result, indent=2))
        elif args.command == "doctor":
            import importlib.util
            print(json.dumps({"python": sys.version, "data": str(args.data),
                              "snapshots": len(store.list()), "cases": len(cases.list()),
                              "torch": bool(importlib.util.find_spec("torch")),
                              "transformers": bool(importlib.util.find_spec("transformers")),
                              "web_built": (Path(__file__).parent / "web" / "index.html").exists()},
                             indent=2))
    except (ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
