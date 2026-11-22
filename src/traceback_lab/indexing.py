"""Build immutable indexes, publishing complete directories atomically."""
from dataclasses import asdict
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import tokenize
import tarfile

from .config import PARSER_VERSION, SCHEMA_VERSION, fingerprint
from .parser import build_graph, parse_file
from .storage import SnapshotStore
from .text import lexical_text

EXCLUDED = frozenset({"tests", "test", "testing", "vendor", "vendored", "node_modules",
                      ".venv", "venv", "__pycache__", "build", "dist", ".git", "migrations",
                      "_vendor"})
EXCLUDED_PREFIXES = ("requests/packages/", "src/requests/packages/")


def in_scope(path):
    parts = PurePosixPath(path).parts
    return (path.endswith(".py") and not any(p in EXCLUDED for p in parts)
            and not parts[-1].startswith("test_") and not parts[-1].endswith("_test.py")
            and not path.startswith(EXCLUDED_PREFIXES))


def read_repository(path: Path, revision: str | None = None):
    path = path.resolve()
    if revision:
        if revision.startswith("-"):
            raise ValueError("Revision must be a commit reference")
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse",
                                          "--verify", f"{revision}^{{commit}}"], text=True).strip()
        files = {}
        process = subprocess.Popen(["git", "-C", str(path), "archive", "--format=tar", commit],
                                   stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    if member.isfile() and in_scope(member.name):
                        stream = archive.extractfile(member)
                        if stream:
                            files[member.name] = stream.read()
            if process.wait() != 0:
                raise RuntimeError("git archive failed")
        finally:
            if process.stdout:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        return files, commit
    files = {}
    for item in sorted(path.rglob("*.py")):
        relative = item.relative_to(path).as_posix()
        if in_scope(relative) and item.is_file() and not item.is_symlink():
            files[relative] = item.read_bytes()
    return files, "working-tree"


def build_snapshot(store: SnapshotStore, files: dict[str, bytes], repository: str,
                   revision: str, source_url: str = "", *, allow_empty: bool = False):
    files = {p: content for p, content in sorted(files.items()) if in_scope(p)}
    hashes = {p: hashlib.sha256(content).hexdigest() for p, content in files.items()}
    snapshot_id = fingerprint([repository, revision, hashes, PARSER_VERSION, SCHEMA_VERSION])[:24]
    final = store.root / snapshot_id
    if (final / "manifest.json").exists():
        return store.manifest(snapshot_id)
    temporary = Path(tempfile.mkdtemp(prefix=".index-", dir=store.root))
    symbols, aliases, source_files, errors = [], {}, {}, []
    try:
        for path, raw in files.items():
            try:
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                text = raw.decode(encoding)
                source_files[path] = (text, encoding)
                found, imports, parse_errors = parse_file(path, text, snapshot_id)
                symbols.extend(s for s in found if not s.parse_error)
                aliases[path] = imports
                errors.extend({"path": path, **e} for e in parse_errors)
            except (UnicodeError, SyntaxError, LookupError, RecursionError) as error:
                errors.append({"path": path, "kind": type(error).__name__, "message": str(error)})
        if allow_empty and not source_files:
            raise ValueError("Не удалось прочитать исходники. Проверьте кодировку Python-файлов.")
        if not symbols and not allow_empty:
            raise ValueError("Snapshot contains no supported Python functions")
        edges = build_graph(symbols, aliases)
        db = sqlite3.connect(temporary / "index.sqlite")
        try:
            db.executescript("""
                CREATE TABLE symbols(id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE files(path TEXT PRIMARY KEY, content TEXT NOT NULL, encoding TEXT);
                CREATE TABLE edges(source TEXT, target TEXT, line INTEGER, data TEXT);
                CREATE INDEX edge_source ON edges(source);
                CREATE INDEX edge_target ON edges(target);
                CREATE VIRTUAL TABLE lexical USING fts5(
                    id UNINDEXED, path, name, signature, docstring, messages, body,
                    tokenize="unicode61 tokenchars '_'");
                CREATE VIRTUAL TABLE terms USING fts5vocab(lexical, 'row');
            """)
            db.executemany("INSERT INTO files VALUES(?,?,?)",
                           [(p, t, e) for p, (t, e) in source_files.items()])
            for symbol in symbols:
                db.execute("INSERT INTO symbols VALUES(?,?)",
                           (symbol.id, json.dumps(asdict(symbol), ensure_ascii=False)))
                # Literal strings preserve exact diagnostics while identifiers are split.
                messages = " ".join(line for line in symbol.search_code.splitlines()
                                    if "raise " in line or "Error(" in line or "warn(" in line)
                db.execute("INSERT INTO lexical VALUES(?,?,?,?,?,?,?)",
                           (symbol.id, lexical_text(symbol.path), lexical_text(symbol.qualname),
                            lexical_text(symbol.signature), lexical_text(symbol.docstring),
                            lexical_text(messages), lexical_text(symbol.search_code)))
            db.executemany("INSERT INTO edges VALUES(?,?,?,?)",
                           [(e["source"], e["target"], e["line"], json.dumps(e)) for e in edges])
            db.commit()
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Index integrity check failed")
        finally:
            db.close()
        manifest = {"id": snapshot_id, "repository": repository, "revision": revision,
                    "source_url": source_url, "tree_hash": fingerprint(hashes),
                    "parser_version": PARSER_VERSION, "schema_version": SCHEMA_VERSION,
                    "symbol_count": len(symbols), "file_count": len(source_files),
                    "edge_count": len(edges), "parse_errors": errors,
                    "excluded_parts": sorted(EXCLUDED),
                    "excluded_prefixes": EXCLUDED_PREFIXES, "file_hashes": hashes}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.rename(temporary, final)
        except OSError:
            if not (final / "manifest.json").exists():
                raise
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

