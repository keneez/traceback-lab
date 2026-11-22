from pathlib import PurePosixPath
import re

from .text import query_terms

FRAME = re.compile(r'File ["\'](?P<path>.+?)["\'], line (?P<line>\d+), in (?P<name>[^\r\n]+)')


def bind_trace(traceback, symbols, revision, trace_revision=None):
    paths = sorted({s["path"] for s in symbols})
    frames = []
    for match in FRAME.finditer(traceback):
        item = match.groupdict()
        line = int(item["line"])
        path = item["path"].replace("\\", "/")
        candidates = [p for p in paths if path == p or path.endswith("/" + p)]
        if not candidates:
            candidates = [p for p in paths if PurePosixPath(path).name == PurePosixPath(p).name]
        frame = {"path": item["path"], "line": line, "name": item["name"].strip(),
                 "symbol_id": None, "status": "external"}
        if len(candidates) > 1:
            frame["status"] = "ambiguous"
        elif candidates:
            local = candidates[0]
            inside = [s for s in symbols if s["path"] == local
                      and s["start_line"] <= line <= s["end_line"]]
            inside.sort(key=lambda s: (s["end_line"] - s["start_line"], s["id"]))
            name = frame["name"]
            target = inside[0] if inside else None
            if target and name not in {target["name"], target["qualname"]}:
                target = None
            if target is None:
                named = [s for s in symbols if s["path"] == local
                         and name in {s["name"], s["qualname"]}]
                if len(named) == 1:
                    target = named[0]
            if target:
                frame.update(symbol_id=target["id"], local_path=local, status="approximate")
                if target in inside and trace_revision == revision:
                    frame["status"] = "confirmed"
            else:
                frame["status"] = "unresolved"
        frames.append(frame)
    return frames


def lexical_search(store, snapshot_id, text, limit=60):
    candidates = query_terms(text, limit=100)
    if not candidates:
        return []
    with store.connect(snapshot_id) as db:
        placeholders = ",".join("?" for _ in candidates)
        frequencies = dict(db.execute(
            f"SELECT term,doc FROM terms WHERE term IN ({placeholders})", candidates))
        # Preserve rare identifiers and exception terms, then break ties deterministically.
        chosen = sorted(candidates, key=lambda t: (frequencies.get(t, 10**9), -len(t), t))[:60]
        query = " OR ".join('"' + term.replace('"', '""') + '"' for term in chosen)
        rows = db.execute("""
            SELECT id,bm25(lexical,0,2,4,4,1,2,1) AS score
            FROM lexical WHERE lexical MATCH ?
            ORDER BY score ASC,id ASC LIMIT ?
        """, (query, limit)).fetchall()
    return [{"id": r["id"], "score": r["score"], "rank": i + 1}
            for i, r in enumerate(rows)]

