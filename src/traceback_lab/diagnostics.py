"""Small, deterministic source checks independent of the learned retrieval pipeline."""
import ast
from collections import Counter
import warnings

MAX_FINDINGS = 400
RULES = {
    "PY001": ("Ошибка синтаксиса", "error"),
    "PY002": ("Изменяемое значение по умолчанию", "warning"),
    "PY003": ("Повторяющийся ключ словаря", "warning"),
    "PY004": ("Кортеж в условии всегда истинен", "warning"),
    "PY005": ("Исключение молча пропускается", "warning"),
}


def mutable_literal(node):
    if isinstance(node, (ast.List, ast.Set, ast.Dict, ast.ListComp, ast.SetComp, ast.DictComp)):
        return True
    return isinstance(node, ast.Tuple) and any(mutable_literal(item) for item in node.elts)


def scan_source(path, source):
    def finding(rule, line, message):
        title, level = RULES[rule]
        return {"rule": rule, "level": level, "title": title, "path": path,
                "line": line, "message": message}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError) as exc:
        yield finding("PY001", getattr(exc, "lineno", 1) or 1, getattr(exc, "msg", str(exc)))
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for value in [*args.defaults, *args.kw_defaults]:
                if mutable_literal(value):
                    yield finding("PY002", value.lineno,
                                  "Объект создаётся один раз и может сохранять изменения между вызовами.")
        elif isinstance(node, ast.Dict):
            seen = {}
            for key in node.keys:
                if not isinstance(key, ast.Constant):
                    continue
                if key.value in seen:
                    yield finding("PY003", key.lineno,
                                  f"Этот ключ уже задан в строке {seen[key.value]}. Последнее значение заменяет первое.")
                seen[key.value] = key.lineno
        elif isinstance(node, (ast.If, ast.While, ast.Assert)):
            test = node.test
            if isinstance(test, ast.Tuple) and any(not isinstance(item, ast.Starred) for item in test.elts):
                yield finding("PY004", test.lineno,
                              "Непустой кортеж имеет истинное значение независимо от значений его элементов.")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                yield finding("PY005", node.lineno,
                              "Пустой обработчик перехватывает также KeyboardInterrupt и SystemExit.")


def inspect_snapshot(store, snapshot_id):
    manifest = store.manifest(snapshot_id)
    rows, counts, paths = [], Counter(), set()
    for path, source in store.sources(snapshot_id):
        paths.add(path)
        for item in sorted(scan_source(path, source), key=lambda row: (row["line"], row["rule"])):
            counts[item["level"]] += 1
            if len(rows) < MAX_FINDINGS:
                rows.append(item)
    return {"snapshot_id": snapshot_id, "findings": rows, "total": sum(counts.values()),
            "counts": counts, "files_scanned": len(paths), "rule_count": len(RULES),
            "unreadable": sorted(set(manifest.get("file_hashes", {})) - paths)}
