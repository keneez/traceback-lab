"""Map changed OLD coordinates to the innermost existing Python function."""
import re
import ast
import textwrap
import warnings

from unidiff import PatchSet


def semantic_tree(code):
    lines = code.splitlines(True)
    if lines and lines[0].lstrip().startswith("@"):
        definition = next((line for line in lines if re.match(r"\s*(?:async\s+)?def ", line)), "")
        indent = len(definition) - len(definition.lstrip())
        lines[0] = " " * indent + lines[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(textwrap.dedent("".join(lines)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    value = node.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        node.body = node.body[1:]
        return ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError):
        return None


def owner(symbols, line):
    found = [s for s in symbols if s["start_line"] <= line <= s["end_line"]]
    return min(found, key=lambda s: (s["end_line"] - s["start_line"], s["id"])) if found else None


def map_patch(patch, symbols):
    by_path = {}
    for symbol in symbols:
        by_path.setdefault(symbol["path"], []).append(symbol)
    positives, changes = set(), []
    parsed = PatchSet(patch)
    for file in parsed:
        path = file.source_file.removeprefix("a/")
        targets = by_path.get(path, [])
        for hunk in file:
            old_line = hunk.source_start
            index = 0
            lines = list(hunk)
            while index < len(lines):
                line = lines[index]
                if line.is_context:
                    old_line += 1
                    index += 1
                    continue
                start = old_line
                removed, added = [], []
                while index < len(lines) and not lines[index].is_context:
                    current = lines[index]
                    if current.is_removed:
                        removed.append((old_line, current.value))
                        old_line += 1
                    elif current.is_added:
                        added.append(current.value)
                    index += 1
                if not removed and not added:
                    continue
                ids = set()
                category = "unmapped"
                nonempty = [v.strip() for _, v in removed] + [v.strip() for v in added]
                auxiliary = all(not v or v.startswith("#") for v in nonempty)
                if auxiliary:
                    category = "auxiliary"
                elif removed:
                    for position, _ in removed:
                        target = owner(targets, position)
                        if target:
                            ids.add(target["id"])
                    category = "function_change" if ids else "outside_function"
                else:
                    before, after = owner(targets, start - 1), owner(targets, start)
                    additions = "".join(added)
                    if re.search(r"(?m)^\s*(?:async\s+)?(?:def|class)\s+\w+", additions):
                        category = "added_symbol"
                    elif before and after and before["id"] == after["id"]:
                        ids.add(before["id"])
                        category = "function_insertion"
                    elif after and additions.lstrip().startswith("@") and start == after["start_line"]:
                        ids.add(after["id"])
                        category = "decorator_insertion"
                    elif before:
                        def_line = before["code"].splitlines()[0]
                        def_indent = len(def_line) - len(def_line.lstrip())
                        first = next((v for v in added if v.strip()), "")
                        indent = len(first) - len(first.lstrip())
                        if first and indent > def_indent and start <= before["end_line"] + 1:
                            ids.add(before["id"])
                            category = "function_boundary_insertion"
                positives.update(ids)
                changes.append({"path": path, "old_start": start,
                                "old_count": len(removed), "new_count": len(added),
                                "category": category, "symbol_ids": sorted(ids),
                                "_added": added})
    by_id = {s["id"]: s for s in symbols}
    auxiliary_ids = set()
    for symbol_id in positives:
        symbol = by_id[symbol_id]
        modifications = [c for c in changes if symbol_id in c["symbol_ids"]]
        lines = (symbol["code"] + "\n").splitlines(True)
        for change in sorted(modifications, key=lambda c: c["old_start"], reverse=True):
            start = change["old_start"] - symbol["start_line"]
            end = start + change["old_count"]
            lines[start:end] = change["_added"]
        original, modified = semantic_tree(symbol["code"]), semantic_tree("".join(lines))
        if original is not None and original == modified:
            auxiliary_ids.add(symbol_id)
    positives -= auxiliary_ids
    for change in changes:
        previously_mapped = bool(change["symbol_ids"])
        change["symbol_ids"] = [i for i in change["symbol_ids"] if i not in auxiliary_ids]
        if previously_mapped and not change["symbol_ids"]:
            change["category"] = "auxiliary"
        change.pop("_added")
    meaningful = [c for c in changes if c["category"] != "auxiliary"]
    mapped = sum(bool(c["symbol_ids"]) for c in meaningful)
    return {
        "label_version": "old-coordinates-semantic-v2",
        "positive_ids": sorted(positives), "changes": changes,
        "mapped_groups": mapped, "total_groups": len(meaningful),
        "mapping_coverage": mapped / len(meaningful) if meaningful else 0.0,
        "scope": "fully_mapped" if meaningful and mapped == len(meaningful) else "mixed",
    }

