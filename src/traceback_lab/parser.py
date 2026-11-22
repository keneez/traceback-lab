"""Tolerant Python syntax extraction. Original UTF-8 source is retained for display."""
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
import ast
import re
import warnings

from tree_sitter import Language, Parser
import tree_sitter_python

from .config import fingerprint


@dataclass
class Symbol:
    id: str
    path: str
    module: str
    qualname: str
    name: str
    class_name: str | None
    parent: str | None
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: str
    docstring: str
    code: str
    search_code: str
    source_hash: str
    parse_error: bool
    calls: list[dict] = field(default_factory=list)
    shadowed: list[str] = field(default_factory=list)

    def public(self):
        row = asdict(self)
        row.pop("calls")
        row.pop("shadowed")
        return row


def module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def parse_file(path: str, text: str, snapshot_id: str) -> tuple[list[Symbol], dict, list[dict]]:
    source = text.encode("utf-8")
    parser = Parser(Language(tree_sitter_python.language()))
    root = parser.parse(source).root_node
    symbols, imports, errors = [], {}, []
    by_id = {}
    module = module_name(path)

    def content(node):
        return source[node.start_byte:node.end_byte].decode("utf-8")

    def walk(node, scopes=(), class_name=None, owner=None):
        if node.type == "ERROR" or node.is_missing:
            errors.append({"start_line": node.start_point.row + 1,
                           "end_line": node.end_point.row + 1, "kind": node.type})
        if node.type == "class_definition":
            name = content(node.child_by_field_name("name"))
            for child in node.named_children:
                walk(child, (*scopes, name), name, owner)
            return
        if node.type == "function_definition":
            name = content(node.child_by_field_name("name"))
            qualname = ".".join((*scopes, name))
            outer = node.parent if node.parent.type == "decorated_definition" else node
            body = node.child_by_field_name("body")
            code = source[outer.start_byte:outer.end_byte].decode("utf-8")
            symbol_id = fingerprint([snapshot_id, path, qualname, outer.start_byte, code])[:24]
            docstring = ""
            if body and body.named_children:
                first = body.named_children[0]
                if first.type == "expression_statement" and first.named_children:
                    value = first.named_children[0]
                    if value.type in {"string", "concatenated_string"}:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", SyntaxWarning)
                                docstring = ast.literal_eval(content(value))
                        except (ValueError, SyntaxError):
                            docstring = content(value)
            signature = source[node.start_byte:body.start_byte].decode("utf-8").strip()
            symbol = Symbol(symbol_id, path, module, qualname, name, class_name, owner,
                            outer.start_point.row + 1, outer.end_point.row + 1,
                            outer.start_byte, outer.end_byte, signature, str(docstring),
                            code, code, fingerprint(code), node.has_error)
            params = node.child_by_field_name("parameters")
            if params:
                symbol.shadowed = re.findall(r"\b[a-zA-Z_]\w*\b", content(params))
            symbols.append(symbol)
            by_id[symbol.id] = symbol
            for child in node.named_children:
                walk(child, (*scopes, name), class_name, symbol_id)
            return
        if node.type == "call" and owner:
            function = node.child_by_field_name("function")
            symbol = by_id[owner]
            symbol.calls.append({"expression": content(function),
                                 "line": node.start_point.row + 1})
        if node.type in {"assignment", "augmented_assignment", "named_expression"} and owner:
            left = node.child_by_field_name("left") or node.child_by_field_name("name")
            if left:
                symbol = by_id[owner]
                symbol.shadowed.extend(re.findall(r"\b[a-zA-Z_]\w*\b", content(left)))
        if node.type in {"import_statement", "import_from_statement"} and owner is None:
            try:
                statement = ast.parse(content(node)).body[0]
                if isinstance(statement, ast.Import):
                    for alias in statement.names:
                        imports[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0])
                elif isinstance(statement, ast.ImportFrom):
                    prefix = statement.module or ""
                    if statement.level:
                        package = module.split(".")
                        if not path.endswith("__init__.py"):
                            package = package[:-1]
                        package = package[:len(package) - statement.level + 1]
                        prefix = ".".join([*package, prefix]).strip(".")
                    for alias in statement.names:
                        imports[alias.asname or alias.name] = ".".join(
                            p for p in [prefix, alias.name] if p)
            except (ValueError, SyntaxError):
                pass
        for child in node.named_children:
            walk(child, scopes, class_name, owner)

    walk(root)
    # Nested implementations should not duplicate their lexical weight in the parent.
    children = defaultdict(list)
    for symbol in symbols:
        if symbol.parent:
            children[symbol.parent].append(symbol)
    for parent_id, nested in children.items():
        parent = by_id[parent_id]
        body = parent.code.encode()
        for child in sorted(nested, key=lambda s: s.start_byte, reverse=True):
            start, end = child.start_byte - parent.start_byte, child.end_byte - parent.start_byte
            body = body[:start] + f"\n# nested symbol {child.name}\n".encode() + body[end:]
        parent.search_code = body.decode()
    if root.has_error and not errors:
        errors.append({"start_line": 1, "end_line": len(text.splitlines()), "kind": "has_error"})
    return symbols, imports, errors


def build_graph(symbols: list[Symbol], imports: dict[str, dict]) -> list[dict]:
    targets = defaultdict(list)
    by_id = {symbol.id: symbol for symbol in symbols}
    for symbol in symbols:
        targets[f"{symbol.module}.{symbol.qualname}"].append(symbol)
    edges = []
    for caller in symbols:
        aliases = imports.get(caller.path, {})
        for call in caller.calls:
            expression = call["expression"]
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", expression):
                continue
            parts = expression.split(".")
            root = parts[0]
            options, resolution = [], "unresolved"
            if root in caller.shadowed and root not in {"self", "cls"}:
                continue
            if len(parts) == 1:
                scope = caller
                while scope:
                    options.append(f"{caller.module}.{scope.qualname}.{root}")
                    scope = by_id.get(scope.parent)
                options.append(f"{caller.module}.{root}")
                if root in aliases:
                    options.append(aliases[root])
                resolution = "lexical"
            elif root in {"self", "cls"} and caller.class_name:
                path = caller.qualname.split(".")[:-1]
                class_index = len(path) - 1 - path[::-1].index(caller.class_name)
                class_scope = path[:class_index + 1]
                options = [".".join([caller.module, *class_scope, *parts[1:]])]
                resolution = "method"
            elif root in aliases:
                options = [".".join([aliases[root], *parts[1:]])]
                resolution = "import"
            for option in options:
                found = targets.get(option, [])
                if len(found) == 1:
                    target = found[0]
                    edge_resolution = ("import" if root in aliases and option == aliases[root]
                                       else resolution)
                    edges.append({"source": caller.id, "target": target.id,
                                  "line": call["line"], "path": caller.path,
                                  "expression": expression, "resolution": edge_resolution,
                                  "approximate": edge_resolution == "method"})
                    break
                if len(found) > 1:
                    break
    return sorted(edges, key=lambda e: (e["source"], e["line"], e["target"]))

