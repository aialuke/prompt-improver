import ast
import os
import sys
from pathlib import Path

LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}


class LoggingFStringTransformer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in LOG_METHODS:
            if node.args and isinstance(node.args[0], ast.JoinedStr):
                joined: ast.JoinedStr = node.args[0]
                fmt_parts: list[str] = []
                fmt_args: list[ast.AST] = []
                for value in joined.values:
                    if isinstance(value, ast.Str):
                        fmt_parts.append(value.s.replace("%", "%%"))
                    elif isinstance(value, ast.FormattedValue):
                        fmt_parts.append("%s")
                        expr = value.value
                        if value.format_spec is not None:
                            spec_str = _joinedstr_to_literal(value.format_spec)
                            if spec_str is not None:
                                expr = ast.Call(
                                    func=ast.Name(id="format", ctx=ast.Load()),
                                    args=[expr, ast.Constant(value=spec_str)],
                                    keywords=[],
                                )
                            else:
                                expr = ast.Call(
                                    func=ast.Name(id="str", ctx=ast.Load()),
                                    args=[expr],
                                    keywords=[],
                                )
                        fmt_args.append(expr)
                    else:
                        fmt_parts.append("%s")
                        fmt_args.append(
                            ast.Call(
                                func=ast.Name(id="str", ctx=ast.Load()),
                                args=[value],
                                keywords=[],
                            )
                        )
                fmt_literal = ast.Constant(value="".join(fmt_parts))
                new_args = [fmt_literal]
                new_args.extend(fmt_args)
                node.args = new_args + node.args[1:]
        return node


def _joinedstr_to_literal(node: ast.AST) -> str | None:
    """Best-effort convert a JoinedStr format_spec to a plain string.

    Returns None if conversion is not trivial.
    """
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Str):
                parts.append(v.s)
            else:
                return None
        return "".join(parts)
    return None


def process_file(path: Path) -> bool:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    transformer = LoggingFStringTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    if ast.unparse:
        new_src = ast.unparse(new_tree)
        if src.endswith("\n") and (not new_src.endswith("\n")):
            new_src += "\n"
    else:
        return False
    if new_src != src:
        path.write_text(new_src, encoding="utf-8")
        return True
    return False


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    changed = 0
    for dirpath, _, filenames in os.walk(root):
        if any(part.startswith(".") for part in Path(dirpath).parts):
            pass
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            if (
                ".venv" in path.parts
                or ".ruff_cache" in path.parts
                or ".mypy_cache" in path.parts
            ):
                continue
            if process_file(path):
                changed += 1
    print(f"Rewrote logging f-strings in {changed} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
