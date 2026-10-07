"""Bounded, deterministic, read-only code search and navigation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from ..workspace import Workspace, WorkspaceError
from .errors import CodingLimitError, CodingWorkspaceError
from .limits import CodingLimits
from .models import CodeSymbol, CodingProject


@dataclass(frozen=True)
class CodeSearchHit:
    path: str
    line: int
    column: int
    kind: str
    symbol: str | None
    context: str


@dataclass(frozen=True)
class CodeSearchResult:
    query: str
    mode: str
    hits: tuple[CodeSearchHit, ...]
    truncated: bool = False


_JS_SYMBOL_PATTERNS = (
    (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
    (
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function|(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>)"
        ),
        "function",
    ),
)


class CodeSearchRuntime:
    """Search validated project source snapshots without executing them."""

    def __init__(self, workspace: Workspace, limits: CodingLimits) -> None:
        self.workspace = workspace
        self.limits = limits

    def text_search(
        self, project: CodingProject, query: str, *, max_results: int = 50
    ) -> CodeSearchResult:
        return self._search(project, query, mode="text", max_results=max_results)

    def symbol_search(
        self, project: CodingProject, query: str, *, max_results: int = 50
    ) -> CodeSearchResult:
        result = self._search(project, query, mode="symbol", max_results=max_results)
        return result

    def file_search(
        self, project: CodingProject, query: str, *, max_results: int = 50
    ) -> CodeSearchResult:
        root = self._root(project)
        q = self._bounded_query(query)
        hits: list[CodeSearchHit] = []
        truncated = False
        for path in self._files(root):
            rel = path.relative_to(root).as_posix()
            if q.casefold() in rel.casefold():
                hits.append(CodeSearchHit(rel, 1, 1, "file", None, rel[:240]))
                if len(hits) >= max_results:
                    truncated = True
                    break
        return CodeSearchResult(q, "file", tuple(hits), truncated)

    def find_definition(
        self, project: CodingProject, name: str, *, max_results: int = 20
    ) -> CodeSearchResult:
        q = self._bounded_query(name)
        root = self._root(project)
        hits: list[CodeSearchHit] = []
        for path in self._files(root):
            rel = path.relative_to(root).as_posix()
            text = self._read(path)
            if text is None:
                continue
            for symbol in self._symbols(text, rel):
                if (
                    symbol.name.casefold() == q.casefold()
                    or (symbol.qualified_name or "").casefold() == q.casefold()
                ):
                    hits.append(
                        CodeSearchHit(
                            rel,
                            symbol.region.start_line,
                            symbol.region.start_column or 1,
                            symbol.kind.value,
                            symbol.name,
                            self._context(text, symbol.region.start_line),
                        )
                    )
                    if len(hits) >= max_results:
                        return CodeSearchResult(q, "definition", tuple(hits), True)
        return CodeSearchResult(q, "definition", tuple(hits), False)

    def _search(
        self, project: CodingProject, query: str, *, mode: str, max_results: int
    ) -> CodeSearchResult:
        q = self._bounded_query(query)
        root = self._root(project)
        hits: list[CodeSearchHit] = []
        truncated = False
        for path in self._files(root):
            rel = path.relative_to(root).as_posix()
            text = self._read(path)
            if text is None:
                continue
            if mode == "symbol":
                rows = (
                    (
                        s.region.start_line,
                        s.region.start_column or 1,
                        s.kind.value,
                        s.name,
                        self._context(text, s.region.start_line),
                    )
                    for s in self._symbols(text, rel)
                    if q.casefold() in s.name.casefold()
                    or q.casefold() in (s.qualified_name or "").casefold()
                )
            else:
                rows = self._text_rows(text, q)
            for line, column, kind, symbol, context in rows:
                hits.append(CodeSearchHit(rel, line, column, kind, symbol, context))
                if len(hits) >= max_results:
                    return CodeSearchResult(q, mode, tuple(hits), True)
        return CodeSearchResult(q, mode, tuple(hits), truncated)

    def _root(self, project: CodingProject) -> Path:
        try:
            return self.workspace.resolve(project.root_path)
        except WorkspaceError:
            raise CodingWorkspaceError() from None

    def _bounded_query(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise CodingLimitError()
        return query.strip()

    def _files(self, root: Path):
        count = 0
        for path in sorted(root.rglob("*")):
            if count >= self.limits.max_project_files:
                break
            if not path.is_file() or any(
                part
                in {
                    ".git",
                    ".venv",
                    "venv",
                    "node_modules",
                    "__pycache__",
                    "dist",
                    "build",
                    "secrets",
                    "credentials",
                }
                for part in path.relative_to(root).parts
            ):
                continue
            if path.suffix.lower() not in {
                ".py",
                ".pyi",
                ".js",
                ".jsx",
                ".mjs",
                ".cjs",
                ".ts",
                ".tsx",
                ".mts",
                ".cts",
                ".md",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
            }:
                continue
            try:
                resolved = self.workspace.resolve(path.relative_to(self.workspace.root).as_posix())
            except WorkspaceError:
                continue
            if resolved != path.resolve() or path.stat().st_size > self.limits.max_file_size_bytes:
                continue
            count += 1
            yield path

    def _read(self, path: Path) -> str | None:
        try:
            data = path.read_bytes()
            if b"\x00" in data or len(data) > self.limits.max_file_size_bytes:
                return None
            text = data.decode("utf-8")
            if len(text) > self.limits.max_source_chars:
                return None
            return text
        except (OSError, UnicodeDecodeError):
            return None

    @staticmethod
    def _text_rows(text: str, query: str):
        for lineno, line in enumerate(text.splitlines(), 1):
            start = line.casefold().find(query.casefold())
            if start >= 0:
                yield lineno, start + 1, "text", None, line[:500]

    @staticmethod
    def _context(text: str, line: int) -> str:
        rows = text.splitlines()
        return rows[line - 1][:500] if 0 < line <= len(rows) else ""

    @staticmethod
    def _symbols(text: str, path: str) -> tuple[CodeSymbol, ...]:
        from .models import CodeRegion, CodeSymbolKind

        suffix = Path(path).suffix.lower()
        out: list[CodeSymbol] = []
        if suffix in {".py", ".pyi"}:
            try:
                tree = ast.parse(text)
            except SyntaxError:
                return ()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = (
                        CodeSymbolKind.CLASS
                        if isinstance(node, ast.ClassDef)
                        else CodeSymbolKind.FUNCTION
                    )
                    end = getattr(node, "end_lineno", node.lineno)
                    out.append(
                        CodeSymbol(
                            name=node.name,
                            qualified_name=node.name,
                            kind=kind,
                            region=CodeRegion(path=path, start_line=node.lineno, end_line=end),
                            signature=text.splitlines()[node.lineno - 1][:500]
                            if text.splitlines()
                            else None,
                        )
                    )
        elif suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}:
            from .models import CodeSymbolKind

            for lineno, line in enumerate(text.splitlines(), 1):
                for pattern, kind_name in _JS_SYMBOL_PATTERNS:
                    kind = CodeSymbolKind(kind_name)
                    m = pattern.match(line)
                    if m:
                        out.append(
                            CodeSymbol(
                                name=m.group(1),
                                qualified_name=m.group(1),
                                kind=kind,
                                region=CodeRegion(path=path, start_line=lineno, end_line=lineno),
                                signature=line[:500],
                            )
                        )
                        break
        return tuple(out)
