"""Code ingestion: clone a repo and chunk it by function/class with tree-sitter.

Each function, class, and method becomes a :class:`Chunk` (source="code") carrying the
metadata needed for citations: repo-relative file path, qualified symbol name, kind,
language, and 1-based start/end lines. Classes are captured both as a whole and split
into their methods, giving retrieval both coarse and fine granularity.

Parsers come from ``tree-sitter-language-pack`` (prebuilt wheels — no compilation, works
on Windows). Note this binding exposes the node API as *methods* (``node.kind()``,
``node.child(i)``, ``node.start_byte()``, ``node.start_position().row``) and ``parse()``
takes ``str``; the small adapter helpers below hide that so the walk stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from paper_to_code.models import Chunk

# File extension -> tree-sitter language name.
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
}

# Node types (across languages) that represent a function/class/method definition.
_DEF_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
    "class_definition",
    "class_declaration",
    "function_item",
    "impl_item",
    "struct_item",
    "interface_declaration",
    "enum_declaration",
    "constructor_declaration",
}
_CLASS_HINTS = ("class", "struct", "impl", "interface", "enum")

_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".tox", "site-packages", ".idea", ".vscode",
    "vendor", "third_party", ".eggs",
}


@lru_cache(maxsize=None)
def _get_parser(language: str):
    from tree_sitter_language_pack import get_parser

    return get_parser(language)  # type: ignore[arg-type]


# --- tree-sitter node adapter (this binding exposes accessors as methods) ---

def _kind(node: Any) -> str:
    return node.kind()


def _children(node: Any) -> list[Any]:
    return [node.child(i) for i in range(node.child_count())]


def _text(src: bytes, node: Any) -> str:
    return src[node.start_byte() : node.end_byte()].decode("utf-8", errors="replace")


def _name(src: bytes, node: Any) -> str | None:
    name_node = node.child_by_field_name("name")
    return _text(src, name_node) if name_node is not None else None


def _row(point: Any) -> int:
    row = point.row
    return row() if callable(row) else row


def _start_line(node: Any) -> int:
    return _row(node.start_position()) + 1


def _end_line(node: Any) -> int:
    return _row(node.end_position()) + 1


def _kind_of(node_type: str) -> str:
    return "class" if any(hint in node_type for hint in _CLASS_HINTS) else "function"


def _walk_definitions(node: Any, src: bytes, parent: str | None = None) -> Iterator[tuple[Any, str, str]]:
    """Yield (node, qualified_symbol, kind) for every definition under ``node``."""
    for child in _children(node):
        ntype = _kind(child)
        if ntype in _DEF_TYPES:
            name = _name(src, child) or "<anonymous>"
            qualified = f"{parent}.{name}" if parent else name
            kind = _kind_of(ntype)
            yield child, qualified, kind
            if kind == "class":  # recurse to capture methods
                yield from _walk_definitions(child, src, qualified)
        else:  # descend through wrappers (decorators, blocks) to find defs
            yield from _walk_definitions(child, src, parent)


def chunk_file(path: Path, repo_root: Path, repo_id: str) -> list[Chunk]:
    """Parse one source file into definition-level chunks. Returns [] on parse failure."""
    language = LANGUAGE_BY_EXT.get(path.suffix.lower())
    if language is None:
        return []
    try:
        # The parser takes str; byte offsets index the UTF-8 encoding of that same str.
        source = path.read_text(encoding="utf-8", errors="replace")
        src_bytes = source.encode("utf-8")
        tree = _get_parser(language).parse(source)
        root = tree.root_node()
    except Exception:  # noqa: BLE001 - skip unreadable/unparseable files
        return []

    rel = path.relative_to(repo_root).as_posix()
    chunks: list[Chunk] = []
    for node, symbol, kind in _walk_definitions(root, src_bytes):
        text = _text(src_bytes, node)
        if not text.strip():
            continue
        start_line = _start_line(node)
        end_line = _end_line(node)
        metadata: dict[str, str | int | float | bool] = {
            "repo_id": repo_id,
            "source": "code",
            "file": rel,
            "symbol": symbol,
            "kind": kind,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
        }
        chunk_id = f"{repo_id}::{rel}::{symbol}::{start_line}"
        chunks.append(Chunk(id=chunk_id, text=text, source="code", metadata=metadata))
    return chunks


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _IGNORE_DIRS for part in rel_parts):
            continue
        if path.suffix.lower() in LANGUAGE_BY_EXT:
            yield path


def _looks_like_url(source: str) -> bool:
    return "://" in source or source.startswith("git@")


def clone_repo(url: str, repos_dir: str | Path, repo_id: str | None = None) -> tuple[str, Path]:
    """Shallow-clone ``url`` into ``repos_dir``; reuse an existing non-empty clone."""
    from git import Repo

    repo_id = repo_id or url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = Path(repos_dir) / repo_id
    if dest.exists() and any(dest.iterdir()):
        return repo_id, dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(url, str(dest), multi_options=["--depth=1"])
    return repo_id, dest


@dataclass
class CodeIngestResult:
    repo_id: str
    repo_path: Path
    chunks: list[Chunk]


def ingest_code(
    source: str | Path,
    repo_id: str | None = None,
    *,
    repos_dir: str | Path = "data/repos",
) -> CodeIngestResult:
    """Ingest a repo from a git URL or a local path into code chunks."""
    if isinstance(source, str) and _looks_like_url(source):
        repo_id, repo_path = clone_repo(source, repos_dir, repo_id)
    else:
        repo_path = Path(source)
        repo_id = repo_id or repo_path.name

    chunks: list[Chunk] = []
    for file_path in iter_source_files(repo_path):
        chunks.extend(chunk_file(file_path, repo_path, repo_id))
    return CodeIngestResult(repo_id=repo_id, repo_path=repo_path, chunks=chunks)
