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

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from paper_to_code.errors import IngestionError
from paper_to_code.models import Chunk
from paper_to_code.progress import ProgressFn, report

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
    ".cu": "cuda",   # CUDA kernels (parsed by the cuda grammar)
    ".cuh": "cuda",
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


_GIT_URL_SCHEMES = {"http", "https", "git", "ssh"}


def _looks_like_url(source: str) -> bool:
    return "://" in source or source.startswith("git@")


def _derive_repo_id(url: str) -> str:
    """Best-effort repo name from a clone URL (``.../owner/repo.git`` -> ``repo``)."""
    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    return name or "repo"


def _validate_git_url(url: str) -> None:
    """Reject obviously-malformed URLs before shelling out to git."""
    if url.startswith("git@"):  # scp-style SSH: git@host:owner/repo.git
        if ":" not in url:
            raise IngestionError(f"Malformed SSH git URL: {url!r}.")
        return
    parsed = urlparse(url)
    if parsed.scheme not in _GIT_URL_SCHEMES or not parsed.netloc:
        raise IngestionError(
            f"{url!r} is not a valid repository. Provide a public git URL "
            "(e.g. https://github.com/owner/repo) or a path to a local folder."
        )


def _friendly_clone_error(url: str, exc: Exception) -> str:
    """Map a low-level git failure to an actionable, user-facing message."""
    text = str(exc).lower()
    if any(s in text for s in ("could not resolve host", "unable to access", "timed out", "timeout")):
        return f"Could not reach {url}. Check your internet connection and the URL."
    if any(s in text for s in ("repository not found", "not found", "404")):
        return f"Repository not found: {url}. Check the URL; only public repositories are supported."
    if any(s in text for s in ("authentication", "permission denied", "403", "could not read username", "access denied")):
        return (
            f"Access denied for {url}. This looks like a private repository — only public "
            "repositories can be cloned here."
        )
    if "git" in text and any(s in text for s in ("not found", "executable", "no such file")):
        return "Git is not installed or not on PATH. Install git, then try again."
    first_line = (str(exc).strip().splitlines() or [""])[0]
    return f"Failed to clone {url}: {first_line[:200]}"


def clone_repo(url: str, repos_dir: str | Path, repo_id: str | None = None) -> tuple[str, Path]:
    """Shallow-clone ``url`` into ``repos_dir``, reusing a previous *healthy* clone.

    A leftover directory from a failed clone (no ``.git``) is discarded and re-cloned, and
    a clone that fails part-way is cleaned up so it can't be mistaken for a good one.
    Predictable failures are raised as :class:`IngestionError` with an actionable message.
    """
    from git import Repo

    _validate_git_url(url)
    repo_id = repo_id or _derive_repo_id(url)
    dest = Path(repos_dir) / repo_id

    if dest.exists() and any(dest.iterdir()):
        if (dest / ".git").exists():
            return repo_id, dest  # reuse a healthy clone
        shutil.rmtree(dest, ignore_errors=True)  # leftover from an earlier failed clone

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        Repo.clone_from(url, str(dest), multi_options=["--depth=1"])
    except Exception as exc:  # noqa: BLE001 - normalize every git failure for the front-ends
        shutil.rmtree(dest, ignore_errors=True)  # never leave a half-clone behind
        raise IngestionError(_friendly_clone_error(url, exc)) from exc
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
    on_progress: ProgressFn | None = None,
) -> CodeIngestResult:
    """Ingest a repo from a git URL or a local path into code chunks.

    Raises :class:`IngestionError` for bad URLs, clone failures, or a missing local path.
    """
    if isinstance(source, str) and _looks_like_url(source):
        report(on_progress, f"Cloning {source} …", None)
        repo_id, repo_path = clone_repo(source, repos_dir, repo_id)
    else:
        repo_path = Path(source)
        if not repo_path.exists():
            raise IngestionError(
                f"Local path not found: {source}. Provide an existing folder or a public "
                "git URL (e.g. https://github.com/owner/repo)."
            )
        if not repo_path.is_dir():
            raise IngestionError(f"Not a folder: {source}. Point at a repository directory.")
        repo_id = repo_id or repo_path.name

    report(on_progress, "Scanning source files …", None)
    files = list(iter_source_files(repo_path))
    chunks: list[Chunk] = []
    for file_path in files:
        chunks.extend(chunk_file(file_path, repo_path, repo_id))
    report(on_progress, f"Parsed {len(chunks)} definitions from {len(files)} files.", None)
    return CodeIngestResult(repo_id=repo_id, repo_path=repo_path, chunks=chunks)
