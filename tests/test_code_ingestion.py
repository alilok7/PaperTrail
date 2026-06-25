"""Stage tests for code ingestion (real tree-sitter parse, no network/clone)."""

from __future__ import annotations

import textwrap

from paper_to_code.ingestion.code import chunk_file, iter_source_files


def _write(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(textwrap.dedent(body), encoding="utf-8")
    return f


def test_chunk_python_classes_and_functions(tmp_path):
    src = """
        import torch

        class CausalSelfAttention:
            def __init__(self):
                self.n_head = 4

            def forward(self, x):
                return x

        def make_model():
            return CausalSelfAttention()
    """
    f = _write(tmp_path, "model.py", src)
    chunks = chunk_file(f, tmp_path, "nano")
    by_symbol = {c.metadata["symbol"]: c for c in chunks}

    assert "CausalSelfAttention" in by_symbol
    assert "CausalSelfAttention.__init__" in by_symbol
    assert "CausalSelfAttention.forward" in by_symbol
    assert "make_model" in by_symbol

    cls = by_symbol["CausalSelfAttention"]
    assert cls.metadata["kind"] == "class"
    assert cls.metadata["language"] == "python"
    assert cls.metadata["file"] == "model.py"
    assert cls.metadata["source"] == "code"
    assert cls.metadata["start_line"] >= 1
    assert cls.metadata["end_line"] >= cls.metadata["start_line"]
    assert "def forward" in cls.text  # whole class body is captured

    fn = by_symbol["make_model"]
    assert fn.metadata["kind"] == "function"
    assert all(isinstance(v, (str, int, float, bool)) for v in fn.metadata.values())


def test_unsupported_extension_yields_no_chunks(tmp_path):
    f = _write(tmp_path, "notes.txt", "just some text, not code")
    assert chunk_file(f, tmp_path, "repo") == []


def test_iter_source_files_skips_ignored_dirs(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    (tmp_path / "node_modules").mkdir()
    _write(tmp_path, "node_modules/b.py", "def b():\n    return 2\n")
    (tmp_path / "__pycache__").mkdir()
    _write(tmp_path, "__pycache__/c.py", "def c():\n    return 3\n")

    found = {p.name for p in iter_source_files(tmp_path)}
    assert found == {"a.py"}
