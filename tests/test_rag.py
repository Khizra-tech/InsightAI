import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from document_processor import chunk_text  # noqa: E402


def test_chunk_text_respects_size():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_covers_all_content():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert chunks[0].startswith("w0")
    assert chunks[-1].endswith("w499")


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)
