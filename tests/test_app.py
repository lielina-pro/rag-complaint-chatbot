"""Tests for app.py (Task 4).

Importing app.py itself doubles as a smoke test: the module builds the full
Gradio UI at import time, and gracefully captures pipeline-init failures
into INIT_ERROR instead of crashing. Whether that actually happens on any
given machine depends on real local state (has the vector store been built?
is HF_TOKEN set?) -- which is exactly what makes it unsuitable to assert on
directly. The two tests below instead force each scenario deterministically
via monkeypatch, so they pass identically whether run in CI (nothing built
yet) or on a fully set-up dev machine (everything built and working).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def test_app_module_imports_without_crashing():
    # If we get here at all, the import (and Gradio Blocks construction)
    # succeeded -- the real assertion is just "no exception was raised".
    assert app.demo is not None


def test_build_pipeline_raises_when_vector_store_missing(monkeypatch, tmp_path):
    """Force the 'vector store not built yet' scenario directly, regardless
    of whether the real vector store happens to exist on this machine."""
    monkeypatch.setattr(app, "INDEX_PATH", str(tmp_path / "missing.faiss"))
    monkeypatch.setattr(app, "METADATA_PATH", str(tmp_path / "missing.parquet"))

    with pytest.raises(FileNotFoundError, match="build_faiss_index"):
        app.build_pipeline()


def test_format_sources_empty_list():
    assert app.format_sources([]) == "_No sources retrieved._"


def test_format_sources_includes_product_and_text():
    chunks = [{
        "product_category": "Credit Card",
        "company": "Acme Bank",
        "score": 0.873,
        "chunk_text": "Customer reports an unauthorized charge on their statement.",
    }]
    result = app.format_sources(chunks)
    assert "Credit Card" in result
    assert "Acme Bank" in result
    assert "0.87" in result
    assert "unauthorized charge" in result


def test_format_sources_truncates_long_text():
    long_text = "x" * 500
    chunks = [{"product_category": "Credit Card", "chunk_text": long_text}]
    result = app.format_sources(chunks)
    assert "..." in result
    assert len(result) < len(long_text) + 100


def test_format_sources_numbers_multiple_chunks():
    chunks = [
        {"product_category": "Credit Card", "chunk_text": "first"},
        {"product_category": "Savings Account", "chunk_text": "second"},
    ]
    result = app.format_sources(chunks)
    assert "Source 1" in result
    assert "Source 2" in result


def test_clear_all_returns_three_empty_strings():
    assert app.clear_all() == ("", "", "")


def test_ask_question_with_empty_string_prompts_for_input():
    outputs = list(app.ask_question("", 5))
    assert len(outputs) == 1
    assert "enter a question" in outputs[0][0].lower()


def test_ask_question_shows_init_error_if_pipeline_not_ready(monkeypatch):
    """Force INIT_ERROR directly so this doesn't depend on whether the real
    pipeline actually failed to initialize on this machine."""
    monkeypatch.setattr(
        app, "INIT_ERROR",
        "Vector store not found at 'vector_store/full_dataset.faiss'. Run "
        "`python src/build_faiss_index.py` first.",
    )
    outputs = list(app.ask_question("Why are people unhappy with Credit Cards?", 5))
    assert len(outputs) == 1
    assert "set up yet" in outputs[0][0].lower()
