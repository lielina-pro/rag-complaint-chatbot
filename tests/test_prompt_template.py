"""Tests for src/prompt_template.py.

The module exposes:
  - SYSTEM_CONTEXT : str   (formerly PROMPT_TEMPLATE)
  - build_prompt(chunks, question) -> str
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt_template import SYSTEM_CONTEXT, build_prompt


# ── SYSTEM_CONTEXT ────────────────────────────────────────────────────────────

def test_system_context_is_nonempty_string():
    assert isinstance(SYSTEM_CONTEXT, str)
    assert len(SYSTEM_CONTEXT) > 20


def test_system_context_mentions_financial_role():
    assert any(word in SYSTEM_CONTEXT.lower() for word in ["financial", "analyst", "creditrust"])


# ── build_prompt ──────────────────────────────────────────────────────────────

def _make_chunk(text: str, product: str = "Credit Card", company: str = "Acme Bank") -> dict:
    return {"chunk_text": text, "product_category": product, "company": company, "score": 0.85}


def test_build_prompt_contains_question():
    question = "Why are customers unhappy with Credit Cards?"
    chunks = [_make_chunk("Customer reports unexpected fee.")]
    prompt = build_prompt(chunks, question)
    assert question in prompt


def test_build_prompt_contains_chunk_text():
    chunk_text = "Customer reports an unexpected fee on their statement."
    prompt = build_prompt([_make_chunk(chunk_text)], "Test question?")
    assert chunk_text in prompt


def test_build_prompt_contains_system_context():
    prompt = build_prompt([_make_chunk("some text")], "Any question?")
    assert SYSTEM_CONTEXT in prompt


def test_build_prompt_numbers_sources():
    chunks = [_make_chunk(f"complaint text {i}") for i in range(3)]
    prompt = build_prompt(chunks, "Any question?")
    assert "Source 1" in prompt
    assert "Source 2" in prompt
    assert "Source 3" in prompt


def test_build_prompt_includes_metadata():
    chunk = _make_chunk("complaint text", product="Personal Loan", company="Big Bank")
    prompt = build_prompt([chunk], "Any question?")
    assert "Personal Loan" in prompt
    assert "Big Bank" in prompt


def test_build_prompt_empty_chunks_still_returns_string():
    prompt = build_prompt([], "Any question?")
    assert isinstance(prompt, str)
    assert "Any question?" in prompt


def test_build_prompt_handles_missing_chunk_text_key():
    chunk = {"document": "fallback text field", "product_category": "Credit Card"}
    prompt = build_prompt([chunk], "Test?")
    assert "fallback text field" in prompt


def test_build_prompt_returns_string():
    prompt = build_prompt([_make_chunk("text")], "question?")
    assert isinstance(prompt, str)
