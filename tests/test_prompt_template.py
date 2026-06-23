from src.prompt_template import PROMPT_TEMPLATE, build_context_block, build_prompt


def test_build_context_block_numbers_sources():
    chunks = [
        {"chunk_text": "First complaint text.", "product_category": "Credit Card"},
        {"chunk_text": "Second complaint text.", "product_category": "Savings Account"},
    ]
    block = build_context_block(chunks)
    assert "[Source 1 | Credit Card]: First complaint text." in block
    assert "[Source 2 | Savings Account]: Second complaint text." in block


def test_build_context_block_falls_back_to_text_column():
    chunks = [{"text": "Fallback text column.", "product_category": "Money Transfer"}]
    block = build_context_block(chunks)
    assert "Fallback text column." in block


def test_build_context_block_empty_list():
    assert build_context_block([]) == ""


def test_build_prompt_includes_question_and_context():
    chunks = [{"chunk_text": "Customers report delayed refunds.", "product_category": "Credit Card"}]
    prompt = build_prompt(chunks, "Why are people unhappy with Credit Cards?")
    assert "Why are people unhappy with Credit Cards?" in prompt
    assert "Customers report delayed refunds." in prompt
    assert "financial analyst assistant for CrediTrust" in prompt


def test_prompt_template_has_required_placeholders():
    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE
    assert "don't have" in PROMPT_TEMPLATE.lower() or "do not have" in PROMPT_TEMPLATE.lower()
