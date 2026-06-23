from src.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_call = None

    def retrieve(self, question, k=5):
        self.last_call = (question, k)
        return self.chunks[:k]


class FakeGenerator:
    def __init__(self, canned_response="This is a generated answer."):
        self.canned_response = canned_response
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt
        return self.canned_response


def _sample_chunks():
    return [
        {"chunk_text": "Customers report delayed refunds.", "product_category": "Credit Card", "score": 0.91},
        {"chunk_text": "Fees were charged without notice.", "product_category": "Credit Card", "score": 0.87},
    ]


def test_answer_returns_expected_keys():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator()
    pipeline = RAGPipeline(retriever, generator, k=2)

    result = pipeline.answer("Why are people unhappy with Credit Cards?")
    assert set(result.keys()) == {"answer", "sources", "prompt"}
    assert result["answer"] == "This is a generated answer."
    assert len(result["sources"]) == 2


def test_answer_passes_question_and_k_to_retriever():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator()
    pipeline = RAGPipeline(retriever, generator, k=2)

    pipeline.answer("some question", k=1)
    assert retriever.last_call == ("some question", 1)


def test_answer_default_k_used_when_not_overridden():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator()
    pipeline = RAGPipeline(retriever, generator, k=2)

    pipeline.answer("some question")
    assert retriever.last_call == ("some question", 2)


def test_prompt_sent_to_generator_contains_question_and_context():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator()
    pipeline = RAGPipeline(retriever, generator, k=2)

    pipeline.answer("Why are people unhappy with Credit Cards?")
    assert "Why are people unhappy with Credit Cards?" in generator.last_prompt
    assert "Customers report delayed refunds." in generator.last_prompt
