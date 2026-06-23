from src.rag_pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_call = None

    def retrieve(self, question, k=5):
        self.last_call = (question, k)
        return self.chunks[:k]


class FakeGenerator:
    def __init__(self, canned_response="This is a generated answer.", stream_tokens=None):
        self.canned_response = canned_response
        # stream_tokens=None means "derive streamed tokens from canned_response";
        # an explicit list (including []) is used as-is, e.g. to simulate a
        # provider that streams nothing for a given prompt.
        self.stream_tokens = stream_tokens
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt
        return self.canned_response

    def generate_stream(self, prompt):
        self.last_prompt = prompt
        tokens = (
            self.stream_tokens
            if self.stream_tokens is not None
            else [word + " " for word in self.canned_response.split(" ")]
        )
        for token in tokens:
            yield token


class FakeEmptyStreamGenerator(FakeGenerator):
    """Simulates a provider that returns zero delta chunks for a short
    answer -- exercises the non-streaming fallback in answer_stream."""

    def generate_stream(self, prompt):
        self.last_prompt = prompt
        return iter([])


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


def test_answer_stream_yields_growing_partial_answer():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator(stream_tokens=["The ", "answer ", "grows."])
    pipeline = RAGPipeline(retriever, generator, k=2)

    partials = [partial for partial, _ in pipeline.answer_stream("some question")]
    assert partials == ["The ", "The answer ", "The answer grows."]


def test_answer_stream_sources_constant_across_yields():
    chunks = _sample_chunks()
    retriever = FakeRetriever(chunks)
    generator = FakeGenerator(stream_tokens=["a", "b"])
    pipeline = RAGPipeline(retriever, generator, k=2)

    results = list(pipeline.answer_stream("some question"))
    assert all(sources == chunks[:2] for _, sources in results)


def test_answer_stream_retrieves_once_with_correct_k():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator(stream_tokens=["x"])
    pipeline = RAGPipeline(retriever, generator, k=2)

    list(pipeline.answer_stream("some question", k=1))
    assert retriever.last_call == ("some question", 1)


def test_answer_stream_falls_back_to_generate_on_empty_stream():
    """Some providers occasionally yield zero delta chunks for very short
    answers -- the pipeline should fall back to a plain generate() call
    rather than silently returning nothing."""
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator(canned_response="fallback answer", stream_tokens=[])
    pipeline = RAGPipeline(retriever, generator, k=2)

    results = list(pipeline.answer_stream("some question"))
    assert len(results) == 1
    assert results[0][0] == "fallback answer"


def test_answer_stream_yields_growing_answer_and_constant_sources():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeGenerator("This is a generated answer.")
    pipeline = RAGPipeline(retriever, generator, k=2)

    outputs = list(pipeline.answer_stream("some question"))
    assert len(outputs) == 5  # one yield per word in "This is a generated answer."

    # answer text should be monotonically growing and end up matching the full response
    answers = [a for a, _ in outputs]
    assert answers[-1].strip() == "This is a generated answer."
    assert all(len(answers[i]) <= len(answers[i + 1]) for i in range(len(answers) - 1))

    # sources should be present (and identical) at every step
    for _, sources in outputs:
        assert sources == retriever.chunks[:2]


def test_answer_stream_falls_back_when_stream_yields_nothing():
    retriever = FakeRetriever(_sample_chunks())
    generator = FakeEmptyStreamGenerator("Fallback answer.")
    pipeline = RAGPipeline(retriever, generator, k=2)

    outputs = list(pipeline.answer_stream("some question"))
    assert len(outputs) == 1
    assert outputs[0][0] == "Fallback answer."
