import pytest

from src.generator import Generator


class FakeDelta:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.delta = FakeDelta(content)


class FakeStreamChunk:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeClient:
    """Stands in for huggingface_hub.InferenceClient -- returns a canned
    sequence of streaming chunks instead of making a real network call."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_call_kwargs = None

    def chat_completion(self, **kwargs):
        self.last_call_kwargs = kwargs
        return iter(FakeStreamChunk(c) for c in self._chunks)


def _make_generator(monkeypatch, stream_chunks):
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-tests")
    gen = Generator()
    gen.client = FakeClient(stream_chunks)
    return gen


def test_generate_stream_yields_concatenated_deltas(monkeypatch):
    gen = _make_generator(monkeypatch, ["Hello", " ", "world", "."])
    tokens = list(gen.generate_stream("some prompt"))
    assert tokens == ["Hello", " ", "world", "."]
    assert "".join(tokens) == "Hello world."


def test_generate_stream_skips_empty_deltas(monkeypatch):
    gen = _make_generator(monkeypatch, ["Hi", None, "", " there"])
    tokens = list(gen.generate_stream("some prompt"))
    assert tokens == ["Hi", " there"]


def test_generate_stream_passes_stream_true(monkeypatch):
    gen = _make_generator(monkeypatch, ["ok"])
    list(gen.generate_stream("some prompt"))
    assert gen.client.last_call_kwargs["stream"] is True


def test_generator_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError):
        Generator()
