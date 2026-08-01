"""LLM generation via the Hugging Face Inference API.

Requires a free HF account and access token:
  1. Create one at https://huggingface.co/settings/tokens (read access is enough)
  2. Put HF_TOKEN=hf_xxx in a .env file at the project root (already gitignored)

Model note: use a standard instruct model, NOT a reasoning model (e.g. avoid
gpt-oss, DeepSeek-R1, QwQ). Reasoning models generate hidden chain-of-thought
tokens before writing visible output — with a small max_tokens budget they can
exhaust it on thinking and return None for content.
"""

from __future__ import annotations

import os
from typing import Generator, Optional

from huggingface_hub import InferenceClient

DEFAULT_MODEL: str = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_TEMPERATURE: float = 0.7


class Generator:
    """Wraps the HF Inference API for text generation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_name: str = model_name
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if not token:
            raise ValueError(
                "HF_TOKEN is required. Add it to your .env file or set it as "
                "an environment variable before running the app."
            )
        self.client: InferenceClient = InferenceClient(
            model=model_name, token=token
        )

    @classmethod
    def from_config(cls, config: "RAGConfig") -> "Generator":  # type: ignore[name-defined]
        """Construct a Generator from a RAGConfig instance."""
        return cls(
            model_name=config.generator_model,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            hf_token=config.hf_token or None,
        )

    def generate(self, prompt: str) -> str:
        """Generate a complete answer for *prompt* (blocking)."""
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream the answer token-by-token."""
        stream = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                yield token
