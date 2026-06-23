"""LLM generation for Task 3, via the Hugging Face Inference API.

Requires a free Hugging Face account and access token:
  1. Create a token at https://huggingface.co/settings/tokens (read access is enough)
  2. Set it as an environment variable before running anything that uses this module:
       Windows (PowerShell):  $env:HF_TOKEN = "hf_xxxxxxxx"
       Windows (cmd):         set HF_TOKEN=hf_xxxxxxxx
       Mac/Linux:             export HF_TOKEN=hf_xxxxxxxx

HF Inference is now routed through a multi-provider system (router.huggingface.co)
rather than HF's own single serverless backend, so which models are servable as
chat-completion models shifts over time depending on which third-party providers
(Cerebras, Together, Novita, DeepInfra, SambaNova, Groq, ...) currently host them.
deepseek-ai/DeepSeek-V3-0324 is used below since it's a standard instruct chat
model (no hidden reasoning/chain-of-thought tokens to budget for, unlike
"reasoning" models such as gpt-oss or DeepSeek-R1/QwQ, which can silently return
empty content if max_tokens is too low to cover their internal thinking step)
and is confirmed as one of HF's current canonical Inference Providers examples.

If this default ever returns a "model not supported"/"not a chat model" error,
check https://huggingface.co/docs/inference-providers for a currently-supported
model — any chat_completion-compatible instruct model works with zero other
code changes, just swap the string. Prefer standard instruct models over
"reasoning" models here unless you also raise max_new_tokens substantially
(several hundred to thousands) to leave room for their hidden thinking step.
"""

from __future__ import annotations

import os
from typing import Iterator

from huggingface_hub import InferenceClient

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3-0324"


class Generator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        hf_token: str | None = None,
        max_new_tokens: int = 300,
        temperature: float = 0.2,
    ):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "No Hugging Face token found. Set the HF_TOKEN environment variable "
                "(see the module docstring in src/generator.py) or pass hf_token= explicitly."
            )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.client = InferenceClient(model=model_name, token=token)

    def generate(self, prompt: str) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yield the answer token-by-token (or chunk-by-chunk, depending on
        the provider) as it's generated, for the Task 4 UI's streaming
        display. Falls back gracefully -- if a chunk has no text delta
        (e.g. a final empty chunk some providers send), it's simply skipped."""
        stream = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
