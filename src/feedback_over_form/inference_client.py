"""Async inference client for Ollama.

Sends chat completion requests to a local Ollama instance.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

# Patterns for thinking/reasoning tags across model families
_THINKING_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL),
]

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks from model output."""
    for pattern in _THINKING_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


class InferenceClient:
    """Async inference client for Ollama."""

    def __init__(
        self,
        backend: str = "ollama",
        ollama_base_url: str = OLLAMA_BASE_URL,
        timeout: float = 120.0,
    ):
        self.backend = backend
        self.ollama_base_url = ollama_base_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.ollama_base_url,
                timeout=httpx.Timeout(self.timeout, connect=30.0),
            )
        return self._client

    async def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the model. Returns cleaned text."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 4096,
                "temperature": temperature,
            },
        }
        # Disable thinking for qwen3/3.5 and gemma4
        if model.startswith(("qwen3:", "qwen3.5:", "gemma4:")):
            payload["think"] = False

        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "")
        return _strip_thinking(text)

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
