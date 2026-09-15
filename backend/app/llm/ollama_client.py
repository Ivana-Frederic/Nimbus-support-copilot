"""Async client for Ollama's REST API. No API key needed and no per-token
cost since it's just a local server.
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    async def chat(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

    async def chat_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={"model": self._model, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json

                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
