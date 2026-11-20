"""Local Ollama HTTP API client."""

from __future__ import annotations

import requests

from src.llm.base import BaseLLM


class OllamaClient(BaseLLM):
    """Calls a locally running Ollama server via its /api/chat endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.1,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.model_label = f"ollama/{model}"

    def is_available(self) -> bool:
        """Check whether Ollama server is reachable and has the configured model."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(m == self.model or m.startswith(f"{self.model}:") for m in models)
        except requests.RequestException:
            return False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an assistant response using Ollama /api/chat."""
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
