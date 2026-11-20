"""Factory for selecting the configured local LLM backend."""

from __future__ import annotations

from typing import Any

from src.llm.base import BaseLLM
from src.llm.hf_client import HuggingFaceLocalClient
from src.llm.ollama_client import OllamaClient


def create_llm_client(llm_config: dict[str, Any]) -> BaseLLM:
    """Instantiate an Ollama or local HuggingFace LLM client based on config."""
    provider = str(llm_config.get("provider", "auto")).lower()
    ollama_cfg = llm_config.get("ollama", {})
    hf_cfg = llm_config.get("huggingface", {})

    ollama_client = OllamaClient(
        base_url=str(ollama_cfg.get("base_url", "http://localhost:11434")),
        model=str(ollama_cfg.get("model", "qwen2.5:1.5b")),
        temperature=float(ollama_cfg.get("temperature", 0.1)),
        timeout_seconds=int(ollama_cfg.get("timeout_seconds", 120)),
    )

    if provider == "ollama":
        return ollama_client
    if provider == "auto" and ollama_client.is_available():
        return ollama_client
    if provider in {"auto", "huggingface"}:
        return HuggingFaceLocalClient(
            model_id=str(hf_cfg.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct")),
            max_new_tokens=int(hf_cfg.get("max_new_tokens", 220)),
            temperature=float(hf_cfg.get("temperature", 0.1)),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
