"""Abstract interface for local LLM inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Common interface implemented by all local LLM clients."""

    model_label: str

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts."""
