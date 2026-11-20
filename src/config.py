"""Configuration and prompt template loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PromptTemplates:
    """System and user prompt templates for RAG generation."""

    rag_system_prompt: str
    rag_user_prompt: str


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file relative to project root or an absolute path."""
    resolved = path if path.is_absolute() else ROOT_DIR / path
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load main application configuration."""
    return load_yaml(Path(config_path))


def load_prompts(prompts_path: str | Path = "configs/prompts.yaml") -> PromptTemplates:
    """Load prompt templates into a validated PromptTemplates container."""
    data = load_yaml(Path(prompts_path))
    return PromptTemplates(
        rag_system_prompt=str(data["rag_system_prompt"]).strip(),
        rag_user_prompt=str(data["rag_user_prompt"]).strip(),
    )


def resolve_path(relative_or_abs: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(relative_or_abs)
    return path if path.is_absolute() else ROOT_DIR / path
