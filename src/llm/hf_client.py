"""Pure-Python local Hugging Face Transformers instruction model client."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.base import BaseLLM


class HuggingFaceLocalClient(BaseLLM):
    """Runs an open-source instruction-tuned causal LM locally via Transformers."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 220,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model_label = f"huggingface/{model_id}"
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Load tokenizer and model weights once on CPU (or CUDA if available)."""
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
        ).to(device)
        self._model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a chat completion using the model's chat template."""
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        do_sample = self.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
