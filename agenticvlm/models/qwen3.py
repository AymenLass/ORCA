"""Qwen3 model wrapper for text-only reasoning."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from agenticvlm.models.base import BaseVLM, GenerationConfig
from agenticvlm.utils.text_processing import clean_generated_text

logger = logging.getLogger(__name__)

# Debate / language expert default config (long generation for reasoning)
DEBATE_GEN_CONFIG = GenerationConfig(
    max_new_tokens=32768,
    temperature=0.6,
    do_sample=True,
    top_p=0.95,
    min_p=0.0,
)

CHECKER_GEN_CONFIG = GenerationConfig(
    max_new_tokens=200,
    temperature=0.6,
    do_sample=True,
    top_p=0.95,
)


class Qwen3Model(BaseVLM):
    """Qwen3 text-only model wrapper for debate / reasoning agents.

    Args:
        model_path: Path to model weights.
        device_map: Device mapping strategy.
        enable_thinking: Whether to enable thinking mode in chat template.
            When ``False`` (default for agents), thinking is suppressed.
    """

    THINK_TOKEN_ID: int = 151668  # </think> token in Qwen3 tokenizer

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(model_path, device_map, torch_dtype=torch.bfloat16)
        self.enable_thinking = enable_thinking

    def load(self) -> None:
        """Load Qwen3 model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Qwen3 model from %s", self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._loaded = True
        logger.info("Qwen3 model loaded (thinking=%s)", self.enable_thinking)

    def generate(
        self,
        image_path: str | None,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text response (ignores image_path — text-only model).

        Args:
            image_path: Ignored for text-only models.
            prompt: The text prompt.
            gen_config: Generation configuration.
            system_message: Optional system message.

        Returns:
            Generated text, with thinking portion stripped if present.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = gen_config or DEBATE_GEN_CONFIG

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, **config.to_dict())

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        content = self._extract_content(output_ids)
        return clean_generated_text(content)

    def _extract_content(self, output_ids: list[int]) -> str:
        """Parse output token ids, separating thinking from content."""
        try:
            think_idx = output_ids.index(self.THINK_TOKEN_ID)
            content_ids = output_ids[think_idx + 1 :]
        except ValueError:
            content_ids = output_ids

        return self.tokenizer.decode(
            content_ids,
            skip_special_tokens=True,
        ).strip()

    def unload(self) -> None:
        """Release model, tokenizer, and GPU memory."""
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        super().unload()

    def generate_text(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """Convenience alias — generates text without image."""
        return self.generate(
            image_path=None,
            prompt=prompt,
            gen_config=gen_config,
            system_message=system_message,
        )
