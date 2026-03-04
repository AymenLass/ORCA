"""InternVL3 model wrapper."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from agenticvlm.models.base import BaseVLM, GenerationConfig
from agenticvlm.utils.text_processing import clean_generated_text

logger = logging.getLogger(__name__)

VISION_EXPERT_GEN_CONFIG = GenerationConfig(max_new_tokens=500, temperature=1e-20)


class InternVL3Model(BaseVLM):
    """InternVL3 wrapper for visual expert agents.

    Args:
        model_path: Path to InternVL3 model weights.
        device_map: Device mapping strategy.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
    ) -> None:
        super().__init__(model_path, device_map, torch_dtype=torch.bfloat16)

    def load(self) -> None:
        """Load InternVL3 model and processor."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading InternVL3 model from %s", self.model_path)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._loaded = True
        logger.info("InternVL3 model loaded successfully")

    def generate(
        self,
        image_path: str,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text grounded on a document image.

        Args:
            image_path: Path to the document image.
            prompt: Text prompt for the model.
            gen_config: Generation configuration.

        Returns:
            Generated text response.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = gen_config or VISION_EXPERT_GEN_CONFIG

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        generate_ids = self.model.generate(**inputs, **config.to_dict())
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        result = decoded_output.strip() if decoded_output else ""
        return clean_generated_text(result)
