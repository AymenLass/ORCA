"""GLM-4.5V-9B model wrapper."""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

import torch
from PIL import Image

from agenticvlm.models.base import BaseVLM, GenerationConfig
from agenticvlm.prompts.thinker_prompts import THINKER_QUESTION_SUFFIX
from agenticvlm.utils.text_processing import extract_boxed_answer

logger = logging.getLogger(__name__)

THINKER_GEN_CONFIG = GenerationConfig(max_new_tokens=1200)


class GLM4VModel(BaseVLM):
    """GLM-4.5V-9B wrapper for the Thinker Agent.

    Args:
        model_path: Path to GLM-4.5V-9B model weights.
        device_map: Device mapping strategy.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
    ) -> None:
        super().__init__(model_path, device_map, torch_dtype=torch.bfloat16)

    def load(self) -> None:
        """Load GLM-4V model and processor."""
        from transformers import AutoProcessor, Glm4vForConditionalGeneration

        logger.info("Loading GLM-4.5V model (Thinker Agent) from %s", self.model_path)

        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self._loaded = True
        logger.info("GLM-4.5V model loaded successfully")

    def generate(
        self,
        image_path: str,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generate an answer with chain-of-thought reasoning.

        Args:
            image_path: Path to the document image.
            prompt: The question to answer.
            gen_config: Generation configuration.

        Returns:
            Full model output including thinking and boxed answer.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        config = gen_config or THINKER_GEN_CONFIG

        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": data_uri},
                    {"type": "text", "text": prompt + THINKER_QUESTION_SUFFIX},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **config.to_dict())
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        )

        return output_text

    def get_answer(
        self,
        image_path: str,
        question: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> tuple[str, str]:
        """Get both the full output and extracted answer.

        Args:
            image_path: Path to the document image.
            question: The question to answer.
            gen_config: Generation configuration.

        Returns:
            Tuple of (full_output, extracted_answer). If no boxed/tagged
            answer is found, the full output is returned as the answer.
        """
        full_output = self.generate(image_path, question, gen_config)

        extracted = extract_boxed_answer(full_output)
        if extracted is not None:
            return full_output, extracted

        return full_output, full_output.strip()
