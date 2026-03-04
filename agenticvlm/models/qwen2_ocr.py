"""Qwen2-VL-OCR-2B-Instruct model wrapper."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from agenticvlm.models.base import BaseVLM, GenerationConfig
from agenticvlm.utils.text_processing import clean_generated_text

logger = logging.getLogger(__name__)

OCR_GEN_CONFIG = GenerationConfig(max_new_tokens=128, temperature=1e-20)


class Qwen2OCRModel(BaseVLM):
    """Qwen2-VL-OCR-2B-Instruct wrapper for handwritten text OCR.

    Args:
        model_path: Path to Qwen2-VL-OCR-2B model weights.
        device_map: Device mapping strategy.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
    ) -> None:
        super().__init__(model_path, device_map, torch_dtype=torch.bfloat16)

    def load(self) -> None:
        """Load Qwen2-VL-OCR model and processor."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading Qwen2-VL-OCR model from %s", self.model_path)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        )

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True,
        )
        self._loaded = True
        logger.info("Qwen2-VL-OCR model loaded successfully")

    def generate(
        self,
        image_path: str,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate OCR transcription from a document image.

        Args:
            image_path: Path to the document image.
            prompt: Text prompt for the model.
            gen_config: Generation configuration.
            system_message: Optional system message.

        Returns:
            Generated text transcription.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from qwen_vl_utils import process_vision_info

        config = gen_config or OCR_GEN_CONFIG

        messages: list[dict[str, Any]] = []
        if system_message:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_message}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **config.to_dict())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        result = output_text[0] if output_text else ""
        return clean_generated_text(result)
