"""Router model wrapper — LoRA-finetuned Qwen2.5-VL with Turbo DFS decoding.

The router generates short label tokens and Turbo DFS explores multiple
likely continuations.  The union of labels decoded from all candidate
sequences whose cumulative probability exceeds ``min_prob`` yields the
binary activation vector ``v ∈ {0,1}^9``.

Paper parameters:
    * ``min_prob  = 0.02``
    * ``max_new_tokens = 3``
    * ``temperature = 0.9``
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import torch

from agenticvlm.decoding.turbo_dfs import inference_turbo_dfs
from agenticvlm.models.base import BaseVLM, GenerationConfig
from agenticvlm.prompts.router_prompts import ROUTER_CLASSIFICATION_PROMPT, ROUTER_LABELS
from agenticvlm.utils.image_utils import load_image_grayscale
from agenticvlm.utils.text_processing import clean_generated_text

logger = logging.getLogger(__name__)

# Router generation config — kept for the non-Turbo fallback path
ROUTER_GEN_CONFIG = GenerationConfig(
    max_new_tokens=128,
    temperature=0.1,
    do_sample=True,
    min_p=0.1,
)

# Turbo DFS defaults (from the paper, Section 3.2)
TURBO_DFS_MIN_PROB: float = 0.02
TURBO_DFS_MAX_NEW_TOKENS: int = 3
TURBO_DFS_TEMPERATURE: float = 0.9


class RouterModel(BaseVLM):
    """Qwen2.5-VL router with LoRA adapter and Turbo DFS decoding.

    The model receives a grayscale document image together with the
    classification prompt and generates candidate label sequences.
    The union strategy combines labels from all candidates whose
    cumulative probability exceeds ``min_prob``.

    Args:
        model_path: Path to the base Qwen2.5-VL model.
        adapter_path: Optional path to the LoRA adapter directory.
        device_map: Device mapping strategy.
        use_unsloth: Whether to load via Unsloth's ``FastVisionModel``.
        use_turbo_dfs: Enable Turbo DFS decoding (default ``True``).
        turbo_min_prob: Probability threshold for Turbo DFS pruning.
        turbo_max_new_tokens: Maximum generated tokens per candidate.
        turbo_temperature: Temperature for Turbo DFS scoring.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        use_unsloth: bool = True,
        use_turbo_dfs: bool = True,
        turbo_min_prob: float = TURBO_DFS_MIN_PROB,
        turbo_max_new_tokens: int = TURBO_DFS_MAX_NEW_TOKENS,
        turbo_temperature: float = TURBO_DFS_TEMPERATURE,
    ) -> None:
        super().__init__(model_path, device_map, torch_dtype=torch.float16)
        self.adapter_path = adapter_path
        self.use_unsloth = use_unsloth
        self.use_turbo_dfs = use_turbo_dfs
        self.turbo_min_prob = turbo_min_prob
        self.turbo_max_new_tokens = turbo_max_new_tokens
        self.turbo_temperature = turbo_temperature

    def load(self) -> None:
        """Load router model with optional LoRA adapter."""
        if self.use_unsloth:
            self._load_unsloth()
        else:
            self._load_hf()
        self._loaded = True

    def _load_unsloth(self) -> None:
        """Load with Unsloth for 4-bit QLoRA inference."""
        from unsloth import FastVisionModel

        logger.info(
            "Loading router via Unsloth from %s (adapter=%s)",
            self.model_path,
            self.adapter_path,
        )

        model, tokenizer = FastVisionModel.from_pretrained(
            self.adapter_path or self.model_path,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(model)

        self.model = model
        self.tokenizer = tokenizer

        from transformers import AutoProcessor

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True,
        )
        logger.info("Router model loaded via Unsloth (4-bit)")

    def _load_hf(self) -> None:
        """Load merged model via HuggingFace Transformers."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading merged router from %s", self.model_path)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
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
        self.tokenizer = self.processor
        logger.info("Router model loaded via HuggingFace")

    def _prepare_inputs(self, image_path: str, question: str) -> dict:
        """Prepare model inputs from an image path and question."""
        gray_image = load_image_grayscale(image_path)
        prompt = ROUTER_CLASSIFICATION_PROMPT.format(question=question)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[gray_image],
            padding=True,
            return_tensors="pt",
        )
        device = self.model.device if hasattr(self.model, "device") else "cuda"
        return {k: v.to(device) for k, v in inputs.items()}

    def unload(self) -> None:
        """Release model, tokenizer, and GPU memory."""
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        super().unload()

    def classify(
        self,
        image_path: str,
        question: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Classify a (question, image) pair into document-type labels.

        Uses **Turbo DFS** to explore multiple candidate label sequences
        and returns the **union** of all labels whose cumulative
        probability exceeds the threshold (multi-label activation).

        When ``use_turbo_dfs`` is ``False``, falls back to standard
        greedy decoding with single-label output.

        Args:
            image_path: Path to the document image.
            question: The question to classify.
            gen_config: Optional generation config override (greedy only).

        Returns:
            List of activated label strings (one or more from
            ``ROUTER_LABELS``).  Guaranteed to be non-empty.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.use_turbo_dfs:
            return self._classify_turbo_dfs(image_path, question)
        return [self._classify_greedy(image_path, question, gen_config)]

    def _classify_turbo_dfs(self, image_path: str, question: str) -> List[str]:
        """Multi-label classification via Turbo DFS union strategy."""
        inputs = self._prepare_inputs(image_path, question)
        input_ids = inputs["input_ids"]

        eos_token_id = getattr(
            self.processor, "tokenizer", self.tokenizer
        ).eos_token_id
        if eos_token_id is None:
            eos_token_id = 2  # safe fallback

        candidates = inference_turbo_dfs(
            model=self.model,
            input_ids=input_ids,
            eos_token_id=eos_token_id,
            max_new_tokens=self.turbo_max_new_tokens,
            min_prob=self.turbo_min_prob,
            temperature=self.turbo_temperature,
        )

        if not candidates:
            logger.warning("Turbo DFS produced no candidates — falling back to greedy")
            return [self._classify_greedy(image_path, question)]

        activated: list[str] = []
        seen: set[str] = set()

        for score_val, token_seq, _ in candidates:
            prob = np.exp(-score_val)
            if prob < self.turbo_min_prob:
                continue

            if len(token_seq) == 0:
                continue
            decoded_text = getattr(
                self.processor, "tokenizer", self.tokenizer
            ).decode(token_seq, skip_special_tokens=True)
            decoded_text = clean_generated_text(decoded_text).strip()

            labels = self._extract_labels(decoded_text)
            for label in labels:
                if label not in seen:
                    activated.append(label)
                    seen.add(label)

        if not activated:
            best_text = getattr(
                self.processor, "tokenizer", self.tokenizer
            ).decode(candidates[0][1], skip_special_tokens=True)
            best_text = clean_generated_text(best_text).strip()
            labels = self._extract_labels(best_text)
            activated = labels if labels else ["others"]

        logger.info(
            "Turbo DFS routing: %d candidates → activated labels: %s",
            len(candidates),
            activated,
        )
        return activated

    def _classify_greedy(
        self,
        image_path: str,
        question: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Fallback single-label classification via standard generation."""
        config = gen_config or ROUTER_GEN_CONFIG
        inputs = self._prepare_inputs(image_path, question)

        generated_ids = self.model.generate(**inputs, **config.to_dict())
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        raw_label = clean_generated_text(output_text[0] if output_text else "").strip()
        return self._normalize_label(raw_label)

    @staticmethod
    def _extract_labels(text: str) -> List[str]:
        """Extract one or more canonical labels from decoded text.

        Supports comma-separated multi-label output as well as single
        labels.
        """
        extracted: list[str] = []
        parts = [p.strip() for p in text.split(",")]
        for part in parts:
            if not part:
                continue
            label = RouterModel._normalize_label(part)
            if label not in extracted:
                extracted.append(label)
        return extracted

    @staticmethod
    def _normalize_label(raw: str) -> str:
        """Map raw model output to a canonical label."""
        raw_lower = raw.strip().lower()
        for label in ROUTER_LABELS:
            if label.lower() in raw_lower:
                return label
        fuzzy = {
            "figure": "figure/diagram",
            "diagram": "figure/diagram",
            "chart": "figure/diagram",
            "image": "Image/Photo",
            "photo": "Image/Photo",
            "table": "table/list",
            "list": "table/list",
            "yes": "Yes/No",
            "no": "Yes/No",
            "hand": "handwritten",
            "ocr": "handwritten",
            "text": "free_text",
            "free": "free_text",
        }
        for keyword, label in fuzzy.items():
            if keyword in raw_lower:
                return label
        logger.warning("Router produced unrecognized label: '%s' → defaulting to 'others'", raw)
        return "others"

    def generate(
        self,
        image_path: str,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generic generate interface — delegates to ``classify``."""
        question = kwargs.get("question", prompt)
        labels = self.classify(image_path, question, gen_config)
        return ", ".join(labels)
