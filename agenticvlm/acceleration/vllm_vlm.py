"""vLLM-accelerated wrapper for vision-language models (Qwen2.5-VL, InternVL3, GLM-4.5V)."""

from __future__ import annotations

import base64
import gc
import logging
from pathlib import Path
from typing import Any, Optional

from agenticvlm.models.base import BaseVLM, GenerationConfig
from agenticvlm.utils.text_processing import clean_generated_text

logger = logging.getLogger(__name__)


class VLLMVisionModel(BaseVLM):
    """VLM served through a vLLM engine.

    Args:
        model_path: Path or HF hub id of the model.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use (0-1).
        max_model_len: Maximum sequence length (tokens).  ``None`` = auto.
        trust_remote_code: Whether to trust remote code for custom models.
        quantization: Quantization method (``"awq"``, ``"gptq"``, or ``None``).
        dtype: Data type (``"bfloat16"``, ``"float16"``, ``"auto"``).
        enforce_eager: Disable CUDA graph capture.
        limit_mm_per_prompt: Max multimodal items per prompt, e.g.
            ``{"image": 1}``.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        quantization: Optional[str] = None,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__(model_path, device_map="auto", torch_dtype=dtype)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.quantization = quantization
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.limit_mm_per_prompt = limit_mm_per_prompt or {"image": 1}
        self._llm: Any = None
        self._sampling_params_cls: Any = None

    def load(self) -> None:
        """Initialize the vLLM engine."""
        from vllm import LLM, SamplingParams  # noqa: F811

        logger.info(
            "Initializing vLLM engine for %s (tp=%d, mem=%.0f%%)",
            self.model_path,
            self.tensor_parallel_size,
            self.gpu_memory_utilization * 100,
        )

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "enforce_eager": self.enforce_eager,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.quantization is not None:
            kwargs["quantization"] = self.quantization

        self._llm = LLM(**kwargs)
        self._sampling_params_cls = SamplingParams
        self._loaded = True
        logger.info("vLLM engine ready for %s", self.model_path)

    def _build_sampling_params(
        self, gen_config: Optional[GenerationConfig] = None,
    ) -> Any:
        """Convert a ``GenerationConfig`` into vLLM ``SamplingParams``."""
        cfg = gen_config or GenerationConfig()
        kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_new_tokens,
            "temperature": max(cfg.temperature, 1e-7),
        }
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.min_p is not None:
            kwargs["min_p"] = cfg.min_p
        return self._sampling_params_cls(**kwargs)

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Return a ``data:`` URI for the image at *image_path*."""
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(
            suffix, "png",
        )
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/{mime};base64,{b64}"

    def generate(
        self,
        image_path: str,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from an image+prompt pair via vLLM."""
        if not self._loaded:
            raise RuntimeError("vLLM engine not loaded. Call load() first.")

        messages: list[dict[str, Any]] = []
        if system_message:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
            )

        data_uri = self._encode_image(image_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            },
        )

        sampling = self._build_sampling_params(gen_config)
        outputs = self._llm.chat(messages=[messages], sampling_params=sampling)
        text = outputs[0].outputs[0].text if outputs else ""
        return clean_generated_text(text.strip())

    def generate_batch(
        self,
        image_paths: list[str],
        prompts: list[str],
        gen_config: Optional[GenerationConfig] = None,
        system_message: Optional[str] = None,
    ) -> list[str]:
        """Generate answers for a batch of image+prompt pairs."""
        if not self._loaded:
            raise RuntimeError("vLLM engine not loaded. Call load() first.")
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        all_messages: list[list[dict[str, Any]]] = []
        for img, text in zip(image_paths, prompts):
            conv: list[dict[str, Any]] = []
            if system_message:
                conv.append(
                    {"role": "system", "content": [{"type": "text", "text": system_message}]},
                )
            data_uri = self._encode_image(img)
            conv.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": text},
                    ],
                },
            )
            all_messages.append(conv)

        sampling = self._build_sampling_params(gen_config)
        outputs = self._llm.chat(messages=all_messages, sampling_params=sampling)
        return [clean_generated_text(o.outputs[0].text.strip()) for o in outputs]

    def unload(self) -> None:
        """Shut down the vLLM engine and free GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        logger.info("vLLM engine unloaded: %s", self.model_path)
