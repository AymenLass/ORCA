"""vLLM-accelerated wrapper for text-only LLMs (Qwen3-8B, Qwen3-1.7B)."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

from agenticvlm.models.base import BaseVLM, GenerationConfig
from agenticvlm.utils.text_processing import clean_generated_text

logger = logging.getLogger(__name__)


class VLLMLLMModel(BaseVLM):
    """Text-only LLM served through a vLLM engine.

    Args:
        model_path: Path or HF hub id.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_model_len: Maximum sequence length. ``None`` = auto.
        enable_thinking: Whether to enable Qwen3-style thinking mode.
        dtype: Data type.
    """

    THINK_END = "</think>"

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        enable_thinking: bool = False,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__(model_path, device_map="auto", torch_dtype=dtype)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.dtype = dtype
        self._llm: Any = None
        self._tokenizer: Any = None
        self._sampling_params_cls: Any = None

    def load(self) -> None:
        """Initialize the vLLM engine for text generation."""
        from vllm import LLM, SamplingParams

        logger.info(
            "Initializing vLLM LLM engine for %s (tp=%d, thinking=%s)",
            self.model_path,
            self.tensor_parallel_size,
            self.enable_thinking,
        )

        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**kwargs)
        self._sampling_params_cls = SamplingParams
        self._tokenizer = self._llm.get_tokenizer()
        self._loaded = True
        logger.info("vLLM LLM engine ready for %s", self.model_path)

    def _build_sampling_params(
        self, gen_config: Optional[GenerationConfig] = None,
    ) -> Any:
        cfg = gen_config or GenerationConfig(max_new_tokens=32768)
        kwargs: dict[str, Any] = {
            "max_tokens": cfg.max_new_tokens,
            "temperature": max(cfg.temperature, 1e-7),
        }
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.min_p is not None:
            kwargs["min_p"] = cfg.min_p
        return self._sampling_params_cls(**kwargs)

    def _strip_thinking(self, text: str) -> str:
        """Remove ``<think>...</think>`` block if present."""
        idx = text.find(self.THINK_END)
        if idx != -1:
            return text[idx + len(self.THINK_END) :].strip()
        return text.strip()

    def generate(
        self,
        image_path: str | None,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text (text-only model, *image_path* is ignored)."""
        if not self._loaded:
            raise RuntimeError("vLLM engine not loaded. Call load() first.")

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        sampling = self._build_sampling_params(gen_config)
        chat_kwargs: dict[str, Any] = {
            "messages": [messages],
            "sampling_params": sampling,
        }
        if self.enable_thinking and hasattr(self._tokenizer, "apply_chat_template"):
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

        outputs = self._llm.chat(**chat_kwargs)
        raw = outputs[0].outputs[0].text if outputs else ""
        content = self._strip_thinking(raw) if self.enable_thinking else raw
        return clean_generated_text(content)

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

    def unload(self) -> None:
        """Shut down the vLLM engine and free memory."""
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
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
        logger.info("vLLM LLM engine unloaded: %s", self.model_path)
