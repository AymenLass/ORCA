"""Abstract base class for all vision-language models in AgenticVLM.

Subclasses must implement :meth:`load`, :meth:`generate`, and :meth:`unload`.
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Generation hyperparameters shared across models.

    Attributes:
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature. Near-zero for deterministic output.
        do_sample: Whether to use sampling (vs greedy decoding).
        min_p: Minimum probability threshold for nucleus sampling.
    """

    max_new_tokens: int = 128
    temperature: float = 1e-20
    do_sample: bool = False
    min_p: Optional[float] = None
    top_p: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to kwargs dict for ``model.generate()``."""
        d: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.do_sample:
            d["do_sample"] = True
        if self.min_p is not None:
            d["min_p"] = self.min_p
        if self.top_p is not None:
            d["top_p"] = self.top_p
        d.update(self.extra)
        return d


class BaseVLM(ABC):
    """Abstract base for VLM wrappers.

    Args:
        model_path: Path to the pretrained model weights.
        device_map: Device mapping strategy (default ``"auto"``).
        torch_dtype: Tensor dtype for model weights.
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: Any = None,
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.model: Any = None
        self.processor: Any = None
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""

    @abstractmethod
    def generate(
        self,
        image_path: str,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text given an image and prompt.

        Args:
            image_path: Path to the input image.
            prompt: Text prompt for the model.
            gen_config: Generation configuration. Uses model defaults if None.

        Returns:
            Generated text string.
        """

    def unload(self) -> None:
        """Release model resources and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model unloaded: %s", self.model_path)

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(path={self.model_path!r}, {status})"
