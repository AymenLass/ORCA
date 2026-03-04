"""vLLM-accelerated inference backends for ORCA."""

from agenticvlm.acceleration.vllm_vlm import VLLMVisionModel
from agenticvlm.acceleration.vllm_llm import VLLMLLMModel

__all__ = [
    "VLLMVisionModel",
    "VLLMLLMModel",
]
