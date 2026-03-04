"""Device management and GPU utilities."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """Resolve the compute device.

    Args:
        device: Explicit device string (e.g. ``'cuda:0'``, ``'cpu'``).
            If ``None``, auto-selects CUDA if available.

    Returns:
        Resolved ``torch.device``.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_memory_info() -> dict[str, float]:
    """Get current GPU memory statistics in GB.

    Returns:
        Dictionary with ``total``, ``reserved``, ``allocated``, and ``free``
        memory in GB. Returns zeros if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {"total": 0.0, "reserved": 0.0, "allocated": 0.0, "free": 0.0}

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024**3)
    reserved = torch.cuda.max_memory_reserved() / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    free = total - allocated

    return {
        "total": round(total, 3),
        "reserved": round(reserved, 3),
        "allocated": round(allocated, 3),
        "free": round(free, 3),
    }


def log_gpu_stats() -> None:
    """Log current GPU memory statistics."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return

    info = get_gpu_memory_info()
    gpu_name = torch.cuda.get_device_properties(0).name
    logger.info(
        "GPU: %s | Total: %.1f GB | Allocated: %.1f GB | Free: %.1f GB",
        gpu_name,
        info["total"],
        info["allocated"],
        info["free"],
    )


def clear_gpu_cache() -> None:
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.debug("GPU cache cleared")


def safe_model_cleanup(*models: Any) -> None:
    """Safely delete model objects and free GPU memory.

    Args:
        *models: Model objects to delete. ``None`` values are skipped.
    """
    for model in models:
        if model is not None:
            del model

    clear_gpu_cache()
    logger.info("Model(s) cleaned up and GPU cache cleared")
