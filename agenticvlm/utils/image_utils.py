"""Image loading and preprocessing utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


def load_and_convert_image(
    image_path: Union[str, Path],
    mode: str = "L",
) -> Optional[Image.Image]:
    """Load an image from disk and convert to the specified mode.

    Args:
        image_path: Path to the image file.
        mode: PIL image mode (``'L'`` for grayscale, ``'RGB'`` for colour).

    Returns:
        PIL Image in the requested mode, or ``None`` on failure.
    """
    try:
        img = Image.open(image_path)
        if img.mode != mode:
            img = img.convert(mode)
        return img
    except Exception as e:
        logger.error("Error loading image %s: %s", image_path, e)
        return None


def load_image_rgb(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """Load an image and convert to RGB mode.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image in RGB mode, or ``None`` on failure.
    """
    return load_and_convert_image(image_path, mode="RGB")


def load_image_grayscale(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """Load an image and convert to grayscale (mode L)."""
    return load_and_convert_image(image_path, mode="L")
