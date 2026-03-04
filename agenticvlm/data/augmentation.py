"""Data augmentation for router training (Appendix B.1)."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Back-translation augmentation
# ---------------------------------------------------------------------------

_PARAPHRASE_TEMPLATES: List[str] = [
    "Based on the document, {q}",
    "From the information shown, {q}",
    "Looking at this document, {q}",
    "According to the content presented, {q}",
    "Referring to the document image, {q}",
    "As indicated in the document, {q}",
]


def back_translate(
    question: str,
    *,
    translator: Any | None = None,
    intermediate_languages: Sequence[str] = ("fr", "zh"),
    n_variants: int = 2,
    seed: int | None = None,
) -> List[str]:
    """Generate paraphrased question variants via back-translation.

    If a ``translator`` callable is provided it is used to round-trip
    translate the question through each intermediate language::

        en → fr → en
        en → zh → en

    Otherwise lightweight rule-based paraphrasing templates are applied.

    Args:
        question: The original English question.
        translator: Optional callable with signature
            ``translator(text: str, src: str, tgt: str) -> str``.
        intermediate_languages: Languages to translate through.
        n_variants: Number of paraphrased variants to return.
        seed: Random seed for reproducibility.

    Returns:
        List of paraphrased question strings (may include the original).
    """
    rng = random.Random(seed)

    if translator is not None:
        variants: List[str] = []
        for lang in intermediate_languages:
            try:
                intermediate = translator(question, "en", lang)
                back = translator(intermediate, lang, "en")
                if back and back.strip() != question.strip():
                    variants.append(back.strip())
            except Exception as e:
                logger.warning(
                    "Back-translation via '%s' failed: %s", lang, e,
                )
        if variants:
            return variants[:n_variants]

    templates = list(_PARAPHRASE_TEMPLATES)
    rng.shuffle(templates)
    variants = []
    for tmpl in templates[:n_variants]:
        q_lower = question[0].lower() + question[1:] if question else question
        if q_lower.endswith("?"):
            variants.append(tmpl.format(q=q_lower))
        else:
            variants.append(tmpl.format(q=q_lower + "?"))
    return variants


# ---------------------------------------------------------------------------
# 2. Document image perturbations
# ---------------------------------------------------------------------------

def apply_document_perturbations(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    rotation_range: tuple[float, float] = (-5.0, 5.0),
    contrast_range: tuple[float, float] = (0.7, 1.3),
    brightness_range: tuple[float, float] = (0.8, 1.2),
    add_noise: bool = True,
    n_variants: int = 2,
    seed: int | None = None,
) -> List[Path]:
    """Generate augmented document image variants.

    Applies minor transformations simulating scanning variations:
    - Small rotation (default ±5°)
    - Contrast adjustment
    - Brightness adjustment
    - Optional Gaussian noise

    Args:
        image_path: Path to the original document image.
        output_dir: Directory to save augmented images.
        rotation_range: Min/max rotation in degrees.
        contrast_range: Min/max contrast factor (1.0 = original).
        brightness_range: Min/max brightness factor.
        add_noise: Whether to add light Gaussian blur as noise proxy.
        n_variants: Number of perturbed variants to create.
        seed: Random seed for reproducibility.

    Returns:
        List of paths to the generated augmented images.
    """
    rng = random.Random(seed)
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.warning("Failed to open image %s: %s", image_path, e)
        return []

    augmented_paths: List[Path] = []
    stem = image_path.stem
    suffix = image_path.suffix or ".png"

    for i in range(n_variants):
        aug_img = img.copy()

        # Rotation
        angle = rng.uniform(*rotation_range)
        aug_img = aug_img.rotate(
            angle, resample=Image.BICUBIC, expand=False, fillcolor="white",
        )

        # Contrast
        factor = rng.uniform(*contrast_range)
        aug_img = ImageEnhance.Contrast(aug_img).enhance(factor)

        # Brightness
        factor = rng.uniform(*brightness_range)
        aug_img = ImageEnhance.Brightness(aug_img).enhance(factor)

        # Light blur as noise proxy
        if add_noise and rng.random() > 0.5:
            aug_img = aug_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        out_path = output_dir / f"{stem}_aug{i}{suffix}"
        aug_img.save(out_path)
        augmented_paths.append(out_path)

    return augmented_paths


# ---------------------------------------------------------------------------
# 3. Combined augmentation pipeline
# ---------------------------------------------------------------------------

def augment_training_sample(
    sample: Dict[str, Any],
    output_dir: str | Path,
    *,
    do_back_translation: bool = True,
    do_image_perturbation: bool = True,
    translator: Any | None = None,
    n_text_variants: int = 2,
    n_image_variants: int = 2,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Augment a single router training sample.

    Generates new samples by combining back-translated question
    variants with perturbed document images.

    Args:
        sample: Original training sample with ``messages`` in the
            SFTTrainer conversation format.
        output_dir: Directory for augmented images.
        do_back_translation: Whether to apply question back-translation.
        do_image_perturbation: Whether to apply image perturbations.
        translator: Optional translation callable.
        n_text_variants: Number of question paraphrases.
        n_image_variants: Number of image perturbations.
        seed: Random seed.

    Returns:
        List of augmented sample dictionaries (excluding the original).
    """
    output_dir = Path(output_dir)
    messages = sample.get("messages", [])
    if not messages:
        return []

    user_msg = messages[0]
    assistant_msg = messages[1] if len(messages) > 1 else None
    if assistant_msg is None:
        return []

    original_image = None
    original_text = None
    for content_item in user_msg.get("content", []):
        if isinstance(content_item, dict):
            if content_item.get("type") == "image":
                original_image = content_item.get("image", "")
            elif content_item.get("type") == "text":
                original_text = content_item.get("text", "")

    if not original_text:
        return []

    label_content = assistant_msg.get("content", [])
    label = ""
    if isinstance(label_content, list):
        for item in label_content:
            if isinstance(item, dict) and item.get("type") == "text":
                label = item.get("text", "")
                break
    elif isinstance(label_content, str):
        label = label_content

    text_variants: List[str] = [original_text]
    if do_back_translation:

        q_match = re.search(r'Question:\s*"?(.+?)"?\s*$', original_text, re.MULTILINE)
        if q_match:
            raw_question = q_match.group(1)
            paraphrases = back_translate(
                raw_question,
                translator=translator,
                n_variants=n_text_variants,
                seed=seed,
            )
            for para in paraphrases:
                text_variants.append(
                    original_text.replace(raw_question, para)
                )

    image_variants: List[str] = [original_image] if original_image else []
    if do_image_perturbation and original_image:
        aug_paths = apply_document_perturbations(
            original_image,
            output_dir,
            n_variants=n_image_variants,
            seed=seed,
        )
        image_variants.extend(str(p) for p in aug_paths)

    # Cross-product of text and image variants (excluding original × original)
    augmented: List[Dict[str, Any]] = []
    for t_idx, text_var in enumerate(text_variants):
        for i_idx, img_var in enumerate(image_variants):
            if t_idx == 0 and i_idx == 0:
                continue  # skip original
            new_user_content = []
            if img_var:
                new_user_content.append({"type": "image", "image": img_var})
            new_user_content.append({"type": "text", "text": text_var})

            new_sample: Dict[str, Any] = {
                "messages": [
                    {"role": "user", "content": new_user_content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": label}],
                    },
                ]
            }
            augmented.append(new_sample)

    return augmented
