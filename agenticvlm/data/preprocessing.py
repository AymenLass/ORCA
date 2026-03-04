"""Data preprocessing utilities for router training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from agenticvlm.data.augmentation import augment_training_sample
from agenticvlm.data.label_definitions import ROUTER_LABELS
from agenticvlm.prompts.router_prompts import ROUTER_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


def prepare_router_training_data(
    csv_path: str | Path,
    image_dir: str | Path,
    output_path: Optional[str | Path] = None,
    question_col: str = "Question",
    label_col: str = "label",
    image_col: str = "image_path",
    grayscale: bool = True,
    augment: bool = False,
    augment_dir: Optional[str | Path] = None,
    n_text_variants: int = 2,
    n_image_variants: int = 2,
    translator: Optional[Callable] = None,
    seed: int = 3407,
) -> List[Dict[str, Any]]:
    """Convert a labeled CSV into the conversation format for router training.

    Each sample becomes a conversation with:
    - A user message containing the grayscale image and the classification prompt.
    - An assistant message containing the label.

    When ``augment=True``, the two strategies from Appendix B.1 are applied:

    1. **Back-translation** — questions are paraphrased via round-trip
       translation through French and Chinese (or via lightweight
       templates when no translator API is available).
    2. **Document perturbations** — rotation, contrast, and brightness
       adjustments simulate scanning variations.

    Args:
        csv_path: Path to the labeled CSV file.
        image_dir: Base directory for resolving image paths.
        output_path: Optional path to save the processed data as JSON.
        question_col: Column name for the question text.
        label_col: Column name for the label.
        image_col: Column name for image paths.
        grayscale: Whether to convert images to grayscale (mode 'L').
        augment: Whether to apply data augmentation (B.1).
        augment_dir: Directory for augmented images (defaults to
            ``image_dir / _augmented``).
        n_text_variants: Number of back-translated question variants.
        n_image_variants: Number of perturbed image variants.
        translator: Optional translation callable with signature
            ``translator(text, src_lang, tgt_lang) -> str``.
        seed: Random seed for augmentation reproducibility.

    Returns:
        List of conversation dictionaries.
    """
    df = pd.read_csv(csv_path)
    image_dir = Path(image_dir)

    conversations: List[Dict[str, Any]] = []
    skipped = 0

    for _, row in df.iterrows():
        question = str(row[question_col])
        label = str(row[label_col]).strip()
        image_ref = str(row[image_col])

        if label not in ROUTER_LABELS and label not in {l.lower() for l in ROUTER_LABELS}:
            logger.warning("Skipping unknown label: %s", label)
            skipped += 1
            continue

        if label not in ROUTER_LABELS:
            label = next(l for l in ROUTER_LABELS if l.lower() == label.lower())

        image_path = Path(image_ref)
        if not image_path.is_absolute():
            image_path = image_dir / image_ref

        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            skipped += 1
            continue

        if grayscale:
            try:
                img = Image.open(image_path)
                if img.mode != "L":
                    gray_path = image_path.parent / f"gray_{image_path.name}"
                    img.convert("L").save(gray_path)
                    image_path = gray_path
            except Exception as e:
                logger.warning("Failed to convert %s to grayscale: %s", image_path, e)
                skipped += 1
                continue

        prompt = ROUTER_CLASSIFICATION_PROMPT.format(question=question)

        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label},
                    ],
                },
            ]
        }
        conversations.append(conversation)

    logger.info(
        "Prepared %d training samples (%d skipped) from %s",
        len(conversations),
        skipped,
        csv_path,
    )

    # ----- Data Augmentation (Appendix B.1) -----
    if augment:
        aug_out = Path(augment_dir) if augment_dir else (image_dir / "_augmented")
        aug_out.mkdir(parents=True, exist_ok=True)

        augmented_samples: List[Dict[str, Any]] = []
        for idx, sample in enumerate(conversations):
            extras = augment_training_sample(
                sample,
                output_dir=aug_out,
                do_back_translation=True,
                do_image_perturbation=True,
                translator=translator,
                n_text_variants=n_text_variants,
                n_image_variants=n_image_variants,
                seed=seed + idx if seed else None,
            )
            augmented_samples.extend(extras)

        logger.info(
            "Augmentation produced %d extra samples (%.1f× expansion)",
            len(augmented_samples),
            1 + len(augmented_samples) / max(len(conversations), 1),
        )
        conversations.extend(augmented_samples)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(conversations, f, indent=2)
        logger.info("Saved processed data to %s", output_path)

    return conversations


def multilabel_stratified_kfold(
    csv_path: str | Path,
    label_col: str = "label",
    n_splits: int = 8,
    seed: int = 3407,
) -> List[tuple[List[int], List[int]]]:
    """Multilabel Stratified K-Fold split for router evaluation.

    As described in Appendix B.1, this preserves the distribution of
    label combinations across folds — critical because some agent
    combinations are significantly rarer than others.

    Args:
        csv_path: Path to the labeled CSV.
        label_col: Column containing labels (comma-separated for multi-label).
        n_splits: Number of folds (default 8, per paper).
        seed: Random seed.

    Returns:
        List of ``(train_indices, val_indices)`` tuples, one per fold.
    """
    df = pd.read_csv(csv_path)
    labels_raw = df[label_col].astype(str).tolist()

    all_labels = sorted({
        lbl.strip()
        for row in labels_raw
        for lbl in row.split(",")
        if lbl.strip()
    })
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    n_samples = len(labels_raw)
    n_labels = len(all_labels)
    label_matrix = np.zeros((n_samples, n_labels), dtype=int)
    for i, row in enumerate(labels_raw):
        for lbl in row.split(","):
            lbl = lbl.strip()
            if lbl in label_to_idx:
                label_matrix[i, label_to_idx[lbl]] = 1

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        skf = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed,
        )
        folds = list(skf.split(np.zeros(n_samples), label_matrix))
    except ImportError:
        logger.warning(
            "iterative-stratification not installed — "
            "falling back to simple stratified split. "
            "Install via: pip install iterative-stratification"
        )
        from sklearn.model_selection import StratifiedKFold

        primary = [row.split(",")[0].strip() for row in labels_raw]
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed,
        )
        folds = list(skf.split(np.zeros(n_samples), primary))

    logger.info(
        "Created %d-fold stratified split (%d samples, %d unique labels)",
        n_splits,
        n_samples,
        n_labels,
    )
    return folds
