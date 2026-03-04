"""DocVQA dataset loader for inference and evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DocVQADataset:
    """Iterable dataset for Document VQA samples.

    Supports loading from CSV (the format used in the original notebooks)
    and JSON (the standard DocVQA challenge format).

    Each yielded sample is a dictionary with at minimum:
    - ``question_id``
    - ``question``
    - ``image_path``
    - ``answers`` (list of acceptable ground-truth answers, if available)

    Args:
        data_path: Path to CSV or JSON data file.
        image_dir: Base directory for resolving relative image paths.
        split: Optional split name (e.g. ``"val"``, ``"test"``).
    """

    def __init__(
        self,
        data_path: str | Path,
        image_dir: Optional[str | Path] = None,
        split: Optional[str] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir) if image_dir else self.data_path.parent
        self.split = split
        self._samples: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            self._load_csv()
        elif suffix == ".json":
            self._load_json()
        else:
            raise ValueError(f"Unsupported data format: {suffix}")
        logger.info("Loaded %d samples from %s", len(self._samples), self.data_path)

    def _load_csv(self) -> None:
        df = pd.read_csv(self.data_path)
        for idx, row in df.iterrows():
            sample: Dict[str, Any] = {
                "question_id": str(row.get("questionId", row.get("question_id", idx))),
                "question": str(row.get("Question", row.get("question", ""))),
                "image_path": self._resolve_image(
                    str(row.get("image_path", row.get("image", "")))
                ),
            }
            # Ground truth (may not exist in test sets)
            gt = row.get("Ground Truth", row.get("answers", None))
            if gt is not None:
                if isinstance(gt, str):
                    try:
                        sample["answers"] = json.loads(gt)
                    except (json.JSONDecodeError, TypeError):
                        sample["answers"] = [str(gt)]
                else:
                    sample["answers"] = [str(gt)]
            else:
                sample["answers"] = []

            # Optional: predicted answer from prior stage
            pred = row.get("Predicted Answer", row.get("predicted", None))
            if pred is not None:
                sample["predicted_answer"] = str(pred).strip()

            self._samples.append(sample)

    def _load_json(self) -> None:
        with open(self.data_path) as f:
            data = json.load(f)

        items = data if isinstance(data, list) else data.get("data", [])
        for item in items:
            sample: Dict[str, Any] = {
                "question_id": str(item.get("questionId", item.get("question_id", ""))),
                "question": str(item.get("question", "")),
                "image_path": self._resolve_image(str(item.get("image", ""))),
                "answers": item.get("answers", []),
            }
            self._samples.append(sample)

    def _resolve_image(self, image_ref: str) -> str:
        """Resolve an image reference to an absolute path."""
        p = Path(image_ref)
        if p.is_absolute() and p.exists():
            return str(p)
        resolved = self.image_dir / image_ref
        return str(resolved)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._samples[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._samples)
