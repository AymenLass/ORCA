"""Document VQA evaluator — batch evaluation with ANLS scoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import pandas as pd

from agenticvlm.evaluation.anls import ANLSCalculator

logger = logging.getLogger(__name__)


class DocVQAEvaluator:
    """Batch evaluator for Document VQA predictions.

    Args:
        threshold: ANLS threshold (default 0.5).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.calculator = ANLSCalculator(threshold=threshold)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        prediction_col: str = "Predicted Answer",
        ground_truth_col: str = "Ground Truth",
    ) -> Dict[str, Any]:
        """Evaluate predictions stored in a DataFrame.

        Args:
            df: DataFrame with prediction and ground-truth columns.
            prediction_col: Column name for predictions.
            ground_truth_col: Column name for ground truths.

        Returns:
            Dictionary with ``anls``, ``num_samples``, ``per_sample``.
        """
        predictions = df[prediction_col].astype(str).tolist()
        ground_truths = df[ground_truth_col].astype(str).tolist()

        per_sample = []
        for idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            score = self.calculator.anls_score(pred, gt)
            per_sample.append(
                {
                    "index": idx,
                    "prediction": pred,
                    "ground_truth": gt,
                    "anls": score,
                }
            )

        avg_anls = sum(s["anls"] for s in per_sample) / len(per_sample) if per_sample else 0.0

        return {
            "anls": avg_anls,
            "num_samples": len(per_sample),
            "per_sample": per_sample,
        }

    def evaluate_lists(
        self,
        predictions: Sequence[str],
        ground_truths: Sequence[Union[str, List[str]]],
    ) -> Dict[str, Any]:
        """Evaluate from parallel lists.

        Args:
            predictions: Predicted answer strings.
            ground_truths: Ground-truth answer strings (or lists).

        Returns:
            Dictionary with ``anls`` and ``per_sample``.
        """
        avg = self.calculator.batch_anls(predictions, ground_truths)
        per_sample = []
        for idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            if isinstance(gt, str):
                score = self.calculator.anls_score(pred, gt)
            else:
                score = self.calculator.anls_score_multi(pred, gt)
            per_sample.append(
                {
                    "index": idx,
                    "prediction": pred,
                    "ground_truth": gt,
                    "anls": score,
                }
            )
        return {"anls": avg, "num_samples": len(per_sample), "per_sample": per_sample}

    def evaluate_json(
        self,
        predictions_file: str | Path,
        ground_truth_file: str | Path,
        pred_key: str = "answer",
        gt_key: str = "answers",
        question_id_key: str = "questionId",
    ) -> Dict[str, Any]:
        """Evaluate from JSON prediction and ground-truth files.

        Expects both files to contain lists of dicts with matching
        ``question_id_key``.

        Args:
            predictions_file: Path to predictions JSON.
            ground_truth_file: Path to ground-truth JSON.

        Returns:
            Dictionary with ``anls``, ``num_samples``, ``per_sample``.
        """
        with open(predictions_file) as f:
            preds_data = json.load(f)
        with open(ground_truth_file) as f:
            gt_data = json.load(f)

        gt_index: Dict[str, Any] = {}
        for item in gt_data:
            qid = str(item[question_id_key])
            gt_index[qid] = item.get(gt_key, item.get("answer", ""))

        predictions = []
        ground_truths = []
        for item in preds_data:
            qid = str(item[question_id_key])
            pred = str(item.get(pred_key, ""))
            gt = gt_index.get(qid, "")
            predictions.append(pred)
            ground_truths.append(gt if isinstance(gt, list) else str(gt))

        return self.evaluate_lists(predictions, ground_truths)

    def save_report(
        self,
        results: Dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Save evaluation results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Evaluation report saved to %s", output_path)
