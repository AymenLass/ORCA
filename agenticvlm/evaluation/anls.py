"""ANLS (Average Normalized Levenshtein Similarity) metric.

The primary evaluation metric for Document Visual Question Answering,
as used in the SP-DocVQA and MP-DocVQA benchmarks.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Union

import Levenshtein

logger = logging.getLogger(__name__)


class ANLSCalculator:
    """Computes ANLS between predicted and ground-truth answers.

    The ANLS metric is defined as:

    .. math::

        \\text{ANLS}(p, g) = \\begin{cases}
            1 - \\text{NLD}(p, g) & \\text{if } \\text{NLD}(p, g) < \\tau \\\\
            0 & \\text{otherwise}
        \\end{cases}

    where NLD is the Normalized Levenshtein Distance and :math:`\\tau`
    is the threshold (default 0.5).

    Args:
        threshold: ANLS threshold. Pairs with NLD >= threshold score 0.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def anls_score(self, prediction: str, ground_truth: str) -> float:
        """Compute ANLS between a prediction and ground truth.

        Args:
            prediction: The predicted answer string.
            ground_truth: The ground-truth answer string.

        Returns:
            ANLS score in [0, 1].
        """
        pred = prediction.lower().strip()
        gt = ground_truth.lower().strip()

        if not gt and not pred:
            return 1.0
        if not gt or not pred:
            return 0.0

        distance = Levenshtein.distance(pred, gt)
        max_len = max(len(pred), len(gt))
        nld = distance / max_len

        if nld < self.threshold:
            return 1.0 - nld
        return 0.0

    def anls_score_multi(
        self,
        prediction: str,
        ground_truths: Sequence[str],
    ) -> float:
        """Compute ANLS against multiple acceptable ground truths.

        Args:
            prediction: The predicted answer string.
            ground_truths: One or more acceptable ground-truth strings.

        Returns:
            Maximum ANLS score in [0, 1].
        """
        if not ground_truths:
            return 0.0
        return max(self.anls_score(prediction, gt) for gt in ground_truths)

    def batch_anls(
        self,
        predictions: Sequence[str],
        ground_truths: Sequence[Union[str, List[str]]],
    ) -> float:
        """Compute average ANLS over a batch of predictions.

        Args:
            predictions: List of predicted answer strings.
            ground_truths: List of ground-truth strings (or lists of
                acceptable ground-truth variants per sample).

        Returns:
            Average ANLS score in [0, 1].

        Raises:
            ValueError: If lengths of predictions and ground_truths differ.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(ground_truths)} ground truths"
            )

        if not predictions:
            return 0.0

        total = 0.0
        for pred, gt in zip(predictions, ground_truths):
            if isinstance(gt, str):
                total += self.anls_score(pred, gt)
            else:
                total += self.anls_score_multi(pred, gt)

        return total / len(predictions)
