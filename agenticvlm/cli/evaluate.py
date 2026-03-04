"""Evaluation CLI — compute ANLS metrics on predictions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Evaluate predictions against ground-truth answers."""
    parser = argparse.ArgumentParser(
        description="AgenticVLM — evaluate DocVQA predictions",
    )
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        required=True,
        help="Path to predictions CSV or JSON.",
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        default=None,
        help="Path to separate ground-truth file (if not in predictions).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the evaluation report (JSON).",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="predicted_answer",
        help="Column name for predictions (CSV mode).",
    )
    parser.add_argument(
        "--gt-col",
        type=str,
        default="Ground Truth",
        help="Column name for ground truth (CSV mode).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="ANLS threshold.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    from agenticvlm.evaluation.evaluator import DocVQAEvaluator

    evaluator = DocVQAEvaluator(threshold=args.threshold)

    pred_path = Path(args.predictions)

    if pred_path.suffix == ".json" and args.ground_truth:
        results = evaluator.evaluate_json(
            predictions_file=pred_path,
            ground_truth_file=args.ground_truth,
        )
    elif pred_path.suffix == ".csv":
        df = pd.read_csv(pred_path)
        if args.ground_truth:
            gt_df = pd.read_csv(args.ground_truth)
            df[args.gt_col] = gt_df[args.gt_col]
        results = evaluator.evaluate_dataframe(
            df,
            prediction_col=args.pred_col,
            ground_truth_col=args.gt_col,
        )
    else:
        logger.error("Unsupported file format: %s", pred_path.suffix)
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  AgenticVLM Evaluation Report")
    print(f"{'='*50}")
    print(f"  ANLS Score:    {results['anls']:.4f}")
    print(f"  Num Samples:   {results['num_samples']}")
    print(f"  Threshold:     {args.threshold}")
    print(f"{'='*50}\n")

    if args.output:
        evaluator.save_report(results, args.output)


if __name__ == "__main__":
    main()
