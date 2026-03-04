"""Prediction CLI — run the AgenticVLM pipeline on a dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(cfg: dict):
    """Instantiate the full AgenticVLM pipeline from config."""
    from agenticvlm.models.glm4v import GLM4VModel
    from agenticvlm.models.internvl3 import InternVL3Model
    from agenticvlm.models.qwen25vl import Qwen25VLModel
    from agenticvlm.models.qwen2_ocr import Qwen2OCRModel
    from agenticvlm.models.qwen3 import Qwen3Model
    from agenticvlm.models.router import RouterModel
    from agenticvlm.pipeline.pipeline import AgenticVLMPipeline

    models_cfg = cfg["models"]

    thinker = GLM4VModel(
        model_path=models_cfg["thinker"]["path"],
        device_map=models_cfg["thinker"].get("device_map", "auto"),
    )
    router = RouterModel(
        model_path=models_cfg["router"]["base_model"],
        adapter_path=models_cfg["router"].get("adapter_path"),
        use_unsloth=models_cfg["router"].get("use_unsloth", True),
    )
    specialist = Qwen25VLModel(
        model_path=models_cfg["specialist"]["path"],
        device_map=models_cfg["specialist"].get("device_map", "auto"),
    )
    vision_expert = InternVL3Model(
        model_path=models_cfg["vision_expert"]["path"],
        device_map=models_cfg["vision_expert"].get("device_map", "auto"),
    )
    ocr_model = Qwen2OCRModel(
        model_path=models_cfg["ocr"]["path"],
        device_map=models_cfg["ocr"].get("device_map", "auto"),
    )
    debate_llm = Qwen3Model(
        model_path=models_cfg["debate_llm"]["path"],
        device_map=models_cfg["debate_llm"].get("device_map", "auto"),
        enable_thinking=models_cfg["debate_llm"].get("enable_thinking", False),
    )
    checker_llm = Qwen3Model(
        model_path=models_cfg["checker_llm"]["path"],
        device_map=models_cfg["checker_llm"].get("device_map", "auto"),
        enable_thinking=models_cfg["checker_llm"].get("enable_thinking", True),
    )

    return AgenticVLMPipeline(
        thinker=thinker,
        router=router,
        specialist_model=specialist,
        vision_expert=vision_expert,
        ocr_model=ocr_model,
        debate_llm=debate_llm,
        checker_llm=checker_llm,
    )


def main(argv: list[str] | None = None) -> None:
    """Run predictions on a dataset."""
    parser = argparse.ArgumentParser(
        description="AgenticVLM — run the multi-agent DocVQA pipeline",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to input data CSV/JSON file.",
    )
    parser.add_argument(
        "--image-dir", "-i",
        type=str,
        default=None,
        help="Base directory for image paths.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="predictions.csv",
        help="Output file path for predictions.",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only the first N samples.",
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

    cfg = load_config(args.config)

    pipeline = build_pipeline(cfg)
    pipeline.load_all_models()

    from agenticvlm.data.dataset import DocVQADataset

    dataset = DocVQADataset(
        data_path=args.data,
        image_dir=args.image_dir or cfg.get("data", {}).get("image_dir", "."),
    )

    results = []
    samples = list(dataset)
    if args.limit:
        samples = samples[: args.limit]

    for idx, sample in enumerate(samples):
        logger.info("Processing %d/%d: %s", idx + 1, len(samples), sample["question"][:60])
        try:
            result = pipeline.predict(sample["image_path"], sample["question"])
            results.append(
                {
                    "question_id": sample["question_id"],
                    "question": sample["question"],
                    "image_path": sample["image_path"],
                    "predicted_answer": result.final_answer,
                    "router_labels": result.router_labels,
                    "primary_label": result.primary_label,
                    "thinker_answer": result.thinker_answer,
                }
            )
        except Exception as e:
            logger.error("Error processing sample %d: %s", idx, e)
            results.append(
                {
                    "question_id": sample["question_id"],
                    "question": sample["question"],
                    "image_path": sample["image_path"],
                    "predicted_answer": "",
                    "error": str(e),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        pd.DataFrame(results).to_csv(output_path, index=False)

    logger.info("Saved %d predictions to %s", len(results), output_path)


if __name__ == "__main__":
    main()
