"""vLLM-accelerated prediction CLI — drop-in replacement for predict.py."""

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


def build_pipeline_vllm(cfg: dict):
    """Instantiate the AgenticVLM pipeline with vLLM-backed models.

    The router retains its HuggingFace backend because Turbo DFS
    requires direct logit access.
    """
    from agenticvlm.acceleration.vllm_llm import VLLMLLMModel
    from agenticvlm.acceleration.vllm_vlm import VLLMVisionModel
    from agenticvlm.models.router import RouterModel
    from agenticvlm.pipeline.pipeline import AgenticVLMPipeline

    models_cfg = cfg["models"]
    vllm_cfg = cfg.get("vllm", {})

    tp = vllm_cfg.get("tensor_parallel_size", 1)
    mem = vllm_cfg.get("gpu_memory_utilization", 0.85)

    thinker = VLLMVisionModel(
        model_path=models_cfg["thinker"]["path"],
        tensor_parallel_size=tp,
        gpu_memory_utilization=mem,
        dtype=models_cfg["thinker"].get("dtype", "bfloat16"),
    )

    router = RouterModel(
        model_path=models_cfg["router"]["base_model"],
        adapter_path=models_cfg["router"].get("adapter_path"),
        use_unsloth=models_cfg["router"].get("use_unsloth", True),
    )

    specialist = VLLMVisionModel(
        model_path=models_cfg["specialist"]["path"],
        tensor_parallel_size=tp,
        gpu_memory_utilization=mem,
        dtype=models_cfg["specialist"].get("dtype", "auto"),
    )

    vision_expert = VLLMVisionModel(
        model_path=models_cfg["vision_expert"]["path"],
        tensor_parallel_size=tp,
        gpu_memory_utilization=mem,
        dtype=models_cfg["vision_expert"].get("dtype", "bfloat16"),
    )

    ocr_model = VLLMVisionModel(
        model_path=models_cfg["ocr"]["path"],
        tensor_parallel_size=tp,
        gpu_memory_utilization=mem,
        dtype=models_cfg["ocr"].get("dtype", "bfloat16"),
    )

    debate_llm = VLLMLLMModel(
        model_path=models_cfg["debate_llm"]["path"],
        tensor_parallel_size=tp,
        gpu_memory_utilization=mem,
        enable_thinking=models_cfg["debate_llm"].get("enable_thinking", False),
    )

    checker_llm = VLLMLLMModel(
        model_path=models_cfg["checker_llm"]["path"],
        tensor_parallel_size=tp,
        gpu_memory_utilization=mem,
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
    """Run vLLM-accelerated predictions on a dataset."""
    parser = argparse.ArgumentParser(
        description="AgenticVLM — vLLM-accelerated inference",
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
        "--tensor-parallel", "-tp",
        type=int,
        default=None,
        help="Override tensor_parallel_size from config.",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=None,
        help="Override gpu_memory_utilization (0-1) from config.",
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

    if args.tensor_parallel is not None:
        cfg.setdefault("vllm", {})["tensor_parallel_size"] = args.tensor_parallel
    if args.gpu_mem is not None:
        cfg.setdefault("vllm", {})["gpu_memory_utilization"] = args.gpu_mem

    pipeline = build_pipeline_vllm(cfg)
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
