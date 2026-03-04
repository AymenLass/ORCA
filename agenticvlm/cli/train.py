"""Training CLI — finetune the router model with LoRA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Train the AgenticVLM router via LoRA finetuning."""
    parser = argparse.ArgumentParser(
        description="AgenticVLM — train the question-type router",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data CSV with labels.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Base directory for training images.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for the trained adapter.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save merged model (base + LoRA) after training.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation (back-translation + image perturbations).",
    )
    parser.add_argument(
        "--n-text-variants",
        type=int,
        default=2,
        help="Number of back-translated question variants per sample.",
    )
    parser.add_argument(
        "--n-image-variants",
        type=int,
        default=2,
        help="Number of perturbed image variants per sample.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Run Multilabel Stratified K-Fold CV (default: no CV, train on all).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    models_cfg = cfg.get("models", {})

    from agenticvlm.training.router_trainer import RouterTrainer, RouterTrainingConfig

    config = RouterTrainingConfig(
        base_model=models_cfg.get("router", {}).get("base_model", "unsloth/Qwen2.5-VL-7B-Instruct"),
        output_dir=args.output_dir or train_cfg.get("output_dir", "outputs/router"),
        num_train_epochs=args.epochs or train_cfg.get("num_epochs", 4),
        per_device_train_batch_size=args.batch_size or train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=args.lr or train_cfg.get("learning_rate", 5e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 5),
        optim=train_cfg.get("optimizer", "adamw_8bit"),
        seed=train_cfg.get("seed", 3407),
        lora_r=train_cfg.get("lora_r", 32),
        lora_alpha=train_cfg.get("lora_alpha", 64),
        lora_dropout=train_cfg.get("lora_dropout", 0.1),
    )

    from agenticvlm.data.preprocessing import prepare_router_training_data

    aug_cfg = cfg.get("augmentation", {})
    do_augment = args.augment or aug_cfg.get("enabled", False)

    logger.info("Preparing training data from %s", args.train_data)
    conversations = prepare_router_training_data(
        csv_path=args.train_data,
        image_dir=args.image_dir,
        augment=do_augment,
        n_text_variants=args.n_text_variants,
        n_image_variants=args.n_image_variants,
        seed=train_cfg.get("seed", 3407),
    )

    from datasets import Dataset

    train_dataset = Dataset.from_list(conversations)
    logger.info("Training dataset: %d samples", len(train_dataset))

    trainer = RouterTrainer(config)
    trainer.setup()
    trainer.train(train_dataset)

    output_dir = Path(config.output_dir)
    trainer.save(output_dir / "adapter")

    if args.save_merged:
        trainer.save_merged(output_dir / "merged")

    logger.info("Training complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
