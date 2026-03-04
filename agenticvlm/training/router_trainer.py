"""Router trainer — LoRA finetuning of Qwen2.5-VL for question routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

logger = logging.getLogger(__name__)


@dataclass
class RouterTrainingConfig:
    """Configuration for router LoRA training.

    Mirrors the hyperparameters from the original notebook:
    ``fork-of-fork-of-train-allversion.ipynb``.
    """

    base_model: str = "unsloth/Qwen2.5-VL-7B-Instruct"
    load_in_4bit: bool = True

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    use_rslora: bool = False
    bias: str = "none"

    output_dir: str = "outputs/router"
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 5
    optim: str = "adamw_8bit"
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 1
    seed: int = 3407
    max_seq_length: int = 1024
    dataset_num_proc: int = 4

    save_strategy: str = "epoch"
    save_total_limit: int = 2

    remove_unused_columns: bool = False
    dataset_text_field: str = ""
    dataset_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"skip_prepare_dataset": True}
    )


class RouterTrainer:
    """LoRA-based router trainer using Unsloth + SFTTrainer.

    Example usage::

        config = RouterTrainingConfig(
            base_model="path/to/qwen2.5-vl-7b",
            output_dir="outputs/router_v1",
        )
        trainer = RouterTrainer(config)
        trainer.train(train_dataset)
        trainer.save("outputs/router_v1_final")

    Args:
        config: Training configuration.
    """

    def __init__(self, config: RouterTrainingConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self._trainer = None

    def setup(self) -> None:
        """Load the base model and apply LoRA adapters."""
        from unsloth import FastVisionModel

        logger.info("Loading base model: %s", self.config.base_model)

        model, tokenizer = FastVisionModel.from_pretrained(
            self.config.base_model,
            load_in_4bit=self.config.load_in_4bit,
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            random_state=self.config.seed,
            use_rslora=self.config.use_rslora,
            loftq_config=None,
        )

        self.model = model
        self.tokenizer = tokenizer
        logger.info("Model loaded and LoRA adapters applied.")

    def train(self, train_dataset: Any) -> Any:
        """Run the training loop.

        Args:
            train_dataset: A HuggingFace ``Dataset`` with conversation-format
                samples (as produced by ``prepare_router_training_data``).

        Returns:
            The training result from ``SFTTrainer.train()``.
        """
        if self.model is None:
            self.setup()

        from trl import SFTConfig, SFTTrainer
        from unsloth import UnslothVisionDataCollator

        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            optim=self.config.optim,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            seed=self.config.seed,
            max_seq_length=self.config.max_seq_length,
            dataset_num_proc=self.config.dataset_num_proc,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=self.config.remove_unused_columns,
            dataset_text_field=self.config.dataset_text_field,
            dataset_kwargs=self.config.dataset_kwargs,
        )

        self._trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            args=sft_config,
        )

        logger.info("Starting router training...")
        result = self._trainer.train()
        logger.info("Training complete.")
        return result

    def save(self, output_path: str | Path) -> None:
        """Save the trained LoRA adapter.

        Args:
            output_path: Directory to save the adapter weights.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Run training first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        logger.info("Router adapter saved to %s", output_path)

    def save_merged(self, output_path: str | Path, quantization: Optional[str] = None) -> None:
        """Save a fully merged model (base + LoRA).

        Args:
            output_path: Directory for the merged model.
            quantization: Optional quantization method (e.g. ``"q4_k_m"``).
        """
        if self.model is None:
            raise RuntimeError("No model to save. Run training first.")

        output_path = Path(output_path)

        if quantization:
            self.model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method=f"merged_{quantization}",
            )
        else:
            self.model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method="merged_16bit",
            )
        logger.info("Merged model saved to %s", output_path)


def shrink_tokenizer_vocab(
    tokenizer: Any,
    required_tokens: Set[str],
) -> tuple[Any, Dict[int, int]]:
    """Shrink a tokenizer's vocabulary to only the required tokens.

    Args:
        tokenizer: HuggingFace tokenizer.
        required_tokens: Set of token strings to keep.

    Returns:
        Tuple of (modified tokenizer, old_id → new_id mapping).
    """
    required_ids: Set[int] = set()
    for token in required_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        required_ids.update(ids)

    special_ids = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)
    required_ids.update(special_ids)

    for char in "0123456789,/_ \n\t":
        ids = tokenizer.encode(char, add_special_tokens=False)
        required_ids.update(ids)

    sorted_old_ids = sorted(required_ids)
    old_to_new = {old: new for new, old in enumerate(sorted_old_ids)}

    logger.info(
        "Vocabulary shrunk: %d → %d tokens",
        len(tokenizer),
        len(sorted_old_ids),
    )
    return tokenizer, old_to_new


def shrink_model_embeddings(
    model: Any,
    old_to_new: Dict[int, int],
    device: Optional[str] = None,
) -> None:
    """Shrink model embedding and LM head layers to match reduced vocabulary.

    Args:
        model: HuggingFace model with ``model.embed_tokens`` and ``lm_head``.
        old_to_new: Mapping from original token IDs to new compact IDs.
        device: Target device for the new tensors.
    """
    new_vocab_size = len(old_to_new)
    sorted_old_ids = sorted(old_to_new.keys())

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        old_embed = model.model.embed_tokens.weight.data
        embed_dim = old_embed.shape[1]
        new_embed = torch.zeros(
            new_vocab_size, embed_dim, dtype=old_embed.dtype
        )
        for new_id, old_id in enumerate(sorted_old_ids):
            if old_id < old_embed.shape[0]:
                new_embed[new_id] = old_embed[old_id]
        if device:
            new_embed = new_embed.to(device)
        model.model.embed_tokens.weight = torch.nn.Parameter(new_embed)
        model.model.embed_tokens.num_embeddings = new_vocab_size

    if hasattr(model, "lm_head"):
        old_head = model.lm_head.weight.data
        head_dim = old_head.shape[1]
        new_head = torch.zeros(
            new_vocab_size, head_dim, dtype=old_head.dtype
        )
        for new_id, old_id in enumerate(sorted_old_ids):
            if old_id < old_head.shape[0]:
                new_head[new_id] = old_head[old_id]
        if device:
            new_head = new_head.to(device)
        model.lm_head.weight = torch.nn.Parameter(new_head)
        model.lm_head.out_features = new_vocab_size

    if hasattr(model, "config"):
        model.config.vocab_size = new_vocab_size

    logger.info("Model embeddings shrunk to %d tokens", new_vocab_size)


def shrink_embeddings(
    model: Any,
    tokenizer: Any,
    label_tokens: Optional[Set[str]] = None,
) -> tuple[Any, Any]:
    """Combined vocabulary + embedding shrinking for router training.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        label_tokens: Token strings to keep. Defaults to the 9 routing
            labels used in the paper.

    Returns:
        Tuple of (modified model, modified tokenizer).
    """
    if label_tokens is None:
        label_tokens = {
            "handwritten", "layout", "form", "table/list", "table", "list",
            "free_text", "Yes/No", "yes", "no", "figure/diagram", "figure",
            "diagram", "image/photo", "image", "photo", "others",
        }

    tokenizer, old_to_new = shrink_tokenizer_vocab(tokenizer, label_tokens)
    shrink_model_embeddings(model, old_to_new)
    return model, tokenizer
