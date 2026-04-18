from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA + MI fine-tuning for summarization.")

    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--output_dir", default="data/outputs/ft_lora_mi")

    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--train_split", default="train[:2000]")
    parser.add_argument("--eval_split", default="validation[:200]")
    parser.add_argument("--train_file", default="")
    parser.add_argument("--eval_file", default="")

    parser.add_argument("--text_column", default="article")
    parser.add_argument("--summary_column", default="abstract")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lambda_mi", type=float, default=0.1, help="Weight for MI surrogate loss.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", default="q,v")
    parser.add_argument("--use_entity_prior", action="store_true", help="Use biomedical entity prior for MI loss.")
    parser.add_argument(
        "--entity_column",
        default="entity_text",
        help="Preprocessed entity text column, generated in preprocessing stage.",
    )
    parser.add_argument("--max_entity_length", type=int, default=128)

    return parser


def load_data(args: argparse.Namespace) -> DatasetDict:
    # 与预处理脚本保持一致：
    # - 若给 train_file/eval_file 则按文件加载
    # - 若仅给 train_file 则按 9:1 自动切分验证集
    # - 否则按 HF dataset + split 加载
    if args.train_file:
        data_files: dict[str, str] = {"train": args.train_file}
        if args.eval_file:
            data_files["validation"] = args.eval_file
        dataset = load_dataset("json", data_files=data_files)
        if "validation" not in dataset:
            split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
            return DatasetDict(train=split["train"], validation=split["test"])
        return DatasetDict(train=dataset["train"], validation=dataset["validation"])

    if not args.dataset_name:
        raise ValueError("Provide either --train_file or --dataset_name.")

    dataset = load_dataset(args.dataset_name, args.dataset_config or None)
    return DatasetDict(
        train=dataset[args.train_split],
        validation=dataset[args.eval_split],
    )


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    expanded = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * expanded).sum(dim=1)
    denom = expanded.sum(dim=1).clamp_min(1.0)
    return summed / denom


class LoRAMITrainer(Seq2SeqTrainer):
    """L_total = L_mle + lambda * L_mi (MI surrogate via cosine alignment)."""

    def __init__(
        self,
        *args: Any,
        lambda_mi: float = 0.1,
        use_entity_prior: bool = False,
        tokenizer: AutoTokenizer,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.lambda_mi = float(lambda_mi)
        self.use_entity_prior = bool(use_entity_prior)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        model_inputs = dict(inputs)
        entity_ids = model_inputs.pop("entity_input_ids", None)
        entity_mask = model_inputs.pop("entity_attention_mask", None)

        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
        mle_loss = outputs.loss
        if mle_loss is None:
            raise RuntimeError("Model did not return loss; make sure labels are included.")

        encoder_hidden = outputs.encoder_last_hidden_state
        if encoder_hidden is None:
            raise RuntimeError("encoder_last_hidden_state is required for MI loss.")

        source_repr = masked_mean(encoder_hidden, model_inputs.get("attention_mask"))
        if self.use_entity_prior and entity_ids is not None:
            entity_emb = model.get_input_embeddings()(entity_ids)
            source_repr = masked_mean(entity_emb, entity_mask)

        labels = model_inputs["labels"]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        label_ids = labels.masked_fill(labels == -100, pad_id)
        label_emb = model.get_input_embeddings()(label_ids)
        label_mask = (labels != -100).to(label_emb.dtype).unsqueeze(-1)
        target_repr = (label_emb * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp_min(1.0)

        mi_loss = 1.0 - F.cosine_similarity(source_repr, target_repr, dim=-1).mean()
        total_loss = mle_loss + self.lambda_mi * mi_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_ds = load_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    text_col = args.text_column
    summary_col = args.summary_column
    entity_col = args.entity_column
    if args.use_entity_prior:
        # 只要求 train split 含实体列，不强制 validation 含实体列。
        train_cols = set(raw_ds["train"].column_names)
        if entity_col not in train_cols:
            raise ValueError(
                f"--use_entity_prior requires column '{entity_col}' in train split. "
                "Please run data preprocessing first."
            )

    def preprocess_train(batch: dict[str, list[Any]]) -> dict[str, Any]:
        # 训练阶段：常规输入 + 摘要标签。
        inputs = [str(x) for x in batch[text_col]]
        targets = [str(x) for x in batch[summary_col]]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        if args.use_entity_prior:
            # 仅训练阶段注入实体先验，参与 MI 对齐损失。
            entity_texts = [str(x) for x in batch[entity_col]]
            if not any(text.strip() for text in entity_texts):
                raise ValueError(
                    f"Column '{entity_col}' is empty in current batch. "
                    "Please verify preprocessing output."
                )
            entity_inputs = tokenizer(
                entity_texts,
                max_length=args.max_entity_length,
                truncation=True,
                padding="max_length",
            )
            model_inputs["entity_input_ids"] = entity_inputs["input_ids"]
            model_inputs["entity_attention_mask"] = entity_inputs["attention_mask"]
        return model_inputs

    def preprocess_eval(batch: dict[str, list[Any]]) -> dict[str, Any]:
        # 验证阶段：只走原始文本->摘要监督，不读取实体列。
        inputs = [str(x) for x in batch[text_col]]
        targets = [str(x) for x in batch[summary_col]]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        # 验证/测试阶段不使用实体先验，避免影响评估分布。
        return model_inputs
    tokenized_train = raw_ds["train"].map(
        preprocess_train,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Tokenizing train",
    )
    tokenized_eval = raw_ds["validation"].map(
        preprocess_eval,
        batched=True,
        remove_columns=raw_ds["validation"].column_names,
        desc="Tokenizing validation",
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(float(v), 4) for k, v in result.items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=args.seed,
    )

    trainer = LoRAMITrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
        lambda_mi=args.lambda_mi,
        use_entity_prior=args.use_entity_prior,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    print("=== Eval Metrics ===")
    for key, value in metrics.items():
        if key.startswith("eval_"):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
