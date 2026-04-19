from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=str(_ENV_PATH))

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# 函数作用：构建命令行参数解析器并定义可配置项。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq model for summarization.")

    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--output_dir", default="data/outputs/ft_model")

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
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def load_data(args: argparse.Namespace) -> DatasetDict:
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

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config or None,
    )
    return DatasetDict(
        train=dataset[args.train_split],
        validation=dataset[args.eval_split],
    )


# 函数作用：程序入口，串联参数解析与主执行流程。
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_ds = load_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    text_col = args.text_column
    summary_col = args.summary_column

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
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
        return model_inputs

    tokenized = raw_ds.map(
        preprocess,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Tokenizing",
    )

    rouge = evaluate.load("rouge")

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
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
