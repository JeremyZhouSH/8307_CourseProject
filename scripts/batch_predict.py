"""
批量推理：加载训练好的模型，在验证集/测试集上生成预测摘要。

用法：
    python scripts/batch_predict.py \
        --model_path data/outputs/ft_lora_mi \
        --input_file data/pubmed/val_ner.jsonl \
        --output_file data/outputs/predictions_val.jsonl \
        --max_samples 200

输入格式（JSONL）：每行含 "article" 和 "abstract" 字段
输出格式（JSONL）：每行含 "article", "prediction", "reference"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch prediction for summarization.")
    parser.add_argument("--model_path", required=True, help="训练好的模型路径")
    parser.add_argument("--input_file", required=True, help="输入 JSONL（含 article + abstract）")
    parser.add_argument("--output_file", required=True, help="输出 JSONL（含 prediction）")
    parser.add_argument("--text_column", default="article")
    parser.add_argument("--summary_column", default="abstract")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="0=全部")
    parser.add_argument("--device", default="auto")
    return parser


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_batch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_input_length: int,
    max_target_length: int,
    device: torch.device,
) -> list[str]:
    inputs = tokenizer(
        texts,
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = model.to(device)
    model.eval()

    print(f"Loading input from {args.input_file}...")
    rows = load_jsonl(args.input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating predictions for {len(rows)} samples...")
    with output_path.open("w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch_rows = rows[i : i + args.batch_size]
            texts = [str(row.get(args.text_column, "")) for row in batch_rows]
            refs = [str(row.get(args.summary_column, "")) for row in batch_rows]

            preds = generate_batch(
                model, tokenizer, texts,
                args.max_input_length, args.max_target_length, device
            )

            for text, pred, ref in zip(texts, preds, refs):
                out_f.write(
                    json.dumps(
                        {"article": text, "prediction": pred, "reference": ref},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
