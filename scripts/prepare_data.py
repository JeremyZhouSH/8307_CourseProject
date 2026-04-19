"""
从 HuggingFace 下载 ccdv/pubmed-summarization 到本地 data/pubmed/，
并自动生成带实体标注的训练/验证/测试集。

用法：
    python scripts/prepare_data.py \
        --num_train 2000 \
        --num_val 200 \
        --num_test 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset


# 函数作用：构建命令行参数解析器并定义可配置项。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local PubMed dataset.")
    parser.add_argument("--dataset_name", default="ccdv/pubmed-summarization")
    parser.add_argument("--dataset_config", default="document")
    parser.add_argument("--num_train", type=int, default=16000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--output_dir", default="data/pubmed")
    parser.add_argument("--ner_model", default="en_ner_bionlp13cg_md")
    return parser


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def save_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 函数作用：程序入口，串联参数解析与主执行流程。
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"

    print(f"Loading {args.dataset_name} from HuggingFace...")
    ds = load_dataset(args.dataset_name, args.dataset_config or None)

    # 1. 保存原始数据
    train_rows = [
        {"article": str(item["article"]), "abstract": str(item["abstract"])}
        for item in ds["train"].select(range(min(args.num_train, len(ds["train"]))))
    ]
    val_rows = [
        {"article": str(item["article"]), "abstract": str(item["abstract"])}
        for item in ds["validation"].select(range(min(args.num_val, len(ds["validation"]))))
    ]
    test_rows = [
        {"article": str(item["article"]), "abstract": str(item["abstract"])}
        for item in ds["test"].select(range(min(args.num_test, len(ds["test"]))))
    ]

    save_jsonl(train_rows, raw_dir / "train.jsonl")
    save_jsonl(val_rows, raw_dir / "val.jsonl")
    save_jsonl(test_rows, raw_dir / "test.jsonl")

    print(f"Raw data saved to {raw_dir}")
    print(f"  train: {len(train_rows)}  val: {len(val_rows)}  test: {len(test_rows)}")

    # 2. 运行实体预处理（只标注训练集，验证/测试不标注）
    print("\nRunning entity preprocessing...")
    cmd = [
        sys.executable,
        "data/preprocess_entities.py",
        "--train_file", str(raw_dir / "train.jsonl"),
        "--eval_file", str(raw_dir / "val.jsonl"),
        "--ner_model", args.ner_model,
        "--output_train", str(output_dir / "train_ner.jsonl"),
        "--output_eval", str(output_dir / "val_ner.jsonl"),
    ]
    subprocess.run(cmd, check=True)

    # 3. 测试集单独保存一份无实体版本（保持评估分布纯净）
    save_jsonl(test_rows, output_dir / "test.jsonl")

    print(f"\nAll data prepared in {output_dir}/")
    print("  train_ner.jsonl  -> 训练集（含实体标注）")
    print("  val_ner.jsonl    -> 验证集（无实体标注）")
    print("  test.jsonl       -> 测试集（无实体标注，用于最终评估）")


if __name__ == "__main__":
    main()
