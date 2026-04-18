from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess dataset and annotate biomedical entities.")
    # 数据来源：二选一
    # 1) HuggingFace 数据集（dataset_name / dataset_config）
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--test_split", default="test")
    # 2) 本地 json/jsonl 文件（train_file / eval_file）
    parser.add_argument("--train_file", default="")
    parser.add_argument("--eval_file", default="")
    parser.add_argument("--test_file", default="")
    # 监督字段列名：输入文本 + 目标摘要
    parser.add_argument("--text_column", default="article")
    parser.add_argument("--summary_column", default="abstract")
    # 预处理新增字段：实体串，默认列名 entity_text
    parser.add_argument("--entity_column", default="entity_text")
    parser.add_argument("--ner_model", default="en_ner_bionlp13cg_md")
    parser.add_argument("--max_entities", type=int, default=64)
    parser.add_argument("--output_train", default="data/samples/train_ner.jsonl")
    parser.add_argument("--output_eval", default="data/samples/dev_ner.jsonl")
    parser.add_argument("--output_test", default="data/samples/test_ner.jsonl")
    parser.add_argument(
        "--annotate_eval",
        action="store_true",
        help="Also annotate entities for eval split. Default is False (train only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_data(args: argparse.Namespace) -> DatasetDict:
    # 路径 A：从本地文件读取。
    if args.train_file:
        data_files: dict[str, str] = {"train": args.train_file}
        if args.eval_file:
            data_files["validation"] = args.eval_file
        dataset = load_dataset("json", data_files=data_files)
        if "validation" not in dataset:
            # 未提供验证集时，从 train 按 9:1 划分 train/validation。
            split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
            return DatasetDict(train=split["train"], validation=split["test"])#type: ignore
        return DatasetDict(train=dataset["train"], validation=dataset["validation"])#type: ignore

    if not args.dataset_name:
        raise ValueError("Provide either --train_file or --dataset_name.")

    # 路径 B：从 HF 数据集读取指定 split。
    dataset = load_dataset(args.dataset_name, args.dataset_config or None)
    return DatasetDict(
        train=dataset[args.train_split],
        validation=dataset[args.eval_split],
    ) #type: ignore 


def load_ner(model_name: str) -> Any:
    try:
        import spacy
    except Exception as exc:
        raise RuntimeError("spaCy is required for entity preprocessing.") from exc
    try:
        return spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load NER model '{model_name}'. Install model first."
        ) from exc


def extract_entities(text: str, nlp: Any, max_entities: int) -> str:
    # 对单条文本做 NER，按“小写归一”去重，并截断最大实体数。
    doc = nlp(text)
    dedup: list[str] = []
    seen: set[str] = set()
    for ent in doc.ents:
        value = ent.text.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(value)
        if len(dedup) >= max_entities:
            break
    return " ; ".join(dedup)


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    # 统一写出 JSONL（每行一个样本），便于下游 HF json loader 直接读取。
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ds = load_data(args)
    nlp = load_ner(args.ner_model)

    text_col = args.text_column
    summary_col = args.summary_column
    entity_col = args.entity_column

    def annotate_split(split_name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in ds[split_name]:
            text = str(item.get(text_col, ""))#type: ignore
            summary = str(item.get(summary_col, ""))#type: ignore
            if split_name == "train" or args.annotate_eval:
                # 训练集默认打实体；验证集仅在 annotate_eval=True 时打实体。
                entities = extract_entities(text=text, nlp=nlp, max_entities=args.max_entities)
                rows.append(
                    {
                        text_col: text,
                        summary_col: summary,
                        entity_col: entities,
                    }
                )
            else:
                # 默认不改动验证/测试分布：仅保留原始监督字段。
                rows.append(
                    {
                        text_col: text,
                        summary_col: summary,
                    }
                )
        return rows

    # 生成两个输出文件：
    # - train：默认包含 entity_column
    # - validation：默认不包含 entity_column（除非 --annotate_eval）
    train_rows = annotate_split("train")
    eval_rows = annotate_split("validation")
    write_jsonl(train_rows, args.output_train)
    write_jsonl(eval_rows, args.output_eval)

    print("Preprocessing completed.")
    print(f"Train output: {args.output_train} (rows={len(train_rows)})")
    print(f"Eval output: {args.output_eval} (rows={len(eval_rows)})")


if __name__ == "__main__":
    main()
