from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


# 函数作用：构建命令行参数解析器并定义可配置项。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess dataset and annotate biomedical entities.")
    # 数据来源：二选一
    # 1) HuggingFace 数据集（dataset_name / dataset_config）
    parser.add_argument("--dataset_name", default="ccdv/pubmed-summarization")
    parser.add_argument("--dataset_config", default="document")
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
    parser.add_argument("--entity_types_column", default="entity_types")
    parser.add_argument("--entity_spans_column", default="entity_spans")
    parser.add_argument("--summary_entity_column", default="summary_entities")
    parser.add_argument("--summary_entity_types_column", default="summary_entity_types")
    parser.add_argument("--summary_entity_spans_column", default="summary_entity_spans")
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


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def load_data(args: argparse.Namespace) -> DatasetDict:
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _resolve_hf_split(dataset: DatasetDict, split_name_or_expr: str) -> Dataset:
        if split_name_or_expr in dataset:
            return dataset[split_name_or_expr]
        return load_dataset(  # type: ignore[return-value]
            args.dataset_name,
            args.dataset_config or None,
            split=split_name_or_expr,
        )

    # 路径 A：从本地文件读取。
    if args.train_file:
        data_files: dict[str, str] = {"train": args.train_file}
        if args.eval_file:
            data_files["validation"] = args.eval_file
        if args.test_file:
            data_files["test"] = args.test_file
        dataset = load_dataset("json", data_files=data_files)

        result = DatasetDict()
        if "validation" not in dataset:
            # 未提供验证集时，从 train 按 9:1 划分 train/validation。
            split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
            result["train"] = split["train"]
            result["validation"] = split["test"]
        else:
            result["train"] = dataset["train"]
            result["validation"] = dataset["validation"]

        if "test" in dataset:
            result["test"] = dataset["test"]
        return result

    # 路径 B：从 HF 数据集读取指定 split。
    dataset = load_dataset(args.dataset_name, args.dataset_config or None)
    result = DatasetDict(
        train=_resolve_hf_split(dataset, args.train_split),
        validation=_resolve_hf_split(dataset, args.eval_split),
    )#type: ignore  
    if args.test_split:
        try:
            result["test"] = _resolve_hf_split(dataset, args.test_split)
        except Exception:
            # Some datasets have no test split; keep train/validation only.
            pass
    return result


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
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


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def extract_entities(text: str, nlp: Any, max_entities: int) -> dict[str, Any]:
    """
    Extract entities with type and character-span information.

    Returns
    -------
    dict with keys:
        - entity_text: "; " joined entity texts
        - entity_types: "; " joined entity type labels
        - entity_spans: JSON string of [[start, end], ...]
    """
    doc = nlp(text)
    texts: list[str] = []
    types: list[str] = []
    spans: list[list[int]] = []
    seen: set[str] = set()
    for ent in doc.ents:
        value = ent.text.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        texts.append(value)
        types.append(ent.label_)
        spans.append([int(ent.start_char), int(ent.end_char)])
        if len(texts) >= max_entities:
            break
    return {
        "entity_text": " ; ".join(texts),
        "entity_types": " ; ".join(types),
        "entity_spans": json.dumps(spans),
    }


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    # 统一写出 JSONL（每行一个样本），便于下游 HF json loader 直接读取。
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# 函数作用：程序入口，串联参数解析与主执行流程。
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ds = load_data(args)
    nlp = load_ner(args.ner_model)

    text_col = args.text_column
    summary_col = args.summary_column
    entity_col = args.entity_column
    entity_types_col = args.entity_types_column
    entity_spans_col = args.entity_spans_column
    summary_entity_col = args.summary_entity_column
    summary_entity_types_col = args.summary_entity_types_column
    summary_entity_spans_col = args.summary_entity_spans_column

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def annotate_split(split_name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in ds[split_name]:
            text = str(item.get(text_col, ""))  # type: ignore
            summary = str(item.get(summary_col, ""))  # type: ignore
            if split_name == "train" or args.annotate_eval:
                # Annotate both article and abstract (summary) for train.
                src_ents = extract_entities(text=text, nlp=nlp, max_entities=args.max_entities)
                sum_ents = extract_entities(text=summary, nlp=nlp, max_entities=args.max_entities)
                rows.append(
                    {
                        text_col: text,
                        summary_col: summary,
                        entity_col: src_ents["entity_text"],
                        entity_types_col: src_ents["entity_types"],
                        entity_spans_col: src_ents["entity_spans"],
                        summary_entity_col: sum_ents["entity_text"],
                        summary_entity_types_col: sum_ents["entity_types"],
                        summary_entity_spans_col: sum_ents["entity_spans"],
                    }
                )
            else:
                # Eval/test: keep only original supervision fields to avoid
                # leaking entity priors into the evaluation distribution.
                rows.append(
                    {
                        text_col: text,
                        summary_col: summary,
                    }
                )
        return rows

    # train：默认包含实体特征
    # validation/test：默认只保留监督字段（除非 --annotate_eval）
    train_rows = annotate_split("train")
    eval_rows = annotate_split("validation")
    write_jsonl(train_rows, args.output_train)
    write_jsonl(eval_rows, args.output_eval)
    test_rows: list[dict[str, Any]] = []
    if "test" in ds:
        test_rows = annotate_split("test")
        write_jsonl(test_rows, args.output_test)

    print("Preprocessing completed.")
    print(f"Train output: {args.output_train} (rows={len(train_rows)})")
    print(f"Eval output: {args.output_eval} (rows={len(eval_rows)})")
    if test_rows:
        print(f"Test output: {args.output_test} (rows={len(test_rows)})")


if __name__ == "__main__":
    main()
