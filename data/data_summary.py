from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess dataset and annotate biomedical entities.")
    # 数据集配置
    parser.add_argument("--dataset_name", default="ccdv/pubmed-summarization")
    parser.add_argument("--dataset_config", default="document")
    parser.add_argument("--hf_token", default="", help="HuggingFace token for dataset access")
    
    # 监督字段列名
    parser.add_argument("--text_column", default="article")
    parser.add_argument("--summary_column", default="abstract")
    parser.add_argument("--entity_column", default="entity_text")
    
    # NER 与多进程批处理配置
    parser.add_argument("--ner_model", default="en_ner_bionlp13cg_md")
    parser.add_argument("--max_entities", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for SpaCy pipe")
    parser.add_argument("--n_process", type=int, default=4, help="Number of CPU cores for SpaCy multiprocessing")
    parser.add_argument("--map_batch_size", type=int, default=1000, help="Batch size for HuggingFace dataset map")
    
    # 输出路径
    parser.add_argument("--output_train", default="data/samples/train_ner.jsonl")
    parser.add_argument("--output_eval", default="data/samples/validation_ner.jsonl")
    parser.add_argument("--output_test", default="data/samples/test_ner.jsonl")
    
    parser.add_argument(
        "--annotate_eval",
        action="store_true",
        help="Also annotate entities for eval and test splits. Default is False (train only).",
    )
    return parser


def load_ner(model_name: str) -> Any:
    try:
        import spacy
    except Exception as exc:
        raise RuntimeError("spaCy is required for entity preprocessing.") from exc
    try:
        # 禁用不需要的管道组件以提升推理速度
        return spacy.load(model_name, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load NER model '{model_name}'. Ensure it is installed via `python -m spacy download {model_name}`."
        ) from exc


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. 加载目标数据集 (包含 train, validation, test)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    ds_full = load_dataset(args.dataset_name, args.dataset_config, token=token)
    
    if not isinstance(ds_full, DatasetDict):
        raise TypeError("Expected a DatasetDict from load_dataset.")

    # 2. 加载 NER 模型
    nlp = load_ner(args.ner_model)

    text_col = args.text_column
    entity_col = args.entity_column
    max_entities = args.max_entities

    # 3. 构造批处理处理函数
    def process_batch(batch: dict[str, list[Any]], is_train: bool) -> dict[str, list[Any]]:
        # 如果不是训练集且未开启 --annotate_eval，则返回空实体列表
        if not is_train and not args.annotate_eval:
            batch[entity_col] = [""] * len(batch[text_col])
            return batch

        texts = [str(text) for text in batch[text_col]]
        # 截断optional
        texts = [text for text in texts]

        entities_batch = []
        # 使用 nlp.pipe 开启多进程和批处理
        docs = nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process)
        
        for doc in docs:
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
            entities_batch.append(" ; ".join(dedup))
            
        batch[entity_col] = entities_batch
        return batch

    # 4. 应用 map 函数进行并行处理
    print("Processing train split...")
    ds_train = ds_full["train"].map(
        lambda b: process_batch(b, is_train=True),
        batched=True,
        batch_size=args.map_batch_size,
        desc="Annotating train"
    )

    print("Processing validation split...")
    ds_val = ds_full["validation"].map(
        lambda b: process_batch(b, is_train=False),
        batched=True,
        batch_size=args.map_batch_size,
        desc="Annotating validation"
    )

    print("Processing test split...")
    ds_test = ds_full["test"].map(
        lambda b: process_batch(b, is_train=False),
        batched=True,
        batch_size=args.map_batch_size,
        desc="Annotating test"
    )

    # 5. 导出处理后的数据为 JSONL，以便 Hugging Face 直接 load_dataset("json", ...) 读取
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    
    ds_train.to_json(args.output_train, force_ascii=False)
    ds_val.to_json(args.output_eval, force_ascii=False)
    ds_test.to_json(args.output_test, force_ascii=False)

    print("\nPreprocessing completed.")
    print(f"Train output: {args.output_train} (rows={len(ds_train)})")
    print(f"Eval output:  {args.output_eval} (rows={len(ds_val)})")
    print(f"Test output:  {args.output_test} (rows={len(ds_test)})")


if __name__ == "__main__":
    main()