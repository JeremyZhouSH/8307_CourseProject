from __future__ import annotations
import os


import argparse
import json
from pathlib import Path
from typing import Any

import spacy
from datasets import DatasetDict, load_dataset
from tqdm import tqdm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess summarization data with biomedical NER annotations."
    )

    # Data source: either local files or HF dataset
    parser.add_argument("--train_file", default="", help="Local train json/jsonl file")
    parser.add_argument("--eval_file", default="", help="Local eval json/jsonl file")
    parser.add_argument("--dataset_name", default="", help="HF dataset name")
    parser.add_argument("--dataset_config", default="", help="HF dataset config")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="validation")

    # Column mapping
    parser.add_argument("--text_column", default="article")
    parser.add_argument("--summary_column", default="abstract")

    # Output fields expected by train_lora_mi_v2.py
    parser.add_argument("--entity_column", default="entity_text")
    parser.add_argument("--entity_types_column", default="entity_types")
    parser.add_argument("--entity_spans_column", default="entity_spans")
    parser.add_argument("--summary_entity_column", default="summary_entities")
    parser.add_argument("--summary_entity_types_column", default="summary_entity_types")
    parser.add_argument("--summary_entity_spans_column", default="summary_entity_spans")

    parser.add_argument("--ner_model", default="en_ner_bionlp13cg_md")
    parser.add_argument("--max_entities", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument(
        "--deduplicate_case_insensitive",
        action="store_true",
        help="Deduplicate entities by lowercase text + label.",
    )

    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_eval", required=True)

    return parser


def load_data(args: argparse.Namespace) -> DatasetDict:
    if args.train_file:
        data_files: dict[str, str] = {"train": args.train_file}
        if args.eval_file:
            data_files["validation"] = args.eval_file
        dataset = load_dataset("json", data_files=data_files)
        if "validation" not in dataset:
            split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            return DatasetDict(train=split["train"], validation=split["test"])
        return DatasetDict(train=dataset["train"], validation=dataset["validation"])

    if not args.dataset_name:
        raise ValueError("Provide either --train_file/--eval_file or --dataset_name.")

    train_ds = load_dataset(
        args.dataset_name,
        args.dataset_config or None,
        split=args.train_split,
    )
    eval_ds = load_dataset(
        args.dataset_name,
        args.dataset_config or None,
        split=args.eval_split,
    )
    return DatasetDict(train=train_ds, validation=eval_ds)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value)


def extract_entities_from_doc(
    doc: Any,
    max_entities: int,
    deduplicate_case_insensitive: bool,
) -> tuple[list[str], list[str], list[list[int]]]:
    texts: list[str] = []
    types: list[str] = []
    spans: list[list[int]] = []
    seen: set[str] = set()

    for ent in doc.ents:
        text = ent.text.strip()
        if not text:
            continue

        if deduplicate_case_insensitive:
            key = f"{text.lower()}::{ent.label_}"
        else:
            key = f"{ent.start_char}:{ent.end_char}:{ent.label_}"

        if key in seen:
            continue
        seen.add(key)

        texts.append(text)
        types.append(str(ent.label_))
        spans.append([int(ent.start_char), int(ent.end_char)])

        if len(texts) >= max_entities:
            break

    return texts, types, spans


def annotate_split(
    split,
    nlp: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    text_col = args.text_column
    summary_col = args.summary_column

    entity_col = args.entity_column
    entity_types_col = args.entity_types_column
    entity_spans_col = args.entity_spans_column
    summary_entity_col = args.summary_entity_column
    summary_entity_types_col = args.summary_entity_types_column
    summary_entity_spans_col = args.summary_entity_spans_column

    # Keep all original fields, then append NER fields.
    examples = [dict(row) for row in split]
    src_texts = [normalize_text(row.get(text_col, "")) for row in examples]
    sum_texts = [normalize_text(row.get(summary_col, "")) for row in examples]

    src_docs = nlp.pipe(src_texts, batch_size=args.batch_size, n_process=args.num_proc)
    sum_docs = nlp.pipe(sum_texts, batch_size=args.batch_size, n_process=args.num_proc)

    rows: list[dict[str, Any]] = []
    non_empty_summary_types = 0
    summary_types_ge2 = 0

    for example, src_doc, sum_doc in tqdm(
        zip(examples, src_docs, sum_docs),
        total=len(examples),
        desc="Annotating",
    ):
        src_entities, src_types, src_spans = extract_entities_from_doc(
            src_doc,
            max_entities=args.max_entities,
            deduplicate_case_insensitive=args.deduplicate_case_insensitive,
        )
        sum_entities, sum_types, sum_spans = extract_entities_from_doc(
            sum_doc,
            max_entities=args.max_entities,
            deduplicate_case_insensitive=args.deduplicate_case_insensitive,
        )

        if sum_types:
            non_empty_summary_types += 1
        if len(set(sum_types)) >= 2:
            summary_types_ge2 += 1

        example[text_col] = normalize_text(example.get(text_col, ""))
        example[summary_col] = normalize_text(example.get(summary_col, ""))

        example[entity_col] = "; ".join(src_entities)
        example[entity_types_col] = "; ".join(src_types)
        example[entity_spans_col] = json.dumps(src_spans, ensure_ascii=False)

        example[summary_entity_col] = "; ".join(sum_entities)
        example[summary_entity_types_col] = "; ".join(sum_types)
        example[summary_entity_spans_col] = json.dumps(sum_spans, ensure_ascii=False)

        rows.append(example)

    denom = max(len(examples), 1)
    stats = {
        "samples": float(len(examples)),
        "summary_types_non_empty_ratio": non_empty_summary_types / denom,
        "summary_types_unique_ge2_ratio": summary_types_ge2 / denom,
    }
    return rows, stats


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        nlp = spacy.load(args.ner_model)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load spaCy/scispaCy model '{args.ner_model}'. "
            "Install the model first."
        ) from exc

    ds = load_data(args)

    train_rows, train_stats = annotate_split(ds["train"], nlp, args)
    eval_rows, eval_stats = annotate_split(ds["validation"], nlp, args)

    write_jsonl(args.output_train, train_rows)
    write_jsonl(args.output_eval, eval_rows)

    print("Preprocessing finished.")
    print(
        f"Train stats: samples={int(train_stats['samples'])}, "
        f"summary_types_non_empty_ratio={train_stats['summary_types_non_empty_ratio']:.2%}, "
        f"summary_types_unique_ge2_ratio={train_stats['summary_types_unique_ge2_ratio']:.2%}"
    )
    print(
        f"Eval stats: samples={int(eval_stats['samples'])}, "
        f"summary_types_non_empty_ratio={eval_stats['summary_types_non_empty_ratio']:.2%}, "
        f"summary_types_unique_ge2_ratio={eval_stats['summary_types_unique_ge2_ratio']:.2%}"
    )


if __name__ == "__main__":
    main()
