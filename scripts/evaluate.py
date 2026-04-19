"""
综合评估脚本：支持 HuggingFace 数据集在线加载或本地 JSONL。

用法 A（在线加载参考数据）：
    python scripts/evaluate.py \
        --pred_file data/outputs/predictions.jsonl \
        --dataset_name ccdv/pubmed-summarization \
        --dataset_config document \
        --split validation \
        --max_samples 200 \
        --output_file data/outputs/eval_results.json

用法 B（本地参考文件）：
    python scripts/evaluate.py \
        --pred_file data/outputs/predictions.jsonl \
        --ref_file data/samples/refs.jsonl \
        --output_file data/outputs/eval_results.json

输入格式（predictions.jsonl）：
    每行 {"article": "...", "prediction": "生成的摘要", "reference": "参考摘要（可选）"}

若 predictions.jsonl 中已有 reference 字段，可省略 --dataset_name / --ref_file。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import spacy
from datasets import load_dataset

from src.verifier.faithfulness_checker import FaithfulnessChecker


# 函数作用：构建命令行参数解析器并定义可配置项。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate summarization outputs.")
    parser.add_argument("--pred_file", required=True, help="预测摘要 JSONL（每行含 prediction）")
    parser.add_argument("--ref_file", default="", help="本地参考摘要 JSONL（与 pred 一一对应）")
    parser.add_argument("--dataset_name", default="", help="HuggingFace 数据集名称（如 ccdv/pubmed-summarization）")
    parser.add_argument("--dataset_config", default="", help="数据集 config 名称（如 document）")
    parser.add_argument("--split", default="validation", help="数据集 split（默认 validation）")
    parser.add_argument("--max_samples", type=int, default=0, help="最大评估样本数，0=全部")
    parser.add_argument("--text_column", default="article", help="原文字段名")
    parser.add_argument("--summary_column", default="abstract", help="摘要字段名")
    parser.add_argument("--output_file", default="data/outputs/eval_results.json")
    parser.add_argument("--ner_model", default="en_ner_bionlp13cg_md")
    return parser


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def extract_entities(text: str, nlp: spacy.Language) -> list[tuple[str, str]]:
    """返回 [(entity_text_lower, entity_type), ...]"""
    doc = nlp(text)
    return [
        (ent.text.strip().lower(), ent.label_)
        for ent in doc.ents
        if ent.text.strip()
    ]


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def compute_entity_metrics(pred_text: str, ref_text: str, nlp: spacy.Language) -> dict[str, float]:
    pred_ents = extract_entities(pred_text, nlp)
    ref_ents = extract_entities(ref_text, nlp)

    if not ref_ents:
        return {"entity_recall": 0.0, "entity_precision": 0.0, "entity_f1": 0.0}

    pred_set = set(pred_ents)
    ref_set = set(ref_ents)
    overlap = pred_set & ref_set

    recall = len(overlap) / len(ref_set) if ref_set else 0.0
    precision = len(overlap) / len(pred_set) if pred_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "entity_recall": round(recall, 4),
        "entity_precision": round(precision, 4),
        "entity_f1": round(f1, 4),
    }


# 函数作用：程序入口，串联参数解析与主执行流程。
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pred_rows = load_jsonl(args.pred_file)
    predictions = [str(row.get("prediction", "")) for row in pred_rows]
    articles = [str(row.get("article", "")) for row in pred_rows]

    # --- 获取参考摘要 ---
    references: list[str] = []

    # 优先级1：predictions.jsonl 自带 reference
    if pred_rows and "reference" in pred_rows[0]:
        references = [str(row.get("reference", "")) for row in pred_rows]
    # 优先级2：本地 ref_file
    elif args.ref_file:
        ref_rows = load_jsonl(args.ref_file)
        references = [str(row.get(args.summary_column, row.get("abstract", ""))) for row in ref_rows]
    # 优先级3：在线 HuggingFace 数据集
    elif args.dataset_name:
        ds = load_dataset(args.dataset_name, args.dataset_config or None)
        split_data = ds[args.split]
        refs = [str(item[args.summary_column]) for item in split_data]
        references = refs[: len(predictions)]
    else:
        raise ValueError("请提供参考摘要：--ref_file、--dataset_name，或在 pred_file 中包含 reference 字段")

    n = min(len(predictions), len(references), len(articles))
    if args.max_samples > 0:
        n = min(n, args.max_samples)

    predictions = predictions[:n]
    references = references[:n]
    articles = articles[:n]

    if n == 0:
        raise ValueError("没有可用样本进行评估")

    print(f"Evaluating {n} samples...")

    # --- 加载评估工具 ---
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("sacrebleu")
    checker = FaithfulnessChecker()
    nlp = spacy.load(args.ner_model)

    # 1. ROUGE
    rouge_results = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )

    # 2. BERTScore
    bs_results = bertscore.compute(
        predictions=predictions, references=references, lang="en", device="cpu"
    )

    # 3. BLEU
    bleu_results = bleu.compute(
        predictions=predictions, references=[[ref] for ref in references]
    )

    # 4. Faithfulness（摘要 vs 原文的词汇重叠率）
    faithfulness_scores = []
    for pred, article in zip(predictions, articles):
        if not article:
            faithfulness_scores.append(0.0)
            continue
        try:
            ver = checker.check(pred, {}, article)
            faithfulness_scores.append(ver.get("faithfulness_score", 0.0))
        except Exception:
            faithfulness_scores.append(0.0)

    # 5. 实体指标
    entity_recalls = []
    entity_precisions = []
    entity_f1s = []
    for pred, ref in zip(predictions, references):
        metrics = compute_entity_metrics(pred, ref, nlp)
        entity_recalls.append(metrics["entity_recall"])
        entity_precisions.append(metrics["entity_precision"])
        entity_f1s.append(metrics["entity_f1"])

    results = {
        "rouge1": round(float(rouge_results["rouge1"]), 4),
        "rouge2": round(float(rouge_results["rouge2"]), 4),
        "rougeL": round(float(rouge_results["rougeL"]), 4),
        "bleu": round(float(bleu_results["score"]), 4),
        "bertscore_precision": round(float(np.mean(bs_results["precision"])), 4),
        "bertscore_recall": round(float(np.mean(bs_results["recall"])), 4),
        "bertscore_f1": round(float(np.mean(bs_results["f1"])), 4),
        "faithfulness": round(float(np.mean(faithfulness_scores)), 4),
        "entity_recall": round(float(np.mean(entity_recalls)), 4),
        "entity_precision": round(float(np.mean(entity_precisions)), 4),
        "entity_f1": round(float(np.mean(entity_f1s)), 4),
        "num_samples": n,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
