from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import evaluate
import json
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from finetune.mi_layers import NodeLayerLoss, LinkLayerLoss, NetworkLayerLoss


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
    parser.add_argument("--entity_types_column", default="entity_types")
    parser.add_argument("--entity_spans_column", default="entity_spans")
    parser.add_argument("--summary_entity_column", default="summary_entities")
    parser.add_argument("--summary_entity_types_column", default="summary_entity_types")
    parser.add_argument("--summary_entity_spans_column", default="summary_entity_spans")
    parser.add_argument("--max_entities", type=int, default=32)
    parser.add_argument("--max_entity_token_length", type=int, default=16)
    parser.add_argument("--max_summary_entities", type=int, default=16)

    # Three-layer MI loss weights
    parser.add_argument("--lambda_node", type=float, default=0.1, help="Weight for node-layer (entity-type InfoNCE) loss.")
    parser.add_argument("--lambda_link", type=float, default=0.05, help="Weight for link-layer (TransE) loss.")
    parser.add_argument("--lambda_network", type=float, default=0.03, help="Weight for network-layer (spectral graph) loss.")
    parser.add_argument("--cooccurrence_window", type=int, default=200, help="Character window for entity co-occurrence.")
    parser.add_argument("--missing_entity_penalty", type=float, default=0.5, help="Penalty when summary misses an entity type.")
    parser.add_argument("--use_link_layer", action="store_true", help="Enable link-layer TransE constraint.")
    parser.add_argument("--use_network_layer", action="store_true", help="Enable network-layer spectral alignment.")

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


class CustomDataCollator(DataCollatorForSeq2Seq):
    """
    Extends DataCollatorForSeq2Seq to preserve string-list fields
    (entity types / spans) that cannot be tensor-stacked.
    """

    def __call__(self, features, return_tensors=None):
        # Extract custom fields before parent collator touches them.
        entity_type_list = [f.pop("entity_type_list", []) for f in features]
        entity_span_list = [f.pop("entity_span_list", []) for f in features]
        summary_entity_type_list = [f.pop("summary_entity_type_list", []) for f in features]
        summary_entity_span_list = [f.pop("summary_entity_span_list", []) for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        batch["entity_type_list"] = entity_type_list
        batch["entity_span_list"] = entity_span_list
        batch["summary_entity_type_list"] = summary_entity_type_list
        batch["summary_entity_span_list"] = summary_entity_span_list
        return batch


class LoRAMITrainer(Seq2SeqTrainer):
    """
    L_total = L_mle + lambda_node * L_node + lambda_link * L_link + lambda_network * L_network
    """

    def __init__(
        self,
        *args: Any,
        lambda_node: float = 0.1,
        lambda_link: float = 0.05,
        lambda_network: float = 0.03,
        use_entity_prior: bool = False,
        use_link_layer: bool = False,
        use_network_layer: bool = False,
        missing_entity_penalty: float = 0.5,
        cooccurrence_window: int = 200,
        hidden_dim: int = 512,
        tokenizer: AutoTokenizer,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.lambda_node = float(lambda_node)
        self.lambda_link = float(lambda_link)
        self.lambda_network = float(lambda_network)
        self.use_entity_prior = bool(use_entity_prior)
        self.use_link_layer = bool(use_link_layer)
        self.use_network_layer = bool(use_network_layer)

        self.node_layer = NodeLayerLoss(missing_penalty=missing_entity_penalty)
        self.link_layer: LinkLayerLoss | None = None
        self.network_layer: NetworkLayerLoss | None = None
        if self.use_link_layer:
            self.link_layer = LinkLayerLoss(
                hidden_dim=hidden_dim,
                cooccurrence_window=cooccurrence_window,
            )
        if self.use_network_layer:
            self.network_layer = NetworkLayerLoss(
                k=8,
                hidden_dim=hidden_dim,
                cooccurrence_window=cooccurrence_window,
            )

    def _aux_modules(self) -> list[torch.nn.Module]:
        modules: list[torch.nn.Module] = [self.node_layer]
        if self.link_layer is not None:
            modules.append(self.link_layer)
        if self.network_layer is not None:
            modules.append(self.network_layer)
        return modules

    def _move_aux_modules_to_device(self, device: torch.device) -> None:
        for module in self._aux_modules():
            module.to(device)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        aux_params = []
        for module in self._aux_modules():
            aux_params.extend([p for p in module.parameters() if p.requires_grad])
        if not aux_params:
            return optimizer

        existing = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }
        new_params = [p for p in aux_params if id(p) not in existing]
        if new_params:
            optimizer.add_param_group(
                {
                    "params": new_params,
                    "weight_decay": self.args.weight_decay,
                }
            )
        return optimizer

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        model_inputs = dict(inputs)

        # Pop custom non-tensor fields.
        entity_type_list = model_inputs.pop("entity_type_list", None)
        entity_span_list = model_inputs.pop("entity_span_list", None)
        summary_entity_type_list = model_inputs.pop("summary_entity_type_list", None)
        summary_entity_span_list = model_inputs.pop("summary_entity_span_list", None)

        # Pop entity tensors [B, N, T].
        entity_input_ids = model_inputs.pop("entity_input_ids", None)
        entity_attention_mask = model_inputs.pop("entity_attention_mask", None)
        summary_entity_input_ids = model_inputs.pop("summary_entity_input_ids", None)
        summary_entity_attention_mask = model_inputs.pop("summary_entity_attention_mask", None)

        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
        mle_loss = outputs.loss
        if mle_loss is None:
            raise RuntimeError("Model did not return loss; make sure labels are included.")

        self._move_aux_modules_to_device(mle_loss.device)
        total_mi_loss = torch.tensor(0.0, device=mle_loss.device, dtype=mle_loss.dtype)

        if self.use_entity_prior and entity_input_ids is not None:
            if summary_entity_input_ids is None or summary_entity_attention_mask is None:
                raise ValueError(
                    "Missing summary entity features. "
                    "Please ensure preprocessing includes summary entity columns."
                )
            if entity_type_list is None or summary_entity_type_list is None:
                raise ValueError(
                    "Missing entity type lists in batch. "
                    "Check preprocess + collator outputs."
                )

            # ---- Node Layer ----
            # Entity embeddings: [B, N, T] -> [B, N, T, D]
            B, N_src, T_src = entity_input_ids.shape
            entity_emb = model.get_input_embeddings()(entity_input_ids.view(B * N_src, T_src))
            entity_emb = entity_emb.view(B, N_src, T_src, -1)

            B_sum, N_sum, T_sum = summary_entity_input_ids.shape
            sum_entity_emb = model.get_input_embeddings()(summary_entity_input_ids.view(B_sum * N_sum, T_sum))
            sum_entity_emb = sum_entity_emb.view(B_sum, N_sum, T_sum, -1)

            node_loss = self.node_layer(
                entity_emb,
                entity_attention_mask,
                entity_type_list,
                sum_entity_emb,
                summary_entity_attention_mask,
                summary_entity_type_list,
            )
            total_mi_loss += self.lambda_node * node_loss

            # ---- Link Layer ----
            if self.link_layer is not None and entity_span_list is not None and summary_entity_span_list is not None:
                link_loss = self.link_layer(
                    entity_emb,
                    entity_attention_mask,
                    entity_span_list,
                    entity_type_list,
                    sum_entity_emb,
                    summary_entity_attention_mask,
                    summary_entity_span_list,
                    summary_entity_type_list,
                )
                total_mi_loss += self.lambda_link * link_loss

            # ---- Network Layer ----
            if self.network_layer is not None and entity_span_list is not None:
                # Decoder final hidden state: last layer, last token.
                decoder_hidden = outputs.decoder_hidden_states[-1]  # [B, T, D]
                decoder_final_hidden = decoder_hidden[:, -1, :]     # [B, D]

                network_loss = self.network_layer(
                    entity_emb,
                    entity_attention_mask,
                    entity_span_list,
                    decoder_final_hidden,
                )
                total_mi_loss += self.lambda_network * network_loss

        total_loss = mle_loss + total_mi_loss

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
    entity_types_col = args.entity_types_column
    entity_spans_col = args.entity_spans_column
    summary_entity_col = args.summary_entity_column
    summary_entity_types_col = args.summary_entity_types_column
    summary_entity_spans_col = args.summary_entity_spans_column

    if args.use_entity_prior:
        train_cols = set(raw_ds["train"].column_names)
        if entity_col not in train_cols:
            raise ValueError(
                f"--use_entity_prior requires column '{entity_col}' in train split. "
                "Please run data preprocessing first."
            )

    def _tokenize_entity_batch(
        entity_texts_strs: list[str],
        entity_types_strs: list[str],
        entity_spans_strs: list[str],
        max_entities: int,
        max_token_length: int,
    ) -> tuple[list[list[list[int]]], list[list[list[int]]], list[list[str]], list[list[list[int]]]]:
        """Tokenize each entity individually and pad to [B, N, T]."""
        batch_ids: list[list[list[int]]] = []
        batch_masks: list[list[list[int]]] = []
        batch_types: list[list[str]] = []
        batch_spans: list[list[list[int]]] = []

        for text_str, type_str, span_str in zip(entity_texts_strs, entity_types_strs, entity_spans_strs):
            texts = [e.strip() for e in text_str.split(";") if e.strip()]
            types = [t.strip() for t in type_str.split(";") if t.strip()]
            spans: list[list[int]] = []
            if span_str.strip():
                try:
                    spans = json.loads(span_str)
                except json.JSONDecodeError:
                    spans = []

            while len(types) < len(texts):
                types.append("UNKNOWN")
            while len(spans) < len(texts):
                spans.append([-1, -1])

            texts = texts[:max_entities]
            types = types[:max_entities]
            spans = spans[:max_entities]

            sample_ids: list[list[int]] = []
            sample_masks: list[list[int]] = []
            for text in texts:
                tok = tokenizer(
                    text,
                    max_length=max_token_length,
                    truncation=True,
                    padding="max_length",
                )
                sample_ids.append(tok["input_ids"])
                sample_masks.append(tok["attention_mask"])

            while len(sample_ids) < max_entities:
                sample_ids.append([tokenizer.pad_token_id] * max_token_length)
                sample_masks.append([0] * max_token_length)
                types.append("PAD")
                spans.append([-1, -1])

            batch_ids.append(sample_ids)
            batch_masks.append(sample_masks)
            batch_types.append(types)
            batch_spans.append(spans)

        return batch_ids, batch_masks, batch_types, batch_spans

    def preprocess_train(batch: dict[str, list[Any]]) -> dict[str, Any]:
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
            # Source entities
            src_texts = [str(x) for x in batch[entity_col]]
            src_types = [str(x) for x in batch.get(entity_types_col, [""] * len(src_texts))]
            src_spans = [str(x) for x in batch.get(entity_spans_col, [""] * len(src_texts))]

            src_ids, src_masks, src_type_list, src_span_list = _tokenize_entity_batch(
                src_texts, src_types, src_spans,
                args.max_entities, args.max_entity_token_length,
            )
            model_inputs["entity_input_ids"] = src_ids
            model_inputs["entity_attention_mask"] = src_masks
            model_inputs["entity_type_list"] = src_type_list
            model_inputs["entity_span_list"] = src_span_list

            # Summary entities
            sum_texts = [str(x) for x in batch.get(summary_entity_col, [""] * len(src_texts))]
            sum_types = [str(x) for x in batch.get(summary_entity_types_col, [""] * len(src_texts))]
            sum_spans = [str(x) for x in batch.get(summary_entity_spans_col, [""] * len(src_texts))]

            sum_ids, sum_masks, sum_type_list, sum_span_list = _tokenize_entity_batch(
                sum_texts, sum_types, sum_spans,
                args.max_summary_entities, args.max_entity_token_length,
            )
            model_inputs["summary_entity_input_ids"] = sum_ids
            model_inputs["summary_entity_attention_mask"] = sum_masks
            model_inputs["summary_entity_type_list"] = sum_type_list
            model_inputs["summary_entity_span_list"] = sum_span_list

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

    # Determine hidden_dim for link/network layer projections.
    hidden_dim = model.config.d_model if hasattr(model.config, "d_model") else 512

    trainer = LoRAMITrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
        lambda_node=args.lambda_node,
        lambda_link=args.lambda_link,
        lambda_network=args.lambda_network,
        use_entity_prior=args.use_entity_prior,
        use_link_layer=args.use_link_layer,
        use_network_layer=args.use_network_layer,
        missing_entity_penalty=args.missing_entity_penalty,
        cooccurrence_window=args.cooccurrence_window,
        hidden_dim=hidden_dim,
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
