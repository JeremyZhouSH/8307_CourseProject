from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from mi_layers_v2 import LinkLayerLossV2, NetworkLayerLossV2, NodeLayerLossV2

# Batch keys shared by preprocess -> collator -> trainer.
SRC_ENTITY_TEXT_KEY = "src_entity_text"
SRC_ENTITY_TYPES_KEY = "src_entity_types"
SRC_ENTITY_SPANS_KEY = "src_entity_spans"
SUM_ENTITY_TEXT_KEY = "sum_entity_text"
SUM_ENTITY_TYPES_KEY = "sum_entity_types"
SUM_ENTITY_SPANS_KEY = "sum_entity_spans"

ENTITY_INPUT_IDS_KEY = "entity_input_ids"
ENTITY_ATTN_MASK_KEY = "entity_attention_mask"
ENTITY_TYPES_LIST_KEY = "entity_type_list"
ENTITY_SPANS_LIST_KEY = "entity_span_list"
SUM_ENTITY_INPUT_IDS_KEY = "summary_entity_input_ids"
SUM_ENTITY_ATTN_MASK_KEY = "summary_entity_attention_mask"
SUM_ENTITY_TYPES_LIST_KEY = "summary_entity_type_list"
SUM_ENTITY_SPANS_LIST_KEY = "summary_entity_span_list"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA + MI v2 fine-tuning for summarization.")

    parser.add_argument("--model_name", default="google/flan-t5-xxl")
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
    parser.add_argument("--generation_max_length", type=int, default=256)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", default="q,v")

    parser.add_argument("--use_entity_prior", action="store_true")
    parser.add_argument("--entity_column", default="entity_text")
    parser.add_argument("--entity_types_column", default="entity_types")
    parser.add_argument("--entity_spans_column", default="entity_spans")
    parser.add_argument("--summary_entity_column", default="summary_entities")
    parser.add_argument("--summary_entity_types_column", default="summary_entity_types")
    parser.add_argument("--summary_entity_spans_column", default="summary_entity_spans")
    parser.add_argument("--max_entities", type=int, default=32)
    parser.add_argument("--max_entity_token_length", type=int, default=16)
    parser.add_argument("--max_summary_entities", type=int, default=16)

    parser.add_argument("--lambda_node", type=float, default=0.1)
    parser.add_argument("--lambda_link", type=float, default=0.05)
    parser.add_argument("--lambda_network", type=float, default=0.03)
    parser.add_argument("--cooccurrence_window", type=int, default=200)
    parser.add_argument("--distance_tau", type=float, default=120.0)
    parser.add_argument("--relation_buckets", type=int, default=4096)
    parser.add_argument("--link_margin", type=float, default=0.2)
    parser.add_argument("--missing_entity_penalty", type=float, default=0.5)
    parser.add_argument("--use_link_layer", action="store_true")
    parser.add_argument("--use_network_layer", action="store_true")
    parser.add_argument("--compute_bertscore", action="store_true", help="Compute BERTScore during evaluation (slower).")
    parser.add_argument("--resume_from_checkpoint", default="", help="Path to a checkpoint to resume training from.")
    parser.add_argument("--lora_pretrained", default="", help="Path to a saved LoRA adapter directory to initialize from.")
    parser.add_argument("--freeze_lora", action="store_true", help="Freeze LoRA parameters when training MI layers (only valid with --lora_pretrained).")

    return parser


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

    train_split = load_dataset(
        args.dataset_name,
        args.dataset_config or None,
        split=args.train_split,
    )
    eval_split = load_dataset(
        args.dataset_name,
        args.dataset_config or None,
        split=args.eval_split,
    )
    return DatasetDict(train=train_split, validation=eval_split)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return str(value[0])
        return "; ".join(str(x) for x in value)
    return str(value)


def _parse_spans(span_value: Any) -> list[list[int]]:
    if span_value is None:
        return []

    if isinstance(span_value, str):
        text = span_value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
    elif isinstance(span_value, (list, tuple, np.ndarray)):
        parsed = span_value
    else:
        return []

    spans: list[list[int]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start = int(item[0])
            end = int(item[1])
        except (TypeError, ValueError):
            continue
        spans.append([start, end])
    return spans


def _split_semicolon_items(raw_text: Any) -> list[str]:
    text = _normalize_text(raw_text)
    return [x.strip() for x in text.split(";") if x.strip()]


def _tensor_to_nested_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


class CustomDataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        *args: Any,
        max_entities: int = 32,
        max_summary_entities: int = 16,
        max_entity_token_length: int = 16,
        include_entity_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_entities = max_entities
        self.max_summary_entities = max_summary_entities
        self.max_entity_token_length = max_entity_token_length
        self.include_entity_features = include_entity_features

    def _tokenize_entity_batch(
        self,
        entity_text_values: list[Any],
        entity_type_values: list[Any],
        entity_span_values: list[Any],
        max_entities: int,
    ) -> tuple[list[list[list[int]]], list[list[list[int]]], list[list[str]], list[list[list[int]]]]:
        batch_ids: list[list[list[int]]] = []
        batch_masks: list[list[list[int]]] = []
        batch_types: list[list[str]] = []
        batch_spans: list[list[list[int]]] = []

        for text_value, type_value, span_value in zip(entity_text_values, entity_type_values, entity_span_values):
            texts = _split_semicolon_items(text_value)
            types = _split_semicolon_items(type_value)
            spans = _parse_spans(span_value)

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
                tokenized = self.tokenizer(
                    text,
                    max_length=self.max_entity_token_length,
                    truncation=True,
                    padding="max_length",
                )
                sample_ids.append(tokenized["input_ids"])
                sample_masks.append(tokenized["attention_mask"])

            while len(sample_ids) < max_entities:
                sample_ids.append([self.tokenizer.pad_token_id] * self.max_entity_token_length)
                sample_masks.append([0] * self.max_entity_token_length)
                types.append("PAD")
                spans.append([-1, -1])

            batch_ids.append(sample_ids)
            batch_masks.append(sample_masks)
            batch_types.append(types)
            batch_spans.append(spans)

        return batch_ids, batch_masks, batch_types, batch_spans

    def __call__(self, features: list[dict[str, Any]], return_tensors: str | None = None) -> dict[str, Any]:
        if not self.include_entity_features:
            return super().__call__(features, return_tensors=return_tensors)

        entity_text_values = [feature.pop(SRC_ENTITY_TEXT_KEY, "") for feature in features]
        entity_type_values = [feature.pop(SRC_ENTITY_TYPES_KEY, "") for feature in features]
        entity_span_values = [feature.pop(SRC_ENTITY_SPANS_KEY, "") for feature in features]
        summary_entity_text_values = [feature.pop(SUM_ENTITY_TEXT_KEY, "") for feature in features]
        summary_entity_type_values = [feature.pop(SUM_ENTITY_TYPES_KEY, "") for feature in features]
        summary_entity_span_values = [feature.pop(SUM_ENTITY_SPANS_KEY, "") for feature in features]

        src_ids, src_masks, src_types, src_spans = self._tokenize_entity_batch(
            entity_text_values,
            entity_type_values,
            entity_span_values,
            self.max_entities,
        )
        sum_ids, sum_masks, sum_types, sum_spans = self._tokenize_entity_batch(
            summary_entity_text_values,
            summary_entity_type_values,
            summary_entity_span_values,
            self.max_summary_entities,
        )

        batch = super().__call__(features, return_tensors=return_tensors)
        batch[ENTITY_INPUT_IDS_KEY] = torch.tensor(src_ids, dtype=torch.long)
        batch[ENTITY_ATTN_MASK_KEY] = torch.tensor(src_masks, dtype=torch.long)
        batch[ENTITY_TYPES_LIST_KEY] = src_types
        batch[ENTITY_SPANS_LIST_KEY] = torch.tensor(src_spans, dtype=torch.long)

        batch[SUM_ENTITY_INPUT_IDS_KEY] = torch.tensor(sum_ids, dtype=torch.long)
        batch[SUM_ENTITY_ATTN_MASK_KEY] = torch.tensor(sum_masks, dtype=torch.long)
        batch[SUM_ENTITY_TYPES_LIST_KEY] = sum_types
        batch[SUM_ENTITY_SPANS_LIST_KEY] = torch.tensor(sum_spans, dtype=torch.long)
        return batch


class LoRAMITrainerV2(Seq2SeqTrainer):
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
        distance_tau: float = 120.0,
        relation_buckets: int = 4096,
        link_margin: float = 0.2,
        hidden_dim: int = 512,
        tokenizer: AutoTokenizer,
        log_dir: str = "report/logs",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.lambda_node = float(lambda_node)
        self.lambda_link = float(lambda_link)
        self.lambda_network = float(lambda_network)
        self.use_entity_prior = bool(use_entity_prior)
        self.use_link_layer = bool(use_link_layer)
        self.use_network_layer = bool(use_network_layer)
        self.distance_tau = float(distance_tau)
        self.relation_buckets = int(relation_buckets)
        self.link_margin = float(link_margin)

        self.node_layer = NodeLayerLossV2(missing_penalty=missing_entity_penalty)
        self.link_layer: LinkLayerLossV2 | None = None
        self.network_layer: NetworkLayerLossV2 | None = None

        if self.use_link_layer:
            self.link_layer = LinkLayerLossV2(
                hidden_dim=hidden_dim,
                cooccurrence_window=cooccurrence_window,
                distance_tau=self.distance_tau,
                relation_buckets=self.relation_buckets,
                margin=self.link_margin,
            )
        if self.use_network_layer:
            self.network_layer = NetworkLayerLossV2(
                k=8,
                hidden_dim=hidden_dim,
                cooccurrence_window=cooccurrence_window,
                distance_tau=self.distance_tau,
            )

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_losses: dict[str, float] = {}
        self.training_records: list[dict[str, Any]] = []
        self.start_time: float | None = None

    def _aux_modules(self) -> list[torch.nn.Module]:
        modules: list[torch.nn.Module] = [self.node_layer]
        if self.link_layer is not None:
            modules.append(self.link_layer)
        if self.network_layer is not None:
            modules.append(self.network_layer)
        return modules

    def _move_aux_modules_to_device(self, device: torch.device, dtype: torch.dtype) -> None:
        for module in self._aux_modules():
            module.to(device=device, dtype=dtype)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        aux_params: list[torch.nn.Parameter] = []
        for module in self._aux_modules():
            aux_params.extend([p for p in module.parameters() if p.requires_grad])
        if not aux_params:
            return optimizer

        existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
        new_params = [p for p in aux_params if id(p) not in existing]
        if new_params:
            optimizer.add_param_group({"params": new_params, "weight_decay": self.args.weight_decay})
        return optimizer

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch

        model_inputs = dict(inputs)
        entity_type_list = model_inputs.pop(ENTITY_TYPES_LIST_KEY, None)
        entity_span_list = model_inputs.pop(ENTITY_SPANS_LIST_KEY, None)
        summary_entity_type_list = model_inputs.pop(SUM_ENTITY_TYPES_LIST_KEY, None)
        summary_entity_span_list = model_inputs.pop(SUM_ENTITY_SPANS_LIST_KEY, None)

        entity_input_ids = model_inputs.pop(ENTITY_INPUT_IDS_KEY, None)
        entity_attention_mask = model_inputs.pop(ENTITY_ATTN_MASK_KEY, None)
        summary_entity_input_ids = model_inputs.pop(SUM_ENTITY_INPUT_IDS_KEY, None)
        summary_entity_attention_mask = model_inputs.pop(SUM_ENTITY_ATTN_MASK_KEY, None)

        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
        mle_loss = outputs.loss
        if mle_loss is None:
            raise RuntimeError("Model did not return loss; make sure labels are provided.")

        self._move_aux_modules_to_device(mle_loss.device, mle_loss.dtype)
        total_mi_loss = torch.zeros((), device=mle_loss.device, dtype=mle_loss.dtype)

        node_loss = torch.zeros((), device=mle_loss.device, dtype=mle_loss.dtype)
        link_loss = torch.zeros((), device=mle_loss.device, dtype=mle_loss.dtype)
        network_loss = torch.zeros((), device=mle_loss.device, dtype=mle_loss.dtype)

        has_entity_features = (
            self.use_entity_prior
            and entity_input_ids is not None
            and entity_attention_mask is not None
            and summary_entity_input_ids is not None
            and summary_entity_attention_mask is not None
            and entity_type_list is not None
            and summary_entity_type_list is not None
            and int(entity_attention_mask.sum().item()) > 0
            and int(summary_entity_attention_mask.sum().item()) > 0
        )

        if has_entity_features:
            encoder_hidden = outputs.encoder_last_hidden_state  # [B, S, D]
            d_model = encoder_hidden.size(-1)

            # Source entities: attend over main encoder output (context-aware, gradients flow through LoRA)
            enc_dtype = encoder_hidden.dtype
            ent_mask_f = entity_attention_mask.to(enc_dtype).unsqueeze(-1)  # [B, N, T, 1]
            entity_query = model.get_input_embeddings()(entity_input_ids).to(enc_dtype)  # [B, N, T, D]
            entity_query_pooled = (entity_query * ent_mask_f).sum(dim=2) / ent_mask_f.sum(dim=2).clamp_min(1.0)  # [B, N, D]
            attn_scores = torch.bmm(entity_query_pooled, encoder_hidden.transpose(1, 2)) / math.sqrt(d_model)  # [B, N, S]
            attn_weights = F.softmax(attn_scores, dim=-1)
            entity_emb_ctx = torch.bmm(attn_weights, encoder_hidden)  # [B, N, D]
            # Zero out PAD entities, reshape to [B, N, 1, D] for NodeLayerLoss
            entity_valid = (entity_attention_mask.sum(dim=2) > 0).to(enc_dtype).unsqueeze(-1)  # [B, N, 1]
            entity_emb = (entity_emb_ctx * entity_valid).unsqueeze(2)  # [B, N, 1, D]
            entity_attention_mask = entity_valid.long().squeeze(-1).unsqueeze(-1)  # [B, N, 1]

            # Summary entities: encode separately (not visible to encoder)
            encoder = model.get_encoder()
            bsz_sum, n_sum, t_sum = summary_entity_input_ids.shape
            summary_entity_encoder_out = encoder(
                input_ids=summary_entity_input_ids.view(bsz_sum * n_sum, t_sum),
                attention_mask=summary_entity_attention_mask.view(bsz_sum * n_sum, t_sum),
            ).last_hidden_state
            summary_entity_emb = summary_entity_encoder_out.view(bsz_sum, n_sum, t_sum, -1)

            node_loss = self.node_layer(
                entity_emb,
                entity_attention_mask,
                entity_type_list,
                summary_entity_emb,
                summary_entity_attention_mask,
                summary_entity_type_list,
            )
            node_loss = node_loss.to(dtype=mle_loss.dtype)
            total_mi_loss = total_mi_loss + self.lambda_node * node_loss

            entity_span_list_py = _tensor_to_nested_list(entity_span_list)
            summary_entity_span_list_py = _tensor_to_nested_list(summary_entity_span_list)

            if (
                self.link_layer is not None
                and entity_span_list_py is not None
                and summary_entity_span_list_py is not None
            ):
                link_loss = self.link_layer(
                    entity_emb,
                    entity_attention_mask,
                    entity_span_list_py,
                    entity_type_list,
                    summary_entity_emb,
                    summary_entity_attention_mask,
                    summary_entity_span_list_py,
                    summary_entity_type_list,
                )
                link_loss = link_loss.to(dtype=mle_loss.dtype)
                total_mi_loss = total_mi_loss + self.lambda_link * link_loss

            if self.network_layer is not None and entity_span_list_py is not None:
                network_loss = self.network_layer(
                    entity_emb,
                    entity_attention_mask,
                    entity_span_list_py,
                    outputs.encoder_last_hidden_state,
                )
                network_loss = network_loss.to(dtype=mle_loss.dtype)
                total_mi_loss = total_mi_loss + self.lambda_network * network_loss

        total_loss = mle_loss + total_mi_loss
        self.step_losses = {
            "mle_loss": round(float(mle_loss.item()), 6),
            "total_mi_loss": round(float(total_mi_loss.item()), 6),
            "total_loss": round(float(total_loss.item()), 6),
            "node_loss": round(float(node_loss.item()), 6),
            "link_loss": round(float(link_loss.item()), 6),
            "network_loss": round(float(network_loss.item()), 6),
        }

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Remove entity-specific keys before calling model.generate() during eval."""
        inputs = dict(inputs)
        for key in (
            ENTITY_TYPES_LIST_KEY,
            ENTITY_SPANS_LIST_KEY,
            SUM_ENTITY_TYPES_LIST_KEY,
            SUM_ENTITY_SPANS_LIST_KEY,
            ENTITY_INPUT_IDS_KEY,
            ENTITY_ATTN_MASK_KEY,
            SUM_ENTITY_INPUT_IDS_KEY,
            SUM_ENTITY_ATTN_MASK_KEY,
        ):
            inputs.pop(key, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def log(self, logs: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> None:
        if logs is None:
            logs = {}
        merged_logs = dict(logs)
        merged_logs.update(self.step_losses)

        record = {
            "step": int(self.state.global_step),
            "epoch": round(float(self.state.epoch or 0.0), 4),
            **merged_logs,
        }
        self.training_records.append(record)
        if len(self.training_records) % 10 == 0:
            self._save_training_logs()

        super().log(merged_logs, *args, **kwargs)

    def _save_training_logs(self) -> None:
        path = self.log_dir / "training_log.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.training_records, f, ensure_ascii=False, indent=2)

    def _save_training_summary(self, elapsed_seconds: float) -> None:
        summary = {
            "total_steps": int(self.state.global_step),
            "total_epochs": float(self.args.num_train_epochs),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "elapsed_minutes": round(elapsed_seconds / 60.0, 2),
            "best_metric": self.state.best_metric,
            "best_model_checkpoint": self.state.best_model_checkpoint,
        }
        if self.state.log_history:
            last_log = self.state.log_history[-1]
            summary["final_train_loss"] = last_log.get("loss")
            summary["final_learning_rate"] = last_log.get("learning_rate")

        path = self.log_dir / "training_summary.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def save_aux_state(self, path: str) -> None:
        """Save MI layer weights (node/link/network) separately."""
        state: dict[str, Any] = {}
        if self.node_layer is not None:
            state["node"] = self.node_layer.state_dict()
        if self.link_layer is not None:
            state["link"] = self.link_layer.state_dict()
        if self.network_layer is not None:
            state["network"] = self.network_layer.state_dict()
        if state:
            torch.save(state, path)
            print(f"[MI] Saved auxiliary layers to {path}")

    def load_aux_state(self, path: str) -> None:
        """Load MI layer weights if present."""
        if not Path(path).exists():
            return
        state = torch.load(path, map_location="cpu")
        if "node" in state and self.node_layer is not None:
            self.node_layer.load_state_dict(state["node"])
        if "link" in state and self.link_layer is not None:
            self.link_layer.load_state_dict(state["link"])
        if "network" in state and self.network_layer is not None:
            self.network_layer.load_state_dict(state["network"])
        print(f"[MI] Loaded auxiliary layers from {path}")

    def train(self, *args: Any, **kwargs: Any) -> Any:
        self.start_time = time.time()
        result = super().train(*args, **kwargs)
        elapsed_seconds = time.time() - self.start_time
        self._save_training_logs()
        self._save_training_summary(elapsed_seconds)
        return result


def _require_columns(ds: DatasetDict, split: str, cols: list[str]) -> None:
    available = set(ds[split].column_names)
    missing = [c for c in cols if c not in available]
    if missing:
        raise ValueError(f"Missing required columns in {split} split: {missing}")


def _build_config_for_report(args: argparse.Namespace, tokenized_train_len: int, tokenized_eval_len: int) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "base_model": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "generation_max_length": args.generation_max_length,
        "lambda_node": args.lambda_node,
        "lambda_link": args.lambda_link,
        "lambda_network": args.lambda_network,
        "use_entity_prior": args.use_entity_prior,
        "use_link_layer": args.use_link_layer,
        "use_network_layer": args.use_network_layer,
        "missing_entity_penalty": args.missing_entity_penalty,
        "cooccurrence_window": args.cooccurrence_window,
        "distance_tau": args.distance_tau,
        "relation_buckets": args.relation_buckets,
        "link_margin": args.link_margin,
        "mi_version": "v2",
        "compute_bertscore": args.compute_bertscore,
        "train_samples": tokenized_train_len,
        "eval_samples": tokenized_eval_len,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Cannot use both --bf16 and --fp16.")

    raw_ds = load_data(args)
    _require_columns(raw_ds, "train", [args.text_column, args.summary_column])
    _require_columns(raw_ds, "validation", [args.text_column, args.summary_column])

    if args.use_entity_prior and args.entity_column not in set(raw_ds["train"].column_names):
        raise ValueError(
            f"--use_entity_prior requires column '{args.entity_column}' in train split. "
            "Please run data preprocessing first."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch_dtype)
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    if args.lora_pretrained:
        model = PeftModel.from_pretrained(base_model, args.lora_pretrained, is_trainable=True)
        print(f"Loaded LoRA weights from {args.lora_pretrained}")
        if args.freeze_lora:
            frozen = 0
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
                    frozen += 1
            print(f"[Freeze] Frozen {frozen} LoRA parameters, only MI layers will be trained.")
    else:
        if args.freeze_lora:
            raise ValueError("--freeze_lora requires --lora_pretrained.")
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

    def _batch_get_str(batch: dict[str, list[Any]], key: str, size: int) -> list[str]:
        if key not in batch:
            return [""] * size
        return [_normalize_text(x) for x in batch[key]]

    def preprocess_train(batch: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = ["summarize: " + _normalize_text(x) for x in batch[text_col]]
        targets = [_normalize_text(x) for x in batch[summary_col]]

        model_inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        if args.use_entity_prior:
            bsz = len(inputs)
            model_inputs[SRC_ENTITY_TEXT_KEY] = _batch_get_str(batch, args.entity_column, bsz)
            model_inputs[SRC_ENTITY_TYPES_KEY] = _batch_get_str(batch, args.entity_types_column, bsz)
            model_inputs[SRC_ENTITY_SPANS_KEY] = _batch_get_str(batch, args.entity_spans_column, bsz)
            model_inputs[SUM_ENTITY_TEXT_KEY] = _batch_get_str(batch, args.summary_entity_column, bsz)
            model_inputs[SUM_ENTITY_TYPES_KEY] = _batch_get_str(batch, args.summary_entity_types_column, bsz)
            model_inputs[SUM_ENTITY_SPANS_KEY] = _batch_get_str(batch, args.summary_entity_spans_column, bsz)

        return model_inputs

    def preprocess_eval(batch: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = ["summarize: " + _normalize_text(x) for x in batch[text_col]]
        targets = [_normalize_text(x) for x in batch[summary_col]]

        model_inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Build a cache fingerprint from key preprocessing args so that
    # changing split / length / entity settings invalidates stale caches.
    cache_key = "|".join(
        [
            args.model_name,
            str(args.train_file),
            str(args.eval_file),
            str(args.dataset_name),
            str(args.train_split),
            str(args.eval_split),
            str(args.max_input_length),
            str(args.max_target_length),
            str(args.use_entity_prior),
            str(args.text_column),
            str(args.summary_column),
        ]
    )
    cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
    cache_dir = Path(args.output_dir) / "tokenized_datasets" / cache_hash
    train_cache = cache_dir / "train"
    eval_cache = cache_dir / "eval"

    if train_cache.exists() and eval_cache.exists():
        print(f"[Cache] Loading tokenized datasets from {cache_dir}")
        tokenized_train = load_from_disk(str(train_cache))
        tokenized_eval = load_from_disk(str(eval_cache))
    else:
        # Let HF datasets use its own arrow cache (load_from_cache_file=True by default).
        # The manual save_to_disk below gives an extra fast-load layer.
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
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenized_train.save_to_disk(str(train_cache))
        tokenized_eval.save_to_disk(str(eval_cache))
        print(f"[Cache] Tokenized datasets saved to {cache_dir}")

    try:
        import evaluate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'evaluate'. Install with: pip install evaluate rouge_score"
        ) from exc

    rouge = evaluate.load("rouge")
    bertscore = None
    if args.compute_bertscore:
        try:
            bertscore = evaluate.load("bertscore")
        except Exception as exc:
            print(f"[WARN] Failed to load bertscore metric: {exc}")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)
        labels = np.asarray(labels)

        # Some eval paths pad predictions with -100; decode requires valid token ids.
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = preds.astype(np.int64, copy=False)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = labels.astype(np.int64, copy=False)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        if bertscore is not None:
            try:
                bs = bertscore.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    lang="en",
                    model_type="distilbert-base-uncased",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                result["bertscore_f1"] = round(float(bs["f1"].mean()), 4)
            except Exception:
                pass

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
        generation_max_length=args.generation_max_length,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy != "no" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy != "no" else None,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=(args.eval_strategy != "no"),
        metric_for_best_model="rougeL",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
    )

    hidden_dim = 512
    if hasattr(model.config, "d_model"):
        hidden_dim = int(model.config.d_model)
    elif hasattr(model.config, "hidden_size"):
        hidden_dim = int(model.config.hidden_size)

    report_log_dir = Path("report/logs")
    report_log_dir.mkdir(parents=True, exist_ok=True)

    with (report_log_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            _build_config_for_report(args, len(tokenized_train), len(tokenized_eval)),
            f,
            ensure_ascii=False,
            indent=2,
        )

    trainer = LoRAMITrainerV2(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(
            tokenizer=tokenizer,
            model=model,
            max_entities=args.max_entities,
            max_summary_entities=args.max_summary_entities,
            max_entity_token_length=args.max_entity_token_length,
            include_entity_features=args.use_entity_prior,
        ),
        compute_metrics=compute_metrics,
        lambda_node=args.lambda_node,
        lambda_link=args.lambda_link,
        lambda_network=args.lambda_network,
        use_entity_prior=args.use_entity_prior,
        use_link_layer=args.use_link_layer,
        use_network_layer=args.use_network_layer,
        missing_entity_penalty=args.missing_entity_penalty,
        cooccurrence_window=args.cooccurrence_window,
        distance_tau=args.distance_tau,
        relation_buckets=args.relation_buckets,
        link_margin=args.link_margin,
        hidden_dim=hidden_dim,
        log_dir=str(report_log_dir),
    )

    if args.lora_pretrained:
        mi_state_path = Path(args.lora_pretrained) / "mi_layers.pt"
        if mi_state_path.exists():
            trainer.load_aux_state(str(mi_state_path))

    resume_ckpt = args.resume_from_checkpoint if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_aux_state(str(output_dir / "mi_layers.pt"))

    metrics = trainer.evaluate()
    print("=== Eval Metrics ===")
    for key, value in metrics.items():
        if key.startswith("eval_"):
            print(f"{key}: {value}")

    with (report_log_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k.startswith("eval_")}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
