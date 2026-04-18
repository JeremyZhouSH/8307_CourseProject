# 微调目录说明

这个目录提供一个可直接运行的 Seq2Seq 微调模板，适用于论文摘要任务。

## 1. 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 2. 方案 A：用 Hugging Face 数据集微调

默认示例使用 `ccdv/pubmed-summarization`：

```bash
python finetune/train_seq2seq.py \
  --dataset_name ccdv/pubmed-summarization \
  --dataset_config document \
  --train_split "train[:2000]" \
  --eval_split "validation[:200]" \
  --output_dir data/outputs/ft_pubmed
```

## 3. 方案 B：用本地 JSONL 数据微调

本地 JSONL 每行需要包含两个字段：

- `article`：原文
- `abstract`：目标摘要

示例：

```json
{"article":"...","abstract":"..."}
{"article":"...","abstract":"..."}
```

训练命令：

```bash
python finetune/train_seq2seq.py \
  --train_file data/samples/train.jsonl \
  --eval_file data/samples/dev.jsonl \
  --output_dir data/outputs/ft_local
```

## 4. 推理测试

```bash
python finetune/infer_seq2seq.py \
  --model_path data/outputs/ft_local \
  --input_file data/samples/sample_paper.txt
```

## 5. 常用参数

- `--model_name`：基础模型，默认 `google/flan-t5-small`
- `--max_input_length`：输入截断长度，默认 `1024`
- `--max_target_length`：目标摘要长度，默认 `256`
- `--learning_rate`：学习率，默认 `2e-5`
- `--num_train_epochs`：训练轮数，默认 `2`

## 6. LoRA + MI 训练（对应 method.md）

如果你要跑“LoRA + 互信息约束”版本，推荐先做实体预处理，再训练。

### 6.1 实体预处理（在训练前）

```bash
python data/preprocess_entities.py \
  --dataset_name ccdv/pubmed-summarization \
  --dataset_config document \
  --train_split "train[:2000]" \
  --eval_split "validation[:200]" \
  --ner_model en_ner_bionlp13cg_md \
  --output_train data/samples/train_ner.jsonl \
  --output_eval data/samples/dev_ner.jsonl
```

说明：

- 默认只给 `train` 生成 `entity_text`；
- `eval`（以及你后续测试集）保持原字段，不加实体列；
- 若你确实要给 eval 也标注，可显式加 `--annotate_eval`。

### 6.2 LoRA + MI 训练（读取已标注实体列）

```bash
python finetune/train_lora_mi.py \
  --train_file data/samples/train_ner.jsonl \
  --eval_file data/samples/dev_ner.jsonl \
  --output_dir data/outputs/ft_lora_mi \
  --lambda_mi 0.1 \
  --use_entity_prior \
  --entity_column entity_text \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1
```

说明：

- 总损失：`L_total = L_mle + lambda_mi * L_mi`
- 开启 `--use_entity_prior` 后，训练会读取 `entity_column`（如 `entity_text`）参与 MI 对齐
- 训练阶段不再运行 NER，实体识别完全在预处理脚本完成
- 验证阶段默认不使用实体先验（仅训练集使用）

### 6.3 安装 `en_ner_bionlp13cg_md`

先安装 spaCy/scispaCy（版本可按你的环境调整）：

```bash
pip install spacy scispacy
```

再安装模型（示例版本）：

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```
