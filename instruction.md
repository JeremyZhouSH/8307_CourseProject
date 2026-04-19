# 系统搭建流程与运行要求（instruction，当前版本）

## 1. 系统目标

本项目是一个科研论文摘要 Agent，支持：

- 论文文本加载与章节切分
- 关键信息抽取（含 CRF/HMM + ILP）
- 结构化摘要与最终摘要生成
- 启发式 Faithfulness 校验
- 多轮 Agent 决策循环（动态工具选择、重试、澄清、长期记忆）

---

## 2. 运行环境要求

建议环境：

- Python `3.10+`
- macOS / Linux（Windows 也可，但命令示例以类 Unix 为主）
- 可选 GPU（微调时建议）

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含：

- 基础：`PyYAML` `datasets` `pytest`
- 算法：`sklearn-crfsuite` `pulp`
- 微调：`torch` `transformers` `accelerate` `evaluate` `rouge_score` `peft` 等

---

## 3. 项目搭建流程（当前）

### Step 1. 准备配置

核心配置文件：`config/default.yaml`

关键参数：

- `llm.use_mock`：是否使用 mock LLM
- `llm.provider`：`openai_compatible` 或 `hf_local`
- `extractor.strategy`：`ilp` / `rule` / `hybrid`
- `extractor.word_budget`：摘要词预算（默认 200）
- `agent.max_steps`：Agent 最大循环步数
- `agent.retry_limit`：失败重试上限
- `agent.memory_path`：长期记忆文件路径

### Step 2. 准备输入数据

默认输入：`data/samples/sample_paper.txt`  
输出目录：`data/outputs/`

### Step 3. 运行基础流水线

```bash
python3 -m src.main \
  --input data/samples/sample_paper.txt \
  --output data/outputs/summary.json
```

### Step 3.1 医学大模型接入（可选）

#### A. API 模型（OpenAI-Compatible）

**方式一：使用 `.env` 文件（推荐）**

```bash
# 1. 复制模板
 cp .env.example .env

# 2. 编辑 .env，填入你的密钥
# SMART_LLM__API_KEY=sk-your-key
# SMART_LLM__MODEL_NAME=deepseek-chat
# SMART_LLM__BASE_URL=https://api.deepseek.com/v1
# SMART_LLM__USE_MOCK=false
```

程序启动时会自动加载项目根目录的 `.env` 文件，无需手动 export。

**方式二：命令行 export**

```bash
export SMART_LLM__USE_MOCK=false
export SMART_LLM__PROVIDER=openai_compatible
export SMART_LLM__API_KEY="your_key"
export SMART_LLM__BASE_URL="https://your-endpoint/v1"
export SMART_LLM__MODEL_NAME="your-med-model"
```

#### B. 本地 HF 医学模型

在 `config/default.yaml` 设置：

```yaml
llm:
  use_mock: false
  provider: hf_local
  model: your-local-med-model
  local_device_map: auto
  local_torch_dtype: auto
```

### Step 4. 运行 Agent 决策循环

```python
from src.agent.controller import AgentController
from src.pipeline import SummarizationPipeline

controller = AgentController(pipeline=SummarizationPipeline())
state = controller.run(
    input_path="data/samples/sample_paper.txt",
    output_path="data/outputs/my_agent_summary.json",
    user_request="生成学术风格摘要并输出JSON"
)
```

---

## 4. 系统结构与细节

### 4.1 主流水线（`src/pipeline.py`）

流程顺序：

1. 解析输入输出路径
2. 文档读取与章节切分
3. 关键信息抽取
4. 结构化摘要生成
5. 最终摘要生成（可选 LLM 重写）
6. Faithfulness 校验
7. 结果写出 JSON

### 4.2 抽取模块（`src/extractor/`）

- `role_tagger_crf.py`：角色标注，优先 `CRF`，失败回退 `HMM`，再回退启发式
- `ilp_sentence_selector.py`：在词预算、角色覆盖、冗余惩罚下做 ILP 选句
- `key_info_extractor.py`：把角色标注与 ILP 组合为统一抽取接口

### 4.3 Agent 循环（`src/agent/`）

- `planner.py`：按状态选择下一动作（动态工具选择）
- `tools.py`：工具注册与执行（load/split/extract/verify/write 等）
- `reviewer.py`：依据异常、faithfulness、unsupported_claims 做重试决策
- `memory.py`：长期记忆（JSONL 持久化、历史检索、策略建议）
- `dialogue.py`：澄清接口（结构化问题对象）
- `controller.py`：`plan -> act -> review -> retry/finish` 循环控制

---

## 5. 对话澄清交互协议

当输入缺失或目标不明确时，`state` 返回：

- `needs_clarification = True`
- `clarification_request`（结构化对象）

示例字段：

- `question_id`
- `question`
- `fields`（每个字段包含 `name/prompt/required/field_type`）

回填继续执行：

```python3
state = controller.run(
  output_path="data/outputs/my_agent_summary.json",
  clarification_answers={
    "input_path": "data/samples/sample_paper.txt",
    "user_request": "生成200词学术摘要并输出JSON"
  }
)
```

`clarification_request` 包含：

- `question_id`
- `question`
- `fields`（每项含 `name/prompt/required/field_type`）

---

## 6. 长期记忆机制

默认存储路径：`data/outputs/agent_memory.jsonl`

每条记录包含：

- request
- summary
- extractor_strategy
- faithfulness_score
- retry_count

用途：

- 检索相似历史任务
- 推荐历史上效果更好的抽取策略

---

## 7. 微调流程（`finetune/`，当前推荐）

### 7.1 标准 Seq2Seq 微调

```bash
python3 finetune/train_seq2seq.py \
  --dataset_name ccdv/pubmed-summarization \
  --dataset_config document \
  --train_split "train[:2000]" \
  --eval_split "validation[:200]" \
  --output_dir data/outputs/ft_pubmed
```

### 7.2 使用本地 JSONL 微调

```bash
python3 finetune/train_seq2seq.py \
  --train_file data/samples/train.jsonl \
  --eval_file data/samples/dev.jsonl \
  --output_dir data/outputs/ft_local
```

### 7.3 推理

```bash
python3 finetune/infer_seq2seq.py \
  --model_path data/outputs/ft_local \
  --input_file data/samples/sample_paper.txt
```

### 7.4 LoRA + 三层 MI 对齐框架

系统支持三种训练模式：仅节点层（默认）、节点+链路层、完整三层。

#### 7.4.1 预处理实体（默认仅训练集标注）

```bash
python3 data/preprocess_entities.py \
  --dataset_name ccdv/pubmed-summarization \
  --dataset_config document \
  --train_split "train[:2000]" \
  --eval_split "validation[:200]" \
  --ner_model en_ner_bionlp13cg_md \
  --output_train data/samples/train_ner.jsonl \
  --output_eval data/samples/dev_ner.jsonl
```

说明：

- 同时生成源文与摘要的实体文本、类型、位置信息
- 默认只给 `train` 添加实体列；`eval` 默认不标注
- 若确实需要给 eval 标注，手动加 `--annotate_eval`

#### 7.4.2 训练（三层框架）

**仅节点层（实体类型 InfoNCE）**：

```bash
python3 finetune/train_lora_mi.py \
  --train_file data/samples/train_ner.jsonl \
  --eval_file data/samples/dev_ner.jsonl \
  --output_dir data/outputs/ft_lora_mi_node \
  --use_entity_prior \
  --lambda_node 0.1
```

**节点层 + 链路层（增加 TransE 约束）**：

```bash
python3 finetune/train_lora_mi.py \
  --train_file data/samples/train_ner.jsonl \
  --eval_file data/samples/dev_ner.jsonl \
  --output_dir data/outputs/ft_lora_mi_link \
  --use_entity_prior \
  --use_link_layer \
  --lambda_node 0.1 \
  --lambda_link 0.05 \
  --cooccurrence_window 200
```

**完整三层（增加谱图网络对齐）**：

```bash
python3 finetune/train_lora_mi.py \
  --train_file data/samples/train_ner.jsonl \
  --eval_file data/samples/dev_ner.jsonl \
  --output_dir data/outputs/ft_lora_mi_full \
  --use_entity_prior \
  --use_link_layer \
  --use_network_layer \
  --lambda_node 0.1 \
  --lambda_link 0.05 \
  --lambda_network 0.03 \
  --cooccurrence_window 200 \
  --missing_entity_penalty 0.5
```

损失形式：

- `L_total = L_mle + lambda_node * L_node + lambda_link * L_link + lambda_network * L_network`
- 所有实体先验仅用于 `train_dataset`
- `eval_dataset` 不走任何实体分支，保持评估分布纯净

---

## 7.5 完整工作流：数据准备 → 训练 → 预测 → 评估

### Step 1. 数据准备（落地到本地）

```bash
python3 scripts/prepare_data.py \
  --num_train 2000 \
  --num_val 200 \
  --num_test 200
```

执行后 `data/pubmed/` 目录结构：

```
data/pubmed/
├── raw/
│   ├── train.jsonl       # 原始训练数据
│   ├── val.jsonl         # 原始验证数据
│   └── test.jsonl        # 原始测试数据
├── train_ner.jsonl       # 训练集（含实体标注，用于训练）
├── val_ner.jsonl         # 验证集（无实体标注，用于验证）
└── test.jsonl            # 测试集（无实体标注，用于最终评估）
```

### Step 2. 训练（Baseline vs 三层框架）

**Baseline（标准 LoRA）**：

```bash
python3 finetune/train_seq2seq.py \
  --train_file data/pubmed/train_ner.jsonl \
  --eval_file data/pubmed/val_ner.jsonl \
  --output_dir data/outputs/baseline
```

**三层框架（完整版）**：

```bash
python3 finetune/train_lora_mi.py \
  --train_file data/pubmed/train_ner.jsonl \
  --eval_file data/pubmed/val_ner.jsonl \
  --output_dir data/outputs/three_layer \
  --use_entity_prior \
  --use_link_layer \
  --use_network_layer \
  --lambda_node 0.1 \
  --lambda_link 0.05 \
  --lambda_network 0.03
```

### Step 3. 批量预测

在**验证集**上生成预测（用于调参对比）：

```bash
python3 scripts/batch_predict.py \
  --model_path data/outputs/three_layer \
  --input_file data/pubmed/val_ner.jsonl \
  --output_file data/outputs/predictions_val.jsonl \
  --max_samples 200
```

在**测试集**上生成预测（用于最终报告，只跑一次）：

```bash
python3 scripts/batch_predict.py \
  --model_path data/outputs/three_layer \
  --input_file data/pubmed/test.jsonl \
  --output_file data/outputs/predictions_test.jsonl
```

### Step 4. 评估

**方式 A：用本地预测文件 + 本地参考文件**

```bash
python3 scripts/evaluate.py \
  --pred_file data/outputs/predictions_val.jsonl \
  --ref_file data/pubmed/raw/val.jsonl \
  --output_file data/outputs/eval_val.json
```

**方式 B：用本地预测文件 + HuggingFace 在线参考**

```bash
python3 scripts/evaluate.py \
  --pred_file data/outputs/predictions_val.jsonl \
  --dataset_name ccdv/pubmed-summarization \
  --dataset_config document \
  --split validation \
  --max_samples 200 \
  --output_file data/outputs/eval_val.json
```

评估输出示例：

```json
{
  "rouge1": 0.4521,
  "rouge2": 0.2134,
  "rougeL": 0.3892,
  "bleu": 18.56,
  "bertscore_f1": 0.8912,
  "faithfulness": 0.7234,
  "entity_recall": 0.6543,
  "entity_f1": 0.6123,
  "num_samples": 200
}
```

### Step 5. 对比实验（Baseline vs 三层）

```bash
# Baseline 预测
python3 scripts/batch_predict.py \
  --model_path data/outputs/baseline \
  --input_file data/pubmed/test.jsonl \
  --output_file data/outputs/predictions_baseline.jsonl

# 三层框架预测
python3 scripts/batch_predict.py \
  --model_path data/outputs/three_layer \
  --input_file data/pubmed/test.jsonl \
  --output_file data/outputs/predictions_three_layer.jsonl

# 分别评估
python3 scripts/evaluate.py \
  --pred_file data/outputs/predictions_baseline.jsonl \
  --ref_file data/pubmed/raw/val.jsonl \
  --output_file data/outputs/eval_baseline.json

python3 scripts/evaluate.py \
  --pred_file data/outputs/predictions_three_layer.jsonl \
  --ref_file data/pubmed/raw/val.jsonl \
  --output_file data/outputs/eval_three_layer.json
```

---

## 8. 测试与验收

运行测试：

```bash
pytest -q
```

验收重点：

- 基础 pipeline 能端到端产出 `summary.json`
- Agent 能输出 `agent_trace`
- 低质量时能触发重试
- 触发澄清时能返回 `clarification_request`
- 回填答案后可继续执行并成功写出结果

---

## 9. 常见问题

1. `needs_clarification=True`  
说明输入路径或目标描述有问题，按 `clarification_request.fields` 回填后重跑。

2. `LLM` 失败  
系统会自动回退本地摘要，不会中断整体流程。

3. `pulp` 或 `sklearn-crfsuite` 不可用  
抽取模块会自动回退策略，但建议安装完整依赖以获得最佳效果。

4. 医学模型接入失败  
先检查 `llm.provider`、模型名、API 地址和密钥；`hf_local` 模式需本地可加载该模型。
