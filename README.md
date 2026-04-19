# Scientific Paper Summarization Agent

当前版本是一个可本地运行的科研论文摘要系统，包含：

- 规则与统计混合抽取（CRF/HMM + ILP）
- 结构化摘要与最终摘要生成
- Faithfulness 启发式校验
- 多轮 Agent 决策循环（动态工具、重试、澄清、长期记忆）
- 医学模型接入（OpenAI-compatible API / 本地 HF）
- LoRA + MI 微调流程（实体先验）

## 1. 环境安装

```bash
pip install -r requirements.txt
```

建议 Python `3.10+`。

## 2. 快速运行

### 2.1 基础流水线

```bash
python -m src.main \
  --input data/samples/sample_paper.txt \
  --output data/outputs/summary.json
```

### 2.2 Agent 决策循环

```python
from src.agent.controller import AgentController
from src.pipeline import SummarizationPipeline

controller = AgentController(pipeline=SummarizationPipeline())
state = controller.run(
    input_path="data/samples/sample_paper.txt",
    output_path="data/outputs/my_agent_summary.json",
    user_request="生成学术风格摘要并输出JSON"
)
print(state.final_summary)
```

## 3. 医学模型接入

### 3.1 API 模型（OpenAI-compatible）

**方式一：`.env` 文件（推荐）**

```bash
cp .env.example .env
# 编辑 .env 填入 SMART_LLM__API_KEY
```

**方式二：命令行 export**

```bash
export SMART_LLM__USE_MOCK=false
export SMART_LLM__PROVIDER=openai_compatible
export SMART_LLM__API_KEY="your_key"
export SMART_LLM__BASE_URL="https://your-endpoint/v1"
export SMART_LLM__MODEL_NAME="your-med-model"
```

### 3.2 本地 HF 模型

`config/default.yaml` 示例：

```yaml
llm:
  use_mock: false
  provider: hf_local
  model: your-local-med-model
  local_device_map: auto
  local_torch_dtype: auto
```

## 4. 对话澄清与恢复

当 `state.needs_clarification=True` 时，读取 `state.clarification_request` 并回填：

```python
state = controller.run(
    output_path="data/outputs/my_agent_summary.json",
    clarification_answers={
        "input_path": "data/samples/sample_paper.txt",
        "user_request": "生成200词学术摘要并输出JSON"
    }
)
```

## 5. 微调（LoRA + MI）

### 5.1 先做实体预处理（默认只标训练集）

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

- 默认只给 `train` 添加 `entity_text`
- `eval` 默认不标注实体，不影响验证/测试分布

### 5.2 训练（三层框架）

```bash
python finetune/train_lora_mi.py \
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

### 5.3 推理

```bash
python finetune/infer_seq2seq.py \
  --model_path data/outputs/ft_lora_mi \
  --input_file data/samples/sample_paper.txt
```

## 6. 测试

```bash
pytest -q
```

## 7. 配置入口

主要配置文件：`config/default.yaml`  
关键项：

- `llm.*`
- `extractor.*`
- `agent.*`

---

完整构建流程与细节说明见：

- [instruction.md](instruction.md) 详细命令与配置
- [roadmap.md](roadmap.md) 项目架构与代码原理
- [docs/three_layer_mi_framework.md](docs/three_layer_mi_framework.md) 三层 MI 对齐框架详解
