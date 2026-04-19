# Method（与当前代码一致版）

## 1. 任务定义

目标：对科研/医学论文进行自动摘要，输出结构化信息与最终摘要，并进行基础忠实度校验。

当前系统由两部分组成：

1. 推理系统（Pipeline + Agent）
2. 微调系统（LoRA + MI）

---

## 2. 推理系统方法

### 2.1 流水线（`src/pipeline.py`）

执行顺序：

1. 文档加载
2. 章节切分
3. 关键信息抽取
4. 结构化摘要生成
5. 最终摘要生成（可选 LLM 重写）
6. Faithfulness 启发式校验
7. 结果写出 JSON

### 2.2 关键信息抽取（`src/extractor/`）

实现为“角色标注 + 选句优化”：

- 角色标注：`CRF -> HMM -> heuristic` 逐级回退
- 选句：ILP（词预算、角色覆盖、冗余惩罚）  
  若 ILP 不可用或失败，回退贪心策略

结构化角色：

- objective
- methods
- results
- limitations

### 2.3 Agent 决策循环（`src/agent/`）

采用 `plan -> act -> review -> retry/finish`：

- Planner：按状态动态选工具
- Tools：执行 load/split/extract/verify/write 等动作
- Reviewer：根据异常、faithfulness、unsupported_claims 决定是否重试
- Dialogue：输入缺失时返回结构化澄清问题
- Memory：将历史运行写入 JSONL 并支持检索策略建议

---

## 3. 微调系统方法（LoRA + MI）

实现脚本：`finetune/train_lora_mi.py`

### 3.1 LoRA

在 Seq2Seq 基础模型上注入 LoRA（默认 target modules：`q,v`），仅训练低秩增量参数。

关键参数（默认）：

- `r=16`
- `alpha=32`
- `dropout=0.1`

### 3.2 损失函数（三层 MI 对齐框架）

当前实现将单一的余弦对齐升级为**三层信息对齐框架**：

$$
L_{total} = L_{mle} + \lambda_{node} L_{node} + \lambda_{link} L_{link} + \lambda_{network} L_{network}
$$

- `L_mle`：标准 seq2seq 监督损失
- `L_node`：**节点层** — 实体类型级 InfoNCE 对比学习 + 缺失实体类型惩罚  
  对每种医学实体类型（如 `CHEMICAL`、`GENE`），分别计算源文与摘要的实体平均 embedding，用 InfoNCE 拉近正样本对；若某类型在源文存在但摘要缺失，施加固定惩罚。
- `L_link`：**链路层** — 共现实体的 TransE 几何约束  
  将原文中字符距离小于窗口的实体视为共现，对每对共现类型在摘要中施加 `v_{type_i} + r \approx v_{type_j}` 的 TransE 约束。关系向量 `r` 为可学习参数。
- `L_network`：**网络层** — 实体共现图的谱嵌入对齐  
  构建实体共现无向图，计算归一化图拉普拉斯的谱特征（前 `k=8` 个特征向量），经可学习投影后与 decoder 最终 hidden state 做 MSE 对齐。

三层损失均只作用于训练集，验证/测试阶段完全关闭。

### 3.3 医学实体先验（预处理阶段）

实现要求：

- **仅训练集标注实体**（含摘要端实体）
- 验证/测试集默认不标注实体，不改变评估分布
- 预处理同时生成源文与摘要的实体信息

预处理输出字段：

| 字段 | 说明 |
|------|------|
| `entity_text` | 源文实体文本（`;` 分隔） |
| `entity_types` | 源文实体类型（`;` 分隔） |
| `entity_spans` | 源文实体字符位置（JSON `[[start,end],...]`） |
| `summary_entities` | 摘要实体文本（`;` 分隔） |
| `summary_entity_types` | 摘要实体类型（`;` 分隔） |
| `summary_entity_spans` | 摘要实体字符位置（JSON） |

流程：

1. 运行 `data/preprocess_entities.py`（使用 `en_ner_bionlp13cg_md`）
2. 运行 `train_lora_mi.py --use_entity_prior [--use_link_layer] [--use_network_layer]`

训练阶段不再在线跑 NER。

---

## 4. 已实现与未实现边界

### 已实现

- CRF/HMM + ILP 抽取
- Agent 多轮决策、重试、澄清、长期记忆
- 医学模型接入（API / HF 本地）
- LoRA + MI 训练
- 训练集实体预处理（`en_ner_bionlp13cg_md`）

### 未实现（可作为后续工作）

- 开放域关系抽取替代共现窗口
- 医学实体编码器 `phi(.)` 独立预训练与对齐
- 细粒度人工事实一致性评测管线

---

## 5. 复现实验最小命令

### 5.1 预处理（仅训练集实体）

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

### 5.2 训练（LoRA + MI）

```bash
python finetune/train_lora_mi.py \
  --train_file data/samples/train_ner.jsonl \
  --eval_file data/samples/dev_ner.jsonl \
  --output_dir data/outputs/ft_lora_mi \
  --use_entity_prior \
  --entity_column entity_text \
  --lambda_node 0.1
```

### 5.3 推理（Agent）

```python
from src.agent.controller import AgentController
from src.pipeline import SummarizationPipeline

controller = AgentController(pipeline=SummarizationPipeline())
state = controller.run(
    input_path="data/samples/sample_paper.txt",
    output_path="data/outputs/agent_summary.json",
    user_request="生成学术风格摘要并输出JSON"
)
print(state.final_summary)
```
