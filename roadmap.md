# Project Roadmap: 科研论文自动摘要 Agent

## 1. 项目概述

本项目是一个面向**科研/医学论文**的自动摘要系统，包含两个子系统：

1. **推理系统**：基于规则的流水线 + LLM Agent 决策循环，支持结构化摘要生成与忠实度校验
2. **微调系统**：基于 LoRA 的参数高效微调，引入**三层互信息对齐框架**（Node-Link-Network）提升摘要的实体忠实度

---

## 2. 系统架构全景

```
┌─────────────────────────────────────────────────────────────┐
│                        推理系统                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Parser     │→│  Extractor   │→│   Summarizer     │  │
│  │ (文档切分)    │  │ (CRF/HMM+ILP)│  │ (结构化摘要)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         ↓                  ↓                    ↓           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Agent 决策循环 (Plan-Act-Review)          │  │
│  │  Planner → Tools → Reviewer → (Retry / Finish)        │  │
│  └──────────────────────────────────────────────────────┘  │
│                              ↓                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Verifier (Faithfulness Check)             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        微调系统                               │
│  ┌──────────────┐  ┌─────────────────────────────────────┐  │
│  │  数据预处理   │→│         LoRA + 三层 MI 对齐           │  │
│  │ (NER+实体信息)│  │  Node(InfoNCE) → Link(TransE)      │  │
│  └──────────────┘  │  → Network(谱图对齐)                 │  │
│                    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 推理系统详解

### 3.1 主流水线 (`src/pipeline.py`)

```python
SummarizationPipeline.run(input_path, output_path):
    1. 文档加载      → DocumentLoader.read()
    2. 章节切分      → SectionSplitter.split()
    3. 关键信息抽取   → KeyInfoExtractor.extract()
    4. 结构化摘要    → StructuredSummarizer.summarize()
    5. 最终摘要      → FinalSummarizer.generate() [可选 LLM 重写]
    6. 忠实度校验    → FaithfulnessChecker.check()
    7. 结果输出      → JSON 文件
```

### 3.2 文档解析 (`src/parser/`)

**`document_loader.py`**
- 功能：读取纯文本文件（UTF-8），支持 `.txt` 格式
- 输出：原始文本字符串

**`section_splitter.py`**
- 功能：基于规则 + 启发式的章节切分
- 切分逻辑：
  - 识别标题行（如 `ABSTRACT`, `1 INTRODUCTION`, `METHODS`）
  - 标题行特征：不以标点结尾、大写开头、长度适中
  - 按标题将文档切分为 `Section(title, content)` 列表
- 回退策略：若无法识别标题，按固定段落数均匀分段

### 3.3 关键信息抽取 (`src/extractor/`)

核心创新：**角色标注 + 优化选句** 的两阶段架构

**`role_tagger_crf.py`**
```
输入：章节列表 (List[Section])
输出：带角色标签的句子列表 (List[TaggedSentence])

角色标签集：{objective, methods, results, limitations, other}

处理流程：
  CRF → HMM → heuristic 逐级回退

CRF 特征工程：
  - 句子位置（全局位置、章节内位置）
  - 关键词命中（如 "propose"→objective, "accuracy"→results）
  - 章节标题提示（如 Methods 章节 → methods 概率提升）
  - 上下文特征（前后句的章节提示）

弱监督训练：
  - 利用章节标题做弱标签（Introduction→objective）
  - 在单篇文档上即时拟合 CRF（无需预训练语料）
```

**`ilp_sentence_selector.py`**
```
输入：带角色标签的句子候选列表
输出：满足约束的最优句子子集

ILP 模型：
  变量：x_i ∈ {0,1} 表示是否选中句子 i
  
  目标函数：
    max Σ score_i · x_i - λ Σ sim_ij · z_ij
    
    （最大化信息覆盖 - 冗余惩罚）
  
  约束：
    1. 词预算：Σ len_i · x_i ≤ word_budget (默认 200)
    2. 角色覆盖：每类角色至少选 min_role_coverage 句
    3. 冗余线性化：z_ij ≤ x_i, z_ij ≤ x_j, z_ij ≥ x_i + x_j - 1

求解器：PuLP (CBC)
回退：若 ILP 失败，使用贪心策略（按 score/word_count 排序）
```

**`key_info_extractor.py`**
- 统一接口：组合 `role_tagger_crf.py` + `ilp_sentence_selector.py`
- 自动策略回退：CRF 未安装 → HMM；PuLP 未安装 → 贪心

### 3.4 摘要生成 (`src/summarizer/`)

**`structured_summarizer.py`**
- 按角色（objective/methods/results/limitations）组织抽取的句子
- 输出结构化 JSON：`{"objective": "...", "methods": "...", ...}`

**`final_summarizer.py`**
- 可选 LLM 重写：将结构化摘要整合为连贯段落
- 若 LLM 不可用，直接拼接结构化内容

### 3.5 忠实度校验 (`src/verifier/faithfulness_checker.py`)

```python
检查维度：
  1. 实体重叠率：摘要中的术语有多少出现在原文中
  2. 数字一致性：摘要中的数字是否能在原文找到支持
  3. 句子可溯源性：摘要句子与原文最佳匹配句的相似度

输出：faithfulness_score ∈ [0, 1]
```

### 3.6 Agent 决策循环 (`src/agent/`)

```
状态机：plan → act → review → (retry | finish)

Planner (planner.py):
  根据当前状态选择下一步工具
  工具集：load, split, extract, summarize, verify, write

Tools (tools.py):
  每个工具执行一个 pipeline 步骤
  工具注册表支持动态扩展

Reviewer (reviewer.py):
  检查异常、faithfulness_score、unsupported_claims
  若质量不达标，触发重试（最多 retry_limit 次）

Memory (memory.py):
  将历史运行写入 JSONL
  支持检索相似任务的历史策略建议

Dialogue (dialogue.py):
  输入缺失时返回结构化澄清问题（question_id, fields）
  用户回填后可继续执行
```

---

## 4. 微调系统详解

### 4.1 标准 Seq2Seq 微调 (`finetune/train_seq2seq.py`)

```
基础模型：T5 / FLAN-T5 (默认 google/flan-t5-small)
训练方式：全参数微调或 LoRA 微调
评估指标：ROUGE-1 / ROUGE-2 / ROUGE-L
数据集：PubMed Summarization (ccdv/pubmed-summarization)
```

### 4.2 LoRA 配置 (`finetune/train_lora_mi.py`)

```python
LoRA 参数：
  r = 16                # 低秩维度
  alpha = 32            # 缩放系数
  dropout = 0.1         # LoRA 层 dropout
  target_modules = ["q", "v"]  # 仅注入 attention 的 query/value

可训练参数占比：~0.1% - 1%（取决于模型大小）
```

### 4.3 三层 MI 对齐框架 (`finetune/mi_layers.py`)

#### 核心动机

传统 seq2seq 仅优化 `P(summary|article)`，容易导致：
- **实体遗漏**：关键医学术语未出现在摘要中
- **关系颠倒**：因果关系表述错误（如 "A 抑制 B" 写成 "B 抑制 A"）
- **全局结构失真**：摘要未能反映原文的论证结构

三层框架从**微观→中观→宏观**逐层约束生成过程：

```
Node Layer   : 微观 — 确保关键术语不丢失
Link Layer   : 中观 — 确保实体关系不颠倒  
Network Layer: 宏观 — 确保全局结构一致
```

#### 4.3.1 节点层 (`NodeLayerLoss`)

```
原理：实体类型级 InfoNCE 对比学习

对于每种实体类型 t（如 CHEMICAL, GENE）：
  h_src(t)  = MeanPool({embedding(e) | e ∈ source, type(e) = t})
  h_sum(t)  = MeanPool({embedding(e) | e ∈ summary, type(e) = t})
  
  L_node += InfoNCE(h_src(t), h_sum(t))    # 拉近同类型表示
  
  if t 存在于 source 但不存在于 summary:
      L_node += missing_penalty              # 惩罚遗漏

InfoNCE 实现：
  使用 batch 内其他样本作为负样本
  sim = anchor · positive^T / temperature
  loss = CrossEntropy(sim, diagonal_labels)  # 对角线为正样本
```

#### 4.3.2 链路层 (`LinkLayerLoss`)

```
原理：TransE 几何约束保持共现关系

共现定义：
  两个实体在原文中的字符距离 < cooccurrence_window (默认200)
  
关系表示：
  不抽开放域三元组，用一个共享的可学习向量 r 表示"共现关系"

约束：
  对于共现实体对 (type_i, type_j)：
      v_sum(type_i) + r ≈ v_sum(type_j)
  
  即：如果原文中 CHEMICAL 和 GENE 共现，
      摘要中这两种类型的平均表示应满足 TransE 几何关系

损失：
  L_link = MSE(v_sum(t_i) + r, v_sum(t_j))
```

#### 4.3.3 网络层 (`NetworkLayerLoss`)

```
原理：全图结构向量与解码器输出的对齐

图构建：
  节点：实体
  边：共现关系（无向，基于字符距离窗口）

谱嵌入：
  1. 构建邻接矩阵 A
  2. 计算归一化图拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
  3. 特征分解 L = U Λ U^T
  4. 取前 k 个特征向量（跳过 λ=0 对应的全局分量）
  5. 对节点维度平均池化 → graph_vec ∈ R^k

对齐：
  projected = W_proj · graph_vec    # 投影到 decoder hidden dim
  L_network = MSE(projected, h_decoder_final)
```

#### 4.3.4 总体损失

```
L_total = L_mle 
        + λ_node · L_node 
        + λ_link · L_link 
        + λ_network · L_network
```

### 4.4 数据预处理 (`data/preprocess_entities.py`)

```
输入：原始论文数据集（article + abstract）
工具：spacy + en_ner_bionlp13cg_md（生物医学 NER 模型）

输出字段（每篇论文）：
  article                 : 原文
  abstract                : 摘要
  entity_text             : 源文实体文本（"; " 分隔）
  entity_types            : 源文实体类型（"; " 分隔）
  entity_spans            : 源文实体字符位置（JSON [[start,end],...]）
  summary_entities        : 摘要实体文本
  summary_entity_types    : 摘要实体类型
  summary_entity_spans    : 摘要实体位置

策略：
  - 仅训练集标注实体（避免评估分布泄漏）
  - 验证/测试集默认不标注（除非加 --annotate_eval）
```

### 4.5 自定义 Collator (`CustomDataCollator`)

```python
问题：DataCollatorForSeq2Seq 无法处理字符串列表（entity types / spans）

解决：继承 DataCollatorForSeq2Seq
  1. 在父类处理前提取字符串字段
  2. 父类处理标准 tensor 字段（padding + batching）
  3. 将字符串字段以 Python list 形式放回 batch

结果：compute_loss 中可直接访问
  batch["entity_type_list"]      → List[List[str]]
  batch["entity_span_list"]      → List[List[List[int]]]
  batch["summary_entity_type_list"] → ...
```

---

## 5. 构建与运行流程

### 5.1 环境准备

```bash
# Python 3.10+ recommended
pip install -r requirements.txt

# 安装 spacy 生物医学 NER 模型
python -m spacy download en_ner_bionlp13cg_md
```

### 5.2 推理系统运行

```bash
# 基础流水线
python -m src.main \
  --input data/samples/sample_paper.txt \
  --output data/outputs/summary.json

# Agent 决策循环
python scripts/run_demo.py
```

### 5.3 微调系统运行

```bash
# Step 1: 预处理实体（仅训练集）
python data/preprocess_entities.py \
  --dataset_name ccdv/pubmed-summarization \
  --dataset_config document \
  --train_split "train[:2000]" \
  --eval_split "validation[:200]" \
  --ner_model en_ner_bionlp13cg_md \
  --output_train data/samples/train_ner.jsonl \
  --output_eval data/samples/dev_ner.jsonl

# Step 2: 训练（三层框架）
python finetune/train_lora_mi.py \
  --train_file data/samples/train_ner.jsonl \
  --eval_file data/samples/dev_ner.jsonl \
  --output_dir data/outputs/ft_lora_mi \
  --use_entity_prior \
  --use_link_layer \
  --use_network_layer \
  --lambda_node 0.1 \
  --lambda_link 0.05 \
  --lambda_network 0.03

# Step 3: 推理
python finetune/infer_seq2seq.py \
  --model_path data/outputs/ft_lora_mi \
  --input_file data/samples/sample_paper.txt
```

### 5.4 测试

```bash
pytest -q
```

---

## 6. 技术栈

| 层级 | 技术/库 |
|------|---------|
| 语言 | Python 3.10+ |
| 深度学习 | PyTorch, Transformers (HuggingFace) |
| 参数高效微调 | PEFT (LoRA) |
| NER | spaCy, scispaCy |
| 序列标注 | sklearn-crfsuite |
| 优化求解 | PuLP (ILP) |
| 数据集 | datasets (HuggingFace) |
| 评估 | evaluate, rouge_score |
| 配置 | PyYAML |
| 测试 | pytest |

---

## 7. 扩展方向

| 方向 | 当前状态 | 可能的深化 |
|------|---------|-----------|
| 关系抽取 | 共现窗口（简化） | 开放域关系抽取（OpenIE / BioRE） |
| 图神经网络 | 谱方法（简化） | GCN / GAT / GraphSAGE |
| 实体编码器 | 共享 embedding | 独立的 biomedical entity encoder φ(·) |
| 评估维度 | ROUGE + 启发式 Faithfulness | 实体级 F1 / 关系一致性 / 人工评测 |
| 模型规模 | FLAN-T5-small | FLAN-T5-base / large / PubMedGPT |
