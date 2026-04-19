# 三层互信息对齐框架（Three-Layer MI Alignment Framework）

## 1. 背景与动机

### 1.1 传统 Seq2Seq 摘要的问题

标准 seq2seq 训练仅优化条件概率：

```
max log P(summary | article)
```

这种单一监督信号存在三个层面的信息损失：

| 层面 | 问题 | 具体表现 |
|------|------|---------|
| **微观（术语）** | 实体遗漏 | 医学术语（如 `COX-2`、`Aspirin`）未出现在摘要中 |
| **中观（关系）** | 关系颠倒 | "A 抑制 B" 被写成 "B 抑制 A" |
| **宏观（结构）** | 全局失真 | 摘要未能反映原文的论证结构 |

### 1.2 从"余弦对齐"到"三层约束"

早期方案使用简单的余弦对齐：

```
L_mi = 1 - cos(h_source, h_target)
```

这只能保证"源文和摘要的整体向量方向相近"，无法解决上述三层问题。因此，我们将其扩展为**分层信息对齐框架**：

```
L_total = L_mle + λ_node·L_node + λ_link·L_link + λ_network·L_network
```

---

## 2. 框架总览

```
┌─────────────────────────────────────────────────────────────┐
│                    三层信息对齐框架                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   节点层 (Node Layer)                                        │
│   ├── 粒度：实体类型级                                         │
│   ├── 目标：关键术语不丢失                                     | 
│   ├── 机制：InfoNCE 对比学习 + 缺失惩罚                         |
│   └── 数学：L_node = Σ_t InfoNCE(h_src(t), h_sum(t))         │
│                                                             │
│   链路层 (Link Layer)                                        │
│   ├── 粒度：实体关系对                                         │
│   ├── 目标：共现关系不颠倒                                      │
│   ├── 机制：TransE 几何约束                                    │
│   └── 数学：L_link = MSE(v_i + r, v_j)                       │
│                                                             │
│   网络层 (Network Layer)                                     │
│   ├── 粒度：全图结构                                           │
│   ├── 目标：全局拓扑一致                                       │
│   ├── 机制：谱图嵌入 + 投影对齐                                 │
│   └── 数学：L_network = MSE(W·spectral(G), h_decoder)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

三层损失仅在**训练阶段**生效，验证/测试阶段完全关闭。

---

## 3. 节点层（Node Layer）

### 3.1 核心思想

按**实体类型**分组对比，而非对所有实体做全局平均。这样"DISEASE 实体要对上 DISEASE 实体，CHEMICAL 要对上 CHEMICAL"。

### 3.2 数学定义

对于每种实体类型 $t$（如 CHEMICAL, GENE）：

$$
h_{src}(t) = \frac{1}{|E_{src}^t|} \sum_{e \in E_{src}^t} \text{Embed}(e)
$$

$$
h_{sum}(t) = \frac{1}{|E_{sum}^t|} \sum_{e \in E_{sum}^t} \text{Embed}(e)
$$

**InfoNCE 损失**（使用 batch 内负样本）：

$$
\mathcal{L}_{\text{InfoNCE}}(a, p) = -\log \frac{\exp(\text{sim}(a, p)/\tau)}{\sum_{i} \exp(\text{sim}(a, n_i)/\tau)}
$$

其中 $a$ = anchor（源端类型均值），$p$ = positive（摘要端同类型均值），$n_i$ = batch 内其他样本的摘要表示（负样本）。

**缺失实体惩罚**：

若类型 $t$ 在源端存在但摘要端缺失：

$$
\mathcal{L}_{\text{node}} += \alpha_{\text{missing}}
$$

### 3.3 代码实现

```python
# finetune/mi_layers.py

class NodeLayerLoss(nn.Module):
    def forward(self, src_entity_emb, src_entity_mask, src_type_lists,
                sum_entity_emb, sum_entity_mask, sum_type_lists):
        
        for b in range(B):
            # 按类型分组
            src_by_type = group_by_type(src_emb, src_types)
            sum_by_type = group_by_type(sum_emb, sum_types)
            
            for t, src_embs in src_by_type.items():
                src_mean = torch.stack(src_embs).mean(0)
                if t in sum_by_type:
                    sum_mean = torch.stack(sum_by_type[t]).mean(0)
                    loss += info_nce(src_mean, sum_mean)
                else:
                    loss += self.missing_penalty  # 摘要遗漏该类型
```

**关键设计**：
- 使用模型共享的 `get_input_embeddings()` 获取实体 embedding
- 每个实体单独 tokenize（非拼接），保留边界信息
- InfoNCE 双向损失（anchor→positive 和 positive→anchor）

---

## 4. 链路层（Link Layer）

### 4.1 核心思想

医学文献中，共现实体往往存在语义关联（如 `Aspirin` 与 `COX-2`）。我们约束：**如果原文中两类实体共现，摘要中它们的关系向量应满足几何一致性**。

### 4.2 简化策略：共现代替开放域关系抽取

不抽取显式三元组（如 `<Aspirin, inhibits, COX-2>`），而是使用**字符距离窗口**定义共现：

```
共现定义：两个实体的字符距离 < cooccurrence_window (默认 200)
```

这样做的好处：
- 避免开放域关系抽取的错误传播
- 无需预训练关系分类器
- 计算简单，训练稳定

### 4.3 TransE 几何约束

关系类型不显式区分，用一个**共享的可学习向量** $r \in \mathbb{R}^d$ 表示"共现关系"。

对于共现类型对 $(t_i, t_j)$：

$$
v_{sum}(t_i) + r \approx v_{sum}(t_j)
$$

**损失**：

$$
\mathcal{L}_{\text{link}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \| v_{sum}(t_i) + r - v_{sum}(t_j) \|^2
$$

其中 $\mathcal{P}$ 是源文中的共现类型对集合。

### 4.4 代码实现

```python
# finetune/mi_layers.py

class LinkLayerLoss(nn.Module):
    def __init__(self, hidden_dim, cooccurrence_window=200):
        self.relation_vector = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
    def forward(self, ...):
        # 1. 找出源文中的共现对（基于字符距离窗口）
        pairs = _build_cooccurrence_pairs(src_spans, window=200)
        
        # 2. 按类型聚合实体 embedding
        src_type_mean = {t: mean(emb for emb in src_embs if type==t)}
        sum_type_mean = {t: mean(emb for emb in sum_embs if type==t)}
        
        # 3. 对共现类型对施加 TransE
        for (i, j) in pairs:
            t_i, t_j = src_types[i], src_types[j]
            if t_i in sum_type_mean and t_j in sum_type_mean:
                predicted = sum_type_mean[t_i] + self.relation_vector
                loss += F.mse_loss(predicted, sum_type_mean[t_j])
```

**关键设计**：
- $r$ 是可学习参数，无需预训练
- 类型级聚合（非实体级），避免实体数量不匹配问题
- 若摘要中不存在对应类型对，不施加约束（避免过度惩罚）

---

## 5. 网络层（Network Layer）

### 5.1 核心思想

实体共现图的整体结构应被 decoder 的隐状态所"记住"。如果源文有一个紧密的实体关系网络，摘要生成器的最终状态应能重构该网络的结构特征。

### 5.2 谱图嵌入

**步骤 1：构建共现无向图**

```
节点：实体
边：共现关系（无向，基于字符距离窗口）
```

**步骤 2：归一化图拉普拉斯**

$$
L = I - D^{-1/2} A D^{-1/2}
$$

其中 $A$ 为邻接矩阵，$D$ 为度矩阵。

**步骤 3：特征分解**

$$
L = U \Lambda U^T
$$

取前 $k$ 个特征向量（跳过 $\lambda=0$ 对应的全局分量）：

$$
g = \frac{1}{N} \sum_{i=1}^{N} u_i^{(1:k)} \in \mathbb{R}^k
$$

### 5.3 投影对齐

通过可学习投影矩阵 $W_{proj} \in \mathbb{R}^{d \times k}$ 将谱特征映射到 decoder hidden 维度：

$$
\mathcal{L}_{\text{network}} = \| W_{proj} \cdot g - h_{\text{decoder}}^{\text{final}} \|^2
$$

### 5.4 代码实现

```python
# finetune/mi_layers.py

def spectral_embedding(adjacency, k=8):
    # 度矩阵
    degree = adjacency.sum(dim=-1)
    D_inv_sqrt = torch.diag(torch.rsqrt(degree.clamp_min(1.0)))
    
    # 归一化拉普拉斯
    I = torch.eye(N)
    L = I - D_inv_sqrt @ adjacency @ D_inv_sqrt
    
    # 特征分解
    _, eigvecs = torch.linalg.eigh(L)
    
    # 取前 k 个特征向量，跳过第一个（全局分量）
    vectors = eigvecs[:, 1:k+1]
    return vectors.mean(dim=0)  # [k]

class NetworkLayerLoss(nn.Module):
    def __init__(self, k=8, hidden_dim=512):
        self.projection = nn.Linear(k, hidden_dim)
        
    def forward(self, src_entity_emb, src_entity_mask, src_span_lists, decoder_final_hidden):
        # 构建共现邻接矩阵
        adj = build_cooccurrence_adjacency(src_spans)
        
        # 谱嵌入
        graph_vec = spectral_embedding(adj, self.k)  # [k]
        
        # 投影并对齐
        projected = self.projection(graph_vec)         # [hidden_dim]
        return F.mse_loss(projected, decoder_final_hidden[b])
```

**关键设计**：
- 使用**谱方法**而非 GCN，避免引入额外训练参数和过平滑问题
- $k=8$ 是超参数，控制保留的图结构信息量
- 实体数 $<3$ 时退化为节点层损失（图太小无意义）

---

## 6. 总体损失与训练配置

### 6.1 总体损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mle}} + \lambda_{\text{node}} \mathcal{L}_{\text{node}} + \lambda_{\text{link}} \mathcal{L}_{\text{link}} + \lambda_{\text{network}} \mathcal{L}_{\text{network}}
$$

### 6.2 推荐配置

| 配置 | 仅节点层 | 节点+链路 | 完整三层 |
|------|---------|----------|---------|
| `--use_entity_prior` | ✅ | ✅ | ✅ |
| `--use_link_layer` | ❌ | ✅ | ✅ |
| `--use_network_layer` | ❌ | ❌ | ✅ |
| `--lambda_node` | 0.1 | 0.1 | 0.1 |
| `--lambda_link` | — | 0.05 | 0.05 |
| `--lambda_network` | — | — | 0.03 |
| `--cooccurrence_window` | — | 200 | 200 |
| `--missing_entity_penalty` | 0.5 | 0.5 | 0.5 |

### 6.3 训练命令

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
  --lambda_network 0.03 \
  --cooccurrence_window 200 \
  --missing_entity_penalty 0.5
```

---

## 7. 与单层余弦对齐的对比

| 维度 | 单层余弦对齐 | 三层框架 |
|------|------------|---------|
| **表示粒度** | 全局平均（所有实体压成一个向量） | 类型级分组（按 CHEMICAL/GENE 等分别对比） |
| **负样本** | 无（仅拉近正样本） | 有（InfoNCE batch 内负样本） |
| **关系约束** | 无 | TransE 几何约束 |
| **结构约束** | 无 | 谱图嵌入对齐 |
| **缺失惩罚** | 无 | 有（摘要遗漏实体类型时惩罚） |
| **理论保证** | 无（不是任何 MI 下界） | InfoNCE 是互信息的变分下界 |

---

## 8. 数据流

```
预处理阶段
  article + abstract
      ↓  spacy NER (en_ner_bionlp13cg_md)
  entity_text / entity_types / entity_spans  （源文）
  summary_entities / summary_entity_types / summary_entity_spans  （摘要）
      ↓ 保存为 JSONL
  train_ner.jsonl  →  训练集（含实体标注）
  val_ner.jsonl    →  验证集（无实体标注）

训练阶段
  batch 输入
      ├── article → tokenizer → input_ids
      ├── abstract → tokenizer → labels
      ├── entities → per-entity tokenizer → entity_input_ids [B, N, T]
      ├── entity_types → List[List[str]]
      └── entity_spans → List[List[List[int]]]
      
  forward
      ├── L_mle（标准 seq2seq 损失）
      ├── L_node（InfoNCE + 缺失惩罚）
      ├── L_link（TransE，可选）
      └── L_network（谱图对齐，可选）
```

---

## 9. 局限与扩展方向

| 当前简化 | 潜在深化 |
|---------|---------|
| 共现窗口定义关系 | 开放域关系抽取（BioRE / OpenIE） |
| 谱方法（无参） | GCN / GAT / GraphSAGE |
| 共享关系向量 $r$ | 类型特定的关系嵌入 |
| 词汇级 faithfulness | 实体级 F1 / 关系一致性评测 |
| 英文医学文献 | 多语言医学 NER 适配 |

---

## 10. 参考文献

1. Oord, A. et al. "Representation Learning with Contrastive Predictive Coding." arXiv:1807.03748.
2. Bordes, A. et al. "Translating Embeddings for Modeling Multi-relational Data." NeurIPS 2013.
3. Von Luxburg, U. "A Tutorial on Spectral Clustering." Statistics and Computing, 2007.
4. Hu, Z. et al. "Entity-Level Cross-Attention for Abstractive Summarization." ACL 2022.
