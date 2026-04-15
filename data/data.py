import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. 设定环境变量与加载数据集 (此处抽取 train 集前 5000 条作为样本空间)
hf_token = os.getenv("HF_TOKEN")
ds = load_dataset("ccdv/pubmed-summarization", "document", split="train[:5000]", token=hf_token)

# 2. 加载目标分词器
# 必须使用你拟采用的模型的分词器，因为不同模型的切词规则 (BPE, WordPiece) 会导致 Token 数量差异
tokenizer_name = "allenai/led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# 3. 定义 Token 计算逻辑
def compute_token_lengths(example):
    # truncation=False 确保统计到完整的绝对长度
    return {
        "article_len": len(tokenizer(example["article"], truncation=False)["input_ids"]),
        "abstract_len": len(tokenizer(example["abstract"], truncation=False)["input_ids"])
    }

# 4. 并行映射计算 (num_proc 可根据你的 CPU 核心数调整)
print("开始计算 Token 长度...")
ds_with_lengths = ds.map(compute_token_lengths)

# 5. 提取数组并计算经验分布特征
article_lengths = np.array(ds_with_lengths["article_len"])
abstract_lengths = np.array(ds_with_lengths["abstract_len"])

def print_statistics(name, lengths):
    print(f"\n--- {name} 统计特征 ---")
    print(f"样本量 (N): {len(lengths)}")
    print(f"均值 (Mean): {np.mean(lengths):.2f}")
    print(f"标准差 (Std Dev): {np.std(lengths):.2f}")
    print(f"中位数 (Median): {np.median(lengths):.2f}")
    print(f"90% 分位数 (90th Percentile): {np.percentile(lengths, 90):.2f}")
    print(f"95% 分位数 (95th Percentile): {np.percentile(lengths, 95):.2f}")
    print(f"99% 分位数 (99th Percentile): {np.percentile(lengths, 99):.2f}")
    print(f"最大值 (Max): {np.max(lengths)}")

print_statistics("原论文 (Article)", article_lengths)
print_statistics("摘要 (Abstract)", abstract_lengths)