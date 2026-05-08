"""
使用纯 LoRA 微调后的 Flan-T5 模型生成医学摘要
加载 lora_baseline 训练得到的 LoRA 权重进行推理
"""

import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ==================== 1. 配置路径 ====================
# 基础模型名称（必须与训练时一致）
base_model_name = "google/flan-t5-large"

# LoRA 权重路径（训练时保存的 output_dir）
lora_weights_path = "/root/autodl-tmp/8307_CourseProject/outputs/lora_baseline"

print(f"基础模型: {base_model_name}")
print(f"LoRA 权重路径: {lora_weights_path}")

# ==================== 2. 加载模型和分词器 ====================
print("正在加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,  # 与训练时保持一致（使用了 --bf16）
    device_map="auto"            # 自动分配设备
)
lora_weights_path = "/root/autodl-tmp/8307_CourseProject/outputs/lora_baseline"
print("正在加载 LoRA 权重...")
model = PeftModel.from_pretrained(base_model, lora_weights_path)

model.eval()
print(f"模型加载完成！可训练参数: {model.num_parameters(only_trainable=True):,}")

# ==================== 3. 读取测试集 ====================
test_file_path = "/root/autodl-fs/test_ner_clean.jsonl"

print(f"正在读取测试集: {test_file_path}")
test_data = []
with open(test_file_path, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line.strip()))

print(f"测试集共 {len(test_data)} 条数据")

# ==================== 4. 生成摘要 ====================
def generate_summary(article_text, max_input_length=1024, max_output_length=256):
    """
    使用微调后的 T5 模型生成摘要
    """
    input_text = f"summarize: {article_text}"
    
    inputs = tokenizer(
        input_text, 
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],           # 明确指定参数名
            attention_mask=inputs["attention_mask"], # 也传入 attention_mask
            max_length=max_output_length,
            min_length=30,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# ==================== 5. 生成摘要 ====================
print("\n开始生成摘要...")
results = []

# 设置测试模式：True=只跑前100条快速验证，False=跑全部数据
test_mode = True
sample_limit = 100 if test_mode else len(test_data)

for i, item in enumerate(tqdm(test_data[:sample_limit], desc="生成进度")):
    article = item.get("article") or item.get("text") or item.get("source")
    
    if article is None:
        print(f"警告: 第 {i} 条数据没有找到文章字段，跳过")
        continue
    
    summary = generate_summary(article)
    
    results.append({
        "index": i,
        "original_article": article[:200] + "..." if len(article) > 200 else article,
        "generated_summary": summary,
        "reference_abstract": item.get("abstract") or item.get("target")
    })

# ==================== 6. 保存结果 ====================
output_file = "generated_summaries_lora_baseline.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n生成完成！共生成 {len(results)} 条摘要")
print(f"结果已保存到: {output_file}")

# 打印一个示例
if results:
    print("\n===== 示例摘要 =====")
    print(f"原文片段: {results[0]['original_article']}")
    print(f"生成摘要: {results[0]['generated_summary']}")
    if results[0].get('reference_abstract'):
        print(f"参考摘要: {results[0]['reference_abstract'][:200]}...")
import json
from evaluate import load

print("加载评估指标...")
rouge_metric = load("rouge")

# 读取之前生成的摘要结果
results_file = "generated_summaries_lora_baseline.json"

with open(results_file, "r", encoding="utf-8") as f:
    results = json.load(f)

# 提取预测摘要和参考答案
predictions = [item["generated_summary"] for item in results]
references = [item["reference_abstract"] for item in results]

print(f"共加载 {len(predictions)} 条数据，开始计算 ROUGE 指标...")

# 计算 ROUGE 指标
rouge_result = rouge_metric.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True
)

# 只输出 ROUGE 结果
print("\n" + "="*50)
print("📊 评估结果 (ROUGE)")
print("="*50)
print(f"ROUGE-1:     {rouge_result['rouge1']:.4f}")
print(f"ROUGE-2:     {rouge_result['rouge2']:.4f}")
print(f"ROUGE-L:     {rouge_result['rougeL']:.4f}")
print("="*50)
print("注: BERTScore 因网络连接问题未能计算。")

# 保存结果
metrics = {
    "rouge1": round(rouge_result["rouge1"], 4),
    "rouge2": round(rouge_result["rouge2"], 4),
    "rougeL": round(rouge_result["rougeL"], 4),
    "bertscore_f1": None,
    "note": "BERTScore skipped due to network connection error to huggingface.co",
    "num_samples": len(predictions)
}

with open("evaluation_metrics.json_lora_baseline", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\nROUGE 指标已保存到 evaluation_metrics_lora_baseline.json")
