"""
训练报告可视化生成器。

读取 report/logs/ 下的训练日志，生成可视化图像和报告 Markdown。

用法：
    python scripts/generate_report.py \
        --log_dir report/logs \
        --output_dir report/figures \
        --report_path report/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# 函数作用：构建命令行参数解析器并定义可配置项。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate training report visuals.")
    parser.add_argument("--log_dir", default="report/logs")
    parser.add_argument("--output_dir", default="report/figures")
    parser.add_argument("--report_path", default="report/report.md")
    return parser


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def plot_loss_curves(records: list[dict[str, Any]], output_dir: Path) -> None:
    """绘制损失曲线。"""
    steps = [r["step"] for r in records]
    mle_losses = [r.get("mle_loss", 0.0) for r in records]
    total_losses = [r.get("total_loss", 0.0) for r in records]
    mi_losses = [r.get("total_mi_loss", 0.0) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    # Total Loss
    ax = axes[0, 0]
    ax.plot(steps, total_losses, color="#2E86AB", linewidth=1.2)
    ax.set_title("Total Loss (L_mle + L_mi)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # MLE Loss
    ax = axes[0, 1]
    ax.plot(steps, mle_losses, color="#A23B72", linewidth=1.2)
    ax.set_title("MLE Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # MI Loss
    ax = axes[1, 0]
    ax.plot(steps, mi_losses, color="#F18F01", linewidth=1.2)
    ax.set_title("Total MI Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Layer-wise breakdown (if available)
    ax = axes[1, 1]
    has_layer = False
    for key, color in [
        ("node_loss", "#C73E1D"),
        ("link_loss", "#3B1F2B"),
        ("network_loss", "#2E86AB"),
    ]:
        values = [r.get(key) for r in records]
        if any(v is not None for v in values):
            values = [v if v is not None else 0.0 for v in values]
            ax.plot(steps, values, label=key.replace("_", " ").title(), color=color, linewidth=1.2)
            has_layer = True

    if has_layer:
        ax.set_title("Layer-wise MI Losses")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("Layer-wise MI Losses")
        ax.text(0.5, 0.5, "No layer losses recorded", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    path = output_dir / "loss_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves to {path}")


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def plot_eval_metrics(eval_metrics: dict[str, Any], output_dir: Path) -> None:
    """绘制验证指标柱状图。"""
    metrics = {k.replace("eval_", ""): v for k, v in eval_metrics.items() if k.startswith("eval_")}
    if not metrics:
        print("No eval metrics found, skipping eval chart.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics.keys())
    values = [float(v) for v in metrics.values()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics (Validation Set)", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2 if values else 1.0)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = output_dir / "eval_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved eval metrics chart to {path}")


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def generate_report_markdown(
    config: dict[str, Any],
    summary: dict[str, Any],
    eval_metrics: dict[str, Any],
    report_path: Path,
    figures_dir: Path,
) -> None:
    """生成 Markdown 报告。"""

    lines: list[str] = [
        "# 训练报告：三层 MI 对齐框架",
        "",
        "## 1. 训练配置",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
    ]

    key_params = [
        "model_name", "lora_r", "lora_alpha", "lora_dropout",
        "learning_rate", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "max_input_length", "max_target_length",
        "lambda_node", "lambda_link", "lambda_network",
        "use_entity_prior", "use_link_layer", "use_network_layer",
        "cooccurrence_window", "missing_entity_penalty",
        "train_samples", "eval_samples",
    ]
    for key in key_params:
        if key in config:
            val = config[key]
            if isinstance(val, bool):
                val = "✅" if val else "❌"
            lines.append(f"| {key} | {val} |")

    lines.extend([
        "",
        "## 2. 训练耗时",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
    ])
    if "elapsed_minutes" in summary:
        lines.append(f"| 总耗时 | {summary['elapsed_minutes']:.2f} 分钟 |")
    if "elapsed_seconds" in summary:
        lines.append(f"| 总耗时（秒）| {summary['elapsed_seconds']:.2f} 秒 |")
    if "total_steps" in summary:
        lines.append(f"| 总步数 | {summary['total_steps']} |")
    if "total_epochs" in summary:
        lines.append(f"| 总轮数 | {summary['total_epochs']} |")
    if "best_metric" in summary and summary["best_metric"] is not None:
        lines.append(f"| 最佳验证指标 | {summary['best_metric']:.4f} |")
    if "final_train_loss" in summary and summary["final_train_loss"] is not None:
        lines.append(f"| 最终训练损失 | {summary['final_train_loss']:.4f} |")

    lines.extend([
        "",
        "## 3. 验证指标",
        "",
    ])
    if eval_metrics:
        lines.extend(["| 指标 | 值 |", "|------|-----|"])
        for k, v in eval_metrics.items():
            k_clean = k.replace("eval_", "")
            lines.append(f"| {k_clean} | {float(v):.4f} |")
    else:
        lines.append("暂无验证指标数据。")

    lines.extend([
        "",
        "## 4. 训练过程可视化",
        "",
        "### 4.1 损失曲线",
        "",
        f"![损失曲线]({figures_dir / 'loss_curves.png'})",
        "",
        "上图展示了训练过程中的损失变化：",
        "- **左上**：总损失（MLE + MI 对齐）",
        "- **右上**：MLE 监督损失",
        "- **左下**：总 MI 对齐损失",
        "- **右下**：各层 MI 损失分解（节点层 / 链路层 / 网络层）",
        "",
        "### 4.2 验证指标",
        "",
    ])

    if (figures_dir / "eval_metrics.png").exists():
        lines.extend([
            f"![验证指标]({figures_dir / 'eval_metrics.png'})",
            "",
        ])
    else:
        lines.append("暂无验证指标可视化。\n")

    lines.extend([
        "## 5. 说明",
        "",
        "- 训练日志保存在 `report/logs/training_log.json`",
        "- 训练摘要保存在 `report/logs/training_summary.json`",
        "- 超参数配置保存在 `report/logs/config.json`",
        "- 本报告由 `scripts/generate_report.py` 自动生成",
        "",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {report_path}")


# 函数作用：程序入口，串联参数解析与主执行流程。
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    config = {}
    if (log_dir / "config.json").exists():
        config = load_json(log_dir / "config.json")

    summary = {}
    if (log_dir / "training_summary.json").exists():
        summary = load_json(log_dir / "training_summary.json")

    eval_metrics = {}
    if (log_dir / "eval_metrics.json").exists():
        eval_metrics = load_json(log_dir / "eval_metrics.json")

    records: list[dict[str, Any]] = []
    if (log_dir / "training_log.json").exists():
        records = load_json(log_dir / "training_log.json")

    # Generate figures
    if records:
        plot_loss_curves(records, output_dir)
    else:
        print("No training log found. Run training first.")

    if eval_metrics:
        plot_eval_metrics(eval_metrics, output_dir)

    # Generate markdown report
    generate_report_markdown(config, summary, eval_metrics, report_path, output_dir)

    print("\nReport generation complete!")
    print(f"  Figures: {output_dir}")
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()
