from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件（如果存在）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=str(_ENV_PATH))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许以 `python -m src.main` 方式从项目根目录导入模块。
    sys.path.insert(0, str(ROOT))

from src.pipeline import SummarizationPipeline


# 函数作用：构建命令行参数解析器并定义可配置项。
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the summarization pipeline.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--prompts", default="config/prompts.yaml", help="Path to prompts YAML")
    parser.add_argument("--input", dest="input_path", default=None, help="Override input paper path")
    parser.add_argument("--output", dest="output_path", default=None, help="Override output JSON path")
    return parser


# 函数作用：程序入口，串联参数解析与主执行流程。
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # CLI 参数仅覆盖输入/输出与配置路径，不改动内部流程。
    pipeline = SummarizationPipeline(config_path=args.config, prompts_path=args.prompts)
    state = pipeline.run(input_path=args.input_path, output_path=args.output_path)

    print("=== Final Academic Summary ===")
    print(state.final_summary)
    print()
    print(f"Output written to: {state.output_path}")
    print(f"Faithfulness score: {state.verification.get('faithfulness_score', 0.0)}")


if __name__ == "__main__":
    main()
