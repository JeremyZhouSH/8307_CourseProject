from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=str(_ENV_PATH))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 让脚本在仓库任意位置执行时都能找到 src 包。
    sys.path.insert(0, str(ROOT))

from src.pipeline import SummarizationPipeline


def main() -> None:
    # demo 使用默认配置跑完整流水线。
    pipeline = SummarizationPipeline()
    state = pipeline.run()

    print("Demo completed.")
    print(f"Input: {state.input_path}")
    print(f"Output: {state.output_path}")
    print(f"Faithfulness score: {state.verification.get('faithfulness_score', 0.0)}")


if __name__ == "__main__":
    main()
