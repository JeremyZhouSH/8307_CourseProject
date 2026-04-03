from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import SummarizationPipeline


def main() -> None:
    pipeline = SummarizationPipeline()
    state = pipeline.run()

    print("Demo completed.")
    print(f"Input: {state.input_path}")
    print(f"Output: {state.output_path}")
    print(f"Faithfulness score: {state.verification.get('faithfulness_score', 0.0)}")


if __name__ == "__main__":
    main()
