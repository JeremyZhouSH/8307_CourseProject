from __future__ import annotations

import argparse

from src.pipeline import SummarizationPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the summarization pipeline.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--prompts", default="config/prompts.yaml", help="Path to prompts YAML")
    parser.add_argument("--input", dest="input_path", default=None, help="Override input paper path")
    parser.add_argument("--output", dest="output_path", default=None, help="Override output JSON path")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = SummarizationPipeline(config_path=args.config, prompts_path=args.prompts)
    state = pipeline.run(input_path=args.input_path, output_path=args.output_path)

    print("=== Final Academic Summary ===")
    print(state.final_summary)
    print()
    print(f"Output written to: {state.output_path}")
    print(f"Faithfulness score: {state.verification.get('faithfulness_score', 0.0)}")


if __name__ == "__main__":
    main()
