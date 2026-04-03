from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate summary output JSON.")
    parser.add_argument(
        "--output",
        default="data/outputs/summary.json",
        help="Path to the summary output JSON file.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = Path(args.output)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    section_count = len(payload.get("sections", []))
    faithfulness = payload.get("verification", {}).get("faithfulness_score", 0.0)
    unsupported_count = len(payload.get("verification", {}).get("unsupported_claims", []))

    print("=== Evaluation ===")
    print(f"Sections parsed: {section_count}")
    print(f"Faithfulness score: {faithfulness}")
    print(f"Unsupported claims: {unsupported_count}")


if __name__ == "__main__":
    main()
