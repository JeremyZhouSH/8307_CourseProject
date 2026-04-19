from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=str(_ENV_PATH))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned seq2seq model.")
    parser.add_argument("--model_path", required=True, help="Path to local model dir.")
    parser.add_argument("--input_file", required=True, help="Path to text file to summarize.")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input_file)
    text = input_path.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=args.max_input_length,
        return_tensors="pt",
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
