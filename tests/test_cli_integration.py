from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 函数作用：内部辅助逻辑，服务当前类/模块主流程。
def _mock_env() -> dict[str, str]:
    env = os.environ.copy()
    env["SMART_LLM__USE_MOCK"] = "true"
    return env


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_src_main_cli_runs_end_to_end(tmp_path) -> None:
    input_text = """
ABSTRACT
This paper proposes a robust summarization pipeline.

METHODS
We use parser extractor summarizer and verifier modules.

RESULTS
The system generates stable summaries.
""".strip()
    input_file = tmp_path / "paper.txt"
    output_file = tmp_path / "summary.json"
    input_file.write_text(input_text, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=_mock_env(),
    )

    assert result.returncode == 0, result.stderr
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["final_summary"]
    assert "verification" in payload


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_run_demo_script_executes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_demo.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=_mock_env(),
    )

    assert result.returncode == 0, result.stderr
    assert "Demo completed." in result.stdout
