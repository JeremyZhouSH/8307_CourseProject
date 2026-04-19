from __future__ import annotations

import subprocess
import sys


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_pipeline_import_has_no_circular_dependency() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import src.pipeline"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
