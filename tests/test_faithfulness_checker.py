from __future__ import annotations

from src.verifier.faithfulness_checker import FaithfulnessChecker


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_faithfulness_checker_reports_multi_signal_scores() -> None:
    checker = FaithfulnessChecker()
    document = "The trial enrolled 120 patients in 2024. Accuracy reached 92.5 percent."
    summary = "The study enrolled 120 patients in 2024 and reported 85 percent accuracy."

    result = checker.check(
        final_summary=summary,
        key_info={"results": ["Accuracy reached 92.5 percent."]},
        document_text=document,
    )

    assert 0.0 <= result["faithfulness_score"] <= 1.0
    assert "subscores" in result
    assert result["subscores"]["number_consistency"] < 1.0
    assert "85" in result["unsupported_numbers"]


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_faithfulness_checker_empty_summary_is_zero() -> None:
    checker = FaithfulnessChecker()
    result = checker.check(final_summary="", key_info={}, document_text="Some document text.")

    assert result["faithfulness_score"] == 0.0


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_faithfulness_checker_no_numbers_defaults_to_full_number_consistency() -> None:
    checker = FaithfulnessChecker()
    result = checker.check(
        final_summary="The model improves stability and quality.",
        key_info={},
        document_text="This paper reports improved stability and quality.",
    )

    assert result["subscores"]["number_consistency"] == 1.0
