from src.parser.section_splitter import SectionSplitter


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_section_splitter_detects_common_sections() -> None:
    text = """
INTRODUCTION
This section introduces the task.

METHODS
This section explains the approach.

RESULTS
This section reports findings.
""".strip()

    splitter = SectionSplitter()
    sections = splitter.split(text)
    titles = [section.title for section in sections]

    assert "Introduction" in titles
    assert "Methods" in titles
    assert "Results" in titles


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_section_splitter_fallback_for_plain_text() -> None:
    text = "Single paragraph without explicit headings."

    splitter = SectionSplitter()
    sections = splitter.split(text)

    assert len(sections) == 1
    assert sections[0].title == "Document"
