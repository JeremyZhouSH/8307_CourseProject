from __future__ import annotations

from src.extractor.ilp_sentence_selector import ILPSentenceSelector, SentenceCandidate
from src.extractor.key_info_extractor import KeyInfoExtractor
from src.parser.section_splitter import SectionSplitter


def test_ilp_selector_respects_budget_and_role_coverage() -> None:
    selector = ILPSentenceSelector(
        word_budget=20,
        redundancy_penalty=0.2,
        min_role_coverage={
            "objective": 1,
            "methods": 1,
            "results": 1,
            "limitations": 1,
        },
    )

    candidates = [
        SentenceCandidate(
            sentence_id=0,
            text="This paper aims to improve paper summarization quality.",
            role="objective",
            score=0.9,
            word_count=5,
        ),
        SentenceCandidate(
            sentence_id=1,
            text="We train a modular model with a lightweight pipeline.",
            role="methods",
            score=0.85,
            word_count=5,
        ),
        SentenceCandidate(
            sentence_id=2,
            text="Experiments show stable gains across several benchmark sets.",
            role="results",
            score=0.88,
            word_count=5,
        ),
        SentenceCandidate(
            sentence_id=3,
            text="A key limitation is weaker robustness on rare terms.",
            role="limitations",
            score=0.82,
            word_count=5,
        ),
        SentenceCandidate(
            sentence_id=4,
            text="An ablation confirms that section features are important.",
            role="results",
            score=0.5,
            word_count=5,
        ),
    ]

    selected = selector.select(candidates)
    assert selected

    total_words = sum(candidate.word_count for candidate in selected)
    assert total_words <= 20

    selected_roles = {candidate.role for candidate in selected}
    assert "objective" in selected_roles
    assert "methods" in selected_roles
    assert "results" in selected_roles
    assert "limitations" in selected_roles


def test_key_info_extractor_ilp_strategy_returns_structured_info() -> None:
    text = """
ABSTRACT
This paper aims to build a robust and interpretable scientific summarizer.

METHODS
We design a modular pipeline with sentence role tagging and optimization.

RESULTS
Experiments show improved coherence and reduced redundancy across examples.

LIMITATIONS
The approach may underperform when domain terms are highly ambiguous.
""".strip()

    sections = SectionSplitter().split(text)
    extractor = KeyInfoExtractor(
        extractor_cfg={
            "strategy": "ilp",
            "role_tagger": "hmm",
            "word_budget": 80,
            "max_sentences_per_role": 2,
            "min_sentence_chars": 20,
            "min_role_coverage": {
                "objective": 1,
                "methods": 1,
                "results": 1,
                "limitations": 1,
            },
        }
    )

    info = extractor.extract(sections)

    assert set(info.keys()) == {"objective", "methods", "results", "limitations"}
    assert info["objective"]
    assert info["methods"]
    assert info["results"]
    assert info["limitations"]
    assert all(len(items) <= 2 for items in info.values())
