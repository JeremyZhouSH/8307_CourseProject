"""关键信息抽取组件。"""

from src.extractor.ilp_sentence_selector import ILPSentenceSelector, SentenceCandidate
from src.extractor.key_info_extractor import KeyInfoExtractor
from src.extractor.role_tagger_crf import SentenceRoleTagger, TaggedSentence

__all__ = [
    "ILPSentenceSelector",
    "SentenceCandidate",
    "KeyInfoExtractor",
    "SentenceRoleTagger",
    "TaggedSentence",
]
