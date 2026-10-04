"""Citation extraction and validation against retrieved context pages."""

import re
from dataclasses import dataclass
from typing import List, Set, Tuple


@dataclass
class Citation:
    """Represents an extracted inline citation from the answer text."""
    raw_text: str
    section_name: str
    page_start: int
    page_end: int
    is_valid: bool = True


@dataclass
class CitationValidationReport:
    citations: List[Citation]
    total_citations: int
    valid_citations: int
    hallucinated_citations: int
    precision: float
    has_citations: bool


class CitationValidator:
    """
    Parses and validates inline citations in the format [Section Name, Page X].
    Checks every cited page number against the ground-truth retrieved pages
    to detect fake or hallucinated page numbers.
    """

    CITATION_PATTERN = re.compile(
        r"\[([A-Za-z0-9\s,\-\–\(\)\'\’]+?),\s*(?:Pages?|Page)\s*(\d+)(?:\s*[\-–]\s*(\d+))?\]"
    )

    @classmethod
    def extract_citations(cls, answer_text: str) -> List[Citation]:
        citations = []
        for match in cls.CITATION_PATTERN.finditer(answer_text):
            raw = match.group(0)
            sec = match.group(1).strip()
            p_start = int(match.group(2))
            p_end = int(match.group(3)) if match.group(3) else p_start
            citations.append(
                Citation(
                    raw_text=raw,
                    section_name=sec,
                    page_start=p_start,
                    page_end=p_end,
                )
            )
        return citations

    @classmethod
    def validate_citations(
        cls, answer_text: str, retrieved_pages: List[int]
    ) -> CitationValidationReport:
        extracted = cls.extract_citations(answer_text)
        retrieved_set = set(retrieved_pages)

        if not extracted:
            return CitationValidationReport(
                citations=[],
                total_citations=0,
                valid_citations=0,
                hallucinated_citations=0,
                precision=0.0,
                has_citations=False,
            )

        valid_count = 0
        for c in extracted:
            cited_range = set(range(c.page_start, c.page_end + 1))
            if cited_range.intersection(retrieved_set):
                c.is_valid = True
                valid_count += 1
            else:
                c.is_valid = False

        hallucinated_count = len(extracted) - valid_count
        precision = valid_count / len(extracted) if extracted else 0.0

        return CitationValidationReport(
            citations=extracted,
            total_citations=len(extracted),
            valid_citations=valid_count,
            hallucinated_citations=hallucinated_count,
            precision=round(precision, 4),
            has_citations=len(extracted) > 0,
        )
