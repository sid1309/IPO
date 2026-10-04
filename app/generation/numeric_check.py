"""Deterministic Numeric Verifier module for grounding numeric claims."""

import re
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class ExtractedNumber:
    """Represents a numeric entity found in an answer."""
    raw_string: str
    normalized_digits: str
    category: str
    is_verified: bool = False
    context_match: str = ""


@dataclass
class NumericVerificationReport:
    """Summary report of numeric verification on an answer."""
    total_numbers: int
    verified_numbers: int
    unverified_numbers: int
    precision: float
    all_verified: bool
    details: List[ExtractedNumber] = field(default_factory=list)


class NumericVerifier:
    """
    Deterministic regex-based numeric verification engine.
    Scans generated answers for numbers and checks if they exist in source context.
    """

    CURRENCY_PATTERN = re.compile(
        r"(?:₹|rs\.?|inr|\$)\s*([0-9]+(?:,[0-9]{2,3})*(?:\.[0-9]+)?)\s*(?:million|crore|cr|lakh|bn|billion)?",
        re.IGNORECASE,
    )
    PERCENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)
    MULTIPLE_PATTERN = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:x|times)\b", re.IGNORECASE)
    GENERAL_NUMBER_PATTERN = re.compile(
        r"\b([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]+\.[0-9]+)\b"
    )

    @classmethod
    def extract_numbers(cls, text: str) -> List[ExtractedNumber]:
        extracted = []
        seen_spans = set()

        def add_entity(match, category: str, digit_group: int):
            span = match.span()
            for s_start, s_end in seen_spans:
                if not (span[1] <= s_start or span[0] >= s_end):
                    return
            seen_spans.add(span)
            raw = match.group(0).strip()
            digits = match.group(digit_group).replace(",", "").strip()

            # Ignore single digit numbers (1-5 often used for numbered lists)
            if digits in {"1", "2", "3", "4", "5"}:
                return
            extracted.append(
                ExtractedNumber(
                    raw_string=raw,
                    normalized_digits=digits,
                    category=category,
                )
            )

        for m in cls.CURRENCY_PATTERN.finditer(text):
            add_entity(m, "currency", 1)

        for m in cls.PERCENT_PATTERN.finditer(text):
            add_entity(m, "percentage", 1)

        for m in cls.MULTIPLE_PATTERN.finditer(text):
            add_entity(m, "multiple", 1)

        for m in cls.GENERAL_NUMBER_PATTERN.finditer(text):
            add_entity(m, "number", 1)

        return extracted

    @classmethod
    def verify(cls, answer: str, context_text: str) -> NumericVerificationReport:
        entities = cls.extract_numbers(answer)
        if not entities:
            return NumericVerificationReport(
                total_numbers=0,
                verified_numbers=0,
                unverified_numbers=0,
                precision=1.0,
                all_verified=True,
                details=[],
            )

        normalized_context = context_text.lower()
        uncomma_context = re.sub(r",", "", normalized_context)
        verified_count = 0

        for entity in entities:
            raw_clean = entity.raw_string.lower().strip()
            digits = entity.normalized_digits

            if raw_clean in normalized_context:
                entity.is_verified = True
                entity.context_match = raw_clean
                verified_count += 1
                continue

            digit_pattern = rf"\b{re.escape(digits)}\b"
            digit_comma_pattern = rf"\b{re.escape(cls._add_commas(digits))}\b"

            if re.search(digit_pattern, uncomma_context) or re.search(
                digit_comma_pattern, normalized_context
            ):
                entity.is_verified = True
                entity.context_match = digits
                verified_count += 1
            else:
                entity.is_verified = False

        total = len(entities)
        unverified = total - verified_count
        precision = verified_count / total if total > 0 else 1.0

        return NumericVerificationReport(
            total_numbers=total,
            verified_numbers=verified_count,
            unverified_numbers=unverified,
            precision=precision,
            all_verified=unverified == 0,
            details=entities,
        )

    @classmethod
    def _add_commas(cls, digits_str: str) -> str:
        if "." in digits_str:
            parts = digits_str.split(".")
            integer_part = parts[0]
            decimal_part = "." + parts[1]
        else:
            integer_part = digits_str
            decimal_part = ""

        try:
            val = int(integer_part)
            return f"{val:,}" + decimal_part
        except ValueError:
            return digits_str
