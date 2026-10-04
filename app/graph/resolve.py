"""Entity resolution and name canonicalization to avoid duplicate graph nodes."""

import re
from typing import Dict, Optional


class EntityResolver:
    """
    Normalizes company, promoter, and litigation names to prevent duplicate nodes.
    For example:
      - 'Info Edge (India) Limited' and 'Info Edge India Ltd.' -> 'info_edge_india'
      - 'Mr. Deepinder Goyal' and 'Deepinder Goyal' -> 'deepinder_goyal'
    """

    CORPORATE_SUFFIXES = (
        r"\b(?:private\s+limited|pvt\.?\s*ltd\.?)\b",
        r"\b(?:limited|ltd\.?)\b",
        r"\b(?:corporation|corp\.?)\b",
        r"\b(?:incorporated|inc\.?)\b",
        r"\b(?:llc|llp)\b",
    )

    HONORIFICS = (
        r"\b(?:mr\.?|ms\.?|mrs\.?|dr\.?|shri|smt)\b",
    )

    def __init__(self):
        self._canonical_cache: Dict[str, str] = {}

    @classmethod
    def clean_name(cls, name: str) -> str:
        if not name:
            return ""

        cleaned = name.strip()

        for pat in cls.HONORIFICS:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

        for pat in cls.CORPORATE_SUFFIXES:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

        cleaned = cleaned.replace("(", " ").replace(")", " ")
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)
        cleaned = " ".join(cleaned.split()).strip()

        return cleaned

    @classmethod
    def generate_node_id(cls, entity_type: str, raw_name: str, ipo_id: str) -> str:
        base = cls.clean_name(raw_name).lower().replace(" ", "_")
        if not base:
            base = "unknown"
        type_prefix = entity_type.lower()
        return f"{ipo_id}_{type_prefix}_{base}"

    def canonicalize(self, raw_name: str, entity_type: str, ipo_id: str) -> str:
        cache_key = f"{ipo_id}_{entity_type}_{raw_name.lower().strip()}"
        if cache_key in self._canonical_cache:
            return self._canonical_cache[cache_key]

        canonical_id = self.generate_node_id(entity_type, raw_name, ipo_id)
        self._canonical_cache[cache_key] = canonical_id
        return canonical_id
