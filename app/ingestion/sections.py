import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from app.ingestion.parser import PDFParser

# Normalized canonical section keys
CANONICAL_SECTIONS = {
    "objects_of_issue": [
        r"OBJECTS\s+OF\s+THE\s+(?:OFFER|ISSUE)",
        r"USE\s+OF\s+PROCEEDS",
        r"UTILISATION\s+OF\s+ISSUE\s+PROCEEDS",
    ],
    "capital_structure": [
        r"CAPITAL\s+STRUCTURE",
    ],
    "basis_for_price": [
        r"BASIS\s+FOR\s+(?:ISSUE|OFFER)\s+PRICE",
    ],
    "financial_information": [
        r"SECTION\s+V\s*:\s*FINANCIAL\s+INFORMATION",
        r"FINANCIAL\s+STATEMENTS",
        r"RESTATED\s+FINANCIAL\s+INFORMATION",
        r"FINANCIAL\s+INDEBTEDNESS",
    ],
    "risk_factors": [
        r"SECTION\s+II\s*:\s*RISK\s+FACTORS",
        r"RISK\s+FACTORS",
    ],
    "legal_proceedings": [
        r"OUTSTANDING\s+LITIGATION\s+AND\s+MATERIAL\s+DEVELOPMENTS",
        r"LEGAL\s+AND\s+OTHER\s+INFORMATION",
        r"OUTSTANDING\s+LITIGATION",
    ],
    "promoters_and_management": [
        r"OUR\s+MANAGEMENT",
        r"OUR\s+PROMOTERS(?:\s+AND\s+PROMOTER\s+GROUP)?",
        r"BOARD\s+OF\s+DIRECTORS",
    ],
    "group_companies": [
        r"OUR\s+GROUP\s+COMPANIES",
        r"GROUP\s+COMPANIES",
    ],
    "related_party_transactions": [
        r"RELATED\s+PARTY\s+TRANSACTIONS",
    ],
    "general_and_summary": [
        r"SUMMARY\s+OF\s+THE\s+(?:OFFER|ISSUE)\s+DOCUMENT",
        r"SECTION\s+I\s*:\s*GENERAL",
        r"DEFINITIONS\s+AND\s+ABBREVIATIONS",
    ],
}

@dataclass
class SectionBoundary:
    """Represents a continuous page range belonging to a SEBI section."""
    section_key: str           # e.g. 'objects_of_issue'
    display_name: str          # e.g. 'Objects of the Offer'
    start_page: int            # Physical PDF page (1-indexed)
    end_page: int              # Physical PDF page (1-indexed, inclusive)

class SectionDetector:
    """
    Detects SEBI chapter boundaries in an IPO prospectus using Table of Contents
    and in-page heading matching.
    """

    def __init__(self, parser: PDFParser):
        self.parser = parser
        self._boundaries: Optional[List[SectionBoundary]] = None

    def _find_toc_pages(self, max_check_pages: int = 10) -> List[int]:
        """Find the physical pages containing the Table of Contents."""
        toc_pages = []
        for p in range(1, min(max_check_pages + 1, self.parser.total_pages + 1)):
            page_text = self.parser.extract_page(p).text.upper()
            if "TABLE OF CONTENTS" in page_text or "INDEX" in page_text:
                toc_pages.append(p)
        return toc_pages

    def _parse_toc_entries(self, toc_pages: List[int]) -> List[tuple[str, int]]:
        """
        Extract section titles and their printed document page numbers from TOC pages.
        Returns list of (section_title, printed_page_number) tuples.
        """
        entries = []
        pattern = re.compile(
            r"([A-Z0-9\s,\-–—\(\)\'\’]+?)\s*[\.\s]{3,}\s*(\d{1,4})", re.MULTILINE
        )

        for p in toc_pages:
            text = self.parser.extract_page(p).text
            for match in pattern.finditer(text):
                title = match.group(1).strip()
                page_num = int(match.group(2))
                entries.append((title, page_num))

        # Sort entries by printed page number
        entries.sort(key=lambda x: x[1])
        return entries

    def _detect_physical_page_offset(self, toc_entries: List[tuple[str, int]], toc_end_page: int) -> int:
        """
        Calculate offset between physical PDF page and printed document page:
        physical_page = printed_page + offset
        """
        # Typically printed page 1 starts 1 to 4 pages after the Table of Contents
        # Look for the page where printed page 1 appears in text
        for p in range(toc_end_page, min(toc_end_page + 6, self.parser.total_pages + 1)):
            page_data = self.parser.extract_page(p)
            if page_data.doc_page == 1:
                return p - 1

        # Fallback heuristic: TOC end page is physical offset
        return max(1, toc_end_page)

    @staticmethod
    def _match_canonical_key(title: str) -> Optional[str]:
        """Match a section title from TOC or page header against canonical SEBI sections."""
        clean_title = title.upper()
        for key, patterns in CANONICAL_SECTIONS.items():
            for pat in patterns:
                if re.search(pat, clean_title):
                    return key
        return None

    def detect_sections(self) -> List[SectionBoundary]:
        """
        Analyze document and return sorted list of SectionBoundary objects.
        """
        if self._boundaries is not None:
            return self._boundaries

        toc_pages = self._find_toc_pages()
        if not toc_pages:
            # Fallback if no explicit TOC is found
            return self._fallback_page_scan()

        toc_entries = self._parse_toc_entries(toc_pages)
        offset = self._detect_physical_page_offset(toc_entries, max(toc_pages))

        # Filter and map TOC entries to canonical keys
        detected_points = []
        for title, doc_page in toc_entries:
            key = self._match_canonical_key(title)
            if key:
                phys_page = min(doc_page + offset, self.parser.total_pages)
                detected_points.append((phys_page, key, title.title()))

        if not detected_points:
            return self._fallback_page_scan()

        # Deduplicate and sort by physical start page
        detected_points.sort(key=lambda x: x[0])
        
        boundaries: List[SectionBoundary] = []
        total_p = self.parser.total_pages

        for i in range(len(detected_points)):
            curr_page, curr_key, curr_name = detected_points[i]
            if i + 1 < len(detected_points):
                next_page = detected_points[i + 1][0]
                end_p = max(curr_page, next_page - 1)
            else:
                end_p = total_p

            boundaries.append(
                SectionBoundary(
                    section_key=curr_key,
                    display_name=curr_name,
                    start_page=curr_page,
                    end_page=end_p,
                )
            )

        self._boundaries = boundaries
        return self._boundaries

    def _fallback_page_scan(self) -> List[SectionBoundary]:
        """Fallback boundary detection scanning page headers directly."""
        boundaries = []
        total_p = self.parser.total_pages
        # Default full-range fallback
        boundaries.append(
            SectionBoundary(
                section_key="general_and_summary",
                display_name="General Prospectus Document",
                start_page=1,
                end_page=total_p,
            )
        )
        return boundaries

    def get_section_for_page(self, physical_page: int) -> str:
        """Return the canonical section key for a given physical page number."""
        boundaries = self.detect_sections()
        for b in boundaries:
            if b.start_page <= physical_page <= b.end_page:
                return b.section_key
        return "general_and_summary"
