import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional
import fitz  # PyMuPDF

@dataclass
class ExtractedPage:
    """Represents a single parsed page from a PDF document."""
    physical_page: int         # 1-indexed page in the physical PDF file
    doc_page: Optional[int]    # Printed page number on the page (if detected)
    text: str                  # Clean extracted text content
    character_count: int
    has_tables: bool = False
    metadata: dict = field(default_factory=dict)

@dataclass
class DocumentMetadata:
    """High-level metadata extracted from the prospectus filing."""
    filename: str
    filepath: str
    total_pages: int
    company_name: Optional[str] = None
    doc_type: str = "RHP"      # DRHP or RHP
    filing_date: Optional[str] = None

class PDFParser:
    """
    High-performance PDF parser using PyMuPDF (fitz).
    Preserves exact 1-indexed physical page numbers for reliable user citations.
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {self.filepath}")
        self._doc: Optional[fitz.Document] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> fitz.Document:
        if self._doc is None or self._doc.is_closed:
            self._doc = fitz.open(str(self.filepath))
        return self._doc

    def close(self) -> None:
        if self._doc is not None and not self._doc.is_closed:
            self._doc.close()
            self._doc = None

    @property
    def total_pages(self) -> int:
        doc = self.open()
        return len(doc)

    def extract_document_metadata(self) -> DocumentMetadata:
        """Infer company name, document type (DRHP/RHP), and date from the cover page."""
        doc = self.open()
        cover_text = doc[0].get_text() if len(doc) > 0 else ""

        # Detect Document Type (DRHP vs RHP)
        doc_type = "RHP"
        if "DRAFT RED HERRING PROSPECTUS" in cover_text.upper():
            doc_type = "DRHP"
        elif "RED HERRING PROSPECTUS" in cover_text.upper():
            doc_type = "RHP"
        elif "PROSPECTUS" in cover_text.upper():
            doc_type = "Prospectus"

        # Detect Filing Date
        filing_date = None
        date_match = re.search(r"Dated\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", cover_text, re.IGNORECASE)
        if date_match:
            filing_date = date_match.group(1)

        # Detect Company Name
        company_name = None
        comp_match = re.search(r"([A-Z0-9\s,\.\(\)]+?\s+(?:LIMITED|PRIVATE LIMITED))", cover_text)
        if comp_match:
            cleaned = comp_match.group(1).strip()
            # Avoid generic heading phrases
            if not any(w in cleaned for w in ["RED HERRING", "PROSPECTUS", "COMPANIES ACT"]):
                company_name = cleaned

        if not company_name:
            company_name = self.filepath.stem.replace("_", " ").title()

        return DocumentMetadata(
            filename=self.filepath.name,
            filepath=str(self.filepath),
            total_pages=len(doc),
            company_name=company_name,
            doc_type=doc_type,
            filing_date=filing_date,
        )

    @staticmethod
    def _extract_printed_page_number(text: str) -> Optional[int]:
        """Detect printed page number in top header or bottom footer lines."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return None

        # Check top 3 and bottom 3 lines
        candidate_lines = lines[:3] + lines[-3:]
        for line in candidate_lines:
            # Standalone digits e.g. "74" or "Page 74" or "- 74 -"
            m = re.match(r"^(?:page\s*)?[-–—]?\s*(\d{1,4})\s*[-–—]?$", line, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        return None

    def extract_page(self, physical_page: int) -> ExtractedPage:
        """
        Extract text from a specific physical page (1-indexed).
        """
        doc = self.open()
        if physical_page < 1 or physical_page > len(doc):
            raise IndexError(f"Page {physical_page} out of bounds (1..{len(doc)})")

        page = doc[physical_page - 1]
        raw_text = page.get_text("text")

        # Clean null characters and trailing carriage returns
        clean_text = raw_text.replace("\x00", "").strip()

        # Check if page has vector drawings/rectangles indicating tables
        has_drawings = len(page.get_drawings()) > 5

        doc_page_num = self._extract_printed_page_number(clean_text)

        return ExtractedPage(
            physical_page=physical_page,
            doc_page=doc_page_num,
            text=clean_text,
            character_count=len(clean_text),
            has_tables=has_drawings,
        )

    def iterate_pages(
        self, start_page: int = 1, end_page: Optional[int] = None
    ) -> Generator[ExtractedPage, None, None]:
        """
        Generator yielding parsed pages one by one for memory efficiency.
        """
        doc = self.open()
        last_page = end_page if end_page is not None else len(doc)
        last_page = min(last_page, len(doc))

        for p in range(start_page, last_page + 1):
            yield self.extract_page(p)
