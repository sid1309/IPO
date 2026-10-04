from dataclasses import dataclass
from typing import List, Optional
import fitz

@dataclass
class ExtractedTable:
    """Represents a clean extracted financial table."""
    physical_page: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    markdown: str
    row_count: int
    col_count: int
    table_summary: str = ""

class TableExtractor:
    """
    Extracts and normalizes financial tables from PDF pages into Markdown format.
    Preserves column alignment and financial numbers without splitting across rows.
    """

    @staticmethod
    def _clean_cell(cell: Optional[str]) -> str:
        """Clean line breaks, null bytes, and excess whitespace inside table cells."""
        if cell is None:
            return ""
        # Replace newlines inside cells with space
        cleaned = cell.replace("\x00", "").replace("\n", " ").strip()
        # Escape pipe symbols for markdown
        cleaned = cleaned.replace("|", "/")
        return " ".join(cleaned.split())

    @classmethod
    def _rows_to_markdown(cls, cleaned_rows: List[List[str]]) -> str:
        """Convert 2D array of cleaned cells into a GitHub Flavored Markdown table."""
        if not cleaned_rows:
            return ""

        # Remove columns that are completely empty across all rows
        num_cols = max(len(row) for row in cleaned_rows)
        active_cols = []
        for col_idx in range(num_cols):
            has_content = any(
                len(row) > col_idx and bool(row[col_idx].strip())
                for row in cleaned_rows
            )
            if has_content:
                active_cols.append(col_idx)

        if not active_cols:
            return ""

        # Filter rows to only keep active columns
        filtered_rows = []
        for row in cleaned_rows:
            new_row = [row[c] if c < len(row) else "" for c in active_cols]
            # Only keep if row has at least one non-empty cell
            if any(cell.strip() for cell in new_row):
                filtered_rows.append(new_row)

        if not filtered_rows:
            return ""

        # Header row
        header = filtered_rows[0]
        # If header has empty values, give default column names
        header = [h if h.strip() else f"Col_{i+1}" for i, h in enumerate(header)]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]

        # Body rows
        for row in filtered_rows[1:]:
            padded_row = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(header)]) + " |")

        return "\n".join(lines)

    @classmethod
    def generate_synthetic_summary(cls, headers: List[str], sample_rows: List[List[str]], page_num: int) -> str:
        """
        Rule-based synthetic table summary for vector indexing.
        Provides high-precision keyword cues without requiring external API calls.
        """
        clean_headers = [h for h in headers if h and not h.startswith("Col_")]
        header_str = ", ".join(clean_headers) if clean_headers else "Financial particulars"
        
        # Collect sample entity or particulars names
        sample_items = []
        for r in sample_rows[:3]:
            for cell in r:
                if cell and not cell.replace(",", "").replace(".", "").isdigit() and len(cell) > 3:
                    sample_items.append(cell)
                    break
        items_str = f" including {', '.join(sample_items[:3])}" if sample_items else ""
        return f"Financial table on page {page_num} covering {header_str}{items_str}."

    def extract_tables_from_page(self, page: fitz.Page, physical_page: int) -> List[ExtractedTable]:
        """Extract all tables on a given physical page."""
        extracted: List[ExtractedTable] = []
        try:
            table_finder = page.find_tables()
            tables = getattr(table_finder, "tables", [])
        except Exception:
            tables = []

        for idx, tab in enumerate(tables):
            raw_rows = tab.extract()
            if not raw_rows or len(raw_rows) < 2:
                continue

            cleaned_rows = [
                [self._clean_cell(cell) for cell in row]
                for row in raw_rows
            ]

            md = self._rows_to_markdown(cleaned_rows)
            if not md or "|" not in md:
                continue

            headers = cleaned_rows[0]
            summary = self.generate_synthetic_summary(headers, cleaned_rows[1:], physical_page)

            extracted.append(
                ExtractedTable(
                    physical_page=physical_page,
                    table_index=idx + 1,
                    headers=headers,
                    rows=cleaned_rows[1:],
                    markdown=md,
                    row_count=len(cleaned_rows),
                    col_count=len(headers),
                    table_summary=summary,
                )
            )

        return extracted
