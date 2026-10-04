import uuid
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from app.ingestion.parser import PDFParser, ExtractedPage, DocumentMetadata
from app.ingestion.sections import SectionDetector, SectionBoundary
from app.ingestion.tables import TableExtractor, ExtractedTable

@dataclass
class DocumentChunk:
    """Represents a chunk indexed into Qdrant or retained for parent context."""
    chunk_id: str
    ipo_id: str
    company_name: str
    doc_type: str
    doc_version: str
    section: str
    page_start: int
    page_end: int
    chunk_type: str            # 'child_text', 'table', 'parent_section'
    text: str
    parent_id: Optional[str] = None
    table_summary: Optional[str] = None
    metadata: dict = field(default_factory=dict)

class HierarchicalChunker:
    """
    Implements Hierarchical (Parent-Child) chunking tailored to SEBI prospectuses:
    1. Parent Chunks: ~1,500-2,000 tokens macro context per sub-section.
    2. Child Chunks: ~400 tokens with 50-token overlap for high-precision retrieval.
    3. Table Chunks: Preserves whole financial tables without chopping rows.
    4. Hard Boundaries: Chunks never cross across distinct SEBI chapters.
    """

    def __init__(
        self,
        child_chunk_size: int = 1500,     # ~375 tokens in characters
        child_overlap: int = 200,          # ~50 tokens in characters
        parent_chunk_size: int = 6000,     # ~1,500 tokens in characters
    ):
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.parent_chunk_size = parent_chunk_size
        self.table_extractor = TableExtractor()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text cleanly at sentence boundaries to avoid splitting mid-number or mid-word."""
        # Avoid splitting on abbreviations like Pvt., Ltd., No., Dr.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_child_chunks(
        self,
        parent_text: str,
        parent_id: str,
        ipo_id: str,
        company_name: str,
        doc_type: str,
        doc_version: str,
        section: str,
        page_start: int,
        page_end: int,
    ) -> List[DocumentChunk]:
        """Split a parent section into overlapping child chunks."""
        sentences = self._split_into_sentences(parent_text)
        if not sentences:
            sentences = [parent_text]

        child_chunks: List[DocumentChunk] = []
        current_chunk_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.child_chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                child_chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        ipo_id=ipo_id,
                        company_name=company_name,
                        doc_type=doc_type,
                        doc_version=doc_version,
                        section=section,
                        page_start=page_start,
                        page_end=page_end,
                        chunk_type="child_text",
                        text=chunk_text,
                        parent_id=parent_id,
                    )
                )

                # Overlap: keep the last sentence(s) up to child_overlap characters
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) <= self.child_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk_sentences = overlap_sentences
                current_len = sum(len(s) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_len += sentence_len

        # Final trailing child chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            child_chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    ipo_id=ipo_id,
                    company_name=company_name,
                    doc_type=doc_type,
                    doc_version=doc_version,
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_type="child_text",
                    text=chunk_text,
                    parent_id=parent_id,
                )
            )

        return child_chunks

    def chunk_document(
        self,
        parser: PDFParser,
        section_detector: SectionDetector,
        ipo_id: str,
        doc_metadata: Optional[DocumentMetadata] = None,
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Process a complete prospectus document.
        Returns:
            (retrieval_chunks, parent_context_chunks)
            - retrieval_chunks: child_text chunks and whole table chunks (for Qdrant embedding)
            - parent_context_chunks: parent_section chunks (for expanding context during generation)
        """
        meta = doc_metadata or parser.extract_document_metadata()
        company_name = meta.company_name or ipo_id
        doc_type = meta.doc_type
        doc_version = meta.filing_date or "latest"

        boundaries = section_detector.detect_sections()
        doc = parser.open()

        retrieval_chunks: List[DocumentChunk] = []
        parent_chunks: List[DocumentChunk] = []

        for b in boundaries:
            section_pages_text: List[Tuple[int, str]] = []
            section_tables: List[ExtractedTable] = []

            # Collect text and tables strictly within this section boundary
            for p_num in range(b.start_page, b.end_page + 1):
                page_data = parser.extract_page(p_num)
                if page_data.text:
                    section_pages_text.append((p_num, page_data.text))

                # Extract financial tables if page has drawing vectors
                if page_data.has_tables:
                    fitz_page = doc[p_num - 1]
                    tables = self.table_extractor.extract_tables_from_page(fitz_page, p_num)
                    section_tables.extend(tables)

            if not section_pages_text and not section_tables:
                continue

            # Group pages into Parent Chunks (~6000 chars / 1500 tokens)
            current_parent_text = []
            current_parent_len = 0
            parent_start_page = b.start_page

            for p_num, p_text in section_pages_text:
                current_parent_text.append(p_text)
                current_parent_len += len(p_text)

                if current_parent_len >= self.parent_chunk_size:
                    parent_id = str(uuid.uuid4())
                    parent_combined_text = "\n\n".join(current_parent_text)

                    parent_chunk = DocumentChunk(
                        chunk_id=parent_id,
                        ipo_id=ipo_id,
                        company_name=company_name,
                        doc_type=doc_type,
                        doc_version=doc_version,
                        section=b.section_key,
                        page_start=parent_start_page,
                        page_end=p_num,
                        chunk_type="parent_section",
                        text=parent_combined_text,
                    )
                    parent_chunks.append(parent_chunk)

                    # Create child chunks linked to this parent
                    children = self._create_child_chunks(
                        parent_text=parent_combined_text,
                        parent_id=parent_id,
                        ipo_id=ipo_id,
                        company_name=company_name,
                        doc_type=doc_type,
                        doc_version=doc_version,
                        section=b.section_key,
                        page_start=parent_start_page,
                        page_end=p_num,
                    )
                    retrieval_chunks.extend(children)

                    current_parent_text = []
                    current_parent_len = 0
                    parent_start_page = p_num + 1

            # Trailing parent chunk in section
            if current_parent_text:
                parent_id = str(uuid.uuid4())
                parent_combined_text = "\n\n".join(current_parent_text)
                parent_chunk = DocumentChunk(
                    chunk_id=parent_id,
                    ipo_id=ipo_id,
                    company_name=company_name,
                    doc_type=doc_type,
                    doc_version=doc_version,
                    section=b.section_key,
                    page_start=parent_start_page,
                    page_end=b.end_page,
                    chunk_type="parent_section",
                    text=parent_combined_text,
                )
                parent_chunks.append(parent_chunk)

                children = self._create_child_chunks(
                    parent_text=parent_combined_text,
                    parent_id=parent_id,
                    ipo_id=ipo_id,
                    company_name=company_name,
                    doc_type=doc_type,
                    doc_version=doc_version,
                    section=b.section_key,
                    page_start=parent_start_page,
                    page_end=b.end_page,
                )
                retrieval_chunks.extend(children)

            # Insert Table Chunks as distinct whole retrieval units
            for tbl in section_tables:
                nearest_parent_id = parent_chunks[-1].chunk_id if parent_chunks else None
                table_chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    ipo_id=ipo_id,
                    company_name=company_name,
                    doc_type=doc_type,
                    doc_version=doc_version,
                    section=b.section_key,
                    page_start=tbl.physical_page,
                    page_end=tbl.physical_page,
                    chunk_type="table",
                    text=tbl.markdown,
                    parent_id=nearest_parent_id,
                    table_summary=tbl.table_summary,
                )
                retrieval_chunks.append(table_chunk)

        return retrieval_chunks, parent_chunks
