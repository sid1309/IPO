from pathlib import Path
import pytest
from app.ingestion.parser import PDFParser
from app.ingestion.sections import SectionDetector
from app.ingestion.chunker import HierarchicalChunker
from app.ingestion.tables import TableExtractor

PDF_PATH = Path("data/raw_pdfs/ZOMATO.pdf")

@pytest.mark.skipif(not PDF_PATH.exists(), reason="ZOMATO.pdf not downloaded")
def test_table_extractor_finds_markdown_tables():
    parser = PDFParser(PDF_PATH)
    doc = parser.open()
    extractor = TableExtractor()

    # Physical page 117 has the Use of Proceeds table
    page_117 = doc[116]
    tables = extractor.extract_tables_from_page(page_117, physical_page=117)

    assert len(tables) >= 1
    t = tables[0]
    assert t.physical_page == 117
    assert "|" in t.markdown
    assert "---" in t.markdown
    assert len(t.table_summary) > 10


@pytest.mark.skipif(not PDF_PATH.exists(), reason="ZOMATO.pdf not downloaded")
def test_hierarchical_chunker_parent_child_linking():
    parser = PDFParser(PDF_PATH)
    detector = SectionDetector(parser)
    chunker = HierarchicalChunker(child_chunk_size=1200, child_overlap=150)

    retrieval_chunks, parent_chunks = chunker.chunk_document(
        parser, detector, ipo_id="zomato-2021"
    )

    assert len(retrieval_chunks) > 100
    assert len(parent_chunks) > 10

    parent_ids = {p.chunk_id for p in parent_chunks}

    # Verify every child chunk links to a real parent chunk
    child_text_chunks = [c for c in retrieval_chunks if c.chunk_type == "child_text"]
    assert len(child_text_chunks) > 0
    for child in child_text_chunks:
        assert child.parent_id is not None
        assert child.parent_id in parent_ids
        assert child.section != ""
        assert child.page_start <= child.page_end


@pytest.mark.skipif(not PDF_PATH.exists(), reason="ZOMATO.pdf not downloaded")
def test_table_chunks_preserved_whole():
    parser = PDFParser(PDF_PATH)
    detector = SectionDetector(parser)
    chunker = HierarchicalChunker()

    retrieval_chunks, _ = chunker.chunk_document(
        parser, detector, ipo_id="zomato-2021"
    )

    table_chunks = [c for c in retrieval_chunks if c.chunk_type == "table"]
    assert len(table_chunks) > 0
    for tbl in table_chunks:
        assert tbl.table_summary is not None
        assert "|" in tbl.text
        # Single page assertion for tables extracted from a single page
        assert tbl.page_start == tbl.page_end
