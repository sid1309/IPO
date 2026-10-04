from pathlib import Path
import pytest
from app.ingestion.parser import PDFParser
from app.ingestion.sections import SectionDetector

PDF_PATH = Path("data/raw_pdfs/ZOMATO.pdf")

@pytest.mark.skipif(not PDF_PATH.exists(), reason="ZOMATO.pdf not downloaded")
def test_pdf_metadata_extraction():
    parser = PDFParser(PDF_PATH)
    assert parser.total_pages == 420

    meta = parser.extract_document_metadata()
    assert "ZOMATO" in meta.company_name.upper()
    assert meta.doc_type == "RHP"
    assert "July" in (meta.filing_date or "")


@pytest.mark.skipif(not PDF_PATH.exists(), reason="ZOMATO.pdf not downloaded")
def test_page_extraction_preserves_page_numbers():
    parser = PDFParser(PDF_PATH)
    page_data = parser.extract_page(physical_page=117)

    assert page_data.physical_page == 117
    assert len(page_data.text) > 50
    # Page 117 in Zomato RHP is Objects of the Offer
    assert "OBJECTS" in page_data.text.upper() or "PROCEEDS" in page_data.text.upper()


@pytest.mark.skipif(not PDF_PATH.exists(), reason="ZOMATO.pdf not downloaded")
def test_section_detector_identifies_sebi_chapters():
    parser = PDFParser(PDF_PATH)
    detector = SectionDetector(parser)
    boundaries = detector.detect_sections()

    assert len(boundaries) >= 5
    section_keys = {b.section_key for b in boundaries}
    assert "objects_of_issue" in section_keys
    assert "risk_factors" in section_keys
    assert "capital_structure" in section_keys

    # Test get_section_for_page
    sec_117 = detector.get_section_for_page(117)
    assert sec_117 == "objects_of_issue"
