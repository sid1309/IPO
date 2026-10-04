"""API Data Models for Request & Response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User query request for prospectus question answering."""
    ipo_id: str = Field(description="Unique IPO identifier (e.g. 'zomato-2021')")
    question: str = Field(description="The user's question about the prospectus")


class CitationItem(BaseModel):
    """Individual inline citation extracted from prospectus context."""
    section: str
    page: int
    quote: str
    is_valid: bool = True


class QueryResponse(BaseModel):
    """Structured response from the prospectus QA engine."""
    question: str
    answer: str
    is_refusal: bool = False
    is_advice_refusal: bool = False
    is_listing_analysis: bool = False
    cited_pages: List[int] = []
    sections_covered: List[str] = []
    citations: List[CitationItem] = []
    numeric_precision: float = 1.0
    all_numbers_verified: bool = True
    unverified_numbers_count: int = 0
    disclaimer: str = ""


class IPOInfo(BaseModel):
    """Metadata summary of an available IPO prospectus."""
    ipo_id: str
    company_name: str
    doc_type: str
    filename: str
    page_count: int
    is_indexed: bool = False
    has_summary: bool = False
    has_graph: bool = False


class UploadResponse(BaseModel):
    """Response returned upon successful PDF ingestion."""
    ipo_id: str
    company_name: str
    filename: str
    total_pages: int
    total_chunks: int
    message: str
