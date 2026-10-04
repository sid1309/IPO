"""FastAPI router endpoints for IPO Prospectus Analyst."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.models import (
    CitationItem,
    IPOInfo,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.core.config import settings
from app.extraction.extractor import SummaryCardExtractor
from app.extraction.summary_schema import IPOSummaryCard
from app.generation.answer import AnswerGenerator
from app.generation.guardrails import InputGuardrail, SEBI_DISCLAIMER
from app.graph.schema import ExtractedGraph
from app.graph.extract import GraphEntityExtractor
from app.graph.graph_query import GraphRAGQueryEngine
from app.graph.load_neo4j import Neo4jLoader
from app.ingestion.chunker import HierarchicalChunker
from app.ingestion.indexer import QdrantIndexer
from app.ingestion.parser import PDFParser
from app.ingestion.sections import SectionDetector
from app.llm.client import LLMClient
from app.retrieval.parent_expand import AssembledContext, ParentContextManager
from app.retrieval.rerank import CrossEncoderReranker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

llm_client = LLMClient()
indexer = QdrantIndexer()
reranker = CrossEncoderReranker()
answer_generator = AnswerGenerator(llm_client=llm_client)
summary_extractor = SummaryCardExtractor(llm_client=llm_client)
graph_extractor = GraphEntityExtractor(llm_client=llm_client)
neo4j_loader = Neo4jLoader()
graph_engine = GraphRAGQueryEngine(loader=neo4j_loader)


@router.get("/ipos", response_model=List[IPOInfo])
def list_available_ipos() -> List[IPOInfo]:
    """Lists all detected or preprocessed IPO prospectuses."""
    raw_dir = Path(settings.RAW_PDFS_DIR)
    processed_dir = Path(settings.PROCESSED_DIR)
    ipos: Dict[str, IPOInfo] = {}

    # Check processed JSON summaries
    if processed_dir.exists():
        for summary_file in processed_dir.glob("*_summary.json"):
            ipo_id = summary_file.stem.replace("_summary", "")
            has_graph = (processed_dir / f"{ipo_id}_graph.json").exists()
            company_clean = ipo_id.replace("-", " ").title()
            ipos[ipo_id] = IPOInfo(
                ipo_id=ipo_id,
                company_name=company_clean,
                doc_type="RHP",
                filename=f"{company_clean}.pdf",
                page_count=0,
                is_indexed=True,
                has_summary=True,
                has_graph=has_graph,
            )

    # Check raw PDFs
    if raw_dir.exists():
        for pdf_file in raw_dir.glob("*.pdf"):
            ipo_id = pdf_file.stem.lower().replace(" ", "-")
            has_summary = (processed_dir / f"{ipo_id}_summary.json").exists()
            has_graph = (processed_dir / f"{ipo_id}_graph.json").exists()

            if ipo_id not in ipos:
                try:
                    parser = PDFParser(pdf_file)
                    page_count = parser.total_pages
                    doc_type = parser.doc_type
                    company_name = parser.extract_document_metadata().company_name or pdf_file.stem.title()
                except Exception:
                    page_count = 0
                    doc_type = "RHP"
                    company_name = pdf_file.stem.title()

                ipos[ipo_id] = IPOInfo(
                    ipo_id=ipo_id,
                    company_name=company_name,
                    doc_type=doc_type,
                    filename=pdf_file.name,
                    page_count=page_count,
                    is_indexed=True,
                    has_summary=has_summary,
                    has_graph=has_graph,
                )
            else:
                try:
                    parser = PDFParser(pdf_file)
                    ipos[ipo_id].page_count = parser.total_pages
                except Exception:
                    pass

    return list(ipos.values())


@router.get("/summary/{ipo_id}", response_model=IPOSummaryCard)
def get_ipo_summary(ipo_id: str) -> IPOSummaryCard:
    """Retrieves the pre-extracted or cached structured summary card."""
    cache_path = Path(settings.PROCESSED_DIR) / f"{ipo_id}_summary.json"
    if not cache_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No summary found for '{ipo_id}'. Please upload and index the filing first.",
        )

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IPOSummaryCard.model_validate(data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading summary card: {e}",
        )


@router.get("/graph/{ipo_id}")
def get_ipo_graph(ipo_id: str) -> Dict[str, Any]:
    """Retrieves the entity network graph and multi-hop queries."""
    cache_path = Path(settings.PROCESSED_DIR) / f"{ipo_id}_graph.json"
    graph_data = {"nodes": [], "relationships": []}

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            extracted_g = ExtractedGraph.model_validate(graph_data)
            neo4j_loader.insert_graph(extracted_g)
        except Exception as e:
            logger.warning(f"Failed to read graph cache: {e}")

    litigations = graph_engine.get_promoter_litigations(ipo_id)
    directorships = graph_engine.get_common_directorships(ipo_id)

    return {
        "ipo_id": ipo_id,
        "nodes": graph_data.get("nodes", []),
        "relationships": graph_data.get("relationships", []),
        "promoter_litigations": litigations,
        "common_directorships": directorships,
    }


@router.post("/upload", response_model=UploadResponse)
def upload_and_process_prospectus(file: UploadFile = File(...)) -> UploadResponse:
    """Uploads a PDF, chunks it hierarchically, builds hybrid vector index, summary card, and graph."""
    raw_dir = Path(settings.RAW_PDFS_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / file.filename

    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ipo_id = destination.stem.lower().replace(" ", "-")

    try:
        parser = PDFParser(destination)
        total_pages = parser.total_pages
        meta = parser.extract_document_metadata()
        company_name = meta.company_name or destination.stem.title()

        detector = SectionDetector(parser)
        chunker = HierarchicalChunker()
        retrieval_chunks, parent_chunks = chunker.chunk_document(
            parser=parser,
            detector=detector,
            ipo_id=ipo_id,
            doc_metadata=meta,
        )

        indexer.index_chunks(retrieval_chunks)
        summary_extractor.extract_summary_card(ipo_id=ipo_id, chunks=retrieval_chunks, force_refresh=True)
        graph = graph_extractor.extract_graph(ipo_id=ipo_id, chunks=retrieval_chunks, force_refresh=True)
        neo4j_loader.insert_graph(graph)

        return UploadResponse(
            ipo_id=ipo_id,
            company_name=company_name,
            filename=file.filename,
            total_pages=total_pages,
            total_chunks=len(retrieval_chunks),
            message=f"Successfully ingested and indexed {file.filename} across Vector DB and Knowledge Graph.",
        )
    except Exception as e:
        logger.error(f"Ingestion pipeline failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion pipeline error: {e}",
        )


@router.post("/query", response_model=QueryResponse)
def query_prospectus(req: QueryRequest) -> QueryResponse:
    """Answers a question about the prospectus with citations, guardrails, and numeric verification."""
    question = req.question.strip()
    guard_res = InputGuardrail.check_input(question)

    if not guard_res.is_safe:
        return QueryResponse(
            question=question,
            answer=guard_res.refusal_message or "Request refused by guardrail.",
            is_refusal=True,
            is_advice_refusal=guard_res.reason == "advice_refusal",
            is_listing_analysis=False,
            cited_pages=[],
            sections_covered=[],
            citations=[],
            numeric_precision=1.0,
            all_numbers_verified=True,
            unverified_numbers_count=0,
            disclaimer=SEBI_DISCLAIMER,
        )

    candidates = indexer.hybrid_search(query=question, ipo_id=req.ipo_id, top_k=15)
    augmented_pages = graph_engine.get_graph_augmented_pages(query=question, ipo_id=req.ipo_id)

    if not candidates and not augmented_pages:
        return QueryResponse(
            question=question,
            answer="The provided prospectus does not contain information regarding this topic.",
            is_refusal=True,
            is_advice_refusal=False,
            is_listing_analysis=guard_res.analysis_mode == "listing_analysis",
            cited_pages=[],
            sections_covered=[],
            citations=[],
            numeric_precision=1.0,
            all_numbers_verified=True,
            unverified_numbers_count=0,
            disclaimer=SEBI_DISCLAIMER,
        )

    reranked = reranker.rerank(query=question, candidate_points=candidates, top_k=5)
    context_mgr = ParentContextManager()
    context = context_mgr.assemble_context(reranked_chunks=reranked)

    result = answer_generator.generate_answer(
        question=question,
        context=context,
        company_name=req.ipo_id.replace("-", " ").title(),
        doc_type="RHP",
        use_cache=True,
    )

    seen_citations = set()
    citation_items = []
    for c in result.citations.citations:
        key = (c.section_name.strip().lower(), c.page_start)
        if key not in seen_citations:
            seen_citations.add(key)
            citation_items.append(
                CitationItem(
                    section=c.section_name,
                    page=c.page_start,
                    quote=c.raw_text,
                    raw_citation=c.raw_text,
                    is_valid=c.is_valid,
                )
            )

    return QueryResponse(
        question=question,
        answer=result.answer,
        is_refusal=result.is_refusal,
        is_advice_refusal=result.is_advice_refusal,
        is_listing_analysis=result.is_listing_analysis,
        cited_pages=result.cited_pages,
        sections_covered=result.sections_covered,
        citations=citation_items,
        numeric_precision=result.numeric_verification.precision,
        all_numbers_verified=result.numeric_verification.all_verified,
        unverified_numbers_count=result.numeric_verification.unverified_numbers,
        disclaimer=result.disclaimer,
    )


@router.post("/index-cached/{ipo_id}")
def index_cached_prospectus(ipo_id: str) -> Dict[str, Any]:
    """Indexes a cached prospectus from disk into Qdrant."""
    raw_dir = Path(settings.RAW_PDFS_DIR)
    target_file = None

    for pdf_file in raw_dir.glob("*.pdf"):
        if pdf_file.stem.lower().replace(" ", "-") == ipo_id:
            target_file = pdf_file
            break

    if not target_file:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF found for IPO ID '{ipo_id}' in {raw_dir}",
        )

    parser = PDFParser(target_file)
    detector = SectionDetector(parser)
    chunker = HierarchicalChunker()
    meta = parser.extract_document_metadata()

    retrieval_chunks, _ = chunker.chunk_document(
        parser=parser,
        detector=detector,
        ipo_id=ipo_id,
        doc_metadata=meta,
    )

    indexed_count = indexer.index_chunks(retrieval_chunks)

    return {
        "ipo_id": ipo_id,
        "filename": target_file.name,
        "total_chunks": len(retrieval_chunks),
        "indexed_chunks": indexed_count,
        "message": f"Successfully indexed {indexed_count} chunks into Qdrant for {ipo_id}.",
    }
