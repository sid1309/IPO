# IPO Prospectus Analyst: Phase 1 (RAG & GraphRAG) Technical Specification & Implementation Plan

> **Compliance & Legal Notice**: This project is an educational, document-intelligence research system. It extracts, indexes, and queries statements from public regulatory filings (SEBI DRHP / RHP prospectuses) with exact page-level citations. It strictly refuses investment recommendations ("buy", "sell", "apply", "avoid") to comply with SEBI regulations and displays disclaimers across all UI and API interfaces.

---

## 1. Executive Summary & System Objectives

Phase 1 delivers an enterprise-grade, verifiable Retrieval-Augmented Generation (RAG) system with a Knowledge Graph (GraphRAG) layer tailored to Indian IPO Prospectuses (Draft Red Herring Prospectus - DRHP, and Red Herring Prospectus - RHP).

### Key Deliverables
1. **Document-Grounded Q&A**: Answers plain-English queries on Use of Proceeds, Capital Structure, Financials, Risk Factors, and Legal Proceedings with exact section and page citations.
2. **Deterministic Numeric Verification**: Guarantees that every currency figure (₹ Crore / Lakhs), percentage, share count, and ratio cited in the response appears verbatim in the source filing chunks.
3. **Structured Summary Extraction**: Pydantic-validated extraction of issue fundamentals (Fresh vs. Offer for Sale split, Objects of the Issue table, Price Band, Pre/Post Promoter Holding).
4. **GraphRAG Layer**: Neo4j knowledge graph mapping entity relationships (Promoters $\leftrightarrow$ Directors $\leftrightarrow$ Subsidiaries $\leftrightarrow$ Group Companies $\leftrightarrow$ Litigations $\leftrightarrow$ Related Party Transactions) to answer multi-hop questions where plain vector chunking fails.
5. **Zero-Cost / Free-Tier Cloud & Local Architecture**: Leverages Google AI Studio free tier (Gemini Flash), Groq (Llama-3.3 70B), Qdrant (Hybrid Dense + Sparse), and Neo4j AuraDB Free.

---

## 2. Model & Infrastructure Analysis

### 2.1 Large Language Models (LLMs)

| Role | Primary Model | Fallback Model | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Answer Generation** | **Gemini 2.5 / 1.5 Flash** (AI Studio Free Tier) | **Groq Llama-3.3 70B** | 1M+ token context window, native structured JSON schema output, high instruction-following for strict citations, zero cloud cost. |
| **Query Router & Classifier** | **Gemini Flash-Lite** | **Groq Llama-3.1 8B** | Ultra-low latency (<300ms) for fast query intent routing; saves generation quota. |
| **Graph Entity Extraction** | **Gemini Flash** (Structured JSON) | **Groq Llama-3.3 70B** | Strong entity-relation extraction conforming to strict Pydantic/JSON schemas. |
| **Eval Judge (LLM-as-a-Judge)**| **Groq Llama-3.3 70B Versatile** | **Gemini Flash** | **Crucial Best Practice**: Using a model family distinct from the generator eliminates self-preference bias during evaluation. |

### 2.2 Embedding & Reranking Architecture

```
User Query / Document Chunks
            │
            ▼
┌────────────────────────────────────────────────────────┐
│              Embedding: BAAI/bge-m3                    │
│   ┌─────────────────────┬──────────────────────────┐   │
│   │ Dense Vector (1024) │ Sparse Lexical Weights   │   │
│   └─────────────────────┴──────────────────────────┘   │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│      Hybrid Retrieval in Qdrant (RRF Score Fusion)     │
│             Payload Filter: ipo_id == X                │
└────────────────────────────────────────────────────────┘
            │  (Top 25 Candidates)
            ▼
┌────────────────────────────────────────────────────────┐
│     Reranker: BAAI/bge-reranker-v2-m3 (or FastRank)    │
└────────────────────────────────────────────────────────┘
            │  (Top 5 Refined Chunks + Parent Expansion)
            ▼
Context Assembly for LLM
```

- **BAAI/bge-m3** (568M params, 1024-d, 8192-token context):
  - **Dense + Sparse Hybrid**: Captures high-level semantic intent while simultaneously preserving exact statutory terms, clause numbers (e.g., *Section 62(1)(c)*), and ISIN identifiers.
  - **Execution Mode**: Runs via **FastEmbed (ONNX)** on CPU for optimized multi-threaded execution without requiring raw PyTorch or a dedicated GPU.
  - **Hosted Alternative**: Google `text-embedding-004` can be evaluated in the embedding bake-off as a zero-CPU hosted comparison.
- **BAAI/bge-reranker-v2-m3**:
  - Re-scores top-25 hybrid candidates down to top-5. Eliminates false positives from boilerplate prospectus disclaimers and cross-references.

### 2.3 Storage Layer

- **Vector Database**: **Qdrant**
  - Native hybrid search (Dense + Sparse).
  - Multi-tenant payload filtering (`ipo_id`, `section`, `doc_type`, `chunk_type`).
  - Deployable via **Embedded Mode** (`QdrantClient(path="./data/qdrant_storage")`) or **Qdrant Cloud Free Tier** (1 GB).
- **Graph Database**: **Neo4j**
  - Stores complex multi-hop entities and relationships with Cypher querying.
  - Deployable on **Neo4j AuraDB Free Tier** (up to 200,000 nodes, 400,000 relationships) or local Docker.
- **Relational & Cache**:
  - **SQLite / PostgreSQL**: Stores IPO metadata, document status, eval results, and user audit logs.
  - **DiskCache / Redis**: Caches LLM responses keyed by prompt hash to eliminate redundant quota usage.

---

## 3. Detailed Data & Ingestion Pipeline

### 3.1 PDF Parsing & Section Detection
Prospectuses range from 300 to 800+ pages. The parser maintains a precise mapping between physical PDF page numbers and document content.
- **Standard SEBI Sections Detected**:
  1. `objects_of_issue`: Objects of the Issue, deployment schedule, bridge financing.
  2. `capital_structure`: Equity share capital, fresh issue vs OFS, shareholding pattern.
  3. `basis_for_price`: Quantitative factors, EPS, P/E, RoNW, peer comparison.
  4. `financial_information`: Restated standalone and consolidated balance sheet, P&L, cash flows.
  5. `risk_factors`: Internal risks, external risks, company-specific risks.
  6. `legal_proceedings`: Outstanding litigation against company, directors, promoters, subsidiaries.
  7. `promoters_and_management`: Board of directors, KMPs, group companies.
  8. `related_party_transactions`: Material RPTs, holding structures.

### 3.2 Table Handling Strategy
Financial tables in prospectuses (e.g., Use of Proceeds breakdown) are dense and easily broken by naive text splitting.
1. **Extraction**: Parsed into clean GitHub Flavored Markdown tables.
2. **Dual-Representation Indexing**:
   - **Table Summary Chunk**: An LLM-generated summary (e.g., *"Table showing Object-wise utilization of fresh issue proceeds totaling ₹500 Cr across debt repayment and capex"*) is embedded for vector search.
   - **Table Data Chunk**: The exact Markdown table is stored as the chunk payload and injected into the LLM context when the summary is retrieved.

### 3.3 Chunk Metadata Schema
```json
{
  "chunk_id": "uuid-v4",
  "ipo_id": "zomato-2021",
  "company_name": "Zomato Limited",
  "doc_type": "RHP",
  "doc_version": "2021-07-06",
  "section": "objects_of_issue",
  "page_start": 74,
  "page_end": 76,
  "chunk_type": "table",
  "parent_id": "parent-section-chunk-id",
  "text": "| Objects | Amount (₹ Cr) |\n| Debt Repayment | 450.00 | ...",
  "table_summary": "Schedule of deployment of fresh issue proceeds..."
}
```

---

## 4. Query Pipeline, Citations & Guardrails

### 4.1 Query Routing & Execution Flow
1. **Input Guardrail & Compliance**:
   - Intercepts intent: if user asks for advice (*"Should I buy?", "Is this IPO good?", "Will it make profit?"*), return an immediate disclaimer and refusal redirect.
2. **Intent Classification**:
   - `DOCUMENT_LOOKUP`: Direct facts (Proceeds, price band, risks) $\rightarrow$ Qdrant Hybrid Search.
   - `RELATIONSHIP_QUERY`: Promoters, group entities, litigations $\rightarrow$ Neo4j GraphRAG.
   - `COMPARISON`: DRHP vs RHP differences $\rightarrow$ Dual-version filtered retrieval.
3. **Retrieval & Reranking**:
   - Fetch top-25 chunks from Qdrant via dense + sparse fusion with `ipo_id` filter.
   - Cross-encoder reranks top-25 to top-5.
   - Retrieve parent chunk text to ensure complete context without truncation.
4. **Answer Generation**:
   - System prompt dictates that every statement must cite `[Section Name, Page X]`.
   - Strict ungrounded refusal: If the filing does not provide the answer, say so explicitly.

### 4.2 Strict Numeric Verification Engine
To eliminate financial hallucinations:
```python
def verify_numeric_grounding(answer_text: str, context_chunks: list[str]) -> VerificationResult:
    # 1. Extract all numbers, percentages, and amounts (e.g., 450.50, 18.5%, ₹1,200 Cr)
    extracted_figures = extract_numeric_tokens(answer_text)
    
    # 2. Check each figure against the retrieved context text
    unmatched_figures = []
    for fig in extracted_figures:
        if not figure_exists_in_context(fig, context_chunks):
            unmatched_figures.append(fig)
            
    # 3. If any figure is hallucinated, flag or redact
    return VerificationResult(
        is_verified=len(unmatched_figures) == 0,
        unmatched_figures=unmatched_figures
    )
```

---

## 5. Structured Summary Extraction

In addition to free-form Q&A, the pipeline extracts a strongly typed `IPOSummary` record:

```python
from pydantic import BaseModel, Field

class ObjectOfIssue(BaseModel):
    name: str = Field(description="Name of the object, e.g. Prepayment of certain borrowings")
    amount_inr_crore: float = Field(description="Amount allocated in ₹ Crore")
    schedule_year_wise: dict[str, float] = Field(default_factory=dict, description="e.g. {'FY25': 200.0, 'FY26': 250.0}")
    evidence_page: int = Field(description="Page number in prospectus")

class IPOSummary(BaseModel):
    ipo_id: str
    company_name: str
    fresh_issue_crore: float | None
    offer_for_sale_crore: float | None
    total_issue_crore: float | None
    price_band_low: float | None
    price_band_high: float | None
    lot_size: int | None
    objects: list[ObjectOfIssue]
    promoter_holding_pre_pct: float | None
    promoter_holding_post_pct: float | None
    lead_managers: list[str]
    registrar: str | None
    evidence: dict[str, int]  # Field -> Page number
```

**Consistency Rule**:
$$\sum \text{Object Amounts} + \text{Estimated Issue Expenses} \approx \text{Fresh Issue Amount}$$

---

## 6. GraphRAG Layer (Neo4j)

### 6.1 Schema Definition
- **Node Labels**:
  - `Company`, `Promoter`, `Director`, `KMP`, `Subsidiary`, `GroupCompany`, `Litigation`, `RelatedPartyTransaction`, `ObjectOfIssue`.
- **Relationship Types**:
  - `(:Promoter)-[:PROMOTES]->(:Company)`
  - `(:Director)-[:DIRECTOR_OF]->(:Company | :GroupCompany | :Subsidiary)`
  - `(:Company)-[:SUBSIDIARY_OF]->(:Company)`
  - `(:Company)-[:FUNDS]->(:ObjectOfIssue)`
  - `(:Person | :Company)-[:PARTY_TO]->(:Litigation)`
  - `(:Company)-[:TRANSACTS_WITH {amount_crore: float, nature: str}]->(:GroupCompany | :Promoter)`

### 6.2 Entity Resolution Pipeline
Prospectuses refer to the same entity under multiple variants (*"Infosys Limited"*, *"Infosys Ltd"*, *"Infosys"*).
1. **Text Normalization**: Strips legal suffixes (`Pvt Ltd`, `Private Limited`, `LLP`, `Inc`).
2. **Fuzzy Deduplication**: `rapidfuzz.fuzz.token_sort_ratio` with threshold $> 90$.
3. **LLM Tie-Break**: For borderline candidates (score 80–90), an LLM call resolves alias identity.
4. **Neo4j Loading**: Merges via idempotent Cypher queries tagged with `ipo_id`:
   ```cypher
   MERGE (c:Company {name: $canonical_name, ipo_id: $ipo_id})
   SET c.aliases = $aliases, c.evidence_page = $page
   ```

### 6.3 Graph Retrieval Paths
1. **Template Cypher**: Pre-validated, parameterized Cypher queries for recurring questions (*"Show all litigations involving promoters"*).
2. **Graph-Augmented Vector Retrieval**: Detect entities in the user question, traverse 1–2 hops in Neo4j to collect linked chunk IDs, and merge those chunks with the vector search results before reranking.

---

## 7. Phase 1 Step-by-Step Implementation Roadmap

```
Milestone 1: Project Skeleton & LLM Client Wrapper
   │
Milestone 2: PDF Parsing & SEBI Section Detector
   │
Milestone 3: Financial Table Extraction & Summary Chunker
   │
Milestone 4: Qdrant Setup & Hybrid Search (Dense + Sparse)
   │
Milestone 5: Cross-Encoder Reranking & Parent Expansion
   │
Milestone 6: Prompt Synthesis, Citations & Guardrails
   │
Milestone 7: Exact Numeric Verification Engine
   │
Milestone 8: Structured Summary & Proceeds Extractor
   │
Milestone 9: GraphRAG Neo4j Layer & Entity Resolution
   │
Milestone 10: FastAPI Endpoints & Streamlit UI with PDF Viewer
   │
Milestone 11: Benchmark Suite (50 Golden Questions + Evals)
```

### Detailed Breakdown of Milestones

| Milestone | Deliverables | Verification Criteria |
| :--- | :--- | :--- |
| **M1: Skeleton & LLM Wrapper** | Directory setup, virtual environment, `app/core/config.py`, `app/llm/client.py` (Gemini + Groq adapters, exponential backoff, prompt cache). | Unit test proving fallback to Groq when Gemini returns 429; prompt cache hit returns without API call. |
| **M2: PDF Parser & Sections** | `app/ingestion/parser.py`, `app/ingestion/sections.py`. Page tracking, TOC/regex boundary detector for the 8 SEBI chapters. | Runs on a sample prospectus; correctly segments text and records exact page numbers. |
| **M3: Tables & Chunking** | `app/ingestion/tables.py`, `app/ingestion/chunker.py`. Markdown table converter, synthetic table summaries, parent-child chunk hierarchy. | Table chunks preserve columns; parent-child mapping correctly established. |
| **M4: Qdrant Hybrid Indexing** | `app/ingestion/embed.py`, `app/ingestion/indexer.py`. FastEmbed `bge-m3` dense + sparse vector generation, Qdrant collection setup. | Hybrid search retrieves relevant chunks with `ipo_id` filter; keyword match works for exact clauses. |
| **M5: Reranker & Context** | `app/retrieval/rerank.py`, `app/retrieval/context.py`. Cross-encoder scoring (top-25 to top-5) and parent chunk expansion. | Evaluated top-5 chunk relevance before vs after reranking; parent context enriched. |
| **M6: Generation & Citations** | `app/generation/prompts.py`, `app/generation/answer.py`. System prompt enforcing `[Section, Page]` citations and ungrounded refusal. | Answers cite exact pages; questions outside the document return explicit refusal. |
| **M7: Numeric Verification** | `app/generation/numeric_check.py`. Regex token matcher checking all numerical claims against source chunks. | Injects a deliberately modified figure; verifier flags and alerts hallucination. |
| **M8: Structured Extractor** | `app/extraction/summary_schema.py`, `app/extraction/extractor.py`. Pydantic models for `IPOSummary` and `ObjectOfIssue`. | Extracted objects sum to fresh issue proceeds (within margin); evidence pages recorded. |
| **M9: GraphRAG (Neo4j)** | `app/graph/schema.py`, `app/graph/extract.py`, `app/graph/resolve.py`, `app/graph/load_neo4j.py`, `app/graph/graph_query.py`. | Multi-hop query (Promoter $\rightarrow$ Litigation $\rightarrow$ Group Company) succeeds where vector search misses. |
| **M10: FastAPI & Streamlit UI** | `app/main.py`, `app/api/`, `frontend/streamlit_app.py`. REST endpoints (`/ipos`, `/ask`, `/summary`, `/graph`, `/charts/proceeds`) + interactive UI. | Streamlit displays answer, side-by-side PDF page viewer, Plotly proceeds bar chart, and graph tab. |
| **M11: Evaluation Benchmark** | `eval/questions_phase1.jsonl`, `eval/run_eval.py`. 50 golden Q&A pairs; Recall@5, MRR, Faithfulness, Numeric Accuracy, Refusal rate. | Complete evaluation matrix generated and logged into `eval/results/`. |

---

## 8. Directory Structure

```
ipo/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_ipos.py          # /ipos upload & status
│   │   ├── routes_ask.py           # /ask Q&A endpoint
│   │   ├── routes_summary.py       # /summary structured card
│   │   ├── routes_graph.py         # /graph visualization data
│   │   └── routes_charts.py        # /charts/proceeds Plotly data
│   ├── core/
│   │   ├── config.py               # Pydantic Settings & environment variables
│   │   ├── schemas.py              # Common API request/response models
│   │   └── logging.py              # Structured logging
│   ├── ingestion/
│   │   ├── parser.py               # PyMuPDF text & page extractor
│   │   ├── sections.py             # SEBI section boundary detection
│   │   ├── tables.py               # pdfplumber / markdown table processor
│   │   ├── chunker.py              # Hierarchical parent-child chunker
│   │   ├── embed.py                # bge-m3 dense + sparse embedding wrapper
│   │   └── indexer.py              # Qdrant collection manager
│   ├── retrieval/
│   │   ├── hybrid.py               # Dense + sparse RRF retrieval
│   │   ├── rerank.py               # Cross-encoder reranker
│   │   ├── parent_expand.py        # Swapping child chunk for parent context
│   │   └── router.py               # Query classifier (Lookup vs Graph vs Refusal)
│   ├── generation/
│   │   ├── prompts.py              # Grounded Q&A prompts with citation rules
│   │   ├── answer.py               # Answer synthesis engine
│   │   ├── citations.py            # Citation parser and validator
│   │   ├── numeric_check.py        # Deterministic numerical accuracy verifier
│   │   └── guardrails.py           # SEBI compliance & advice refusal check
│   ├── extraction/
│   │   ├── summary_schema.py       # Pydantic IPOSummary & ObjectOfIssue
│   │   └── extractor.py            # Structured JSON schema extractor
│   ├── graph/
│   │   ├── schema.py               # Neo4j entity & relation definitions
│   │   ├── extract.py              # LLM relation extraction over key chapters
│   │   ├── resolve.py              # rapidfuzz entity deduplication
│   │   ├── load_neo4j.py           # Idempotent Cypher merge loader
│   │   ├── cypher_templates.py     # Parameterized Cypher query library
│   │   └── graph_query.py          # Graph-augmented chunk retrieval
│   ├── llm/
│   │   ├── client.py               # Unified LLM client interface
│   │   ├── cache.py                # Response caching (disk/redis)
│   │   └── providers/
│   │       ├── gemini.py           # Google AI Studio Gemini adapter
│   │       ├── groq.py             # Groq Llama adapter
│   │       └── ollama.py           # Local model adapter
│   └── charts/
│       └── proceeds.py             # Use of proceeds Plotly generator
├── data/
│   ├── raw_pdfs/                   # Input DRHP/RHP filings
│   ├── processed/                  # Cached parsed sections & tables
│   └── qdrant_storage/             # Local embedded Qdrant data (if not using cloud)
├── eval/
│   ├── questions_phase1.jsonl      # 50 golden benchmark questions
│   ├── run_eval.py                 # Evaluation benchmark runner
│   └── results/                    # Eval score reports
├── frontend/
│   └── streamlit_app.py            # Interactive Streamlit UI
├── tests/
│   ├── test_parser.py
│   ├── test_numeric_check.py
│   ├── test_guardrails.py
│   └── test_llm_client.py
├── .env.example
├── requirements.txt
├── README.md
└── PHASE_1_SPECIFICATION.md
```
