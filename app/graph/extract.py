"""Graph Entity and Relationship Extractor for Prospectus Filings."""

import json
import logging
from pathlib import Path
from typing import List, Optional
from app.core.config import settings
from app.graph.schema import ExtractedGraph, GraphNode, GraphRelationship, NodeType, RelationType
from app.graph.resolve import EntityResolver
from app.ingestion.chunker import DocumentChunk
from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

GRAPH_SYSTEM_PROMPT = """You are an expert financial and corporate graph analyst extracting entities and relationships from SEBI prospectus excerpts.
Extract ONLY verified relationships that appear in the text.
Return a valid JSON object matching the requested schema.
Allowed node labels: 'Company', 'Promoter', 'Director', 'GroupCompany', 'Litigation', 'ObjectOfIssue'.
Allowed relationship types: 'PROMOTER_OF', 'DIRECTOR_OF', 'HAS_GROUP_COMPANY', 'PARTY_TO', 'ALLOCATES_PROCEEDS_TO'.
"""

GRAPH_USER_PROMPT = """Extract the entities and their directed relationships from the provided prospectus excerpts.

Excerpts:
================================================================================
{context_text}
================================================================================

JSON Schema format:
{{
  "nodes": [
    {{"label": "Company", "name": "Zomato Limited", "page": 1, "properties": {{}}}},
    {{"label": "Promoter", "name": "Deepinder Goyal", "page": 112, "properties": {{"role": "Managing Director"}}}},
    {{"label": "GroupCompany", "name": "Zomato Ireland Limited", "page": 168, "properties": {{"country": "Ireland"}}}},
    {{"label": "Litigation", "name": "VAT Appeal 2020", "page": 425, "properties": {{"amount": "₹ 45 million", "status": "Pending"}}}}
  ],
  "relationships": [
    {{"source_name": "Deepinder Goyal", "source_label": "Promoter", "target_name": "Zomato Limited", "target_label": "Company", "rel_type": "PROMOTER_OF", "page": 112}},
    {{"source_name": "Deepinder Goyal", "source_label": "Promoter", "target_name": "Zomato Ireland Limited", "target_label": "GroupCompany", "rel_type": "DIRECTOR_OF", "page": 168}},
    {{"source_name": "Zomato Ireland Limited", "source_label": "GroupCompany", "target_name": "VAT Appeal 2020", "target_label": "Litigation", "rel_type": "PARTY_TO", "page": 425}}
  ]
}}

Return ONLY the raw JSON object.
"""


class GraphEntityExtractor:
    def __init__(self, llm_client: Optional[LLMClient] = None, resolver: Optional[EntityResolver] = None):
        self.llm_client = llm_client or LLMClient()
        self.resolver = resolver or EntityResolver()
        self.cache_dir = Path(settings.PROCESSED_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, ipo_id: str) -> Path:
        return self.cache_dir / f"{ipo_id}_graph.json"

    def extract_graph(
        self,
        ipo_id: str,
        chunks: List[DocumentChunk],
        force_refresh: bool = False,
    ) -> ExtractedGraph:
        cache_path = self.get_cache_path(ipo_id)
        if cache_path.exists() and not force_refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ExtractedGraph.model_validate(data)
            except Exception as e:
                logger.warning(f"Failed to read graph cache from {cache_path}: {e}")

        rel_chapters = {
            "capital_structure",
            "our_promoters",
            "outstanding_litigation",
            "objects_of_issue",
            "group_companies",
            "promoters_and_management",
        }

        targeted = [c for c in chunks if any(ch in c.section.lower() for ch in rel_chapters)]
        if not targeted:
            targeted = chunks[:15]

        context_parts = []
        char_count = 0
        for c in targeted:
            if char_count + len(c.text) > 9500:
                break
            header = f"\n--- [Section: {c.section}, Page {c.page_start}, Chunk: {c.chunk_id}] ---"
            context_parts.append(header)
            context_parts.append(c.text)
            char_count += len(c.text) + len(header)

        context_text = "\n\n".join(context_parts)
        prompt = GRAPH_USER_PROMPT.format(context_text=context_text)

        raw_response = self.llm_client.generate(
            prompt=prompt,
            system_instruction=GRAPH_SYSTEM_PROMPT,
            temperature=0.0,
            use_cache=False,
        )

        parsed_data = self._clean_and_parse_json(raw_response)

        nodes_dict = {}
        for n in parsed_data.get("nodes", []):
            name = n.get("name", "").strip()
            label = n.get("label", "Company")
            if not name:
                continue
            canonical_id = self.resolver.canonicalize(name, label, ipo_id)
            if canonical_id not in nodes_dict:
                nodes_dict[canonical_id] = GraphNode(
                    node_id=canonical_id,
                    label=label,
                    name=name,
                    ipo_id=ipo_id,
                    page=n.get("page"),
                    properties=n.get("properties", {}),
                )

        relationships = []
        for r in parsed_data.get("relationships", []):
            s_name = r.get("source_name", "").strip()
            s_label = r.get("source_label", "Company")
            t_name = r.get("target_name", "").strip()
            t_label = r.get("target_label", "Company")
            rel_type = r.get("rel_type")

            if not s_name or not t_name or not rel_type:
                continue

            s_id = self.resolver.canonicalize(s_name, s_label, ipo_id)
            t_id = self.resolver.canonicalize(t_name, t_label, ipo_id)

            if s_id not in nodes_dict:
                nodes_dict[s_id] = GraphNode(
                    node_id=s_id,
                    label=s_label,
                    name=s_name,
                    ipo_id=ipo_id,
                    page=r.get("page"),
                )

            if t_id not in nodes_dict:
                nodes_dict[t_id] = GraphNode(
                    node_id=t_id,
                    label=t_label,
                    name=t_name,
                    ipo_id=ipo_id,
                    page=r.get("page"),
                )

            relationships.append(
                GraphRelationship(
                    source_id=s_id,
                    target_id=t_id,
                    rel_type=rel_type,
                    page=r.get("page"),
                    properties=r.get("properties", {}),
                )
            )

        graph = ExtractedGraph(
            ipo_id=ipo_id,
            nodes=list(nodes_dict.values()),
            relationships=relationships,
        )

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(graph.model_dump(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache graph to {cache_path}: {e}")

        return graph

    def _clean_and_parse_json(self, raw_text: str) -> dict:
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        try:
            return json.loads(clean)
        except Exception as e:
            logger.error(f"Failed to parse graph JSON: {e}")
            return {"nodes": [], "relationships": []}
