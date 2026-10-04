"""Cross-Encoder Reranker using FastEmbed BAAI/bge-reranker-v2-m3."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from fastembed.rerank.cross_encoder import TextCrossEncoder


@dataclass
class RerankedChunk:
    chunk_id: str
    text: str
    relevance_score: float
    section: str
    page_start: int
    page_end: int
    chunk_type: str
    parent_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", top_n: int = 5):
        self.model = TextCrossEncoder(model_name=model_name)
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        candidate_points: List[Any],
        top_k: Optional[int] = None,
    ) -> List[RerankedChunk]:
        top_n = top_k if top_k is not None else self.top_n
        if not candidate_points:
            return []

        documents = []
        for pt in candidate_points:
            payload = getattr(pt, "payload", {}) or {}
            if payload.get("chunk_type") == "table" and payload.get("table_summary"):
                doc_text = f"{payload['table_summary']}\n{payload.get('text', '')[:500]}"
            else:
                doc_text = payload.get("text", "")
            documents.append(doc_text)

        scores = list(self.model.rerank(query, documents))

        scored_chunks = []
        for pt, score in zip(candidate_points, scores):
            payload = getattr(pt, "payload", {}) or {}
            scored_chunks.append(
                RerankedChunk(
                    chunk_id=str(getattr(pt, "id", "")),
                    text=payload.get("text", ""),
                    relevance_score=float(score),
                    section=payload.get("section", "general"),
                    page_start=payload.get("page_start", 1),
                    page_end=payload.get("page_end", 1),
                    chunk_type=payload.get("chunk_type", "child_text"),
                    parent_id=payload.get("parent_id"),
                    payload=payload,
                )
            )

        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_chunks[:top_n]
