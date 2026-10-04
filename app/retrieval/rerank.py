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
    def __init__(self, model_name: Optional[str] = None, top_n: int = 5):
        self.top_n = top_n
        self.model = None

        if not model_name:
            try:
                from app.core.config import settings
                model_name = getattr(settings, "RERANKER_MODEL_NAME", "Xenova/ms-marco-MiniLM-L-6-v2")
            except Exception:
                model_name = "Xenova/ms-marco-MiniLM-L-6-v2"

        # Map unsupported names to FastEmbed supported models
        # Xenova/ms-marco-MiniLM-L-6-v2 is ultra-lightweight (80MB) and fits inside free cloud tiers (512MB RAM)
        target_model = model_name
        if "v2-m3" in str(model_name):
            target_model = "Xenova/ms-marco-MiniLM-L-6-v2"

        for candidate in [target_model, "Xenova/ms-marco-MiniLM-L-6-v2"]:
            try:
                self.model = TextCrossEncoder(model_name=candidate)
                break
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Could not load reranker '{candidate}': {e}")

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

        if self.model is not None:
            try:
                scores = list(self.model.rerank(query, documents))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Reranking error, using candidate scores: {e}")
                scores = [getattr(pt, "score", 1.0) for pt in candidate_points]
        else:
            scores = [getattr(pt, "score", 1.0) for pt in candidate_points]

        scored_chunks = []
        for pt, score in zip(candidate_points, scores):
            payload = getattr(pt, "payload", {}) or {}
            scored_chunks.append(
                RerankedChunk(
                    chunk_id=str(getattr(pt, "id", "")),
                    text=payload.get("text", ""),
                    relevance_score=float(score) if score is not None else 0.0,
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
