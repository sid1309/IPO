"""Hybrid text embedder producing dense vectors (Gemini Cloud API or FastEmbed) and sparse lexical weights (BM25)."""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class HybridEmbeddingOutput:
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]


class HybridEmbedder:
    def __init__(
        self,
        dense_model_name: Optional[str] = None,
        sparse_model_name: str = "Qdrant/bm25",
        provider: Optional[str] = None,
    ):
        self.provider = (provider or getattr(settings, "EMBEDDING_PROVIDER", "gemini")).lower()
        self.gemini_client = None
        self.dense_model = None

        # 1. Initialize Cloud Gemini Provider if configured (0 local RAM footprint)
        if self.provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
                self.dense_dimension = getattr(settings, "EMBEDDING_DIMENSION", 768)
                model_candidate = getattr(settings, "EMBEDDING_MODEL_NAME", "gemini-embedding-001")
                if not model_candidate or "bge" in model_candidate.lower() or "baai" in model_candidate.lower():
                    self.cloud_model_name = "gemini-embedding-001"
                else:
                    self.cloud_model_name = model_candidate
                logger.info(f"Initialized Cloud Gemini Embeddings ({self.cloud_model_name}, dim={self.dense_dimension}) - 0 RAM usage.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini cloud embeddings, falling back to local: {e}")
                self.gemini_client = None

        # 2. Fall back to local FastEmbed ONNX model if cloud is not active
        if self.gemini_client is None:
            from fastembed import TextEmbedding
            model_name = dense_model_name or "BAAI/bge-small-en-v1.5"
            if model_name.lower() in ("baai/bge-m3", "bge-m3", "gemini-embedding-001"):
                model_name = "BAAI/bge-small-en-v1.5"
            logger.info(f"Loading local FastEmbed model: {model_name}")
            self.dense_model = TextEmbedding(model_name=model_name)
            sample_vec = list(self.dense_model.embed(["test"]))[0]
            self.dense_dimension = len(sample_vec)

        # 3. Fast lightweight BM25 sparse tokenizer (~5 MB RAM, no heavy neural network)
        from fastembed import SparseTextEmbedding
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    def _embed_dense_cloud(self, texts: List[str]) -> List[List[float]]:
        from google.genai import types
        all_embeddings: List[List[float]] = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                res = self.gemini_client.models.embed_content(
                    model=self.cloud_model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=self.dense_dimension),
                )
                for emb in res.embeddings:
                    all_embeddings.append(emb.values)
            except Exception as e:
                logger.error(f"Gemini embedding batch failed: {e}")
                # Fallback to zero vector if individual call fails
                for _ in batch:
                    all_embeddings.append([0.0] * self.dense_dimension)

        return all_embeddings

    def embed_texts(self, texts: List[str]) -> List[HybridEmbeddingOutput]:
        if not texts:
            return []

        # Dense embeddings: cloud API or local FastEmbed
        if self.gemini_client is not None:
            dense_embeddings = self._embed_dense_cloud(texts)
        else:
            dense_raw = list(self.dense_model.embed(texts))
            dense_embeddings = [
                d.tolist() if hasattr(d, "tolist") else list(d) for d in dense_raw
            ]

        # Sparse embeddings (BM25)
        sparse_embeddings = list(self.sparse_model.embed(texts))

        outputs = []
        for dense_vec, sparse in zip(dense_embeddings, sparse_embeddings):
            sparse_idx = (
                sparse.indices.tolist()
                if hasattr(sparse.indices, "tolist")
                else list(sparse.indices)
            )
            sparse_val = (
                sparse.values.tolist()
                if hasattr(sparse.values, "tolist")
                else list(sparse.values)
            )

            outputs.append(
                HybridEmbeddingOutput(
                    dense_vector=dense_vec,
                    sparse_indices=sparse_idx,
                    sparse_values=sparse_val,
                )
            )

        return outputs

    def embed_query(self, query: str) -> HybridEmbeddingOutput:
        results = self.embed_texts([query])
        return results[0]
