"""Hybrid text embedder producing dense vectors (FastEmbed) and sparse lexical weights (BM25)."""

import os
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding
from app.core.config import settings

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
    ):
        model_name = dense_model_name or settings.EMBEDDING_MODEL_NAME
        if model_name.lower() in ("baai/bge-m3", "bge-m3"):
            model_name = "BAAI/bge-small-en-v1.5"

        self.dense_model = TextEmbedding(model_name=model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

        sample_vec = list(self.dense_model.embed(["test"]))[0]
        self.dense_dimension = len(sample_vec)

    def embed_texts(self, texts: List[str]) -> List[HybridEmbeddingOutput]:
        if not texts:
            return []

        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.sparse_model.embed(texts))

        outputs = []
        for dense, sparse in zip(dense_embeddings, sparse_embeddings):
            dense_vec = dense.tolist() if hasattr(dense, "tolist") else list(dense)
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
