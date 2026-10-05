"""Hybrid Qdrant indexer storing dense and sparse vectors with RRF retrieval."""

import logging
from typing import Any, List, Optional
from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.ingestion.chunker import DocumentChunk
from app.ingestion.embed import HybridEmbedder, HybridEmbeddingOutput

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ipo_prospectus"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantIndexer:
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[HybridEmbedder] = None,
        storage_path: Optional[str] = None,
    ):
        self.embedder = embedder or HybridEmbedder()

        if client is not None:
            self.client = client
        elif settings.QDRANT_URL:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )
        else:
            path = storage_path or settings.QDRANT_STORAGE_DIR
            try:
                self.client = QdrantClient(path=path)
            except Exception as e:
                err_msg = str(e).lower()
                if "already accessed" in err_msg or "alreadylocked" in err_msg or "permission denied" in err_msg:
                    logger.warning("Qdrant disk storage locked by another process; falling back to in-memory client.")
                    self.client = QdrantClient(":memory:")
                else:
                    raise e

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)

        if exists:
            try:
                coll_info = self.client.get_collection(COLLECTION_NAME)
                vectors_cfg = coll_info.config.params.vectors
                dense_cfg = vectors_cfg.get(DENSE_VECTOR_NAME) if isinstance(vectors_cfg, dict) else vectors_cfg
                current_size = getattr(dense_cfg, "size", None)
                if current_size is not None and current_size != self.embedder.dense_dimension:
                    logger.info(f"Vector size changed ({current_size} -> {self.embedder.dense_dimension}). Re-creating collection '{COLLECTION_NAME}'...")
                    self.client.delete_collection(COLLECTION_NAME)
                    exists = False
            except Exception as e:
                logger.warning(f"Could not verify existing collection vector dimension: {e}")

        if not exists:
            logger.info(f"Creating Qdrant hybrid collection '{COLLECTION_NAME}'...")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.embedder.dense_dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )

            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="ipo_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="section",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="chunk_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        if not chunks:
            return 0

        total_indexed = 0
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            if batch_num % 5 == 1 or batch_num == total_batches:
                logger.info(
                    f"Hybrid Indexing batch {batch_num}/{total_batches} ({total_indexed}/{len(chunks)} chunks)..."
                )

            texts_to_embed = [
                c.table_summary if c.chunk_type == "table" and c.table_summary else c.text
                for c in batch
            ]
            embeddings = self.embedder.embed_texts(texts_to_embed)

            points = []
            for chunk, emb in zip(batch, embeddings):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "ipo_id": chunk.ipo_id,
                    "company_name": chunk.company_name,
                    "doc_type": chunk.doc_type,
                    "doc_version": chunk.doc_version,
                    "section": chunk.section,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_type": chunk.chunk_type,
                    "text": chunk.text,
                    "parent_id": chunk.parent_id,
                    "table_summary": chunk.table_summary,
                }

                import uuid
                try:
                    point_id = str(uuid.UUID(chunk.chunk_id))
                except Exception:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: emb.dense_vector,
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=emb.sparse_indices,
                            values=emb.sparse_values,
                        ),
                    },
                    payload=payload,
                )
                points.append(point)

            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_indexed += len(points)

        return total_indexed

    def hybrid_search(
        self,
        query: str,
        ipo_id: str,
        section: Optional[str] = None,
        top_k: int = 15,
        limit: Optional[int] = None,
    ) -> List[Any]:
        if limit is not None:
            top_k = limit

        query_emb = self.embedder.embed_query(query)

        must_conditions = [
            models.FieldCondition(
                key="ipo_id",
                match=models.MatchValue(value=ipo_id),
            )
        ]

        if section:
            must_conditions.append(
                models.FieldCondition(
                    key="section",
                    match=models.MatchValue(value=section),
                )
            )

        query_filter = models.Filter(must=must_conditions)

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=query_emb.dense_vector,
                    using=DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_emb.sparse_indices,
                        values=query_emb.sparse_values,
                    ),
                    using=SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        return results.points
