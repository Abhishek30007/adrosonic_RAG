"""Dense Vector Retriever Module for Phase 1 RAG Pipeline.

Embeds natural language queries using FastEmbed (BAAI/bge-small-en-v1.5) and executes
high-speed Cosine similarity search against Qdrant, tracking latency against the p95 < 300ms SLA.
"""

from dataclasses import dataclass
import logging
import os
import sys
import time
from typing import Any
# pyrefly: ignore [missing-import]
from fastembed import TextEmbedding
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.http import models as rest_models

from config import config

# Configure structured console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DenseRetriever")


@dataclass
class RetrievedDocument:
    """Standardized retrieved passage representation."""

    id: int | str
    score: float
    text: str
    metadata: dict[str, Any]
    latency_ms: float


class DenseRetriever:
    """Production-grade dense retriever querying Qdrant using FastEmbed embeddings."""

    def __init__(
        self,
        collection_name: str = config.COLLECTION_NAME,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        qdrant_url: str | None = config.QDRANT_URL,
        qdrant_api_key: str | None = config.QDRANT_API_KEY,
        qdrant_path: str = config.QDRANT_STORAGE_PATH,
        qdrant_host: str | None = config.QDRANT_HOST,
        qdrant_port: int = config.QDRANT_PORT,
    ):
        self.collection_name = collection_name
        self.model_name = model_name

        # 1. Connect to Qdrant Vector Database (Cloud vs Remote vs Local Fallback)
        if qdrant_url and qdrant_api_key:
            logger.info("Connecting to Qdrant Cloud Cluster at: %s", qdrant_url)
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        elif qdrant_host:
            logger.info("Connecting to remote Qdrant at %s:%s", qdrant_host, qdrant_port)
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        else:
            logger.info("Connecting to local Qdrant at: %s", qdrant_path)
            os.makedirs(qdrant_path, exist_ok=True)
            self.client = QdrantClient(path=qdrant_path)

        # 2. Initialize FastEmbed Embedding Model (CPU ONNX accelerated)
        logger.info("Initializing query embedding model: %s", self.model_name)
        self.embed_model = TextEmbedding(
            model_name=self.model_name,
            threads=config.EMBED_NUM_THREADS,
        )

        # 3. Verify collection existence
        self._verify_collection()

    def _verify_collection(self) -> None:
        """Ensure collection exists in Qdrant before executing queries."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.warning(
                    "Collection '%s' does not exist yet. Run `ingest.py` before querying.",
                    self.collection_name,
                )
        except Exception as e:
            logger.error("Failed to verify Qdrant collections: %s", str(e))

    def _build_filter(self, category_filter: str | None) -> rest_models.Filter | None:
        """Construct pre-retrieval metadata filter for payload category attribute."""
        if not category_filter or category_filter.lower() in ("all", "none", ""):
            return None

        return rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="category",
                    match=rest_models.MatchValue(value=category_filter.strip().lower()),
                )
            ]
        )

    def search(
        self,
        query: str,
        category_filter: str | None = None,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[RetrievedDocument]:
        """Perform dense vector search with query latency instrumentation and optional category filtering.

        Args:
            query: Natural language search string.
            category_filter: Optional category metadata filter (e.g., 'tech', 'science').
            top_k: Number of most similar passages to retrieve.

        Returns:
            List of `RetrievedDocument` sorted by descending similarity score.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to DenseRetriever.search.")
            return []

        start_time = time.perf_counter()
        qdrant_filter = self._build_filter(category_filter)

        try:
            # 1. Embed query on CPU
            embed_start = time.perf_counter()
            query_embedding = list(self.embed_model.embed([query]))[0].tolist()
            embed_latency = (time.perf_counter() - embed_start) * 1000

            # 2. Execute Cosine similarity vector search in Qdrant
            search_start = time.perf_counter()
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                search_results = response.points
            else:
                search_results = getattr(self.client, "search")(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
            qdrant_latency = (time.perf_counter() - search_start) * 1000

            # Total round-trip latency
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            # 3. Format retrieved documents
            retrieved_docs: list[RetrievedDocument] = []
            for hit in search_results:
                payload = hit.payload or {}
                text = payload.get("text", "")
                meta = {k: v for k, v in payload.items() if k != "text"}

                retrieved_docs.append(
                    RetrievedDocument(
                        id=hit.id,
                        score=float(hit.score),
                        text=text,
                        metadata=meta,
                        latency_ms=total_latency_ms,
                    )
                )

            # 4. Log latency metrics and SLA verification
            sla_status = (
                "PASS" if total_latency_ms <= config.LATENCY_TARGET_P95_MS else "WARN"
            )
            logger.info(
                "[LATENCY %s] Query: '%s' | Total: %.2f ms (Embed: %.2f ms, Qdrant: %.2f ms) | Filter: %s | Retrieved: %d",
                sla_status,
                query[:40] + ("..." if len(query) > 40 else ""),
                total_latency_ms,
                embed_latency,
                qdrant_latency,
                category_filter or "None",
                len(retrieved_docs),
            )

            return retrieved_docs

        except Exception as e:
            logger.exception("Error during dense retrieval for query '%s': %s", query, str(e))
            return []

    def upsert_document(
        self,
        passage_id: int | str,
        text: str,
        source: str = "manual_crud",
        category: str = "general",
    ) -> bool:
        """Embed and upsert a document directly into the live Dense index without full reindexing.

        Args:
            passage_id: Unique integer or string ID.
            text: Text content to embed and store.
            source: Document source URL or identifier.
            category: Classification category tag.

        Returns:
            True if upsert succeeded, False otherwise.
        """
        clean_text = text.strip()
        if not clean_text:
            logger.error("Cannot upsert empty document text.")
            return False

        try:
            dense_embedding = list(self.embed_model.embed([clean_text]))[0].tolist()
            payload = {
                "text": clean_text,
                "passage_id": passage_id,
                "source": source,
                "category": category.strip().lower(),
                "category_tag": category.strip().lower(),
                "updated_at": time.time(),
            }
            point = rest_models.PointStruct(
                id=int(passage_id) if str(passage_id).isdigit() else str(passage_id),
                vector=dense_embedding,
                payload=payload,
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
            )
            logger.info("Successfully upserted document ID %s into %s", passage_id, self.collection_name)
            return True
        except Exception as e:
            logger.exception("Failed to upsert document ID %s: %s", passage_id, str(e))
            return False

    def delete_document(self, passage_id: int | str) -> bool:
        """Delete a document by ID directly from the live Dense index.

        Args:
            passage_id: Document ID to delete.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        try:
            point_id = int(passage_id) if str(passage_id).isdigit() else str(passage_id)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.PointIdsList(points=[point_id]),
                wait=True,
            )
            logger.info("Successfully deleted document ID %s from %s", passage_id, self.collection_name)
            return True
        except Exception as e:
            logger.exception("Failed to delete document ID %s: %s", passage_id, str(e))
            return False
