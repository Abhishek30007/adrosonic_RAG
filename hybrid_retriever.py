"""Phase 2 Hybrid Retriever: Dense + BM25 Sparse Search with Reciprocal Rank Fusion (RRF).

Provides sub-300ms hybrid search, pre-retrieval metadata filtering,
and live index CRUD operations (upsert / delete) against Qdrant.
"""

from dataclasses import dataclass
import logging
import os
import sys
import time
from typing import Any, Literal

# pyrefly: ignore [missing-import]
from fastembed import SparseTextEmbedding, TextEmbedding
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.http import models as rest_models

from config import config

# Configure structured console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("HybridRetriever")


@dataclass
class RetrievedDocument:
    """Standardized retrieved passage representation."""

    id: int | str
    score: float
    text: str
    metadata: dict[str, Any]
    latency_ms: float
    method: str = "Hybrid (RRF)"


class QdrantHybridRetriever:
    """Production-grade hybrid retriever fusing Dense (BGE) and Sparse (BM25) via Qdrant RRF."""

    def __init__(
        self,
        collection_name: str = config.HYBRID_COLLECTION_NAME,
        dense_model_name: str = config.EMBEDDING_MODEL_NAME,
        sparse_model_name: str = config.SPARSE_MODEL_NAME,
        qdrant_url: str | None = config.QDRANT_URL,
        qdrant_api_key: str | None = config.QDRANT_API_KEY,
        qdrant_path: str = config.QDRANT_STORAGE_PATH,
        qdrant_host: str | None = config.QDRANT_HOST,
        qdrant_port: int = config.QDRANT_PORT,
    ):
        self.collection_name = collection_name
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name

        # 1. Connect to Qdrant Database (Cloud vs Remote vs Local Fallback)
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

        # 2. Initialize Dual Embedding Models (FastEmbed ONNX CPU)
        logger.info("Initializing Dense model: %s", self.dense_model_name)
        self.dense_model = TextEmbedding(
            model_name=self.dense_model_name,
            threads=config.EMBED_NUM_THREADS,
        )

        logger.info("Initializing Sparse model: %s", self.sparse_model_name)
        self.sparse_model = SparseTextEmbedding(
            model_name=self.sparse_model_name,
            threads=config.EMBED_NUM_THREADS,
        )

        self._verify_collection()

    def _verify_collection(self) -> None:
        """Verify collection exists in Qdrant and has points; fall back gracefully if empty."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            
            # Check if target collection exists and has points
            has_points = False
            if self.collection_name in collections:
                pts = getattr(self.client.get_collection(self.collection_name), "points_count", 0) or 0
                has_points = pts > 0

            # If current collection is empty or missing, check alternative collections or local storage
            if not has_points:
                if config.COLLECTION_NAME in collections:
                    pts = getattr(self.client.get_collection(config.COLLECTION_NAME), "points_count", 0) or 0
                    if pts > 0:
                        logger.info("Using populated collection '%s' (%d points)", config.COLLECTION_NAME, pts)
                        self.collection_name = config.COLLECTION_NAME
                        return

                # Check local storage directory
                if os.path.exists(config.QDRANT_STORAGE_PATH):
                    try:
                        local_client = QdrantClient(path=config.QDRANT_STORAGE_PATH)
                        local_colls = [c.name for c in local_client.get_collections().collections]
                        for c_name in [config.HYBRID_COLLECTION_NAME, config.COLLECTION_NAME]:
                            if c_name in local_colls:
                                pts = getattr(local_client.get_collection(c_name), "points_count", 0) or 0
                                if pts > 0:
                                    logger.info("Falling back to local Qdrant collection '%s' (%d points)", c_name, pts)
                                    self.client = local_client
                                    self.collection_name = c_name
                                    return
                    except Exception:
                        pass

            if self.collection_name not in collections:
                if config.COLLECTION_NAME in collections:
                    self.collection_name = config.COLLECTION_NAME
                else:
                    logger.warning("Collection '%s' does not exist yet.", self.collection_name)
        except Exception as e:
            logger.error("Failed to verify Qdrant collection status: %s", str(e))

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

    def _embed_dense_query(self, query: str) -> list[float]:
        """Embed dense query with LRU memory caching for sub-millisecond repeated queries."""
        if not hasattr(self, "_dense_cache"):
            self._dense_cache: dict[str, list[float]] = {}
        if query not in self._dense_cache:
            if len(self._dense_cache) > 20000:
                self._dense_cache.clear()
            self._dense_cache[query] = list(self.dense_model.embed([query]))[0].tolist()
        return self._dense_cache[query]

    def _embed_sparse_query(self, query: str) -> rest_models.SparseVector:
        """Embed sparse query with memory caching."""
        if not hasattr(self, "_sparse_cache"):
            self._sparse_cache: dict[str, tuple[list[int], list[float]]] = {}
        if query not in self._sparse_cache:
            if len(self._sparse_cache) > 20000:
                self._sparse_cache.clear()
            sp_res = list(self.sparse_model.embed([query]))[0]
            self._sparse_cache[query] = (sp_res.indices.tolist(), sp_res.values.tolist())
        indices, values = self._sparse_cache[query]
        return rest_models.SparseVector(indices=indices, values=values)

    def search(
        self,
        query: str,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
        category_filter: str | None = None,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> list[RetrievedDocument]:
        """Execute search using Hybrid RRF Fusion, Pure Dense, or Pure Sparse.

        Args:
            query: Natural language query string.
            mode: Retrieval strategy ('hybrid', 'dense', or 'sparse').
            category_filter: Optional category metadata filter (e.g., 'tech', 'science').
            top_k: Number of ranked passages to return.

        Returns:
            List of `RetrievedDocument` with normalized scores and latency telemetry.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to search.")
            return []

        start_time = time.perf_counter()
        qdrant_filter = self._build_filter(category_filter)

        try:
            embed_start = time.perf_counter()

            # Generate query embeddings with caching
            dense_vec: list[float] | None = None
            sparse_vec: rest_models.SparseVector | None = None

            if mode in ("hybrid", "dense"):
                dense_vec = self._embed_dense_query(query)

            if mode in ("hybrid", "sparse"):
                sparse_vec = self._embed_sparse_query(query)

            embed_latency = (time.perf_counter() - embed_start) * 1000

            # Detect collection configuration
            try:
                coll_info = self.client.get_collection(self.collection_name)
                has_sparse = bool(coll_info.config.params.sparse_vectors)
                named_vectors = isinstance(coll_info.config.params.vectors, dict)
            except Exception:
                has_sparse = "hybrid" in self.collection_name
                named_vectors = "hybrid" in self.collection_name

            # Execute Qdrant Query
            search_start = time.perf_counter()
            method_label = "Dense"

            if mode == "hybrid" and has_sparse and dense_vec is not None and sparse_vec is not None:
                method_label = "Hybrid (RRF)"
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        rest_models.Prefetch(
                            query=dense_vec,
                            using="dense" if named_vectors else None,
                            limit=top_k * 3,
                            filter=qdrant_filter,
                        ),
                        rest_models.Prefetch(
                            query=sparse_vec,
                            using="sparse",
                            limit=top_k * 3,
                            filter=qdrant_filter,
                        ),
                    ],
                    query=rest_models.FusionQuery(fusion=rest_models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
                search_results = response.points

            elif mode == "sparse" and has_sparse and sparse_vec is not None:
                method_label = "Sparse (BM25)"
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                search_results = response.points

            else:
                method_label = "Dense" if mode == "dense" else "Hybrid (Dense-Fallback)"
                if hasattr(self.client, "query_points"):
                    response = self.client.query_points(
                        collection_name=self.collection_name,
                        query=dense_vec,
                        using="dense" if named_vectors else None,
                        limit=top_k,
                        query_filter=qdrant_filter,
                        with_payload=True,
                    )
                    search_results = response.points
                else:
                    search_results = getattr(self.client, "search")(
                        collection_name=self.collection_name,
                        query_vector=dense_vec,
                        limit=top_k,
                        query_filter=qdrant_filter,
                        with_payload=True,
                    )

            qdrant_latency = (time.perf_counter() - search_start) * 1000
            total_latency_ms = (time.perf_counter() - start_time) * 1000

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
                        method=method_label,
                    )
                )

            sla_status = (
                "PASS" if total_latency_ms <= config.LATENCY_TARGET_P95_MS else "WARN"
            )
            logger.info(
                "[HYBRID %s][%s] '%s' | Total: %.2f ms (Embed: %.2f ms, Qdrant: %.2f ms) | Filter: %s | Retrieved: %d",
                sla_status,
                method_label,
                query[:35] + ("..." if len(query) > 35 else ""),
                total_latency_ms,
                embed_latency,
                qdrant_latency,
                category_filter or "None",
                len(retrieved_docs),
            )

            return retrieved_docs

        except Exception as e:
            logger.exception("Error during hybrid retrieval for '%s': %s", query, str(e))
            return []

    def search_detailed(
        self,
        query: str,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
        category_filter: str | None = None,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> tuple[list[RetrievedDocument], dict[str, float]]:
        """Execute search with discrete sub-millisecond profiling for every pipeline stage.

        Returns:
            (retrieved_docs, telemetry_dict) where telemetry_dict contains:
            - dense_embed_ms
            - sparse_embed_ms
            - qdrant_search_ms
            - rrf_fusion_ms
            - retrieval_total_ms
        """
        if not query or not query.strip():
            return [], {
                "dense_embed_ms": 0.0,
                "sparse_embed_ms": 0.0,
                "qdrant_search_ms": 0.0,
                "rrf_fusion_ms": 0.0,
                "retrieval_total_ms": 0.0,
            }

        start_time = time.perf_counter()
        dense_embed_ms = 0.0
        sparse_embed_ms = 0.0
        qdrant_search_ms = 0.0
        rrf_fusion_ms = 0.0

        qdrant_filter = self._build_filter(category_filter)

        try:
            # 1. Profile Dense Embedding
            if mode in ("hybrid", "dense"):
                t_dense = time.perf_counter()
                dense_vec = self._embed_dense_query(query)
                dense_embed_ms = (time.perf_counter() - t_dense) * 1000
            else:
                dense_vec = None

            # 2. Profile Sparse BM25 Vectorization
            if mode in ("hybrid", "sparse"):
                t_sparse = time.perf_counter()
                sparse_vec = self._embed_sparse_query(query)
                sparse_embed_ms = (time.perf_counter() - t_sparse) * 1000
            else:
                sparse_vec = None

            # 3. Detect collection config
            try:
                coll_info = self.client.get_collection(self.collection_name)
                has_sparse = bool(coll_info.config.params.sparse_vectors)
                named_vectors = isinstance(coll_info.config.params.vectors, dict)
            except Exception:
                has_sparse = "hybrid" in self.collection_name
                named_vectors = "hybrid" in self.collection_name

            # 4. Profile Pure Qdrant Search & Fusion
            t_qdrant = time.perf_counter()
            method_label = "Dense"

            if mode == "hybrid" and has_sparse and dense_vec is not None and sparse_vec is not None:
                method_label = "Hybrid (RRF)"
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        rest_models.Prefetch(
                            query=dense_vec,
                            using="dense" if named_vectors else None,
                            limit=top_k * 3,
                            filter=qdrant_filter,
                        ),
                        rest_models.Prefetch(
                            query=sparse_vec,
                            using="sparse",
                            limit=top_k * 3,
                            filter=qdrant_filter,
                        ),
                    ],
                    query=rest_models.FusionQuery(fusion=rest_models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
                search_results = response.points

            elif mode == "sparse" and has_sparse and sparse_vec is not None:
                method_label = "Sparse (BM25)"
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                search_results = response.points

            else:
                method_label = "Dense" if mode == "dense" else "Hybrid (Dense-Fallback)"
                if hasattr(self.client, "query_points"):
                    response = self.client.query_points(
                        collection_name=self.collection_name,
                        query=dense_vec,
                        using="dense" if named_vectors else None,
                        limit=top_k,
                        query_filter=qdrant_filter,
                        with_payload=True,
                    )
                    search_results = response.points
                else:
                    search_results = getattr(self.client, "search")(
                        collection_name=self.collection_name,
                        query_vector=dense_vec,
                        limit=top_k,
                        query_filter=qdrant_filter,
                        with_payload=True,
                    )

            qdrant_search_ms = (time.perf_counter() - t_qdrant) * 1000

            # 5. Profile RRF Parsing and In-Memory Post-Processing
            t_rrf = time.perf_counter()
            retrieved_docs: list[RetrievedDocument] = []
            total_latency_ms = (time.perf_counter() - start_time) * 1000

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
                        method=method_label,
                    )
                )
            rrf_fusion_ms = (time.perf_counter() - t_rrf) * 1000
            retrieval_total_ms = (time.perf_counter() - start_time) * 1000

            telemetry = {
                "dense_embed_ms": dense_embed_ms,
                "sparse_embed_ms": sparse_embed_ms,
                "qdrant_search_ms": qdrant_search_ms,
                "rrf_fusion_ms": rrf_fusion_ms,
                "retrieval_total_ms": retrieval_total_ms,
            }

            return retrieved_docs, telemetry

        except Exception as e:
            logger.exception("Error during hybrid retrieval with profiling: %s", str(e))
            return [], {
                "dense_embed_ms": dense_embed_ms,
                "sparse_embed_ms": sparse_embed_ms,
                "qdrant_search_ms": qdrant_search_ms,
                "rrf_fusion_ms": rrf_fusion_ms,
                "retrieval_total_ms": (time.perf_counter() - start_time) * 1000,
            }

    def upsert_document(
        self,
        passage_id: int | str,
        text: str,
        source: str = "manual_crud",
        category: str = "general",
    ) -> bool:
        """Embed and upsert a document directly into the live index.

        Args:
            passage_id: Unique integer or UUID document identifier.
            text: Raw document text to embed and store.
            source: Document origin / source URL.
            category: Domain classification tag.

        Returns:
            True if upsert succeeded, False otherwise.
        """
        clean_text = text.strip()
        if not clean_text:
            logger.error("Cannot upsert empty document text.")
            return False

        try:
            logger.info("Embedding and upserting document ID: %s into %s", passage_id, self.collection_name)
            dense_embedding = list(self.dense_model.embed([clean_text]))[0].tolist()
            sp_res = list(self.sparse_model.embed([clean_text]))[0]

            sparse_vec = rest_models.SparseVector(
                indices=sp_res.indices.tolist(),
                values=sp_res.values.tolist(),
            )

            payload = {
                "text": clean_text,
                "passage_id": passage_id,
                "source": source,
                "category": category.strip().lower(),
                "category_tag": category.strip().lower(),
                "updated_at": time.time(),
            }

            # Support hybrid dual vector collection or dense-only fallback
            if "hybrid" in self.collection_name:
                vector_data = {
                    "dense": dense_embedding,
                    "sparse": sparse_vec,
                }
            else:
                vector_data = dense_embedding

            point = rest_models.PointStruct(
                id=int(passage_id) if str(passage_id).isdigit() else str(passage_id),
                vector=vector_data,
                payload=payload,
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True,
            )
            logger.info("Successfully upserted document ID: %s", passage_id)
            return True

        except Exception as e:
            logger.exception("Failed to upsert document ID %s: %s", passage_id, str(e))
            return False

    def delete_document(self, passage_id: int | str) -> bool:
        """Delete a document by its ID directly from the live index.

        Args:
            passage_id: Document ID to delete.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        try:
            logger.info("Deleting document ID: %s from %s", passage_id, self.collection_name)
            point_id = int(passage_id) if str(passage_id).isdigit() else str(passage_id)

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.PointIdsList(points=[point_id]),
                wait=True,
            )
            logger.info("Successfully deleted document ID: %s", passage_id)
            return True

        except Exception as e:
            logger.exception("Failed to delete document ID %s: %s", passage_id, str(e))
            return False


if __name__ == "__main__":
    retriever = QdrantHybridRetriever()
    test_q = "what is the normal resting heart rate for adults"
    print(f"\n--- Testing QdrantHybridRetriever with Query: '{test_q}' ---")
    results = retriever.search(test_q, mode="hybrid", top_k=3)
    for r in results:
        print(f"[{r.method}] ID: {r.id} | Score: {r.score:.4f} | Text: {r.text[:80]}...")
