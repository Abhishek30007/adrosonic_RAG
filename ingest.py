"""Ingestion Pipeline for Phase 1 RAG: Indexing 100,000+ MS MARCO passages into Qdrant.

Uses FastEmbed (BAAI/bge-small-en-v1.5) and Qdrant local disk persistence or Docker server.
Optimized for high-throughput CPU vector indexing with streaming batch ingestion.
"""

from collections.abc import Generator
import logging
import os
import sys
import time
from typing import Any

from datasets import load_dataset
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from tqdm import tqdm

from config import config

# Configure structured console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("IngestionPipeline")


def get_qdrant_client() -> QdrantClient:
    """Initialize a high-performance Qdrant client based on configuration.

    Connects to Qdrant Cloud Cluster if QDRANT_URL and QDRANT_API_KEY are configured,
    or falls back gracefully to local file persistence.
    """
    if config.QDRANT_URL and config.QDRANT_API_KEY:
        logger.info("Connecting to Qdrant Cloud Cluster at: %s", config.QDRANT_URL)
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    if config.QDRANT_HOST:
        logger.info(
            "Connecting to remote Qdrant instance at %s:%s",
            config.QDRANT_HOST,
            config.QDRANT_PORT,
        )
        return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    logger.info("Initializing local Qdrant storage at: %s", config.QDRANT_STORAGE_PATH)
    os.makedirs(config.QDRANT_STORAGE_PATH, exist_ok=True)
    return QdrantClient(path=config.QDRANT_STORAGE_PATH)


def init_collection(client: QdrantClient, recreate: bool = True) -> None:
    """Ensure the target collection exists with exact Cosine distance configuration."""
    collections = [c.name for c in client.get_collections().collections]

    if config.COLLECTION_NAME in collections:
        if recreate:
            logger.warning(
                "Collection '%s' already exists. Recreating collection for fresh index...",
                config.COLLECTION_NAME,
            )
            client.delete_collection(collection_name=config.COLLECTION_NAME)
        else:
            logger.info(
                "Collection '%s' already exists. Preserving existing data.",
                config.COLLECTION_NAME,
            )
            return

    logger.info(
        "Creating Qdrant collection: '%s' (Vector Dimension: %d, Metric: %s)",
        config.COLLECTION_NAME,
        config.VECTOR_DIMENSION,
        config.DISTANCE_METRIC,
    )
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=rest_models.VectorParams(
            size=config.VECTOR_DIMENSION,
            distance=rest_models.Distance.COSINE,
            on_disk=True,  # Keeps memory footprint small on CPU architectures
        ),
        optimizers_config=rest_models.OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=20000,
        ),
    )


def stream_msmarco_passages(
    target_count: int,
) -> Generator[tuple[int, str, dict[str, Any]], None, None]:
    """Stream passages from Hugging Face MS MARCO dataset to avoid RAM saturation.

    Yields:
        Tuple of (passage_id, passage_text, metadata_dict)
    """
    logger.info(
        "Loading Hugging Face dataset: %s (%s, split: %s) in streaming mode...",
        config.DATASET_NAME,
        config.DATASET_CONFIG,
        config.DATASET_SPLIT,
    )
    dataset = load_dataset(
        config.DATASET_NAME,
        config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
        streaming=True,
    )

    yielded_count = 0

    for item in dataset:
        passages_data = item.get("passages", {})
        passage_texts = passages_data.get("passage_text", [])
        is_selected_list = passages_data.get("is_selected", [])
        query_id = item.get("query_id", "")
        query_text = item.get("query", "")

        categories = ["general", "tech", "science", "finance", "health", "news"]

        for idx, text in enumerate(passage_texts):
            clean_text = text.strip()
            if not clean_text:
                continue

            category_tag = categories[yielded_count % len(categories)]
            metadata = {
                "passage_id": yielded_count,
                "query_id": query_id,
                "associated_query": query_text,
                "is_selected": (
                    bool(is_selected_list[idx]) if idx < len(is_selected_list) else False
                ),
                "source": f"https://microsoft.com/msmarco/passages/{yielded_count}",
                "category": category_tag,
                "category_tag": category_tag,
            }

            yield (yielded_count, clean_text, metadata)
            yielded_count += 1

            if yielded_count >= target_count:
                return


def run_ingestion(
    target_count: int = config.TARGET_INGESTION_COUNT,
    batch_size: int = config.EMBED_BATCH_SIZE,
) -> None:
    """Execute the full batch embedding and indexing pipeline."""
    start_time = time.perf_counter()
    logger.info("=== STARTING PHASE 1 INGESTION: %d Passages ===", target_count)

    # 1. Initialize Vector DB Client & Collection
    client = get_qdrant_client()
    init_collection(client, recreate=True)

    # 2. Initialize FastEmbed Model (ONNX optimized for CPU)
    logger.info("Loading FastEmbed model: %s", config.EMBEDDING_MODEL_NAME)
    embedding_model = TextEmbedding(
        model_name=config.EMBEDDING_MODEL_NAME,
        threads=config.EMBED_NUM_THREADS,
    )

    # 3. Stream, Embed, and Ingest in Batches
    passage_stream = stream_msmarco_passages(target_count=target_count)

    batch_ids: list[int] = []
    batch_texts: list[str] = []
    batch_payloads: list[dict[str, Any]] = []

    indexed_count = 0

    with tqdm(
        total=target_count,
        desc="Indexing Passages (FastEmbed + Qdrant)",
        unit="docs",
    ) as pbar:
        for p_id, text, meta in passage_stream:
            batch_ids.append(p_id)
            batch_texts.append(text)
            batch_payloads.append({"text": text, **meta})

            if len(batch_texts) >= batch_size:
                # Compute dense embeddings on CPU via ONNX Runtime
                embeddings = list(
                    embedding_model.embed(batch_texts, batch_size=batch_size)
                )

                # Upsert into Qdrant
                points = [
                    rest_models.PointStruct(
                        id=batch_ids[i],
                        vector=embeddings[i].tolist(),
                        payload=batch_payloads[i],
                    )
                    for i in range(len(batch_ids))
                ]

                client.upsert(
                    collection_name=config.COLLECTION_NAME,
                    points=points,
                    wait=False,
                )

                indexed_count += len(batch_texts)
                pbar.update(len(batch_texts))

                # Reset batch buffers
                batch_ids.clear()
                batch_texts.clear()
                batch_payloads.clear()

        # Ingest remaining items
        if batch_texts:
            embeddings = list(
                embedding_model.embed(batch_texts, batch_size=batch_size)
            )
            points = [
                rest_models.PointStruct(
                    id=batch_ids[i],
                    vector=embeddings[i].tolist(),
                    payload=batch_payloads[i],
                )
                for i in range(len(batch_ids))
            ]
            client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points,
                wait=True,
            )
            indexed_count += len(batch_texts)
            pbar.update(len(batch_texts))

    total_time = time.perf_counter() - start_time
    docs_per_sec = indexed_count / total_time if total_time > 0 else 0

    logger.info("=== INGESTION SUMMARY ===")
    logger.info("Successfully indexed %d passages into Qdrant", indexed_count)
    logger.info("Total execution time: %.2f seconds (%.2f minutes)", total_time, total_time / 60)
    logger.info("Average throughput: %.2f passages/sec", docs_per_sec)
    logger.info("Target SLA (<2 hours for 100k): %s", "PASSED" if total_time < 7200 else "EXCEEDED")


if __name__ == "__main__":
    try:
        run_ingestion()
    except KeyboardInterrupt:
        logger.warning("Ingestion stopped by user.")
    except Exception as e:
        logger.exception("Ingestion failed with unhandled error: %s", str(e))
        sys.exit(1)
