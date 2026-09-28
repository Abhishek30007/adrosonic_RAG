"""Fast & Resilient Cloud Sync with Retry and Timeout."""

import logging
import sys
import time
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from tqdm import tqdm

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CloudSync")


def main():
    logger.info("Opening local Qdrant storage at: %s", config.QDRANT_STORAGE_PATH)
    local_client = QdrantClient(path=config.QDRANT_STORAGE_PATH)

    logger.info("Connecting to Qdrant Cloud with 60s timeout at: %s", config.QDRANT_URL)
    cloud_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)

    # Initialize Sparse Embedding model
    logger.info("Initializing SparseTextEmbedding (%s)...", config.SPARSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL_NAME)

    # Ensure collection exists on Cloud
    cloud_collections = [c.name for c in cloud_client.get_collections().collections]
    if config.HYBRID_COLLECTION_NAME not in cloud_collections:
        logger.info("Creating collection '%s' on Qdrant Cloud...", config.HYBRID_COLLECTION_NAME)
        cloud_client.create_collection(
            collection_name=config.HYBRID_COLLECTION_NAME,
            vectors_config={
                "dense": rest_models.VectorParams(
                    size=config.VECTOR_DIMENSION,
                    distance=rest_models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": rest_models.SparseVectorParams(
                    modifier=rest_models.Modifier.IDF,
                )
            },
        )

    try:
        cloud_client.create_payload_index(
            collection_name=config.HYBRID_COLLECTION_NAME,
            field_name="category",
            field_schema=rest_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass

    # Sync in batches of 64
    limit = 64
    offset = None
    total_synced = 0
    pbar = tqdm(total=8448, desc="Syncing points to Qdrant Cloud", unit="pts")

    while True:
        records, next_offset = local_client.scroll(
            collection_name="msmarco_100k_dense",
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        texts = [r.payload.get("text", "") for r in records]
        sparse_embeddings = list(sparse_model.embed(texts))

        cloud_points = []
        for r, sp in zip(records, sparse_embeddings):
            dense_vec = r.vector
            sparse_vec = rest_models.SparseVector(
                indices=sp.indices.tolist(),
                values=sp.values.tolist(),
            )
            payload = dict(r.payload or {})
            if "category" not in payload:
                payload["category"] = "general"

            cloud_points.append(
                rest_models.PointStruct(
                    id=r.id,
                    vector={
                        "dense": dense_vec,
                        "sparse": sparse_vec,
                    },
                    payload=payload,
                )
            )

        for attempt in range(3):
            try:
                cloud_client.upsert(
                    collection_name=config.HYBRID_COLLECTION_NAME,
                    points=cloud_points,
                    wait=False,
                )
                break
            except Exception as e:
                if attempt == 2:
                    logger.error("Failed batch after 3 attempts: %s", str(e))
                time.sleep(1)

        total_synced += len(cloud_points)
        pbar.update(len(cloud_points))

        if next_offset is None:
            break
        offset = next_offset

    pbar.close()
    logger.info("Successfully synced %d passages to Qdrant Cloud cluster!", total_synced)


if __name__ == "__main__":
    main()
