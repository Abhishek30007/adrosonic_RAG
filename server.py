"""High-Throughput Enterprise FastAPI Server for RAG Retrieval.

Provides REST endpoints for Dense, Sparse, and Hybrid RRF search,
health checks, and SLA metrics telemetry.
"""

from contextlib import asynccontextmanager
import logging
import sys
import time
from typing import Any, Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import config
from hybrid_retriever import QdrantHybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RAG_API_Server")

retriever: QdrantHybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Initializing QdrantHybridRetriever instance...")
    retriever = QdrantHybridRetriever()
    logger.info("QdrantHybridRetriever initialized and ready for traffic.")
    yield
    logger.info("Shutting down RAG API Server...")


app = FastAPI(
    title="Enterprise RAG Retrieval Service",
    description="High-Throughput Dense & Hybrid Search API with FastEmbed and Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., example="what is machine learning", min_length=1)
    mode: Literal["hybrid", "dense", "sparse"] = Field("hybrid", example="hybrid")
    top_k: int = Field(5, ge=1, le=50)
    category_filter: str | None = Field(None, example="tech")


class SearchHit(BaseModel):
    id: int | str
    score: float
    text: str
    metadata: dict[str, Any]
    method: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    results_count: int
    latency_ms: float
    sla_pass: bool
    results: list[SearchHit]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "collection": retriever.collection_name if retriever else "uninitialized",
        "sla_target_p95_ms": config.LATENCY_TARGET_P95_MS,
    }


@app.post("/search", response_model=SearchResponse)
def search_post(payload: SearchRequest):
    if retriever is None:
        raise RuntimeError("Retriever is not initialized")

    start = time.perf_counter()
    docs = retriever.search(
        query=payload.query,
        mode=payload.mode,
        category_filter=payload.category_filter,
        top_k=payload.top_k,
    )
    total_latency_ms = (time.perf_counter() - start) * 1000

    hits = [
        SearchHit(
            id=d.id,
            score=round(d.score, 4),
            text=d.text,
            metadata=d.metadata,
            method=d.method,
        )
        for d in docs
    ]

    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        results_count=len(hits),
        latency_ms=round(total_latency_ms, 2),
        sla_pass=total_latency_ms < config.LATENCY_TARGET_P95_MS,
        results=hits,
    )


@app.get("/search", response_model=SearchResponse)
def search_get(
    q: str = Query(..., min_length=1),
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    top_k: int = 5,
    category: str | None = None,
):
    req = SearchRequest(query=q, mode=mode, top_k=top_k, category_filter=category)
    return search_post(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1)
