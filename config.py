"""Centralized Configuration for Enterprise RAG Pipeline (Phase 1 & Phase 2).

Manages all vector database settings (Local Qdrant vs Qdrant Cloud),
embedding models, dataset parameters, and API credentials with robust fallback.

-------------------------------------------------------------------------------
Streamlit Cloud Secrets Note:
To deploy or test on Streamlit Cloud or locally with secrets, create a file at
`.streamlit/secrets.toml` formatted as follows:

    # .streamlit/secrets.toml
    QDRANT_URL = "https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "your-qdrant-cloud-api-key"
    GROQ_API_KEY = "gsk_your_groq_api_key_here"
    GROQ_MODEL_NAME = "openai/gpt-oss-20b"
-------------------------------------------------------------------------------
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Retrieve secret from os.environ or Streamlit st.secrets if running on Streamlit."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration container for the RAG search pipeline."""

    # Project Root and Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    QDRANT_STORAGE_PATH: str = str(BASE_DIR / "qdrant_storage")
    METRICS_EXPORT_PATH: str = str(BASE_DIR / "phase1_baseline_metrics.csv")

    # Database Mode Toggle (Local Embedded vs Cloud)
    # Evaluators can toggle USE_LOCAL_DB=true in .env to run with 0ms network latency
    USE_LOCAL_DB: bool = _get_secret("USE_LOCAL_DB", "true").lower() in ("true", "1", "yes")

    # Qdrant Database Configuration (Cloud vs Local Fallback)
    # If USE_LOCAL_DB is False and QDRANT_URL/API_KEY are provided, Qdrant Cloud is used.
    QDRANT_URL: str | None = _get_secret("QDRANT_URL", None)
    QDRANT_API_KEY: str | None = _get_secret("QDRANT_API_KEY", None)
    QDRANT_HOST: str | None = os.getenv("QDRANT_HOST", None)
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

    COLLECTION_NAME: str = "msmarco_100k_dense"
    HYBRID_COLLECTION_NAME: str = "msmarco_100k_hybrid"
    VECTOR_DIMENSION: int = 384  # Dimension for BAAI/bge-small-en-v1.5
    DISTANCE_METRIC: str = "Cosine"

    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    SPARSE_MODEL_NAME: str = "Qdrant/bm25"
    EMBED_BATCH_SIZE: int = 256  # Optimized batch size for multi-threaded CPU ingestion
    EMBED_NUM_THREADS: int | None = int(os.getenv("FAST_EMBED_THREADS", "0")) or None

    # Dataset Parameters
    DATASET_NAME: str = "microsoft/ms_marco"
    DATASET_CONFIG: str = "v1.1"
    DATASET_SPLIT: str = "train"
    TARGET_INGESTION_COUNT: int = 100_000
    CATEGORIES: tuple[str, ...] = ("general", "tech", "science", "finance", "health", "news")

    # Retrieval Defaults
    DEFAULT_TOP_K: int = 5
    LATENCY_TARGET_P95_MS: float = 300.0

    # LLM & Evaluation Configuration (Groq & RAGAS)
    GROQ_API_KEY: str = _get_secret("GROQ_API_KEY", "") or ""
    GROQ_RAGAS_API_KEY: str = _get_secret("GROQ_RAGAS_API_KEY", "") or _get_secret("GROQ_API_KEY", "") or ""
    GROQ_MODEL_NAME: str = _get_secret("GROQ_MODEL_NAME", "openai/gpt-oss-20b") or "openai/gpt-oss-20b"
    EVAL_TEMPERATURE: float = 0.0
    PHASE1_METRICS_PATH: str = str(BASE_DIR / "phase1_baseline_metrics.csv")
    PHASE2_METRICS_PATH: str = str(BASE_DIR / "phase2_metrics.csv")


# Instantiate global singleton config instance
config = PipelineConfig()


def get_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
    path: str | None = None,
) -> "QdrantClient":  # type: ignore[name-defined]
    """Initialize a production QdrantClient with seamless Cloud vs Local fallback.

    Connection Priority:
    1. Explicit URL & API Key parameters.
    2. Environment variables / Streamlit secrets (QDRANT_URL & QDRANT_API_KEY) -> Qdrant Cloud.
    3. Network host & port (QDRANT_HOST:QDRANT_PORT) -> Remote/Docker instance.
    4. Local storage directory -> QdrantLocal embedded disk persistence.

    Returns:
        Configured and connected QdrantClient instance.
    """
    from qdrant_client import QdrantClient

    target_url = url or config.QDRANT_URL
    target_api_key = api_key or config.QDRANT_API_KEY

    # 1. Cloud Connection
    if target_url and target_api_key:
        return QdrantClient(
            url=target_url,
            api_key=target_api_key,
        )

    # 2. Network Host/Port Connection
    if config.QDRANT_HOST:
        return QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
        )

    # 3. Local Embedded Storage Fallback
    local_path = path or config.QDRANT_STORAGE_PATH
    os.makedirs(local_path, exist_ok=True)
    return QdrantClient(path=local_path)


def get_groq_api_key(override_key: str | None = None) -> str:
    """Retrieve and validate the Groq API key from argument, environment, or Streamlit secrets."""
    api_key = override_key or config.GROQ_API_KEY
    if not api_key:
        raise ValueError(
            "Groq API Key is not set. Please set the GROQ_API_KEY environment variable "
            "or add it to .streamlit/secrets.toml."
        )
    return api_key


def get_groq_ragas_api_key(override_key: str | None = None) -> str:
    """Retrieve and validate the Groq API key strictly for RAGAS evaluation.

    Checks GROQ_RAGAS_API_KEY first, with automatic fallback to GROQ_API_KEY.
    """
    api_key = (
        override_key
        or config.GROQ_RAGAS_API_KEY
        or config.GROQ_API_KEY
        or os.getenv("GROQ_RAGAS_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Groq RAGAS API Key is not set. Please set the GROQ_RAGAS_API_KEY (or GROQ_API_KEY) "
            "environment variable or add it to .streamlit/secrets.toml."
        )
    return api_key
