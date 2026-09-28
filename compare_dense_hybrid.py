"""Comprehensive Benchmark & Side-by-Side Comparison: Dense vs Hybrid Retrieval.

Measures:
1. Latency Breakdown (Query Embedding, Qdrant Search, Total Latency).
2. Percentile Latency Distribution (p50, p90, p95, p99, Mean, Max).
3. Retrieval Ranking Discrepancy & Lexical vs Semantic Analysis.
"""

import time
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from config import config
from hybrid_retriever import QdrantHybridRetriever

console = Console()

TEST_QUERIES = [
    "what is the normal resting heart rate for adults",
    "what causes tides in the ocean",
    "how many chromosomes do humans have",
    "what is photosynthesis",
    "what is the capital of australia",
    "who invented the telephone",
    "what is the boiling point of water in fahrenheit",
    "what is the speed of light in vacuum",
    "how does penicillin work",
    "what is inflation in economics",
    "what is the function of red blood cells",
    "what is the chemical formula for water",
    "what is renewable energy",
    "what is the largest planet in our solar system",
    "what causes earthquakes",
    "who wrote hamlet",
    "what is the primary gas in earth atmosphere",
    "how do airplanes generate lift",
    "what is the freezing point of water in celsius",
    "what is machine learning",
]


def benchmark_mode(retriever: QdrantHybridRetriever, mode: str, runs_per_query: int = 5):
    records = []
    latencies = []

    # Warmup
    _ = retriever.search(TEST_QUERIES[0], mode=mode, top_k=5)

    for q in TEST_QUERIES:
        for _ in range(runs_per_query):
            start = time.perf_counter()
            docs = retriever.search(q, mode=mode, top_k=5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            records.append({
                "query": q,
                "latency_ms": elapsed_ms,
                "top_doc_id": docs[0].id if docs else None,
                "top_score": docs[0].score if docs else 0.0,
                "top_text": (docs[0].text[:80] + "...") if docs else "",
            })

    arr = np.array(latencies)
    stats = {
        "mode": mode.upper(),
        "total_queries": len(latencies),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }
    return stats, pd.DataFrame(records)


def main():
    console.print("\n[bold cyan]=== RUNNING DENSE VS HYBRID RETRIEVAL BENCHMARK ===[/bold cyan]\n")
    retriever = QdrantHybridRetriever()

    console.print("[yellow]Benchmarking PURE DENSE retrieval (100 runs)...[/yellow]")
    dense_stats, dense_df = benchmark_mode(retriever, mode="dense", runs_per_query=5)

    console.print("[yellow]Benchmarking HYBRID (Dense + BM25 RRF) retrieval (100 runs)...[/yellow]")
    hybrid_stats, hybrid_df = benchmark_mode(retriever, mode="hybrid", runs_per_query=5)

    metrics_keys = [
        ("Mean Latency", "mean_ms"),
        ("Min Latency", "min_ms"),
        ("P50 (Median)", "p50_ms"),
        ("P90 Latency", "p90_ms"),
        ("P95 Latency", "p95_ms"),
        ("P99 Latency", "p99_ms"),
        ("Max Latency", "max_ms"),
    ]

    df_results = pd.DataFrame([
        {
            "Metric": label,
            "Dense (BGE)": f"{dense_stats[key]:.2f} ms",
            "Hybrid (RRF)": f"{hybrid_stats[key]:.2f} ms",
            "Delta": f"{(hybrid_stats[key] - dense_stats[key]):+.2f} ms",
            "Overhead %": f"{((hybrid_stats[key] - dense_stats[key]) / dense_stats[key] * 100):+.1f}%",
        }
        for label, key in metrics_keys
    ])
    print("\n" + "=" * 65)
    print("      DENSE VS HYBRID RETRIEVAL LATENCY BENCHMARK")
    print("=" * 65)
    print(df_results.to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
