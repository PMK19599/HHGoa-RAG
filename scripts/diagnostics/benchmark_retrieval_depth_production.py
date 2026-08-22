import sys
import time
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rag_pipeline


QUERIES = [
    "what is a corporation?",
    "what is a felony?",
    "how much water can a running toilet waste?",
    "what are symptoms of a subaru head gasket?",
    "what is arbitrary?",
    "what is a government corporation?",
]

CANDIDATE_DEPTHS = [
    20,
    15,
    10,
    8,
    5,
]


def percentile(values, p):

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        int(len(ordered) * p / 100),
    )

    return ordered[index]


def retrieve_with_depth(query, depth):

    start = time.perf_counter()

    query_embedding = rag_pipeline.embed_model.encode(
        [query],
        normalize_embeddings=True,
    )

    import numpy as np

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    dense_scores, indices = rag_pipeline.index.search(
        query_embedding,
        depth,
    )

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        indices[0],
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(rag_pipeline.metadata):
            continue

        candidates.append(
            {
                "metadata_index": idx,
                "dense_score": float(dense_score),
                "cross_score": None,
                "text": rag_pipeline.metadata[idx]["text"],
            }
        )

    if not candidates:
        return [], (
            time.perf_counter() - start
        ) * 1000

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    cross_scores = rag_pipeline.cross_model.predict(
        pairs
    )

    for candidate, cross_score in zip(
        candidates,
        cross_scores,
    ):

        candidate["cross_score"] = float(
            cross_score
        )

    candidates.sort(
        key=lambda x: x["cross_score"],
        reverse=True,
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return candidates[:rag_pipeline.FINAL_TOP_K], elapsed


def main():

    print()
    print("=" * 78)
    print("       HH GOA RAG - RETRIEVAL DEPTH BENCHMARK")
    print("=" * 78)

    print()
    print("Warming up...")

    for depth in CANDIDATE_DEPTHS:

        retrieve_with_depth(
            QUERIES[0],
            depth,
        )

    print("Warmup complete.")

    for depth in CANDIDATE_DEPTHS:

        print()
        print("-" * 78)
        print(f"DENSE CANDIDATE DEPTH: {depth}")
        print("-" * 78)

        latencies = []
        answerable = 0

        for i, query in enumerate(
            QUERIES,
            1,
        ):

            results, latency = retrieve_with_depth(
                query,
                depth,
            )

            latencies.append(latency)

            if results:

                best = results[0]

                if (
                    best["dense_score"]
                    >= rag_pipeline.DENSE_THRESHOLD
                    and
                    best["cross_score"]
                    >= rag_pipeline.CROSS_THRESHOLD
                ):

                    answerable += 1

            print(
                f"{i:02d} | "
                f"{latency:8.2f} ms | "
                f"{'ANSWER' if results and answerable else 'RESULT'} | "
                f"{query}"
            )

        print()
        print(
            f"P50:  "
            f"{statistics.median(latencies):.2f} ms"
        )

        print(
            f"P70:  "
            f"{percentile(latencies, 70):.2f} ms"
        )

        print(
            f"P100: "
            f"{max(latencies):.2f} ms"
        )

        print(
            f"Mean: "
            f"{statistics.mean(latencies):.2f} ms"
        )

        print(
            f"Answerable: "
            f"{answerable}/{len(QUERIES)}"
        )

    print()
    print("=" * 78)
    print("RETRIEVAL DEPTH BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
