
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

EVIDENCE_DEPTHS = [3, 2, 1]


def percentile(values, p):

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        int(len(ordered) * p / 100),
    )

    return ordered[index]


def build_evidence(results, depth):

    parts = []

    for rank, result in enumerate(
        results[:depth],
        1,
    ):

        parts.append(
            f"[Evidence {rank}]\n"
            f"{result['text']}"
        )

    return "\n\n".join(parts)


def main():

    print()
    print("=" * 78)
    print("       HH GOA RAG - EVIDENCE DEPTH BENCHMARK")
    print("=" * 78)

    print()
    print("Preparing production retrieval evidence...")

    prepared = []

    for i, query in enumerate(QUERIES, 1):

        start = time.perf_counter()

        results = rag_pipeline.retrieve(query)

        retrieval_ms = (
            time.perf_counter() - start
        ) * 1000

        if not results:
            continue

        best = results[0]

        answerable = (
            best["dense_score"]
            >= rag_pipeline.DENSE_THRESHOLD
            and
            best["cross_score"]
            >= rag_pipeline.CROSS_THRESHOLD
        )

        if not answerable:
            continue

        prepared.append(
            (
                query,
                results,
                retrieval_ms,
            )
        )

        print(
            f"{i:02d} | "
            f"{retrieval_ms:7.2f} ms | "
            f"ANSWER | "
            f"{query}"
        )

    for depth in EVIDENCE_DEPTHS:

        print()
        print("-" * 78)
        print(f"EVIDENCE DEPTH: TOP-{depth}")
        print("-" * 78)

        total_values = []
        generation_values = []
        ttft_values = []

        for i, (
            query,
            results,
            retrieval_ms,
        ) in enumerate(prepared, 1):

            evidence = build_evidence(
                results,
                depth,
            )

            generation_start = time.perf_counter()

            answer, ttft_ms, generation_ms = (
                rag_pipeline.generate_answer(
                    query,
                    evidence,
                )
            )

            generation_wall_ms = (
                time.perf_counter()
                - generation_start
            ) * 1000

            total_ms = (
                retrieval_ms
                + generation_wall_ms
            )

            total_values.append(total_ms)
            generation_values.append(
                generation_wall_ms
            )

            if ttft_ms is not None:
                ttft_values.append(ttft_ms)

            print(
                f"{i:02d} | "
                f"TTFT={ttft_ms:7.2f} ms | "
                f"generation={generation_wall_ms:7.2f} ms | "
                f"RAG+generation={total_ms:7.2f} ms | "
                f"chars={len(answer)}"
            )

        print()

        print(
            f"TTFT P50:           "
            f"{statistics.median(ttft_values):.2f} ms"
        )

        print(
            f"Generation P50:     "
            f"{statistics.median(generation_values):.2f} ms"
        )

        print(
            f"Generation P70:     "
            f"{percentile(generation_values, 70):.2f} ms"
        )

        print(
            f"Generation P100:    "
            f"{max(generation_values):.2f} ms"
        )

        print(
            f"RAG+generation P50: "
            f"{statistics.median(total_values):.2f} ms"
        )

        print(
            f"RAG+generation P70: "
            f"{percentile(total_values, 70):.2f} ms"
        )

        print(
            f"RAG+generation P100:"
            f" {max(total_values):.2f} ms"
        )

    print()
    print("=" * 78)
    print("EVIDENCE DEPTH BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
