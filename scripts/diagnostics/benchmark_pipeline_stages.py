import sys
import statistics
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag_pipeline import retrieve, generate_answer


QUERIES = [
    "what is a corporation?",
    "what is a felony?",
    "how much water can a running toilet waste?",
    "what are symptoms of a subaru head gasket?",
    "what is arbitrary?",
    "what is a government corporation?",
]


def percentile(values, percentile):
    ordered = sorted(values)

    index = int(
        (percentile / 100) * len(ordered)
    )

    index = min(
        len(ordered) - 1,
        index
    )

    return ordered[index]


def main():

    print()
    print("=" * 70)
    print("           HH GOA RAG STAGE BENCHMARK")
    print("=" * 70)

    print("\nWarming up...")

    retrieve("what is a corporation?")

    print("Warmup complete.")

    retrieval_times = []
    generation_times = []
    total_times = []

    for i, query in enumerate(QUERIES, 1):

        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()

        results = retrieve(query)

        retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        if not results:

            total_ms = (
                time.perf_counter() - total_start
            ) * 1000

            print(
                f"{i:02d} | "
                f"retrieval={retrieval_ms:8.2f} ms | "
                f"generation=SKIP | "
                f"total={total_ms:8.2f} ms"
            )

            continue

        evidence_parts = []

        for rank, result in enumerate(
            results[:3],
            1
        ):

            evidence_parts.append(
                f"[Evidence {rank}]\n"
                f"{result['text']}"
            )

        evidence = "\n\n".join(evidence_parts)

        generation_start = time.perf_counter()

        answer, ttft_ms, generation_ms = generate_answer(
            query,
            evidence
        )

        generation_elapsed = (
            time.perf_counter() - generation_start
        ) * 1000

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        retrieval_times.append(retrieval_ms)
        generation_times.append(generation_elapsed)
        total_times.append(total_ms)

        print(
            f"{i:02d} | "
            f"retrieval={retrieval_ms:8.2f} ms | "
            f"TTFT={ttft_ms:8.2f} ms | "
            f"generation={generation_elapsed:8.2f} ms | "
            f"total={total_ms:8.2f} ms"
        )

    print()
    print("=" * 70)
    print("                  RESULTS")
    print("=" * 70)

    print()
    print("RETRIEVAL")

    if retrieval_times:
        print(
            f"P50:  {percentile(retrieval_times, 50):.2f} ms"
        )
        print(
            f"P70:  {percentile(retrieval_times, 70):.2f} ms"
        )
        print(
            f"P100: {percentile(retrieval_times, 100):.2f} ms"
        )

    print()
    print("GENERATION")

    if generation_times:
        print(
            f"P50:  {percentile(generation_times, 50):.2f} ms"
        )
        print(
            f"P70:  {percentile(generation_times, 70):.2f} ms"
        )
        print(
            f"P100: {percentile(generation_times, 100):.2f} ms"
        )

    print()
    print("TOTAL")

    if total_times:
        print(
            f"P50:  {percentile(total_times, 50):.2f} ms"
        )
        print(
            f"P70:  {percentile(total_times, 70):.2f} ms"
        )
        print(
            f"P100: {percentile(total_times, 100):.2f} ms"
        )

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
