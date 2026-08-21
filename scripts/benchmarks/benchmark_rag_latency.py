import sys
import statistics
import time
from pathlib import Path


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag_pipeline import run_query


# ------------------------------------------------------------
# TEST QUERIES
# ------------------------------------------------------------

QUERIES = [
    "what is a corporation?",
    "what is photosynthesis?",
    "what is a felony?",
    "what is basal in dna?",
    "how much water can a running toilet waste?",
    "what are symptoms of a subaru head gasket?",
    "what is arbitrary?",
    "what is a sub slab?",
    "how much does a stamp cover?",
    "what is a government corporation?",
]


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("             HH GOA RAG LATENCY HARNESS")
    print("=" * 70)

    print("\nWarming up...")

    run_query(
        "what is a corporation?",
        verbose=False
    )

    print("Warmup complete.")

    latencies = []
    decisions = []

    print("\nRunning benchmark...")

    for i, query in enumerate(QUERIES, 1):

        start = time.perf_counter()

        result = run_query(
            query,
            verbose=False
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)
        decisions.append(result["decision"])

        print(
            f"{i:02d}/10 | "
            f"{elapsed_ms:8.2f} ms | "
            f"{result['decision']:6s} | "
            f"{query}"
        )

    ordered = sorted(latencies)

    p50 = statistics.median(ordered)

    p70_index = min(
        len(ordered) - 1,
        int(len(ordered) * 0.70)
    )

    p70 = ordered[p70_index]

    p100 = max(ordered)
    mean = statistics.mean(ordered)

    print()
    print("=" * 70)
    print("                 RESULTS")
    print("=" * 70)

    print(f"Queries:       {len(ordered)}")
    print(f"Mean:          {mean:.2f} ms")
    print(f"P50:           {p50:.2f} ms")
    print(f"P70:           {p70:.2f} ms")
    print(f"P100:          {p100:.2f} ms")

    print()
    print("Decisions:")
    print(f"ANSWER:        {decisions.count('ANSWER')}")
    print(f"REFUSE:        {decisions.count('REFUSE')}")
    print(f"ERROR:         {decisions.count('ERROR')}")

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
