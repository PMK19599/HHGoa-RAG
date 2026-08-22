
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rag_pipeline


QUERY = "what is a corporation?"


print()
print("=" * 72)
print("       HH GOA RAG - PRODUCTION QUERY PROFILE")
print("=" * 72)

print()
print("Profiling:", QUERY)

# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

start = time.perf_counter()

results = rag_pipeline.retrieve(QUERY)

retrieval_ms = (
    time.perf_counter() - start
) * 1000

print()
print(f"retrieve(): {retrieval_ms:.2f} ms")

if not results:
    print("No results.")
    raise SystemExit

best = results[0]

print(
    f"Best dense: {best['dense_score']:.4f}"
)

print(
    f"Best cross: {best['cross_score']:.4f}"
)

# ------------------------------------------------------------
# Guardrail
# ------------------------------------------------------------

guardrail_start = time.perf_counter()

answerable = (
    best["dense_score"]
    >= rag_pipeline.DENSE_THRESHOLD
    and
    best["cross_score"]
    >= rag_pipeline.CROSS_THRESHOLD
)

guardrail_ms = (
    time.perf_counter() - guardrail_start
) * 1000

print(
    f"guardrail: {guardrail_ms:.4f} ms"
)

print(
    "Decision:",
    "ANSWER" if answerable else "REFUSE"
)

if not answerable:
    raise SystemExit

# ------------------------------------------------------------
# Evidence construction
# ------------------------------------------------------------

evidence_start = time.perf_counter()

evidence_parts = []

for rank, result in enumerate(
    results[:rag_pipeline.EVIDENCE_TOP_K],
    1,
):

    evidence_parts.append(
        f"[Evidence {rank}]\n"
        f"{result['text']}"
    )

evidence = "\n\n".join(evidence_parts)

evidence_ms = (
    time.perf_counter() - evidence_start
) * 1000

print(
    f"evidence construction: {evidence_ms:.4f} ms"
)

# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

generation_start = time.perf_counter()

answer, ttft_ms, generation_ms = (
    rag_pipeline.generate_answer(
        QUERY,
        evidence,
    )
)

generation_wall_ms = (
    time.perf_counter() - generation_start
) * 1000

print()
print("=== GENERATION ===")
print(f"TTFT:        {ttft_ms:.2f} ms")
print(f"Generation:  {generation_ms:.2f} ms")
print(f"Wall time:   {generation_wall_ms:.2f} ms")

print()
print("=== ANSWER ===")
print(answer)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

total = (
    retrieval_ms
    + guardrail_ms
    + evidence_ms
    + generation_wall_ms
)

print()
print("=== SUMMARY ===")
print(f"Retrieval:   {retrieval_ms:.2f} ms")
print(f"Guardrail:   {guardrail_ms:.4f} ms")
print(f"Evidence:    {evidence_ms:.4f} ms")
print(f"Generation:  {generation_wall_ms:.2f} ms")
print(f"Estimated:   {total:.2f} ms")

print()
print("=" * 72)
