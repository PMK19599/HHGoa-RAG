import os
import sys
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import rag_pipeline


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# CONFIG
# ============================================================

MODEL = "qwen/qwen3.6-27b"

TOKEN_LIMITS = [
    64,
    48,
    32,
    24,
]

QUERIES = [
    "what is a corporation?",
    "what is a felony?",
    "how much water can a running toilet waste?",
    "what are symptoms of a subaru head gasket?",
    "what is arbitrary?",
    "what is a government corporation?",
]


SYSTEM_PROMPT = """
You are an answer-generation component of a retrieval-augmented
question answering system.

Answer using ONLY the supplied evidence.

Rules:
1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Give the shortest complete answer possible.
5. Prefer one or two concise sentences.
6. Do not mention the retrieval system, models, scores, or guardrails.
7. Do not expose reasoning or internal analysis.
8. Do not begin with phrases such as "Based on the evidence"
   or "According to the provided evidence".
""".strip()


# ============================================================
# HELPERS
# ============================================================

def percentile(values, p):

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        int(len(ordered) * p / 100),
    )

    return ordered[index]


# ============================================================
# GENERATION
# ============================================================

def generate(query, evidence, max_tokens):

    prompt = f"""
Question:
{query}

Retrieved evidence:

{evidence}

Answer the question using only the retrieved evidence.
""".strip()

    start = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        reasoning_effort="none",
        temperature=0,
        max_completion_tokens=max_tokens,
        stream=True,
    )

    first_content = None
    parts = []

    for chunk in stream:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if not content:
            continue

        if first_content is None:
            first_content = time.perf_counter()

        parts.append(content)

    end = time.perf_counter()

    answer = "".join(parts).strip()

    ttft = (
        (first_content - start) * 1000
        if first_content
        else None
    )

    generation_ms = (
        end - start
    ) * 1000

    return answer, ttft, generation_ms


# ============================================================
# PREPARE REAL RETRIEVAL EVIDENCE
# ============================================================

def prepare_cases():

    print()
    print("=" * 78)
    print("PREPARING REAL RAG EVIDENCE")
    print("=" * 78)
    print()

    cases = []

    for i, query in enumerate(QUERIES, 1):

        start = time.perf_counter()

        results = rag_pipeline.retrieve(query)

        retrieval_ms = (
            time.perf_counter() - start
        ) * 1000

        if not results:

            print(
                f"{i:02d} | "
                f"NO RESULTS | "
                f"{query}"
            )

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

            print(
                f"{i:02d} | "
                f"REFUSE | "
                f"retrieval={retrieval_ms:.2f} ms | "
                f"{query}"
            )

            continue

        evidence_parts = []

        for rank, result in enumerate(
            results[:rag_pipeline.EVIDENCE_TOP_K],
            1
        ):

            evidence_parts.append(
                f"[Evidence {rank}]\n"
                f"{result['text']}"
            )

        evidence = "\n\n".join(evidence_parts)

        print(
            f"{i:02d} | "
            f"ANSWER | "
            f"retrieval={retrieval_ms:.2f} ms | "
            f"{query}"
        )

        cases.append(
            {
                "query": query,
                "evidence": evidence,
                "retrieval_ms": retrieval_ms,
            }
        )

    return cases


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    cases = prepare_cases()

    print()
    print("=" * 78)
    print("       HH GOA RAG - QWEN TOKEN BUDGET BENCHMARK")
    print("=" * 78)

    print()
    print(f"Model: {MODEL}")
    print()

    # Warmup
    print("Warming up...")

    warmup = cases[0]

    generate(
        warmup["query"],
        warmup["evidence"],
        64,
    )

    print("Warmup complete.")

    for token_limit in TOKEN_LIMITS:

        print()
        print("-" * 78)
        print(f"MAX COMPLETION TOKENS: {token_limit}")
        print("-" * 78)

        ttfts = []
        generation_times = []
        answer_lengths = []

        for i, case in enumerate(cases, 1):

            try:

                answer, ttft, generation_ms = generate(
                    case["query"],
                    case["evidence"],
                    token_limit,
                )

            except Exception as e:

                print(
                    f"{i:02d} | ERROR | {e}"
                )

                continue

            ttfts.append(ttft)
            generation_times.append(generation_ms)
            answer_lengths.append(len(answer))

            total_ms = (
                case["retrieval_ms"]
                + generation_ms
            )

            print(
                f"{i:02d} | "
                f"TTFT={ttft:7.2f} ms | "
                f"generation={generation_ms:7.2f} ms | "
                f"RAG+generation={total_ms:7.2f} ms | "
                f"chars={len(answer)}"
            )

            print(
                f"    {answer[:180]}"
            )

        if not generation_times:
            continue

        print()
        print(
            f"TTFT P50:        "
            f"{statistics.median(ttfts):.2f} ms"
        )

        print(
            f"TTFT P70:        "
            f"{percentile(ttfts, 70):.2f} ms"
        )

        print(
            f"TTFT P100:       "
            f"{max(ttfts):.2f} ms"
        )

        print(
            f"Generation P50:  "
            f"{statistics.median(generation_times):.2f} ms"
        )

        print(
            f"Generation P70:  "
            f"{percentile(generation_times, 70):.2f} ms"
        )

        print(
            f"Generation P100: "
            f"{max(generation_times):.2f} ms"
        )

        print(
            f"Chars P50:       "
            f"{statistics.median(answer_lengths):.0f}"
        )

        print(
            f"Chars P100:      "
            f"{max(answer_lengths):.0f}"
        )

    print()
    print("=" * 78)
    print("TOKEN BUDGET BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
