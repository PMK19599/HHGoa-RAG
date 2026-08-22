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


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

import rag_pipeline


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]


# ============================================================
# TEST QUERIES
# ============================================================

QUERIES = [
    "what is a corporation?",
    "what is a felony?",
    "how much water can a running toilet waste?",
    "what are symptoms of a subaru head gasket?",
    "what is arbitrary?",
    "what is a government corporation?",
]


# ============================================================
# GENERATION CONFIG
# ============================================================

MAX_TOKENS = 100


SYSTEM_PROMPT = """
You are the answer-generation component of a retrieval-augmented
question answering system.

You MUST answer using only the supplied evidence.

Rules:
1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent or assume facts that are not supported by the evidence.
4. If the evidence does not contain enough information to answer,
   respond exactly with:

"I don't have enough reliable evidence in the retrieved passages
to answer this question."

5. Keep the answer concise and direct.
6. Do not mention the retrieval system, models, scores, or guardrails.
7. Do not expose reasoning or internal analysis.
""".strip()


REFUSAL_MESSAGE = (
    "I don't have enough reliable evidence in the retrieved "
    "passages to answer this question."
)


# ============================================================
# GENERATION
# ============================================================

def generate(model, query, evidence):

    user_prompt = f"""
Question:
{query}

Retrieved evidence:

{evidence}

Answer the question using only the retrieved evidence.
""".strip()

    if model == "qwen/qwen3.6-27b":

        extra_params = {
            "reasoning_effort": "none",
        }

    else:

        extra_params = {
            "reasoning_effort": "low",
            "include_reasoning": False,
        }

    start = time.perf_counter()

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=MAX_TOKENS,
        stream=True,
        **extra_params,
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

    total = (
        end - start
    ) * 1000

    if not answer:
        answer = REFUSAL_MESSAGE

    return answer, ttft, total


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
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("       HH GOA RAG - PRODUCTION GENERATION A/B TEST")
    print("=" * 78)

    print()
    print("Using the real production retrieval pipeline.")
    print("Generation is benchmarked separately.")
    print()

    # --------------------------------------------------------
    # Retrieve evidence once
    # --------------------------------------------------------

    cases = []

    print("Preparing retrieval evidence...")
    print()

    for i, query in enumerate(QUERIES, 1):

        retrieval_start = time.perf_counter()

        results = rag_pipeline.retrieve(query)

        retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        if not results:

            print(
                f"{i:02d} | "
                f"NO RESULTS | "
                f"{query}"
            )

            cases.append(
                {
                    "query": query,
                    "results": [],
                    "answerable": False,
                    "retrieval_ms": retrieval_ms,
                }
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

        decision = (
            "ANSWER"
            if answerable
            else
            "REFUSE"
        )

        print(
            f"{i:02d} | "
            f"retrieval={retrieval_ms:7.2f} ms | "
            f"{decision:6s} | "
            f"{query}"
        )

        cases.append(
            {
                "query": query,
                "results": results,
                "answerable": answerable,
                "evidence": evidence,
                "retrieval_ms": retrieval_ms,
            }
        )

    # --------------------------------------------------------
    # Benchmark each model
    # --------------------------------------------------------

    for model in MODELS:

        print()
        print("=" * 78)
        print(f"MODEL: {model}")
        print("=" * 78)

        # Warmup
        print("Warming up...")

        warmup_case = next(
            (
                case
                for case in cases
                if case["answerable"]
            ),
            None,
        )

        if warmup_case:

            try:

                generate(
                    model,
                    warmup_case["query"],
                    warmup_case["evidence"],
                )

            except Exception as e:

                print(f"Warmup failed: {e}")
                continue

        print("Warmup complete.")
        print()

        ttfts = []
        generation_times = []
        total_times = []

        for i, case in enumerate(cases, 1):

            query = case["query"]

            # ------------------------------------------------
            # Refusal path
            # ------------------------------------------------

            if not case["answerable"]:

                print(
                    f"{i:02d} | "
                    f"REFUSE | "
                    f"generation=SKIP | "
                    f"{query}"
                )

                continue

            # ------------------------------------------------
            # Generation
            # ------------------------------------------------

            try:

                answer, ttft, generation_ms = generate(
                    model,
                    query,
                    case["evidence"],
                )

            except Exception as e:

                print(
                    f"{i:02d} | "
                    f"ERROR | "
                    f"{e}"
                )

                continue

            total_ms = (
                case["retrieval_ms"]
                + generation_ms
            )

            ttfts.append(ttft)
            generation_times.append(generation_ms)
            total_times.append(total_ms)

            print(
                f"{i:02d} | "
                f"TTFT={ttft:7.2f} ms | "
                f"generation={generation_ms:7.2f} ms | "
                f"RAG+generation={total_ms:7.2f} ms | "
                f"chars={len(answer)}"
            )

            print(
                f"    Answer: {answer[:220]}"
            )

        if not generation_times:
            continue

        print()
        print("RESULTS")
        print("-" * 78)

        print(
            f"Generation TTFT P50:  "
            f"{statistics.median(ttfts):.2f} ms"
        )

        print(
            f"Generation TTFT P70:  "
            f"{percentile(ttfts, 70):.2f} ms"
        )

        print(
            f"Generation TTFT P100: "
            f"{max(ttfts):.2f} ms"
        )

        print(
            f"Generation P50:       "
            f"{statistics.median(generation_times):.2f} ms"
        )

        print(
            f"Generation P70:       "
            f"{percentile(generation_times, 70):.2f} ms"
        )

        print(
            f"Generation P100:      "
            f"{max(generation_times):.2f} ms"
        )

        print(
            f"RAG+generation P50:   "
            f"{statistics.median(total_times):.2f} ms"
        )

        print(
            f"RAG+generation P70:   "
            f"{percentile(total_times, 70):.2f} ms"
        )

        print(
            f"RAG+generation P100:  "
            f"{max(total_times):.2f} ms"
        )

    print()
    print("=" * 78)
    print("A/B TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
