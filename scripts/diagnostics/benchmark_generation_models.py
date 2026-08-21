import os
import statistics
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

QUESTION = "What is a corporation?"

EVIDENCE = """
[Evidence 1]
Corporation definition, an association of individuals, created by law or
under authority of law, having a continuous existence independent of the
existences of its members, and powers and liabilities distinct from those
of its members.

[Evidence 2]
A corporation is a company or group of people authorized to act as a
single entity and recognized as such in law.

[Evidence 3]
A corporation may issue stock, either private or public, or may be
classified as a non-stock corporation.
""".strip()

SYSTEM_PROMPT = """
You are the answer-generation component of a retrieval-augmented
question answering system.

You MUST answer using only the supplied evidence.

Rules:
1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent or assume facts.
4. Keep the answer concise and direct.
5. Do not mention the retrieval system, models, scores, or guardrails.
6. Do not expose reasoning or internal analysis.
""".strip()

RUNS = 5
MAX_TOKENS = 100


def run(model):

    prompt = f"""
Question:
{QUESTION}

Retrieved evidence:

{EVIDENCE}

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
                "content": prompt,
            },
        ],
        temperature=0,
        max_completion_tokens=MAX_TOKENS,
        stream=True,
        **extra_params,
    )

    first = None
    parts = []

    for chunk in stream:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if not content:
            continue

        if first is None:
            first = time.perf_counter()

        parts.append(content)

    end = time.perf_counter()

    answer = "".join(parts).strip()

    ttft = (
        (first - start) * 1000
        if first
        else None
    )

    total = (
        end - start
    ) * 1000

    return answer, ttft, total


def percentile(values, p):

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        int(len(ordered) * p / 100),
    )

    return ordered[index]


def main():

    print()
    print("=" * 70)
    print("       HH GOA GENERATION MODEL BENCHMARK")
    print("=" * 70)

    for model in MODELS:

        print()
        print("=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)

        print("Warming up...")

        try:
            warmup_answer, _, _ = run(model)
        except Exception as e:
            print(f"Warmup FAILED: {e}")
            continue

        print("Warmup complete.")

        ttfts = []
        totals = []
        answers = []

        for i in range(RUNS):

            try:

                answer, ttft, total = run(model)

                ttfts.append(ttft)
                totals.append(total)
                answers.append(answer)

                print(
                    f"Run {i + 1}: "
                    f"TTFT={ttft:.2f} ms | "
                    f"Total={total:.2f} ms | "
                    f"Chars={len(answer)}"
                )

            except Exception as e:

                print(
                    f"Run {i + 1}: ERROR: {e}"
                )

        if not totals:
            continue

        print()
        print(
            f"TTFT P50:   "
            f"{statistics.median(ttfts):.2f} ms"
        )

        print(
            f"TTFT P70:   "
            f"{percentile(ttfts, 70):.2f} ms"
        )

        print(
            f"TTFT P100:  "
            f"{max(ttfts):.2f} ms"
        )

        print(
            f"Total P50:  "
            f"{statistics.median(totals):.2f} ms"
        )

        print(
            f"Total P70:  "
            f"{percentile(totals, 70):.2f} ms"
        )

        print(
            f"Total P100: "
            f"{max(totals):.2f} ms"
        )

        print()
        print("Sample answer:")
        print(answers[0])

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
