import os
import statistics
import time

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "openai/gpt-oss-20b"

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
""".strip()


TOKEN_LIMITS = [150, 100, 64, 48]
RUNS = 5


def percentile(values, p):

    values = sorted(values)

    index = min(
        len(values) - 1,
        int(len(values) * p / 100)
    )

    return values[index]


def run(limit):

    prompt = f"""
Question:
{QUESTION}

Retrieved evidence:

{EVIDENCE}

Answer the question using only the retrieved evidence.
""".strip()

    start = time.perf_counter()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning_effort="low",
        include_reasoning=False,
        temperature=0,
        max_completion_tokens=limit,
        stream=True
    )

    first_token = None
    parts = []

    for chunk in stream:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if not content:
            continue

        if first_token is None:
            first_token = time.perf_counter()

        parts.append(content)

    end = time.perf_counter()

    answer = "".join(parts).strip()

    ttft = (
        (first_token - start) * 1000
        if first_token
        else None
    )

    total = (
        end - start
    ) * 1000

    return answer, ttft, total


print()
print("=" * 70)
print("          HH GOA GENERATION TOKEN BENCHMARK")
print("=" * 70)

# Warmup
print("\nWarming up...")
run(150)
print("Warmup complete.")


for limit in TOKEN_LIMITS:

    ttfts = []
    totals = []
    answers = []

    print()
    print("-" * 70)
    print(f"MAX COMPLETION TOKENS: {limit}")
    print("-" * 70)

    for i in range(RUNS):

        answer, ttft, total = run(limit)

        ttfts.append(ttft)
        totals.append(total)
        answers.append(answer)

        print(
            f"Run {i + 1}: "
            f"TTFT={ttft:.2f} ms | "
            f"Total={total:.2f} ms | "
            f"Chars={len(answer)}"
        )

    print()
    print(
        f"TTFT P50:   {statistics.median(ttfts):.2f} ms"
    )

    print(
        f"Total P50:  {statistics.median(totals):.2f} ms"
    )

    print(
        f"Total P100: {max(totals):.2f} ms"
    )

    print()
    print("Sample answer:")
    print(answers[0])


print()
print("=" * 70)
print("Benchmark complete.")
print("=" * 70)
