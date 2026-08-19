import os
import json
import time
import re
import requests
import faiss
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

NUM_QUERIES = 100

DENSE_CANDIDATE_K = 20
FINAL_TOP_K = 5

# FROZEN production guardrail
DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0

# Evidence verifier gets the strongest reranked passages.
VERIFIER_TOP_K = 3

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Same model that already worked in test_groq.py
GROQ_MODEL = "openai/gpt-oss-20b"

# Keep verifier output tiny and deterministic.
MAX_COMPLETION_TOKENS = 20


# ============================================================
# ENVIRONMENT
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file before running this benchmark."
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross-encoder...")
cross_model = CrossEncoder(CROSS_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata...")

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Index ready: {index.ntotal} passages")


# ============================================================
# LOAD DATASET
# ============================================================

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))


# ============================================================
# WARMUP
# ============================================================

print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["warmup query", "warmup passage"]]
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    dense_scores, dense_indices = index.search(
        query_embedding,
        DENSE_CANDIDATE_K
    )

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        dense_indices[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append({
            "metadata_index": idx,
            "dense_score": float(dense_score),
            "text": metadata[idx]["text"]
        })

    if not candidates:
        return []

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    cross_scores = cross_model.predict(pairs)

    for candidate, cross_score in zip(
        candidates,
        cross_scores
    ):
        candidate["cross_score"] = float(cross_score)

    candidates.sort(
        key=lambda x: x["cross_score"],
        reverse=True
    )

    return candidates[:FINAL_TOP_K]


# ============================================================
# FROZEN PRODUCTION RULE
# ============================================================

def production_guardrail(results):

    if not results:
        return False

    best = results[0]

    return (
        best["dense_score"] >= DENSE_THRESHOLD
        and best["cross_score"] >= CROSS_THRESHOLD
    )


# ============================================================
# LLM EVIDENCE VERIFIER
# ============================================================

def verify_evidence(query, evidence):

    system_prompt = """
You are an evidence-support verifier inside a retrieval system.

Your task is NOT to answer the question.

Determine whether the supplied evidence contains enough information
to directly support an answer to the question.

IMPORTANT:
- Relevance alone is not enough.
- Keyword overlap alone is not enough.
- The evidence must actually support the requested claim.
- Do not use outside knowledge.
- Do not infer missing facts.
- If the evidence discusses the opposite direction of a relationship,
  it does NOT support the question.
- If the evidence mentions the topic but omits the requested fact,
  it does NOT support the question.

Return exactly one token:

SUPPORTED

or

NOT_SUPPORTED
""".strip()

    user_prompt = f"""
Question:
{query}

Evidence:
{evidence}

Does the evidence directly support answering the question?

Return exactly:
SUPPORTED
or
NOT_SUPPORTED
""".strip()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": "low",
        "include_reasoning": False
    }

    start = time.perf_counter()

    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq verifier error {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    raw = data["choices"][0]["message"]["content"]

    normalized = raw.strip().upper()

    # Extract only the expected decision.
    if re.search(r"\bSUPPORTED\b", normalized):
        decision = "SUPPORTED"
    elif re.search(r"\bNOT_SUPPORTED\b", normalized):
        decision = "NOT_SUPPORTED"
    else:
        decision = "NOT_SUPPORTED"

    return decision, elapsed, raw.strip()


# ============================================================
# EVALUATION
# ============================================================

print("\n")
print("=" * 72)
print("           LLM EVIDENCE VERIFIER EXPERIMENT")
print("=" * 72)

print(f"\nQueries:             {NUM_QUERIES}")
print(f"Dense candidates:    {DENSE_CANDIDATE_K}")
print(f"Final K:              {FINAL_TOP_K}")
print(f"Verifier evidence K:  {VERIFIER_TOP_K}")
print(f"Dense threshold:     {DENSE_THRESHOLD}")
print(f"Cross threshold:     {CROSS_THRESHOLD}")
print(f"Verifier model:      {GROQ_MODEL}")

print("\nRunning evaluation...")


baseline_correct = 0
baseline_false_answers = 0
baseline_false_refusals = 0

verifier_correct = 0
verifier_false_answers = 0
verifier_false_refusals = 0

verifier_calls = 0
verifier_latencies = []

false_answer_catches = 0
false_refusal_breaks = 0

interesting_cases = []


for count, record in enumerate(
    records[:NUM_QUERIES],
    start=1
):

    query = record["eng_query"]

    # Ground truth from dataset.
    actual_answerable = bool(
        any(record["is_selected"])
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = retrieve(query)

    # --------------------------------------------------------
    # Frozen production decision
    # --------------------------------------------------------

    baseline_answer = production_guardrail(results)

    if baseline_answer == actual_answerable:
        baseline_correct += 1

    if baseline_answer and not actual_answerable:
        baseline_false_answers += 1

    if not baseline_answer and actual_answerable:
        baseline_false_refusals += 1

    # --------------------------------------------------------
    # If production already refuses, we do NOT call verifier.
    #
    # The verifier is specifically testing whether it can improve
    # the cases where the production rule says ANSWER.
    # --------------------------------------------------------

    verifier_answer = baseline_answer
    verifier_latency = 0
    verifier_raw = ""

    if baseline_answer and results:

        evidence_parts = []

        for i, result in enumerate(
            results[:VERIFIER_TOP_K],
            start=1
        ):

            evidence_parts.append(
                f"[Evidence {i}]\n{result['text']}"
            )

        evidence = "\n\n".join(evidence_parts)

        try:

            verifier_decision, verifier_latency, verifier_raw = (
                verify_evidence(
                    query,
                    evidence
                )
            )

            verifier_calls += 1
            verifier_latencies.append(
                verifier_latency
            )

            verifier_answer = (
                verifier_decision == "SUPPORTED"
            )

        except Exception as e:

            # Fail closed.
            verifier_answer = False
            verifier_raw = f"ERROR: {e}"

            verifier_calls += 1

        # ----------------------------------------------------
        # Track interesting cases
        # ----------------------------------------------------

        if (
            baseline_answer
            and not actual_answerable
            and not verifier_answer
        ):
            false_answer_catches += 1

            interesting_cases.append({
                "type": "FALSE_ANSWER_CAUGHT",
                "query": query,
                "baseline": "ANSWER",
                "verifier": "NOT_SUPPORTED",
                "top_dense": results[0]["dense_score"],
                "top_cross": results[0]["cross_score"],
                "evidence": results[0]["text"],
                "verifier_raw": verifier_raw
            })

        elif (
            baseline_answer
            and actual_answerable
            and not verifier_answer
        ):
            false_refusal_breaks += 1

            interesting_cases.append({
                "type": "TRUE_ANSWER_BLOCKED",
                "query": query,
                "baseline": "ANSWER",
                "verifier": "NOT_SUPPORTED",
                "top_dense": results[0]["dense_score"],
                "top_cross": results[0]["cross_score"],
                "evidence": results[0]["text"],
                "verifier_raw": verifier_raw
            })

    # --------------------------------------------------------
    # Final verifier evaluation
    # --------------------------------------------------------

    if verifier_answer == actual_answerable:
        verifier_correct += 1

    if verifier_answer and not actual_answerable:
        verifier_false_answers += 1

    if not verifier_answer and actual_answerable:
        verifier_false_refusals += 1

    if count % 10 == 0:
        print(
            f"Processed {count}/{NUM_QUERIES}"
        )


# ============================================================
# FINAL METRICS
# ============================================================

baseline_accuracy = (
    baseline_correct / NUM_QUERIES
)

verifier_accuracy = (
    verifier_correct / NUM_QUERIES
)

answerable_count = sum(
    1
    for record in records[:NUM_QUERIES]
    if any(record["is_selected"])
)

unanswerable_count = (
    NUM_QUERIES - answerable_count
)

baseline_answer_recall = (
    (
        answerable_count
        - baseline_false_refusals
    )
    / answerable_count
)

verifier_answer_recall = (
    (
        answerable_count
        - verifier_false_refusals
    )
    / answerable_count
)

baseline_refusal_precision = (
    (
        unanswerable_count
        - baseline_false_answers
    )
    / unanswerable_count
)

verifier_refusal_precision = (
    (
        unanswerable_count
        - verifier_false_answers
    )
    / unanswerable_count
)


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 72)
print("                  BASELINE VS VERIFIER")
print("=" * 72)

print("\nFROZEN PRODUCTION BASELINE")

print(
    f"Correct answers:       {answerable_count - baseline_false_refusals}"
)

print(
    f"Correct refusals:      {unanswerable_count - baseline_false_answers}"
)

print(
    f"False answers:         {baseline_false_answers}"
)

print(
    f"False refusals:        {baseline_false_refusals}"
)

print(
    f"Accuracy:              {baseline_accuracy:.3f}"
)

print(
    f"Answer recall:         {baseline_answer_recall:.3f}"
)

print(
    f"Refusal precision:     {baseline_refusal_precision:.3f}"
)


print("\nLLM EVIDENCE VERIFIER")

print(
    f"Correct answers:       {answerable_count - verifier_false_refusals}"
)

print(
    f"Correct refusals:      {unanswerable_count - verifier_false_answers}"
)

print(
    f"False answers:         {verifier_false_answers}"
)

print(
    f"False refusals:        {verifier_false_refusals}"
)

print(
    f"Accuracy:              {verifier_accuracy:.3f}"
)

print(
    f"Answer recall:         {verifier_answer_recall:.3f}"
)

print(
    f"Refusal precision:     {verifier_refusal_precision:.3f}"
)


print("\nVERIFIER EFFECT")

print(
    f"False answers caught:  {false_answer_catches}"
)

print(
    f"True answers blocked:  {false_refusal_breaks}"
)

print(
    f"Verifier calls:        {verifier_calls}"
)


if verifier_latencies:

    verifier_latencies = np.asarray(
        verifier_latencies
    )

    print("\nVERIFIER LATENCY")

    print(
        f"P50:                  "
        f"{np.percentile(verifier_latencies, 50):.2f} ms"
    )

    print(
        f"P70:                  "
        f"{np.percentile(verifier_latencies, 70):.2f} ms"
    )

    print(
        f"P100:                 "
        f"{np.max(verifier_latencies):.2f} ms"
    )

    print(
        f"Mean:                  "
        f"{np.mean(verifier_latencies):.2f} ms"
    )


# ============================================================
# VERDICT
# ============================================================

print("\n")
print("=" * 72)
print("                         VERDICT")
print("=" * 72)

if verifier_accuracy > baseline_accuracy:

    print(
        f"\nVERIFIER IMPROVED BASELINE:"
        f" {verifier_accuracy:.3f} > {baseline_accuracy:.3f}"
    )

    print(
        "\nThe LLM evidence verifier is promising."
    )

elif verifier_accuracy == baseline_accuracy:

    print(
        f"\nNO ACCURACY CHANGE:"
        f" {verifier_accuracy:.3f} == {baseline_accuracy:.3f}"
    )

    print(
        "\nThe verifier did not improve overall accuracy."
    )

else:

    print(
        f"\nVERIFIER DID NOT IMPROVE:"
        f" {verifier_accuracy:.3f} <= {baseline_accuracy:.3f}"
    )

    print(
        "\nKeep the production pipeline frozen."
    )


print("\nDo NOT modify rag_pipeline.py based on this experiment alone.")


# ============================================================
# INTERESTING CASES
# ============================================================

print("\n")
print("=" * 72)
print("                    INTERESTING CASES")
print("=" * 72)

for i, case in enumerate(
    interesting_cases,
    start=1
):

    print("\n" + "-" * 72)

    print(
        f"CASE #{i}: {case['type']}"
    )

    print(
        f"Query: {case['query']}"
    )

    print(
        f"Dense: {case['top_dense']:.4f}"
    )

    print(
        f"Cross: {case['top_cross']:.4f}"
    )

    print(
        f"Verifier raw: {case['verifier_raw']}"
    )

    print(
        f"Evidence:\n{case['evidence'][:1000]}"
    )


print("\n")
print("=" * 72)
print("                  Evaluation complete.")
print("=" * 72)