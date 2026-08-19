import os
import json
import time
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# ------------------------------------------------------------
# Retrieval configuration
# ------------------------------------------------------------

# Retrieve a larger dense candidate pool first.
CANDIDATE_K = 20

# Keep only the strongest reranked passages.
TOP_K = 5


# ------------------------------------------------------------
# Guardrail configuration
# ------------------------------------------------------------

# Validated using benchmark_combined_guardrail.py:
#
# Cross >= 7.0 + Dense >= 0.70
#
# Result:
# Accuracy       : 0.790
# False answers  : 11
# False refusals : 10
#
# Gap signal was tested but intentionally NOT included because
# it increased false refusals substantially.

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0


# ------------------------------------------------------------
# Groq configuration
# ------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# This is the model verified successfully with test_groq.py.
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# ENVIRONMENT
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file before running the pipeline."
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

embed_model = SentenceTransformer(EMBED_MODEL)

cross_model = CrossEncoder(CROSS_MODEL)

print("Loading FAISS index...")

index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata...")

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Index ready: {index.ntotal} passages")


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    # --------------------------------------------------------
    # 1. Dense retrieval
    # --------------------------------------------------------

    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    dense_scores, indices = index.search(
        query_embedding,
        CANDIDATE_K
    )

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        indices[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        passage = metadata[idx]["text"]

        candidates.append({
            "dense_score": float(dense_score),
            "cross_score": None,
            "text": passage,
            "metadata_index": idx
        })

    if not candidates:
        return []


    # --------------------------------------------------------
    # 2. Cross-encoder reranking
    # --------------------------------------------------------

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


    # Highest cross-encoder score first
    candidates.sort(
        key=lambda x: x["cross_score"],
        reverse=True
    )


    # --------------------------------------------------------
    # 3. Final Top-K
    # --------------------------------------------------------

    return candidates[:TOP_K]


# ============================================================
# GROQ GENERATION
# ============================================================

def generate_answer(query, evidence):

    system_prompt = """
You are the answer-generation component of a retrieval-augmented
question answering system.

You MUST answer using only the supplied evidence.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the evidence does not contain enough information to answer,
   say exactly:
   "I don't have enough reliable evidence in the retrieved passages
   to answer this question."
4. Keep answers concise and direct.
5. Do not mention the retrieval system unless necessary.
"""

    user_prompt = f"""
Question:
{query}

Retrieved evidence:

{evidence}

Answer the question using only the evidence above.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 150,
        "stream": False
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
            f"Groq API error {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    answer = data["choices"][0]["message"]["content"].strip()

    return answer, elapsed


# ============================================================
# QUERY PIPELINE
# ============================================================

def answer_query(query):

    total_start = time.perf_counter()


    # --------------------------------------------------------
    # Retrieval + reranking
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    results = retrieve(query)

    retrieval_elapsed = (
        time.perf_counter() - retrieval_start
    ) * 1000


    if not results:

        print("\nDecision: REFUSE")

        print(
            "\nI don't have enough reliable evidence "
            "in the retrieved passages to answer this question."
        )

        return


    # --------------------------------------------------------
    # Show reranked Top-5
    # --------------------------------------------------------

    print("\n=== RERANKED TOP-5 ===")

    for i, result in enumerate(results, 1):

        print(f"\n#{i}")

        print(
            f"Dense: "
            f"{result['dense_score']:.4f}"
        )

        print(
            f"Cross: "
            f"{result['cross_score']:.4f}"
        )

        print(
            result["text"][:500]
        )


    # --------------------------------------------------------
    # Guardrail
    # --------------------------------------------------------

    best = results[0]

    best_dense = best["dense_score"]
    best_cross = best["cross_score"]

    answerable = (
        best_dense >= DENSE_THRESHOLD
        and best_cross >= CROSS_THRESHOLD
    )


    print("\n=== GUARDRAIL ===")

    print(
        f"Selected dense score: "
        f"{best_dense:.4f}"
    )

    print(
        f"Selected cross score: "
        f"{best_cross:.4f}"
    )

    print(
        f"Dense threshold: "
        f"{DENSE_THRESHOLD}"
    )

    print(
        f"Cross threshold: "
        f"{CROSS_THRESHOLD}"
    )


    # --------------------------------------------------------
    # REFUSE
    # --------------------------------------------------------

    if not answerable:

        print("\nDecision: REFUSE")

        print(
            "\nI don't have enough reliable evidence "
            "in the retrieved passages to answer this question."
        )

        total_elapsed = (
            time.perf_counter() - total_start
        ) * 1000

        print("\n=== LATENCY ===")

        print(
            f"Dense + reranking: "
            f"{retrieval_elapsed:.2f} ms"
        )

        print(
            f"Total pipeline: "
            f"{total_elapsed:.2f} ms"
        )

        return


    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    print("\nDecision: ANSWER")


    # --------------------------------------------------------
    # Evidence selection
    #
    # Use the strongest 3 reranked passages for generation.
    # --------------------------------------------------------

    evidence_parts = []

    for i, result in enumerate(
        results[:3],
        1
    ):

        evidence_parts.append(
            f"[Evidence {i}]\n"
            f"{result['text']}"
        )

    evidence = "\n\n".join(evidence_parts)


    print("\n=== SELECTED EVIDENCE ===")

    print(
        best["text"]
    )


    # --------------------------------------------------------
    # Groq generation
    # --------------------------------------------------------

    try:

        answer, generation_latency = generate_answer(
            query,
            evidence
        )

    except Exception as e:

        print("\n=== GENERATION ERROR ===")
        print(str(e))

        return


    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    print("\n=== FINAL ANSWER ===")

    print(answer)


    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter() - total_start
    ) * 1000

    print("\n=== LATENCY ===")

    print(
        f"Dense + reranking: "
        f"{retrieval_elapsed:.2f} ms"
    )

    print(
        f"Generation: "
        f"{generation_latency:.2f} ms"
    )

    print(
        f"Total pipeline: "
        f"{total_elapsed:.2f} ms"
    )


# ============================================================
# MAIN LOOP
# ============================================================

print("\nHH Goa RAG")

print(
    "Architecture: "
    "BGE Top-20 -> Cross-Encoder -> Top-5 "
    "-> Guardrail -> Groq"
)

print("Type 'exit' to quit.")


while True:

    query = input("\nEnter query: ").strip()

    if query.lower() == "exit":
        break

    if not query:
        continue

    answer_query(query)