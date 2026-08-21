import os
import json
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

DENSE_CANDIDATE_K = 20
FINAL_TOP_K = 5
EVIDENCE_TOP_K = 3

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0

GROQ_MODEL = "openai/gpt-oss-20b"

MAX_COMPLETION_TOKENS = 150


# ============================================================
# CONSTANTS
# ============================================================

REFUSAL_MESSAGE = (
    "I don't have enough reliable evidence in the retrieved "
    "passages to answer this question."
)


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
7. Do not add information merely because it is generally known.
""".strip()


# ============================================================
# ENVIRONMENT
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to the .env file before running the pipeline."
    )

groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# MODEL LOADING
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
    """
    Retrieve dense Top-20 candidates and rerank them with
    the cross-encoder. Return the final Top-5.
    """

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
        DENSE_CANDIDATE_K
    )

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        indices[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append({
            "metadata_index": idx,
            "dense_score": float(dense_score),
            "cross_score": None,
            "text": metadata[idx]["text"]
        })

    if not candidates:
        return []

    # --------------------------------------------------------
    # Cross-encoder reranking
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

    candidates.sort(
        key=lambda x: x["cross_score"],
        reverse=True
    )

    return candidates[:FINAL_TOP_K]


# ============================================================
# GENERATION
# ============================================================

def generate_answer(query, evidence):
    """
    Generate an answer using only the selected evidence.

    Returns:
        answer
        TTFT in milliseconds
        total generation latency in milliseconds
    """

    user_prompt = f"""
Question:
{query}

Retrieved evidence:

{evidence}

Answer the question using only the retrieved evidence.
""".strip()

    start = time.perf_counter()

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        reasoning_effort="low",
        include_reasoning=False,
        temperature=0,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        stream=True
    )

    first_content_time = None
    answer_parts = []

    for chunk in stream:

        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if not content:
            continue

        if first_content_time is None:
            first_content_time = time.perf_counter()

        answer_parts.append(content)

    end = time.perf_counter()

    answer = "".join(answer_parts).strip()

    if first_content_time is None:
        ttft_ms = None
    else:
        ttft_ms = (
            first_content_time - start
        ) * 1000

    total_ms = (
        end - start
    ) * 1000

    if not answer:
        answer = REFUSAL_MESSAGE

    return answer, ttft_ms, total_ms


# ============================================================
# QUERY PIPELINE
# ============================================================

def run_query(query, verbose=True):
    """
    Run the complete RAG pipeline and return a structured result.

    This is the reusable API used by both the CLI and voice harness.
    """

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    results = retrieve(query)

    retrieval_elapsed = (
        time.perf_counter() - retrieval_start
    ) * 1000

    # --------------------------------------------------------
    # Empty retrieval
    # --------------------------------------------------------

    if not results:

        total_elapsed = (
            time.perf_counter() - total_start
        ) * 1000

        if verbose:
            print("\n=== GUARDRAIL ===")
            print("Decision: REFUSE")
            print(f"\n{REFUSAL_MESSAGE}")

            print("\n=== LATENCY ===")
            print(
                f"Retrieval + reranking: "
                f"{retrieval_elapsed:.2f} ms"
            )
            print(
                f"Total pipeline: "
                f"{total_elapsed:.2f} ms"
            )

        return {
            "query": query,
            "decision": "REFUSE",
            "answer": REFUSAL_MESSAGE,
            "evidence": [],
            "retrieval_ms": retrieval_elapsed,
            "ttft_ms": None,
            "generation_ms": 0.0,
            "total_ms": total_elapsed,
        }

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

    if verbose:
        print("\n=== RERANKED TOP-5 ===")

        for rank, result in enumerate(results, 1):

            print(f"\n#{rank}")

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
            f"{DENSE_THRESHOLD:.2f}"
        )

        print(
            f"Cross threshold: "
            f"{CROSS_THRESHOLD:.2f}"
        )

    # --------------------------------------------------------
    # REFUSAL
    # --------------------------------------------------------

    if not answerable:

        total_elapsed = (
            time.perf_counter() - total_start
        ) * 1000

        if verbose:
            print("\nDecision: REFUSE")
            print(f"\n{REFUSAL_MESSAGE}")

            print("\n=== LATENCY ===")

            print(
                f"Retrieval + reranking: "
                f"{retrieval_elapsed:.2f} ms"
            )

            print(
                f"Total pipeline: "
                f"{total_elapsed:.2f} ms"
            )

        return {
            "query": query,
            "decision": "REFUSE",
            "answer": REFUSAL_MESSAGE,
            "evidence": [],
            "retrieval_ms": retrieval_elapsed,
            "ttft_ms": None,
            "generation_ms": 0.0,
            "total_ms": total_elapsed,
        }

    # --------------------------------------------------------
    # Top-3 evidence
    # --------------------------------------------------------

    evidence_parts = []

    for rank, result in enumerate(
        results[:EVIDENCE_TOP_K],
        1
    ):

        evidence_parts.append(
            f"[Evidence {rank}]\n"
            f"{result['text']}"
        )

    evidence = "\n\n".join(evidence_parts)

    if verbose:
        print("\nDecision: ANSWER")
        print("\n=== SELECTED EVIDENCE ===")
        print(best["text"])

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    try:

        answer, ttft_ms, generation_ms = generate_answer(
            query,
            evidence
        )

    except Exception as e:

        total_elapsed = (
            time.perf_counter() - total_start
        ) * 1000

        if verbose:
            print("\n=== GENERATION ERROR ===")
            print(str(e))

            print("\n=== LATENCY ===")

            print(
                f"Retrieval + reranking: "
                f"{retrieval_elapsed:.2f} ms"
            )

            print(
                f"Total pipeline: "
                f"{total_elapsed:.2f} ms"
            )

        return {
            "query": query,
            "decision": "ERROR",
            "answer": REFUSAL_MESSAGE,
            "evidence": results[:EVIDENCE_TOP_K],
            "retrieval_ms": retrieval_elapsed,
            "ttft_ms": None,
            "generation_ms": 0.0,
            "total_ms": total_elapsed,
            "error": str(e),
        }

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    total_elapsed = (
        time.perf_counter() - total_start
    ) * 1000

    if verbose:
        print("\n=== FINAL ANSWER ===")
        print(answer)

        print("\n=== LATENCY ===")

        print(
            f"Retrieval + reranking: "
            f"{retrieval_elapsed:.2f} ms"
        )

        if ttft_ms is not None:
            print(
                f"Generation TTFT: "
                f"{ttft_ms:.2f} ms"
            )
        else:
            print(
                "Generation TTFT: N/A"
            )

        print(
            f"Generation total: "
            f"{generation_ms:.2f} ms"
        )

        print(
            f"Total pipeline: "
            f"{total_elapsed:.2f} ms"
        )

    return {
        "query": query,
        "decision": "ANSWER",
        "answer": answer,
        "evidence": results[:EVIDENCE_TOP_K],
        "retrieval_ms": retrieval_elapsed,
        "ttft_ms": ttft_ms,
        "generation_ms": generation_ms,
        "total_ms": total_elapsed,
    }


def answer_query(query):
    """
    Backward-compatible CLI wrapper.
    """

    return run_query(query, verbose=True)


# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":

    print("\nHH Goa RAG")

    print(
        "Architecture: "
        "BGE Top-20 -> Cross-Encoder -> Top-5 "
        "-> Guardrail -> Top-3 Evidence -> GPT-OSS-20B"
    )

    print(
        f"Guardrail: "
        f"Dense >= {DENSE_THRESHOLD:.2f} "
        f"AND Cross >= {CROSS_THRESHOLD:.2f}"
    )

    print("Type 'exit' to quit.")

    while True:

        try:
            query = input("\nEnter query: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\nExiting.")

            break

        if query.lower() == "exit":
            break

        if not query:
            continue

        answer_query(query)

