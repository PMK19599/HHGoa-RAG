import json
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIG
# ============================================================

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

TOP_K = 5
NUM_QUERIES = 100

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross-encoder...")
cross_model = CrossEncoder(CROSS_MODEL)


# ============================================================
# LOAD INDEX + METADATA
# ============================================================

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


records = records[:NUM_QUERIES]

print(f"Evaluating {len(records)} queries...")


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
# METRIC STORAGE
# ============================================================

dense_hits = 0
rerank_hits = 0

guardrail_correct = 0
false_answers = 0
false_refusals = 0

dense_ranks = []
rerank_ranks = []

retrieval_latencies = []
rerank_latencies = []
total_latencies = []


# ============================================================
# EVALUATION
# ============================================================

print("\nRunning final evaluation...")

for query_number, record in enumerate(records, start=1):

    query = record["eng_query"]
    query_id = record["query_id"]

    # --------------------------------------------------------
    # Dataset ground truth
    # --------------------------------------------------------

    answerable = any(
        int(flag) == 1
        for flag in record["is_selected"]
    )

    # --------------------------------------------------------
    # Stage 1: Dense retrieval
    # --------------------------------------------------------

    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()

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
        TOP_K
    )

    retrieval_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000

    retrieval_latencies.append(retrieval_ms)

    # --------------------------------------------------------
    # Build candidates
    # --------------------------------------------------------

    candidates = []

    for rank, (score, idx) in enumerate(
        zip(
            dense_scores[0],
            dense_indices[0]
        ),
        start=1
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]

        candidates.append({
            "dense_rank": rank,
            "dense_score": float(score),
            "metadata_index": idx,
            "query_id": meta["query_id"],
            "passage_index": meta["passage_index"],
            "is_selected": int(meta["is_selected"]),
            "text": meta["text"]
        })

    # --------------------------------------------------------
    # Dense Recall@5
    #
    # Correct passage must belong to this query and be selected.
    # --------------------------------------------------------

    dense_rank = None

    for candidate in candidates:

        if (
            candidate["query_id"] == query_id
            and candidate["is_selected"] == 1
        ):

            dense_rank = candidate["dense_rank"]
            break

    if dense_rank is not None:

        dense_hits += 1
        dense_ranks.append(dense_rank)

    else:

        dense_ranks.append(0)

    # --------------------------------------------------------
    # Stage 2: Cross-encoder reranking
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    if pairs:

        cross_scores = cross_model.predict(pairs)

        for candidate, cross_score in zip(
            candidates,
            cross_scores
        ):

            candidate["cross_score"] = float(
                cross_score
            )

        candidates.sort(
            key=lambda x: x["cross_score"],
            reverse=True
        )

    rerank_ms = (
        time.perf_counter() - rerank_start
    ) * 1000

    rerank_latencies.append(rerank_ms)

    # --------------------------------------------------------
    # Reranking Recall@1
    # --------------------------------------------------------

    rerank_rank = None

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        if (
            candidate["query_id"] == query_id
            and candidate["is_selected"] == 1
        ):

            rerank_rank = rank
            break

    if rerank_rank is not None:

        rerank_hits += 1
        rerank_ranks.append(rerank_rank)

    else:

        rerank_ranks.append(0)

    # --------------------------------------------------------
    # Final evidence
    # --------------------------------------------------------

    if candidates:

        best = candidates[0]

        best_dense = best["dense_score"]
        best_cross = best["cross_score"]

    else:

        best = None
        best_dense = 0.0
        best_cross = -999.0

    # --------------------------------------------------------
    # Stage 3: Guardrail
    # --------------------------------------------------------

    predicted_answerable = (
        best_dense >= DENSE_THRESHOLD
        and best_cross >= CROSS_THRESHOLD
    )

    if predicted_answerable == answerable:

        guardrail_correct += 1

    if predicted_answerable and not answerable:

        false_answers += 1

    if not predicted_answerable and answerable:

        false_refusals += 1

    # --------------------------------------------------------
    # Total latency
    # --------------------------------------------------------

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000

    total_latencies.append(total_ms)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if query_number % 10 == 0:

        print(
            f"Processed {query_number}/{len(records)}"
        )


# ============================================================
# SAFE METRIC HELPERS
# ============================================================

def mean_or_zero(values):

    if not values:
        return 0.0

    return float(np.mean(values))


def percentile_or_zero(values, percentile):

    if not values:
        return 0.0

    return float(
        np.percentile(values, percentile)
    )


def reciprocal_rank(rank):

    if rank <= 0:
        return 0.0

    return 1.0 / rank


# ============================================================
# MRR
# ============================================================

dense_mrr = mean_or_zero(
    [
        reciprocal_rank(rank)
        for rank in dense_ranks
    ]
)

rerank_mrr = mean_or_zero(
    [
        reciprocal_rank(rank)
        for rank in rerank_ranks
    ]
)


# ============================================================
# RESULTS
# ============================================================

total_queries = len(records)

dense_recall = (
    dense_hits / total_queries
    if total_queries
    else 0.0
)

rerank_recall = (
    rerank_hits / total_queries
    if total_queries
    else 0.0
)

guardrail_accuracy = (
    guardrail_correct / total_queries
    if total_queries
    else 0.0
)

false_answer_rate = (
    false_answers / total_queries
    if total_queries
    else 0.0
)

false_refusal_rate = (
    false_refusals / total_queries
    if total_queries
    else 0.0
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("              FINAL RAG EVALUATION")
print("=" * 60)

print("\nDATASET")
print(f"Queries evaluated: {total_queries}")
print(f"FAISS passages:    {index.ntotal}")

print("\nRETRIEVAL")

print(
    f"Dense Recall@{TOP_K}: "
    f"{dense_recall:.3f}"
)

print(
    f"Dense MRR@{TOP_K}:    "
    f"{dense_mrr:.3f}"
)

print(
    f"Reranked Recall@{TOP_K}: "
    f"{rerank_recall:.3f}"
)

print(
    f"Reranked MRR@{TOP_K}:    "
    f"{rerank_mrr:.3f}"
)

print("\nGUARDRAIL")

print(
    f"Dense threshold: "
    f"{DENSE_THRESHOLD:.2f}"
)

print(
    f"Cross threshold: "
    f"{CROSS_THRESHOLD:.2f}"
)

print(
    f"Guardrail accuracy: "
    f"{guardrail_accuracy:.3f}"
)

print(
    f"False answers: "
    f"{false_answers}"
)

print(
    f"False answer rate: "
    f"{false_answer_rate:.3f}"
)

print(
    f"False refusals: "
    f"{false_refusals}"
)

print(
    f"False refusal rate: "
    f"{false_refusal_rate:.3f}"
)

print("\nLATENCY")

print(
    f"Dense retrieval P50: "
    f"{percentile_or_zero(retrieval_latencies, 50):.2f} ms"
)

print(
    f"Dense retrieval P70: "
    f"{percentile_or_zero(retrieval_latencies, 70):.2f} ms"
)

print(
    f"Cross-encoder P50: "
    f"{percentile_or_zero(rerank_latencies, 50):.2f} ms"
)

print(
    f"Cross-encoder P70: "
    f"{percentile_or_zero(rerank_latencies, 70):.2f} ms"
)

print(
    f"Total pipeline P50: "
    f"{percentile_or_zero(total_latencies, 50):.2f} ms"
)

print(
    f"Total pipeline P70: "
    f"{percentile_or_zero(total_latencies, 70):.2f} ms"
)

print(
    f"Total pipeline P100: "
    f"{percentile_or_zero(total_latencies, 100):.2f} ms"
)

print(
    f"Total pipeline Mean: "
    f"{mean_or_zero(total_latencies):.2f} ms"
)

print("\n")
print("=" * 60)
print("Evaluation complete.")
print("=" * 60)
