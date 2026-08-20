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

NUM_QUERIES = 100

DENSE_CANDIDATE_K = 20
FINAL_TOP_K = 5

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")

embed_model = SentenceTransformer(
    EMBED_MODEL
)

print("Loading cross-encoder...")

cross_model = CrossEncoder(
    CROSS_MODEL
)

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

print("Loading metadata...")

with META_PATH.open(
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

print(
    f"Index ready: {index.ntotal} passages"
)


# ============================================================
# LOAD DATASET
# ============================================================

records = []

with DATA.open(
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        records.append(
            json.loads(line)
        )


records = records[:NUM_QUERIES]


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
# STORAGE
# ============================================================

answerable_total = 0
unanswerable_total = 0

correct_answers = 0
correct_refusals = 0

false_answers = 0
false_refusals = 0

retrieval_hits = 0

dense_latencies = []
cross_latencies = []
total_latencies = []

reciprocal_ranks = []

guardrail_results = []


# ============================================================
# BENCHMARK
# ============================================================

print()
print("=" * 70)
print("              FINAL END-TO-END RAG BENCHMARK")
print("=" * 70)

print()
print(f"Queries: {len(records)}")
print(f"Dense candidates: {DENSE_CANDIDATE_K}")
print(f"Final K: {FINAL_TOP_K}")
print(f"Dense threshold: {DENSE_THRESHOLD}")
print(f"Cross threshold: {CROSS_THRESHOLD}")

print()
print("Running evaluation...")


for count, record in enumerate(
    records,
    start=1
):

    query = record["eng_query"]

    ground_truth_answerable = any(
        record["is_selected"]
    )

    if ground_truth_answerable:

        answerable_total += 1

    else:

        unanswerable_total += 1


    total_start = time.perf_counter()


    # ========================================================
    # 1. DENSE RETRIEVAL
    # ========================================================

    dense_start = time.perf_counter()

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

    dense_latency = (
        time.perf_counter()
        - dense_start
    ) * 1000

    dense_latencies.append(
        dense_latency
    )


    # ========================================================
    # 2. RETRIEVAL RECALL
    # ========================================================

    dense_found_rank = None

    for rank, idx in enumerate(
        dense_indices[0],
        start=1
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]

        if (
            meta["query_id"]
            == record["query_id"]
            and meta["is_selected"] == 1
        ):

            dense_found_rank = rank
            break


    if dense_found_rank is not None:

        retrieval_hits += 1

        reciprocal_ranks.append(
            1 / dense_found_rank
        )

    else:

        reciprocal_ranks.append(0)


    # ========================================================
    # 3. CROSS-ENCODER RERANKING
    # ========================================================

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        dense_indices[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append({

            "index": idx,

            "dense_score": float(
                dense_score
            ),

            "text": metadata[idx]["text"]
        })


    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]


    cross_start = time.perf_counter()

    cross_scores = cross_model.predict(
        pairs
    )

    cross_latency = (
        time.perf_counter()
        - cross_start
    ) * 1000

    cross_latencies.append(
        cross_latency
    )


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


    final_candidates = candidates[
        :FINAL_TOP_K
    ]


    # ========================================================
    # 4. RERANKED RETRIEVAL RECALL
    # ========================================================

    reranked_found = False

    for rank, candidate in enumerate(
        final_candidates,
        start=1
    ):

        meta = metadata[
            candidate["index"]
        ]

        if (
            meta["query_id"]
            == record["query_id"]
            and meta["is_selected"] == 1
        ):

            reranked_found = True

            break


    # ========================================================
    # 5. GUARDRAIL
    # ========================================================

    if final_candidates:

        best = final_candidates[0]

        best_dense = best["dense_score"]
        best_cross = best["cross_score"]

        predicted_answerable = (
            best_dense >= DENSE_THRESHOLD
            and best_cross >= CROSS_THRESHOLD
        )

    else:

        predicted_answerable = False

        best_dense = -999
        best_cross = -999


    guardrail_results.append({

        "query": query,

        "ground_truth": ground_truth_answerable,

        "predicted": predicted_answerable,

        "dense": best_dense,

        "cross": best_cross
    })


    # ========================================================
    # 6. CLASSIFICATION METRICS
    # ========================================================

    if predicted_answerable:

        if ground_truth_answerable:

            correct_answers += 1

        else:

            false_answers += 1

    else:

        if ground_truth_answerable:

            false_refusals += 1

        else:

            correct_refusals += 1


    # ========================================================
    # 7. TOTAL LATENCY
    # ========================================================

    total_latency = (
        time.perf_counter()
        - total_start
    ) * 1000

    total_latencies.append(
        total_latency
    )


    if count % 10 == 0:

        print(
            f"Processed {count}/{len(records)}"
        )


# ============================================================
# NUMPY ARRAYS
# ============================================================

dense_latencies = np.asarray(
    dense_latencies
)

cross_latencies = np.asarray(
    cross_latencies
)

total_latencies = np.asarray(
    total_latencies
)

reciprocal_ranks = np.asarray(
    reciprocal_ranks
)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = (
    correct_answers
    + correct_refusals
) / len(records)

guardrail_recall = (
    correct_answers
    / answerable_total
    if answerable_total
    else 0
)

refusal_precision = (
    correct_refusals
    / (
        correct_refusals
        + false_refusals
    )
    if (
        correct_refusals
        + false_refusals
    )
    else 0
)

retrieval_recall = (
    retrieval_hits
    / answerable_total
    if answerable_total
    else 0
)

mrr = np.mean(
    reciprocal_ranks
)


# ============================================================
# REPORT
# ============================================================

print()
print()
print("=" * 70)
print("                 FINAL RAG RESULTS")
print("=" * 70)


print()
print("DATASET")

print(
    f"Total queries:          {len(records)}"
)

print(
    f"Answerable:             {answerable_total}"
)

print(
    f"Unanswerable:           {unanswerable_total}"
)


print()
print("GUARDRAIL")

print(
    f"Correct answers:        {correct_answers}"
)

print(
    f"Correct refusals:       {correct_refusals}"
)

print(
    f"False answers:          {false_answers}"
)

print(
    f"False refusals:         {false_refusals}"
)

print(
    f"Accuracy:               {accuracy:.3f}"
)

print(
    f"Answer recall:          {guardrail_recall:.3f}"
)

print(
    f"Refusal precision:      {refusal_precision:.3f}"
)


print()
print("RETRIEVAL")

print(
    f"Dense Recall@20:        {retrieval_recall:.3f}"
)

print(
    f"Dense MRR@20:           {mrr:.3f}"
)


print()
print("LATENCY")

print(
    f"Dense P50:              "
    f"{np.percentile(dense_latencies, 50):.2f} ms"
)

print(
    f"Dense P70:              "
    f"{np.percentile(dense_latencies, 70):.2f} ms"
)

print(
    f"Cross P50:              "
    f"{np.percentile(cross_latencies, 50):.2f} ms"
)

print(
    f"Cross P70:              "
    f"{np.percentile(cross_latencies, 70):.2f} ms"
)

print(
    f"Total P50:              "
    f"{np.percentile(total_latencies, 50):.2f} ms"
)

print(
    f"Total P70:              "
    f"{np.percentile(total_latencies, 70):.2f} ms"
)

print(
    f"Total P100:             "
    f"{np.max(total_latencies):.2f} ms"
)

print(
    f"Total Mean:             "
    f"{np.mean(total_latencies):.2f} ms"
)


print()
print("=" * 70)
print("Evaluation complete.")
print("=" * 70)