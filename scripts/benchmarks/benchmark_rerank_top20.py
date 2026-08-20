import json
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder


DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

NUM_QUERIES = 100

CANDIDATE_K = 20
FINAL_TOP_K = 5


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


records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))


print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["test query", "test passage"]]
)


recall_hits = 0
reciprocal_ranks = []

dense_latencies = []
cross_latencies = []
total_latencies = []


print(f"\nEvaluating {NUM_QUERIES} queries...")
print(
    f"Dense candidates: {CANDIDATE_K} | "
    f"Final results: {FINAL_TOP_K}"
)


for count, record in enumerate(records[:NUM_QUERIES], start=1):

    query = record["eng_query"]

    total_start = time.perf_counter()

    # --------------------------------------------------
    # 1. Dense retrieval: Top-20
    # --------------------------------------------------

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
        CANDIDATE_K
    )

    dense_elapsed = (
        time.perf_counter() - dense_start
    ) * 1000

    dense_latencies.append(dense_elapsed)


    # --------------------------------------------------
    # 2. Cross-encoder reranking
    # --------------------------------------------------

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        dense_indices[0]
    ):

        idx = int(idx)

        candidates.append({
            "index": idx,
            "dense_score": float(dense_score),
            "text": metadata[idx]["text"]
        })


    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]


    cross_start = time.perf_counter()

    cross_scores = cross_model.predict(pairs)

    cross_elapsed = (
        time.perf_counter() - cross_start
    ) * 1000

    cross_latencies.append(cross_elapsed)


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


    final_candidates = candidates[:FINAL_TOP_K]


    # --------------------------------------------------
    # 3. Evaluation
    # --------------------------------------------------

    found_rank = None

    for rank, candidate in enumerate(
        final_candidates,
        start=1
    ):

        meta = metadata[candidate["index"]]

        if (
            meta["query_id"] == record["query_id"]
            and meta["is_selected"] == 1
        ):

            found_rank = rank
            break


    if found_rank is not None:

        recall_hits += 1

        reciprocal_ranks.append(
            1 / found_rank
        )

    else:

        reciprocal_ranks.append(0)


    total_elapsed = (
        time.perf_counter() - total_start
    ) * 1000

    total_latencies.append(total_elapsed)


    if count % 10 == 0:
        print(f"Processed {count}/{NUM_QUERIES}")


# ------------------------------------------------------
# Final statistics
# ------------------------------------------------------

dense_latencies = np.asarray(dense_latencies)
cross_latencies = np.asarray(cross_latencies)
total_latencies = np.asarray(total_latencies)
reciprocal_ranks = np.asarray(reciprocal_ranks)


print("\n")
print("=" * 60)
print("          BGE TOP-20 + CROSS-ENCODER RERANK")
print("=" * 60)

print("\nRETRIEVAL")

print(
    f"Dense Candidate K:       {CANDIDATE_K}"
)

print(
    f"Final K:                 {FINAL_TOP_K}"
)

print(
    f"Reranked Recall@5:       "
    f"{recall_hits / NUM_QUERIES:.3f}"
)

print(
    f"Reranked MRR@5:          "
    f"{np.mean(reciprocal_ranks):.3f}"
)


print("\nLATENCY")

print(
    f"Dense P50:               "
    f"{np.percentile(dense_latencies, 50):.2f} ms"
)

print(
    f"Dense P70:               "
    f"{np.percentile(dense_latencies, 70):.2f} ms"
)

print(
    f"Cross-encoder P50:       "
    f"{np.percentile(cross_latencies, 50):.2f} ms"
)

print(
    f"Cross-encoder P70:       "
    f"{np.percentile(cross_latencies, 70):.2f} ms"
)

print(
    f"Total pipeline P50:      "
    f"{np.percentile(total_latencies, 50):.2f} ms"
)

print(
    f"Total pipeline P70:      "
    f"{np.percentile(total_latencies, 70):.2f} ms"
)

print(
    f"Total pipeline P100:     "
    f"{np.max(total_latencies):.2f} ms"
)

print(
    f"Total pipeline Mean:     "
    f"{np.mean(total_latencies):.2f} ms"
)


print("\n")
print("=" * 60)
print("Evaluation complete.")
print("=" * 60)