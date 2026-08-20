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
FINAL_TOP_K = 5

CANDIDATE_K_VALUES = [10, 20, 30, 50]


print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross-encoder...")
cross_model = CrossEncoder(CROSS_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata...")
with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Index ready: {index.ntotal} passages")

print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["test query", "test passage"]]
)


all_results = {}


for candidate_k in CANDIDATE_K_VALUES:

    print("\n" + "=" * 70)
    print(f"Testing candidate depth: {candidate_k}")
    print("=" * 70)

    recall_hits = 0
    reciprocal_ranks = []

    dense_latencies = []
    cross_latencies = []
    total_latencies = []

    dense_recall_hits = 0

    for count, record in enumerate(
        records[:NUM_QUERIES],
        start=1
    ):

        query = record["eng_query"]

        total_start = time.perf_counter()

        # --------------------------------------------------
        # 1. Dense retrieval
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
            candidate_k
        )

        dense_elapsed = (
            time.perf_counter() - dense_start
        ) * 1000

        dense_latencies.append(dense_elapsed)

        # Check whether a relevant passage entered
        # the candidate pool at all.
        dense_found = False

        for idx in dense_indices[0]:

            idx = int(idx)

            if idx < 0 or idx >= len(metadata):
                continue

            meta = metadata[idx]

            if (
                meta["query_id"] == record["query_id"]
                and meta["is_selected"] == 1
            ):
                dense_found = True
                break

        if dense_found:
            dense_recall_hits += 1

        # --------------------------------------------------
        # 2. Cross-encoder reranking
        # --------------------------------------------------

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

        candidates.sort(
            key=lambda x: x["cross_score"],
            reverse=True
        )

        final_candidates = candidates[:FINAL_TOP_K]

        # --------------------------------------------------
        # 3. Reranked evaluation
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

    dense_latencies = np.asarray(dense_latencies)
    cross_latencies = np.asarray(cross_latencies)
    total_latencies = np.asarray(total_latencies)
    reciprocal_ranks = np.asarray(reciprocal_ranks)

    all_results[candidate_k] = {
        "dense_recall": dense_recall_hits / NUM_QUERIES,
        "rerank_recall": recall_hits / NUM_QUERIES,
        "mrr": np.mean(reciprocal_ranks),
        "dense_p50": np.percentile(dense_latencies, 50),
        "cross_p50": np.percentile(cross_latencies, 50),
        "total_p50": np.percentile(total_latencies, 50),
        "total_p70": np.percentile(total_latencies, 70),
        "total_p100": np.max(total_latencies),
        "total_mean": np.mean(total_latencies)
    }


print("\n")
print("=" * 78)
print("              CANDIDATE DEPTH RERANKING BENCHMARK")
print("=" * 78)

print("\n")
print(
    "Candidates | Dense Recall | Recall@5 | MRR@5 | "
    "Dense P50 | Cross P50 | Total P50 | Total P70 | Total P100"
)

print("-" * 110)

for candidate_k in CANDIDATE_K_VALUES:

    r = all_results[candidate_k]

    print(
        f"{candidate_k:9d} | "
        f"{r['dense_recall']:12.3f} | "
        f"{r['rerank_recall']:8.3f} | "
        f"{r['mrr']:6.3f} | "
        f"{r['dense_p50']:9.2f} | "
        f"{r['cross_p50']:9.2f} | "
        f"{r['total_p50']:9.2f} | "
        f"{r['total_p70']:9.2f} | "
        f"{r['total_p100']:10.2f}"
    )


print("\n")
print("=" * 78)
print("Evaluation complete.")
print("=" * 78)
