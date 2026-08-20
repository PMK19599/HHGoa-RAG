import json
import time
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

NUM_QUERIES = 100
FINAL_TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading saved FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

passages = [item["text"] for item in metadata]

print("Preparing BM25...")
tokenized_passages = [
    passage.lower().split()
    for passage in passages
]

bm25 = BM25Okapi(tokenized_passages)

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# Warm up BGE once
model.encode(["warmup"], normalize_embeddings=True)

hits = 0
reciprocal_ranks = []
latencies = []

print(f"Benchmarking {NUM_QUERIES} hybrid queries...")

for record in records[:NUM_QUERIES]:

    query = record["eng_query"]

    start = time.perf_counter()

    # -------- Dense / BGE --------
    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    _, dense_indices = index.search(
        q_emb,
        CANDIDATE_K
    )

    dense_indices = dense_indices[0]

    # -------- BM25 --------
    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    sparse_indices = np.argsort(
        bm25_scores
    )[::-1][:CANDIDATE_K]

    # -------- Reciprocal Rank Fusion --------
    rrf_scores = {}

    for rank, idx in enumerate(dense_indices, start=1):
        rrf_scores[idx] = (
            rrf_scores.get(idx, 0)
            + 1 / (RRF_K + rank)
        )

    for rank, idx in enumerate(sparse_indices, start=1):
        rrf_scores[idx] = (
            rrf_scores.get(idx, 0)
            + 1 / (RRF_K + rank)
        )

    final_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:FINAL_TOP_K]

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    latencies.append(elapsed_ms)

    found_rank = None

    for rank, idx in enumerate(final_indices, start=1):

        meta = metadata[idx]

        if (
            meta["query_id"] == record["query_id"]
            and meta["is_selected"] == 1
        ):
            found_rank = rank
            break

    if found_rank:
        hits += 1
        reciprocal_ranks.append(
            1 / found_rank
        )
    else:
        reciprocal_ranks.append(0)

latencies = np.array(latencies)

print("\n=== HYBRID RESULTS ===")
print(f"Queries: {NUM_QUERIES}")
print(f"Recall@5: {hits / NUM_QUERIES:.3f}")
print(f"MRR@5: {np.mean(reciprocal_ranks):.3f}")

print("\nLATENCY")
print(f"P50: {np.percentile(latencies, 50):.2f} ms")
print(f"P70: {np.percentile(latencies, 70):.2f} ms")
print(f"P100: {np.max(latencies):.2f} ms")
print(f"Mean: {np.mean(latencies):.2f} ms")