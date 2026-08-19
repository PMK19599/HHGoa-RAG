import json
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path("data/sample_hin.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5
NUM_QUERIES = 100

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

passages = []
metadata = []

for record in records:
    for i, passage in enumerate(record["english_passages"]):
        passages.append(passage)
        metadata.append({
            "query_id": record["query_id"],
            "passage_index": i,
            "is_selected": record["is_selected"][i],
        })

print(f"Embedding {len(passages)} passages...")

embeddings = model.encode(
    passages,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print(f"Index ready: {index.ntotal} passages")

latencies = []
hits = 0
reciprocal_ranks = []

print(f"\nBenchmarking {NUM_QUERIES} queries...")

for record in records[:NUM_QUERIES]:

    query = record["eng_query"]

    start = time.perf_counter()

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    scores, indices = index.search(q_emb, TOP_K)

    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies.append(elapsed_ms)

    found_rank = None

    for rank, idx in enumerate(indices[0], start=1):
        meta = metadata[idx]

        if (
            meta["query_id"] == record["query_id"]
            and meta["is_selected"] == 1
        ):
            found_rank = rank
            break

    if found_rank:
        hits += 1
        reciprocal_ranks.append(1 / found_rank)
    else:
        reciprocal_ranks.append(0)

latencies = np.array(latencies)

print("\n=== RESULTS ===")

print(f"Queries: {NUM_QUERIES}")
print(f"Recall@{TOP_K}: {hits / NUM_QUERIES:.3f}")
print(f"MRR@{TOP_K}: {np.mean(reciprocal_ranks):.3f}")

print("\nLATENCY")
print(f"P50: {np.percentile(latencies, 50):.2f} ms")
print(f"P70: {np.percentile(latencies, 70):.2f} ms")
print(f"P100: {np.max(latencies):.2f} ms")
print(f"Mean: {np.mean(latencies):.2f} ms")
print(f"Min: {np.min(latencies):.2f} ms")