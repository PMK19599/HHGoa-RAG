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

INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Index ready: {index.ntotal} passages")

latencies = []
hits = 0
reciprocal_ranks = []

print(f"\nBenchmarking {NUM_QUERIES} queries...")

evaluated = 0

for record in records[:NUM_QUERIES]:
    
    if not any(record["is_selected"]):
        continue

    evaluated += 1

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
print(f"Answerable queries: {evaluated}")
print(f"Recall@{TOP_K}: {hits / evaluated:.3f}")

print("\nLATENCY")
print(f"P50: {np.percentile(latencies, 50):.2f} ms")
print(f"P70: {np.percentile(latencies, 70):.2f} ms")
print(f"P100: {np.max(latencies):.2f} ms")
print(f"Mean: {np.mean(latencies):.2f} ms")
print(f"Min: {np.min(latencies):.2f} ms")