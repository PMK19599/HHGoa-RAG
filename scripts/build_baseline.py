import json
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path("data/sample_hin.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

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

print(f"Total passages: {len(passages)}")

print("Embedding passages...")
start = time.perf_counter()

embeddings = model.encode(
    passages,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embed_time = time.perf_counter() - start

embeddings = np.asarray(embeddings, dtype="float32")

print(f"Embedding completed in {embed_time:.2f}s")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"FAISS index size: {index.ntotal}")

query = records[0]["eng_query"]

print(f"\nTEST QUERY:\n{query}")

start = time.perf_counter()

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

query_embedding = np.asarray(query_embedding, dtype="float32")

embedding_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()

scores, indices = index.search(query_embedding, 5)

retrieval_ms = (time.perf_counter() - start) * 1000

print("\nTOP 5 RESULTS:")

for rank, idx in enumerate(indices[0], start=1):
    meta = metadata[idx]

    print(f"\n#{rank}")
    print(f"Score: {scores[0][rank-1]:.4f}")
    print(f"Relevant: {meta['is_selected']}")
    print(passages[idx][:400])

print(f"\nQuery embedding: {embedding_ms:.2f} ms")
print(f"FAISS retrieval: {retrieval_ms:.2f} ms")
print(f"Total retrieval path: {embedding_ms + retrieval_ms:.2f} ms")