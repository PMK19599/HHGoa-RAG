import json
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

query = input("\nEnter query: ")

for run in range(1, 6):

    start = time.perf_counter()

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    embedding_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()

    scores, indices = index.search(q_emb, TOP_K)

    retrieval_ms = (time.perf_counter() - start) * 1000

    print(
        f"Run {run}: "
        f"Embedding {embedding_ms:.2f} ms | "
        f"FAISS {retrieval_ms:.2f} ms | "
        f"Total {embedding_ms + retrieval_ms:.2f} ms"
    )

print("\n=== TOP RESULTS ===")

for rank, idx in enumerate(indices[0], start=1):
    item = metadata[idx]

    print(f"\n#{rank}")
    print(f"Score: {scores[0][rank-1]:.4f}")
    print(f"Query ID: {item['query_id']}")
    print(item["text"][:500])

print("\n=== LATENCY ===")
print(f"Embedding: {embedding_ms:.2f} ms")
print(f"FAISS: {retrieval_ms:.2f} ms")
print(f"Total: {embedding_ms + retrieval_ms:.2f} ms")