import json
import re
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path("data/sample_hin.jsonl")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5
NUM_QUERIES = 100
MAX_WORDS = 90


def sentence_chunks(text, max_words=90):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()

        if not words:
            continue

        if current_words + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

chunks = []
metadata = []

for record in records:
    for passage_index, passage in enumerate(record["english_passages"]):

        passage_chunks = sentence_chunks(passage, MAX_WORDS)

        for chunk_index, chunk in enumerate(passage_chunks):
            chunks.append(chunk)

            metadata.append({
                "query_id": record["query_id"],
                "passage_index": passage_index,
                "chunk_index": chunk_index,
                "is_selected": record["is_selected"][passage_index],
            })

print(f"Original passages: {sum(len(r['english_passages']) for r in records)}")
print(f"Generated chunks: {len(chunks)}")

print("Embedding chunks...")

embeddings = model.encode(
    chunks,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print(f"Index ready: {index.ntotal} chunks")

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

    latency = (time.perf_counter() - start) * 1000
    latencies.append(latency)

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

print("\n=== SENTENCE CHUNK RESULTS ===")

print(f"Queries: {NUM_QUERIES}")
print(f"Max words: {MAX_WORDS}")
print(f"Recall@{TOP_K}: {hits / NUM_QUERIES:.3f}")
print(f"MRR@{TOP_K}: {np.mean(reciprocal_ranks):.3f}")

print("\nLATENCY")
print(f"P50: {np.percentile(latencies, 50):.2f} ms")
print(f"P70: {np.percentile(latencies, 70):.2f} ms")
print(f"P100: {np.max(latencies):.2f} ms")
print(f"Mean: {np.mean(latencies):.2f} ms")
print(f"Min: {np.min(latencies):.2f} ms")