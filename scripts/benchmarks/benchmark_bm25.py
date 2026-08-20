import json
import time
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA = Path("data/sample_hin.jsonl")

TOP_K = 5
NUM_QUERIES = 100

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

print(f"Preparing BM25 over {len(passages)} passages...")

tokenized_passages = [
    passage.lower().split()
    for passage in passages
]

bm25 = BM25Okapi(tokenized_passages)

hits = 0
reciprocal_ranks = []
latencies = []

print(f"Benchmarking {NUM_QUERIES} queries...")

for record in records[:NUM_QUERIES]:

    query_tokens = record["eng_query"].lower().split()

    start = time.perf_counter()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies.append(elapsed_ms)

    found_rank = None

    for rank, idx in enumerate(top_indices, start=1):

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

print("\n=== BM25 RESULTS ===")
print(f"Queries: {NUM_QUERIES}")
print(f"Recall@{TOP_K}: {hits / NUM_QUERIES:.3f}")
print(f"MRR@{TOP_K}: {np.mean(reciprocal_ranks):.3f}")

print("\nLATENCY")
print(f"P50: {np.percentile(latencies, 50):.2f} ms")
print(f"P70: {np.percentile(latencies, 70):.2f} ms")
print(f"P100: {np.max(latencies):.2f} ms")
print(f"Mean: {np.mean(latencies):.2f} ms")