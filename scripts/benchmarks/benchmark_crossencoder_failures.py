import json
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
FINAL_K = 5


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
embed_model.encode(["warmup"], normalize_embeddings=True)
cross_model.predict([["test query", "test passage"]])


failures = []
dense_misses = 0
rerank_hits = 0

print("\nDiagnosing CrossEncoder ranking...")

for count, record in enumerate(records[:NUM_QUERIES], start=1):

    query = record["eng_query"]

    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    dense_scores, dense_indices = index.search(
        q_emb,
        CANDIDATE_K
    )

    candidates = []

    relevant_dense_ranks = []

    for dense_rank, (dense_score, idx) in enumerate(
        zip(dense_scores[0], dense_indices[0]),
        start=1
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]

        is_relevant = (
            meta["query_id"] == record["query_id"]
            and meta["is_selected"] == 1
        )

        candidates.append({
            "index": idx,
            "dense_rank": dense_rank,
            "dense_score": float(dense_score),
            "text": meta["text"],
            "relevant": is_relevant
        })

        if is_relevant:
            relevant_dense_ranks.append(dense_rank)

    if not relevant_dense_ranks:
        dense_misses += 1
        continue

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    cross_scores = cross_model.predict(pairs)

    for candidate, score in zip(
        candidates,
        cross_scores
    ):
        candidate["cross_score"] = float(score)

    candidates.sort(
        key=lambda x: x["cross_score"],
        reverse=True
    )

    relevant_after_rerank = None
    relevant_candidate = None

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):

        if candidate["relevant"]:
            relevant_after_rerank = rank
            relevant_candidate = candidate
            break

    if relevant_after_rerank is not None:

        if relevant_after_rerank <= FINAL_K:
            rerank_hits += 1

        else:

            top5 = candidates[:FINAL_K]

            failures.append({
                "query": query,
                "dense_rank": relevant_candidate["dense_rank"],
                "rerank_rank": relevant_after_rerank,
                "dense_score": relevant_candidate["dense_score"],
                "cross_score": relevant_candidate["cross_score"],
                "relevant_text": relevant_candidate["text"],
                "top5": [
                    {
                        "rank": i + 1,
                        "dense_rank": c["dense_rank"],
                        "dense_score": c["dense_score"],
                        "cross_score": c["cross_score"],
                        "text": c["text"]
                    }
                    for i, c in enumerate(top5)
                ]
            })

    if count % 10 == 0:
        print(f"Processed {count}/{NUM_QUERIES}")


print("\n" + "=" * 78)
print("             CROSSENCODER RANKING DIAGNOSTIC")
print("=" * 78)

print(f"\nQueries evaluated:        {NUM_QUERIES}")
print(f"Dense Top-{CANDIDATE_K} misses:     {dense_misses}")
print(f"Relevant in Dense Top-{CANDIDATE_K}: {NUM_QUERIES - dense_misses}")
print(f"Relevant reaching Top-{FINAL_K}:     {rerank_hits}")
print(f"CrossEncoder ranking failures: {len(failures)}")


if failures:

    print("\n" + "=" * 78)
    print("        CASES WHERE CROSSENCODER PUSHED RELEVANT DOWN")
    print("=" * 78)

    for number, case in enumerate(failures, start=1):

        print("\n" + "-" * 78)
        print(f"CASE #{number}")
        print(f"Query: {case['query']}")
        print(f"Relevant dense rank: {case['dense_rank']}")
        print(f"Relevant rerank rank: {case['rerank_rank']}")
        print(f"Relevant dense score: {case['dense_score']:.4f}")
        print(f"Relevant cross score: {case['cross_score']:.4f}")

        print("\nRELEVANT PASSAGE:")
        print(case["relevant_text"][:700])

        print("\nTOP-5 THAT BEAT IT:")

        for item in case["top5"]:

            print(
                f"\nRank {item['rank']} | "
                f"Dense rank {item['dense_rank']} | "
                f"Dense {item['dense_score']:.4f} | "
                f"Cross {item['cross_score']:.4f}"
            )

            print(item["text"][:400])


print("\n" + "=" * 78)
print("Evaluation complete.")
print("=" * 78)
