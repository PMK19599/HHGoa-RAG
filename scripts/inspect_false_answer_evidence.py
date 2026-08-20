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

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0


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


false_answers = []


print("\nFinding current false-answer cases...")

for count, record in enumerate(records[:NUM_QUERIES], start=1):

    query = record["eng_query"]
    answerable = bool(any(record["is_selected"]))

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

    for dense_rank, (dense_score, idx) in enumerate(
        zip(dense_scores[0], dense_indices[0]),
        start=1
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]

        candidates.append({
            "index": idx,
            "dense_rank": dense_rank,
            "dense": float(dense_score),
            "text": meta["text"],
            "selected": (
                meta["query_id"] == record["query_id"]
                and meta["is_selected"] == 1
            )
        })

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    cross_scores = cross_model.predict(pairs)

    for candidate, score in zip(
        candidates,
        cross_scores
    ):
        candidate["cross"] = float(score)

    candidates.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    top5 = candidates[:FINAL_K]

    best = top5[0]

    predicted_answerable = (
        best["dense"] >= DENSE_THRESHOLD
        and best["cross"] >= CROSS_THRESHOLD
    )

    if predicted_answerable and not answerable:

        false_answers.append({
            "query": query,
            "best": best,
            "top5": top5
        })

    if count % 10 == 0:
        print(f"Processed {count}/{NUM_QUERIES}")


print("\n" + "=" * 80)
print("              CURRENT FALSE-ANSWER CASE ANALYSIS")
print("=" * 80)

print(f"\nFalse answers found: {len(false_answers)}")


for number, case in enumerate(false_answers, start=1):

    print("\n" + "=" * 80)
    print(f"CASE #{number}")
    print("=" * 80)

    print(f"\nQuery:")
    print(case["query"])

    best = case["best"]

    print("\nGUARDRAIL SIGNALS")
    print(
        f"Dense: {best['dense']:.4f} "
        f"(threshold {DENSE_THRESHOLD:.2f})"
    )

    print(
        f"Cross: {best['cross']:.4f} "
        f"(threshold {CROSS_THRESHOLD:.2f})"
    )

    print(f"Dense rank: {best['dense_rank']}")
    print(f"Dataset selected: {best['selected']}")

    print("\nTOP-5 EVIDENCE")

    for rank, candidate in enumerate(
        case["top5"],
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"Rank {rank} | "
            f"Dense rank {candidate['dense_rank']} | "
            f"Dense {candidate['dense']:.4f} | "
            f"Cross {candidate['cross']:.4f}"
        )

        print(
            f"Dataset selected: "
            f"{candidate['selected']}"
        )

        print("\nPassage:")
        print(candidate["text"][:1200])


print("\n" + "=" * 80)
print("Evaluation complete.")
print("=" * 80)
