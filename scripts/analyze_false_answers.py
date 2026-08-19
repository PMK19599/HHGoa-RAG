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

DENSE_K = 20
FINAL_K = 5
DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0

print("Loading models...")
embed_model = SentenceTransformer(EMBED_MODEL)
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
cross_model.predict([["warmup", "warmup"]])

false_answers = []

print("\nAnalyzing production guardrail...\n")

for record in records[:100]:

    query = record["eng_query"]
    answerable = any(record["is_selected"])

    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    dense_scores, indices = index.search(
        q_emb,
        DENSE_K
    )

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        indices[0]
    ):
        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append({
            "dense": float(dense_score),
            "text": metadata[idx]["text"],
            "index": idx
        })

    if not candidates:
        continue

    pairs = [
        [query, c["text"]]
        for c in candidates
    ]

    cross_scores = cross_model.predict(pairs)

    for candidate, cross_score in zip(candidates, cross_scores):
        candidate["cross"] = float(cross_score)

    candidates.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    top = candidates[:FINAL_K]

    best = top[0]

    predicted_answer = (
        best["dense"] >= DENSE_THRESHOLD
        and best["cross"] >= CROSS_THRESHOLD
    )

    if predicted_answer and not answerable:

        false_answers.append({
            "query": query,
            "dense": best["dense"],
            "cross": best["cross"],
            "results": top
        })


print("\n" + "=" * 80)
print("FALSE-ANSWER ANALYSIS")
print("=" * 80)

print(f"\nFalse answers found: {len(false_answers)}")

for i, case in enumerate(false_answers, 1):

    print("\n" + "-" * 80)
    print(f"CASE #{i}")
    print("-" * 80)

    print(f"Query: {case['query']}")
    print(f"Best dense: {case['dense']:.4f}")
    print(f"Best cross: {case['cross']:.4f}")

    print("\nTOP-5 RERANKED EVIDENCE:")

    for rank, result in enumerate(case["results"], 1):

        print(f"\n[{rank}]")
        print(f"Dense: {result['dense']:.4f}")
        print(f"Cross: {result['cross']:.4f}")
        print(f"Index: {result['index']}")
        print(result["text"][:700])

print("\n" + "=" * 80)
print("END")
print("=" * 80)
