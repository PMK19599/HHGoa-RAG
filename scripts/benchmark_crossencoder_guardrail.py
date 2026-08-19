import json
import time
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

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross-encoder...")
cross_model = CrossEncoder(CROSS_MODEL)

print("Loading index...")
index = faiss.read_index(str(INDEX_PATH))

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# Warm up
embed_model.encode(["warmup"], normalize_embeddings=True)
cross_model.predict([["test query", "test passage"]])

results = []
latencies = []

print("Running semantic guardrail test...")

for record in records[:NUM_QUERIES]:

    query = record["eng_query"]

    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    _, indices = index.search(q_emb, 1)

    top_idx = int(indices[0][0])
    top_passage = metadata[top_idx]["text"]

    start = time.perf_counter()

    cross_score = float(
        cross_model.predict(
            [[query, top_passage]]
        )[0]
    )

    guardrail_ms = (
        time.perf_counter() - start
    ) * 1000

    latencies.append(guardrail_ms)

    results.append({
        "score": cross_score,
        "answerable": any(record["is_selected"])
    })

answerable = [
    r["score"] for r in results
    if r["answerable"]
]

unanswerable = [
    r["score"] for r in results
    if not r["answerable"]
]

print("\n=== CROSS-ENCODER DISTRIBUTION ===")
print(f"Answerable mean: {np.mean(answerable):.4f}")
print(f"Answerable median: {np.median(answerable):.4f}")
print(f"Unanswerable mean: {np.mean(unanswerable):.4f}")
print(f"Unanswerable median: {np.median(unanswerable):.4f}")

print("\n=== GUARDRAIL LATENCY ===")
print(f"P50: {np.percentile(latencies, 50):.2f} ms")
print(f"P70: {np.percentile(latencies, 70):.2f} ms")
print(f"P100: {np.max(latencies):.2f} ms")

print("\n=== THRESHOLD TEST ===")

for threshold in [-4, -2, 0, 2, 4, 6]:

    correct = 0
    false_answers = 0
    false_refusals = 0

    for r in results:

        predicted_answerable = r["score"] >= threshold

        if predicted_answerable == r["answerable"]:
            correct += 1

        if predicted_answerable and not r["answerable"]:
            false_answers += 1

        if not predicted_answerable and r["answerable"]:
            false_refusals += 1

    print(
        f"Threshold {threshold:>2} | "
        f"Accuracy {correct / len(results):.3f} | "
        f"False answers {false_answers} | "
        f"False refusals {false_refusals}"
    )