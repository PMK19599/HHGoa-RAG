import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
NUM_QUERIES = 100

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# Warm-up
model.encode(["warmup"], normalize_embeddings=True)

results = []

for record in records[:NUM_QUERIES]:

    query = record["eng_query"]

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    scores, indices = index.search(q_emb, 1)

    top_score = float(scores[0][0])

    answerable = any(record["is_selected"])

    results.append({
        "query": query,
        "score": top_score,
        "answerable": answerable
    })


answerable_scores = [
    r["score"]
    for r in results
    if r["answerable"]
]

unanswerable_scores = [
    r["score"]
    for r in results
    if not r["answerable"]
]

print("\n=== SCORE DISTRIBUTION ===")

print(f"Answerable queries: {len(answerable_scores)}")
print(
    f"Answerable mean score: "
    f"{np.mean(answerable_scores):.4f}"
)
print(
    f"Answerable min score: "
    f"{np.min(answerable_scores):.4f}"
)

print()

print(f"Unanswerable queries: {len(unanswerable_scores)}")
print(
    f"Unanswerable mean score: "
    f"{np.mean(unanswerable_scores):.4f}"
)
print(
    f"Unanswerable max score: "
    f"{np.max(unanswerable_scores):.4f}"
)


print("\n=== THRESHOLD TEST ===")

for threshold in [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]:

    correct = 0
    false_answers = 0
    false_refusals = 0

    for r in results:

        predicted_answerable = (
            r["score"] >= threshold
        )

        if predicted_answerable == r["answerable"]:
            correct += 1

        if (
            predicted_answerable
            and not r["answerable"]
        ):
            false_answers += 1

        if (
            not predicted_answerable
            and r["answerable"]
        ):
            false_refusals += 1

    accuracy = correct / len(results)

    print(
        f"Threshold {threshold:.2f} | "
        f"Accuracy {accuracy:.3f} | "
        f"False answers {false_answers} | "
        f"False refusals {false_refusals}"
    )