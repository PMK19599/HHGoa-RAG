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

model.encode(["warmup"], normalize_embeddings=True)

results = []

for record in records[:NUM_QUERIES]:

    query = record["eng_query"]

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    scores, indices = index.search(q_emb, 2)

    top1 = float(scores[0][0])
    top2 = float(scores[0][1])
    gap = top1 - top2

    answerable = any(record["is_selected"])

    results.append({
        "query": query,
        "top1": top1,
        "top2": top2,
        "gap": gap,
        "answerable": answerable
    })

answerable_gaps = [
    r["gap"]
    for r in results
    if r["answerable"]
]

unanswerable_gaps = [
    r["gap"]
    for r in results
    if not r["answerable"]
]

print("\n=== GAP DISTRIBUTION ===")

print(f"Answerable mean gap: {np.mean(answerable_gaps):.4f}")
print(f"Answerable median gap: {np.median(answerable_gaps):.4f}")

print(f"Unanswerable mean gap: {np.mean(unanswerable_gaps):.4f}")
print(f"Unanswerable median gap: {np.median(unanswerable_gaps):.4f}")

print("\n=== TWO-SIGNAL TEST ===")

score_thresholds = [0.75, 0.80, 0.85]
gap_thresholds = [0.01, 0.02, 0.03, 0.05]

for s_threshold in score_thresholds:
    for g_threshold in gap_thresholds:

        correct = 0
        false_answers = 0
        false_refusals = 0

        for r in results:

            predicted_answerable = (
                r["top1"] >= s_threshold
                and r["gap"] >= g_threshold
            )

            if predicted_answerable == r["answerable"]:
                correct += 1

            if predicted_answerable and not r["answerable"]:
                false_answers += 1

            if not predicted_answerable and r["answerable"]:
                false_refusals += 1

        accuracy = correct / len(results)

        print(
            f"Score {s_threshold:.2f} | "
            f"Gap {g_threshold:.2f} | "
            f"Accuracy {accuracy:.3f} | "
            f"False answers {false_answers} | "
            f"False refusals {false_refusals}"
        )