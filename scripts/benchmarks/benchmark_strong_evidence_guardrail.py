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
DENSE_K = 20
FINAL_K = 5

DENSE_THRESHOLDS = [0.65, 0.70, 0.75]
CROSS_THRESHOLDS = [6.0, 7.0, 8.0]
STRONG_COUNTS = [1, 2, 3]


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
cross_model.predict([["warmup", "warmup passage"]])


results = []

print("\nBuilding strong-evidence signals...")

for n, record in enumerate(records[:NUM_QUERIES], 1):

    query = record["eng_query"]
    answerable = bool(any(record["is_selected"]))

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
            "text": metadata[idx]["text"]
        })

    if not candidates:
        continue

    pairs = [
        [query, c["text"]]
        for c in candidates
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

    top = candidates[:FINAL_K]

    top1_dense = top[0]["dense"]
    top1_cross = top[0]["cross"]

    cross_values = np.array(
        [x["cross"] for x in top],
        dtype=np.float32
    )

    strong_count_6 = int(
        np.sum(cross_values >= 6.0)
    )

    strong_count_7 = int(
        np.sum(cross_values >= 7.0)
    )

    strong_count_8 = int(
        np.sum(cross_values >= 8.0)
    )

    results.append({
        "answerable": answerable,
        "dense": top1_dense,
        "cross": top1_cross,
        "strong6": strong_count_6,
        "strong7": strong_count_7,
        "strong8": strong_count_8
    })

    if n % 10 == 0:
        print(f"Processed {n}/{NUM_QUERIES}")


def evaluate(condition):

    correct = 0
    false_answers = 0
    false_refusals = 0

    true_answers = 0
    true_refusals = 0

    for r in results:

        predicted = condition(r)

        if predicted and r["answerable"]:
            true_answers += 1

        elif predicted and not r["answerable"]:
            false_answers += 1

        elif not predicted and r["answerable"]:
            false_refusals += 1

        else:
            true_refusals += 1

        if predicted == r["answerable"]:
            correct += 1

    total = len(results)

    accuracy = correct / total
    answer_recall = (
        true_answers / 51
        if 51
        else 0
    )

    refusal_precision = (
        true_refusals / (true_refusals + false_answers)
        if (true_refusals + false_answers)
        else 0
    )

    return (
        accuracy,
        answer_recall,
        refusal_precision,
        false_answers,
        false_refusals
    )


print("\n" + "=" * 78)
print("        STRONG EVIDENCE COUNT GUARDRAIL EXPERIMENT")
print("=" * 78)

print("\nBASELINE")
print("Dense >= 0.70 AND Cross >= 7.00")
print("Accuracy:        0.790")
print("False answers:   11")
print("False refusals:  10")

print("\n" + "-" * 78)
print(
    "Dense | Cross | Strong>= | Accuracy | Answer recall | "
    "Refusal precision | False answers | False refusals"
)
print("-" * 78)


for dense_threshold in DENSE_THRESHOLDS:

    for cross_threshold in CROSS_THRESHOLDS:

        for strong_key, strong_threshold in [
            ("strong6", 6),
            ("strong7", 7),
            ("strong8", 8)
        ]:

            for minimum_count in STRONG_COUNTS:

                def rule(
                    r,
                    d=dense_threshold,
                    c=cross_threshold,
                    key=strong_key,
                    minimum=minimum_count
                ):
                    return (
                        r["dense"] >= d
                        and r["cross"] >= c
                        and r[key] >= minimum
                    )

                (
                    accuracy,
                    answer_recall,
                    refusal_precision,
                    false_answers,
                    false_refusals
                ) = evaluate(rule)

                print(
                    f"{dense_threshold:5.2f} | "
                    f"{cross_threshold:5.1f} | "
                    f"{strong_threshold:8d} | "
                    f"{accuracy:8.3f} | "
                    f"{answer_recall:13.3f} | "
                    f"{refusal_precision:17.3f} | "
                    f"{false_answers:13d} | "
                    f"{false_refusals:14d}"
                )


print("\n" + "=" * 78)
print("Evaluation complete.")
print("=" * 78)
