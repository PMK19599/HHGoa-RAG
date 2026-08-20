import json
import time
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIG
# ============================================================

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

NUM_QUERIES = 100
CANDIDATE_K = 20


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
dense_model = SentenceTransformer(DENSE_MODEL)

print("Loading cross-encoder...")
cross_model = CrossEncoder(CROSS_MODEL)


# ============================================================
# LOAD INDEX + METADATA
# ============================================================

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata...")
with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Index ready: {index.ntotal} passages")


# ============================================================
# LOAD DATASET
# ============================================================

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))


# ============================================================
# WARMUP
# ============================================================

print("Warming up models...")

dense_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["test query", "test passage"]]
)


# ============================================================
# BUILD CROSS-ENCODER DATASET
# ============================================================

scores = []

dense_latencies = []
cross_latencies = []
total_latencies = []

print(f"\nEvaluating {NUM_QUERIES} queries...")
print(f"Dense candidate K: {CANDIDATE_K}")


for count, record in enumerate(
    records[:NUM_QUERIES],
    start=1
):

    query = record["eng_query"]

    answerable = any(
        record["is_selected"]
    )

    total_start = time.perf_counter()


    # --------------------------------------------------------
    # 1. Dense retrieval
    # --------------------------------------------------------

    dense_start = time.perf_counter()

    query_embedding = dense_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    dense_scores, dense_indices = index.search(
        query_embedding,
        CANDIDATE_K
    )

    dense_elapsed = (
        time.perf_counter() - dense_start
    ) * 1000

    dense_latencies.append(
        dense_elapsed
    )


    # --------------------------------------------------------
    # 2. Cross-encoder
    # --------------------------------------------------------

    pairs = []

    for idx in dense_indices[0]:

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        passage = metadata[idx]["text"]

        pairs.append([
            query,
            passage
        ])


    cross_start = time.perf_counter()

    cross_scores = cross_model.predict(
        pairs
    )

    cross_elapsed = (
        time.perf_counter() - cross_start
    ) * 1000

    cross_latencies.append(
        cross_elapsed
    )


    # Highest cross score = strongest evidence
    best_cross = float(
        np.max(cross_scores)
    )

    best_cross_rank = int(
        np.argmax(cross_scores) + 1
    )


    # Best dense score from the same candidate set
    best_dense = float(
        dense_scores[0][0]
    )


    scores.append({
        "cross": best_cross,
        "dense": best_dense,
        "cross_rank": best_cross_rank,
        "answerable": answerable
    })


    total_elapsed = (
        time.perf_counter() - total_start
    ) * 1000

    total_latencies.append(
        total_elapsed
    )


    if count % 10 == 0:
        print(
            f"Processed {count}/{NUM_QUERIES}"
        )


# ============================================================
# DISTRIBUTIONS
# ============================================================

answerable_scores = [
    r["cross"]
    for r in scores
    if r["answerable"]
]

unanswerable_scores = [
    r["cross"]
    for r in scores
    if not r["answerable"]
]


print("\n")
print("=" * 60)
print("       CROSS-ENCODER GUARDRAIL EVALUATION")
print("=" * 60)


print("\nDATASET")

print(
    f"Queries evaluated: {len(scores)}"
)

print(
    f"Answerable:        "
    f"{sum(r['answerable'] for r in scores)}"
)

print(
    f"Unanswerable:      "
    f"{sum(not r['answerable'] for r in scores)}"
)


print("\nCROSS-ENCODER SCORE DISTRIBUTION")

print(
    f"Answerable mean:   "
    f"{np.mean(answerable_scores):.4f}"
)

print(
    f"Answerable median: "
    f"{np.median(answerable_scores):.4f}"
)

print(
    f"Answerable min:    "
    f"{np.min(answerable_scores):.4f}"
)

print(
    f"Unanswerable mean: "
    f"{np.mean(unanswerable_scores):.4f}"
)

print(
    f"Unanswerable median:"
    f" {np.median(unanswerable_scores):.4f}"
)

print(
    f"Unanswerable max:  "
    f"{np.max(unanswerable_scores):.4f}"
)


# ============================================================
# THRESHOLD TEST
# ============================================================

print("\nCROSS-ENCODER THRESHOLD TEST")

thresholds = [
    -10,
    -8,
    -6,
    -4,
    -2,
    0,
    2,
    4,
    5,
    6,
    7,
    8,
    9
]


for threshold in thresholds:

    correct = 0
    false_answers = 0
    false_refusals = 0

    for result in scores:

        predicted_answerable = (
            result["cross"] >= threshold
        )

        if (
            predicted_answerable
            == result["answerable"]
        ):
            correct += 1

        if (
            predicted_answerable
            and not result["answerable"]
        ):
            false_answers += 1

        if (
            not predicted_answerable
            and result["answerable"]
        ):
            false_refusals += 1


    accuracy = (
        correct / len(scores)
    )

    print(
        f"Threshold {threshold:>5} | "
        f"Accuracy {accuracy:.3f} | "
        f"False answers {false_answers:>2} | "
        f"False refusals {false_refusals:>2}"
    )


# ============================================================
# LATENCY
# ============================================================

dense_latencies = np.asarray(
    dense_latencies
)

cross_latencies = np.asarray(
    cross_latencies
)

total_latencies = np.asarray(
    total_latencies
)


print("\nLATENCY")

print(
    f"Dense P50:       "
    f"{np.percentile(dense_latencies, 50):.2f} ms"
)

print(
    f"Cross P50:       "
    f"{np.percentile(cross_latencies, 50):.2f} ms"
)

print(
    f"Cross P70:       "
    f"{np.percentile(cross_latencies, 70):.2f} ms"
)

print(
    f"Total P50:       "
    f"{np.percentile(total_latencies, 50):.2f} ms"
)

print(
    f"Total P70:       "
    f"{np.percentile(total_latencies, 70):.2f} ms"
)

print(
    f"Total P100:      "
    f"{np.max(total_latencies):.2f} ms"
)

print(
    f"Total Mean:      "
    f"{np.mean(total_latencies):.2f} ms"
)


print("\n")
print("=" * 60)
print("Evaluation complete.")
print("=" * 60)
