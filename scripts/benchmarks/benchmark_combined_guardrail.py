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


print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["test", "test"]]
)


results = []

print("\nBuilding production-matching combined signals...")


for count, record in enumerate(
    records[:NUM_QUERIES],
    start=1
):

    query = record["eng_query"]

    # --------------------------------------------------------
    # Dense Top-20
    # --------------------------------------------------------

    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(
        q_emb,
        dtype="float32"
    )

    dense_scores, dense_indices = index.search(
        q_emb,
        CANDIDATE_K
    )

    dense_scores = dense_scores[0]
    dense_indices = dense_indices[0]


    # --------------------------------------------------------
    # Dense signals
    # --------------------------------------------------------

    top1 = float(dense_scores[0])
    top2 = float(dense_scores[1])
    gap12 = top1 - top2

    top5 = float(dense_scores[4])
    gap15 = top1 - top5


    # --------------------------------------------------------
    # Cross-encoder over all Top-20
    # --------------------------------------------------------

    pairs = []

    valid_dense_scores = []

    for dense_score, idx in zip(
        dense_scores,
        dense_indices
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        pairs.append([
            query,
            metadata[idx]["text"]
        ])

        valid_dense_scores.append(
            float(dense_score)
        )


    cross_scores = cross_model.predict(
        pairs
    )

    best_cross_index = int(
        np.argmax(cross_scores)
    )

    best_cross = float(
        cross_scores[best_cross_index]
    )

    best_dense_for_cross = float(
        valid_dense_scores[best_cross_index]
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    answerable = any(
        record["is_selected"]
    )


    results.append({
        "top1": top1,
        "top2": top2,
        "gap12": gap12,
        "top5": top5,
        "gap15": gap15,
        "cross": best_cross,
        "cross_dense": best_dense_for_cross,
        "answerable": answerable
    })


    if count % 10 == 0:
        print(
            f"Processed {count}/{NUM_QUERIES}"
        )


# ============================================================
# TEST FUNCTION
# ============================================================

def evaluate(
    cross_threshold,
    dense_threshold,
    gap_threshold=None
):

    correct = 0
    false_answers = 0
    false_refusals = 0

    for r in results:

        predicted = (
            r["cross"] >= cross_threshold
            and r["cross_dense"] >= dense_threshold
        )

        if gap_threshold is not None:

            predicted = (
                predicted
                and r["gap12"] >= gap_threshold
            )

        if predicted == r["answerable"]:
            correct += 1

        if predicted and not r["answerable"]:
            false_answers += 1

        if not predicted and r["answerable"]:
            false_refusals += 1


    return (
        correct / len(results),
        false_answers,
        false_refusals
    )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("       PRODUCTION-MATCHING COMBINED GUARDRAIL")
print("=" * 70)


print("\nDATASET")

print(
    f"Queries evaluated: {len(results)}"
)

print(
    f"Answerable:        "
    f"{sum(r['answerable'] for r in results)}"
)

print(
    f"Unanswerable:      "
    f"{sum(not r['answerable'] for r in results)}"
)


# ============================================================
# CROSS + DENSE
# ============================================================

print("\n")
print("=== CROSS + DENSE ===")

print(
    "Cross | Dense | Accuracy | "
    "False answers | False refusals"
)

for cross_threshold in [6, 7, 8]:

    for dense_threshold in [
        0.70,
        0.75,
        0.80,
        0.85
    ]:

        accuracy, false_answers, false_refusals = evaluate(
            cross_threshold,
            dense_threshold
        )

        print(
            f"{cross_threshold:>5} | "
            f"{dense_threshold:.2f} | "
            f"{accuracy:.3f} | "
            f"{false_answers:>13} | "
            f"{false_refusals:>14}"
        )


# ============================================================
# CROSS + DENSE + GAP
# ============================================================

print("\n")
print("=== CROSS + DENSE + GAP ===")

print(
    "Cross | Dense | Gap | Accuracy | "
    "False answers | False refusals"
)

for cross_threshold in [6, 7, 8]:

    for dense_threshold in [
        0.70,
        0.75,
        0.80
    ]:

        for gap_threshold in [
            0.00,
            0.01,
            0.02,
            0.03
        ]:

            accuracy, false_answers, false_refusals = evaluate(
                cross_threshold,
                dense_threshold,
                gap_threshold
            )

            print(
                f"{cross_threshold:>5} | "
                f"{dense_threshold:.2f} | "
                f"{gap_threshold:.2f} | "
                f"{accuracy:.3f} | "
                f"{false_answers:>13} | "
                f"{false_refusals:>14}"
            )


print("\n")
print("=" * 70)
print("Evaluation complete.")
print("=" * 70)
