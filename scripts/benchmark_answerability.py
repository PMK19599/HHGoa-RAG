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

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["test query", "test passage"]]
)


def get_top5(query):

    query_embedding = embed_model.encode(
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

    candidates = []

    for dense_score, idx in zip(
        dense_scores[0],
        dense_indices[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append({
            "index": idx,
            "dense": float(dense_score),
            "text": metadata[idx]["text"]
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

    return candidates[:FINAL_K]


def lexical_answerability(query, evidence):

    """
    Diagnostic heuristic only.

    It estimates whether the retrieved evidence contains
    meaningful lexical overlap with important query terms.

    This is NOT being added to production.
    """

    query_terms = set(
        word.lower()
        for word in query.split()
        if len(word) >= 4
    )

    evidence_text = " ".join(
        item["text"].lower()
        for item in evidence
    )

    if not query_terms:
        return 0.0

    matches = sum(
        1
        for term in query_terms
        if term in evidence_text
    )

    return matches / len(query_terms)


signals = []


print("\nBuilding answerability signals...")

for count, record in enumerate(
    records[:NUM_QUERIES],
    start=1
):

    query = record["eng_query"]

    top5 = get_top5(query)

    best_cross = top5[0]["cross"]

    mean_cross = np.mean([
        item["cross"]
        for item in top5
    ])

    strong_count = sum(
        item["cross"] >= CROSS_THRESHOLD
        for item in top5
    )

    lexical_score = lexical_answerability(
        query,
        top5
    )

    answerable = bool(any(record["is_selected"]))

    production_answer = (
        top5[0]["dense"] >= DENSE_THRESHOLD
        and best_cross >= CROSS_THRESHOLD
    )

    signals.append({
        "query": query,
        "answerable": answerable,
        "production_answer": production_answer,
        "best_cross": best_cross,
        "mean_cross": mean_cross,
        "strong_count": strong_count,
        "lexical_score": lexical_score
    })

    if count % 10 == 0:
        print(f"Processed {count}/{NUM_QUERIES}")


print("\n" + "=" * 78)
print("              ANSWERABILITY DIAGNOSTIC")
print("=" * 78)


answerable = [
    x for x in signals
    if x["answerable"]
]

unanswerable = [
    x for x in signals
    if not x["answerable"]
]


def report(name, values):

    if not values:
        return

    print(f"\n{name}")

    for key in [
        "best_cross",
        "mean_cross",
        "strong_count",
        "lexical_score"
    ]:

        arr = np.asarray([
            item[key]
            for item in values
        ])

        print(
            f"{key:16s} "
            f"mean={np.mean(arr):.4f} "
            f"median={np.median(arr):.4f} "
            f"min={np.min(arr):.4f} "
            f"max={np.max(arr):.4f}"
        )


print(f"\nAnswerable:     {len(answerable)}")
print(f"Unanswerable:   {len(unanswerable)}")

report("ANSWERABLE", answerable)
report("UNANSWERABLE", unanswerable)


print("\n" + "=" * 78)
print("       SIMPLE LEXICAL ANSWERABILITY THRESHOLD TEST")
print("=" * 78)

print(
    "\nThreshold | Accuracy | False answers | False refusals"
)

for threshold in np.arange(
    0.10,
    1.01,
    0.05
):

    correct = 0
    false_answers = 0
    false_refusals = 0

    for item in signals:

        predicted = (
            item["production_answer"]
            and item["lexical_score"] >= threshold
        )

        actual = item["answerable"]

        if predicted == actual:
            correct += 1

        if predicted and not actual:
            false_answers += 1

        if not predicted and actual:
            false_refusals += 1

    print(
        f"{threshold:9.2f} | "
        f"{correct / NUM_QUERIES:8.3f} | "
        f"{false_answers:13d} | "
        f"{false_refusals:14d}"
    )


print("\n" + "=" * 78)
print("CURRENT PRODUCTION BASELINE")
print("=" * 78)

correct = sum(
    item["production_answer"] == item["answerable"]
    for item in signals
)

false_answers = sum(
    item["production_answer"]
    and not item["answerable"]
    for item in signals
)

false_refusals = sum(
    not item["production_answer"]
    and item["answerable"]
    for item in signals
)

print(f"Accuracy:        {correct / NUM_QUERIES:.3f}")
print(f"False answers:   {false_answers}")
print(f"False refusals:  {false_refusals}")


print("\n" + "=" * 78)
print("Diagnostic only. Production code was NOT changed.")
print("=" * 78)
