import json
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

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

NUM_QUERIES = 100
CANDIDATE_K = 20
FINAL_TOP_K = 5

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0


REFUSAL_TEXT = (
    "I don't have enough reliable evidence in the retrieved passages "
    "to answer this question."
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross-encoder...")
cross_model = CrossEncoder(CROSS_MODEL)

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


records = records[:NUM_QUERIES]


# ============================================================
# WARMUP
# ============================================================

print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["warmup query", "warmup passage"]]
)


# ============================================================
# RETRIEVAL + RERANKING
# ============================================================

def evaluate_query(query):

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

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
            "dense_score": float(dense_score),
            "text": metadata[idx]["text"]
        })


    if not candidates:
        return None


    # --------------------------------------------------------
    # Cross-encoder reranking
    # --------------------------------------------------------

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    cross_scores = cross_model.predict(pairs)

    for candidate, cross_score in zip(
        candidates,
        cross_scores
    ):
        candidate["cross_score"] = float(cross_score)


    candidates.sort(
        key=lambda x: x["cross_score"],
        reverse=True
    )


    final_candidates = candidates[:FINAL_TOP_K]

    best = final_candidates[0]

    # --------------------------------------------------------
    # Production guardrail
    # --------------------------------------------------------

    answerable = (
        best["dense_score"] >= DENSE_THRESHOLD
        and best["cross_score"] >= CROSS_THRESHOLD
    )


    return {
        "best": best,
        "results": final_candidates,
        "answerable": answerable
    }


# ============================================================
# BASELINE DATASET EVALUATION
# ============================================================

print()
print("=" * 72)
print("              ADVERSARIAL GUARDRAIL BENCHMARK")
print("=" * 72)

print()
print("Production guardrail:")
print(
    f"Dense >= {DENSE_THRESHOLD:.2f} AND "
    f"Cross >= {CROSS_THRESHOLD:.2f}"
)

print()
print(f"Queries evaluated: {len(records)}")


results = []


for count, record in enumerate(records, start=1):

    query = record["eng_query"]

    expected_answerable = any(
        record["is_selected"]
    )

    result = evaluate_query(query)

    if result is None:
        continue

    result["query"] = query
    result["query_id"] = record["query_id"]
    result["expected_answerable"] = expected_answerable

    results.append(result)

    if count % 10 == 0:
        print(
            f"Processed {count}/{len(records)}"
        )


# ============================================================
# BASELINE METRICS
# ============================================================

correct_answers = 0
correct_refusals = 0
false_answers = 0
false_refusals = 0


for result in results:

    predicted = result["answerable"]
    expected = result["expected_answerable"]

    if predicted and expected:
        correct_answers += 1

    elif not predicted and not expected:
        correct_refusals += 1

    elif predicted and not expected:
        false_answers += 1

    elif not predicted and expected:
        false_refusals += 1


total = len(results)

accuracy = (
    (correct_answers + correct_refusals) / total
    if total
    else 0
)

answer_recall = (
    correct_answers /
    (correct_answers + false_refusals)
    if (correct_answers + false_refusals)
    else 0
)

refusal_precision = (
    correct_refusals /
    (correct_refusals + false_refusals)
    if (correct_refusals + false_refusals)
    else 0
)


print()
print("=" * 72)
print("                    BASELINE RESULTS")
print("=" * 72)

print()
print(f"Total evaluated:       {total}")
print(f"Correct answers:       {correct_answers}")
print(f"Correct refusals:      {correct_refusals}")
print(f"False answers:         {false_answers}")
print(f"False refusals:        {false_refusals}")

print()
print(f"Accuracy:              {accuracy:.3f}")
print(f"Answer recall:         {answer_recall:.3f}")
print(f"Refusal precision:     {refusal_precision:.3f}")


# ============================================================
# FALSE ANSWER ANALYSIS
# ============================================================

false_answer_results = [
    r
    for r in results
    if r["answerable"]
    and not r["expected_answerable"]
]


print()
print("=" * 72)
print("                  FALSE ANSWER ANALYSIS")
print("=" * 72)

print()
print(
    f"False answers found: {len(false_answer_results)}"
)


for i, result in enumerate(
    false_answer_results,
    start=1
):

    best = result["best"]

    print()
    print("-" * 72)

    print(f"FALSE ANSWER #{i}")

    print()
    print(f"Query:")
    print(result["query"])

    print()
    print(
        f"Dense score:  {best['dense_score']:.4f}"
    )

    print(
        f"Cross score:  {best['cross_score']:.4f}"
    )

    print()
    print("Selected passage:")
    print(best["text"][:700])


# ============================================================
# FALSE REFUSAL ANALYSIS
# ============================================================

false_refusal_results = [
    r
    for r in results
    if not r["answerable"]
    and r["expected_answerable"]
]


print()
print("=" * 72)
print("                 FALSE REFUSAL ANALYSIS")
print("=" * 72)

print()
print(
    f"False refusals found: {len(false_refusal_results)}"
)


for i, result in enumerate(
    false_refusal_results,
    start=1
):

    best = result["best"]

    print()
    print("-" * 72)

    print(f"FALSE REFUSAL #{i}")

    print()
    print("Query:")
    print(result["query"])

    print()
    print(
        f"Dense score:  {best['dense_score']:.4f}"
    )

    print(
        f"Cross score:  {best['cross_score']:.4f}"
    )

    print()
    print("Top passage:")
    print(best["text"][:700])


# ============================================================
# ADVERSARIAL QUERY SET
# ============================================================

adversarial_queries = [

    # --------------------------------------------------------
    # Entity substitution
    # --------------------------------------------------------

    (
        "Who is the president of Mars?",
        "fabricated_entity"
    ),

    (
        "Who is the president of the Moon?",
        "fabricated_entity"
    ),

    (
        "Who is the CEO of the planet Mars?",
        "fabricated_entity"
    ),

    # --------------------------------------------------------
    # Concept substitution
    # --------------------------------------------------------

    (
        "What is photosynthesis?",
        "near_miss"
    ),

    (
        "What is cellular respiration?",
        "near_miss"
    ),

    (
        "What is quantum photosynthesis?",
        "near_miss"
    ),

    # --------------------------------------------------------
    # Unsupported factual claims
    # --------------------------------------------------------

    (
        "What evidence proves that humans live on Mars?",
        "unsupported_claim"
    ),

    (
        "What is the capital city of Mars?",
        "unsupported_claim"
    ),

    (
        "What government currently rules Mars?",
        "unsupported_claim"
    ),

    # --------------------------------------------------------
    # Topic blending
    # --------------------------------------------------------

    (
        "What is the corporation that governs Mars?",
        "topic_blend"
    ),

    (
        "What company owns the Moon?",
        "topic_blend"
    ),

    (
        "What corporation controls photosynthesis?",
        "topic_blend"
    ),
]


print()
print("=" * 72)
print("                 ADVERSARIAL QUERY TEST")
print("=" * 72)


adversarial_results = []


for query, category in adversarial_queries:

    result = evaluate_query(query)

    if result is None:
        continue

    best = result["best"]

    predicted_answer = (
        best["dense_score"] >= DENSE_THRESHOLD
        and best["cross_score"] >= CROSS_THRESHOLD
    )

    adversarial_results.append({
        "query": query,
        "category": category,
        "predicted_answer": predicted_answer,
        "dense": best["dense_score"],
        "cross": best["cross_score"],
        "text": best["text"]
    })


    print()
    print("-" * 72)

    print(f"Category: {category}")
    print(f"Query: {query}")

    print()
    print(
        f"Dense: {best['dense_score']:.4f}"
    )

    print(
        f"Cross: {best['cross_score']:.4f}"
    )

    print()
    print(
        "Decision: "
        + (
            "ANSWER"
            if predicted_answer
            else "REFUSE"
        )
    )

    print()
    print("Top passage:")
    print(best["text"][:450])


# ============================================================
# ADVERSARIAL SUMMARY
# ============================================================

adversarial_answers = sum(
    1
    for r in adversarial_results
    if r["predicted_answer"]
)

adversarial_refusals = (
    len(adversarial_results)
    - adversarial_answers
)


print()
print("=" * 72)
print("                ADVERSARIAL SUMMARY")
print("=" * 72)

print()
print(
    f"Adversarial queries:    "
    f"{len(adversarial_results)}"
)

print(
    f"Answered:               "
    f"{adversarial_answers}"
)

print(
    f"Refused:                "
    f"{adversarial_refusals}"
)

if adversarial_results:

    print()
    print(
        "Adversarial refusal rate: "
        f"{adversarial_refusals / len(adversarial_results):.3f}"
    )


# ============================================================
# THRESHOLD SENSITIVITY
# ============================================================

print()
print("=" * 72)
print("                 THRESHOLD SENSITIVITY")
print("=" * 72)

print()
print(
    "Testing nearby production boundaries."
)

print()

print(
    "Dense | Cross | Accuracy | FalseAns | FalseRef"
)

print("-" * 55)


for dense_threshold in [
    0.65,
    0.70,
    0.75,
    0.80,
]:

    for cross_threshold in [
        5.0,
        6.0,
        7.0,
        8.0,
    ]:

        correct = 0
        fa = 0
        fr = 0

        for result in results:

            best = result["best"]

            predicted = (
                best["dense_score"] >= dense_threshold
                and best["cross_score"] >= cross_threshold
            )

            expected = result["expected_answerable"]

            if predicted == expected:
                correct += 1

            if predicted and not expected:
                fa += 1

            if not predicted and expected:
                fr += 1

        acc = correct / total if total else 0

        print(
            f"{dense_threshold:5.2f} | "
            f"{cross_threshold:5.1f} | "
            f"{acc:8.3f} | "
            f"{fa:8d} | "
            f"{fr:8d}"
        )


# ============================================================
# FINAL DIAGNOSTIC
# ============================================================

print()
print("=" * 72)
print("                     DIAGNOSTIC")
print("=" * 72)

print()

if false_answers > false_refusals:

    print(
        "Primary weakness: FALSE ANSWERS."
    )

    print(
        "The guardrail is allowing unsupported "
        "retrieval results through."
    )

elif false_refusals > false_answers:

    print(
        "Primary weakness: FALSE REFUSALS."
    )

    print(
        "The guardrail is rejecting some "
        "queries that have supporting evidence."
    )

else:

    print(
        "False answers and false refusals are balanced."
    )


print()

print(
    "IMPORTANT:"
)

print(
    "Do NOT change the production threshold yet."
)

print(
    "Use the false-answer and false-refusal cases "
    "above to decide the next architectural change."
)

print()
print("=" * 72)
print("                  Evaluation complete.")
print("=" * 72)