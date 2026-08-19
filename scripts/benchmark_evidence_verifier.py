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

VERIFY_THRESHOLDS = [
    -2.0,
    0.0,
    2.0,
    4.0,
    6.0,
    7.0,
    8.0,
]


# ============================================================
# CONSTANT REFUSAL
# ============================================================

REFUSAL_TEXT = (
    "I don't have enough reliable evidence in the retrieved "
    "passages to answer this question."
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
# RETRIEVE + RERANK
# ============================================================

def retrieve_and_rerank(query):

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
            "cross_score": None,
            "text": metadata[idx]["text"]
        })

    if not candidates:
        return []

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

    return candidates[:FINAL_TOP_K]


# ============================================================
# VERIFICATION SCORE
# ============================================================

def calculate_verification_score(query, passage):

    score = cross_model.predict(
        [[query, passage]]
    )[0]

    return float(score)


# ============================================================
# BUILD SIGNAL DATA
# ============================================================

print()
print("=" * 72)
print("              EVIDENCE VERIFIER BENCHMARK")
print("=" * 72)

print()
print("Production baseline:")
print(
    f"Dense >= {DENSE_THRESHOLD:.2f} AND "
    f"Cross >= {CROSS_THRESHOLD:.2f}"
)

print()
print(f"Queries evaluated: {len(records)}")
print()
print("Building retrieval + verification signals...")


results = []


for count, record in enumerate(records, start=1):

    query = record["eng_query"]

    expected_answerable = any(
        record["is_selected"]
    )

    candidates = retrieve_and_rerank(query)

    if not candidates:
        continue

    best = candidates[0]

    verification_score = calculate_verification_score(
        query,
        best["text"]
    )

    results.append({
        "query": query,
        "query_id": record["query_id"],
        "expected_answerable": expected_answerable,
        "dense": best["dense_score"],
        "cross": best["cross_score"],
        "verification": verification_score,
        "text": best["text"]
    })

    if count % 10 == 0:
        print(
            f"Processed {count}/{len(records)}"
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    results,
    dense_threshold,
    cross_threshold,
    verification_threshold=None
):

    correct_answers = 0
    correct_refusals = 0
    false_answers = 0
    false_refusals = 0

    for result in results:

        predicted = (
            result["dense"] >= dense_threshold
            and result["cross"] >= cross_threshold
        )

        if verification_threshold is not None:
            predicted = (
                predicted
                and result["verification"]
                >= verification_threshold
            )

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
        (correct_refusals + false_answers)
        if (correct_refusals + false_answers)
        else 0
    )

    return {
        "correct_answers": correct_answers,
        "correct_refusals": correct_refusals,
        "false_answers": false_answers,
        "false_refusals": false_refusals,
        "accuracy": accuracy,
        "answer_recall": answer_recall,
        "refusal_precision": refusal_precision
    }


# ============================================================
# BASELINE
# ============================================================

baseline = calculate_metrics(
    results,
    DENSE_THRESHOLD,
    CROSS_THRESHOLD
)

print()
print("=" * 72)
print("                     BASELINE")
print("=" * 72)

print()
print(
    f"Correct answers:       "
    f"{baseline['correct_answers']}"
)

print(
    f"Correct refusals:      "
    f"{baseline['correct_refusals']}"
)

print(
    f"False answers:         "
    f"{baseline['false_answers']}"
)

print(
    f"False refusals:        "
    f"{baseline['false_refusals']}"
)

print()
print(
    f"Accuracy:              "
    f"{baseline['accuracy']:.3f}"
)

print(
    f"Answer recall:         "
    f"{baseline['answer_recall']:.3f}"
)

print(
    f"Refusal precision:     "
    f"{baseline['refusal_precision']:.3f}"
)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

answerable_scores = [
    r["verification"]
    for r in results
    if r["expected_answerable"]
]

unanswerable_scores = [
    r["verification"]
    for r in results
    if not r["expected_answerable"]
]


print()
print("=" * 72)
print("             VERIFICATION SCORE DISTRIBUTION")
print("=" * 72)

print()

if answerable_scores:

    print(
        f"Answerable mean:       "
        f"{np.mean(answerable_scores):.4f}"
    )

    print(
        f"Answerable median:     "
        f"{np.median(answerable_scores):.4f}"
    )

    print(
        f"Answerable min:        "
        f"{np.min(answerable_scores):.4f}"
    )

print()

if unanswerable_scores:

    print(
        f"Unanswerable mean:     "
        f"{np.mean(unanswerable_scores):.4f}"
    )

    print(
        f"Unanswerable median:   "
        f"{np.median(unanswerable_scores):.4f}"
    )

    print(
        f"Unanswerable max:      "
        f"{np.max(unanswerable_scores):.4f}"
    )


# ============================================================
# VERIFICATION THRESHOLD TEST
# ============================================================

print()
print("=" * 72)
print("              VERIFICATION THRESHOLD TEST")
print("=" * 72)

print()
print(
    "Verify | Accuracy | FalseAns | FalseRef | Recall | RefPrec"
)

print("-" * 68)

verification_results = []

for threshold in VERIFY_THRESHOLDS:

    metrics = calculate_metrics(
        results,
        DENSE_THRESHOLD,
        CROSS_THRESHOLD,
        threshold
    )

    verification_results.append(
        (threshold, metrics)
    )

    print(
        f"{threshold:6.1f} | "
        f"{metrics['accuracy']:8.3f} | "
        f"{metrics['false_answers']:9d} | "
        f"{metrics['false_refusals']:9d} | "
        f"{metrics['answer_recall']:6.3f} | "
        f"{metrics['refusal_precision']:7.3f}"
    )


# ============================================================
# KNOWN FALSE ANSWERS
# ============================================================

baseline_false_answers = [
    r
    for r in results
    if (
        r["dense"] >= DENSE_THRESHOLD
        and r["cross"] >= CROSS_THRESHOLD
        and not r["expected_answerable"]
    )
]


print()
print("=" * 72)
print("             KNOWN FALSE-ANSWER ANALYSIS")
print("=" * 72)

print()
print(
    f"Baseline false answers: "
    f"{len(baseline_false_answers)}"
)


for i, result in enumerate(
    baseline_false_answers,
    start=1
):

    print()
    print("-" * 72)

    print(f"FALSE ANSWER #{i}")

    print()
    print(f"Query: {result['query']}")

    print()
    print(
        f"Dense:        {result['dense']:.4f}"
    )

    print(
        f"Cross:        {result['cross']:.4f}"
    )

    print(
        f"Verification: {result['verification']:.4f}"
    )

    print()
    print("Passage:")
    print(result["text"][:650])


# ============================================================
# KNOWN FALSE REFUSALS
# ============================================================

baseline_false_refusals = [
    r
    for r in results
    if (
        not (
            r["dense"] >= DENSE_THRESHOLD
            and r["cross"] >= CROSS_THRESHOLD
        )
        and r["expected_answerable"]
    )
]


print()
print("=" * 72)
print("             KNOWN FALSE-REFUSAL ANALYSIS")
print("=" * 72)

print()
print(
    f"Baseline false refusals: "
    f"{len(baseline_false_refusals)}"
)


for i, result in enumerate(
    baseline_false_refusals,
    start=1
):

    print()
    print("-" * 72)

    print(f"FALSE REFUSAL #{i}")

    print()
    print(f"Query: {result['query']}")

    print()
    print(
        f"Dense:        {result['dense']:.4f}"
    )

    print(
        f"Cross:        {result['cross']:.4f}"
    )

    print(
        f"Verification: {result['verification']:.4f}"
    )

    print()
    print("Passage:")
    print(result["text"][:650])


# ============================================================
# ADVERSARIAL QUERIES
# ============================================================

adversarial_queries = [

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

    (
        "What is photosynthesis?",
        "unsupported_topic"
    ),

    (
        "What is cellular respiration?",
        "unsupported_topic"
    ),

    (
        "What is quantum photosynthesis?",
        "unsupported_topic"
    ),

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
print("                 ADVERSARIAL VERIFICATION")
print("=" * 72)


adversarial_results = []


for query, category in adversarial_queries:

    candidates = retrieve_and_rerank(query)

    if not candidates:
        continue

    best = candidates[0]

    dense_score = best["dense_score"]
    cross_score = best["cross_score"]

    verification = calculate_verification_score(
        query,
        best["text"]
    )

    baseline_decision = (
        dense_score >= DENSE_THRESHOLD
        and cross_score >= CROSS_THRESHOLD
    )

    adversarial_results.append({
        "query": query,
        "category": category,
        "dense": dense_score,
        "cross": cross_score,
        "verification": verification,
        "baseline_decision": baseline_decision,
        "text": best["text"]
    })

    print()
    print("-" * 72)

    print(
        f"Category: {category}"
    )

    print(
        f"Query: {query}"
    )

    print()
    print(
        f"Dense:        {dense_score:.4f}"
    )

    print(
        f"Cross:        {cross_score:.4f}"
    )

    print(
        f"Verification: {verification:.4f}"
    )

    print()
    print(
        "Baseline decision: "
        + (
            "ANSWER"
            if baseline_decision
            else "REFUSE"
        )
    )

    print()
    print("Retrieved passage:")
    print(best["text"][:500])


# ============================================================
# ADVERSARIAL SUMMARY
# ============================================================

print()
print("=" * 72)
print("                 ADVERSARIAL SUMMARY")
print("=" * 72)

print()

baseline_adversarial_answers = sum(
    1
    for r in adversarial_results
    if r["baseline_decision"]
)

print(
    f"Adversarial queries:           "
    f"{len(adversarial_results)}"
)

print(
    f"Baseline ANSWER decisions:     "
    f"{baseline_adversarial_answers}"
)

print(
    f"Baseline REFUSE decisions:     "
    f"{len(adversarial_results) - baseline_adversarial_answers}"
)


for threshold in VERIFY_THRESHOLDS:

    verifier_answers = sum(
        1
        for r in adversarial_results
        if (
            r["baseline_decision"]
            and r["verification"] >= threshold
        )
    )

    print(
        f"Verifier >= {threshold:4.1f}: "
        f"{verifier_answers} adversarial ANSWER decisions"
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

print()
print("=" * 72)
print("                    DIAGNOSTIC")
print("=" * 72)

print()

best_candidate = None

for threshold, metrics in verification_results:

    score = (
        metrics["accuracy"]
        - 0.50 * (
            metrics["false_answers"] /
            len(results)
        )
    )

    if (
        best_candidate is None
        or score > best_candidate["score"]
    ):

        best_candidate = {
            "threshold": threshold,
            "metrics": metrics,
            "score": score
        }


if best_candidate:

    threshold = best_candidate["threshold"]
    metrics = best_candidate["metrics"]

    print(
        f"Most interesting verification threshold "
        f"from this sweep: {threshold:.1f}"
    )

    print()
    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.3f}"
    )

    print(
        f"False answers:     "
        f"{metrics['false_answers']}"
    )

    print(
        f"False refusals:    "
        f"{metrics['false_refusals']}"
    )

    print(
        f"Answer recall:     "
        f"{metrics['answer_recall']:.3f}"
    )

    print(
        f"Refusal precision: "
        f"{metrics['refusal_precision']:.3f}"
    )


print()
print("IMPORTANT:")

print(
    "This experiment does NOT prove that the second "
    "cross-encoder score is true logical entailment."
)

print(
    "The second pass is identical to the cross-encoder "
    "scoring model, so identical query/passage pairs "
    "produce identical scores."
)

print(
    "Therefore, if the verification score exactly matches "
    "the production cross score, this experiment provides "
    "no genuinely new signal."
)

print()
print(
    "DO NOT modify rag_pipeline.py yet."
)

print()
print("=" * 72)
print("                 Evaluation complete.")
print("=" * 72)