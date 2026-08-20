import json
import re
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

NUM_QUERIES = 100
DENSE_K = 20
FINAL_K = 5

DENSE_THRESHOLD = 0.70
CROSS_THRESHOLD = 7.0


# ============================================================
# LOAD
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

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Index ready: {index.ntotal} passages")

print("Warming up models...")
embed_model.encode(["warmup"], normalize_embeddings=True)
cross_model.predict([["warmup", "warmup passage"]])


# ============================================================
# TEXT SIGNALS
# ============================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "would",
    "should", "of", "to", "in", "on", "for", "with",
    "and", "or", "that", "this", "it", "be", "as",
    "from", "by", "about", "into", "than", "their",
    "they", "them", "its", "has", "have", "had"
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        word
        for word in words
        if word not in STOPWORDS and len(word) > 1
    ]


def lexical_features(query, passage):

    query_tokens = tokenize(query)
    passage_tokens = tokenize(passage)

    if not query_tokens:
        return {
            "query_terms": 0,
            "matched_terms": 0,
            "overlap_ratio": 0.0,
            "unique_overlap_ratio": 0.0,
            "phrase_overlap": 0.0,
        }

    query_set = set(query_tokens)
    passage_set = set(passage_tokens)

    matched = query_set.intersection(passage_set)

    overlap_ratio = len(matched) / max(len(query_set), 1)

    unique_overlap_ratio = len(matched) / max(
        len(passage_set),
        1
    )

    # Check whether any adjacent two-word phrase
    # from the query appears in the passage.
    phrase_overlap = 0.0

    if len(query_tokens) >= 2:
        query_phrases = [
            f"{query_tokens[i]} {query_tokens[i + 1]}"
            for i in range(len(query_tokens) - 1)
        ]

        passage_text = " ".join(passage_tokens)

        if any(
            phrase in passage_text
            for phrase in query_phrases
        ):
            phrase_overlap = 1.0

    return {
        "query_terms": len(query_set),
        "matched_terms": len(matched),
        "overlap_ratio": overlap_ratio,
        "unique_overlap_ratio": unique_overlap_ratio,
        "phrase_overlap": phrase_overlap,
    }


# ============================================================
# BUILD FEATURES
# ============================================================

print("\nBuilding independent evidence-quality features...")

X = []
y = []

diagnostics = []

for query_index, record in enumerate(records[:NUM_QUERIES]):

    query = record["eng_query"]

    answerable = bool(any(record["is_selected"]))

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(
        q_emb,
        dtype="float32"
    )

    dense_scores, indices = index.search(
        q_emb,
        DENSE_K
    )

    dense_scores = dense_scores[0]
    indices = indices[0]

    # --------------------------------------------------------
    # Cross-encoder reranking
    # --------------------------------------------------------

    candidates = []

    for dense_score, idx in zip(
        dense_scores,
        indices
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        passage = metadata[idx]["text"]

        candidates.append({
            "dense": float(dense_score),
            "text": passage,
            "idx": idx
        })

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    cross_scores = cross_model.predict(pairs)

    for candidate, cross_score in zip(
        candidates,
        cross_scores
    ):
        candidate["cross"] = float(cross_score)

    candidates.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    top = candidates[:FINAL_K]

    if not top:
        continue

    best = top[0]

    # --------------------------------------------------------
    # Retrieval statistics
    # --------------------------------------------------------

    dense_values = np.array(
        [x["dense"] for x in top],
        dtype=np.float32
    )

    cross_values = np.array(
        [x["cross"] for x in top],
        dtype=np.float32
    )

    top1_dense = float(dense_values[0])
    top2_dense = float(
        dense_values[1] if len(dense_values) > 1 else dense_values[0]
    )

    top1_cross = float(cross_values[0])
    top2_cross = float(
        cross_values[1] if len(cross_values) > 1 else cross_values[0]
    )

    # --------------------------------------------------------
    # Independent lexical evidence
    # --------------------------------------------------------

    lexical = lexical_features(
        query,
        best["text"]
    )

    # --------------------------------------------------------
    # Passage structure
    # --------------------------------------------------------

    query_length = len(tokenize(query))
    passage_length = len(tokenize(best["text"]))

    # --------------------------------------------------------
    # Feature vector
    #
    # IMPORTANT:
    # We intentionally keep the CrossEncoder score as
    # ONE feature rather than pretending it is verification.
    #
    # The experiment asks whether the combination of
    # independent evidence-quality signals improves on
    # the frozen production rule.
    # --------------------------------------------------------

    features = [
        # Dense retrieval
        top1_dense,
        top2_dense,
        top1_dense - top2_dense,

        # Cross-encoder relevance
        top1_cross,
        top2_cross,
        top1_cross - top2_cross,

        # Cross distribution
        float(np.mean(cross_values)),
        float(np.max(cross_values) - np.min(cross_values)),

        # Lexical evidence
        lexical["query_terms"],
        lexical["matched_terms"],
        lexical["overlap_ratio"],
        lexical["unique_overlap_ratio"],
        lexical["phrase_overlap"],

        # Passage structure
        query_length,
        passage_length,

        # Production rule
        float(
            top1_dense >= DENSE_THRESHOLD
            and top1_cross >= CROSS_THRESHOLD
        )
    ]

    X.append(features)
    y.append(1 if answerable else 0)

    diagnostics.append({
        "query": query,
        "answerable": answerable,
        "dense": top1_dense,
        "cross": top1_cross,
        "overlap": lexical["overlap_ratio"],
        "matched": lexical["matched_terms"],
        "production": (
            top1_dense >= DENSE_THRESHOLD
            and top1_cross >= CROSS_THRESHOLD
        ),
        "passage": best["text"]
    })

    if (query_index + 1) % 10 == 0:
        print(
            f"Processed {query_index + 1}/{NUM_QUERIES}"
        )


X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)


# ============================================================
# BASELINE
# ============================================================

baseline_pred = X[:, -1].astype(int)

baseline_accuracy = accuracy_score(
    y,
    baseline_pred
)

cm = confusion_matrix(
    y,
    baseline_pred
)

tn, fp, fn, tp = cm.ravel()

baseline_answer_recall = (
    tp / (tp + fn)
    if (tp + fn)
    else 0
)

baseline_refusal_precision = (
    tn / (tn + fn)
    if (tn + fn)
    else 0
)


print("\n" + "=" * 72)
print("              FROZEN PRODUCTION BASELINE")
print("=" * 72)

print(f"Accuracy:          {baseline_accuracy:.3f}")
print(f"Correct answers:   {tp}")
print(f"Correct refusals:  {tn}")
print(f"False answers:     {fp}")
print(f"False refusals:    {fn}")
print(f"Answer recall:     {baseline_answer_recall:.3f}")
print(f"Refusal precision: {baseline_refusal_precision:.3f}")


# ============================================================
# CLASSIFIER
# ============================================================

print("\n" + "=" * 72)
print("          INDEPENDENT EVIDENCE-QUALITY CLASSIFIER")
print("=" * 72)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced"
)

clf.fit(
    X_train,
    y_train
)

pred = clf.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

cm = confusion_matrix(
    y_test,
    pred
)

tn, fp, fn, tp = cm.ravel()

answer_recall = (
    tp / (tp + fn)
    if (tp + fn)
    else 0
)

refusal_precision = (
    tn / (tn + fn)
    if (tn + fn)
    else 0
)


print(f"Test queries:      {len(y_test)}")
print(f"Accuracy:          {accuracy:.3f}")
print(f"Correct answers:   {tp}")
print(f"Correct refusals:  {tn}")
print(f"False answers:     {fp}")
print(f"False refusals:    {fn}")
print(f"Answer recall:     {answer_recall:.3f}")
print(f"Refusal precision: {refusal_precision:.3f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = [
    "top1_dense",
    "top2_dense",
    "dense_gap",
    "top1_cross",
    "top2_cross",
    "cross_gap",
    "mean_cross_top5",
    "cross_range",

    "query_terms",
    "matched_terms",
    "lexical_overlap",
    "passage_overlap",
    "phrase_overlap",

    "query_length",
    "passage_length",

    "production_rule"
]

print("\nFEATURE IMPORTANCE")
print("-" * 72)

importance_pairs = sorted(
    zip(
        feature_names,
        clf.feature_importances_
    ),
    key=lambda x: x[1],
    reverse=True
)

for name, importance in importance_pairs:
    print(
        f"{name:<24} {importance:.4f}"
    )


# ============================================================
# FALSE ANSWER DIAGNOSTICS
# ============================================================

print("\n" + "=" * 72)
print("                 FALSE ANSWER DIAGNOSTICS")
print("=" * 72)

false_answers = [
    item
    for item in diagnostics
    if item["production"] and not item["answerable"]
]

print(
    f"\nProduction false answers: "
    f"{len(false_answers)}"
)

for i, item in enumerate(
    false_answers,
    1
):

    print("\n" + "-" * 72)
    print(f"FALSE ANSWER #{i}")
    print(f"Query:       {item['query']}")
    print(f"Dense:       {item['dense']:.4f}")
    print(f"Cross:       {item['cross']:.4f}")
    print(f"Lexical:     {item['overlap']:.4f}")
    print(f"Matched:     {item['matched']}")
    print(f"Passage:")
    print(item["passage"][:700])


# ============================================================
# FALSE REFUSAL DIAGNOSTICS
# ============================================================

print("\n" + "=" * 72)
print("                 FALSE REFUSAL DIAGNOSTICS")
print("=" * 72)

false_refusals = [
    item
    for item in diagnostics
    if not item["production"] and item["answerable"]
]

print(
    f"\nProduction false refusals: "
    f"{len(false_refusals)}"
)

for i, item in enumerate(
    false_refusals,
    1
):

    print("\n" + "-" * 72)
    print(f"FALSE REFUSAL #{i}")
    print(f"Query:       {item['query']}")
    print(f"Dense:       {item['dense']:.4f}")
    print(f"Cross:       {item['cross']:.4f}")
    print(f"Lexical:     {item['overlap']:.4f}")
    print(f"Matched:     {item['matched']}")
    print(f"Passage:")
    print(item["passage"][:700])


# ============================================================
# VERDICT
# ============================================================

print("\n" + "=" * 72)
print("                         VERDICT")
print("=" * 72)

print(
    "\nThe production pipeline remains FROZEN."
)

if accuracy > baseline_accuracy:
    print(
        f"Classifier accuracy improved over baseline "
        f"({accuracy:.3f} > {baseline_accuracy:.3f})."
    )
else:
    print(
        f"Classifier did NOT improve over baseline "
        f"({accuracy:.3f} <= {baseline_accuracy:.3f})."
    )

print(
    "\nDo NOT modify rag_pipeline.py based on this experiment alone."
)

print(
    "The next decision depends on whether the independent "
    "evidence-quality features provide a meaningful reduction "
    "in false answers without an unacceptable increase in "
    "false refusals."
)

print("\n" + "=" * 72)
print("                    Evaluation complete.")
print("=" * 72)