import json
import numpy as np
import faiss
import re

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
DENSE_K = 20
FINAL_K = 5


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
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_tokens(text):
    words = normalize(text).split()

    stopwords = {
        "a", "an", "the", "is", "are", "was", "were",
        "what", "who", "when", "where", "why", "how",
        "does", "do", "did", "can", "could", "would",
        "should", "of", "to", "in", "on", "for", "with",
        "and", "or", "that", "this", "it", "be", "as",
        "from", "by", "about", "into", "than", "their",
        "they", "them", "its", "has", "have", "had"
    }

    return [
        w for w in words
        if w not in stopwords and len(w) > 2
    ]


# ============================================================
# SIMPLE CONTRADICTION / AGREEMENT HEURISTICS
# ============================================================

NEGATION_PAIRS = [
    ("can", "cannot"),
    ("can", "can't"),
    ("is", "isn't"),
    ("are", "aren't"),
    ("does", "doesn't"),
    ("do", "don't"),
    ("will", "will not"),
    ("will", "won't"),
    ("yes", "no"),
    ("true", "false"),
    ("possible", "impossible"),
    ("allows", "does not allow"),
    ("allowed", "not allowed"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("cause", "does not cause"),
    ("causes", "does not cause"),
    ("effective", "ineffective"),
    ("safe", "unsafe"),
]


def lexical_similarity(a, b):
    a_set = set(content_tokens(a))
    b_set = set(content_tokens(b))

    if not a_set or not b_set:
        return 0.0

    intersection = len(a_set & b_set)
    union = len(a_set | b_set)

    return intersection / max(union, 1)


def contradiction_signal(a, b):
    a_norm = normalize(a)
    b_norm = normalize(b)

    score = 0.0

    for positive, negative in NEGATION_PAIRS:

        if positive in a_norm and negative in b_norm:
            score += 1.0

        if negative in a_norm and positive in b_norm:
            score += 1.0

    return score


# ============================================================
# BUILD TOP-5 EVIDENCE
# ============================================================

print("\nBuilding evidence agreement signals...")

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
            "text": metadata[idx]["text"],
            "idx": idx
        })

    # --------------------------------------------------------
    # Cross encoder
    # --------------------------------------------------------

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

    if not top:
        continue

    # --------------------------------------------------------
    # Pairwise evidence agreement
    # --------------------------------------------------------

    similarities = []
    contradictions = []

    for i in range(len(top)):
        for j in range(i + 1, len(top)):

            sim = lexical_similarity(
                top[i]["text"],
                top[j]["text"]
            )

            contra = contradiction_signal(
                top[i]["text"],
                top[j]["text"]
            )

            similarities.append(sim)
            contradictions.append(contra)

    mean_similarity = (
        float(np.mean(similarities))
        if similarities else 0.0
    )

    max_similarity = (
        float(np.max(similarities))
        if similarities else 0.0
    )

    contradiction_pairs = sum(
        1 for x in contradictions
        if x > 0
    )

    max_contradiction = (
        float(max(contradictions))
        if contradictions else 0.0
    )

    # --------------------------------------------------------
    # Cross score distribution
    # --------------------------------------------------------

    cross_values = np.array(
        [x["cross"] for x in top],
        dtype=np.float32
    )

    strong_count = int(
        np.sum(cross_values >= 7.0)
    )

    diagnostics.append({
        "query": query,
        "answerable": answerable,
        "top1_cross": float(cross_values[0]),
        "mean_cross": float(np.mean(cross_values)),
        "strong_count": strong_count,
        "mean_similarity": mean_similarity,
        "max_similarity": max_similarity,
        "contradiction_pairs": contradiction_pairs,
        "max_contradiction": max_contradiction,
        "passages": top
    })

    if (query_index + 1) % 10 == 0:
        print(f"Processed {query_index + 1}/{NUM_QUERIES}")


# ============================================================
# DISTRIBUTIONS
# ============================================================

answerable = [
    x for x in diagnostics
    if x["answerable"]
]

unanswerable = [
    x for x in diagnostics
    if not x["answerable"]
]


def mean(items, key):
    return float(
        np.mean([x[key] for x in items])
    ) if items else 0.0


def median(items, key):
    return float(
        np.median([x[key] for x in items])
    ) if items else 0.0


print("\n" + "=" * 72)
print("              EVIDENCE AGREEMENT DIAGNOSTIC")
print("=" * 72)

print(f"\nQueries evaluated: {len(diagnostics)}")
print(f"Answerable:        {len(answerable)}")
print(f"Unanswerable:      {len(unanswerable)}")

signals = [
    "mean_similarity",
    "max_similarity",
    "contradiction_pairs",
    "max_contradiction",
    "strong_count",
    "mean_cross",
]


for signal in signals:

    print(f"\n{signal}")

    print(
        f"  Answerable   "
        f"mean={mean(answerable, signal):.4f} "
        f"median={median(answerable, signal):.4f}"
    )

    print(
        f"  Unanswerable "
        f"mean={mean(unanswerable, signal):.4f} "
        f"median={median(unanswerable, signal):.4f}"
    )


# ============================================================
# CURRENT FALSE ANSWERS
# ============================================================

false_answers = [
    x for x in diagnostics
    if not x["answerable"]
    and x["top1_cross"] >= 7.0
]


print("\n" + "=" * 72)
print("          CURRENT FALSE ANSWERS � AGREEMENT SIGNALS")
print("=" * 72)

for i, item in enumerate(false_answers, 1):

    print("\n" + "-" * 72)
    print(f"CASE #{i}")
    print(f"Query: {item['query']}")
    print(f"Top1 cross: {item['top1_cross']:.4f}")
    print(f"Mean cross: {item['mean_cross']:.4f}")
    print(f"Strong passages: {item['strong_count']}")
    print(
        f"Mean evidence similarity: "
        f"{item['mean_similarity']:.4f}"
    )
    print(
        f"Max evidence similarity: "
        f"{item['max_similarity']:.4f}"
    )
    print(
        f"Contradiction pairs: "
        f"{item['contradiction_pairs']}"
    )
    print(
        f"Max contradiction signal: "
        f"{item['max_contradiction']:.4f}"
    )


# ============================================================
# FALSE REFUSALS
# ============================================================

false_refusals = [
    x for x in diagnostics
    if x["answerable"]
    and not (
        x["top1_cross"] >= 7.0
    )
]


print("\n" + "=" * 72)
print("          CURRENT FALSE REFUSALS � AGREEMENT SIGNALS")
print("=" * 72)

for i, item in enumerate(false_refusals, 1):

    print("\n" + "-" * 72)
    print(f"CASE #{i}")
    print(f"Query: {item['query']}")
    print(f"Top1 cross: {item['top1_cross']:.4f}")
    print(f"Mean cross: {item['mean_cross']:.4f}")
    print(f"Strong passages: {item['strong_count']}")
    print(
        f"Mean evidence similarity: "
        f"{item['mean_similarity']:.4f}"
    )
    print(
        f"Contradiction pairs: "
        f"{item['contradiction_pairs']}"
    )


print("\n" + "=" * 72)
print("                    Evaluation complete.")
print("=" * 72)
