import json
import re
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

with DATA.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print(f"Index ready: {index.ntotal} passages")

print("Warming up models...")
embed_model.encode(["warmup"], normalize_embeddings=True)
cross_model.predict([["warmup", "warmup passage"]])


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):
    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    dense_scores, indices = index.search(q_emb, DENSE_K)

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
            "cross": 0.0,
            "text": metadata[idx]["text"],
            "idx": idx
        })

    if not candidates:
        return []

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


# ============================================================
# CLAIM-SUPPORT SIGNALS
# ============================================================

def normalize_words(text):
    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )


def lexical_overlap(query, passage):
    q = set(normalize_words(query))
    p = set(normalize_words(passage))

    if not q:
        return 0.0

    return len(q & p) / len(q)


def direction_signal(query, passage):
    """
    Lightweight diagnostic for obvious directional mismatch.

    This is NOT treated as a final truth classifier.
    It only highlights cases where the query contains
    directional language and the evidence appears reversed.
    """

    q = query.lower()
    p = passage.lower()

    directional_pairs = [
        ("increase", "increase"),
        ("decrease", "decrease"),
        ("cause", "cause"),
        ("causes", "causes"),
        ("help", "help"),
        ("prevent", "prevent"),
        ("reduce", "reduce"),
        ("lead to", "lead to"),
    ]

    for q_term, p_term in directional_pairs:
        if q_term in q and p_term in p:
            return 1.0

    return 0.0


def classify_support(query, results):
    """
    Diagnostic classification only.

    SUPPORTED:
        Strong evidence directly aligned with the query.

    INSUFFICIENT:
        Relevant evidence exists, but support is weak/incomplete.

    CONTRADICTORY:
        Strong passages disagree with each other.

    This intentionally does NOT modify production behavior.
    """

    if not results:
        return "INSUFFICIENT"

    cross = np.array(
        [r["cross"] for r in results],
        dtype=np.float32
    )

    strong = [
        r for r in results
        if r["cross"] >= CROSS_THRESHOLD
    ]

    # Detect obvious contradiction among strong passages.
    contradiction = False

    for i in range(len(strong)):
        for j in range(i + 1, len(strong)):
            a = strong[i]["text"].lower()
            b = strong[j]["text"].lower()

            # Simple opposing-language diagnostic.
            opposition_pairs = [
                ("cannot", "can"),
                ("can't", "can"),
                ("not", "is"),
                ("no", "yes"),
                ("never", "always"),
                ("unable", "able"),
            ]

            for left, right in opposition_pairs:
                if left in a and right in b:
                    contradiction = True
                if right in a and left in b:
                    contradiction = True

    if contradiction:
        return "CONTRADICTORY"

    best = results[0]

    overlap = lexical_overlap(
        query,
        best["text"]
    )

    if (
        best["dense"] >= DENSE_THRESHOLD
        and best["cross"] >= CROSS_THRESHOLD
        and overlap >= 0.20
    ):
        return "SUPPORTED"

    return "INSUFFICIENT"


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 72)
print("          HH GOA RAG - CLAIM SUPPORT DIAGNOSTIC")
print("=" * 72)

print("\nEvaluating retrieved evidence...")

diagnostics = []

for i, record in enumerate(records[:NUM_QUERIES]):

    query = record["eng_query"]

    answerable = bool(any(record["is_selected"]))

    results = retrieve(query)

    classification = classify_support(
        query,
        results
    )

    diagnostics.append({
        "query": query,
        "answerable": answerable,
        "classification": classification,
        "best_dense": (
            results[0]["dense"]
            if results else 0.0
        ),
        "best_cross": (
            results[0]["cross"]
            if results else 0.0
        ),
        "strong_count": sum(
            r["cross"] >= CROSS_THRESHOLD
            for r in results
        )
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{NUM_QUERIES}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("                         RESULTS")
print("=" * 72)

answerable = [
    x for x in diagnostics
    if x["answerable"]
]

unanswerable = [
    x for x in diagnostics
    if not x["answerable"]
]

print(f"\nQueries evaluated: {len(diagnostics)}")
print(f"Answerable:        {len(answerable)}")
print(f"Unanswerable:      {len(unanswerable)}")

for label in [
    "SUPPORTED",
    "INSUFFICIENT",
    "CONTRADICTORY"
]:
    total = sum(
        x["classification"] == label
        for x in diagnostics
    )

    ans = sum(
        x["classification"] == label
        and x["answerable"]
        for x in diagnostics
    )

    unans = total - ans

    print(
        f"{label:<15} total={total:3d} "
        f"answerable={ans:3d} "
        f"unanswerable={unans:3d}"
    )


# ============================================================
# CONFUSION-STYLE DIAGNOSTIC
# ============================================================

print("\n" + "-" * 72)
print("ANSWERABLE QUERY BREAKDOWN")
print("-" * 72)

for label in [
    "SUPPORTED",
    "INSUFFICIENT",
    "CONTRADICTORY"
]:
    subset = [
        x for x in answerable
        if x["classification"] == label
    ]

    print(
        f"{label:<15}: {len(subset):3d}"
    )


print("\n" + "-" * 72)
print("UNANSWERABLE QUERY BREAKDOWN")
print("-" * 72)

for label in [
    "SUPPORTED",
    "INSUFFICIENT",
    "CONTRADICTORY"
]:
    subset = [
        x for x in unanswerable
        if x["classification"] == label
    ]

    print(
        f"{label:<15}: {len(subset):3d}"
    )


# ============================================================
# IMPORTANT CASES
# ============================================================

print("\n" + "=" * 72)
print("IMPORTANT DIAGNOSTIC CASES")
print("=" * 72)

for item in diagnostics:

    interesting = (
        (item["answerable"]
         and item["classification"] != "SUPPORTED")
        or
        (not item["answerable"]
         and item["classification"] == "SUPPORTED")
        or
        item["classification"] == "CONTRADICTORY"
    )

    if not interesting:
        continue

    print("\n" + "-" * 72)
    print(f"Query: {item['query']}")
    print(f"Ground truth answerable: {item['answerable']}")
    print(f"Classification:           {item['classification']}")
    print(f"Best dense:               {item['best_dense']:.4f}")
    print(f"Best cross:               {item['best_cross']:.4f}")
    print(f"Strong passages:          {item['strong_count']}")


print("\n" + "=" * 72)
print("CLAIM SUPPORT DIAGNOSTIC COMPLETE")
print("=" * 72)
print("\nIMPORTANT: Diagnostic only.")
print("No production configuration was changed.")
