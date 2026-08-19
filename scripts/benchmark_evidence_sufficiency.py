import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import accuracy_score

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

# Current production thresholds
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

# Warmup
embed_model.encode(["warmup"], normalize_embeddings=True)
cross_model.predict([["warmup", "warmup"]])

# ============================================================
# BUILD SIGNALS
# ============================================================

results = []

print("\nBuilding evidence-sufficiency signals...")

for n, record in enumerate(records[:NUM_QUERIES], 1):

    query = record["eng_query"]
    answerable = bool(any(record["is_selected"]))

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    dense_scores, indices = index.search(
        q_emb,
        DENSE_K
    )

    dense_scores = dense_scores[0]
    indices = indices[0]

    candidates = []

    for dense_score, idx in zip(dense_scores, indices):

        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        candidates.append({
            "dense": float(dense_score),
            "text": metadata[idx]["text"],
            "index": idx
        })

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
        candidate["cross"] = float(cross_score)

    candidates.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    top = candidates[:FINAL_K]

    if not top:
        continue

    # --------------------------------------------------------
    # Evidence signals
    # --------------------------------------------------------

    dense = np.array(
        [x["dense"] for x in top],
        dtype=float
    )

    cross = np.array(
        [x["cross"] for x in top],
        dtype=float
    )

    top1_dense = float(dense[0])
    top1_cross = float(cross[0])

    # Number of passages clearing useful thresholds
    dense_hits = int(
        np.sum(dense >= DENSE_THRESHOLD)
    )

    cross_hits = int(
        np.sum(cross >= CROSS_THRESHOLD)
    )

    # Average quality of the reranked evidence
    mean_cross = float(np.mean(cross))

    # Mean of passages that actually clear the cross threshold
    strong_cross = cross[cross >= CROSS_THRESHOLD]

    if len(strong_cross) > 0:
        mean_strong_cross = float(np.mean(strong_cross))
    else:
        mean_strong_cross = 0.0

    # How much of the evidence is strongly supported?
    strong_fraction = cross_hits / FINAL_K

    # Gap between strongest and second strongest evidence
    if len(cross) >= 2:
        cross_gap = float(cross[0] - cross[1])
    else:
        cross_gap = 0.0

    # Combined evidence score:
    # rewards multiple strong passages rather than only top-1.
    evidence_score = (
        0.50 * top1_cross
        + 0.25 * mean_cross
        + 0.15 * mean_strong_cross
        + 2.0 * strong_fraction
    )

    results.append({
        "query": query,
        "answerable": answerable,
        "top1_dense": top1_dense,
        "top1_cross": top1_cross,
        "dense_hits": dense_hits,
        "cross_hits": cross_hits,
        "mean_cross": mean_cross,
        "mean_strong_cross": mean_strong_cross,
        "strong_fraction": strong_fraction,
        "cross_gap": cross_gap,
        "evidence_score": evidence_score
    })

    if n % 10 == 0:
        print(f"Processed {n}/{NUM_QUERIES}")

# ============================================================
# SUMMARY
# ============================================================

y_true = np.array(
    [int(r["answerable"]) for r in results]
)

print("\n")
print("=" * 72)
print("              EVIDENCE SUFFICIENCY DIAGNOSTIC")
print("=" * 72)

print(f"\nQueries evaluated: {len(results)}")
print(f"Answerable:        {sum(y_true)}")
print(f"Unanswerable:      {len(y_true) - sum(y_true)}")

# ============================================================
# SIGNAL DISTRIBUTIONS
# ============================================================

answerable = [
    r for r in results
    if r["answerable"]
]

unanswerable = [
    r for r in results
    if not r["answerable"]
]

def stats(name):

    a = np.array(
        [r[name] for r in answerable],
        dtype=float
    )

    u = np.array(
        [r[name] for r in unanswerable],
        dtype=float
    )

    print(f"\n{name}")
    print(
        f"  Answerable   mean={a.mean():.4f} "
        f"median={np.median(a):.4f} "
        f"min={a.min():.4f}"
    )
    print(
        f"  Unanswerable mean={u.mean():.4f} "
        f"median={np.median(u):.4f} "
        f"max={u.max():.4f}"
    )

stats("top1_cross")
stats("mean_cross")
stats("mean_strong_cross")
stats("strong_fraction")
stats("cross_gap")
stats("evidence_score")

# ============================================================
# CURRENT PRODUCTION BASELINE
# ============================================================

print("\n")
print("=" * 72)
print("CURRENT PRODUCTION BASELINE")
print("=" * 72)

pred = np.array([
    int(
        r["top1_dense"] >= DENSE_THRESHOLD
        and r["top1_cross"] >= CROSS_THRESHOLD
    )
    for r in results
])

print(
    f"Accuracy: {accuracy_score(y_true, pred):.3f}"
)

# ============================================================
# EVIDENCE SCORE TEST
# ============================================================

print("\n")
print("=" * 72)
print("EVIDENCE SCORE THRESHOLD TEST")
print("=" * 72)

print(
    "\nThreshold | Accuracy | False answers | False refusals"
)

best = None

for threshold in np.arange(4.0, 10.01, 0.25):

    pred = np.array([
        int(r["evidence_score"] >= threshold)
        for r in results
    ])

    accuracy = accuracy_score(y_true, pred)

    false_answers = int(
        np.sum(
            (pred == 1) &
            (y_true == 0)
        )
    )

    false_refusals = int(
        np.sum(
            (pred == 0) &
            (y_true == 1)
        )
    )

    print(
        f"{threshold:9.2f} | "
        f"{accuracy:8.3f} | "
        f"{false_answers:13d} | "
        f"{false_refusals:14d}"
    )

    # Prefer fewer false answers first,
    # then higher accuracy.
    ranking = (
        false_answers,
        -accuracy,
        false_refusals
    )

    if best is None or ranking < best["ranking"]:
        best = {
            "threshold": threshold,
            "accuracy": accuracy,
            "false_answers": false_answers,
            "false_refusals": false_refusals,
            "ranking": ranking
        }

# ============================================================
# STRONG-PASSAGE COUNT TEST
# ============================================================

print("\n")
print("=" * 72)
print("STRONG-PASSAGE COUNT TEST")
print("=" * 72)

print(
    "\nMinimum cross>=7 passages | Accuracy | "
    "False answers | False refusals"
)

for minimum_hits in range(0, FINAL_K + 1):

    pred = np.array([
        int(r["cross_hits"] >= minimum_hits)
        for r in results
    ])

    accuracy = accuracy_score(y_true, pred)

    false_answers = int(
        np.sum(
            (pred == 1) &
            (y_true == 0)
        )
    )

    false_refusals = int(
        np.sum(
            (pred == 0) &
            (y_true == 1)
        )
    )

    print(
        f"{minimum_hits:25d} | "
        f"{accuracy:8.3f} | "
        f"{false_answers:13d} | "
        f"{false_refusals:14d}"
    )

# ============================================================
# CURRENT FALSE ANSWERS
# ============================================================

print("\n")
print("=" * 72)
print("CURRENT FALSE-ANSWER CASES")
print("=" * 72)

false_answer_cases = [
    r for r in results
    if not r["answerable"]
    and r["top1_dense"] >= DENSE_THRESHOLD
    and r["top1_cross"] >= CROSS_THRESHOLD
]

for i, r in enumerate(false_answer_cases, 1):

    print(f"\nCASE #{i}")
    print(f"Query: {r['query']}")
    print(f"Top1 dense:       {r['top1_dense']:.4f}")
    print(f"Top1 cross:       {r['top1_cross']:.4f}")
    print(f"Cross >= 7 hits:  {r['cross_hits']}")
    print(f"Mean cross:       {r['mean_cross']:.4f}")
    print(f"Strong fraction:  {r['strong_fraction']:.2f}")
    print(f"Evidence score:   {r['evidence_score']:.4f}")

# ============================================================
# CONCLUSION
# ============================================================

print("\n")
print("=" * 72)
print("DIAGNOSTIC CONCLUSION")
print("=" * 72)

print(
    f"\nBest evidence-score threshold under the "
    f"false-answer-first criterion: {best['threshold']:.2f}"
)

print(
    f"Accuracy at that threshold: {best['accuracy']:.3f}"
)

print(
    f"False answers: {best['false_answers']}"
)

print(
    f"False refusals: {best['false_refusals']}"
)

print(
    "\nIMPORTANT:"
)

print(
    "This script is diagnostic only."
)

print(
    "Do NOT change rag_pipeline.py yet."
)

print(
    "We will use these numbers to decide whether "
    "evidence sufficiency is actually worth adding "
    "to the production guardrail."
)

print("\n" + "=" * 72)
print("Evaluation complete.")
print("=" * 72)