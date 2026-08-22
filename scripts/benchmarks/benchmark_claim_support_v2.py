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

records = records[:NUM_QUERIES]

print(f"Index ready: {index.ntotal} passages")

print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["warmup query", "warmup passage"]]
)


# ============================================================
# TEXT HELPERS
# ============================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "would",
    "should", "of", "to", "in", "on", "for", "with",
    "and", "or", "that", "this", "it", "be", "as",
    "from", "by", "about", "into", "than", "their",
    "they", "them", "its", "has", "have", "had",
    "your", "you", "me"
}


def words(text):
    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )


def content_words(text):
    return [
        w for w in words(text)
        if w not in STOPWORDS and len(w) > 1
    ]


def numbers(text):
    return re.findall(
        r"""
        (?:
            \$\s?\d+(?:[.,]\d+)*
            |
            \d+(?:[.,]\d+)*\s?
            (?:
                gallons?|liters?|litres?|miles?|years?|
                months?|days?|hours?|minutes?|seconds?|
                dollars?|usd|cents?|kg|lbs?|pounds?|
                percent|%
            )
            |
            \d+(?:[.,]\d+)*
        )
        """,
        text.lower(),
        re.VERBOSE
    )


# ============================================================
# QUERY TYPE
# ============================================================

def query_type(query):

    q = query.lower().strip()

    if (
        "price" in q
        or "cost" in q
        or "how much" in q
        or "how many" in q
        or "how far" in q
        or "how long" in q
        or "what percentage" in q
    ):
        return "QUANTITY"

    if (
        "definition" in q
        or q.startswith("define ")
        or q.startswith("what is ")
        or q.startswith("what are ")
    ):
        return "DEFINITION"

    if (
        "symptoms" in q
        or "signs" in q
        or "features" in q
        or "types" in q
        or "examples" in q
        or "ways" in q
    ):
        return "LIST"

    if (
        q.startswith("does ")
        or q.startswith("do ")
        or q.startswith("did ")
        or q.startswith("can ")
        or q.startswith("is ")
        or q.startswith("are ")
        or q.startswith("will ")
        or q.startswith("should ")
    ):
        return "YES_NO"

    if any(
        term in q
        for term in [
            "cause",
            "causes",
            "increase",
            "increases",
            "decrease",
            "decreases",
            "reduce",
            "reduces",
            "prevent",
            "prevents",
            "lead to",
            "leads to",
            "result in",
            "results in",
            "help",
            "helps",
            "affect",
            "affects"
        ]
    ):
        return "CAUSAL"

    return "FACTUAL"


# ============================================================
# SIGNALS
# ============================================================

def lexical_topic_overlap(query, passage):

    q = set(content_words(query))
    p = set(content_words(passage))

    if not q:
        return 0.0

    return len(q & p) / len(q)


def requested_value_signal(query, passage):

    q = query.lower()

    quantity_request = any(
        phrase in q
        for phrase in [
            "how much",
            "how many",
            "price",
            "cost",
            "how far",
            "how long",
            "percentage",
            "amount"
        ]
    )

    if not quantity_request:
        return 1.0

    return 1.0 if numbers(passage) else 0.0


def definition_support(query, passage):

    q = query.lower()

    definition_request = (
        "definition" in q
        or q.startswith("define ")
        or q.startswith("what is ")
        or q.startswith("what are ")
    )

    if not definition_request:
        return 1.0

    p = passage.lower()

    patterns = [
        r"\bis defined as\b",
        r"\bmeans\b",
        r"\brefers to\b",
        r"\bis a\b",
        r"\bis an\b",
        r"\bare\b",
        r"\bknown as\b",
        r"\bdefinition of\b"
    ]

    return 1.0 if any(
        re.search(pattern, p)
        for pattern in patterns
    ) else 0.0


def list_support(query, passage):

    q = query.lower()

    list_request = any(
        term in q
        for term in [
            "symptoms",
            "signs",
            "features",
            "types",
            "examples",
            "ways",
            "causes"
        ]
    )

    if not list_request:
        return 1.0

    p = passage.lower()

    # A list-like answer usually contains multiple separated
    # items or explicit enumeration.
    separators = (
        p.count(",")
        + p.count(";")
        + len(re.findall(r"\b(?:first|second|third|also|including)\b", p))
    )

    return min(separators / 3.0, 1.0)


def causal_direction_support(query, passage):

    q = query.lower()
    p = passage.lower()

    causal_pairs = [
        ("increase", "increase"),
        ("increases", "increases"),
        ("decrease", "decrease"),
        ("decreases", "decreases"),
        ("cause", "cause"),
        ("causes", "causes"),
        ("caused", "caused"),
        ("help", "help"),
        ("helps", "helps"),
        ("prevent", "prevent"),
        ("prevents", "prevents"),
        ("reduce", "reduce"),
        ("reduces", "reduces"),
        ("lead to", "lead to"),
        ("leads to", "leads to"),
        ("result in", "result in"),
        ("results in", "results in"),
        ("affect", "affect"),
        ("affects", "affects")
    ]

    requested = [
        pair
        for pair in causal_pairs
        if pair[0] in q
    ]

    if not requested:
        return 1.0

    return 1.0 if any(
        p_term in p
        for _, p_term in requested
    ) else 0.0


def yes_no_support(query, passage):

    q = query.lower()
    p = passage.lower()

    yes_no = (
        q.startswith("does ")
        or q.startswith("do ")
        or q.startswith("did ")
        or q.startswith("can ")
        or q.startswith("is ")
        or q.startswith("are ")
        or q.startswith("will ")
        or q.startswith("should ")
    )

    if not yes_no:
        return 1.0

    # Require language that actually expresses possibility,
    # ability, negation, or the requested relation.
    patterns = [
        r"\bcan\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdoes\b",
        r"\bdoes not\b",
        r"\bdo not\b",
        r"\bdid\b",
        r"\bwill\b",
        r"\bshould\b",
        r"\bis\b",
        r"\bis not\b",
        r"\bpossible\b",
        r"\bimpossible\b",
        r"\bable\b",
        r"\bunable\b"
    ]

    return 1.0 if any(
        re.search(pattern, p)
        for pattern in patterns
    ) else 0.0


def explicit_contradiction(query, passages):

    """
    Conservative contradiction diagnostic.

    Only flag explicit opposing constructions.
    Do NOT use generic 'not' + 'is' matching.
    """

    contradiction_pairs = [
        (r"\bcannot\b", r"\bcan\b"),
        (r"\bcan't\b", r"\bcan\b"),
        (r"\bdoes not\b", r"\bdoes\b"),
        (r"\bdo not\b", r"\bdo\b"),
        (r"\bnever\b", r"\balways\b"),
        (r"\bimpossible\b", r"\bpossible\b"),
        (r"\bunable\b", r"\bable\b"),
        (r"\bdecrease\b", r"\bincrease\b"),
        (r"\bdecreases\b", r"\bincreases\b"),
        (r"\bno\b", r"\byes\b")
    ]

    strong_passages = [
        p.lower()
        for p in passages
    ]

    for i in range(len(strong_passages)):
        for j in range(i + 1, len(strong_passages)):

            a = strong_passages[i]
            b = strong_passages[j]

            for left, right in contradiction_pairs:

                if (
                    re.search(left, a)
                    and re.search(right, b)
                ):
                    return 1.0

                if (
                    re.search(right, a)
                    and re.search(left, b)
                ):
                    return 1.0

    return 0.0


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

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

    scores = cross_model.predict(pairs)

    for candidate, score in zip(
        candidates,
        scores
    ):
        candidate["cross"] = float(score)

    candidates.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    return candidates[:FINAL_K]


# ============================================================
# BUILD SIGNALS
# ============================================================

print()
print("=" * 72)
print("       HH GOA RAG - CLAIM SUPPORT V2 DIAGNOSTIC")
print("=" * 72)

print()
print("Evaluating targeted evidence-support signals...")
print()

diagnostics = []

for i, record in enumerate(
    records,
    start=1
):

    query = record["eng_query"]

    answerable = bool(
        any(record["is_selected"])
    )

    results = retrieve(query)

    if not results:
        continue

    best = results[0]

    passages = [
        r["text"]
        for r in results
        if r["cross"] >= CROSS_THRESHOLD
    ]

    if not passages:
        passages = [best["text"]]

    qtype = query_type(query)

    diagnostics.append({
        "query": query,
        "answerable": answerable,
        "type": qtype,

        "dense": best["dense"],
        "cross": best["cross"],

        "topic_overlap": lexical_topic_overlap(
            query,
            best["text"]
        ),

        "requested_value": requested_value_signal(
            query,
            best["text"]
        ),

        "definition": definition_support(
            query,
            best["text"]
        ),

        "list": list_support(
            query,
            best["text"]
        ),

        "causal": causal_direction_support(
            query,
            best["text"]
        ),

        "yes_no": yes_no_support(
            query,
            best["text"]
        ),

        "contradiction": explicit_contradiction(
            query,
            passages
        ),

        "strong_count": len(passages),

        "passage": best["text"]
    })

    if i % 10 == 0:
        print(
            f"Processed {i}/{len(records)}"
        )


# ============================================================
# DISTRIBUTIONS
# ============================================================

print()
print("=" * 72)
print("                    SIGNAL DISTRIBUTIONS")
print("=" * 72)

signal_names = [
    "topic_overlap",
    "requested_value",
    "definition",
    "list",
    "causal",
    "yes_no",
    "contradiction"
]

for signal in signal_names:

    ans = [
        x[signal]
        for x in diagnostics
        if x["answerable"]
    ]

    unans = [
        x[signal]
        for x in diagnostics
        if not x["answerable"]
    ]

    print()
    print(signal.upper())

    print(
        f"  Answerable   mean={np.mean(ans):.4f} "
        f"median={np.median(ans):.4f}"
    )

    print(
        f"  Unanswerable mean={np.mean(unans):.4f} "
        f"median={np.median(unans):.4f}"
    )


# ============================================================
# SIGNAL SEPARATION
# ============================================================

print()
print("=" * 72)
print("                    SIGNAL SEPARATION")
print("=" * 72)

print()
print(
    "Signal                    Answerable - Unanswerable"
)
print("-" * 72)

for signal in signal_names:

    ans = np.mean([
        x[signal]
        for x in diagnostics
        if x["answerable"]
    ])

    unans = np.mean([
        x[signal]
        for x in diagnostics
        if not x["answerable"]
    ])

    print(
        f"{signal:<25} {ans - unans:+.4f}"
    )


# ============================================================
# FALSE ANSWERS
# ============================================================

false_answers = [
    x for x in diagnostics
    if (
        x["dense"] >= DENSE_THRESHOLD
        and x["cross"] >= CROSS_THRESHOLD
        and not x["answerable"]
    )
]

print()
print("=" * 72)
print("                  FALSE ANSWER CASES")
print("=" * 72)

print()
print(
    f"Known production false answers: {len(false_answers)}"
)

for i, item in enumerate(
    false_answers,
    start=1
):

    print()
    print("-" * 72)

    print(
        f"CASE #{i}"
    )

    print(
        f"Query: {item['query']}"
    )

    print(
        f"Type: {item['type']}"
    )

    print(
        f"Dense: {item['dense']:.4f} | "
        f"Cross: {item['cross']:.4f}"
    )

    print(
        f"Topic overlap:     {item['topic_overlap']:.3f}"
    )

    print(
        f"Requested value:   {item['requested_value']:.1f}"
    )

    print(
        f"Definition:        {item['definition']:.1f}"
    )

    print(
        f"List:              {item['list']:.1f}"
    )

    print(
        f"Causal:            {item['causal']:.1f}"
    )

    print(
        f"Yes/No:            {item['yes_no']:.1f}"
    )

    print(
        f"Contradiction:     {item['contradiction']:.1f}"
    )

    print(
        f"Strong passages:   {item['strong_count']}"
    )

    print()
    print(
        "Evidence:"
    )

    print(
        item["passage"][:600]
    )


# ============================================================
# FALSE REFUSALS
# ============================================================

false_refusals = [
    x for x in diagnostics
    if (
        x["answerable"]
        and not (
            x["dense"] >= DENSE_THRESHOLD
            and x["cross"] >= CROSS_THRESHOLD
        )
    )
]

print()
print("=" * 72)
print("                  FALSE REFUSAL CASES")
print("=" * 72)

print()
print(
    f"Known production false refusals: {len(false_refusals)}"
)

for i, item in enumerate(
    false_refusals[:15],
    start=1
):

    print()
    print("-" * 72)

    print(
        f"CASE #{i}"
    )

    print(
        f"Query: {item['query']}"
    )

    print(
        f"Type: {item['type']}"
    )

    print(
        f"Dense: {item['dense']:.4f} | "
        f"Cross: {item['cross']:.4f}"
    )

    print(
        f"Topic overlap:   {item['topic_overlap']:.3f}"
    )

    print(
        f"Requested value: {item['requested_value']:.1f}"
    )

    print(
        f"Definition:      {item['definition']:.1f}"
    )

    print(
        f"List:            {item['list']:.1f}"
    )

    print(
        f"Causal:          {item['causal']:.1f}"
    )

    print(
        f"Yes/No:          {item['yes_no']:.1f}"
    )

    print(
        f"Evidence: {item['passage'][:400]}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 72)
print("              CLAIM SUPPORT V2 COMPLETE")
print("=" * 72)

print()
print("IMPORTANT:")
print("This is diagnostic only.")
print("No production configuration was changed.")
print("Do NOT modify rag_pipeline.py based on this benchmark alone.")
