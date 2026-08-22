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

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

records = records[:NUM_QUERIES]

print(f"Index ready: {index.ntotal} passages")

print("Warming up models...")

embed_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

cross_model.predict(
    [["warmup", "warmup passage"]]
)


# ============================================================
# TEXT HELPERS
# ============================================================

def words(text):
    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )


def numbers(text):
    return re.findall(
        r"""
        (?:
            \$\s?\d+(?:[.,]\d+)*
            |
            \d+(?:[.,]\d+)*\s?(?:%
            | gallons?
            | gallon
            | miles?
            | mile
            | years?
            | year
            | months?
            | month
            | days?
            | day
            | hours?
            | hour
            | minutes?
            | minute
            | litres?
            | liters?
            | kg
            | lb
            | lbs
            | degrees?
            | °)
            |
            \d+(?:[.,]\d+)*
        )
        """,
        text.lower(),
        re.VERBOSE
    )


def query_tokens(query):
    return set(words(query))


# ============================================================
# ANSWER-SHAPE SIGNALS
# ============================================================

def quantity_signal(query, passage):
    """
    Does the evidence contain a quantitative value when
    the query appears to request one?
    """

    q = query.lower()

    quantity_request = any(
        phrase in q
        for phrase in [
            "how much",
            "how many",
            "what is the price",
            "price",
            "cost",
            "how far",
            "how long",
            "what percentage",
            "how old",
            "what amount"
        ]
    )

    if not quantity_request:
        return 0.0

    return 1.0 if numbers(passage) else 0.0


def price_signal(query, passage):
    """
    Specifically tests whether price/cost evidence contains
    an explicit monetary value.
    """

    q = query.lower()

    price_request = any(
        phrase in q
        for phrase in [
            "price",
            "cost",
            "how much does",
            "how much is",
            "how much to"
        ]
    )

    if not price_request:
        return 0.0

    p = passage.lower()

    monetary_patterns = [
        r"\$\s?\d+",
        r"\d+\s?(?:dollars?|usd)",
        r"\d+\s?(?:cents?)"
    ]

    return 1.0 if any(
        re.search(pattern, p)
        for pattern in monetary_patterns
    ) else 0.0


def list_signal(query, passage):
    """
    Lightweight signal for list-style questions such as
    symptoms, causes, signs, reasons, features, etc.
    """

    q = query.lower()

    list_request = any(
        phrase in q
        for phrase in [
            "what are",
            "symptoms",
            "signs",
            "causes",
            "reasons",
            "features",
            "types",
            "ways",
            "examples"
        ]
    )

    if not list_request:
        return 0.0

    p = passage.lower()

    # Signals that the passage actually enumerates concrete items.
    enumeration_patterns = [
        r"\b1[\).:]",
        r"\b2[\).:]",
        r"\b3[\).:]",
        r"\bfirst\b",
        r"\bsecond\b",
        r"\bthird\b",
        r"\band\b",
        r",",
        r";"
    ]

    hits = sum(
        1
        for pattern in enumeration_patterns
        if re.search(pattern, p)
    )

    return min(hits / 3.0, 1.0)


def definition_signal(query, passage):
    """
    Signal for definition-style questions.
    """

    q = query.lower()

    definition_request = (
        q.startswith("what is ")
        or q.startswith("what are ")
        or "definition" in q
        or q.startswith("define ")
    )

    if not definition_request:
        return 0.0

    p = passage.lower()

    definition_patterns = [
        r"\bis defined as\b",
        r"\bmeans\b",
        r"\bis a\b",
        r"\bis an\b",
        r"\bare defined as\b",
        r"\brefers to\b",
        r"\bknown as\b"
    ]

    return 1.0 if any(
        re.search(pattern, p)
        for pattern in definition_patterns
    ) else 0.0


def yes_no_signal(query, passage):
    """
    Signal for yes/no questions.

    This deliberately does NOT decide whether the answer is yes
    or no. It only checks whether the evidence contains language
    capable of expressing the requested binary relationship.
    """

    q = query.lower()

    yes_no_request = (
        q.startswith("can ")
        or q.startswith("is ")
        or q.startswith("are ")
        or q.startswith("does ")
        or q.startswith("do ")
        or q.startswith("did ")
        or q.startswith("will ")
        or q.startswith("should ")
    )

    if not yes_no_request:
        return 0.0

    p = passage.lower()

    binary_patterns = [
        r"\bcan\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdoes\b",
        r"\bdoes not\b",
        r"\bdo not\b",
        r"\bnot\b",
        r"\bis\b",
        r"\bis not\b",
        r"\bunable\b",
        r"\bpossible\b",
        r"\bimpossible\b",
        r"\bable\b"
    ]

    return 1.0 if any(
        re.search(pattern, p)
        for pattern in binary_patterns
    ) else 0.0


def causal_direction_signal(query, passage):
    """
    Diagnostic for obvious causal/relational direction.

    This is intentionally conservative. It does not claim
    semantic understanding; it only detects whether the
    query's directional language appears in the evidence.
    """

    q = query.lower()
    p = passage.lower()

    directional_terms = [
        "increase",
        "increases",
        "increased",
        "decrease",
        "decreases",
        "decreased",
        "cause",
        "causes",
        "caused",
        "lead to",
        "leads to",
        "result in",
        "results in",
        "help",
        "helps",
        "prevent",
        "prevents",
        "reduce",
        "reduces"
    ]

    requested = [
        term
        for term in directional_terms
        if term in q
    ]

    if not requested:
        return 0.0

    return 1.0 if any(
        term in p
        for term in requested
    ) else 0.0


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    dense_scores, indices = index.search(
        query_embedding,
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
        [query, item["text"]]
        for item in candidates
    ]

    cross_scores = cross_model.predict(pairs)

    for item, score in zip(
        candidates,
        cross_scores
    ):
        item["cross"] = float(score)

    candidates.sort(
        key=lambda x: x["cross"],
        reverse=True
    )

    return candidates[:FINAL_K]


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 72)
print("          HH GOA RAG - ANSWER SHAPE DIAGNOSTIC")
print("=" * 72)

print()
print("Building answer-shape signals...")
print()

results = []

for count, record in enumerate(
    records,
    start=1
):

    query = record["eng_query"]

    answerable = bool(
        any(record["is_selected"])
    )

    retrieved = retrieve(query)

    if not retrieved:
        continue

    best = retrieved[0]

    result = {
        "query": query,
        "answerable": answerable,
        "dense": best["dense"],
        "cross": best["cross"],
        "quantity": quantity_signal(
            query,
            best["text"]
        ),
        "price": price_signal(
            query,
            best["text"]
        ),
        "list": list_signal(
            query,
            best["text"]
        ),
        "definition": definition_signal(
            query,
            best["text"]
        ),
        "yes_no": yes_no_signal(
            query,
            best["text"]
        ),
        "causal": causal_direction_signal(
            query,
            best["text"]
        ),
        "passage": best["text"]
    }

    results.append(result)

    if count % 10 == 0:
        print(
            f"Processed {count}/{len(records)}"
        )


# ============================================================
# AGGREGATE SIGNAL ANALYSIS
# ============================================================

print()
print("=" * 72)
print("                 SIGNAL DISTRIBUTIONS")
print("=" * 72)

signals = [
    "quantity",
    "price",
    "list",
    "definition",
    "yes_no",
    "causal"
]

for signal in signals:

    answerable_values = [
        x[signal]
        for x in results
        if x["answerable"]
    ]

    unanswerable_values = [
        x[signal]
        for x in results
        if not x["answerable"]
    ]

    print()
    print(f"{signal.upper()}")

    print(
        f"  Answerable   mean={np.mean(answerable_values):.4f} "
        f"rate={np.mean(answerable_values):.3f}"
    )

    print(
        f"  Unanswerable mean={np.mean(unanswerable_values):.4f} "
        f"rate={np.mean(unanswerable_values):.3f}"
    )


# ============================================================
# FALSE ANSWERS
# ============================================================

false_answers = [
    x
    for x in results
    if (
        x["dense"] >= DENSE_THRESHOLD
        and x["cross"] >= CROSS_THRESHOLD
        and not x["answerable"]
    )
]

print()
print("=" * 72)
print("                    FALSE ANSWERS")
print("=" * 72)

print()
print(f"False answers found: {len(false_answers)}")

for i, item in enumerate(
    false_answers,
    start=1
):

    print()
    print("-" * 72)

    print(f"CASE #{i}")
    print(f"Query: {item['query']}")

    print()
    print(
        f"Dense: {item['dense']:.4f} | "
        f"Cross: {item['cross']:.4f}"
    )

    print(
        f"Quantity: {item['quantity']:.2f} | "
        f"Price: {item['price']:.2f} | "
        f"List: {item['list']:.2f}"
    )

    print(
        f"Definition: {item['definition']:.2f} | "
        f"YesNo: {item['yes_no']:.2f} | "
        f"Causal: {item['causal']:.2f}"
    )

    print()
    print("Evidence:")
    print(item["passage"][:500])


# ============================================================
# ANSWERABLE QUERIES WITH MISSING SHAPE
# ============================================================

print()
print("=" * 72)
print("             ANSWERABLE QUERIES WITH WEAK SHAPE")
print("=" * 72)

for item in results:

    if not item["answerable"]:
        continue

    q = item["query"].lower()

    expected_signal = None

    if (
        "price" in q
        or "cost" in q
        or "how much" in q
        or "how many" in q
    ):
        expected_signal = "quantity"

    elif (
        "symptoms" in q
        or "signs" in q
        or "what are" in q
    ):
        expected_signal = "list"

    elif (
        q.startswith("what is ")
        or "definition" in q
        or q.startswith("define ")
    ):
        expected_signal = "definition"

    if (
        expected_signal is not None
        and item[expected_signal] == 0.0
    ):

        print()
        print(
            f"Query: {item['query']}"
        )

        print(
            f"Expected shape: {expected_signal}"
        )

        print(
            f"Dense: {item['dense']:.4f} | "
            f"Cross: {item['cross']:.4f}"
        )

        print(
            f"Evidence: {item['passage'][:350]}"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 72)
print("              ANSWER SHAPE DIAGNOSTIC COMPLETE")
print("=" * 72)

print()
print("IMPORTANT:")
print("This is diagnostic only.")
print("No production configuration was changed.")
print("Do NOT modify rag_pipeline.py based on this benchmark alone.")
