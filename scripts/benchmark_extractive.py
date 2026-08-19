import json
import time
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DATA_FILE = "data/sample_hin.jsonl"
INDEX_FILE = "data/bge_native.faiss"
META_FILE = "data/bge_native_metadata.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5
NUM_QUERIES = 100


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_answer(query, passage):
    """
    Very lightweight extractive answerer.
    Returns the sentence with the highest word overlap
    with the query.
    """

    sentences = re.split(r"(?<=[.!?])\s+", passage)

    if not sentences:
        return passage

    query_words = set(
        w.lower()
        for w in re.findall(r"\b[a-zA-Z]{3,}\b", query)
    )

    best_sentence = sentences[0]
    best_score = -1

    for sentence in sentences:

        sentence_words = set(
            w.lower()
            for w in re.findall(r"\b[a-zA-Z]{3,}\b", sentence)
        )

        overlap = len(query_words & sentence_words)

        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence

    return clean_text(best_sentence)


print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading metadata...")

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Global passages: {len(metadata)}")
print(f"FAISS vectors: {index.ntotal}")

records = []

with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

queries = records[:NUM_QUERIES]

print(f"Benchmarking {len(queries)} queries...")

latencies = []

retrieved_relevant = 0


for record in queries:

    query = record["eng_query"]

    query_start = time.perf_counter()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        np.asarray(query_embedding, dtype="float32"),
        TOP_K
    )

    # Find the first globally retrieved passage
    # that belongs to this query's passage set.
    retrieved_metadata = None

    for global_index in indices[0]:

        global_index = int(global_index)

        if global_index < 0:
            continue

        meta = metadata[global_index]

        if meta["query_id"] == record["query_id"]:

            retrieved_metadata = meta
            break

    if retrieved_metadata is not None:

        answer = extract_answer(
            query,
            retrieved_metadata["text"]
        )

        if int(retrieved_metadata["is_selected"]) == 1:
            retrieved_relevant += 1

    elapsed = (
        time.perf_counter() - query_start
    ) * 1000

    latencies.append(elapsed)


latencies = np.array(latencies)


print("\n=== EXTRACTIVE RESULTS ===")
print(f"Queries: {len(queries)}")
print(
    f"Retrieved relevant: "
    f"{retrieved_relevant}/{len(queries)}"
)

print(
    f"Recall@{TOP_K}: "
    f"{retrieved_relevant / len(queries):.3f}"
)

print("\nLATENCY")
print(f"P50: {np.percentile(latencies, 50):.2f} ms")
print(f"P70: {np.percentile(latencies, 70):.2f} ms")
print(f"P100: {np.percentile(latencies, 100):.2f} ms")
print(f"Mean: {np.mean(latencies):.2f} ms")
print(f"Min: {np.min(latencies):.2f} ms")