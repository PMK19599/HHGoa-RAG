import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5
MAX_MISSES = 10

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading index...")
index = faiss.read_index(str(INDEX_PATH))

with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

miss_count = 0

for record in records[:100]:

    query = record["eng_query"]

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    scores, indices = index.search(q_emb, TOP_K)

    found = False

    for idx in indices[0]:
        meta = metadata[idx]

        if (
            meta["query_id"] == record["query_id"]
            and meta["is_selected"] == 1
        ):
            found = True
            break

    if not found:

        miss_count += 1

        print("\n" + "=" * 80)
        print(f"MISS #{miss_count}")
        print(f"QUERY: {query}")
        print(f"EXPECTED ANSWER: {record['eng_answer']}")

        print("\nLABELLED RELEVANT PASSAGES:")

        for i, selected in enumerate(record["is_selected"]):
            if selected == 1:
                print(f"- {record['english_passages'][i][:500]}")

        print("\nRETRIEVED TOP 5:")

        for rank, idx in enumerate(indices[0], start=1):
            print(f"\n#{rank} Score: {scores[0][rank-1]:.4f}")
            print(metadata[idx]["text"][:500])

        if miss_count >= MAX_MISSES:
            break