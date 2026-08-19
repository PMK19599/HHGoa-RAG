import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path("data/sample_hin.jsonl")

INDEX_PATH = Path("data/bge_native.faiss")
META_PATH = Path("data/bge_native_metadata.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

passages = []
metadata = []

for record in records:
    for i, passage in enumerate(record["english_passages"]):

        passages.append(passage)

        metadata.append({
            "query_id": record["query_id"],
            "passage_index": i,
            "is_selected": record["is_selected"][i],
            "text": passage
        })

print(f"Embedding {len(passages)} passages...")

embeddings = model.encode(
    passages,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(embeddings)

faiss.write_index(
    index,
    str(INDEX_PATH)
)

with META_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False
    )

print("\nDONE")
print(f"Saved FAISS index: {INDEX_PATH}")
print(f"Saved metadata: {META_PATH}")
print(f"Indexed vectors: {index.ntotal}")