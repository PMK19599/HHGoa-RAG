import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

DATA = Path("data/sample_hin.jsonl")
INDEX_PATH = Path("data/bge_native.faiss")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
NUM_QUERIES = 500

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

records = []

with DATA.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

model.encode(["warmup"], normalize_embeddings=True)

X = []
y = []

print("Building guardrail features...")

for record in records[:NUM_QUERIES]:

    query = record["eng_query"]

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.asarray(q_emb, dtype="float32")

    scores, _ = index.search(q_emb, 5)

    top1 = float(scores[0][0])
    top2 = float(scores[0][1])
    top3 = float(scores[0][2])
    top5 = float(scores[0][4])

    gap12 = top1 - top2
    gap15 = top1 - top5
    mean_top5 = float(np.mean(scores[0]))

    features = [
        top1,
        top2,
        top3,
        top5,
        gap12,
        gap15,
        mean_top5
    ]

    X.append(features)
    y.append(1 if any(record["is_selected"]) else 0)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

clf = LGBMClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.05,
    verbosity=-1
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

tn, fp, fn, tp = cm.ravel()

print("\n=== CLASSIFIER RESULTS ===")
print(f"Test queries: {len(y_test)}")
print(f"Accuracy: {accuracy:.3f}")
print(f"True answers: {tp}")
print(f"True refusals: {tn}")
print(f"False answers: {fp}")
print(f"False refusals: {fn}")

print("\nFeature importance:")
for name, importance in zip(
    [
        "top1",
        "top2",
        "top3",
        "top5",
        "gap12",
        "gap15",
        "mean_top5"
    ],
    clf.feature_importances_
):
    print(f"{name}: {importance}")