import pandas as pd
import json
from pathlib import Path

URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"

OUT = Path("data/sample_hin.jsonl")
N = 500

print(f"Loading validation data...")
df = pd.read_parquet(URL)

df = df.head(N)

OUT.parent.mkdir(exist_ok=True)

with OUT.open("w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        item = {
            "query_id": int(row["query_id"]),
            "query": row["query"],
            "eng_query": row["Eng_Query"],
            "answer": row["Answer"],
            "eng_answer": row["Eng_Answer"],
            "english_passages": row["passages"]["English_passages"].tolist(),
            "translated_passages": row["passages"]["Translated_passages"].tolist(),
            "is_selected": row["passages"]["is_selected"].tolist()
        }

        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(df)} records to {OUT}")