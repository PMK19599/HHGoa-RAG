import pandas as pd

URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"

print("Reading a small sample from Hindi validation data...")

df = pd.read_parquet(URL)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nFIRST RECORD:")
row = df.iloc[0]

for key, value in row.items():
    if key == "passages":
        print("\npassages:")
        print(value)
    else:
        print(f"\n{key}: {value}")