import json
from pathlib import Path

DATA_DIR = Path("data/MSLCD")

# Inspect dataset metadata
with open(DATA_DIR / "dataset.json", encoding="utf-8") as f:
    metadata = json.load(f)

print("Dataset metadata:")
print(json.dumps(metadata, indent=2))

# Inspect the first training sample
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

print("\nFirst training sample:")
print(json.dumps(sample, indent=2))

# Inspect the assets directory
print("\nAssets directory contents:")
for path in list((DATA_DIR / "assets").rglob("*"))[:20]:
    print(path)