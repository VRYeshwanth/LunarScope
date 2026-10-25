import json
from pathlib import Path

DATA_DIR = Path("data/MSLCD")

with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

label_path = DATA_DIR / sample["target"]

print("Sample ID:", sample["id"])
print("Image:", sample["modalities"]["ce2_dom"])
print("Label file:", label_path)

print("\nAnnotation contents:")
print(label_path.read_text(encoding="utf-8"))

print("\nNumber of annotation lines:")
print(len(label_path.read_text(encoding="utf-8").splitlines()))