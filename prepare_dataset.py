import json
import os
import shutil
from pathlib import Path

DATA_DIR = Path("data/MSLCD")
OUTPUT_DIR = Path("datasets/MSLCD_YOLO")

SPLITS = ["train", "val", "test"]

for split in SPLITS:
    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )


def link_or_copy(source, destination):
    """Create a hard link when possible; otherwise copy the file."""
    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


for split in SPLITS:
    manifest = DATA_DIR / f"{split}.jsonl"

    print(f"\nPreparing {split} split...")

    with manifest.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            sample = json.loads(line)

            # Use the primary grayscale lunar surface imagery.
            image_source = DATA_DIR / sample["modalities"]["ce2_dom"]
            label_source = DATA_DIR / sample["target"]

            if not image_source.is_file():
                raise FileNotFoundError(image_source)

            if not label_source.is_file():
                raise FileNotFoundError(label_source)

            sample_id = sample["id"]

            image_destination = (
                OUTPUT_DIR / "images" / split / f"{sample_id}.tif"
            )

            label_destination = (
                OUTPUT_DIR / "labels" / split / f"{sample_id}.txt"
            )

            link_or_copy(image_source, image_destination)

            # Labels are small, so copying them is fine.
            if not label_destination.exists():
                shutil.copy2(label_source, label_destination)

    print(f"Finished {split}")


# Write the Ultralytics dataset configuration.
yaml_content = """\
path: .
train: images/train
val: images/val
test: images/test

names:
  0: crater
"""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "data.yaml").write_text(
    yaml_content, encoding="utf-8"
)

print("\nDataset preparation complete.")
print(f"Dataset location: {OUTPUT_DIR.resolve()}")