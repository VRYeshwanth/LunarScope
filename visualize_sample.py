import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

DATA_DIR = Path("data/MSLCD")

# Read the first training sample
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

# Load the primary lunar image
image_path = DATA_DIR / sample["modalities"]["ce2_dom"]
label_path = DATA_DIR / sample["target"]

image = Image.open(image_path).convert("RGB")
width, height = image.size

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)

# Read YOLO-format annotations
with open(label_path, encoding="utf-8") as f:
    for line in f:
        values = line.strip().split()

        if not values:
            continue

        class_id, x_center, y_center, box_width, box_height = (
            map(float, values)
        )

        # Convert normalized YOLO coordinates to pixel coordinates
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2

        rectangle = patches.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(rectangle)

ax.set_title(f"MSLCD Lunar Crater Detection: {sample['id']}")
ax.axis("off")
plt.tight_layout()
plt.show()