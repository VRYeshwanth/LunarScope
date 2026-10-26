import json
import cv2
from pathlib import Path

DATA_DIR = Path("data/MSLCD")
OUTPUT_DIR = Path("datasets/MSLCD_YOLO")

SPLITS = ["train", "val", "test"]


def main():
    for split in SPLITS:
        image_dir = OUTPUT_DIR / "images" / split
        label_dir = OUTPUT_DIR / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        manifest = DATA_DIR / f"{split}.jsonl"

        print(f"\nPreparing {split} split...")

        with manifest.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                sample = json.loads(line)
                sample_id = sample["id"]

                image_source = (
                    DATA_DIR / sample["modalities"]["ce2_dom"]
                )
                label_source = DATA_DIR / sample["target"]

                image_destination = (
                    image_dir / f"{sample_id}.png"
                )
                label_destination = (
                    label_dir / f"{sample_id}.txt"
                )

                if not image_source.is_file():
                    raise FileNotFoundError(image_source)

                if not label_source.is_file():
                    raise FileNotFoundError(label_source)

                # Convert TIFF to PNG only when needed.
                if not image_destination.exists():
                    image = cv2.imread(
                        str(image_source),
                        cv2.IMREAD_UNCHANGED,
                    )

                    if image is None:
                        raise RuntimeError(
                            f"Could not read {image_source}"
                        )

                    if image.ndim != 2 or image.dtype.name != "uint8":
                        raise ValueError(
                            f"Unexpected image format: "
                            f"{image_source}, "
                            f"shape={image.shape}, "
                            f"dtype={image.dtype}"
                        )

                    if not cv2.imwrite(
                        str(image_destination), image
                    ):
                        raise RuntimeError(
                            f"Could not save {image_destination}"
                        )

                # Preserve the original YOLO annotations.
                if not label_destination.exists():
                    label_destination.write_bytes(
                        label_source.read_bytes()
                    )

                if index % 500 == 0:
                    print(f"Processed {index} samples...")

        print(f"Finished {split}.")

    # Create dataset configuration.
    yaml_content = f"""\
path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: crater
"""

    (OUTPUT_DIR / "data.yaml").write_text(
        yaml_content, encoding="utf-8"
    )

    print("\nDataset conversion complete.")


if __name__ == "__main__":
    main()