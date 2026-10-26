from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "datasets" / "MSLCD_YOLO"

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(dataset_dir / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        workers=4,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(project_dir / "runs" / "detect"),
        name="lunarscope_baseline",
        exist_ok=True,
        seed=42,
        deterministic=True,
        save=True,
        plots=True,
    )

    print("\nTraining complete.")
    print("Best checkpoint:")
    print(
        project_dir
        / "runs"
        / "detect"
        / "lunarscope_baseline"
        / "weights"
        / "best.pt"
    )


if __name__ == "__main__":
    main()