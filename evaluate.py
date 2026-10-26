from pathlib import Path

from ultralytics import YOLO

def main():
    PROJECT_DIR = Path(__file__).resolve().parent

    MODEL_PATH = (
        PROJECT_DIR
        / "runs"
        / "detect"
        / "lunarscope_baseline"
        / "weights"
        / "best.pt"
    )

    DATA_YAML = PROJECT_DIR / "datasets" / "MSLCD_YOLO" / "data.yaml"

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=8,
        max_det=300,
        plots=True,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name="lunarscope_evaluation",
        exist_ok=True,
    )

    print("\nValidation Results")
    print("-" * 35)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()