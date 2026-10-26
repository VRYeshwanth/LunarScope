# LunarScope 🌕
### Lunar Crater Detection Using YOLO

LunarScope is a computer vision project that uses a fine-tuned YOLO object detection model to identify and localize craters in lunar surface images. It uses Chandrayaan-2 imagery and a pretrained YOLO11 model to detect crater bounding boxes.

The goal is to develop an image-based lunar crater detection system and evaluate its performance using object detection metrics.

## Features

- **Lunar crater detection:** Detects craters in lunar surface imagery.
- **Transfer learning:** Fine-tunes a pretrained YOLO11 model for a single-class detection task.
- **Dataset preparation:** Converts geospatial TIFF images into PNG images suitable for YOLO training.
- **Object detection evaluation:** Measures performance using mAP, precision, recall, and confusion matrices.
- **Prediction visualization:** Generates images with predicted crater bounding boxes.

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO11
- OpenCV
- NumPy
- Matplotlib

## Dataset

LunarScope uses the **MSLCD (Multi-Source Lunar Crater Detection) dataset**, distributed through the LANYUE-Bench dataset repository.

**Dataset source:** [LANYUE-Bench on Hugging Face](https://huggingface.co/datasets/LLukaoo/LANYUE-bench)

### Dataset details

| Property | Details |
|---|---|
| Dataset | MSLCD |
| Task | Object detection |
| Target class | Crater |
| Total samples | 8,881 |
| Training samples | 6,485 |
| Validation samples | 1,401 |
| Test samples | 995 |
| Primary imagery | Chandrayaan-2 CE-2 DOM |
| Original image dimensions | 512 × 512 |
| Original image format | TIFF |
| Annotation format | YOLO bounding boxes |
| Number of classes | 1 |

The dataset also provides additional modalities, including elevation data. This project uses **CE-2 DOM imagery** as the primary model input.

### Dataset annotations

Each annotation file contains one bounding box per line, using the YOLO format:

```text
class_id x_center y_center width height
```

The coordinates are normalized relative to the image dimensions.

For this project, class ID `0` represents a crater.

## Project Structure

```text
LunarScope/
│
├── data/
│   └── MSLCD/
│       ├── assets/
│       │   ├── ce2_dom/
│       │   ├── ce2_dem/
│       │   ├── sldem/
│       │   ├── target/
│       │   └── wac_dom/
│       ├── dataset.json
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── checksums.sha256
│
├── datasets/
│   └── MSLCD_YOLO/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
│
├── runs/
│   └── detect/
│       └── lunarscope_baseline/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── ...
│
├── weights/
├── archives/
│
├── inspect_dataset.py
├── inspect_labels.py
├── prepare_dataset.py
├── visualize_sample.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── yolo11n.pt
└── README.md
```

The directory structure above illustrates the expected dataset layout and generated training outputs. Some directories and files are created only after the corresponding steps are completed.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd LunarScope
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your repository.

### 2. Create a virtual environment

Using Python 3.12:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If the file does not contain all required packages, install the main dependencies:

```bash
pip install ultralytics opencv-python numpy matplotlib
```

Verify that Ultralytics is available:

```bash
yolo checks
```

## Dataset Download

The MSLCD dataset is distributed through the LANYUE-Bench repository on Hugging Face.

### 1. Open the dataset repository

Visit:

https://huggingface.co/datasets/LLukaoo/LANYUE-bench

Locate the MSLCD dataset files or archives provided by the repository.

### 2. Download the MSLCD data

Download the archive containing the MSLCD dataset. Follow the dataset repository's instructions if multiple archives are required.

The complete MSLCD dataset is approximately 8.1 GB, so ensure that sufficient storage is available.

### 3. Extract the dataset

Extract the downloaded archive into the project's `data/` directory.

The expected structure is:

```text
data/
└── MSLCD/
    ├── assets/
    │   ├── ce2_dom/
    │   ├── ce2_dem/
    │   ├── sldem/
    │   ├── target/
    │   └── wac_dom/
    ├── dataset.json
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    └── checksums.sha256
```

**Important:** The exact location of the extracted files must match the paths used by the Python scripts. If the archive extracts into an additional nested directory, move or reorganize its contents so that the structure above is preserved.

The manifest files contain references to the corresponding images and annotation files.

## Dataset Inspection

Before preparing the dataset, inspect its metadata and file structure:

```bash
python inspect_dataset.py
```

Inspect the label distribution:

```bash
python inspect_labels.py
```

To visualize a sample image and its ground-truth bounding boxes:

```bash
python visualize_sample.py
```

These scripts help verify that the dataset is extracted correctly and that the annotations can be read.

## Dataset Preparation

The original dataset uses TIFF images and JSONL manifests. Ultralytics YOLO expects a directory structure containing images and corresponding YOLO-format text files.

Run the preparation script:

```bash
python prepare_dataset.py
```

The script:

1. Reads the training, validation, and test manifests.
2. Locates the CE-2 DOM images and their corresponding annotation files.
3. Converts the grayscale TIFF images to PNG.
4. Copies the original YOLO annotation files without changing their contents.
5. Preserves the original dataset splits.
6. Creates the directory structure expected by YOLO.
7. Generates `datasets/MSLCD_YOLO/data.yaml`.

After preparation, the dataset should contain:

```text
datasets/MSLCD_YOLO/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

The generated `data.yaml` identifies the dataset paths and the single target class:

```yaml
path: /path/to/MSLCD_YOLO
train: images/train
val: images/val
test: images/test

names:
  0: crater
```

The actual absolute path is generated by the preparation script.

**Note:** The script may need to be run again if the dataset is moved or deleted. Existing PNG images and labels are not necessarily regenerated when they already exist, so remove the prepared dataset directory before rerunning the script if you need to rebuild it from scratch.

## Model Training

LunarScope uses a pretrained YOLO11 nano model as its baseline detector.

Start training with:

```bash
python train.py
```

The training configuration uses the following settings:

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 8 |
| Early stopping patience | 10 |
| Random seed | 42 |
| Classes | 1 |

The model is trained on the training split and evaluated on the validation split after each epoch.

Training outputs are saved under:

```text
runs/detect/lunarscope_baseline/
```

Important output files include:

- `weights/best.pt` — best checkpoint according to the validation fitness metric.
- `weights/last.pt` — checkpoint from the final completed epoch.
- `results.csv` — per-epoch training and validation metrics.
- `results.png` — training and validation metric plots.
- `confusion_matrix.png` — confusion matrix.
- `PR_curve.png` — precision-recall curve.
- `F1_curve.png` — F1-confidence curve.
- `P_curve.png` — precision-confidence curve.
- `R_curve.png` — recall-confidence curve.

The exact files generated depend on the Ultralytics version and training configuration.

## Model Evaluation

To evaluate the trained model on the validation set, run:

```bash
python evaluate.py
```

The evaluation script loads `best.pt` and calculates object detection metrics.

The principal metrics are:

- **mAP@0.50:** Mean average precision at an intersection-over-union threshold of 0.50.
- **mAP@0.50:0.95:** Mean average precision averaged across IoU thresholds from 0.50 to 0.95.
- **Precision:** The proportion of predicted detections that are correct.
- **Recall:** The proportion of annotated objects that are successfully detected.

Evaluation plots and metrics are saved under the configured output directory.

### Baseline Results

The initial YOLO11n experiment achieved the following validation results:

| Metric | Score |
|---|---:|
| mAP@0.50 | 0.6675 |
| mAP@0.50:0.95 | 0.3797 |
| Precision | 0.6121 |
| Recall | 0.6399 |

These results represent the initial baseline and may vary with dependency versions, hardware, training conditions, and dataset preparation.

The test split has not been used for these reported results. It should be reserved for final evaluation after model selection.

## Inference

After training, the best checkpoint can be used to detect craters in new images.

Example:

```python
from ultralytics import YOLO

model = YOLO(
    "runs/detect/lunarscope_baseline/weights/best.pt"
)

results = model.predict(
    source="path/to/lunar_image.png",
    imgsz=640,
    conf=0.25,
    save=True,
)

print("Prediction completed.")
```

The output image containing the predicted bounding boxes is saved by Ultralytics.

For a directory of images, set `source` to the directory path.

## License and Attribution

The dataset is distributed through [LANYUE-Bench](https://huggingface.co/datasets/LLukaoo/LANYUE-bench). Consult the dataset documentation and the original data providers' terms before redistributing imagery or annotations.

The license of the dataset reader or benchmark repository should not automatically be assumed to cover all third-party imagery.

## Acknowledgements

- [Ultralytics](https://www.ultralytics.com/) for the YOLO object detection framework.
- [LANYUE-Bench](https://huggingface.co/datasets/LLukaoo/LANYUE-bench) for providing access to the MSLCD dataset.
- The original providers of the lunar imagery and crater annotations.