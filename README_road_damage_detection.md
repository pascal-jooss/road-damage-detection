# 🔍 Road Damage Detection

A YOLOv8-based object detection system that automatically identifies and localises road damage — potholes, cracks, and manholes — in images. Trained on real street footage and deployable as an interactive web demo.

![Demo placeholder](docs/screenshot.png)

---

## Why This Matters

Manual road inspection is time-consuming, expensive, and inconsistent. A computer vision system that automatically detects and classifies road damage from smartphone or dashcam images enables municipalities to:

- **Prioritise maintenance** based on damage type and severity
- **Reduce inspection costs** through automated image analysis
- **Build systematic damage databases** from vehicle-mounted cameras
- **Respond faster** to citizen-reported road hazards

This project demonstrates end-to-end deep learning — from raw data preparation through model training, evaluation, and an interactive deployment — using a real-world dataset of urban, suburban, and rural road footage.

---

## Demo

Upload any road image to get instant damage detection with bounding boxes and confidence scores:

```bash
python app.py
# → http://localhost:7860
```

---

## Architecture

```
data/raw/
  images/ + labels-YOLO/
        │
        ▼
src/prepare_dataset.py
  Train / Val / Test split (70/15/15)
  Stratified by class
        │
        ▼
Google Colab (Tesla T4 GPU)
  YOLOv8n — pretrained on COCO
  Fine-tuned on road damage dataset
  50 epochs · imgsz=640 · batch=16
  MLflow experiment tracking
        │
        ▼
models/best.pt  (6.2 MB)
        │
        ├──▶ src/evaluate.py
        │       Validation on held-out test set
        │       Per-class metrics + confusion matrix
        │       Sample predictions with bounding boxes
        │
        └──▶ app.py
                Gradio web interface
                Upload image → bounding boxes + detection table
```

---

## Results

Trained for 50 epochs on ~1,400 images (Tesla T4, ~20 minutes):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | 0.539 | 0.512 | 0.489 | 0.222 |
| Manhole | 0.773 | 0.732 | 0.749 | 0.360 |
| Pothole | 0.439 | 0.419 | 0.363 | 0.157 |
| Crack | 0.404 | 0.386 | 0.356 | 0.148 |

**Key observations:**
- Manholes are detected reliably (mAP50: 0.749) due to their consistent circular shape and high contrast
- Potholes are harder to detect due to variable size, shape, and lighting conditions
- Cracks are the most challenging class — thin linear features are inherently difficult for bounding-box detection and would benefit from a segmentation approach

---

## Dataset

**Source:** [Road Damage Dataset: Potholes, Cracks and Manholes](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes) (Kaggle, MIT License)

| Property | Value |
|---|---|
| Total images | 2,009 |
| Resolution | 640×360 px |
| Capture devices | GoPro + smartphone |
| Annotation format | YOLO bounding boxes |
| Classes | Pothole (0), Crack (1), Manhole (2) |

**Split:**

| Split | Images | Pothole | Crack | Manhole |
|---|---|---|---|---|
| Train | 1,406 | 871 | 1,793 | 677 |
| Val | 301 | 233 | 376 | 142 |
| Test | 302 | 157 | 350 | 138 |

---

## Tech Stack

| Component | Tool | Role |
|---|---|---|
| Model | [YOLOv8n](https://docs.ultralytics.com) (Ultralytics) | Object detection |
| Base weights | COCO pretrained | Transfer learning |
| Training hardware | Google Colab Tesla T4 | GPU training |
| Inference hardware | Apple M2 (MPS) | Local inference |
| Experiment tracking | [MLflow](https://mlflow.org) | Metrics & artefacts |
| Demo | [Gradio](https://gradio.app) | Web interface |
| Python | 3.11 | — |

**Note on PyTorch:** YOLOv8 is built on PyTorch. Training uses CUDA tensors on the T4 GPU; local inference uses Apple MPS acceleration via `torch.backends.mps`. The model weights are standard PyTorch `.pt` files.

---

## Setup

### Prerequisites

- Python 3.11+
- ~2 GB free disk space (dataset not included)

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd road-damage-detection
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
kaggle datasets download -d lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes
unzip road-damage-dataset-potholes-cracks-and-manholes.zip -d data/raw/
```

### 4. Prepare the dataset

```bash
python src/prepare_dataset.py
```

### 5. Download the pretrained model

Place `best.pt` in the `models/` directory. The model is available in the [Releases](../../releases) section of this repository.

### 6. Run the demo

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## Project Structure

```
road-damage-detection/
├── data/
│   ├── raw/                    # Original Kaggle dataset (git-ignored)
│   └── processed/              # Train/val/test split (git-ignored)
│       ├── train/images+labels/
│       ├── val/images+labels/
│       └── test/images+labels/
├── notebooks/
│   └── 01_eda.ipynb            # EDA + dataset analysis + model results
├── src/
│   ├── prepare_dataset.py      # Stratified train/val/test split
│   ├── train.py                # YOLOv8 training + MLflow logging
│   └── evaluate.py             # Test set evaluation + sample predictions
├── models/
│   └── best.pt                 # Trained model weights (git-ignored)
├── runs/                       # Training and eval outputs (git-ignored)
├── app.py                      # Gradio demo
├── dataset.yaml                # YOLO dataset configuration
├── requirements.txt
└── .gitignore
```

---

## Training (Colab)

The model was trained on Google Colab with a free Tesla T4 GPU. To retrain:

1. Open `notebooks/train_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Upload `data/processed.zip` and `dataset.yaml`
4. Run all cells — training takes ~20 minutes
5. Download `best.pt` from the outputs

Training configuration:
```python
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cuda",
    patience=10
)
```

---

## Design Decisions

**YOLOv8n (nano) over larger variants**
YOLOv8n has 3M parameters and runs at ~5ms per image on MPS — fast enough for real-time inference on a laptop. For a portfolio demo that needs to run locally without a GPU, the nano variant is the right choice. mAP50 of 0.489 is respectable given the model size and dataset scale.

**Transfer learning from COCO weights**
Starting from COCO pretrained weights (which include road, vehicle, and person classes) gives the model a meaningful head start for detecting objects on road surfaces. Fine-tuning takes 50 epochs instead of the hundreds required for training from scratch.

**Stratified train/val/test split**
The dataset is imbalanced — cracks appear nearly twice as often as manholes. A naive random split risks concentrating rare classes in one partition. `prepare_dataset.py` reads the dominant class from each label file and stratifies accordingly, ensuring representative class distributions across all three splits.

**Google Colab for training, local MPS for inference**
Training a detection model on CPU is impractical (2+ hours per epoch on this dataset). Colab provides a free T4 GPU that completes 50 epochs in ~20 minutes. The trained weights are then used locally via Apple MPS — a clean separation of training and deployment environments.

**Gradio over Streamlit for the demo**
Gradio's built-in `gr.Image` component handles image upload, display, and example images with minimal boilerplate. For a computer vision demo where the visual output is the primary deliverable, Gradio produces a cleaner interface with less code than Streamlit.

---

## Known Limitations & Learnings

**Crack detection is fundamentally harder with bounding boxes**
Cracks are thin, elongated features that often span irregular paths across an image. A bounding box around a crack captures mostly road surface, giving the model a weak signal. Instance segmentation (YOLOv8-seg) would be more appropriate for this class and is a natural next step.

**Small dataset size limits generalisation**
~2,000 images is sufficient for a proof of concept but not for production deployment. Road damage appearance varies significantly by geography, climate, road material, and camera angle. A production system would require tens of thousands of images from diverse conditions.

**Single-camera perspective only**
The dataset was captured from vehicle-mounted cameras at 1.2–1.5m height. The model may not generalise well to drone imagery, pedestrian-angle photos, or significantly different camera heights.

**No severity scoring**
The model classifies damage type but not severity. A production system for municipal maintenance prioritisation would need a severity score (e.g. small/medium/large pothole) to support resource allocation decisions.

**Potential next steps**

| Enhancement | Expected Benefit |
|---|---|
| YOLOv8-seg (instance segmentation) | Better crack detection via pixel masks |
| Larger model (YOLOv8s or YOLOv8m) | Higher mAP at the cost of inference speed |
| Data augmentation (weather simulation) | Better generalisation across conditions |
| Severity classification head | Actionable maintenance prioritisation |
| Video inference mode | Real-time dashcam processing |

---

## License

MIT
