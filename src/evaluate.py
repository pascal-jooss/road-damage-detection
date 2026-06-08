"""
Evaluate the trained YOLOv8n road-damage model on the test split.

Outputs (all under runs/eval/):
    confusion_matrix.png          – saved by ultralytics when plots=True
    confusion_matrix_normalized.png
    predictions/<filename>.jpg    – 5 random test images with drawn bounding boxes

Usage:
    python src/evaluate.py
    python src/evaluate.py --weights path/to/other.pt --n-samples 10
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = Path("models/best.pt")
DATASET_YAML = "dataset.yaml"
TEST_IMAGES_DIR = Path("data/processed/test/images")
EVAL_DIR = Path("runs/eval")
PREDS_DIR = EVAL_DIR / "predictions"

DEVICE = "mps"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
N_SAMPLES = 5
SEED = 42

CLASS_NAMES: dict[int, str] = {0: "Pothole", 1: "Crack", 2: "Manhole"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Evaluation ─────────────────────────────────────────────────────────────────
def run_validation(model: YOLO, weights: Path, eval_dir: Path) -> DetMetrics:
    """Run model.val() on the test split; confusion matrix PNG saved to eval_dir."""
    log.info("Running validation on test split …")
    log.info("  weights : %s", weights)
    log.info("  save_dir: %s", eval_dir)

    # Pass save_dir as absolute path so ultralytics uses it verbatim
    # (passing project/name triggers the RUNS_DIR / task / project / name prefix)
    metrics: DetMetrics = model.val(
        data=DATASET_YAML,
        split="test",
        device=DEVICE,
        imgsz=IMG_SIZE,
        plots=True,          # saves confusion_matrix.png + confusion_matrix_normalized.png
        save_dir=str(eval_dir.resolve()),
        exist_ok=True,
        verbose=False,
    )
    return metrics


# ── Metric logging ─────────────────────────────────────────────────────────────
def log_overall_metrics(metrics: DetMetrics) -> None:
    """Print mAP50 and mAP50-95 overall."""
    box = metrics.box
    log.info("── Overall metrics ──────────────────────────────")
    log.info("  mAP50    : %.4f", box.map50)
    log.info("  mAP50-95 : %.4f", box.map)
    log.info("  Precision: %.4f", box.mp)
    log.info("  Recall   : %.4f", box.mr)


def log_per_class_metrics(metrics: DetMetrics) -> None:
    """Print Precision, Recall, AP50, AP50-95 for each detected class."""
    box = metrics.box
    ap50_arr = np.asarray(box.ap50)
    ap_arr   = np.asarray(box.ap)
    p_arr    = np.asarray(box.p)
    r_arr    = np.asarray(box.r)
    class_idx = np.asarray(box.ap_class_index, dtype=int)

    log.info("── Per-class metrics ────────────────────────────")
    header = f"  {'Class':<12} {'P':>7} {'R':>7} {'AP50':>7} {'AP50-95':>9}"
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))

    for i, cls_id in enumerate(class_idx):
        name = CLASS_NAMES.get(int(cls_id), f"cls_{cls_id}")
        log.info(
            "  %-12s %7.4f %7.4f %7.4f %9.4f",
            name, p_arr[i], r_arr[i], ap50_arr[i], ap_arr[i],
        )


def log_confusion_matrix_path(metrics: DetMetrics) -> None:
    """Log the path of the saved confusion matrix PNG (uses metrics.save_dir)."""
    # ultralytics sets metrics.save_dir in finalize_metrics()
    save_dir: Path | None = getattr(metrics, "save_dir", None)
    if save_dir is None:
        log.warning("metrics.save_dir not set — confusion matrix path unknown.")
        return
    cm_path = Path(save_dir) / "confusion_matrix_normalized.png"
    if cm_path.exists():
        log.info("Confusion matrix saved: %s", cm_path)
    else:
        log.warning("Confusion matrix PNG not found at %s", cm_path)


# ── Sample predictions ─────────────────────────────────────────────────────────
def save_sample_predictions(
    model: YOLO,
    test_dir: Path,
    preds_dir: Path,
    n: int,
    seed: int,
) -> None:
    """Run inference on n random test images and save annotated results as JPEG."""
    preds_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(test_dir.glob("*"))
    if not all_images:
        log.warning("No test images found in %s", test_dir)
        return

    rng = random.Random(seed)
    samples = rng.sample(all_images, min(n, len(all_images)))
    log.info("Saving %d sample predictions to %s …", len(samples), preds_dir)

    for img_path in samples:
        out_path = preds_dir / img_path.name

        results = model.predict(
            source=str(img_path),
            device=DEVICE,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        # results[0].plot() returns a BGR numpy array
        annotated_bgr: np.ndarray = results[0].plot(
            labels=True,
            boxes=True,
            conf=True,
        )
        # Convert BGR → RGB for PIL
        annotated_rgb = annotated_bgr[:, :, ::-1]
        Image.fromarray(annotated_rgb).save(out_path, quality=95)

        n_det = len(results[0].boxes) if results[0].boxes is not None else 0
        log.info("  %s  →  %d detection(s)", img_path.name, n_det)


# ── Entry point ────────────────────────────────────────────────────────────────
def main(weights: Path, n_samples: int) -> None:
    if not weights.exists():
        log.error("Model weights not found: %s", weights)
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading model: %s", weights)
    model = YOLO(str(weights))

    # ── Validation + confusion matrix ─────────────────────────────────────────
    metrics = run_validation(model, weights, EVAL_DIR)
    log_overall_metrics(metrics)
    log_per_class_metrics(metrics)
    log_confusion_matrix_path(metrics)

    # ── Sample predictions ────────────────────────────────────────────────────
    save_sample_predictions(model, TEST_IMAGES_DIR, PREDS_DIR, n_samples, SEED)

    log.info("Evaluation complete. Results in: %s", EVAL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8n road-damage model on the test split."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to model weights (default: models/best.pt).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help="Number of random test images to annotate (default: 5).",
    )
    args = parser.parse_args()
    main(weights=args.weights, n_samples=args.n_samples)
