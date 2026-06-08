"""
YOLOv8n training with explicit MLflow tracking.

Usage:
    python src/train.py
    python src/train.py --epochs 10 --batch 8   # quick smoke-test
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# ── Config ─────────────────────────────────────────────────────────────────────
TRACKING_URI = "sqlite:///mlruns/mlflow.db"  # MLflow 3.x requires DB backend; mlruns/ is git-ignored
EXPERIMENT_NAME = "road-damage-detection"

MODEL_WEIGHTS = "yolov8n.pt"
DATASET_YAML = "dataset.yaml"
MODELS_DIR = Path("models")
BEST_PT = MODELS_DIR / "best.pt"

DEFAULT_TRAIN_PARAMS: dict[str, Any] = {
    "data": DATASET_YAML,
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "device": "mps",
    "project": "runs/train",
    "name": "yolov8n_road_damage",
    "exist_ok": True,
    "patience": 10,
    "verbose": False,   # suppress ultralytics wall-of-text; we log ourselves
}

CLASS_NAMES: dict[int, str] = {0: "Pothole", 1: "Crack", 2: "Manhole"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _clean_keys(d: dict[str, Any]) -> dict[str, float]:
    """Strip parentheses from metric names and cast values to float."""
    return {
        k.replace("(", "").replace(")", ""): float(v)
        for k, v in d.items()
        if v is not None
    }


# ── MLflow callbacks ───────────────────────────────────────────────────────────
def _make_callbacks() -> dict[str, Any]:
    """
    Return YOLO trainer callbacks that log to the currently active MLflow run.
    The run must be started before model.train() is called.
    """

    def on_pretrain_routine_end(trainer: BaseTrainer) -> None:
        params = {k: str(v) for k, v in vars(trainer.args).items()}
        mlflow.log_params(params)
        log.info("MLflow params logged (%d keys).", len(params))

    def on_train_epoch_end(trainer: BaseTrainer) -> None:
        step = trainer.epoch
        metrics: dict[str, float] = {}
        if trainer.tloss is not None:
            loss_items = trainer.label_loss_items(trainer.tloss, prefix="train")
            metrics.update(_clean_keys(loss_items))
        metrics.update(_clean_keys(trainer.lr))
        mlflow.log_metrics(metrics, step=step)

    def on_fit_epoch_end(trainer: BaseTrainer) -> None:
        step = trainer.epoch
        val_metrics = _clean_keys(trainer.metrics)
        mlflow.log_metrics(val_metrics, step=step)
        log.info(
            "Epoch %3d  mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
            step + 1,
            val_metrics.get("metrics/mAP50B", 0.0),
            val_metrics.get("metrics/mAP50-95B", 0.0),
            val_metrics.get("metrics/precisionB", 0.0),
            val_metrics.get("metrics/recallB", 0.0),
        )

    return {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
    }


# ── Per-class metrics ──────────────────────────────────────────────────────────
def log_per_class_metrics(model: YOLO, train_params: dict[str, Any]) -> None:
    """Run model.val() on the test split and log per-class AP/P/R to MLflow."""
    log.info("Running validation for per-class metrics …")
    val = model.val(
        data=train_params["data"],
        imgsz=train_params["imgsz"],
        device=train_params["device"],
        split="val",
        verbose=False,
    )
    box = val.box

    # Overall aggregates
    overall: dict[str, float] = {
        "final/mAP50":     float(box.map50),
        "final/mAP50-95":  float(box.map),
        "final/precision": float(box.mp),
        "final/recall":    float(box.mr),
    }
    mlflow.log_metrics(overall)
    log.info(
        "Final  mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
        box.map50, box.map, box.mp, box.mr,
    )

    # Per-class
    ap50_arr: np.ndarray = np.array(box.ap50)
    ap_arr:   np.ndarray = np.array(box.ap)
    p_arr:    np.ndarray = np.array(box.p)
    r_arr:    np.ndarray = np.array(box.r)
    class_idx: np.ndarray = np.array(box.ap_class_index, dtype=int)

    per_class: dict[str, float] = {}
    for i, cls_id in enumerate(class_idx):
        name = CLASS_NAMES.get(int(cls_id), f"cls_{cls_id}")
        per_class[f"cls/{name}/AP50"]     = float(ap50_arr[i])
        per_class[f"cls/{name}/AP50-95"]  = float(ap_arr[i])
        per_class[f"cls/{name}/precision"] = float(p_arr[i])
        per_class[f"cls/{name}/recall"]    = float(r_arr[i])
        log.info(
            "  %-10s  AP50=%.4f  AP50-95=%.4f  P=%.4f  R=%.4f",
            name, ap50_arr[i], ap_arr[i], p_arr[i], r_arr[i],
        )

    mlflow.log_metrics(per_class)


# ── Entry point ────────────────────────────────────────────────────────────────
def main(train_params: dict[str, Any] | None = None) -> None:
    params = {**DEFAULT_TRAIN_PARAMS, **(train_params or {})}

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("MLflow tracking URI: %s  experiment: %s", TRACKING_URI, EXPERIMENT_NAME)

    model = YOLO(MODEL_WEIGHTS)
    callbacks = _make_callbacks()
    for event, fn in callbacks.items():
        model.add_callback(event, fn)

    run_name = params["name"]
    log.info("Starting training run '%s' …", run_name)

    with mlflow.start_run(run_name=run_name) as run:
        log.info("Run ID: %s", run.info.run_id)

        # ── Training ──────────────────────────────────────────────────────────
        results = model.train(**params)

        if results is None:
            log.error("Training returned None — check dataset and parameters.")
            return

        # ── Final metrics + per-class ──────────────────────────────────────────
        log_per_class_metrics(model, params)

        # ── Artifact: weights directory ────────────────────────────────────────
        weights_dir = Path(params["project"]) / params["name"] / "weights"
        best_src = weights_dir / "best.pt"

        if best_src.exists():
            mlflow.log_artifact(str(best_src), artifact_path="weights")
            log.info("Logged artifact: %s", best_src)
        else:
            log.warning("best.pt not found at %s — artifact not logged.", best_src)

        # ── Copy best.pt → models/best.pt ─────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if best_src.exists():
            shutil.copy2(best_src, BEST_PT)
            log.info("Copied %s → %s", best_src, BEST_PT)
        else:
            log.warning("Could not copy best.pt: source not found.")

    log.info("MLflow run finished. View with: mlflow ui --backend-store-uri %s", TRACKING_URI)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n for road damage detection.")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--imgsz",   type=int,   default=None)
    parser.add_argument("--device",  type=str,   default=None)
    parser.add_argument("--patience",type=int,   default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    main(train_params=overrides)
