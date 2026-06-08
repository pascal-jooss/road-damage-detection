"""Gradio demo for road damage detection with YOLOv8n."""

import random
from pathlib import Path
from typing import TypeAlias

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/best.pt")
TEST_IMAGES_DIR = Path("data/processed/test/images")
DEVICE = "mps"
CONF_THRESHOLD = 0.25
N_EXAMPLES = 3
SEED = 42

CLASS_NAMES: dict[int, str] = {0: "Pothole", 1: "Crack", 2: "Manhole"}

TITLE = "🔍 Road Damage Detection"
DESCRIPTION = """
Automatische Erkennung von Straßenschäden mit **YOLOv8n**, trainiert auf eigenem Videomaterial
aus dem kommunalen Straßennetz.

**Erkannte Schadensklassen:** Pothole (Schlagloch) · Crack (Riss) · Manhole (Kanaldeckel)

**Einsatzbereich:** Straßenmeistereien und Kommunen können damit Schadensdokumentation,
Zustandserfassung und Priorisierung von Instandhaltungsmaßnahmen automatisieren –
ohne manuelle Begehung jeder einzelnen Strecke.
""".strip()

TABLE_COLUMNS = ["Klasse", "Konfidenz (%)", "X1", "Y1", "X2", "Y2"]
TABLE_DTYPES  = ["str", "number", "number", "number", "number", "number"]

# ── Type alias ─────────────────────────────────────────────────────────────────
DetectionTable: TypeAlias = pd.DataFrame


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(path: Path) -> YOLO:
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}\n"
            "Run `python src/train.py` first."
        )
    return YOLO(str(path))


# ── Inference ──────────────────────────────────────────────────────────────────
def _build_table(boxes) -> DetectionTable:
    """Convert ultralytics Boxes object to a pandas DataFrame."""
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    xyxy    = boxes.xyxy.cpu().numpy()          # shape (N, 4)
    confs   = boxes.conf.cpu().numpy()           # shape (N,)
    classes = boxes.cls.cpu().numpy().astype(int)  # shape (N,)

    rows = [
        {
            "Klasse":        CLASS_NAMES.get(cls_id, f"cls_{cls_id}"),
            "Konfidenz (%)": round(float(conf) * 100, 1),
            "X1":            int(x1),
            "Y1":            int(y1),
            "X2":            int(x2),
            "Y2":            int(y2),
        }
        for cls_id, conf, (x1, y1, x2, y2) in zip(classes, confs, xyxy)
    ]
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def predict(image: Image.Image) -> tuple[Image.Image, DetectionTable]:
    """Run YOLOv8 inference on a PIL image.

    Returns:
        annotated_image: PIL image with drawn bounding boxes.
        table: DataFrame with one row per detection.
    """
    results = model.predict(
        source=image,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        verbose=False,
    )
    result = results[0]

    # result.plot() returns a BGR numpy array → convert to RGB PIL Image
    annotated_bgr: np.ndarray = result.plot(labels=True, boxes=True, conf=True)
    annotated = Image.fromarray(annotated_bgr[:, :, ::-1])

    return annotated, _build_table(result.boxes)


# ── Example images ─────────────────────────────────────────────────────────────
def _pick_examples(n: int = N_EXAMPLES, seed: int = SEED) -> list[list[str]]:
    """Return n randomly selected test-image paths as Gradio example rows."""
    all_images = sorted(TEST_IMAGES_DIR.glob("*"))
    if not all_images:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(all_images, min(n, len(all_images)))
    return [[str(p)] for p in chosen]


# ── Gradio interface ───────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="pil",
        label="Straßenbild hochladen",
    ),
    outputs=[
        gr.Image(label="Erkannte Schäden"),
        gr.Dataframe(
            headers=TABLE_COLUMNS,
            datatype=TABLE_DTYPES,
            label="Detektionen",
            wrap=False,
        ),
    ],
    title=TITLE,
    description=DESCRIPTION,
    examples=_pick_examples(),
    cache_examples=False,   # don't run inference for every example at startup
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
