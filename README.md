# Road Damage Detection

YOLOv8-based object detection for road damage (potholes, cracks, manholes).

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Workflow

**1. Prepare dataset**
```bash
python src/prepare_dataset.py
```

**2. Train**
```bash
python src/train.py
```

**3. Evaluate**
```bash
python src/evaluate.py --weights models/best.pt --save-plot metrics.png
```

**4. Demo**
```bash
python app.py
```

## Classes

| ID | Name    |
|----|---------|
| 0  | Pothole |
| 1  | Crack   |
| 2  | Manhole |
