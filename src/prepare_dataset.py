"""
Prepare road damage dataset: match → filter → stratified split (70/15/15) → copy.

Usage:
    python src/prepare_dataset.py
    python src/prepare_dataset.py --stats-only   # dry run, no file copying
"""

import argparse
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import TypeAlias

from sklearn.model_selection import train_test_split

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_IMAGES_DIR = Path("data/raw/data/images")
RAW_LABELS_DIR = Path("data/raw/data/labels-YOLO")
PROCESSED_DIR = Path("data/processed")

SEED = 42
VAL_RATIO = 0.15   # fraction of total
TEST_RATIO = 0.15  # fraction of total

CLASS_NAMES: dict[int, str] = {0: "Pothole", 1: "Crack", 2: "Manhole"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────────────────────
Pair: TypeAlias = tuple[Path, Path]  # (image_path, label_path)


# ── Core logic ─────────────────────────────────────────────────────────────────
def find_pairs() -> list[Pair]:
    """Match images to labels by filename stem; log and drop unmatched files."""
    images: dict[str, Path] = {
        p.stem: p for p in RAW_IMAGES_DIR.glob("*") if p.is_file()
    }
    labels: dict[str, Path] = {
        p.stem: p for p in RAW_LABELS_DIR.glob("*.txt")
    }

    matched_stems = sorted(set(images) & set(labels))
    only_images = set(images) - set(labels)
    only_labels = set(labels) - set(images)

    if only_images:
        log.warning("Images without matching label (skipped): %d", len(only_images))
    if only_labels:
        log.warning("Labels without matching image (skipped): %d", len(only_labels))

    log.info("Matched image/label pairs: %d", len(matched_stems))
    return [(images[s], labels[s]) for s in matched_stems]


def primary_class(label_path: Path) -> int:
    """Return the class ID from the first annotation line; -1 for empty files."""
    for line in label_path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            return int(stripped.split()[0])
    return -1  # empty label → own stratum so split still works


def stratified_split(
    pairs: list[Pair],
    strata: list[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Two-step stratified split: train / val / test."""
    temp_ratio = val_ratio + test_ratio

    train_pairs, temp_pairs, _, temp_strata = train_test_split(
        pairs,
        strata,
        test_size=temp_ratio,
        random_state=seed,
        stratify=strata,
    )

    # Within the held-out pool, split val vs test at the same ratio
    val_frac_of_temp = val_ratio / temp_ratio
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=1.0 - val_frac_of_temp,
        random_state=seed,
        stratify=temp_strata,
    )

    return list(train_pairs), list(val_pairs), list(test_pairs)


def copy_pairs(pairs: list[Pair], split: str) -> None:
    """Copy image+label into data/processed/<split>/{images,labels}/; skip existing."""
    img_dst = PROCESSED_DIR / split / "images"
    lbl_dst = PROCESSED_DIR / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for img_src, lbl_src in pairs:
        dst_img = img_dst / img_src.name
        dst_lbl = lbl_dst / lbl_src.name
        if dst_img.exists() and dst_lbl.exists():
            skipped += 1
            continue
        shutil.copy2(img_src, dst_img)
        shutil.copy2(lbl_src, dst_lbl)
        copied += 1

    log.info(
        "  [%s] copied=%d  skipped(already exist)=%d",
        split, copied, skipped,
    )


# ── Statistics ─────────────────────────────────────────────────────────────────
def annotation_counts(pairs: list[Pair]) -> Counter[str]:
    """Count all annotations by class name across a set of pairs."""
    counts: Counter[str] = Counter()
    for _, lbl in pairs:
        for line in lbl.read_text().splitlines():
            stripped = line.strip()
            if stripped:
                cls_id = int(stripped.split()[0])
                counts[CLASS_NAMES.get(cls_id, f"cls_{cls_id}")] += 1
    return counts


def print_stats(split_map: dict[str, list[Pair]]) -> None:
    """Log a table of image counts and per-class annotation counts per split."""
    class_ids = sorted(CLASS_NAMES)
    col_names = [CLASS_NAMES[i] for i in class_ids]
    header = f"{'Split':<8} {'Images':>7}  " + "  ".join(f"{n:>9}" for n in col_names)
    separator = "-" * len(header)

    log.info(separator)
    log.info(header)
    log.info(separator)
    for split_name, pairs in split_map.items():
        counts = annotation_counts(pairs)
        class_cols = "  ".join(f"{counts.get(CLASS_NAMES[i], 0):>9}" for i in class_ids)
        log.info("%-8s %7d  %s", split_name, len(pairs), class_cols)
    log.info(separator)


# ── Entry point ────────────────────────────────────────────────────────────────
def main(stats_only: bool = False) -> None:
    pairs = find_pairs()

    if not pairs:
        log.error("No matched pairs found — check RAW_IMAGES_DIR and RAW_LABELS_DIR.")
        return

    strata = [primary_class(lbl) for _, lbl in pairs]
    train_pairs, val_pairs, test_pairs = stratified_split(
        pairs, strata, VAL_RATIO, TEST_RATIO, SEED
    )

    split_map: dict[str, list[Pair]] = {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
    }

    log.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_pairs), len(val_pairs), len(test_pairs),
    )

    if stats_only:
        log.info("--stats-only: skipping file copy.")
    else:
        for split_name, split_pairs in split_map.items():
            copy_pairs(split_pairs, split_name)

    log.info("Split statistics (annotation counts per class):")
    print_stats(split_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare road damage dataset: stratified train/val/test split."
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print split statistics without copying any files (dry run).",
    )
    args = parser.parse_args()
    main(stats_only=args.stats_only)
