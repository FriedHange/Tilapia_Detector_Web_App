"""
evaluator.py — Academic Model Benchmark Module
===============================================
Evaluates YOLOv8, YOLOv9, and YOLOv10 on a labeled dataset and computes:
  - Precision, Recall, F1-score  (object-level matching, IoU ≥ 0.5)
  - MAE  — Mean Absolute Error of count  (|predicted_count - gt_count|)
  - MAPE — Mean Absolute Percentage Error of count

Dataset format accepted:
  A) YOLO-label directory: images/*.jpg (or .png) + labels/*.txt (class x y w h)
  B) Ground-truth CSV: columns [image_path, gt_count]   (count-only evaluation)

The evaluator is fully synchronous (run in a thread via asyncio.to_thread)
to avoid blocking the event loop during heavy inference.
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np

# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(b1: list[float], b2: list[float]) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter)


def _match_boxes(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
    iou_thresh: float = 0.5,
) -> tuple[int, int, int]:
    """
    Greedy matching (by highest IoU) of predicted vs ground-truth boxes.
    Returns (TP, FP, FN).
    """
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            v = _iou(pred, gt)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, max(fp, 0), max(fn, 0)


# ---------------------------------------------------------------------------
# Label parsers
# ---------------------------------------------------------------------------

def _parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """
    Parse a YOLO-format .txt label file and convert to pixel [x1,y1,x2,y2].
    Returns a list of bounding boxes (ignoring class id for counting purposes).
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Runs single-model or multi-model benchmarks on a labeled dataset.

    Usage
    -----
    evaluator = ModelEvaluator(model_pool)
    results = evaluator.run_benchmark(
        dataset_path="path/to/dataset",   # contains images/ and labels/ dirs
        model_names=["yolov8n", "yolov9c", "yolov10n"],
        conf=0.25,
        iou_thresh=0.50,
    )
    # results is a list of per-model dicts
    """

    def __init__(self, model_pool: dict):
        """
        Parameters
        ----------
        model_pool : dict
            {model_name: ultralytics.YOLO instance}
            The same pool maintained by app.py.
        """
        self._pool = model_pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        dataset_path: Union[str, Path],
        model_names: list[str],
        conf: float = 0.25,
        iou_thresh: float = 0.50,
        gt_csv: Optional[str] = None,
    ) -> list[dict]:
        """
        Evaluate each requested model on the dataset.

        Parameters
        ----------
        dataset_path : str | Path
            Root of dataset.  Must contain:
              - images/   (*.jpg, *.png, *.bmp, *.webp)
              - labels/   (*.txt, YOLO format) — required for bbox metrics.
            OR a CSV at `gt_csv` path with columns [image_path, gt_count].
        model_names : list[str]
            Keys into self._pool to evaluate.
        conf : float
            Inference confidence threshold.
        iou_thresh : float
            IoU threshold for TP matching.
        gt_csv : str | None
            If provided, use CSV for ground-truth counts (count-only mode;
            Precision/Recall/F1 will be None).

        Returns
        -------
        list[dict] — one entry per model with full metrics.
        """
        dataset_path = Path(dataset_path)

        if gt_csv:
            return self._eval_csv_mode(gt_csv, model_names, conf)

        return self._eval_yolo_mode(dataset_path, model_names, conf, iou_thresh)

    # ------------------------------------------------------------------
    # YOLO label mode (full Precision / Recall / F1 + MAE/MAPE)
    # ------------------------------------------------------------------

    def _eval_yolo_mode(
        self,
        dataset_path: Path,
        model_names: list[str],
        conf: float,
        iou_thresh: float,
    ) -> list[dict]:
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"images/ directory not found in {dataset_path}")

        image_paths = sorted(
            [p for p in images_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )
        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")

        results = []

        for model_name in model_names:
            if model_name not in self._pool:
                results.append(
                    {"model_name": model_name, "error": "Model not loaded"}
                )
                continue

            model = self._pool[model_name]
            total_tp = total_fp = total_fn = 0
            abs_errors: list[float] = []
            abs_pct_errors: list[float] = []
            per_image_details: list[dict] = []
            inference_times: list[float] = []

            for img_path in image_paths:
                # --- Ground truth ---
                label_path = labels_dir / (img_path.stem + ".txt")

                # Load image dims (needed for label conversion)
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                gt_boxes = _parse_yolo_label(label_path, img_w, img_h)
                gt_count = len(gt_boxes)

                # --- Inference ---
                t0 = time.perf_counter()
                inference_results = model(img, verbose=False, conf=conf)
                t1 = time.perf_counter()
                inference_times.append((t1 - t0) * 1000)  # ms

                r = inference_results[0]
                pred_boxes: list[list[float]] = []
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        pred_boxes.append([x1, y1, x2, y2])

                pred_count = len(pred_boxes)

                # --- Matching ---
                if gt_boxes:
                    tp, fp, fn = _match_boxes(pred_boxes, gt_boxes, iou_thresh)
                else:
                    # No ground truth → all preds are FP
                    tp, fp, fn = 0, pred_count, 0

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Counting error
                error = abs(pred_count - gt_count)
                abs_errors.append(error)
                if gt_count > 0:
                    abs_pct_errors.append(error / gt_count * 100.0)

                per_image_details.append(
                    {
                        "image": img_path.name,
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                        "tp": tp, "fp": fp, "fn": fn,
                        "abs_error": error,
                    }
                )

            # --- Aggregate metrics ---
            precision = (
                total_tp / (total_tp + total_fp)
                if (total_tp + total_fp) > 0 else 0.0
            )
            recall = (
                total_tp / (total_tp + total_fn)
                if (total_tp + total_fn) > 0 else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            mae = float(np.mean(abs_errors)) if abs_errors else 0.0
            mape = float(np.mean(abs_pct_errors)) if abs_pct_errors else 0.0
            avg_inf_ms = float(np.mean(inference_times)) if inference_times else 0.0

            results.append(
                {
                    "model_name": model_name,
                    "dataset_path": str(dataset_path),
                    "conf": conf,
                    "iou": iou_thresh,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "mae": round(mae, 4),
                    "mape": round(mape, 4),
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "total_images": len(image_paths),
                    "avg_inference_ms": round(avg_inf_ms, 2),
                    "details": per_image_details,
                }
            )

        return results

    # ------------------------------------------------------------------
    # CSV count-only mode (MAE / MAPE only)
    # ------------------------------------------------------------------

    def _eval_csv_mode(
        self,
        gt_csv: str,
        model_names: list[str],
        conf: float,
    ) -> list[dict]:
        """
        Evaluate using a CSV of ground-truth counts.
        CSV format: image_path (absolute or relative), gt_count
        """
        rows: list[dict] = []
        with open(gt_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "image_path": row.get("image_path", row.get("image", "")),
                        "gt_count": int(row.get("gt_count", row.get("count", 0))),
                    }
                )

        results = []
        for model_name in model_names:
            if model_name not in self._pool:
                results.append({"model_name": model_name, "error": "Model not loaded"})
                continue

            model = self._pool[model_name]
            abs_errors: list[float] = []
            abs_pct_errors: list[float] = []
            per_image: list[dict] = []

            import cv2
            for entry in rows:
                img = cv2.imread(entry["image_path"])
                if img is None:
                    continue
                r = model(img, verbose=False, conf=conf)[0]
                pred_count = len(r.boxes) if r.boxes is not None else 0
                gt_count = entry["gt_count"]
                error = abs(pred_count - gt_count)
                abs_errors.append(error)
                if gt_count > 0:
                    abs_pct_errors.append(error / gt_count * 100.0)
                per_image.append(
                    {
                        "image": os.path.basename(entry["image_path"]),
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                        "abs_error": error,
                    }
                )

            mae = float(np.mean(abs_errors)) if abs_errors else 0.0
            mape = float(np.mean(abs_pct_errors)) if abs_pct_errors else 0.0

            results.append(
                {
                    "model_name": model_name,
                    "dataset_path": gt_csv,
                    "conf": conf,
                    "iou": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "mae": round(mae, 4),
                    "mape": round(mape, 4),
                    "tp": None, "fp": None, "fn": None,
                    "total_images": len(rows),
                    "details": per_image,
                }
            )

        return results
