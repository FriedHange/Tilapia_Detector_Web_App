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


def parse_yolo_label_text(text: str, img_w: int, img_h: int) -> list[list[float]]:
    """
    Parse YOLO-format text content (class cx cy w h normalized) into pixel boxes [x1, y1, x2, y2].
    """
    boxes = []
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)])
        except (ValueError, IndexError):
            continue
    return boxes


def compute_academic_metrics(
    pred_boxes: list[list[float]],
    gt_boxes: Optional[list[list[float]]] = None,
    gt_count: Optional[int] = None,
    iou_thresh: float = 0.5,
) -> dict:
    """
    Pure Python/NumPy logic to match predicted bounding boxes against ground-truth boxes
    (Option B) or scalar count (Option A).
    Returns TP, FP, FN, Precision, Recall, F1, MAE, MAPE, Accuracy %, and Confusion Matrix.
    """
    pred_count = len(pred_boxes)

    if gt_boxes is not None:
        # Option B: Annotated bounding boxes available
        actual_count = len(gt_boxes)
        tp, fp, fn = _match_boxes(pred_boxes, gt_boxes, iou_thresh=iou_thresh)
        eval_mode = "yolo_annotations"
    elif gt_count is not None:
        # Option A: Quick Ground Truth Count (scalar count)
        actual_count = max(0, int(gt_count))
        tp = min(pred_count, actual_count)
        fp = max(0, pred_count - actual_count)
        fn = max(0, actual_count - pred_count)
        eval_mode = "ground_truth_count"
    else:
        actual_count = pred_count
        tp = pred_count
        fp = 0
        fn = 0
        eval_mode = "unsupervised"

    # Academic metrics
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    mae = float(abs(pred_count - actual_count))
    mape = float((mae / max(1, actual_count)) * 100.0)
    accuracy_pct = round(max(0.0, 100.0 - mape), 2)

    confusion_matrix = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": 0,
    }

    return {
        "evaluation_mode": eval_mode,
        "actual_count": int(actual_count),
        "predicted_count": int(pred_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "accuracy_pct": accuracy_pct,
        "confusion_matrix": confusion_matrix,
    }


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
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist on disk.")

        # Flexible directory detection: check for images/ subfolder or direct images
        if (dataset_path / "images").is_dir():
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"
        elif any(dataset_path.glob("*.jpg")) or any(dataset_path.glob("*.png")):
            images_dir = dataset_path
            labels_dir = dataset_path / "labels" if (dataset_path / "labels").is_dir() else dataset_path
        else:
            raise ValueError(
                f"No image files (.jpg, .png, etc.) found in '{dataset_path}'. "
                "Ensure the folder contains images directly or has an 'images/' subfolder."
            )

        image_paths = sorted(
            [p for p in images_dir.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )
        if not image_paths:
            raise ValueError(f"No image files found in {images_dir}")

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

    def evaluate_single_frame(
        self,
        frame: np.ndarray,
        model_names: list[str],
        conf: float = 0.25,
        gt_count: Optional[int] = None,
    ) -> dict:
        """
        Benchmark multiple models against a single image or video frame.
        Returns per-model comparison metrics (count, avg_conf, latency_ms, boxes, accuracy).
        """
        models_res = []
        counts = []
        for name in model_names:
            if name not in self._pool:
                continue
            model = self._pool[name]
            t0 = time.perf_counter()
            res = model(frame, verbose=False, conf=conf)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0

            r = res[0]
            boxes = []
            confs = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cf = float(b.conf[0])
                    boxes.append([x1, y1, x2, y2])
                    confs.append(cf)

            cnt = len(boxes)
            counts.append(cnt)
            avg_cf = (sum(confs) / len(confs)) if confs else 0.0

            item = {
                "model_name": name,
                "count": cnt,
                "avg_conf": round(avg_cf, 3),
                "inference_ms": round(lat_ms, 1),
                "boxes": boxes,
            }
            if gt_count is not None:
                err = abs(cnt - gt_count)
                item["abs_error"] = err
                item["accuracy_pct"] = round(max(0.0, 100.0 - (err / max(1, gt_count) * 100.0)), 1)
            models_res.append(item)

        consensus_count = int(np.median(counts)) if counts else 0
        for m in models_res:
            m["consensus_diff"] = m["count"] - consensus_count

        return {
            "consensus_count": consensus_count,
            "models": models_res,
            "ground_truth_count": gt_count,
        }

    def evaluate_sample(
        self,
        frame: np.ndarray,
        model_names: list[str],
        conf: float = 0.5,
        iou_thresh: float = 0.5,
        gt_count: Optional[int] = None,
        gt_boxes: Optional[list[list[float]]] = None,
        source_name: str = "sample",
    ) -> dict:
        """
        Evaluate one or more models on a single frame with complete academic metrics.
        Returns per-model evaluations, consensus comparison, and confusion matrix.
        """
        h, w = frame.shape[:2]
        models_res = []
        counts = []

        for name in model_names:
            if name not in self._pool:
                continue
            model = self._pool[name]
            t0 = time.perf_counter()
            res = model(frame, verbose=False, conf=conf, iou=iou_thresh)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0

            r = res[0]
            boxes = []
            confs = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cf = float(b.conf[0])
                    boxes.append([x1, y1, x2, y2])
                    confs.append(cf)

            cnt = len(boxes)
            counts.append(cnt)
            avg_cf = (sum(confs) / len(confs)) if confs else 0.0

            metrics = compute_academic_metrics(
                pred_boxes=boxes,
                gt_boxes=gt_boxes,
                gt_count=gt_count,
                iou_thresh=iou_thresh,
            )

            metrics.update({
                "model_name": name,
                "confidence_threshold": conf,
                "iou_threshold": iou_thresh,
                "avg_confidence": round(avg_cf, 3),
                "inference_ms": round(lat_ms, 1),
                "boxes": boxes,
                "source_name": source_name,
            })
            models_res.append(metrics)

        consensus_count = int(np.median(counts)) if counts else 0
        for m in models_res:
            m["consensus_diff"] = m["predicted_count"] - consensus_count

        return {
            "source_name": source_name,
            "dimensions": {"width": w, "height": h},
            "consensus_count": consensus_count,
            "models": models_res,
            "gt_count": gt_count if gt_count is not None else (len(gt_boxes) if gt_boxes is not None else None),
            "has_gt_boxes": gt_boxes is not None and len(gt_boxes) > 0,
        }

