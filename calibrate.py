"""Calibrate locked counting settings on COCO validation labels.

The test split is evaluated only after settings have been selected. Predictions
at the lowest candidate confidence can be filtered for higher confidence values:
suppression is score ordered, so removing lower scoring boxes cannot suppress a
retained box. Selected settings are confirmed with fresh native inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from detection_profile import DETECTORS, PROFILE_PATH
from tracker import nms_dedup


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def coco_split(directory):
    directory = Path(directory)
    annotation_file = directory / "_annotations.coco.json"
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {c["id"] for c in data["categories"] if "tilapia" in c["name"].lower() or "fingerling" in c["name"].lower()}
    boxes = {}
    for annotation in data["annotations"]:
        if annotation["category_id"] not in categories or annotation.get("iscrowd"):
            continue
        x, y, w, h = annotation["bbox"]
        if w <= 0 or h <= 0:
            raise ValueError("COCO annotations contain a zero or negative box size.")
        boxes.setdefault(annotation["image_id"], []).append([x, y, x + w, y + h])
    images = []
    for image in data["images"]:
        path = directory / image["file_name"]
        if not path.is_file():
            raise ValueError(f"Missing labeled image: {path.name}")
        images.append({"path": path, "boxes": boxes.get(image["id"], []), "width": image["width"], "height": image["height"]})
    if not images:
        raise ValueError("No labeled images in this split.")
    return images, file_hash(annotation_file)


def match_count(predictions, truth):
    if not predictions or not truth:
        return 0
    pred = np.asarray([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in predictions])
    gt = np.asarray(truth)
    top_left = np.maximum(pred[:, None, :2], gt[None, :, :2])
    bottom_right = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    sizes = np.maximum(bottom_right - top_left, 0)
    intersection = sizes[:, :, 0] * sizes[:, :, 1]
    pred_area = np.prod(np.maximum(pred[:, 2:] - pred[:, :2], 0), axis=1)
    gt_area = np.prod(np.maximum(gt[:, 2:] - gt[:, :2], 0), axis=1)
    overlaps = intersection / np.maximum(pred_area[:, None] + gt_area[None, :] - intersection, 1e-9)
    used = np.zeros(len(gt), dtype=bool)
    matches = 0
    for values in overlaps:
        values[used] = -1
        index = int(np.argmax(values))
        if values[index] >= .5:
            matches += 1
            used[index] = True
    return matches


def score(predictions, images):
    errors, tp, fp, fn = [], 0, 0, 0
    for detected, image in zip(predictions, images):
        matched = match_count(detected, image["boxes"])
        errors.append(abs(len(detected) - len(image["boxes"])))
        tp += matched
        fp += len(detected) - matched
        fn += len(image["boxes"]) - matched
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"mae": round(sum(errors) / len(images), 6),
            "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0,
            "precision": round(precision, 6), "recall": round(recall, 6), "images": len(images),
            "tp": tp, "fp": fp, "fn": fn}


def predictions_for(model, images, conf, iou, device):
    results, latencies = [], []
    for image in images:
        frame = cv2.imread(str(image["path"]))
        start = time.perf_counter()
        result = model(frame, verbose=False, conf=conf, iou=iou, device=device, imgsz=640, max_det=1000)[0]
        if device != "cpu":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        boxes = []
        if result.boxes is not None:
            coordinates = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for coords, confidence, category in zip(coordinates, confidences, classes):
                class_name = str(result.names[int(category)]).lower()
                if "tilapia" not in class_name and "fingerling" not in class_name:
                    continue
                x1, y1, x2, y2 = map(float, coords)
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(confidence), "class_id": 0})
        results.append(boxes)
    return results, round(float(np.median(latencies)), 3)


def groups(images):
    return {re.sub(r"-\d+_jpg.*", "", image["path"].name) for image in images}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROFILE_PATH)
    args = parser.parse_args()
    valid, valid_hash = coco_split(args.dataset / "valid")
    test, test_hash = coco_split(args.dataset / "test")
    train, train_hash = coco_split(args.dataset / "train")
    device = 0 if torch.cuda.is_available() else "cpu"
    confs = [round(i * .05, 2) for i in range(1, 20)]
    ious = [round(.3 + i * .1, 2) for i in range(7)]
    profile = {"calibrated": True, "dataset": str(args.dataset), "annotation_hashes": {"valid": valid_hash, "test": test_hash, "train": train_hash},
        "matching_iou": .5, "max_det": 1000, "imgsz": 640, "models": {}, "ensemble_iou": .5,
        "warnings": ["All validation and test video sources overlap training video sources; results are dataset-specific, not independent video generalization."],
        "source_overlap": {"valid_train": len(groups(valid) & groups(train)), "test_train": len(groups(test) & groups(train))}}
    selected_valid, selected_test, timings = {}, {}, {}
    cache_root = Path(__file__).parent / "calibration_cache"
    cache_root.mkdir(exist_ok=True)
    for name in DETECTORS:
        weight = Path(__file__).parent / "models" / (name + ".pt")
        weight_hash = file_hash(weight)
        model = YOLO(str(weight))
        if not any("tilapia" in str(v).lower() or "fingerling" in str(v).lower() for v in model.names.values()):
            raise ValueError(f"Bundled weight {name} does not contain a fingerling class.")
        print(f"Calibrating {name} on {len(valid)} validation images", flush=True)
        model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device)
        best, best_predictions = None, None
        candidates = []
        for iou in ious:
            cache = cache_root / f"{name}-{weight_hash[:12]}-{valid_hash[:12]}-{iou}.json"
            if cache.exists():
                stored = json.loads(cache.read_text())
                detected, latency = stored["predictions"], stored["latency_ms"]
            else:
                detected, latency = predictions_for(model, valid, confs[0], iou, device)
                cache.write_text(json.dumps({"predictions": detected, "latency_ms": latency}))
            for conf in confs:
                filtered = [[box for box in boxes if box["conf"] >= conf] for boxes in detected]
                metrics = score(filtered, valid)
                candidate = {"conf": conf, "iou": iou, **metrics, "latency_ms": latency}
                candidates.append(candidate)
                rank = (metrics["mae"], -metrics["f1"], latency, conf, iou)
                if best is None or rank < best[0]:
                    best, best_predictions = (rank, candidate), filtered
            print(f"  IoU {iou:.2f} complete; best MAE {best[1]['mae']:.3f}", flush=True)
        selected = best[1]
        verified, verified_latency = predictions_for(model, valid, selected["conf"], selected["iou"], device)
        verified_metrics = score(verified, valid)
        if verified_metrics["mae"] != selected["mae"]:
            raise ValueError("Cached confidence filtering does not match native inference; calibrate candidates directly.")
        test_predictions, test_latency = predictions_for(model, test, selected["conf"], selected["iou"], device)
        baseline, _ = predictions_for(model, valid, .5, .5, device)
        profile["models"][name] = {"conf": selected["conf"], "iou": selected["iou"], "weight_sha256": weight_hash,
            "validation": verified_metrics, "baseline_validation": score(baseline, valid), "test": score(test_predictions, test),
            "latency_ms": verified_latency, "test_latency_ms": test_latency,
            "top_candidates": sorted(candidates, key=lambda r: (r["mae"], -r["f1"], r["latency_ms"], r["conf"], r["iou"]))[:5]}
        selected_valid[name], selected_test[name], timings[name] = verified, test_predictions, verified_latency
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Selected {name}: conf={selected['conf']} IoU={selected['iou']}; validation={verified_metrics}; test={profile['models'][name]['test']}", flush=True)
    ensemble_candidates = []
    for iou in ious:
        fused = [nms_dedup([box for name in DETECTORS for box in selected_valid[name][i]], iou) for i in range(len(valid))]
        ensemble_candidates.append({"iou": iou, **score(fused, valid)})
    winner = min(ensemble_candidates, key=lambda r: (r["mae"], -r["f1"], r["iou"]))
    profile["ensemble_iou"] = winner["iou"]
    test_fused = [nms_dedup([box for name in DETECTORS for box in selected_test[name][i]], winner["iou"]) for i in range(len(test))]
    profile["ensemble"] = {"validation": winner, "test": score(test_fused, test), "median_detector_ms_sum": round(sum(timings.values()), 3),
        "candidates": ensemble_candidates}
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print("Saved locked profile:", args.output, flush=True)
    print("Ensemble:", profile["ensemble"], flush=True)


if __name__ == "__main__":
    main()
