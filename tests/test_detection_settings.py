"""Regression checks for detection threshold changes across inference paths."""

import asyncio
import json
import unittest
from unittest.mock import patch

import numpy as np

import app
from tracker import FingerlngTracker


class FakeBoxes:
    def __init__(self, detections):
        self.xyxy = np.array([box for box, _ in detections], dtype=float).reshape(-1, 4)
        self.conf = np.array([score for _, score in detections], dtype=float)
        self.cls = np.zeros(len(detections), dtype=int)
        self.id = None

    def __len__(self):
        return len(self.conf)


class FakeModel:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        result = type("Result", (), {})()
        result.names = {0: "Tilapia-Fingerlings"}
        result.boxes = FakeBoxes([(box, score) for box, score in self.detections
                                  if score >= kwargs["conf"]])
        return [result]


class DetectionSettingsTests(unittest.TestCase):
    def setUp(self):
        self.saved = (app.state.model_pool, app.state.active_models, app.state.conf,
                      app.state.iou, app.state.cached_frame, app.state.cached_filename)
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)

    def tearDown(self):
        (app.state.model_pool, app.state.active_models, app.state.conf,
         app.state.iou, app.state.cached_frame, app.state.cached_filename) = self.saved

    def test_confidence_changes_count_and_reaches_model(self):
        model = FakeModel([([0, 0, 10, 10], 0.3), ([15, 15, 25, 25], 0.9)])
        app.state.model_pool = {"fish": model}
        app.state.active_models = ["fish"]
        self.assertEqual(len(app._infer_frame(self.frame, False, 0.2, 0.4)[1]), 2)
        self.assertEqual(len(app._infer_frame(self.frame, False, 0.8, 0.7)[1]), 1)
        self.assertEqual([call["conf"] for call in model.calls], [0.2, 0.8])
        self.assertEqual([call["iou"] for call in model.calls], [0.4, 0.7])

    def test_iou_changes_cross_model_deduplication(self):
        first = FakeModel([([0, 0, 10, 10], 0.9)])
        second = FakeModel([([1, 1, 11, 11], 0.8)])
        app.state.model_pool = {"a": first, "b": second}
        app.state.active_models = ["a", "b"]
        self.assertEqual(len(app._infer_frame(self.frame, False, 0.2, 0.5)[1]), 1)
        self.assertEqual(len(app._infer_frame(self.frame, False, 0.2, 0.9)[1]), 2)

    def test_cached_stream_result_invalidates_when_settings_change(self):
        model = FakeModel([([0, 0, 10, 10], 0.3), ([15, 15, 25, 25], 0.9)])
        app.state.model_pool = {"fish": model}
        app.state.active_models = ["fish"]
        tracker = FingerlngTracker()
        with patch.object(app, "_annotate_frame", side_effect=lambda frame, *args, **kwargs: frame), \
             patch.object(app, "_frame_to_b64", return_value="frame"):
            _, _, boxes, cached = app._process_frame_worker(
                self.frame, True, None, tracker, 0.2, 0.5, ["fish"])
            self.assertEqual(len(boxes), 2)
            self.assertTrue(all(box["track_id"] is not None and box["trail"] for box in boxes))
            _, _, boxes, cached = app._process_frame_worker(
                self.frame, False, cached, tracker, 0.8, 0.5, ["fish"])
            self.assertEqual(len(boxes), 1)
            self.assertEqual(len(model.calls), 2)

    def test_cached_image_reprocess_uses_new_threshold(self):
        app.state.model_pool = {"fish": FakeModel([([0, 0, 10, 10], 0.3)])}
        app.state.active_models = ["fish"]
        app.state.cached_frame = self.frame
        app.state.cached_filename = "sample.png"
        with patch.object(app, "_annotate_frame", side_effect=lambda frame, *args, **kwargs: frame), \
             patch.object(app, "_frame_to_b64", return_value="frame"):
            low = asyncio.run(app.reprocess_current({"conf": 0.2, "iou": 0.5}))
            high = asyncio.run(app.reprocess_current({"conf": 0.8, "iou": 0.5}))
        self.assertEqual(json.loads(low.body)["count"], 1)
        self.assertEqual(json.loads(high.body)["count"], 0)


if __name__ == "__main__":
    unittest.main()
