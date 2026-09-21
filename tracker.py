"""
tracker.py — ByteTrack-based Multi-Object Tracker with Virtual Counting Line
=============================================================================
Wraps the Ultralytics built-in ByteTrack tracker to:
  1. Maintain per-track centroid history (visual trails).
  2. Detect line-crossing events (virtual tripwire) using a vector
     cross-product sign-change test.
  3. Deduplicate counts — each track ID is counted at most ONCE per direction.
  4. Return enriched detection dicts ready for WebSocket broadcast and DB logging.

The counting line is defined by two points (x1,y1)→(x2,y2) in **frame pixel
coordinates**.  Fish crossing the line from left→right increments `count_in`,
right→left increments `count_out`.  The definition of "left" vs "right" is
relative to the directed line vector.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cross(ax, ay, bx, by) -> float:
    """2-D cross product of vectors A and B."""
    return ax * by - ay * bx


def _point_side(lx1, ly1, lx2, ly2, px, py) -> float:
    """
    Returns the signed area (cross product) that indicates which side of the
    directed line (lx1,ly1)→(lx2,ly2) the point (px,py) is on.
    Positive = left side, Negative = right side, Zero = on line.
    """
    return _cross(lx2 - lx1, ly2 - ly1, px - lx1, py - ly1)


# ---------------------------------------------------------------------------
# FingerlngTracker
# ---------------------------------------------------------------------------

class FingerlngTracker:
    """
    Stateful tracker combining Ultralytics ByteTrack output with:
      - centroid trail history per track
      - virtual counting line logic
      - cumulative in/out counters
    """

    def __init__(
        self,
        trail_length: int = 30,
        line_pt1: tuple[float, float] = (0.5, 0.0),  # relative [0,1]
        line_pt2: tuple[float, float] = (0.5, 1.0),
    ):
        """
        Parameters
        ----------
        trail_length : int
            Number of historical centroids to keep per track (visual trail).
        line_pt1, line_pt2 : tuple[float,float]
            The counting line endpoints as **relative** coordinates [0,1].
            They are converted to pixel coords on each update call using the
            current frame dimensions.  Set via `set_line()`.
        """
        self._trail_length = trail_length

        # Relative line definition (updated by dashboard drag)
        self._line_rel: tuple[tuple, tuple] = (line_pt1, line_pt2)

        # Per-track state
        # trails: {track_id: deque of (cx, cy) pixel coords}
        self._trails: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self._trail_length)
        )
        # Last known side of the counting line per track
        self._last_side: dict[int, float] = {}

        # Track IDs that have already been counted (deduplicate)
        self._counted_in: set[int] = set()
        self._counted_out: set[int] = set()

        # Cumulative session counters
        self.count_in: int = 0
        self.count_out: int = 0

        # Active tracks seen in last update (pruned on update)
        self._active_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_line(self, rel_x1: float, rel_y1: float,
                 rel_x2: float, rel_y2: float):
        """
        Update the counting line from relative [0,1] coordinates.
        Called when the user drags the line on the dashboard.
        """
        self._line_rel = ((rel_x1, rel_y1), (rel_x2, rel_y2))

    def get_line_pixels(self, frame_w: int, frame_h: int
                        ) -> tuple[tuple, tuple]:
        """Convert relative line coords to frame pixel coords."""
        (rx1, ry1), (rx2, ry2) = self._line_rel
        return (
            (int(rx1 * frame_w), int(ry1 * frame_h)),
            (int(rx2 * frame_w), int(ry2 * frame_h)),
        )

    def reset_counts(self):
        """Reset cumulative counters and per-track dedup sets (new session)."""
        self.count_in = 0
        self.count_out = 0
        self._counted_in.clear()
        self._counted_out.clear()
        self._trails.clear()
        self._last_side.clear()
        self._active_ids.clear()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        results,   # Ultralytics Results object (single frame)
        frame_w: int,
        frame_h: int,
        conf_thresh: float = 0.25,
    ) -> list[dict]:
        """
        Process one frame's Ultralytics results and return enriched detections.

        Parameters
        ----------
        results : ultralytics.engine.results.Results
            The first element of model(frame) output.
        frame_w, frame_h : int
            Pixel dimensions of the current frame (for line conversion).
        conf_thresh : float
            Minimum confidence to include a detection.

        Returns
        -------
        list[dict] with keys:
            track_id, class_id, class_name, conf,
            x1, y1, x2, y2,          # pixel coords
            cx, cy,                    # centroid pixels
            trail,                     # list of (cx,cy) recent history
            crossed_in, crossed_out,   # bool — crossed THIS frame?
        """
        # Pixel coords of counting line endpoints
        (lx1, ly1), (lx2, ly2) = self.get_line_pixels(frame_w, frame_h)

        detections: list[dict] = []
        current_ids: set[int] = set()

        r = results
        if r.boxes is None or len(r.boxes) == 0:
            self._prune_stale_tracks(current_ids)
            return detections

        boxes = r.boxes
        names: dict = r.names if isinstance(r.names, dict) else {}

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < conf_thresh:
                continue

            # Bounding box
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i])
            class_name = names.get(cls_id, str(cls_id))

            # Track ID (ByteTrack assigns .id; may be None on first frame)
            tid_tensor = boxes.id
            if tid_tensor is not None:
                track_id = int(tid_tensor[i])
            else:
                # No tracker — use a synthetic ID from bounding box hash
                track_id = hash((round(x1, 1), round(y1, 1))) & 0xFFFF

            # Centroid
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Update trail
            self._trails[track_id].append((cx, cy))
            current_ids.add(track_id)

            # ---- Counting line crossing detection ----
            crossed_in = False
            crossed_out = False

            side = _point_side(lx1, ly1, lx2, ly2, cx, cy)
            prev_side = self._last_side.get(track_id)

            if prev_side is not None and prev_side != 0:
                # Sign change indicates crossing
                if prev_side > 0 and side <= 0:
                    # Crossed from left side → right side = "in"
                    if track_id not in self._counted_in:
                        self.count_in += 1
                        self._counted_in.add(track_id)
                        crossed_in = True
                elif prev_side < 0 and side >= 0:
                    # Crossed from right side → left side = "out"
                    if track_id not in self._counted_out:
                        self.count_out += 1
                        self._counted_out.add(track_id)
                        crossed_out = True

            self._last_side[track_id] = side

            detections.append(
                {
                    "track_id": track_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "trail": list(self._trails[track_id]),
                    "crossed_in": crossed_in,
                    "crossed_out": crossed_out,
                }
            )

        self._prune_stale_tracks(current_ids)
        self._active_ids = current_ids
        return detections

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_stale_tracks(self, current_ids: set[int]):
        """Remove trail / side data for tracks no longer visible."""
        stale = set(self._trails.keys()) - current_ids
        for tid in stale:
            self._trails.pop(tid, None)
            self._last_side.pop(tid, None)


# ---------------------------------------------------------------------------
# NMS deduplication (preserved from original suite, used for multi-model runs)
# ---------------------------------------------------------------------------

def nms_dedup(boxes: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """
    Cross-model NMS deduplication.  When multiple YOLO models are run on the
    same frame, their outputs are merged and overlapping detections (IoU ≥
    iou_thresh) are collapsed to the highest-confidence detection.

    Parameters
    ----------
    boxes : list[dict]
        Each dict must have keys: conf, x1, y1, x2, y2.
    iou_thresh : float
        Boxes with IoU above this threshold are considered duplicates.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept: list[dict] = []
    for b in boxes:
        discard = False
        for k in kept:
            ix1 = max(b["x1"], k["x1"])
            iy1 = max(b["y1"], k["y1"])
            ix2 = min(b["x2"], k["x2"])
            iy2 = min(b["y2"], k["y2"])
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                continue
            area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            area_k = (k["x2"] - k["x1"]) * (k["y2"] - k["y1"])
            union = area_b + area_k - inter
            if union > 0 and inter / union >= iou_thresh:
                discard = True
                break
        if not discard:
            kept.append(b)
    return kept
