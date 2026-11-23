"""Ultralytics YOLO26 detector with a strict offline OpenCV fallback.

Weights are never downloaded implicitly. If the package, weights or model
initialisation are unavailable, the detector reports ``opencv_fallback`` with
an explicit reason and the rest of the pipeline continues unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DetectorConfig:
    """Configuration for YOLO inference (all fields are JSON-friendly)."""

    model: str = "yolo26n.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 320
    device: str = "cpu"
    weights_dir: str = "weights"
    allow_download: bool = False
    use_tracking: bool = False


@dataclass
class DetectorResult:
    """Normalised detections plus the mode/reason that produced them."""

    mode: str
    reason: str
    model: str
    detections: list[dict[str, Any]] = field(default_factory=list)
    zones: list[dict[str, Any]] = field(default_factory=list)
    frames_analyzed: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary for the report."""
        return {
            "mode": self.mode,
            "reason": self.reason,
            "model": self.model,
            "detections": len(self.detections),
            "zones": [str(zone.get("zone_id", "")) for zone in self.zones],
            "frames_analyzed": self.frames_analyzed,
            "latency_ms": self.latency_ms,
        }


def resolve_weights(model: str, weights_dir: str | Path = "weights") -> Path | None:
    """Find local weights only; never initiate a network download."""
    name = str(model)
    path = Path(name)
    if path.is_file():
        return path
    home = Path.home()
    candidates = [
        Path(weights_dir) / name,
        REPO_ROOT / name,
        REPO_ROOT / "weights" / name,
        home / ".cache" / "ultralytics" / name,
        Path(os.environ.get("ULTRALYTICS_HOME", str(home / "AppData" / "Local" / "Ultralytics"))) / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _to_numpy(value: Any) -> np.ndarray:
    """Normalise torch tensors, lists and arrays to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:  # pragma: no cover - exotic tensor types
            pass
    return np.asarray(value)


class YOLODetector:
    """Runs YOLO26 when locally available, otherwise Phase 1 OpenCV detection."""

    def __init__(self, config: DetectorConfig | None = None, model: Any = None) -> None:
        self.config = config or DetectorConfig()
        self._model = model
        self.mode = "yolo" if model is not None else "opencv_fallback"
        self.reason = "injected_test_model" if model is not None else "not_initialised"
        if model is not None:
            return
        self._initialise()

    def _initialise(self) -> None:
        """Load Ultralytics and local weights, degrading cleanly on failure."""
        weights = resolve_weights(self.config.model, self.config.weights_dir)
        if weights is None and not self.config.allow_download:
            # No network call and no heavy torch import in the offline path.
            self.mode = "opencv_fallback"
            self.reason = "YOLO_WEIGHTS_UNAVAILABLE_OFFLINE"
            return
        try:
            from ultralytics import YOLO  # imported lazily to keep startup light
        except (ImportError, OSError) as exc:
            self.mode, self.reason = "opencv_fallback", f"ULTRALYTICS_NOT_INSTALLED: {exc}"
            return
        if weights is None:
            try:  # explicit opt-in only; default path above never touches network
                self._model = YOLO(self.config.model)
                self.mode, self.reason = "yolo", "yolo26_local_or_downloaded"
                return
            except Exception as exc:  # pragma: no cover - network disabled in tests
                self.mode = "opencv_fallback"
                self.reason = f"YOLO_DOWNLOAD_FAILED: {exc}"
                return
        try:
            self._model = YOLO(str(weights))
            self.mode, self.reason = "yolo", f"weights:{weights.name}"
        except Exception as exc:
            self._model = None
            self.mode = "opencv_fallback"
            self.reason = f"YOLO_LOAD_FAILED: {exc}"

    def detect(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        *,
        width: int,
        height: int,
        rois: list[dict[str, Any]] | None = None,
        scene: str = "",
        auto_zones: bool = False,
    ) -> DetectorResult:
        """Detect objects on sampled frames and attach ROI/zone membership."""
        start = time.perf_counter()
        if self.mode == "yolo" and self._model is not None:
            detections = self._detect_yolo(frames, timestamps)
        else:
            detections = self._detect_opencv(frames, timestamps)
        zones = _zone_definitions(rois or [], width, height, auto_zones)
        detections = _attach_zones(detections, zones, width, height)
        return DetectorResult(
            mode=self.mode,
            reason=self.reason,
            model=self.config.model,
            detections=detections,
            zones=zones,
            frames_analyzed=len(frames),
            latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
        )

    def _detect_yolo(
        self, frames: list[np.ndarray], timestamps: list[float]
    ) -> list[dict[str, Any]]:
        """Parse Ultralytics result objects into plain detection dictionaries."""
        detections: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            timestamp = timestamps[index] if index < len(timestamps) else 0.0
            try:
                if self.config.use_tracking and hasattr(self._model, "track"):
                    results = self._model.track(
                        frame,
                        conf=self.config.conf,
                        iou=self.config.iou,
                        imgsz=self.config.imgsz,
                        device=self.config.device,
                        persist=True,
                        verbose=False,
                    )
                else:
                    results = self._model.predict(
                        frame,
                        conf=self.config.conf,
                        iou=self.config.iou,
                        imgsz=self.config.imgsz,
                        device=self.config.device,
                        verbose=False,
                    )
            except Exception:  # one bad frame must not kill the run
                continue
            for result in results if isinstance(results, (list, tuple)) else [results]:
                detections.extend(_parse_yolo_result(result, index, timestamp, self._model))
        return detections

    def _detect_opencv(
        self, frames: list[np.ndarray], timestamps: list[float]
    ) -> list[dict[str, Any]]:
        """Phase 1 style foreground/blob detector with zero model weights."""
        if not frames:
            return []
        baseline_count = min(6, len(frames))
        grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        baseline = np.median(np.stack(grays[:baseline_count]), axis=0).astype(np.uint8)
        detections: list[dict[str, Any]] = []
        for index, (frame, gray) in enumerate(zip(frames, grays)):
            timestamp = timestamps[index] if index < len(timestamps) else 0.0
            difference = cv2.absdiff(gray, baseline)
            _, mask = cv2.threshold(difference, 28, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((7, 7), dtype=np.uint8), iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = gray.shape[:2]
            for rank, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:6]):
                x, y, box_w, box_h = cv2.boundingRect(contour)
                area_ratio = (box_w * box_h) / max(width * height, 1)
                if area_ratio < 0.004:
                    continue
                detections.append(
                    {
                        "frame_index": index,
                        "timestamp_sec": round(float(timestamp), 3),
                        "label": "moving_object",
                        "confidence": round(min(0.95, 0.55 + area_ratio * 2.0), 4),
                        "bbox": [x, y, box_w, box_h],
                        "center": [x + box_w / 2.0, y + box_h / 2.0],
                        "area_ratio": round(float(area_ratio), 4),
                        "track_id": None,
                        "zone_id": "",
                        "source": "opencv_fallback",
                        "detail": f"blob_{rank}",
                    }
                )
        return detections


def _parse_yolo_result(result: Any, frame_index: int, timestamp: float, model: Any) -> list[dict[str, Any]]:
    """Convert one Ultralytics result's boxes to detection dictionaries."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(boxes.xyxy)
    confidences = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls)
    track_ids = _to_numpy(boxes.id) if getattr(boxes, "id", None) is not None else None
    names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    detections: list[dict[str, Any]] = []
    for row in range(len(xyxy)):
        x1, y1, x2, y2 = [float(value) for value in xyxy[row][:4]]
        class_id = int(classes[row])
        label = str(names.get(class_id, class_id)) if hasattr(names, "get") else str(class_id)
        confidence = float(confidences[row]) if row < len(confidences) else 0.0
        track_id = int(track_ids[row]) if track_ids is not None and row < len(track_ids) else None
        detections.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(float(timestamp), 3),
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "center": [round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1)],
                "area_ratio": round(max(0.0, (x2 - x1) * (y2 - y1)) / 65_536.0, 4),
                "track_id": track_id,
                "zone_id": "",
                "source": "yolo26",
                "detail": f"class_{class_id}",
            }
        )
    return detections


def _zone_definitions(
    rois: list[dict[str, Any]], width: int, height: int, auto_zones: bool
) -> list[dict[str, Any]]:
    """Use manifest ROIs, or derive a coarse grid when auto_zones is enabled."""
    if rois:
        return [
            {
                "zone_id": str(roi.get("zone_id", f"zone_{index}")),
                "kind": str(roi.get("kind", "box")),
                "coords": list(roi.get("coords", [0, 0, 1, 1])),
                "origin": "manifest",
            }
            for index, roi in enumerate(rois)
        ]
    if not auto_zones:
        return []
    zones = []
    for row in range(2):
        for column in range(3):
            zones.append(
                {
                    "zone_id": f"auto_r{row}c{column}",
                    "kind": "grid",
                    "coords": [column / 3.0, row / 2.0, (column + 1) / 3.0, (row + 1) / 2.0],
                    "origin": "detected_grid",
                }
            )
    return zones


def _attach_zones(
    detections: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Label detections with the zone containing their centre point."""
    if not zones:
        return detections
    boxes: list[tuple[str, float, float, float, float]] = []
    for zone in zones:
        x1, y1, x2, y2 = [float(value) for value in zone["coords"][:4]]
        if max(x1, y1, x2, y2) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        boxes.append((str(zone["zone_id"]), x1, y1, x2, y2))
    for detection in detections:
        cx, cy = detection.get("center", [0.0, 0.0])
        for zone_id, x1, y1, x2, y2 in boxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                detection["zone_id"] = zone_id
                break
    return detections


def build_detector(config: DetectorConfig | None = None, model: Any = None) -> YOLODetector:
    """Factory used by the fusion layer and tests."""
    return YOLODetector(config, model=model)
