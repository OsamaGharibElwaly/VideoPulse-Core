"""Five lightweight CV features, optionally enriched by YOLO26 detections.

Every feature keeps its Phase 1 OpenCV/NumPy behaviour when ``detector`` is
``None``; detections only add evidence and are never required to produce a
result.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import cv2
import numpy as np

from core.media_probe import MediaContext
from core.yolo_detector import DetectorResult

FeatureOutput = tuple[list[dict[str, Any]], dict[str, Any]]
Detection = dict[str, Any]


def _gray_frames(context: MediaContext) -> list[np.ndarray]:
    """Convert sampled frames to grayscale."""
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in context.sampled_frames]


def _motion_series(grays: list[np.ndarray]) -> list[float]:
    """Return per-frame motion energy aligned to frame indices."""
    if not grays:
        return []
    series = [0.0]
    series.extend(
        float(np.mean(cv2.absdiff(grays[i], grays[i - 1])))
        for i in range(1, len(grays))
    )
    return series


def _timestamp(context: MediaContext, index: int) -> float:
    """Return the timestamp for a sampled frame index."""
    if index < len(context.timestamps):
        return round(float(context.timestamps[index]), 3)
    return round(index / max(context.fps, 1.0), 3)


def _high_indices(values: list[float], sigma: float = 0.6, minimum: int = 1) -> list[int]:
    """Select values above mean + sigma * std, always keeping at least one."""
    if not values:
        return []
    array = np.asarray(values, dtype=float)
    threshold = float(array.mean() + sigma * array.std())
    selected = [i for i, value in enumerate(values) if value >= threshold]
    if len(selected) < minimum:
        selected = [int(np.argmax(array))]
    return selected


def _detections(detector: DetectorResult | None) -> list[Detection]:
    """Return a detector's detections (empty in Phase 1 mode)."""
    return detector.detections if detector is not None else []


def _at_frame(detections: list[Detection], index: int) -> list[Detection]:
    """Return detections belonging to one sampled frame."""
    return [item for item in detections if item.get("frame_index") == index]


def _near_time(detections: list[Detection], timestamp: float, window: float = 2.5) -> list[Detection]:
    """Return detections inside a timestamp window."""
    return [
        item
        for item in detections
        if abs(float(item.get("timestamp_sec", 0.0)) - timestamp) <= window
    ]


def _mode_stats(detector: DetectorResult | None) -> dict[str, Any]:
    """Add detector provenance to a feature's statistics."""
    if detector is None:
        return {}
    return {"detector_mode": detector.mode, "detector_model": detector.model}


# ---------------------------------------------------------------------------
# 1. Motion keyframe extraction
# ---------------------------------------------------------------------------


def motion_keyframe_extraction(
    context: MediaContext, detector: DetectorResult | None = None
) -> FeatureOutput:
    """Select high-change frames via frame differencing and object novelty."""
    series = _motion_series(_gray_frames(context))
    picked = _high_indices(series)
    detections = _detections(detector)
    if detector is not None:
        by_confidence = sorted(
            {item["frame_index"] for item in detections if float(item.get("confidence", 0)) >= 0.7},
            key=lambda index: max(
                float(item.get("confidence", 0))
                for item in _at_frame(detections, index)
            ),
            reverse=True,
        )[:2]
        for index in by_confidence:
            if index not in picked and index < len(series):
                picked.append(index)
        picked.sort()
    keyframes = []
    for index in picked:
        record: dict[str, Any] = {
            "frame_index": index,
            "timestamp_sec": _timestamp(context, index),
            "motion_score": round(series[index], 4),
        }
        if detector is not None:
            frame_detections = _at_frame(detections, index)
            record["object_labels"] = sorted({str(item["label"]) for item in frame_detections})
            record["object_count"] = len(frame_detections)
            record["max_confidence"] = round(
                max((float(item.get("confidence", 0.0)) for item in frame_detections), default=0.0), 4
            )
        keyframes.append(record)
    total = max(len(series), 1)
    stats = {
        "frames_total": len(series),
        "frames_selected": len(keyframes),
        "frames_skipped_ratio": round(1.0 - len(keyframes) / total, 4),
        **_mode_stats(detector),
    }
    return keyframes, stats


# ---------------------------------------------------------------------------
# 2. ROI & zone tracking
# ---------------------------------------------------------------------------


def _zone_bounds(roi: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a normalised or pixel box to clipped pixel bounds."""
    coords = [float(value) for value in roi.get("coords", [0, 0, width, height])]
    x1, y1, x2, y2 = coords[:4]
    if max(coords[:4]) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(width, max(int(x1) + 1, int(x2))),
        min(height, max(int(y1) + 1, int(y2))),
    )


def roi_zone_tracking(
    context: MediaContext, detector: DetectorResult | None = None
) -> FeatureOutput:
    """Measure motion and detected objects inside each zone."""
    grays = _gray_frames(context)
    detections = _detections(detector)
    definitions = [dict(roi) for roi in context.rois]
    if detector is not None:
        known = {str(roi.get("zone_id")) for roi in definitions}
        definitions.extend(
            dict(zone) for zone in detector.zones if str(zone.get("zone_id")) not in known
        )
    zones: list[dict[str, Any]] = []
    for roi in definitions:
        zone_id = str(roi.get("zone_id", "zone"))
        bounds = _zone_bounds(roi, context.width, context.height)
        x1, y1, x2, y2 = bounds
        crops = [gray[y1:y2, x1:x2] for gray in grays]
        series = [0.0]
        series.extend(
            float(np.mean(cv2.absdiff(crops[i], crops[i - 1])))
            for i in range(1, len(crops))
            if crops[i].shape == crops[i - 1].shape
        )
        active = _high_indices(series, sigma=0.5)
        peak = int(np.argmax(series)) if series else 0
        record: dict[str, Any] = {
            "zone_id": zone_id,
            "kind": str(roi.get("kind", "box")),
            "bounds_px": list(bounds),
            "mean_energy": round(float(np.mean(series)), 4),
            "peak_energy": round(series[peak], 4),
            "peak_timestamp_sec": _timestamp(context, peak),
            "active_timestamps": [_timestamp(context, i) for i in active],
            "active_frames": len(active),
        }
        if detector is not None:
            members = [item for item in detections if item.get("zone_id") == zone_id]
            record["object_count"] = len(members)
            record["object_labels"] = sorted(
                label for label, _ in Counter(str(item["label"]) for item in members).items()
            )
            record["object_timestamps"] = sorted(
                {round(float(item.get("timestamp_sec", 0.0)), 3) for item in members}
            )
            record["max_confidence"] = round(
                max((float(item.get("confidence", 0.0)) for item in members), default=0.0), 4
            )
        zones.append(record)
    return zones, {"zones_tracked": len(zones), **_mode_stats(detector)}


# ---------------------------------------------------------------------------
# 3. Visual anomaly scoring
# ---------------------------------------------------------------------------

_ANOMALY_KINDS = {
    "slides": "layout_shift",
    "conveyor": "object_outlier",
    "facility": "scene_change",
    "aisle": "activity_spike",
    "demo": "motion_spike",
    "assembly": "motion_spike",
}
_OUT_OF_CONTEXT = {"conveyor": {"person", "bicycle"}, "facility": {"car", "bus", "truck"}}


def _rolling_deviations(grays: list[np.ndarray]) -> list[float]:
    """Deviation of each frame from the median of the preceding frames."""
    deviations: list[float] = []
    for index, gray in enumerate(grays):
        if index < 2:
            deviations.append(0.0)
            continue
        window = np.stack(grays[max(0, index - 3) : index])
        baseline = np.median(window, axis=0).astype(np.uint8)
        deviations.append(float(np.mean(cv2.absdiff(gray, baseline))))
    return deviations


def _colour_outlier_counts(frames: list[np.ndarray]) -> list[float]:
    """Count strongly single-hued pixels, catching outlier objects/markings."""
    counts: list[float] = []
    for frame in frames:
        channels = frame.astype(np.int16)
        dominance = np.max(channels, axis=2) - np.median(channels, axis=2)
        counts.append(float(np.count_nonzero(dominance > 90)))
    return counts


def _robust_z(values: list[float]) -> np.ndarray:
    """Median/MAD z-score, robust to a few large deviation spikes."""
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = mad / 0.6745 if mad > 1e-6 else max(1.0, abs(median) * 0.1)
    return (array - median) / scale


def _object_anomalies(
    context: MediaContext, detections: list[Detection], frame_count: int
) -> list[dict[str, Any]]:
    """Flag rare, out-of-context, oversized or count-spike detections."""
    if not detections or not frame_count:
        return []
    labels = [str(item["label"]) for item in detections]
    frequency = Counter(labels)
    counts = [len(_at_frame(detections, index)) for index in range(frame_count)]
    mean_count = float(np.mean(counts))
    std_count = float(np.std(counts))
    forbidden = _OUT_OF_CONTEXT.get(context.scene, set())
    anomalies: list[dict[str, Any]] = []
    for item in detections:
        index = int(item.get("frame_index", 0))
        if index >= frame_count:
            continue
        label = str(item["label"])
        confidence = float(item.get("confidence", 0.0))
        area = float(item.get("area_ratio", 0.0))
        reasons: list[str] = []
        if confidence >= 0.5 and frequency[label] / max(len(detections), 1) < 0.15:
            reasons.append("rare_object")
        if label.lower() in forbidden:
            reasons.append("out_of_context")
        if area > 0.35:
            reasons.append("size_outlier")
        if counts[index] > mean_count + 2 * std_count and std_count > 0:
            reasons.append("count_spike")
        if not reasons:
            continue
        anomalies.append(
            {
                "frame_index": index,
                "timestamp_sec": round(float(item.get("timestamp_sec", 0.0)), 3),
                "score": round(area, 4),
                "colour_outlier_pixels": 0,
                "z_score": round(3.0 if "out_of_context" in reasons else 2.6, 4),
                "kind": reasons[0],
                "object_evidence": [
                    {"label": label, "confidence": round(confidence, 4), "zone_id": item.get("zone_id", "")}
                ],
            }
        )
    return anomalies


def visual_anomaly_scoring(
    context: MediaContext, detector: DetectorResult | None = None
) -> FeatureOutput:
    """Flag deviations from a rolling baseline plus object-level anomalies."""
    grays = _gray_frames(context)
    if not grays:
        return [], {}
    deviations = _rolling_deviations(grays)
    colour_scores = _colour_outlier_counts(context.sampled_frames)
    z_layout = _robust_z(deviations)
    z_colour = _robust_z(colour_scores)
    z_scores = np.maximum(z_layout, z_colour)
    kind = _ANOMALY_KINDS.get(context.scene, "visual_deviation")
    anomalies: list[dict[str, Any]] = [
        {
            "frame_index": index,
            "timestamp_sec": _timestamp(context, index),
            "score": round(deviations[index], 4),
            "colour_outlier_pixels": int(colour_scores[index]),
            "z_score": round(float(z_scores[index]), 4),
            "kind": kind,
        }
        for index in range(len(grays))
        if z_scores[index] > 2.5
    ]
    object_anomalies = _object_anomalies(context, _detections(detector), len(grays))
    anomalies.extend(object_anomalies)
    # One record per frame, keeping the strongest signal.
    best: dict[int, dict[str, Any]] = {}
    for record in anomalies:
        index = record["frame_index"]
        if index not in best or record["z_score"] > best[index]["z_score"]:
            best[index] = record
    anomalies = [best[key] for key in sorted(best)]
    return anomalies, {
        "baseline_frames": 3,
        "mean_deviation": round(float(np.mean(deviations)), 4),
        "object_anomalies": len(object_anomalies),
        **_mode_stats(detector),
    }


# ---------------------------------------------------------------------------
# 4. On-screen text detection
# ---------------------------------------------------------------------------


def _text_contours(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find text-like rectangles using edges and morphology."""
    region = gray[: int(gray.shape[0] * 0.85), :]
    edges = cv2.Canny(region, 80, 180)
    dilated = cv2.dilate(edges, np.ones((5, 9), dtype=np.uint8))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 9 and 4 <= h <= 40 and w * h >= 28 and x + w <= gray.shape[1]:
            boxes.append((x, y, w, h))
    return boxes


try:  # pragma: no cover - optional local install only
    import pytesseract as _pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _pytesseract = None


def _optional_ocr(gray: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    """Use pytesseract only if already installed; otherwise return None."""
    if _pytesseract is None:
        return None
    x, y, w, h = box
    try:  # pragma: no cover - external tesseract binary may be unavailable
        return str(_pytesseract.image_to_string(gray[y : y + h, x : x + w])).strip() or None
    except Exception:  # pragma: no cover
        return None


def on_screen_text_detection(
    context: MediaContext, detector: DetectorResult | None = None
) -> FeatureOutput:
    """Locate high-contrast text-like regions; no OCR dependency required."""
    regions: list[dict[str, Any]] = []
    ocr_budget = 3  # OCR is optional and slow; bound it per video.
    detections = _detections(detector)
    for index, frame in enumerate(context.sampled_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = sorted(_text_contours(gray), key=lambda box: box[0])[:5]
        for box_index, box in enumerate(boxes):
            x, y, w, h = box
            text = None
            if ocr_budget > 0:
                text = _optional_ocr(gray, box)
                ocr_budget -= 1
            record: dict[str, Any] = {
                "region_id": f"f{index}_r{box_index}",
                "frame_index": index,
                "timestamp_sec": _timestamp(context, index),
                "bbox": [x, y, w, h],
                "confidence": round(min(0.99, 0.5 + (w * h) / 5000.0), 4),
                "text": text,
            }
            if detector is not None:
                centre_x, centre_y = x + w / 2.0, y + h / 2.0
                nearby = [
                    str(item["label"])
                    for item in _at_frame(detections, index)
                    if abs(float(item["center"][0]) - centre_x) < 90
                    and abs(float(item["center"][1]) - centre_y) < 90
                ]
                record["nearby_objects"] = sorted(set(nearby))
            regions.append(record)
    return regions, {
        "regions_found": len(regions),
        "ocr_available": any(r["text"] for r in regions),
        **_mode_stats(detector),
    }


# ---------------------------------------------------------------------------
# 5. Action state indexing
# ---------------------------------------------------------------------------


def _classify_state(motion: float, reference: float, vocabulary: list[str]) -> str:
    """Map motion level onto the profile's state vocabulary."""
    upper = set(vocabulary)
    if "CONGESTION" in upper:
        if motion > 1.4 * reference:
            return "CONGESTION"
        return "FLOW" if motion >= 0.6 * reference else "QUIET"
    if "BURST" in upper:
        if motion > 1.4 * reference:
            return "BURST"
        if "IDLE" in upper:
            return "MOVING" if motion >= 0.7 * reference else "IDLE"
        if motion < 0.6 * reference:
            return "STATIC"
        return "TRANSITION" if motion > 1.1 * reference else "ACTIVE"
    return vocabulary[0] if vocabulary else "ACTIVE"


def action_state_indexing(
    context: MediaContext, detector: DetectorResult | None = None
) -> FeatureOutput:
    """Divide the timeline into labelled motion/object state segments."""
    grays = _gray_frames(context)
    series = _motion_series(grays)
    if not series:
        return [], {}
    vocabulary = [str(v) for v in context.profile.get("state_vocabulary", [])] or [
        "STATIC",
        "ACTIVE",
        "TRANSITION",
        "BURST",
    ]
    reference = max(float(np.mean(series)), 1e-6)
    detections = _detections(detector)
    bins = min(6, len(series))
    window = context.duration_sec / bins
    per_bin_counts = [
        len(_near_time(detections, (index + 0.5) * window, window / 2.0))
        for index in range(bins)
    ]
    count_reference = max(float(np.mean(per_bin_counts)), 1e-6)
    states: list[dict[str, Any]] = []
    for bin_index in range(bins):
        start = bin_index * window
        end = (bin_index + 1) * window
        values = [
            series[i]
            for i, timestamp in enumerate(context.timestamps)
            if start <= timestamp < end
        ]
        if not values:
            continue
        motion = float(np.mean(values))
        state = _classify_state(motion, reference, vocabulary)
        record: dict[str, Any] = {
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "state": state,
            "motion_mean": round(motion, 4),
        }
        if detector is not None:
            members = _near_time(detections, (start + end) / 2.0, window / 2.0)
            record["object_count"] = len(members)
            record["top_labels"] = [
                label for label, _ in Counter(str(item["label"]) for item in members).most_common(3)
            ]
            if per_bin_counts[bin_index] > 1.6 * count_reference and per_bin_counts[bin_index] > 1:
                record["state"] = "CONGESTION" if "CONGESTION" in vocabulary else "BURST"
        states.append(record)
    return states, {"state_bins": len(states), "vocabulary": vocabulary, **_mode_stats(detector)}


DISPATCH: dict[str, Callable[[MediaContext, DetectorResult | None], FeatureOutput]] = {
    "motion_keyframe_extraction": motion_keyframe_extraction,
    "roi_zone_tracking": roi_zone_tracking,
    "visual_anomaly_scoring": visual_anomaly_scoring,
    "on_screen_text_detection": on_screen_text_detection,
    "action_state_indexing": action_state_indexing,
}


def run_cv_feature(
    name: str, context: MediaContext, detector: DetectorResult | None = None
) -> FeatureOutput:
    """Execute one CV feature by its canonical name."""
    return DISPATCH[name](context, detector)
