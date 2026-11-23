"""Five lightweight OpenCV/NumPy computer-vision features."""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from core.media_probe import MediaContext

FeatureOutput = tuple[list[dict[str, Any]], dict[str, Any]]


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


# ---------------------------------------------------------------------------
# 1. Motion keyframe extraction
# ---------------------------------------------------------------------------


def motion_keyframe_extraction(context: MediaContext) -> FeatureOutput:
    """Select high-change frames via frame differencing."""
    series = _motion_series(_gray_frames(context))
    picked = _high_indices(series)
    keyframes = [
        {
            "frame_index": index,
            "timestamp_sec": _timestamp(context, index),
            "motion_score": round(series[index], 4),
        }
        for index in picked
    ]
    total = max(len(series), 1)
    stats = {
        "frames_total": len(series),
        "frames_selected": len(keyframes),
        "frames_skipped_ratio": round(1.0 - len(keyframes) / total, 4),
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


def roi_zone_tracking(context: MediaContext) -> FeatureOutput:
    """Measure motion inside each authored zone across the timeline."""
    grays = _gray_frames(context)
    zones: list[dict[str, Any]] = []
    for roi in context.rois:
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
        zones.append(
            {
                "zone_id": str(roi.get("zone_id", "zone")),
                "kind": str(roi.get("kind", "box")),
                "bounds_px": list(bounds),
                "mean_energy": round(float(np.mean(series)), 4),
                "peak_energy": round(series[peak], 4),
                "peak_timestamp_sec": _timestamp(context, peak),
                "active_timestamps": [_timestamp(context, i) for i in active],
                "active_frames": len(active),
            }
        )
    return zones, {"zones_tracked": len(zones)}


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


def visual_anomaly_scoring(context: MediaContext) -> FeatureOutput:
    """Flag frames that deviate from a rolling baseline or colour profile."""
    grays = _gray_frames(context)
    if not grays:
        return [], {}
    deviations = _rolling_deviations(grays)
    colour_scores = _colour_outlier_counts(context.sampled_frames)
    z_layout = _robust_z(deviations)
    z_colour = _robust_z(colour_scores)
    z_scores = np.maximum(z_layout, z_colour)
    kind = _ANOMALY_KINDS.get(context.scene, "visual_deviation")
    anomalies = [
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
    return anomalies, {
        "baseline_frames": 3,
        "mean_deviation": round(float(np.mean(deviations)), 4),
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


def on_screen_text_detection(context: MediaContext) -> FeatureOutput:
    """Locate high-contrast text-like regions; no OCR dependency required."""
    regions: list[dict[str, Any]] = []
    ocr_budget = 3  # OCR is optional and slow; bound it per video.
    for index, frame in enumerate(context.sampled_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = sorted(_text_contours(gray), key=lambda box: box[0])[:5]
        for box_index, box in enumerate(boxes):
            x, y, w, h = box
            text = None
            if ocr_budget > 0:
                text = _optional_ocr(gray, box)
                ocr_budget -= 1
            regions.append(
                {
                    "region_id": f"f{index}_r{box_index}",
                    "frame_index": index,
                    "timestamp_sec": _timestamp(context, index),
                    "bbox": [x, y, w, h],
                    "confidence": round(min(0.99, 0.5 + (w * h) / 5000.0), 4),
                    "text": text,
                }
            )
    return regions, {"regions_found": len(regions), "ocr_available": any(r["text"] for r in regions)}


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


def action_state_indexing(context: MediaContext) -> FeatureOutput:
    """Divide the timeline into labelled motion-state segments."""
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
    bins = min(6, len(series))
    window = context.duration_sec / bins
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
        states.append(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "state": _classify_state(motion, reference, vocabulary),
                "motion_mean": round(motion, 4),
            }
        )
    return states, {"state_bins": len(states), "vocabulary": vocabulary}


DISPATCH: dict[str, Callable[[MediaContext], FeatureOutput]] = {
    "motion_keyframe_extraction": motion_keyframe_extraction,
    "roi_zone_tracking": roi_zone_tracking,
    "visual_anomaly_scoring": visual_anomaly_scoring,
    "on_screen_text_detection": on_screen_text_detection,
    "action_state_indexing": action_state_indexing,
}


def run_cv_feature(name: str, context: MediaContext) -> FeatureOutput:
    """Execute one CV feature by its canonical name."""
    return DISPATCH[name](context)
